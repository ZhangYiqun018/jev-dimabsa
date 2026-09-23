"""Task 2 baseline: training-lexicon spans, Jev pair decisions and VA scores.

Only ID/Text are read at prediction time. Gold Triplet fields are never inputs.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .client import Response, format_va, score_to_va
from .data import _annotation_items, load_jsonl
from .fewshot import normalise
from .rubrics import arousal_question, valence_question

PAIR_BATCH = 16


def span_pattern(terms):
    # CJK has no whitespace word boundaries. Other scripts use word boundaries
    # so e.g. "rice" does not match "price". Prefer longer overlapping phrases.
    parts = []
    for term in sorted(terms, key=lambda s: (-len(s), s)):
        cjk = bool(re.search(r'[\u3040-\u30ff\u3400-\u9fff]', term))
        part = re.escape(term)
        if not cjk:
            part = r'(?<!\w)' + part + r'(?!\w)'
        parts.append(part)
    return re.compile('(?=(' + '|'.join(parts) + '))' if parts else r'(?!)', re.I)


class TripletPredictor:
    def __init__(self, train: Path, heldout: list[Path], calibration: dict):
        banned = {normalise(r['Text']) for p in heldout for r in load_jsonl(p)}
        aspects, opinions = set(), set()
        for row in load_jsonl(train):
            if normalise(row['Text']) in banned:
                continue
            for item in _annotation_items(row):
                for key, target in [('Aspect', aspects), ('Opinion', opinions)]:
                    value = item[key].strip()
                    if value and value.upper() != 'NULL':
                        target.add(value.lower())
        self.aspect_pattern = span_pattern(aspects)
        self.opinion_pattern = span_pattern(opinions)
        self.calibration = calibration
        self.config = {
            'task': 2, 'baseline': 'train-lexicon+noul+score+st1-zero-shrink-v1',
            'lexicon_sha256': hashlib.sha256(json.dumps(
                [sorted(aspects), sorted(opinions)], ensure_ascii=False).encode()).hexdigest(),
            'aspect_terms': len(aspects), 'opinion_terms': len(opinions),
            'pair_threshold': 0.5, 'pair_batch': PAIR_BATCH,
            'calibration': calibration,
        }

    def candidates(self, text):
        aspects = list(dict.fromkeys(m.group(1) for m in self.aspect_pattern.finditer(text)))
        opinions = list(dict.fromkeys(m.group(1) for m in self.opinion_pattern.finditer(text)))
        # Match the scorer's case-insensitive pair identity.
        pairs = {}
        for aspect in aspects:
            for opinion in opinions:
                if aspect.lower() != opinion.lower():
                    pairs.setdefault((aspect.lower(), opinion.lower()), (aspect, opinion))
        return list(pairs.values())

    def __call__(self, client, record):
        text = record['Text']
        pairs = self.candidates(text)
        entry = {'ID': record['ID'], 'Text': text, 'Triplet': [], '_response': None}
        usage, answers, attempts, models = {}, {}, 0, set()
        for start in range(0, len(pairs), PAIR_BATCH):
            questions = {}
            batch = pairs[start:start + PAIR_BATCH]
            for i, (aspect, opinion) in enumerate(batch, start):
                questions[f'p{i}'] = {
                    'type': 'noul',
                    'instructions': {
                        'question': f'Does the opinion phrase "{opinion}" express sentiment '
                                    f'directly toward the aspect "{aspect}" in this review?',
                        'focus': 'Both must be explicit complete aspect/opinion spans. '
                                 'Reject unrelated pairs, incomplete fragments and factual '
                                 'mentions without an evaluative opinion.',
                    },
                }
                questions[f'v{i}'] = valence_question(aspect, opinion)
                questions[f'a{i}'] = arousal_question(aspect, opinion)
            response = client.ask(text, questions)
            models.add(response.model)
            attempts += response.attempts
            answers.update({k: v.raw for k, v in response.answers.items()})
            for key, value in response.usage.items():
                usage[key] = usage.get(key, 0) + value
            for i, (aspect, opinion) in enumerate(batch, start):
                if response.answers[f'p{i}'].noul < 0.5:
                    continue
                va = [score_to_va(response.answers[f'{d}{i}'].score) for d in ('v', 'a')]
                va = [min(9., max(1., b + a * value)) for value, a, b in zip(
                    va, self.calibration['slope'], self.calibration['intercept'])]
                entry['Triplet'].append({'Aspect': aspect, 'Opinion': opinion,
                                         'VA': format_va(*va)})
        if pairs:
            model = ','.join(sorted(models))
            entry['_response'] = Response(model, {}, usage, attempts)
            entry['_jev'] = {'model': model, 'usage': usage, 'attempts': attempts,
                             'answers': answers, 'candidate_pairs': len(pairs)}
        return entry
