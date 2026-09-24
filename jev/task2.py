"""Shared Task 2 helpers: corpora, AO diagnostics, request caching, pair VA, official cF1.

Used by the BIO r3 development pipeline (tools/evaluate_bio_r3_dev.py). The
lexicon baseline (jev/triplets.py via runners/run.py) predates this module and
keeps its own frozen code path.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import sys
import threading
from pathlib import Path

from .client import Answer, Response, format_va, score_to_va
from .data import _annotation_items
from .rubrics import arousal_question, valence_question

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'vendor/DimABSA2026/task-dataset'
CORPORA = ['eng_restaurant', 'eng_laptop', 'zho_restaurant', 'zho_laptop',
           'jpn_hotel', 'rus_restaurant', 'tat_restaurant', 'ukr_restaurant']
PAIR_THRESHOLD = .65
VA_BATCH_PAIRS = 16


def split_path(corpus, split):
    folder = DATA / 'track_a/subtask_2' / corpus[:3]
    name = f'{corpus}_train_alltasks.jsonl' if split == 'train' else f'{corpus}_{split}_task2.jsonl'
    return folder / name


def record_key(corpus, record):
    """Record IDs repeat across corpora, so cache identity includes corpus and text."""
    return hashlib.sha256((corpus + record['ID'] + record['Text']).encode()).hexdigest()[:16]


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


# ------------------------------------------------------------ AO diagnostics

def _counts(pred, gold):
    return [len(pred & gold), len(pred - gold), len(gold - pred)]


def _metric(values):
    tp, fp, fn = values
    return {'tp': tp, 'fp': fp, 'fn': fn,
            'precision': tp / (tp + fp) if tp + fp else 0.,
            'recall': tp / (tp + fn) if tp + fn else 0.,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.}


def ao_metrics(extracted, gold_rows, threshold=PAIR_THRESHOLD):
    """Exact, case-insensitive surface-set AO metrics per record; no VA, not official cF1.

    Also reports candidate-pair recall (before thresholding) and why each missed
    gold pair was missed.
    """
    total = {k: [0, 0, 0] for k in ('aspect', 'opinion', 'linked_aspect',
                                   'linked_opinion', 'pair', 'null_pair')}
    misses = {k: 0 for k in ('aspect_missing', 'opinion_missing',
                            'both_missing', 'pair_rejected')}
    reachable, gold_count = 0, 0
    for pred, row in zip(extracted, gold_rows):
        gold = {(x['Aspect'].lower(), x['Opinion'].lower()) for x in _annotation_items(row)}
        pairs = {(x['Aspect'].lower(), x['Opinion'].lower()) for x in pred['pairs']
                 if x['probability'] >= threshold}
        candidates = {(x['Aspect'].lower(), x['Opinion'].lower()) for x in pred['pairs']}
        reachable += len(candidates & gold)
        gold_count += len(gold)
        available_aspects = {s.lower() for s in pred['spans']['aspect']} | {p[0] for p in candidates}
        available_opinions = {s.lower() for s in pred['spans']['opinion']} | {p[1] for p in candidates}
        for a, o in gold - pairs:
            missing_a, missing_o = a not in available_aspects, o not in available_opinions
            reason = ('both_missing' if missing_a and missing_o else
                      'aspect_missing' if missing_a else
                      'opinion_missing' if missing_o else 'pair_rejected')
            misses[reason] += 1
        for i, kind in enumerate(('aspect', 'opinion')):
            predicted = {x.lower() for x in pred['spans'][kind] if x.upper() != 'NULL'}
            expected = {p[i] for p in gold if p[i] != 'null'}
            total[kind] = [a + b for a, b in zip(total[kind], _counts(predicted, expected))]
            linked = {p[i] for p in pairs if p[i] != 'null'}
            total['linked_' + kind] = [a + b for a, b in zip(
                total['linked_' + kind], _counts(linked, expected))]
        for kind, p, g in [('pair', pairs, gold),
                           ('null_pair', {p for p in pairs if p[0] == 'null'},
                            {p for p in gold if p[0] == 'null'})]:
            total[kind] = [a + b for a, b in zip(total[kind], _counts(p, g))]
    return {**{k: _metric(v) for k, v in total.items()},
            'candidate_pair_recall': reachable / gold_count if gold_count else 0.,
            'pair_misses': misses}


# ------------------------------------------------------------ requests

class CachedClient:
    """Per-request response cache keyed by model, full state and full questions.

    A response is reused only when the model-visible request is identical, never
    because the AO strings match. Thread-safe; concurrent identical requests
    wait for one call.
    """

    def __init__(self, client, directory, model, stage, fallback=()):
        # ``fallback`` directories are read-only historical caches; new calls go to ``directory``.
        self.client, self.directory, self.model, self.stage = client, directory, model, stage
        self.fallback = tuple(fallback)
        self._lock = threading.Lock()
        self._key_locks = {}

    def with_stage(self, stage):
        other = CachedClient(self.client, self.directory, self.model, stage, self.fallback)
        other._lock, other._key_locks = self._lock, self._key_locks
        return other

    def ask(self, state, questions):
        payload = {'model': self.model, 'state': state, 'questions': questions}
        key = digest(payload)
        path = self.directory / f'{key}.json'
        with self._lock:
            key_lock = self._key_locks.setdefault(key, threading.Lock())
        with key_lock:
            hit = next((d / f'{key}.json' for d in (self.directory, *self.fallback)
                        if (d / f'{key}.json').exists()), None)
            if hit:
                saved = json.loads(hit.read_text())
                return Response(saved['model'], {k: Answer(k, questions[k]['type'], v)
                                for k, v in saved['answers'].items()},
                                saved['usage'], saved['attempts'])
            result = self.client.ask(state, questions, attempts=3)
            save_json(path, {**payload, 'stage': self.stage, 'model': result.model,
                             'usage': result.usage, 'attempts': result.attempts,
                             'answers': {k: a.raw for k, a in result.answers.items()}})
            return result


# ------------------------------------------------------------ VA and scoring

def score_pairs(client, record, extracted, calibration, threshold=PAIR_THRESHOLD):
    """Score accepted pairs with the Task 1 V/A questions; raw and calibrated outputs.

    Pairs are deduplicated by the scorer's case-insensitive identity so each is
    output once. Only the review text is sent as state.
    """
    pairs = {}
    for pair in extracted['pairs']:
        if pair['probability'] >= threshold:
            pairs.setdefault((pair['Aspect'].lower(), pair['Opinion'].lower()),
                             (pair['Aspect'], pair['Opinion']))
    return score_pair_list(client, record, list(pairs.values()), calibration)


def score_pair_list(client, record, pairs, calibration):
    """V/A for an ordered list of unique (aspect, opinion) pairs, 16 pairs per request."""
    raw, calibrated, trace = [], [], []
    for offset in range(0, len(pairs), VA_BATCH_PAIRS):
        batch = pairs[offset:offset + VA_BATCH_PAIRS]
        questions = {}
        for i, (a, o) in enumerate(batch):
            questions[f'v{i}'] = valence_question(a, o)
            questions[f'a{i}'] = arousal_question(a, o)
        response = client.ask(record['Text'], questions)
        trace.append({'state': record['Text'], 'questions': questions,
                      'answers': {k: a.raw for k, a in response.answers.items()},
                      'model': response.model, 'usage': response.usage,
                      'attempts': response.attempts})
        for i, (a, o) in enumerate(batch):
            va = [score_to_va(response.answers[f'{d}{i}'].score) for d in ('v', 'a')]
            shrunk = [min(9., max(1., s * x + b)) for x, s, b in
                      zip(va, calibration['slope'], calibration['intercept'])]
            raw.append({'Aspect': a, 'Opinion': o, 'VA': format_va(*va)})
            calibrated.append({'Aspect': a, 'Opinion': o, 'VA': format_va(*shrunk)})
    return {'ID': record['ID'], 'raw': raw, 'calibrated': calibrated, 'trace': trace}


def official_score(gold, pred, log):
    """Run the unmodified official Task 2 scorer; stdout/stderr are kept in ``log``."""
    result = subprocess.run([sys.executable, str(ROOT / 'scoring/score.py'), '--task', '2',
                             '--gold', str(gold), '--pred', str(pred)],
                            capture_output=True, text=True)
    log.write_text(result.stdout + result.stderr)
    match = re.search(r'Final Results: (\{.*\})', result.stdout)
    if result.returncode or not match:
        raise RuntimeError(f'Official scorer failed; see {log}')
    metrics = ast.literal_eval(re.sub(r'np\.float64\(([^)]*)\)', r'\1', match.group(1)))
    metrics['categorical_TP'] = float(re.search(r'True Positives \(TP\): ([\d.]+)',
                                                result.stdout).group(1))
    return metrics
