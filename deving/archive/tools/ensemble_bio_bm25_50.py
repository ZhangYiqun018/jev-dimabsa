#!/usr/bin/env python3
"""Offline 2-of-3 agreement over accepted AO pairs; no Jev calls."""
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
from combined_experiment import save
from diagnose_span_choice import selected
from iterate_extraction import evaluate
from jev.data import _annotation_items

OUT = ROOT / 'reports/bio_bm25_50_20260923'
VARIANTS = ('bigram', 'trigram', 'word')


def agreement(predictions):
    """One vote per arm per lowercased surface pair, after the frozen .65 gate."""
    votes = Counter()
    for pred in predictions:
        votes.update({(p['Aspect'].lower(), p['Opinion'].lower())
                      for p in pred['pairs'] if p['probability'] >= .65})
    return votes


def pair_metrics(predictions, rows):
    return evaluate(predictions, rows, .65)['pair']


def main():
    (OUT / 'ensemble/cache').mkdir(parents=True, exist_ok=True)
    rows = {(c, r['ID']): r for c, r in selected('trial', 6)}
    predictions = {}
    for variant in VARIANTS:
        report = json.loads((OUT / variant / 'summary.json').read_text())
        predictions[variant] = {(e['corpus'], e['ID']): json.loads((ROOT / p).read_text())
                               for e, p in zip(report['examples'], report['cache_files'])}
    common = set(rows).intersection(*(set(p) for p in predictions.values()))
    merged = {}
    histogram = {str(n): {'gold': 0, 'non_gold': 0} for n in (1, 2, 3)}
    for key, row in rows.items():
        if key not in common:
            continue
        votes = agreement([predictions[v][key] for v in VARIANTS])
        gold = {(x['Aspect'].lower(), x['Opinion'].lower()) for x in _annotation_items(row)}
        for pair, count in votes.items():
            histogram[str(count)]['gold' if pair in gold else 'non_gold'] += 1
        pairs = [{'Aspect': a, 'Opinion': o, 'probability': 1.0, 'votes': n}
                 for (a, o), n in sorted(votes.items()) if n >= 2]
        merged[key] = {'ID': row['ID'], 'spans': {'aspect': [], 'opinion': []}, 'pairs': pairs}
        # Surface strings and sample IDs remain in ignored cache only.
        save(OUT / 'ensemble/cache' / f'{key[0]}_{len(merged):02d}.json',
             {'corpus': key[0], 'prediction': merged[key],
              'votes': [{'Aspect': a, 'Opinion': o, 'votes': n} for (a, o), n in sorted(votes.items())]})
    corpora = []
    for corpus in dict.fromkeys(c for c, identity in rows):
        keys = [k for k in rows if k in common and k[0] == corpus]
        if keys:
            corpora.append({'corpus': corpus, 'n': len(keys),
                            'pair': pair_metrics([merged[k] for k in keys], [rows[k] for k in keys])})
    keys = [k for k in rows if k in common]
    result = {'n': len(keys), 'complete': len(keys) == len(rows),
              'macro_pair_f1': sum(c['pair']['f1'] for c in corpora) / len(corpora) if corpora else None,
              'micro_pair': pair_metrics([merged[k] for k in keys], [rows[k] for k in keys]),
              'corpora': corpora, 'vote_histogram': histogram,
              'new_api_calls': 0, 'new_input_tokens': 0, 'new_estimated_usd': 0,
              'protocol': 'Fixed >=2/3 votes on accepted AO pairs; per-arm Noul >=0.65; deduplicate within each arm; lowercase exact surface matching; NULL is an ordinary pair element.',
              'notes': ['AO adaptation: no category, VA, correction or score averaging.',
                        'Vote fraction is not a calibrated probability; probability=1 only marks final acceptance for the existing evaluator.',
                        'Only pair metrics reported: ensemble does not create a new span extractor.',
                        'Existing development trial24; no new independent validation or threshold selection.']}
    base = json.loads((OUT / 'comparison.json').read_text())
    result['baselines_same_rows'] = {v: {'macro_pair_f1': a['macro_pair_f1'], 'micro_pair': a['micro']['pair']}
                                     for v, a in base['arms'].items()} if len(keys) == len(rows) else None
    result['three_arm_logical_input_tokens'] = sum(base['arms'][v]['logical_input_tokens_common'] for v in VARIANTS)
    save(OUT / 'ensemble/summary.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
