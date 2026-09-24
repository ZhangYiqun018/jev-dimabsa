#!/usr/bin/env python3
"""Offline comparison and retrieval overlap for the fixed trial24 experiment."""
import hashlib
import json
from itertools import combinations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
from diagnose_span_choice import selected
from probe_bio_bm25 import summary
from combined_experiment import save

OUT = ROOT / 'reports/bio_bm25_50_20260923'


def main():
    rows = {(c, r['ID']): r for c, r in selected('trial', 6)}
    reports = {v: json.loads((OUT / v / 'summary.json').read_text())
               for v in ('bigram', 'trigram', 'word')}
    reports['bm25_2'] = json.loads((ROOT / 'reports/bio_bm25_20260923/trial/summary.json').read_text())
    predictions = {}
    examples = {}
    for variant, report in reports.items():
        predictions[variant] = {}
        examples[variant] = {}
        for example, path in zip(report['examples'], report['cache_files']):
            key = example['corpus'], example['ID']
            predictions[variant][key] = json.loads((ROOT / path).read_text())
            examples[variant][key] = set(example['example_ids'])
    predictions['bio_r3'] = {}
    for key, row in rows.items():
        corpus, identity = key
        suffix = hashlib.sha256((corpus + identity + row['Text']).encode()).hexdigest()[:16]
        path = ROOT / 'reports/extraction_comparison_20260923/trial_bio/cache' / f'bio_r3_41ea479bf861_{suffix}.json'
        predictions['bio_r3'][key] = json.loads(path.read_text())
    common = set.intersection(*(set(p) for p in predictions.values()))
    arms = {}
    for variant in ('bio_r3', 'bm25_2', 'bigram', 'trigram', 'word'):
        done = [(c, r, predictions[variant][c, identity], None)
                for (c, identity), r in rows.items() if (c, identity) in common]
        result = summary(done)
        result['logical_input_tokens_common'] = sum(t['usage']['input_tokens'] for c, r, p, path in done for t in p['trace'])
        if variant in reports:
            report = reports[variant]
            for field in ('complete', 'new_input_tokens', 'new_estimated_usd', 'new_calls', 'new_http_attempts', 'reused_pair_requests', 'missing'):
                result[field] = report[field]
        arms[variant] = result
    overlaps = {}
    for a, b in combinations(('bigram', 'trigram', 'word'), 2):
        per_row = []
        for c, identity in rows:
            key = c, identity
            if key in examples[a] and key in examples[b]:
                x, y = examples[a][key], examples[b][key]
                per_row.append({'corpus': c, 'ID': identity, 'shared': len(x & y), 'jaccard': len(x & y) / len(x | y)})
        overlaps[f'{a}/{b}'] = {'rows': per_row,
            'mean_shared_of_50': sum(r['shared'] for r in per_row) / len(per_row) if per_row else None,
            'mean_jaccard': sum(r['jaccard'] for r in per_row) / len(per_row) if per_row else None}
    save(OUT / 'comparison.json', {'n_common': len(common), 'n_requested': len(rows), 'arms': arms,
        'retrieval_overlap': overlaps,
        'new_experiment_input_tokens': sum(reports[v]['new_input_tokens'] for v in ('bigram', 'trigram', 'word')),
        'new_experiment_estimated_usd': sum(reports[v]['new_estimated_usd'] for v in ('bigram', 'trigram', 'word')),
        'notes': ['Exact AO surface-pair metrics without VA; not official cF1.',
                  'Common completed rows only; original trial24 is reused development data.',
                  '2-shot vs 50-shot also changes retrieval tokenization; not a pure shot-count ablation.',
                  'No ensemble, correction, threshold search or subsequent model requests.']})
    for v, r in arms.items():
        print(v, 'macro', r['macro_pair_f1'], 'micro', r['micro']['pair'], 'logical tokens', r['logical_input_tokens_common'])
    print('overlap', {k: v['mean_shared_of_50'] for k, v in overlaps.items()})

if __name__ == '__main__':
    main()
