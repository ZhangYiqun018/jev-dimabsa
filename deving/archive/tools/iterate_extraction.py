#!/usr/bin/env python3
"""Small trial/dev-only extraction experiment; caches raw responses for reuse."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jev.client import DEFAULT_MODEL, JevClient
from jev.data import _annotation_items, load_jsonl
from jev.extraction import SpanExtractor

DATA = ROOT / 'vendor/DimABSA2026/task-dataset'
CORPORA = ['eng_restaurant', 'eng_laptop', 'zho_restaurant', 'zho_laptop',
           'jpn_hotel', 'rus_restaurant', 'tat_restaurant', 'ukr_restaurant']


def counts(pred, gold):
    return [len(pred & gold), len(pred - gold), len(gold - pred)]


def metric(values):
    tp, fp, fn = values
    return {'tp': tp, 'fp': fp, 'fn': fn,
            'precision': tp / (tp + fp) if tp + fp else 0.,
            'recall': tp / (tp + fn) if tp + fn else 0.,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.}


def evaluate(extracted, gold_rows, threshold):
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
            vals = counts(predicted, expected)
            total[kind] = [a + b for a, b in zip(total[kind], vals)]
            linked = {p[i] for p in pairs if p[i] != 'null'}
            total['linked_' + kind] = [a + b for a, b in zip(
                total['linked_' + kind], counts(linked, expected))]
        for kind, p, g in [('pair', pairs, gold),
                           ('null_pair', {p for p in pairs if p[0] == 'null'},
                            {p for p in gold if p[0] == 'null'})]:
            total[kind] = [a + b for a, b in zip(total[kind], counts(p, g))]
    return {**{k: metric(v) for k, v in total.items()},
            'candidate_pair_recall': reachable / gold_count if gold_count else 0.,
            'pair_misses': misses}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=['trial', 'dev'], required=True)
    parser.add_argument('--mode', choices=['bio', 'pointer', 'combined'], required=True)
    parser.add_argument('--revision', type=int, choices=[1, 2, 3], default=1)
    parser.add_argument('--per-corpus', type=int, default=6)
    parser.add_argument('--threshold', type=float, default=.5)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--concurrency', type=int, default=3)
    parser.add_argument('--max-input-tokens', type=int)
    args = parser.parse_args()
    if args.mode == 'combined':
        from combined_experiment import run_combined
        args.max_input_tokens = args.max_input_tokens or 5_500_000
        return run_combined(args, CORPORA, evaluate)
    args.max_input_tokens = args.max_input_tokens or 4_000_000
    cache = args.out / 'cache'
    cache.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256((ROOT/'jev/extraction.py').read_bytes()).hexdigest()[:12]
    selected, cached_baseline = {}, {}
    trial_texts = {r['Text'].strip().lower() for p in (DATA/'trial').glob('*_alltasks.jsonl')
                   for r in load_jsonl(p)}
    for corpus in CORPORA:
        if args.split == 'trial':
            path = DATA/'trial'/f'{corpus}_trial_alltasks.jsonl'
        else:
            path = DATA/'track_a/subtask_2'/corpus[:3]/f'{corpus}_dev_task2.jsonl'
        if not path.exists():
            continue
        rows = load_jsonl(path)
        if args.split == 'dev':
            rows = [r for r in rows if r['Text'].strip().lower() not in trial_texts]
            old = ROOT/'reports/st2_baseline_20260923/cache'/f'{corpus}_dev.jsonl'
            cached_baseline[corpus] = {r['ID']: r for r in load_jsonl(old)}
        # Same order for both methods; independent of labels and text length.
        rows.sort(key=lambda r: hashlib.sha256(
            ('20260923:' + corpus + ':' + r['ID']).encode()).digest())
        selected[corpus] = rows[:args.per_corpus]

    lock = threading.Lock()
    spent = 0
    client = JevClient(timeout=60)

    class BudgetClient:
        def ask(self, state, questions):
            nonlocal spent
            with lock:
                if spent >= args.max_input_tokens:
                    raise RuntimeError('Experiment input-token budget reached; cached results retained')
            response = client.ask(state, questions, attempts=2)
            with lock:
                spent += response.usage.get('input_tokens', 0)
            return response

    predictor = SpanExtractor(args.mode, args.revision)
    jobs = [(c, r) for c, rows in selected.items() for r in rows]

    def run(job):
        corpus, row = job
        key = hashlib.sha256((corpus + row['ID'] + row['Text']).encode()).hexdigest()[:16]
        path = cache/f'{args.mode}_r{args.revision}_{fingerprint}_{key}.json'
        if path.exists():
            pred = json.loads(path.read_text())
        else:
            pred = predictor(BudgetClient(), {'ID': row['ID'], 'Text': row['Text']})
            path.write_text(json.dumps(pred, ensure_ascii=False))
        print(f"{corpus} {row['ID']}: {len(pred['spans']['aspect'])} aspects, "
              f"{len(pred['spans']['opinion'])} opinions, {len(pred['trace'])} calls", flush=True)
        return corpus, pred

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        predictions = list(pool.map(run, jobs))
    summaries = []
    for corpus, rows in selected.items():
        preds = [p for c, p in predictions if c == corpus]
        result = {'corpus': corpus, 'n': len(rows), 'ids': [r['ID'] for r in rows],
                  'metrics': evaluate(preds, rows, args.threshold)}
        if args.split == 'dev':
            baseline = []
            for r in rows:
                triplets = cached_baseline[corpus][r['ID']]['Triplet']
                baseline.append({'spans': {k: [t[k.title()] for t in triplets]
                    for k in ('aspect', 'opinion')},
                    'pairs': [dict(t, probability=1.) for t in triplets]})
            result['baseline'] = evaluate(baseline, rows, args.threshold)
        summaries.append(result)
    all_preds = [p for _, p in predictions]
    all_rows = [r for _, r in jobs]
    traces = [t for p in all_preds for t in p['trace']]
    total_tokens = sum(t['usage'].get('input_tokens', 0) for t in traces)
    summary = {'split': args.split, 'mode': args.mode, 'revision': args.revision,
        'fingerprint': fingerprint, 'model': sorted({t['model'] for t in traces}),
        'per_corpus': args.per_corpus, 'threshold': args.threshold,
        'corpora': summaries, 'micro': evaluate(all_preds, all_rows, args.threshold),
        'macro_pair_f1': sum(s['metrics']['pair']['f1'] for s in summaries)/len(summaries),
        'calls': len(traces), 'input_tokens': total_tokens,
        'questions': sum(len(t['questions']) for t in traces),
        'extraction_input_tokens': sum(t['usage'].get('input_tokens', 0) for t in traces
                                      if not next(iter(t['questions'])).startswith('p')),
        'pair_input_tokens': sum(t['usage'].get('input_tokens', 0) for t in traces
                                if next(iter(t['questions'])).startswith('p')),
        'estimated_usd': total_tokens * .042 / 1_000_000,
        'span_limit_hits': sum(bool(p['span_limit_hit']) for p in all_preds),
        'chunked_records': sum(p['chunked'] for p in all_preds),
        'metric_note': 'Case-insensitive exact surface-string set F1 per record; not official cF1. '
                       'Span scores before pair filtering; baseline spans are final outputs.',
        'selection': 'SHA256(seed 20260923, corpus, ID), first N; dev excludes all trial text',
        'cache_files': [str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
                        for p in sorted(cache.glob(f'{args.mode}_r{args.revision}_{fingerprint}_*.json'))]}
    if args.split == 'dev':
        summary['baseline_macro_pair_f1'] = sum(s['baseline']['pair']['f1'] for s in summaries)/len(summaries)
    dest = args.out/f'{args.split}_{args.mode}_r{args.revision}_n{args.per_corpus}.json'
    dest.write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('corpora','cache_files')}, indent=2))


if __name__ == '__main__':
    main()
