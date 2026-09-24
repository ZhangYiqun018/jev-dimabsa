#!/usr/bin/env python3
"""Task 2 BIO r3 on the full dev split: extraction, pair VA, official cF1.

Frozen configuration: BIO r3 extractor, Noul pair threshold 0.65, Task 1
zero-shot shrink calibration transferred unchanged. Every Jev request is cached
under reports/bio_r3_full_dev_20260923/cache/calls, so a rerun with a complete
cache makes no API calls. Record: deving/20260923-bio-r3-full-dev.md.
"""
import hashlib
import json
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jev.client import DEFAULT_MODEL, JevClient
from jev.data import load_jsonl, write_jsonl
from jev.extraction import EXTRACTOR_ID, BIOExtractor
from jev.task2 import (CORPORA, PAIR_THRESHOLD, VA_BATCH_PAIRS, CachedClient, ao_metrics,
                       digest, official_score, record_key, save_json, score_pairs, split_path)

OUT = ROOT / 'reports/bio_r3_full_dev_20260923'
CALIBRATION = ROOT / 'reports/calibration_20260923/parameters.json'
# dev96 extractions from the earlier BIO/SE comparison, reused as-is.
DEV96_CACHE = ROOT / 'reports/extraction_comparison_20260923/dev_bio/cache'
CONCURRENCY = 8


def protocol(params):
    return {'model': DEFAULT_MODEL, 'method': 'bio_r3', 'extractor_id': EXTRACTOR_ID,
            'pair_threshold': PAIR_THRESHOLD, 'va_batch_pairs': VA_BATCH_PAIRS,
            'dataset': json.loads((ROOT / 'data-version.json').read_text())['upstream_commit'],
            'calibration': 'Task1 zero-shot train-fitted shrink; previously selected on Task1 dev; frozen, no refit',
            'calibration_sha256': hashlib.sha256(CALIBRATION.read_bytes()).hexdigest(),
            'parameters': {c: params[c]['shrink'] for c in CORPORA},
            'concurrency': CONCURRENCY, 'null_opinion_supported': False,
            'primary': 'calibrated cF1', 'secondary': 'raw cF1 and exact AO F1'}


def main():
    cache = OUT / 'cache'
    calls = cache / 'calls'
    calls.mkdir(parents=True, exist_ok=True)
    params = json.loads(CALIBRATION.read_text())['0']
    config = protocol(params)
    save_json(OUT / 'protocol.json', config)
    rows = {c: load_jsonl(split_path(c, 'dev')) for c in CORPORA}
    print('Full dev counts: ' + json.dumps({c: len(rs) for c, rs in rows.items()}), flush=True)
    # Round-robin corpora so an interrupted run covers every language.
    jobs = [(c, rs[i]) for i in range(max(map(len, rows.values())))
            for c, rs in rows.items() if i < len(rs)]
    cached = CachedClient(JevClient(timeout=90), calls, DEFAULT_MODEL, 'extract_pair')
    lock, finished = threading.Lock(), 0

    def run(job):
        nonlocal finished
        corpus, row = job
        record = {'ID': row['ID'], 'Text': row['Text']}  # Gold never reaches the model.
        key = record_key(corpus, row)
        reused = DEV96_CACHE / f'bio_r3_{EXTRACTOR_ID}_{key}.json'
        extract_path = cache / f'extract_{EXTRACTOR_ID}_{key}.json'
        va_path = cache / f'va_{digest([config, corpus, row["ID"], row["Text"]])[:20]}.json'
        try:
            if reused.exists():
                extracted = json.loads(reused.read_text())
            elif extract_path.exists():
                extracted = json.loads(extract_path.read_text())
            else:
                extracted = BIOExtractor()(cached, record)
                save_json(extract_path, extracted)
            if va_path.exists():
                va = json.loads(va_path.read_text())
            else:
                va = score_pairs(cached.with_stage('va'), record, extracted, params[corpus]['shrink'])
                save_json(va_path, va)
        except Exception as exc:
            # Details stay in the ignored cache: API errors can echo review text.
            save_json(va_path.with_name(va_path.stem + '_error.json'),
                      {'type': type(exc).__name__, 'error': str(exc)})
            print(f'{corpus} {row["ID"]}: {type(exc).__name__}; cached calls retained', flush=True)
            return corpus, row, None, None, False, type(exc).__name__
        with lock:
            finished += 1
            if finished % 20 == 0 or finished == len(jobs):
                print(f'Completed {finished}/{len(jobs)}', flush=True)
        return corpus, row, extracted, va, reused.exists(), None

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        results = list(pool.map(run, jobs))

    done = [r for r in results if r[2] is not None]
    report_rows = []
    for corpus, records in rows.items():
        group = [r for r in done if r[0] == corpus]
        row = {'corpus': corpus, 'requested': len(records), 'completed': len(group),
               'complete': len(group) == len(records),
               'ao': ao_metrics([r[2] for r in group], [r[1] for r in group])}
        if row['complete']:
            by_id = {r[1]['ID']: r for r in group}
            for variant in ('raw', 'calibrated'):
                pred = cache / f'{corpus}_{variant}.jsonl'
                write_jsonl(pred, [{'ID': r['ID'], 'Triplet': by_id[r['ID']][3][variant]}
                                   for r in records])
                row[variant] = official_score(split_path(corpus, 'dev'), pred,
                                              cache / f'{corpus}_{variant}_scorer.txt')
        report_rows.append(row)

    ledger = [json.loads(p.read_text()) for p in calls.glob('*.json')]
    stages = Counter()
    for call in ledger:
        stages[call['stage']] += call['usage']['input_tokens']
    complete = len(done) == len(jobs)
    macro = (lambda f: sum(f(r) for r in report_rows) / len(report_rows)) if complete else (lambda f: None)
    report = {
        'config': config, 'complete': complete, 'requested': len(jobs), 'completed': len(done),
        'corpora': report_rows,
        'macro_ao_f1': macro(lambda r: r['ao']['pair']['f1']),
        'macro_calibrated_cf1': macro(lambda r: r['calibrated']['cF1']),
        'macro_raw_cf1': macro(lambda r: r['raw']['cF1']),
        'micro_ao': ao_metrics([r[2] for r in done], [r[1] for r in done]),
        'historical_extraction_records_reused': sum(r[4] for r in done),
        'new_calls': len(ledger), 'new_http_attempts': sum(c['attempts'] for c in ledger),
        'new_input_tokens': sum(stages.values()), 'new_stage_input_tokens': dict(stages),
        'new_estimated_usd': sum(stages.values()) * .042 / 1e6,
        'logical_input_tokens_completed': sum(t['usage']['input_tokens'] for r in done
                                              for t in r[2]['trace'] + r[3]['trace']),
        'returned_models': sorted({c['model'] for c in ledger}),
        'missing': [{'corpus': r[0], 'ID': r[1]['ID'], 'reason': r[5]} for r in results if r[2] is None],
        'note': 'Full development evaluation, not official test rank. No refitting. '
                'AO uses surface sets; official cF1 uses unchanged organizer scorer.'}
    save_json(OUT / 'summary.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('config', 'corpora', 'micro_ao')},
                     indent=2))


if __name__ == '__main__':
    main()
