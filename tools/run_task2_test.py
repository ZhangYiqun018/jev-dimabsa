#!/usr/bin/env python3
"""Frozen Task 2 reranked pipeline on the official test split (run once).

Configuration frozen on full dev (deving/20260924-task2-plan.md):
BIO r3 extraction -> lattice r1 candidates (BIO marginals >= 0.2, affix variants,
train-lexicon matches) with Noul per pair -> example-conditioned span and pair
checks -> per-language-group logistic reranker fitted on all of dev, threshold
0.25 -> pair V/A with the Task 2 train calibration -> unchanged official scorer.
Only ID and Text reach Jev; test annotations are read by the scorer alone.
Every request and per-record stage is cached, so reruns resume.

    .venv/bin/python tools/run_task2_test.py
"""
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from evaluate_task2_dev import CALIBRATION, Lexicon
from rerank_task2_dev import lexicon_probabilities, load_items
from jev.client import DEFAULT_MODEL, JevClient
from jev.data import load_jsonl, write_jsonl
from jev.extraction import BIOExtractor
from jev.lattice import LatticePairer
from jev.rerank import features, fit_groups, group_of, score, select
from jev.spancheck import PairChecker, SpanChecker
from jev.task2 import CORPORA, CachedClient, official_score, record_key, save_json, score_pair_list, split_path
from jev.variants import Retriever

OUT = ROOT / 'reports/task2_test_20260924'
VA_PARAMETERS = ROOT / 'reports/task2_dev_20260924/va_calibration/parameters.json'
THRESHOLD = .25
CONCURRENCY = 10


def stage(path, build):
    if path.exists():
        return json.loads(path.read_text())
    value = build()
    save_json(path, value)
    return value


def main():
    cache = OUT / 'cache'
    for d in ('calls', 'extract', 'lattice', 'spancheck', 'paircheck', 'va'):
        (cache / d).mkdir(parents=True, exist_ok=True)
    shrink = json.loads(CALIBRATION.read_text())['0']
    va_params = {c: {k: v for k, v in p.items() if k in ('slope', 'intercept')}
                 for c, p in json.loads(VA_PARAMETERS.read_text()).items()}
    save_json(OUT / 'protocol.json', {
        'model': DEFAULT_MODEL, 'split': 'test', 'threshold': THRESHOLD,
        'reranker': 'per-language-group logistic regression, L2 5, fitted on all full-dev candidates',
        'lattice': 'revision 1 (no merging)', 'va_calibration': va_params,
        'dataset': json.loads((ROOT / 'data-version.json').read_text())['upstream_commit']})
    client = CachedClient(JevClient(timeout=120), cache / 'calls', DEFAULT_MODEL, 'test')
    lexicons = {c: Lexicon(c, shrink[c]['shrink'], split='test') for c in CORPORA}
    retrievers = {c: Retriever(load_jsonl(split_path(c, 'train')),
                               [r['Text'] for s in ('dev', 'test') for r in load_jsonl(split_path(c, s))])
                  for c in CORPORA}
    rows = {c: load_jsonl(split_path(c, 'test')) for c in CORPORA}
    # Round-robin corpora so an interrupted run covers every language.
    jobs = [(c, rs[i]) for i in range(max(map(len, rows.values()))) for c, rs in rows.items() if i < len(rs)]
    lock, done = threading.Lock(), [0]

    def signals(job):
        corpus, row = job
        record = {'ID': row['ID'], 'Text': row['Text']}  # Test annotations never reach Jev.
        key = record_key(corpus, row)
        try:
            extracted = stage(cache / 'extract' / f'{key}.json',
                              lambda: BIOExtractor()(client.with_stage('extract_pair'), record))
            lattice = stage(cache / 'lattice' / f'{key}.json',
                            lambda: LatticePairer()(client.with_stage('lattice_pair'), record, extracted,
                                                    corpus, lexicons[corpus].predictor))
            spans = stage(cache / 'spancheck' / f'{key}.json',
                          lambda: SpanChecker()(client.with_stage('spancheck'), record,
                                                lattice['candidates'], retrievers[corpus]))
            pairs = stage(cache / 'paircheck' / f'{key}.json',
                          lambda: PairChecker()(client.with_stage('paircheck'), record,
                                                lattice['pairs'], retrievers[corpus]))
        except Exception as exc:  # API errors can echo review text; keep details out of logs.
            print(f'{corpus} {row["ID"]}: {type(exc).__name__}; rerun to resume', flush=True)
            return None
        with lock:
            done[0] += 1
            if done[0] % 250 == 0:
                print(f'signals {done[0]}/{len(jobs)}', flush=True)
        return corpus, row, extracted, lattice, spans, pairs

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        collected = list(pool.map(signals, jobs))
    if None in collected:
        sys.exit(f'{collected.count(None)} records failed; rerun to resume')

    dev_items, _, _ = load_items(1)
    models = fit_groups([(c, row['ID'], r, gold) for c, row, r, gold in dev_items])

    def predict(item):
        corpus, row, extracted, lattice, spans, pairs = item
        sig = {'lexicon': lexicon_probabilities(lexicons[corpus], row), 'spancheck': spans['spans'],
               'paircheck': {(p['Aspect'].lower(), p['Opinion'].lower()): p['probability'] for p in pairs['pairs']}}
        scored = score(models[group_of(corpus)], features(corpus, row['Text'], lattice, extracted, sig))
        text, text_l = row['Text'], row['Text'].lower()
        surface = lambda x: 'NULL' if x == 'null' else text[text_l.find(x):text_l.find(x) + len(x)]
        chosen = [(surface(a), surface(o)) for a, o in select(scored, text, THRESHOLD)]
        record = {'ID': row['ID'], 'Text': text}
        va = stage(cache / 'va' / f'{record_key(corpus, row)}.json',
                   lambda: score_pair_list(client.with_stage('va'), record, chosen, va_params[corpus])
                   if chosen else {'calibrated': [], 'raw': [], 'trace': []})
        return corpus, row['ID'], va['calibrated']

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        results = list(pool.map(predict, collected))
    report = {'split': 'test', 'corpora': {}}
    for corpus in CORPORA:
        by_id = {i: t for c, i, t in results if c == corpus}
        pred = OUT / f'{corpus}.jsonl'
        write_jsonl(pred, [{'ID': r['ID'], 'Triplet': by_id[r['ID']]} for r in rows[corpus]])
        m = official_score(split_path(corpus, 'test'), pred, OUT / f'{corpus}_scorer.txt')
        m['categorical_F1'] = 2 * m['categorical_TP'] / (2 * m['categorical_TP'] + m['FP'] + m['FN'])
        report['corpora'][corpus] = m
    report['macro_cF1'] = sum(m['cF1'] for m in report['corpora'].values()) / len(CORPORA)
    ledger = [json.loads(p.read_text()) for p in (cache / 'calls').glob('*.json')]
    report['calls'] = len(ledger)
    report['input_tokens'] = sum(c['usage']['input_tokens'] for c in ledger)
    report['estimated_usd'] = report['input_tokens'] * .042 / 1e6
    report['returned_models'] = sorted({c['model'] for c in ledger})
    save_json(OUT / 'summary.json', report)
    print(f"test macro cF1 {report['macro_cF1'] * 100:.4f}")
    for c, m in report['corpora'].items():
        print(f"  {c:15s} cF1 {m['cF1'] * 100:6.2f}  catF1 {m['categorical_F1'] * 100:6.2f}  "
              f"TP {m['categorical_TP']:.0f} FP {m['FP']} FN {m['FN']}")


if __name__ == '__main__':
    main()
