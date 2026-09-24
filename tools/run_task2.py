#!/usr/bin/env python3
"""Task 2 (DimASTE) pipeline: BIO lattice candidates, example-conditioned Jev checks, reranker.

Prerequisite for each split: the lexicon baseline, whose Noul answers are
reranker features:

    .venv/bin/python runners/run.py --task 2 --split dev  --out reports/st2_baseline_20260923
    .venv/bin/python runners/run.py --task 2 --split test --out reports/st2_baseline_20260923

Then:

    .venv/bin/python tools/run_task2.py va     # pair V/A lines from train gold pairs
    .venv/bin/python tools/run_task2.py dev    # signals, 5-fold CV score, fit the reranker
    .venv/bin/python tools/run_task2.py test   # signals, frozen reranker, official score

Per record: BIO r3 extraction -> lattice candidates with one Noul per pair ->
span and pair checks with retrieved train examples. The per-language-group
logistic reranker scores each candidate pair; pairs are kept in score order
unless they overlap a kept pair on both roles, down to THRESHOLD. Kept pairs get
Jev V/A with the Task 2 train calibration and the unchanged official scorer.
Only ID and Text reach Jev. Every request and every per-record stage is cached
under reports/task2/cache/ (ignored: it contains dataset text), so runs resume;
``--cache-only`` fails instead of calling the API.
"""
import argparse
import hashlib
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jev.checks import PairChecker, SpanChecker
from jev.client import DEFAULT_MODEL, JevClient
from jev.data import _annotation_items, load_jsonl, write_jsonl
from jev.extraction import BIOExtractor
from jev.fewshot import normalise
from jev.lattice import LatticePairer
from jev.rerank import cross_validate, features, fit_groups, group_of, score, select
from jev.retrieval import Retriever
from jev.task2 import CORPORA, CachedClient, official_score, record_key, save_json, score_pair_list, split_path
from jev.triplets import TripletPredictor

OUT = ROOT / 'reports/task2'
CACHE = OUT / 'cache'
LEXICON = ROOT / 'reports/st2_baseline_20260923/cache'
TASK1_SHRINK = ROOT / 'reports/calibration_20260923/parameters.json'
VA_CALIBRATION = OUT / 'va_calibration.json'
RERANKER = OUT / 'reranker.json'
THRESHOLD = .25
VA_SAMPLE_PAIRS = 1000
CONCURRENCY = 10


class CacheOnly:
    def ask(self, *args, **kwargs):
        raise RuntimeError('request not in cache (--cache-only)')


class Lexicon:
    """Lexicon-baseline candidates and cached Noul answers for one corpus and split."""

    def __init__(self, corpus, split):
        base = split_path(corpus, 'train').parent
        shrink = json.loads(TASK1_SHRINK.read_text())['0'][corpus]['shrink']
        self.predictor = TripletPredictor(split_path(corpus, 'train'),
                                          [base / f'{corpus}_{s}_task2.jsonl' for s in ('dev', 'test')], shrink)
        meta = json.loads((LEXICON / f'{corpus}_{split}.jsonl.meta.json').read_text())
        assert meta['config']['lexicon_sha256'] == self.predictor.config['lexicon_sha256']
        self.rows = {r['ID']: r for r in load_jsonl(LEXICON / f'{corpus}_{split}.jsonl')}

    def probabilities(self, row):
        answers = self.rows[row['ID']].get('_jev', {}).get('answers', {})
        return {(a.lower(), o.lower()): answers[f'p{i}']['noul']
                for i, (a, o) in enumerate(self.predictor.candidates(row['Text']))}


def cached_stage(path, build):
    if path.exists():
        return json.loads(path.read_text())
    value = build()
    save_json(path, value)
    return value


def client_for(args):
    (CACHE / 'calls').mkdir(parents=True, exist_ok=True)
    return CachedClient(CacheOnly() if args.cache_only else JevClient(timeout=120),
                        CACHE / 'calls', DEFAULT_MODEL, 'task2')


def collect(split, client):
    """[(corpus, row, feature rows, gold pairs)] after running every signal stage."""
    for stage in ('extract', 'lattice', 'spancheck', 'paircheck'):
        (CACHE / split / stage).mkdir(parents=True, exist_ok=True)
    lexicons = {c: Lexicon(c, split) for c in CORPORA}
    retrievers = {c: Retriever(load_jsonl(split_path(c, 'train')),
                               [r['Text'] for s in ('dev', 'test') for r in load_jsonl(split_path(c, s))])
                  for c in CORPORA}
    rows = {c: load_jsonl(split_path(c, split)) for c in CORPORA}
    # Round-robin corpora so an interrupted run covers every language.
    jobs = [(c, rs[i]) for i in range(max(map(len, rows.values()))) for c, rs in rows.items() if i < len(rs)]
    lock, done = threading.Lock(), [0]

    def run(job):
        corpus, row = job
        record = {'ID': row['ID'], 'Text': row['Text']}  # Annotations never reach Jev.
        path = lambda stage: CACHE / split / stage / f'{record_key(corpus, row)}.json'
        try:
            extracted = cached_stage(path('extract'), lambda: BIOExtractor()(client.with_stage('extract'), record))
            lattice = cached_stage(path('lattice'), lambda: LatticePairer()(
                client.with_stage('lattice'), record, extracted, corpus, lexicons[corpus].predictor))
            spans = cached_stage(path('spancheck'), lambda: SpanChecker()(
                client.with_stage('spancheck'), record, lattice['candidates'], retrievers[corpus]))
            pairs = cached_stage(path('paircheck'), lambda: PairChecker()(
                client.with_stage('paircheck'), record, lattice['pairs'], retrievers[corpus]))
        except Exception as exc:  # API errors can echo review text; keep details out of logs.
            print(f'{corpus} {row["ID"]}: {type(exc).__name__}; rerun to resume', flush=True)
            return None
        signals = {'lexicon': lexicons[corpus].probabilities(row), 'spancheck': spans['spans'],
                   'paircheck': {(p['Aspect'].lower(), p['Opinion'].lower()): p['probability']
                                 for p in pairs['pairs']}}
        gold = {(x['Aspect'].lower(), x['Opinion'].lower()) for x in _annotation_items(row)}
        with lock:
            done[0] += 1
            if done[0] % 500 == 0:
                print(f'{split} signals {done[0]}/{len(jobs)}', flush=True)
        return corpus, row, features(corpus, row['Text'], lattice, extracted, signals), gold

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        items = list(pool.map(run, jobs))
    if None in items:
        sys.exit(f'{items.count(None)} records failed; cached calls are kept, rerun to resume')
    order = {(c, r['ID']): i for c in CORPORA for i, r in enumerate(rows[c])}
    return sorted(items, key=lambda it: (CORPORA.index(it[0]), order[(it[0], it[1]['ID'])]))


def predict_and_score(split, items, scored, client, label):
    """Select pairs, add V/A, write predictions and run the official scorer per corpus."""
    va_params = json.loads(VA_CALIBRATION.read_text())
    (CACHE / split / 'va').mkdir(parents=True, exist_ok=True)

    def run(job):
        (corpus, row, _, _), candidates = job
        text, text_l = row['Text'], row['Text'].lower()
        surface = lambda x: 'NULL' if x == 'null' else text[text_l.find(x):text_l.find(x) + len(x)]
        chosen = [(surface(a), surface(o)) for a, o in select(candidates, text, THRESHOLD)]
        calibration = {k: va_params[corpus][k] for k in ('slope', 'intercept')}
        va = cached_stage(CACHE / split / 'va' / f'{record_key(corpus, row)}.json',
                          lambda: score_pair_list(client.with_stage('va'), {'ID': row['ID'], 'Text': text},
                                                  chosen, calibration)
                          if chosen else {'raw': [], 'calibrated': [], 'trace': []})
        return corpus, row['ID'], va['calibrated']

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        results = list(pool.map(run, zip(items, scored)))
    out = OUT / split
    out.mkdir(parents=True, exist_ok=True)
    report = {'split': split, 'selection': label, 'threshold': THRESHOLD, 'model': DEFAULT_MODEL,
              'dataset': json.loads((ROOT / 'data-version.json').read_text())['upstream_commit'], 'corpora': {}}
    for corpus in CORPORA:
        by_id = {i: t for c, i, t in results if c == corpus}
        pred = out / f'{corpus}.jsonl'
        write_jsonl(pred, [{'ID': r['ID'], 'Triplet': by_id[r['ID']]} for r in load_jsonl(split_path(corpus, split))])
        m = official_score(split_path(corpus, split), pred, out / f'{corpus}_scorer.txt')
        m['categorical_F1'] = 2 * m['categorical_TP'] / (2 * m['categorical_TP'] + m['FP'] + m['FN'])
        report['corpora'][corpus] = m
    report['macro_cF1'] = sum(m['cF1'] for m in report['corpora'].values()) / len(CORPORA)
    save_json(OUT / f'{split}_summary.json', report)
    print(f"{split}: macro cF1 {report['macro_cF1'] * 100:.2f} ({label})")
    for c, m in report['corpora'].items():
        print(f"  {c:15s} cF1 {m['cF1'] * 100:6.2f}  perfect-VA F1 {m['categorical_F1'] * 100:6.2f}  "
              f"TP {m['categorical_TP']:.0f} FP {m['FP']} FN {m['FN']}")


def calibrate_va(client):
    """One least-squares line per V/A dimension from train gold pairs, per corpus.

    Train reviews are taken in SHA-256 order (skipping texts that also occur in
    dev or test) until VA_SAMPLE_PAIRS gold pairs are covered; their gold pairs
    are scored exactly like predictions.
    """
    identity = {'slope': [1., 1.], 'intercept': [0., 0.]}
    params = {}
    for corpus in CORPORA:
        banned = {normalise(r['Text']) for s in ('dev', 'test') for r in load_jsonl(split_path(corpus, s))}
        ordered = sorted(load_jsonl(split_path(corpus, 'train')),
                         key=lambda r: hashlib.sha256(f'task2-va:{corpus}:{r["ID"]}'.encode()).hexdigest())
        chosen, covered = [], 0
        for row in ordered:
            if normalise(row['Text']) in banned or not _annotation_items(row):
                continue
            chosen.append(row)
            covered += len(_annotation_items(row))
            if covered >= VA_SAMPLE_PAIRS:
                break

        def run(row):
            items = _annotation_items(row)
            unique = list(dict.fromkeys((x['Aspect'], x['Opinion']) for x in items))
            va = score_pair_list(client.with_stage('va_train'), {'ID': row['ID'], 'Text': row['Text']},
                                 unique, identity)
            gold = {}
            for x in items:
                gold.setdefault((x['Aspect'].lower(), x['Opinion'].lower()), []).append(
                    [float(v) for v in x['VA'].split('#')])
            return [[*map(float, t['VA'].split('#')), *g] for t in va['raw']
                    for g in gold[(t['Aspect'].lower(), t['Opinion'].lower())]]

        with ThreadPoolExecutor(max_workers=8) as pool:
            data = np.array([x for part in pool.map(run, chosen) for x in part])
        lines = [np.polyfit(data[:, d], data[:, 2 + d], 1) for d in (0, 1)]
        params[corpus] = {'slope': [float(a) for a, _ in lines], 'intercept': [float(b) for _, b in lines],
                          'reviews': len(chosen), 'pairs': len(data)}
        print(corpus, params[corpus], flush=True)
    save_json(VA_CALIBRATION, params)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('step', choices=['va', 'dev', 'test'])
    parser.add_argument('--cache-only', action='store_true', help='fail on any request missing from the cache')
    args = parser.parse_args()
    client = client_for(args)
    if args.step == 'va':
        return calibrate_va(client)
    items = collect(args.step, client)
    keyed = [(c, row['ID'], rows, gold) for c, row, rows, gold in items]
    if args.step == 'dev':
        # Dev is also the reranker's training data: report out-of-fold scores, then fit on all of it.
        predict_and_score('dev', items, cross_validate(keyed), client, '5-fold out-of-fold by record')
        models = fit_groups(keyed)
        save_json(RERANKER, {'threshold': THRESHOLD, 'l2': 5., 'trained_on': 'full dev, all candidates',
                             'weights': {g: [float(x) for x in w] for g, w in models.items()}})
        return
    weights = json.loads(RERANKER.read_text())['weights']
    scored = [score(np.array(weights[group_of(c)]), rows) for c, _, rows, _ in keyed]
    predict_and_score('test', items, scored, client, 'reranker fitted on full dev')


if __name__ == '__main__':
    main()
