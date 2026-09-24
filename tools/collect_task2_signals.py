#!/usr/bin/env python3
"""Collect extra Jev signals for Task 2 lattice candidates on full dev (cached per record).

    .venv/bin/python tools/collect_task2_signals.py spancheck
    .venv/bin/python tools/collect_task2_signals.py paircheck
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jev.client import DEFAULT_MODEL, JevClient
from jev.data import load_jsonl
from jev.spancheck import PairChecker, SpanChecker
from jev.task2 import CORPORA, CachedClient, record_key, save_json, split_path
from jev.variants import Retriever, example
from jev.extraction import BIOExtractor, tokenize
from jev.lattice import LatticePairer
sys.path.insert(0, str(ROOT / 'tools'))
from evaluate_task2_dev import CALIBRATION, Lexicon, bio_extraction

OUT = ROOT / 'reports/task2_dev_20260924/cache'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('signal', choices=['spancheck', 'paircheck', 'lattice2', 'spancheck2', 'paircheck2', 'bioex'])
    parser.add_argument('--split', default='dev')
    args = parser.parse_args()
    target = OUT / args.signal
    target.mkdir(parents=True, exist_ok=True)
    client = CachedClient(JevClient(timeout=90), OUT / 'calls', DEFAULT_MODEL, args.signal)
    retrievers = {c: Retriever(load_jsonl(split_path(c, 'train')),
                               [r['Text'] for s in ('dev', 'test') for r in load_jsonl(split_path(c, s))])
                  for c in CORPORA}
    params = json.loads(CALIBRATION.read_text())['0']
    predictors = {c: Lexicon(c, params[c]['shrink']).predictor for c in CORPORA}
    jobs = [(c, r) for c in CORPORA for r in load_jsonl(split_path(c, args.split))]

    def run(job):
        corpus, row = job
        key = record_key(corpus, row)
        path = target / f'{key}.json'
        if path.exists():
            return True
        try:
            record = {'ID': row['ID'], 'Text': row['Text']}
            if args.signal == 'bioex':
                # BIO r3 label questions with retrieved real examples instead of invented ones.
                def examples(text):
                    return [{**example(r), 'tokens': ' '.join(f'{i}|{t.text}' for i, t in enumerate(tokenize(r['Text'])))}
                            for r in retrievers[corpus].select(text)]
                save_json(path, BIOExtractor(examples=examples, pairs=False)(client, record))
                return True
            if args.signal == 'lattice2':
                # Revision 2 candidates (merged fragments); revision 1 answers are reused.
                old = json.loads((OUT / 'lattice' / f'{key}.json').read_text())
                known = {(p['Aspect'].lower(), p['Opinion'].lower()): p['probability'] for p in old['pairs']}
                result = LatticePairer(merge=True, known=known)(
                    client, record, bio_extraction(corpus, row), corpus, predictors[corpus])
                save_json(path, result)
                return True
            source = 'lattice2' if args.signal.endswith('2') else 'lattice'
            lattice = json.loads((OUT / source / f'{key}.json').read_text())
            if args.signal.startswith('spancheck'):
                done = {}
                if args.signal == 'spancheck2':
                    done = json.loads((OUT / 'spancheck' / f'{key}.json').read_text())['spans']
                pools = {r: [s for s in lattice['candidates'][r] if s not in done.get(r, {})]
                         for r in ('aspect', 'opinion')}
                result = SpanChecker()(client, record, pools, retrievers[corpus])
                for r in done:
                    result['spans'][r].update(done[r])
            else:
                done = {}
                if args.signal == 'paircheck2':
                    done = {(p['Aspect'].lower(), p['Opinion'].lower()): p for p in
                            json.loads((OUT / 'paircheck' / f'{key}.json').read_text())['pairs']}
                todo = [p for p in lattice['pairs'] if (p['Aspect'].lower(), p['Opinion'].lower()) not in done]
                result = PairChecker()(client, record, todo, retrievers[corpus])
                result['pairs'] += list(done.values())
            save_json(path, result)
            return True
        except Exception as exc:  # API errors can echo review text; keep details out of logs.
            print(f'{corpus} {row["ID"]}: {type(exc).__name__}', flush=True)
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        ok = list(pool.map(run, jobs))
    print(f'{sum(ok)}/{len(ok)} records complete')
    if not all(ok):
        sys.exit(1)


if __name__ == '__main__':
    main()
