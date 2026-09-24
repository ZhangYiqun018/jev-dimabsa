#!/usr/bin/env python3
"""Fit Task 2 pair V/A calibration on train gold pairs; frozen for dev and test.

For each corpus, train reviews are taken in SHA-256 order (excluding texts that
also occur in dev or test) until at least SAMPLE_PAIRS gold pairs are covered.
Their gold (aspect, opinion) pairs are scored with the same V/A questions and
batching as predictions (jev.task2.score_pair_list), and one least-squares line
per dimension maps raw to gold. Only train annotations are used.

    .venv/bin/python tools/calibrate_task2_va.py
"""
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jev.client import DEFAULT_MODEL, JevClient
from jev.data import _annotation_items, load_jsonl
from jev.fewshot import normalise
from jev.task2 import CORPORA, CachedClient, save_json, score_pair_list, split_path

OUT = ROOT / 'reports/task2_dev_20260924/va_calibration'
SAMPLE_PAIRS = 1000
IDENTITY = {'slope': [1., 1.], 'intercept': [0., 0.]}


def sample(corpus):
    banned = {normalise(r['Text']) for s in ('dev', 'test') for r in load_jsonl(split_path(corpus, s))}
    rows = sorted(load_jsonl(split_path(corpus, 'train')),
                  key=lambda r: hashlib.sha256(f'task2-va:{corpus}:{r["ID"]}'.encode()).hexdigest())
    chosen, pairs = [], 0
    for row in rows:
        if normalise(row['Text']) in banned or not _annotation_items(row):
            continue
        chosen.append(row)
        pairs += len(_annotation_items(row))
        if pairs >= SAMPLE_PAIRS:
            break
    return chosen


def main():
    (OUT / 'calls').mkdir(parents=True, exist_ok=True)
    client = CachedClient(JevClient(timeout=90), OUT / 'calls', DEFAULT_MODEL, 'va_train')
    params = {}
    for corpus in CORPORA:
        rows = sample(corpus)

        def run(row):
            items = _annotation_items(row)
            unique = list(dict.fromkeys((x['Aspect'], x['Opinion']) for x in items))
            va = score_pair_list(client, {'ID': row['ID'], 'Text': row['Text']}, unique, IDENTITY)
            gold = {}
            for x in items:
                gold.setdefault((x['Aspect'].lower(), x['Opinion'].lower()), []).append(
                    [float(v) for v in x['VA'].split('#')])
            out = []
            for t in va['raw']:
                for g in gold[(t['Aspect'].lower(), t['Opinion'].lower())]:
                    out.append([*map(float, t['VA'].split('#')), *g])
            return out

        with ThreadPoolExecutor(max_workers=8) as pool:
            data = np.array([x for rows_ in pool.map(run, rows) for x in rows_])
        slope, intercept = [], []
        for d in (0, 1):
            a, b = np.polyfit(data[:, d], data[:, 2 + d], 1)
            slope.append(float(a))
            intercept.append(float(b))
        params[corpus] = {'slope': slope, 'intercept': intercept, 'reviews': len(rows), 'pairs': len(data)}
        print(corpus, params[corpus], flush=True)
    save_json(OUT / 'parameters.json', params)


if __name__ == '__main__':
    main()
