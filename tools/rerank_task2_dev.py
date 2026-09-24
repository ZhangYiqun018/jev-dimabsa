#!/usr/bin/env python3
"""Task 2 reranked pipeline on full dev: cross-validated selection, pair VA, official cF1.

Reads the cached signals of tools/collect_task2_signals.py (lattice Noul,
span/pair checks), the BIO r3 extraction and the lexicon-baseline answers.
Selection scores are out-of-fold (5 folds by record), so the reported dev cF1
does not score a record with a model fitted on it; the threshold is chosen on
dev and is therefore optimistic. Record: deving/20260924-task2-plan.md.

    .venv/bin/python tools/rerank_task2_dev.py --revision 2 --threshold 0.25
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from evaluate_task2_dev import BIO_FULL, CALIBRATION, OUT, Lexicon, bio_extraction
from jev.client import DEFAULT_MODEL, JevClient
from jev.data import _annotation_items, load_jsonl, write_jsonl
from jev.rerank import cross_validate, features, select
from jev.task2 import CORPORA, CachedClient, official_score, record_key, save_json, score_pair_list, split_path


def lexicon_probabilities(lexicon, row):
    answers = lexicon.rows[row['ID']].get('_jev', {}).get('answers', {})
    return {(a.lower(), o.lower()): answers[f'p{i}']['noul']
            for i, (a, o) in enumerate(lexicon.predictor.candidates(row['Text']))}


def load_items(revision):
    cache = OUT / 'cache'
    suffix = '2' if revision == 2 else ''
    params = json.loads(CALIBRATION.read_text())['0']
    items, rows_by_corpus = [], {}
    for corpus in CORPORA:
        lexicon = Lexicon(corpus, params[corpus]['shrink'])
        rows_by_corpus[corpus] = load_jsonl(split_path(corpus, 'dev'))
        for row in rows_by_corpus[corpus]:
            key = record_key(corpus, row)
            lattice = json.loads((cache / f'lattice{suffix}' / f'{key}.json').read_text())
            signals = {'lexicon': lexicon_probabilities(lexicon, row),
                       'spancheck': json.loads((cache / f'spancheck{suffix}' / f'{key}.json').read_text())['spans'],
                       'paircheck': {(p['Aspect'].lower(), p['Opinion'].lower()): p['probability'] for p in
                                     json.loads((cache / f'paircheck{suffix}' / f'{key}.json').read_text())['pairs']}}
            rows = features(corpus, row['Text'], lattice, bio_extraction(corpus, row), signals)
            gold = {(x['Aspect'].lower(), x['Opinion'].lower()) for x in _annotation_items(row)}
            items.append((corpus, row, rows, gold))
    return items, rows_by_corpus, params


def categorical(items, scored, threshold):
    macro = {}
    for corpus in CORPORA:
        tp = fp = fn = 0
        for (c, row, _, gold), s in zip(items, scored):
            if c != corpus:
                continue
            kept = set(select(s, row['Text'], threshold))
            tp += len(kept & gold)
            fp += len(kept - gold)
            fn += len(gold - kept)
        macro[corpus] = 2 * tp / (2 * tp + fp + fn)
    return sum(macro.values()) / len(macro), macro


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--revision', type=int, default=2, choices=[1, 2])
    parser.add_argument('--threshold', type=float)
    parser.add_argument('--va', default='task1', choices=['task1', 'task2'],
                        help='task1: frozen Task 1 zero-shot shrink; task2: lines fitted on Task 2 train pairs')
    args = parser.parse_args()
    items, rows_by_corpus, params = load_items(args.revision)
    scored = cross_validate([(c, row['ID'], rows, gold) for c, row, rows, gold in items])
    for t in (.2, .25, .3, .35):
        macro, per = categorical(items, scored, t)
        print(f'threshold {t}: macro catF1 (perfect VA, exact pairs) {macro * 100:.2f} | ' +
              ' '.join(f'{c[:6]} {v * 100:.1f}' for c, v in per.items()))
    if args.threshold is None:
        return
    name = f'rerank_r{args.revision}_t{args.threshold}' + ('_va2' if args.va == 'task2' else '')
    if args.va == 'task2':
        fitted = json.loads((OUT / 'va_calibration/parameters.json').read_text())
        params = {c: {'shrink': {k: fitted[c][k] for k in ('slope', 'intercept')}} for c in CORPORA}
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    client = CachedClient(JevClient(timeout=90), OUT / 'cache/calls', DEFAULT_MODEL, 'va',
                          fallback=[BIO_FULL / 'calls'])

    def va(job):
        (corpus, row, _, _), s = job
        text_l = row['Text'].lower()
        pairs = []
        for a, o in select(s, row['Text'], args.threshold):
            # Output the review's own casing; NULL stays NULL.
            surface = lambda x: 'NULL' if x == 'null' else row['Text'][text_l.find(x):text_l.find(x) + len(x)]
            pairs.append((surface(a), surface(o)))
        record = {'ID': row['ID'], 'Text': row['Text']}
        return corpus, row['ID'], (score_pair_list(client, record, pairs, params[corpus]['shrink'])['calibrated']
                                   if pairs else [])

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(va, zip(items, scored)))
    report = {'revision': args.revision, 'threshold': args.threshold, 'va': args.va, 'corpora': {}}
    for corpus in CORPORA:
        by_id = {i: t for c, i, t in results if c == corpus}
        pred = out / f'{corpus}.jsonl'
        write_jsonl(pred, [{'ID': r['ID'], 'Triplet': by_id[r['ID']]} for r in rows_by_corpus[corpus]])
        m = official_score(split_path(corpus, 'dev'), pred, out / f'{corpus}_scorer.txt')
        m['categorical_F1'] = 2 * m['categorical_TP'] / (2 * m['categorical_TP'] + m['FP'] + m['FN'])
        report['corpora'][corpus] = m
    report['macro_cF1'] = sum(m['cF1'] for m in report['corpora'].values()) / len(CORPORA)
    save_json(out / 'summary.json', report)
    print(f"{name}: macro cF1 {report['macro_cF1'] * 100:.4f}")
    for c, m in report['corpora'].items():
        print(f"  {c:15s} cF1 {m['cF1'] * 100:6.2f}  catF1 {m['categorical_F1'] * 100:6.2f}  "
              f"TP {m['categorical_TP']:.0f} FP {m['FP']} FN {m['FN']}")


if __name__ == '__main__':
    main()
