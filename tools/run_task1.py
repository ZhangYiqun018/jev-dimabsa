#!/usr/bin/env python3
"""Task 1 (DimASR): joint V/A calibration and BM25-retrieved examples.

Builds on log 0005 (9 stratified examples + per-dimension shrink). Two changes
are compared on dev, each fitted on the same train calibration sample:

- ``joint`` calibration: per corpus, ridge regression of (gold V, gold A) on
  [V, A, |V - 5|, V x A] of the Jev scores, so each dimension can use the
  other; the ridge penalty is chosen by grouped 5-fold CV on the train sample.
- ``bm25`` examples: instead of the frozen 9 examples, each review gets the 9
  most similar same-corpus train reviews (BM25 over character bigrams, one
  explicit annotated aspect each, preferring one of the review's aspects). Train texts
  that occur in dev or test, and the review's own text, are never shown.

    .venv/bin/python tools/run_task1.py bm25 --split calibration   # Jev, retrieved examples
    .venv/bin/python tools/run_task1.py bm25 --split dev
    .venv/bin/python tools/run_task1.py dev    # fit, compare with the 0005 system, freeze
    .venv/bin/python tools/run_task1.py test   # frozen choice, official scorer
    .venv/bin/python tools/run_task1.py rerun  # fresh Jev test requests, same frozen choice

The fixed-example arm reuses the 0005 caches (reports/calibration_20260923/cache).
Selection keeps the 0005 rule: the best dev candidate replaces the frozen system
only if it is at least 0.02 RMSE better and the paired cluster-bootstrap 95%
interval is below zero. Only ID and Text reach Jev; requests and predictions are
cached under ignored reports/task1/cache/.
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jev.client import DEFAULT_MODEL, JevClient, format_va, score_to_va
from jev.data import _annotation_items, aspects_for_inference, load_jsonl, write_jsonl
from jev.fewshot import Example, ExampleSet, build_state, normalise, split_path, train_path
from jev.retrieval import Retriever
from jev.task2 import CachedClient
from runners.run_st1 import build_questions
from tools.analyze_st1_design import official_check
from tools.calibrate_st1 import CORPORA as ST1_CORPORA, PARALLEL, SEED, align, apply, digest, fit, infer, metrics, save

OUT = ROOT / 'reports/task1'
CACHE = OUT / 'cache'
FIXED = ROOT / 'reports/calibration_20260923'
SAMPLES = json.loads((FIXED / 'samples.json').read_text())
CORPORA = [f'{lang}_{domain}' for lang, domain in ST1_CORPORA]
SHOTS = 9
LAMBDAS = (.1, 1, 3, 10, 30, 100, 300)
BASELINE = ('fixed', 'shrink')
CONCURRENCY = 10


class CacheOnly:
    def ask(self, *args, **kwargs):
        raise RuntimeError('request not in cache (--cache-only)')


def source_of(c, split):
    lang, domain = c.split('_', 1)
    return FIXED / 'cache' / c / f'{c}_train_alltasks.jsonl' if split == 'calibration' else split_path(lang, domain, split)


def prediction_path(arm, c, split):
    return FIXED / 'cache' / c / f'{split}_s{SHOTS}.jsonl' if arm == 'fixed' else CACHE / c / f'{split}_bm25.jsonl'


# ------------------------------------------------------------ retrieved examples

def retriever_for(c):
    lang, domain = c.split('_', 1)
    banned = [r['Text'] for s in ('dev', 'test') for r in load_jsonl(split_path(lang, domain, s))]
    return Retriever(load_jsonl(train_path(lang, domain)), banned)


def retrieved_examples(retriever, record, c):
    wanted = {a.lower() for a in aspects_for_inference(record)}
    examples = []
    for row in retriever.select(record['Text'], n=3 * SHOTS, exclude=record['Text']):
        items = [x for x in _annotation_items(row) if x['Aspect'] != 'NULL']  # Task 1 aspects are explicit.
        if not items:
            continue
        item = next((x for x in items if x['Aspect'].lower() in wanted), items[0])
        valence, arousal = map(float, item['VA'].split('#'))
        examples.append(Example(row['ID'], row['Text'], item['Aspect'], valence, arousal))
        if len(examples) == SHOTS:
            break
    return ExampleSet(corpus=c, examples=examples, n_requested=SHOTS, strategy='bm25')


def infer_bm25(client, split):
    for c in CORPORA:
        path = prediction_path('bm25', c, split)
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        retriever = retriever_for(c)

        def run(record):
            aspects = aspects_for_inference(record)
            entry = {'ID': record['ID'], 'Text': record['Text'], 'Aspect_VA': []}
            if not aspects:
                return entry
            examples = retrieved_examples(retriever, record, c)
            try:
                response = client.ask(build_state(record['Text'], examples), build_questions(aspects, SHOTS))
            except Exception as exc:  # API errors can echo review text; keep details out of logs.
                print(f'{c} {record["ID"]}: {type(exc).__name__}; rerun to resume', flush=True)
                return None
            entry['_jev'] = {'model': response.model, 'usage': response.usage, 'attempts': response.attempts,
                             'examples': examples.source_ids,
                             'answers': {k: a.raw for k, a in response.answers.items()}}
            entry['Aspect_VA'] = [{'Aspect': a, 'VA': format_va(score_to_va(response.answers[f'v{i}'].score),
                                                                score_to_va(response.answers[f'a{i}'].score))}
                                  for i, a in enumerate(aspects)]
            return entry

        with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            entries = list(pool.map(run, load_jsonl(source_of(c, split))))
        if None in entries:
            sys.exit(f'{entries.count(None)} records failed; cached calls are kept, rerun to resume')
        write_jsonl(path, entries)
        print(f'{c} {split}: {len(entries)} records', flush=True)


# ------------------------------------------------------------ joint calibration

def design(x):
    v, a = x[:, 0], x[:, 1]
    return np.column_stack([v, a, np.abs(v - 5), v * a])


def ridge(z, y, lam):
    mu, sd = z.mean(axis=0), z.std(axis=0) + 1e-9
    s = (z - mu) / sd
    w = np.linalg.solve(s.T @ s + lam * np.eye(s.shape[1]), s.T @ (y - y.mean(axis=0)))
    return {'mu': mu.tolist(), 'sd': sd.tolist(), 'w': w.tolist(), 'b': y.mean(axis=0).tolist()}


def apply_joint(x, p):
    s = (design(x) - np.array(p['mu'])) / np.array(p['sd'])
    return np.clip(s @ np.array(p['w']) + np.array(p['b']), 1, 9)


def fit_joint(x, y, groups):
    unique = sorted(set(groups), key=lambda g: digest([SEED, str(g)]))
    fold_for = {g: i % 5 for i, g in enumerate(unique)}
    folds = np.array([fold_for[g] for g in groups])
    z = design(x)
    loss = {lam: sum(((apply_joint(x[folds == k], ridge(z[folds != k], y[folds != k], lam)) - y[folds == k])**2).sum()
                     for k in range(5)) for lam in LAMBDAS}
    lam = min(LAMBDAS, key=lambda l: (loss[l], l))
    return {**ridge(z, y, lam), 'lambda': lam}


def predict(method, x, p):
    return apply_joint(x, p['joint']) if method == 'joint' else apply(x, p['shrink'])


def load_arm(arm, split):
    """Per corpus: (x, y, groups, keys) aligned with the split's gold entries."""
    out = {}
    for c in CORPORA:
        groups = SAMPLES['corpora'][c]['groups'] if split == 'calibration' else None
        out[c] = align(source_of(c, split), prediction_path(arm, c, split), groups, c)
    return out


def bootstrap(errors_base, errors_new):
    """Paired cluster bootstrap of the micro RMSE difference (strata as in 0005)."""
    strata = {}
    for c, g, b, n in zip(*errors_base, errors_new):
        entry = strata.setdefault('parallel' if c in PARALLEL else c, {}).setdefault(str(g), np.zeros(3))
        entry += (b, n, 1)
    rng = np.random.default_rng(SEED)
    totals = np.zeros((2000, 3))
    for clusters in strata.values():
        values = np.array(list(clusters.values()))
        for i in range(2000):
            totals[i] += values[rng.integers(0, len(values), len(values))].sum(axis=0)
    delta = np.sqrt(totals[:, 1] / totals[:, 2]) - np.sqrt(totals[:, 0] / totals[:, 2])
    return np.quantile(delta, [.025, .975]).tolist()


def run_dev():
    if (OUT / 'selection.json').exists():
        sys.exit('Selection already frozen; use a new output directory for another experiment.')
    arms = ['fixed'] + (['bm25'] if all(prediction_path('bm25', c, s).exists()
                                        for c in CORPORA for s in ('calibration', 'dev')) else [])
    params, errors, report = {}, {}, {'candidates': {}}
    for arm in arms:
        cal, dev = load_arm(arm, 'calibration'), load_arm(arm, 'dev')
        params[arm] = {c: {**fit(*cal[c][:3]), 'joint': fit_joint(*cal[c][:3])} for c in CORPORA}
        for method in ('shrink', 'joint'):
            pred = {c: predict(method, dev[c][0], params[arm][c]) for c in CORPORA}
            y = np.concatenate([dev[c][1] for c in CORPORA])
            report['candidates'][f'{arm}/{method}'] = {
                'aggregate': metrics(np.concatenate([pred[c] for c in CORPORA]), y),
                'corpora': {c: metrics(pred[c], dev[c][1]) for c in CORPORA}}
            errors[(arm, method)] = ([c for c in CORPORA for _ in dev[c][1]],
                                     np.concatenate([dev[c][2] for c in CORPORA]),
                                     np.concatenate([((pred[c] - dev[c][1])**2).sum(axis=1) for c in CORPORA]))
    rmse = {k: report['candidates'][f'{k[0]}/{k[1]}']['aggregate']['RMSE_VA'] for k in errors}
    best = min(rmse, key=rmse.get)
    interval = bootstrap(errors[BASELINE], errors[best][2])
    accepted = best if rmse[BASELINE] - rmse[best] >= .02 and interval[1] < 0 else BASELINE
    report['selection'] = {'candidate': '/'.join(best), 'accepted': '/'.join(accepted),
                           'improvement': rmse[BASELINE] - rmse[best], 'delta_ci95': interval}
    save(OUT / 'parameters.json', params)
    save(OUT / 'dev_summary.json', report)
    save(OUT / 'selection.json', {**report['selection'], 'model': DEFAULT_MODEL,
                                  'parameters_sha256': digest(params), 'samples_sha256': digest(SAMPLES)})
    for k in sorted(rmse, key=rmse.get):
        print(f'dev {"/".join(k):14s} RMSE_VA {rmse[k]:.4f}')
    print(f"selected {report['selection']['accepted']} (candidate {report['selection']['candidate']}, "
          f"improvement {report['selection']['improvement']:.4f}, CI95 {np.round(interval, 4).tolist()})")


def run_test(client, rerun=False):
    """Score the frozen choice on test; ``rerun`` first repeats every fixed-example test request
    through the 0005 runner (no response cache), so the result reflects fresh Jev answers."""
    selection = json.loads((OUT / 'selection.json').read_text())
    params = json.loads((OUT / 'parameters.json').read_text())
    if digest(params) != selection['parameters_sha256'] or digest(SAMPLES) != selection['samples_sha256']:
        sys.exit('Frozen parameters or samples changed')
    arm, method = selection['accepted'].split('/')
    split = 'test'
    if arm == 'bm25':
        infer_bm25(client, 'test')
    elif rerun:
        split = 'test_rerun'
        for c in CORPORA:
            path = infer(FIXED, c, source_of(c, 'test'), split, SHOTS)
            config = lambda stage: json.loads(Path(f'{prediction_path(arm, c, stage)}.meta.json').read_text())['config']
            if path != prediction_path(arm, c, split) or digest(config(split)) != digest(config('test')):
                sys.exit(f'{c}: rerun request differs from the scored test request')
    test = {c: align(source_of(c, 'test'), prediction_path(arm, c, split), None, c) for c in CORPORA}
    report = {'system': selection['accepted'], 'predictions': split, 'model': DEFAULT_MODEL, 'corpora': {}}
    for c in CORPORA:
        x, y, _, keys = test[c]
        values = predict(method, x, params[arm][c])
        local = metrics(np.round(values, 2), y)
        local['official_RMSE_VA'] = official_check(source_of(c, 'test'), keys, values)
        if abs(local['official_RMSE_VA'] - local['RMSE_VA']) > .000051:
            sys.exit('Official scorer disagrees')
        report['corpora'][c] = local
    n = sum(m['n_gold'] for m in report['corpora'].values())
    report['aggregate'] = {'n_gold': n, 'RMSE_VA': float(np.sqrt(
        sum(m['n_gold'] * m['official_RMSE_VA']**2 for m in report['corpora'].values()) / n))}
    save(OUT / f'{split}_summary.json', report)
    print(f"{split} {selection['accepted']}: micro RMSE_VA {report['aggregate']['RMSE_VA']:.4f}")
    for c, m in report['corpora'].items():
        print(f"  {c:15s} {m['official_RMSE_VA']:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('step', choices=['bm25', 'dev', 'test', 'rerun'])
    parser.add_argument('--split', choices=['calibration', 'dev'], help='bm25 only')
    parser.add_argument('--cache-only', action='store_true', help='fail on any request missing from the cache')
    args = parser.parse_args()
    (CACHE / 'calls').mkdir(parents=True, exist_ok=True)
    client = CachedClient(CacheOnly() if args.cache_only else JevClient(timeout=120),
                          CACHE / 'calls', DEFAULT_MODEL, 'task1')
    if args.step == 'bm25':
        return infer_bm25(client.with_stage(f'bm25_{args.split}'), args.split)
    if args.step == 'dev':
        return run_dev()
    run_test(client.with_stage('bm25_test'), rerun=args.step == 'rerun')


if __name__ == '__main__':
    main()
