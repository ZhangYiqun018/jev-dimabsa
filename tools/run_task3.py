#!/usr/bin/env python3
"""Task 3 (DimASQP): a category for every pair of the frozen Task 2 system.

Task 3 dev/test are the Task 2 reviews in the same order (IDs may differ, e.g.
``_aste_`` vs ``_asqp_``); their gold equals Task 2 gold plus Category. Pairs and
V/A are therefore read from the Task 2 predictions (reports/task2/{split}/, dev
out-of-fold), and only the category is added:

    .venv/bin/python tools/run_task3.py fit    # Jev on a train sample, fit combiner weights
    .venv/bin/python tools/run_task3.py dev    # categories for Task 2 dev pairs, official Task 3 score
    .venv/bin/python tools/run_task3.py test --variant full

Per pair: Jev Choice over the corpus's train categories with retrieved train
examples (jev/categories.py), train lookups P(category | aspect) and
P(category | opinion), and the corpus prior, combined per language group by
weights fitted on train only. Train rows whose text occurs in dev or test are
excluded from every statistic and example.
Only ID and Text reach Jev; requests and per-record answers are cached under
reports/task3/cache/ (ignored); ``--cache-only`` fails instead of calling the API.
"""
import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jev.categories import ATTRIBUTES, CategoryChooser, Lookup, choose, fit_weights, glossary, inventory, pair_features, pool
from jev.client import DEFAULT_MODEL, JevClient
from jev.data import _annotation_items, load_jsonl, write_jsonl
from jev.fewshot import normalise
from jev.rerank import GROUPS, group_of
from jev.retrieval import Retriever
from jev.task2 import DATA, CORPORA, CachedClient, digest, official_score, record_key, save_json, split_path

OUT = ROOT / 'reports/task3'
CACHE = OUT / 'cache'
TASK2 = ROOT / 'reports/task2'
WEIGHTS = OUT / 'category_weights.json'
FIT_SAMPLE_PAIRS = 1000
CONCURRENCY = 10
STAGE = f'category_{digest(ATTRIBUTES)[:8]}'  # Per-record answers are tied to the criteria wording.
VARIANTS = {'lookup': ('aspect', 'opinion'), 'jev': ('jev',), 'both': ('jev', 'aspect'),
            'full': ('jev', 'aspect', 'opinion')}


class CacheOnly:
    def ask(self, *args, **kwargs):
        raise RuntimeError('request not in cache (--cache-only)')


def cached_stage(path, build):
    if path.exists():
        return json.loads(path.read_text())
    value = build()
    save_json(path, value)
    return value


def task3_path(corpus, split):
    return DATA / 'track_a/subtask_3' / corpus[:3] / f'{corpus}_{split}_task3.jsonl'


class Corpus:
    """Train statistics, categories, glossary and retriever, all without dev/test texts."""

    def __init__(self, corpus):
        banned = frozenset(normalise(r['Text']) for s in ('dev', 'test') for r in load_jsonl(split_path(corpus, s)))
        self.rows = pool(corpus, banned)
        self.categories = inventory(self.rows)
        self.terms = glossary(self.rows)
        self.lookup = Lookup(self.rows)
        self.retriever = Retriever(list(self.rows))


def unique_pairs(items):
    return list(dict.fromkeys((x['Aspect'], x['Opinion']) for x in items))


def ask_all(jobs, run):
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        results = list(executor.map(run, jobs))
    failed = sum(r is None for r in results)
    if failed:
        sys.exit(f'{failed} records failed; cached calls are kept, rerun to resume')
    return results


def categorise(client, info, corpus, split, record, pairs, exclude=None):
    path = CACHE / split / STAGE / f'{record_key(corpus, record)}.json'
    try:
        answer = cached_stage(path, lambda: CategoryChooser()(
            client.with_stage(f'category_{split}'), {'ID': record['ID'], 'Text': record['Text']},
            pairs, info.retriever, info.categories, info.terms, exclude))
    except Exception as exc:  # API errors can echo review text; keep details out of logs.
        print(f'{corpus} {record["ID"]}: {type(exc).__name__}; rerun to resume', flush=True)
        return None
    return {(p['Aspect'].lower(), p['Opinion'].lower()): p['probabilities'] for p in answer['pairs']}


def fit(client):
    """Jev answers for gold pairs of a train sample (own text excluded), then per-group weights."""
    items = {v: {g: [] for g in GROUPS} for v in VARIANTS}
    for corpus in CORPORA:
        info = Corpus(corpus)
        (CACHE / 'train' / STAGE).mkdir(parents=True, exist_ok=True)
        ordered = sorted(info.rows, key=lambda r: hashlib.sha256(f'task3-cat:{corpus}:{r["ID"]}'.encode()).hexdigest())
        chosen, covered = [], 0
        for row in ordered:
            chosen.append(row)
            covered += len(_annotation_items(row))
            if covered >= FIT_SAMPLE_PAIRS:
                break
        answers = ask_all(chosen, lambda row: categorise(client, info, corpus, 'train', row,
                                                         unique_pairs(_annotation_items(row)), row['Text']))
        for row, probs in zip(chosen, answers):
            for x in _annotation_items(row):
                lookup_rows = info.lookup.features(x['Aspect'], x['Opinion'], info.categories, without=row['Text'])
                gold = info.categories.index(x['Category'])
                for v, use in VARIANTS.items():
                    items[v][group_of(corpus)].append(
                        (pair_features(probs[(x['Aspect'].lower(), x['Opinion'].lower())], lookup_rows,
                                       info.categories, use), gold))
        print(f'{corpus}: {len(chosen)} train reviews, {covered} pairs', flush=True)
    weights = {v: {g: [float(x) for x in fit_weights(items[v][g])] for g in GROUPS} for v in VARIANTS}
    save_json(WEIGHTS, {'model': DEFAULT_MODEL, 'stage': STAGE, 'sample_pairs_per_corpus': FIT_SAMPLE_PAIRS,
                        'features': ['log p_jev', 'log p(c|aspect)', 'seen * log p(c|aspect)', 'log prior',
                                     'log p(c|opinion)', 'seen * log p(c|opinion)'],
                        'weights': weights})
    for v in VARIANTS:
        print(v, {g: [round(x, 3) for x in w] for g, w in weights[v].items()})


def predict(client, split, variants):
    """Categories for the Task 2 predictions of ``split``; official Task 3 score per variant."""
    weights = json.loads(WEIGHTS.read_text())['weights']
    task2 = json.loads((TASK2 / f'{split}_summary.json').read_text())['corpora']
    (CACHE / split / STAGE).mkdir(parents=True, exist_ok=True)
    report = {'split': split, 'model': DEFAULT_MODEL, 'stage': STAGE, 'pairs_from': f'reports/task2/{split}', 'variants': {}}
    per_corpus = {v: {} for v in variants}
    for corpus in CORPORA:
        info = Corpus(corpus)
        rows = load_jsonl(split_path(corpus, split))
        gold3 = load_jsonl(task3_path(corpus, split))
        assert [r['Text'] for r in rows] == [r['Text'] for r in gold3]
        pred2 = {r['ID']: r['Triplet'] for r in load_jsonl(TASK2 / split / f'{corpus}.jsonl')}
        answers = ask_all(rows, lambda row: categorise(client, info, corpus, split, row,
                                                       unique_pairs(pred2[row['ID']])))
        for v in variants:
            w = weights[v][group_of(corpus)]
            out, matched, correct = [], 0, 0
            for row, g3, probs in zip(rows, gold3, answers):
                gold = {}
                for x in g3['Quadruplet']:
                    gold.setdefault((x['Aspect'].lower(), x['Opinion'].lower()), set()).add(x['Category'])
                quads = []
                for t in pred2[row['ID']]:
                    key = (t['Aspect'].lower(), t['Opinion'].lower())
                    features = pair_features(probs[key], info.lookup.features(t['Aspect'], t['Opinion'], info.categories),
                                             info.categories, VARIANTS[v])
                    category = choose(w, features, info.categories)
                    quads.append({'Aspect': t['Aspect'], 'Category': category, 'Opinion': t['Opinion'], 'VA': t['VA']})
                    if key in gold:
                        matched += 1
                        correct += category in gold[key]
                out.append({'ID': g3['ID'], 'Quadruplet': quads})
            folder = OUT / split / v
            folder.mkdir(parents=True, exist_ok=True)
            write_jsonl(folder / f'{corpus}.jsonl', out)
            m = official_score(task3_path(corpus, split), folder / f'{corpus}.jsonl',
                               folder / f'{corpus}_scorer.txt', task=3)
            m['category_accuracy_on_matched_pairs'] = correct / max(1, matched)
            m['retention_vs_task2'] = m['cF1'] / task2[corpus]['cF1']
            per_corpus[v][corpus] = m
    for v in variants:
        corpora = per_corpus[v]
        report['variants'][v] = {'macro_cF1': sum(m['cF1'] for m in corpora.values()) / len(CORPORA),
                                 'corpora': corpora}
        print(f"{split} {v}: macro cF1 {report['variants'][v]['macro_cF1'] * 100:.2f}")
        for c, m in corpora.items():
            print(f"  {c:15s} cF1 {m['cF1'] * 100:6.2f}  retention {m['retention_vs_task2']:.3f}  "
                  f"category acc (matched) {m['category_accuracy_on_matched_pairs']:.3f}")
        if split == 'dev':
            test2 = json.loads((TASK2 / 'test_summary.json').read_text())['corpora']
            projected = sum(test2[c]['cF1'] * corpora[c]['retention_vs_task2'] for c in CORPORA) / len(CORPORA)
            report['variants'][v]['projected_test_macro_cF1'] = projected
            print(f'  projected test macro (Task 2 test x dev retention): {projected * 100:.2f}')
    save_json(OUT / f'{split}_summary.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('step', choices=['fit', 'dev', 'test'])
    parser.add_argument('--variant', default='full', choices=list(VARIANTS), help='test only')
    parser.add_argument('--cache-only', action='store_true', help='fail on any request missing from the cache')
    args = parser.parse_args()
    (CACHE / 'calls').mkdir(parents=True, exist_ok=True)
    client = CachedClient(CacheOnly() if args.cache_only else JevClient(timeout=120),
                          CACHE / 'calls', DEFAULT_MODEL, 'task3')
    if args.step == 'fit':
        return fit(client)
    predict(client, args.step, list(VARIANTS) if args.step == 'dev' else [args.variant])


if __name__ == '__main__':
    main()
