#!/usr/bin/env python3
"""Task 2 development pipeline on full dev: cached extraction, decoding rules, pair VA, official cF1.

Each named configuration chooses, per corpus, the extraction source (BIO r3 or
the frozen lexicon baseline), the deterministic decoding rules of
jev/postprocess.py and the pair threshold. BIO pairs are re-scored for V/A on
their final surface strings; requests identical to historical ones reuse the
BIO r3 cache. Lexicon pairs keep the V/A answers of the frozen baseline run.
Records: deving/20260924-task2-stack.md.

    .venv/bin/python tools/evaluate_task2_dev.py stack
"""
import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jev.client import DEFAULT_MODEL, JevClient, format_va, score_to_va
from jev.data import load_jsonl, write_jsonl
from jev.extraction import EXTRACTOR_ID
from jev.lattice import LATTICE_THRESHOLD, LatticePairer, suppress
from jev.variants import Retriever, VariantChooser, apply_choice
from jev.postprocess import decode
from jev.task2 import (CORPORA, PAIR_THRESHOLD, CachedClient, digest, official_score,
                       record_key, save_json, score_pair_list, split_path)
from jev.triplets import TripletPredictor

OUT = ROOT / 'reports/task2_dev_20260924'
CALIBRATION = ROOT / 'reports/calibration_20260923/parameters.json'
BIO_FULL = ROOT / 'reports/bio_r3_full_dev_20260923/cache'
BIO_DEV96 = ROOT / 'reports/extraction_comparison_20260923/dev_bio/cache'
LEXICON = ROOT / 'reports/st2_baseline_20260923/cache'
CONCURRENCY = 8

BOTH = ('aspect', 'opinion')
A1 = {'null_policy': True, 'overlap': True}


def affix_roles(corpus):
    # English opinion normalisation lowered eng_laptop on dev (plan A3b), so only aspects.
    return ('aspect',) if corpus.startswith('eng_') else BOTH


CONFIGS = {
    'r0': lambda c: {'source': 'bio', 'rules': {}},
    'a1': lambda c: {'source': 'bio', 'rules': A1},
    'a1_affix': lambda c: {'source': 'bio', 'rules': {**A1, 'affix': affix_roles(c)}},
    # jpn routed to the frozen lexicon pipeline at the BIO threshold, plus BIO NULL-aspect pairs.
    'stack': lambda c: ({'source': 'lexicon', 'lexicon_threshold': .65, 'merge_bio_null': True,
                         'rules': {**A1, 'affix': BOTH}}
                        if c == 'jpn_hotel' else
                        {'source': 'bio', 'rules': {**A1, 'affix': affix_roles(c)}}),
    # Stack pairs, then one boundary-variant Choice per pair and role with BM25 train examples.
    'stack_v1': lambda c: {**CONFIGS['stack'](c), 'variants': True},
    'stack_v1_keep': lambda c: {**CONFIGS['stack'](c), 'variants': True, 'drop_none': False},
    # Lattice candidates + lexicon matches, Noul per pair, overlap suppression; one path for all corpora.
    'lattice': lambda c: {'source': 'lattice', 'threshold': PAIR_THRESHOLD, 'rules': {}},
}


def bio_extraction(corpus, row):
    key = record_key(corpus, row)
    for path in (BIO_DEV96 / f'bio_r3_{EXTRACTOR_ID}_{key}.json',
                 BIO_FULL / f'extract_{EXTRACTOR_ID}_{key}.json'):
        if path.exists():
            return json.loads(path.read_text())
    raise FileNotFoundError(f'no cached BIO r3 extraction for {corpus} {row["ID"]}')


class Lexicon:
    """Rebuilds the frozen lexicon baseline's dev output at any pair threshold."""

    def __init__(self, corpus, calibration, split='dev'):
        base = split_path(corpus, 'train').parent
        self.predictor = TripletPredictor(split_path(corpus, 'train'),
                                          [base / f'{corpus}_{s}_task2.jsonl' for s in ('dev', 'test')],
                                          calibration)
        meta = json.loads((LEXICON / f'{corpus}_{split}.jsonl.meta.json').read_text())
        assert meta['config']['lexicon_sha256'] == self.predictor.config['lexicon_sha256']
        self.rows = {r['ID']: r for r in load_jsonl(LEXICON / f'{corpus}_{split}.jsonl')}
        self.calibration = calibration

    def triplets(self, row, threshold):
        saved = self.rows[row['ID']]
        answers = saved.get('_jev', {}).get('answers', {})
        out = []
        for i, (a, o) in enumerate(self.predictor.candidates(row['Text'])):
            if answers[f'p{i}']['noul'] < threshold:
                continue
            va = [score_to_va(answers[f'{d}{i}']['score']) for d in ('v', 'a')]
            va = [min(9., max(1., b + s * x)) for x, s, b in
                  zip(va, self.calibration['slope'], self.calibration['intercept'])]
            out.append({'Aspect': a, 'Opinion': o, 'VA': format_va(*va),
                        'probability': answers[f'p{i}']['noul']})
        return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', choices=CONFIGS)
    args = parser.parse_args()
    out = OUT / args.config
    calls = OUT / 'cache/calls'
    for d in (out, calls):
        d.mkdir(parents=True, exist_ok=True)
    params = json.loads(CALIBRATION.read_text())['0']
    specs = {c: CONFIGS[args.config](c) for c in CORPORA}
    save_json(out / 'protocol.json', {'model': DEFAULT_MODEL, 'config': args.config,
                                      'specs': {c: {**s, 'rules': {k: sorted(v) if isinstance(v, tuple) else v
                                                                    for k, v in s['rules'].items()}}
                                                for c, s in specs.items()},
                                      'bio_extractor_id': EXTRACTOR_ID, 'pair_threshold': PAIR_THRESHOLD,
                                      'lattice_threshold': LATTICE_THRESHOLD,
                                      'calibration': 'Task1 zero-shot shrink, frozen'})
    client = CachedClient(JevClient(timeout=90), calls, DEFAULT_MODEL, 'va', fallback=[BIO_FULL / 'calls'])
    lexicons = {c: Lexicon(c, params[c]['shrink']) for c in CORPORA if specs[c]['source'] == 'lexicon'}
    predictors = {c: Lexicon(c, params[c]['shrink']).predictor for c in CORPORA
                  if specs[c]['source'] == 'lattice'}
    retrievers = {c: Retriever(load_jsonl(split_path(c, 'train')),
                               [r['Text'] for s in ('dev', 'test') for r in load_jsonl(split_path(c, s))])
                  for c in CORPORA if specs[c].get('variants')}
    for d in ('lattice', 'variants'):
        (OUT / 'cache' / d).mkdir(parents=True, exist_ok=True)
    rows = {c: load_jsonl(split_path(c, 'dev')) for c in CORPORA}
    jobs = [(c, r) for c in CORPORA for r in rows[c]]
    lock, done = threading.Lock(), [0]

    def run(job):
        try:
            return attempt(job)
        except Exception as exc:  # Details stay out of logs: API errors can echo review text.
            print(f'{job[0]} {job[1]["ID"]}: {type(exc).__name__}; rerun to resume', flush=True)
            return None

    def attempt(job):
        corpus, row = job
        spec, record = specs[corpus], {'ID': row['ID'], 'Text': row['Text']}
        triplets = []
        bio_pairs = []
        if spec['source'] == 'lexicon':
            triplets = [{k: t[k] for k in ('Aspect', 'Opinion', 'VA')}
                        for t in lexicons[corpus].triplets(row, spec['lexicon_threshold'])]
            if spec.get('merge_bio_null'):
                seen = {(t['Aspect'].lower(), t['Opinion'].lower()) for t in triplets}
                bio_pairs = [p for p in decode(bio_extraction(corpus, row), row['Text'], corpus,
                                               spec['rules'], PAIR_THRESHOLD)
                             if p['Aspect'] == 'NULL' and ('null', p['Opinion'].lower()) not in seen]
        elif spec['source'] == 'lattice':
            path = OUT / 'cache/lattice' / f'{record_key(corpus, row)}.json'
            if path.exists():
                paired = json.loads(path.read_text())
            else:
                paired = LatticePairer()(client.with_stage('lattice_pair'), record,
                                         bio_extraction(corpus, row), corpus, predictors[corpus])
                save_json(path, paired)
            bio_pairs = suppress(paired['pairs'], row['Text'], spec['threshold'])
        else:
            bio_pairs = decode(bio_extraction(corpus, row), row['Text'], corpus,
                               spec['rules'], PAIR_THRESHOLD)
        if spec.get('variants'):
            key = record_key(corpus, row)
            pairs = [(t['Aspect'], t['Opinion']) for t in triplets] + \
                    [(p['Aspect'], p['Opinion']) for p in bio_pairs]
            path = OUT / 'cache/variants' / f'{key}.json'
            if path.exists():
                chosen = json.loads(path.read_text())
            else:
                pools = json.loads((OUT / 'cache/lattice' / f'{key}.json').read_text())['candidates']
                chosen = VariantChooser()(client.with_stage('variants'), record, pairs, pools,
                                          retrievers[corpus])
                save_json(path, chosen)
            pairs = apply_choice([tuple(p) for p in chosen['pairs']], chosen['answers'],
                                 spec.get('drop_none', True))
            triplets = score_pair_list(client, record, pairs, params[corpus]['shrink'])['calibrated'] if pairs else []
        elif bio_pairs:
            va = score_pair_list(client, record, [(p['Aspect'], p['Opinion']) for p in bio_pairs],
                                 params[corpus]['shrink'])
            triplets += va['calibrated']
        with lock:
            done[0] += 1
            if done[0] % 200 == 0:
                print(f'{done[0]}/{len(jobs)}', flush=True)
        return corpus, row['ID'], triplets

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        results = list(pool.map(run, jobs))
    if None in results:
        sys.exit(f'{results.count(None)} records failed; cached calls are kept, rerun to resume')
    report = {'config': args.config, 'corpora': {}}
    for corpus in CORPORA:
        by_id = {i: t for c, i, t in results if c == corpus}
        pred = out / f'{corpus}.jsonl'
        write_jsonl(pred, [{'ID': r['ID'], 'Triplet': by_id[r['ID']]} for r in rows[corpus]])
        metrics = official_score(split_path(corpus, 'dev'), pred, out / f'{corpus}_scorer.txt')
        tp = metrics['categorical_TP']
        metrics['categorical_F1'] = 2 * tp / (2 * tp + metrics['FP'] + metrics['FN'])  # Perfect VA.
        report['corpora'][corpus] = metrics
    report['macro_cF1'] = sum(m['cF1'] for m in report['corpora'].values()) / len(CORPORA)
    report['macro_categorical_F1'] = sum(m['categorical_F1'] for m in report['corpora'].values()) / len(CORPORA)
    save_json(out / 'summary.json', report)
    print(f"{args.config}: macro cF1 {report['macro_cF1'] * 100:.4f}  "
          f"perfect-VA {report['macro_categorical_F1'] * 100:.4f}")
    for c, m in report['corpora'].items():
        print(f"  {c:15s} cF1 {m['cF1'] * 100:6.2f}  catF1 {m['categorical_F1'] * 100:6.2f}  "
              f"TP {m['categorical_TP']:.0f} FP {m['FP']} FN {m['FN']}")


if __name__ == '__main__':
    main()
