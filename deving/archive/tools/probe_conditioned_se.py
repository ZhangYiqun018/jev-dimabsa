#!/usr/bin/env python3
"""Trial-only direct conditioned SE probe, with per-request resume and a small budget."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jev.client import JevClient, Answer, Response, DEFAULT_MODEL
from jev.combined_extraction import Retriever
from jev.conditioned_se import ConditionedSE, CONFIG
from jev.data import load_jsonl, _annotation_items
from iterate_extraction import CORPORA, DATA, evaluate
from combined_experiment import digest, save, BudgetReached
import hashlib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--per-corpus', type=int, default=6)
    parser.add_argument('--out', type=Path, default=ROOT/'reports/conditioned_se_20260923')
    parser.add_argument('--concurrency', type=int, default=3)
    parser.add_argument('--max-input-tokens', type=int, default=1_500_000)
    args = parser.parse_args()
    calls = args.out/'cache/calls'
    calls.mkdir(parents=True, exist_ok=True)
    sources = ['jev/conditioned_se.py', 'jev/combined_extraction.py', 'jev/extraction.py',
               'tools/probe_conditioned_se.py']
    config = {**CONFIG, 'model': DEFAULT_MODEL,
              'fingerprint': digest({p: (ROOT/p).read_text() for p in sources})[:12],
              'dataset': json.loads((ROOT/'data-version.json').read_text())['upstream_commit']}
    trial_texts = [r['Text'] for p in (DATA/'trial').glob('*_alltasks.jsonl') for r in load_jsonl(p)]
    banned = list(trial_texts)
    for c in CORPORA:
        banned.extend(r['Text'] for r in load_jsonl(DATA/'track_a/subtask_2'/c[:3]/f'{c}_dev_task2.jsonl'))
    selected, retrievers = {}, {}
    for corpus in CORPORA:
        path = DATA/'trial'/f'{corpus}_trial_alltasks.jsonl'
        if not path.exists():
            continue
        rows = load_jsonl(path)
        rows.sort(key=lambda r: hashlib.sha256(('20260923:'+corpus+':'+r['ID']).encode()).digest())
        selected[corpus] = rows[:args.per_corpus]
        train = DATA/'track_a/subtask_2'/corpus[:3]/f'{corpus}_train_alltasks.jsonl'
        retrievers[corpus] = Retriever(load_jsonl(train), corpus, banned)
    client = JevClient(timeout=60)
    lock = threading.Lock()
    request_locks = {}
    spent = sum(json.loads(p.read_text())['usage'].get('input_tokens', 0) for p in calls.glob('*.json'))

    class CachedClient:
        def ask(self, state, questions):
            nonlocal spent
            payload = {'config': config, 'state': state, 'questions': questions}
            key = digest(payload)
            path = calls/f'{key}.json'
            with lock:
                key_lock = request_locks.setdefault(key, threading.Lock())
            with key_lock:
                if path.exists():
                    r = json.loads(path.read_text())
                    return Response(r['model'], {k: Answer(k, 'choice', v) for k,v in r['answers'].items()},
                                    r['usage'], r['attempts'])
                with lock:
                    if spent >= args.max_input_tokens:
                        raise BudgetReached()
                result = client.ask(state, questions, attempts=2)
                save(path, {**payload, 'usage': result.usage, 'model': result.model,
                            'attempts': result.attempts,
                            'answers': {k:a.raw for k,a in result.answers.items()}})
                with lock:
                    spent += result.usage.get('input_tokens', 0)
                return result

    def run(job):
        corpus, row = job
        key = digest({'config': config, 'corpus': corpus, 'text': row['Text'], 'ID': row['ID'],
                      'examples': retrievers[corpus].select(row['Text'])})[:20]
        path = args.out/'cache'/f'trial_{key}.json'
        try:
            if path.exists():
                pred = json.loads(path.read_text())
            else:
                pred = ConditionedSE(retrievers[corpus])(CachedClient(), {'ID': row['ID'], 'Text': row['Text']})
                save(path, pred)
            print(f"{corpus} {row['ID']}: {len(pred['spans']['aspect'])} A, "
                  f"{len(pred['pairs'])} pairs, {len(pred['trace'])} calls; tokens={spent}", flush=True)
            return corpus, row, pred, path, None
        except BudgetReached:
            return corpus, row, None, path, 'budget'
        except Exception as exc:
            save(path.with_name(path.stem+'_error.json'), {'error': str(exc), 'type': type(exc).__name__})
            print(f"{corpus} {row['ID']}: {type(exc).__name__}; calls cached", flush=True)
            return corpus, row, None, path, type(exc).__name__

    jobs = [(c, rows[i]) for i in range(args.per_corpus) for c, rows in selected.items() if i < len(rows)]
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(run, jobs))
    done = [r for r in results if r[2] is not None]
    corpora = []
    for c in selected:
        group = [r for r in done if r[0] == c]
        if group:
            corpora.append({'corpus': c, 'ids': [r[1]['ID'] for r in group],
                            'metrics': evaluate([r[2] for r in group], [r[1] for r in group], .5)})
    macro = sum(c['metrics']['pair']['f1'] for c in corpora)/len(corpora) if corpora else None
    ledger = [json.loads(p.read_text()) for p in calls.glob('*.json')]
    stage = Counter()
    for call in ledger:
        stage[call['state']['stage']] += call['usage'].get('input_tokens', 0)
    refs = {}
    for mode in ('bio', 'pointer'):
        by_corpus = {}
        for c, row, *_ in done:
            key = hashlib.sha256((c+row['ID']+row['Text']).encode()).hexdigest()[:16]
            old = ROOT/'reports/extraction_comparison_20260923'/f'trial_{mode}'/'cache'
            matches = list(old.glob(f'{mode}_r3_41ea479bf861_{key}.json'))
            if matches:
                by_corpus.setdefault(c, []).append((json.loads(matches[0].read_text()), row))
        if by_corpus:
            pairs = [p for group in by_corpus.values() for p in group]
            refs[mode] = {'n': len(pairs), 'threshold': .65,
                'macro_pair_f1': sum(evaluate([p for p,r in v], [r for p,r in v], .65)['pair']['f1']
                                     for v in by_corpus.values())/len(by_corpus),
                'micro': evaluate([p for p,r in pairs], [r for p,r in pairs], .65)}
    report = {'split': 'trial', 'config': config, 'complete': len(done)==len(jobs),
              'n_requested': len(jobs), 'n_completed': len(done), 'corpora': corpora,
              'macro_pair_f1': macro if len(done)==len(jobs) else None,
              'partial_macro_pair_f1': macro if len(done)!=len(jobs) else None,
              'micro': evaluate([r[2] for r in done], [r[1] for r in done], .5),
              'null_opinion_gold_pairs': sum(len({(x['Aspect'].lower(),x['Opinion'].lower())
                  for x in _annotation_items(r[1]) if x['Opinion'].upper()=='NULL'}) for r in done),
              'calls': len(ledger), 'input_tokens': sum(stage.values()), 'stage_input_tokens': dict(stage),
              'estimated_usd': sum(stage.values())*.042/1e6,
              'http_attempts': sum(c['attempts'] for c in ledger),
              'returned_models': sorted({c['model'] for c in ledger}),
              'references': refs,
              'cache_files': [str(r[3].relative_to(ROOT)) if r[3].is_relative_to(ROOT) else str(r[3]) for r in done],
              'missing': [{'corpus':r[0], 'ID':r[1]['ID'], 'reason':r[4]} for r in results if r[2] is None],
              'metric_note': 'Exact case-insensitive pair F1, no VA or probability threshold. '
                             'probability=1 is an export marker, not calibrated confidence. '
                             'candidate_pair_recall equals final recall because pairs are directly extracted; '
                             'pair_rejected means both surfaces exist but the conditioned link was not extracted.'}
    save(args.out/f'trial_conditioned_se_n{args.per_corpus}.json', report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('corpora','references','cache_files')}, indent=2))


if __name__ == '__main__':
    main()
