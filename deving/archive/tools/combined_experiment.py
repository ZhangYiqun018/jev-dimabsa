"""Runner for the combined arm, invoked by iterate_extraction --mode combined."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import threading

from jev.client import Answer, Response, JevClient, DEFAULT_MODEL
from jev.combined_extraction import CONFIG, CombinedExtractor, Retriever
from jev.data import _annotation_items, load_jsonl
from jev.fewshot import normalise

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'vendor/DimABSA2026/task-dataset'


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


class BudgetReached(RuntimeError):
    pass


def run_combined(args, corpora, evaluate):
    cache = args.out / 'cache'
    calls = cache / 'calls'
    calls.mkdir(parents=True, exist_ok=True)
    sources = ['jev/combined_extraction.py', 'jev/extraction.py', 'tools/combined_experiment.py']
    fingerprint = digest({p: (ROOT/p).read_text() for p in sources})[:12]
    configuration = {**CONFIG, 'model': DEFAULT_MODEL, 'fingerprint': fingerprint,
                     'dataset': json.loads((ROOT/'data-version.json').read_text())['upstream_commit']}
    selected = {}
    retrievers = {}
    trial_texts = [r['Text'] for p in (DATA/'trial').glob('*_alltasks.jsonl') for r in load_jsonl(p)]
    banned = [*trial_texts]
    # Reuse text exclusion, without scanning test or building another audit stage.
    for corpus in corpora:
        path = DATA/'track_a/subtask_2'/corpus[:3]/f'{corpus}_dev_task2.jsonl'
        banned.extend(r['Text'] for r in load_jsonl(path))
    for corpus in corpora:
        folder = DATA/'track_a/subtask_2'/corpus[:3]
        path = (DATA/'trial'/f'{corpus}_trial_alltasks.jsonl' if args.split == 'trial'
                else folder/f'{corpus}_dev_task2.jsonl')
        if not path.exists():
            continue
        rows = load_jsonl(path)
        if args.split == 'dev':
            # Exactly the same sample exclusion/order as the previous BIO/SE runs.
            excluded = {s.strip().lower() for s in trial_texts}
            rows = [r for r in rows if r['Text'].strip().lower() not in excluded]
        rows.sort(key=lambda r: hashlib.sha256(('20260923:'+corpus+':'+r['ID']).encode()).digest())
        selected[corpus] = rows[:args.per_corpus]
        retrievers[corpus] = Retriever(load_jsonl(folder/f'{corpus}_train_alltasks.jsonl'), corpus, banned)
    selection_path = args.out/'selection.json'
    if args.split == 'dev':
        selection = json.loads(selection_path.read_text())
        if selection['configuration'] != configuration:
            raise ValueError('Dev requires the frozen trial configuration; use a new experiment directory for changes')
        threshold = selection['threshold']
    else:
        threshold = None

    lock = threading.Lock()
    key_locks = {}
    spent = sum(json.loads(p.read_text())['usage'].get('input_tokens', 0) for p in calls.glob('*.json'))
    initial_spent = spent
    client = JevClient(timeout=60)

    class CachedClient:
        def ask(self, state, questions):
            nonlocal spent
            payload = {'configuration': configuration, 'state': state, 'questions': questions}
            key = digest(payload)
            path = calls/f'{key}.json'
            with lock:
                request_lock = key_locks.setdefault(key, threading.Lock())
            with request_lock:
                if path.exists():
                    value = json.loads(path.read_text())
                    return Response(value['model'], {k: Answer(k, questions[k]['type'], v)
                                    for k, v in value['answers'].items()}, value['usage'], value['attempts'])
                with lock:
                    if spent >= args.max_input_tokens:
                        raise BudgetReached('Cumulative experiment token limit reached')
                response = client.ask(state, questions, attempts=2)
                value = {**payload, 'split': args.split, 'usage': response.usage, 'model': response.model,
                         'attempts': response.attempts,
                         'answers': {k: a.raw for k, a in response.answers.items()}}
                # Persist every response, including calls from records interrupted later.
                save(path, value)
                with lock:
                    spent += response.usage.get('input_tokens', 0)
                return response

    def run(job):
        corpus, row = job
        examples = retrievers[corpus].select(row['Text'], CONFIG['shots'])
        key = digest({'configuration': configuration, 'corpus': corpus, 'ID': row['ID'],
                      'text': row['Text'], 'examples': examples})[:20]
        path = cache/f'{args.split}_{key}.json'
        try:
            if path.exists():
                pred = json.loads(path.read_text())
            else:
                pred = CombinedExtractor(retrievers[corpus])(CachedClient(), {'ID': row['ID'], 'Text': row['Text']})
                save(path, pred)
            print(f"{corpus} {row['ID']}: {len(pred['candidates'])} candidates, "
                  f"{len(pred['spans']['aspect'])} A, {len(pred['spans']['opinion'])} O, "
                  f"{len(pred['pairs'])} pairs; cumulative input={spent}", flush=True)
            return corpus, row, pred, str(path), None
        except BudgetReached:
            return corpus, row, None, str(path), 'budget'
        except Exception as exc:
            # Details stay local, not in the shareable report (API errors can echo text).
            save(cache/f'{args.split}_{key}_error.json', {'type': type(exc).__name__, 'message': str(exc)})
            print(f"{corpus} {row['ID']}: {type(exc).__name__}; cached calls retained", flush=True)
            return corpus, row, None, str(path), type(exc).__name__

    # Round-robin corpora so a budget stop does not only evaluate early languages.
    jobs = [(c, rows[i]) for i in range(args.per_corpus) for c, rows in selected.items() if i < len(rows)]
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(run, jobs))
    completed = [r for r in results if r[2] is not None]
    complete = len(completed) == len(jobs)

    def scores_at(t):
        per = {c: evaluate([p for co, _, p, _, _ in completed if co == c],
                           [row for co, row, _, _, _ in completed if co == c], t)
               for c in selected if any(co == c for co, *_ in completed)}
        return {'macro_pair_f1': sum(v['pair']['f1'] for v in per.values())/len(per) if per else None,
                'corpora': per}

    if args.split == 'trial':
        grid = {str(t): scores_at(t) for t in (.5, .65, .8)}
        if complete:
            threshold = max((.5, .65, .8), key=lambda t: (grid[str(t)]['macro_pair_f1'], t))
            save(selection_path, {'configuration': configuration, 'threshold': threshold,
                                 'trial_threshold_grid': grid,
                                 'rule': 'Max trial macro pair F1; ties choose higher threshold; frozen before dev'})
        else:
            threshold = .65  # Reporting only; never freeze a selection from incomplete trial.
    report_scores = scores_at(threshold)
    all_preds = [p for _, _, p, _, _ in completed]
    all_rows = [row for _, row, _, _, _ in completed]
    traces = [t for p in all_preds for t in p['trace']]
    pre = Counter()
    for pred, row in zip(all_preds, all_rows):
        surfaces = {s['text'].lower() for s in pred['candidates']}
        gold = {(x['Aspect'].lower(), x['Opinion'].lower()) for x in _annotation_items(row)}
        pre['gold_pairs'] += len(gold)
        pre['reachable_pairs'] += sum((a == 'null' or a in surfaces) and o in surfaces for a, o in gold)
        pre['null_opinion_pairs'] += sum(o == 'null' for _, o in gold)
    pre['recall'] = pre['reachable_pairs']/pre['gold_pairs'] if pre['gold_pairs'] else 0.
    ledger = [json.loads(p.read_text()) for p in calls.glob('*.json')]
    actual_input = sum(t['usage'].get('input_tokens', 0) for t in ledger)
    split_ledger = [t for t in ledger if t['split'] == args.split]
    stage_usage = {stage: sum(t['usage'].get('input_tokens', 0) for t in split_ledger
                             if t['state']['stage'] == stage)
                   for stage in ('span_screen', 'conditioned_opinion')}
    report = {'split': args.split, 'configuration': configuration, 'complete': complete,
              'n_requested': len(jobs), 'n_completed': len(completed), 'threshold': threshold,
              'corpora': [{'corpus': c, 'ids': [row['ID'] for co, row, *_ in completed if co == c],
                           'metrics': metrics} for c, metrics in report_scores['corpora'].items()],
              'macro_pair_f1': report_scores['macro_pair_f1'] if complete else None,
              'partial_macro_pair_f1': report_scores['macro_pair_f1'] if not complete else None,
              'micro': evaluate(all_preds, all_rows, threshold), 'pre_screen_coverage': dict(pre),
              'candidate_counts': {'full': sum(len(p['candidates']) for p in all_preds),
                  'aspect': sum(len(p['spans']['aspect']) for p in all_preds),
                  'opinion': sum(len(p['spans']['opinion']) for p in all_preds),
                  'pairs': sum(len(p['pairs']) for p in all_preds)},
              'models': sorted({t['model'] for t in traces}),
              'calls': len(split_ledger), 'questions': sum(len(t['questions']) for t in split_ledger),
              'input_tokens': sum(stage_usage.values()), 'stage_input_tokens': stage_usage,
              'experiment_input_tokens': actual_input, 'new_input_tokens': spent-initial_spent,
              'estimated_usd': sum(stage_usage.values())*.042/1e6,
              'experiment_estimated_usd': actual_input*.042/1e6,
              'http_attempts': sum(t['attempts'] for t in split_ledger),
              'cache_files': [str(Path(path).relative_to(ROOT)) if Path(path).is_relative_to(ROOT)
                              else path for _, _, _, path, _ in completed],
              'missing': [{'corpus': c, 'ID': r['ID'], 'reason': error} for c, r, p, _, error in results if p is None],
              'metric_note': 'Exact case-insensitive surface pair F1 without VA; not official cF1. '
                             'Incomplete runs are explicitly partial; usage includes unfinished records.'}
    if args.split == 'trial':
        report['trial_threshold_grid'] = grid
    # Same-record historical reference; no old method API calls.
    references = {}
    for mode in ('bio', 'pointer'):
        old = ROOT/'reports/extraction_comparison_20260923'/f'{args.split}_{mode}'/'cache'
        old_preds, old_rows = [], []
        by_corpus = {}
        for corpus, row, *_ in completed:
            key = hashlib.sha256((corpus+row['ID']+row['Text']).encode()).hexdigest()[:16]
            matches = list(old.glob(f'{mode}_r3_41ea479bf861_{key}.json'))
            if matches:
                pred = json.loads(matches[0].read_text())
                old_preds.append(pred)
                old_rows.append(row)
                by_corpus.setdefault(corpus, []).append((pred, row))
        if old_rows:
            references[mode] = {'n': len(old_rows), 'threshold': .65,
                'micro': evaluate(old_preds, old_rows, .65),
                'macro_pair_f1': sum(evaluate([p for p, _ in v], [r for _, r in v], .65)['pair']['f1']
                                     for v in by_corpus.values())/len(by_corpus)}
    report['references_on_completed_records'] = references
    dest = args.out/f'{args.split}_combined_n{args.per_corpus}.json'
    save(dest, report)
    print(json.dumps({k: v for k, v in report.items() if k not in
                     ('corpora', 'cache_files', 'trial_threshold_grid', 'references_on_completed_records')}, indent=2))
