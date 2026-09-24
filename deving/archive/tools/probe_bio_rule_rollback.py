#!/usr/bin/env python3
"""One BIO extraction-rule rollback; frozen examples and pairing remain unchanged."""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'tools')]
from jev.client import JevClient, Answer, Response, DEFAULT_MODEL
from jev.span_choice import MatchedExtractor
from combined_experiment import digest, save, BudgetReached
from diagnose_span_choice import selected, old
from iterate_extraction import evaluate

PRIOR = ROOT/'reports/span_choice_20260923'
OUT = ROOT/'reports/bio_rule_rollback_20260923'
LIMIT = 194302
# Exact historical scores plus the proposed percentage-point margins.
GATE = {'macro_pair_f1': .6121323529411764 + .03,
        'micro_precision': 22/34 - .02, 'micro_pair_f1': 44/73}


class ExtractionRuleRollback:
    """Intercept only extraction state; retain the exact model-visible trace."""
    def __init__(self, client, rule_block):
        self.client, self.rule_block, self.trace = client, rule_block, []

    def ask(self, state, questions):
        actual = {**state, **self.rule_block} if state['stage'].startswith('extract_') else state
        response = self.client.ask(actual, questions)
        self.trace.append({'state': actual, 'questions': questions,
                           'answers': {k: a.raw for k, a in response.answers.items()},
                           'model': response.model, 'usage': response.usage,
                           'attempts': response.attempts})
        return response


def summarize(done):
    corpora = []
    for c in dict.fromkeys(c for c, r, p in done):
        group = [(r, p) for cc, r, p in done if cc == c]
        corpora.append({'corpus': c, 'ids': [r['ID'] for r, p in group],
                        'metrics': evaluate([p for r, p in group], [r for r, p in group], .65)})
    return {'n': len(done), 'corpora': corpora,
            'macro_pair_f1': sum(c['metrics']['pair']['f1'] for c in corpora)/len(corpora) if corpora else None,
            'micro': evaluate([p for c, r, p in done], [r for c, r, p in done], .65)}


def main():
    calls = OUT/'cache/calls'
    calls.mkdir(parents=True, exist_ok=True)
    source = sorted((ROOT/'reports/extraction_comparison_20260923/trial_bio/cache').glob('bio_r3_41ea479bf861_*.json'))[0]
    historical = json.loads(source.read_text())
    state = next(t['state'] for t in historical['trace'] if 'tokens' in t['state'])
    block = {k: state[k] for k in ('rules', 'boundary_guidance')}
    # Reuse the exact cached examples, including their prior token alignments.
    examples = json.loads((PRIOR/'cache/bio_examples.json').read_text())
    prior = json.loads((PRIOR/'trial_bio.json').read_text())
    config = {'model': DEFAULT_MODEL, 'dataset': prior['config']['dataset'],
              'prior_fingerprint': prior['config']['fingerprint'],
              'rule_source': str(source.relative_to(ROOT)), 'extraction_rule_block': block,
              'examples_digest': digest(examples), 'example_ids': prior['config']['example_ids'],
              'max_new_input_tokens': LIMIT, 'threshold': .65, 'gate': GATE,
              'fingerprint': digest({p: (ROOT/p).read_text() for p in
                  ['tools/probe_bio_rule_rollback.py', 'jev/span_choice.py', 'jev/extraction.py']})[:12]}
    save(OUT/'protocol.json', config)

    def payload(model, state, questions):
        return {'model': model, 'state': state, 'questions': questions}

    # Match the complete request, including batch composition and IDs, not just AO strings.
    prior_calls = {}
    for path in (PRIOR/'cache/bio/calls').glob('*.json'):
        call = json.loads(path.read_text())
        if call['state']['stage'] == 'pair':
            key = digest(payload(call['config']['model'], call['state'], call['questions']))
            prior_calls[key] = path
    spent = sum(json.loads(p.read_text())['usage'].get('input_tokens', 0) for p in calls.glob('*.json'))
    reused = set()
    client = JevClient(timeout=60)

    class Cached:
        def ask(self, state, questions):
            nonlocal spent
            request = payload(DEFAULT_MODEL, state, questions)
            key = digest(request)
            path = calls/f'{key}.json'
            cached = path if path.exists() else prior_calls.get(key)
            if cached:
                record = json.loads(cached.read_text())
                if cached != path:
                    reused.add(str(cached.relative_to(ROOT)))
                return Response(record['model'], {k: Answer(k, questions[k]['type'], v)
                    for k, v in record['answers'].items()}, record['usage'], record['attempts'])
            if spent >= LIMIT:
                raise BudgetReached()
            response = client.ask(state, questions, attempts=2)
            save(path, {**request, 'answers': {k: a.raw for k, a in response.answers.items()},
                        'usage': response.usage, 'attempts': response.attempts, 'model': response.model})
            spent += response.usage.get('input_tokens', 0)
            return response

    selected_rows = selected('trial', 6)
    corpora = list(dict.fromkeys(c for c, r in selected_rows))
    groups = {c: [r for cc, r in selected_rows if cc == c] for c in corpora}
    jobs = [(c, groups[c][i]) for i in range(6) for c in corpora]
    done, missing, paths = [], [], []
    for c, row in jobs:
        path = OUT/'cache'/f'{digest([config,c,row["ID"],row["Text"]])[:20]}.json'
        try:
            if path.exists():
                pred = json.loads(path.read_text())
            else:
                adapter = ExtractionRuleRollback(Cached(), block)
                pred = MatchedExtractor('bio', examples[c])(adapter, {'ID': row['ID'], 'Text': row['Text']})
                pred['trace'] = adapter.trace
                save(path, pred)
            done.append((c, row, pred)); paths.append(str(path.relative_to(ROOT)))
            print(f'{len(done)}/24 {c} {row["ID"]} new input={spent}', flush=True)
        except BudgetReached:
            missing.append({'corpus': c, 'ID': row['ID'], 'reason': 'budget'})
        except Exception as exc:
            save(path.with_name(path.stem+'_error.json'), {'type': type(exc).__name__, 'error': str(exc)})
            missing.append({'corpus': c, 'ID': row['ID'], 'reason': type(exc).__name__})

    result = summarize(done)
    complete = len(done) == len(jobs)
    macro = result.pop('macro_pair_f1')
    ledger = [json.loads(p.read_text()) for p in calls.glob('*.json')]
    stages = Counter()
    for call in ledger:
        stages[call['state']['stage']] += call['usage'].get('input_tokens', 0)
    # Prior BIO cache paths are round-robin, matching the unchanged trial order.
    prior_predictions = {(c, r['ID']): json.loads((ROOT/p).read_text())
                         for (c, r), p in zip(jobs, prior['cache_files'])}
    comparisons = {
        'historical_bio_r3': summarize([(c, r, old(c, r, 'bio')) for c, r, p in done]),
        'fixed_example_bio': summarize([(c, r, prior_predictions[c, r['ID']]) for c, r, p in done])}
    # Reconstruct reuse from traces as well, so resumes report the same provenance.
    for c, r, pred in done:
        for t in pred['trace']:
            key = digest(payload(DEFAULT_MODEL, t['state'], t['questions']))
            if key in prior_calls:
                reused.add(str(prior_calls[key].relative_to(ROOT)))
    gate_pass = complete and macro >= GATE['macro_pair_f1'] and \
        result['micro']['pair']['precision'] >= GATE['micro_precision'] and \
        result['micro']['pair']['f1'] >= GATE['micro_pair_f1']
    report = {**result, 'config': config, 'complete': complete, 'n_requested': len(jobs),
              'macro_pair_f1': macro if complete else None,
              'partial_macro_pair_f1': macro if not complete else None,
              'without_null_opinion': evaluate([{**p, 'pairs': [q for q in p['pairs'] if q['Opinion'] != 'NULL']}
                                                for c, r, p in done], [r for c, r, p in done], .65),
              'comparisons_same_completed_rows': comparisons,
              'new_input_tokens': sum(stages.values()), 'stage_new_input_tokens': dict(stages),
              'new_estimated_usd': sum(stages.values())*.042/1e6,
              'new_calls': len(ledger), 'new_http_attempts': sum(c['attempts'] for c in ledger),
              'reused_pair_requests': len(reused), 'reused_paths': sorted(reused),
              'logical_input_tokens_completed': sum(t['usage']['input_tokens'] for c,r,p in done for t in p['trace']),
              'returned_models': sorted({c['model'] for c in ledger}),
              'gate_pass': gate_pass, 'cache_files': paths, 'missing': missing,
              'metric_note': 'Exact case-insensitive AO surface-pair F1, no VA; not official cF1.',
              'budget_note': 'New successful input only; stop before next request after limit reached. Final call may overrun; failed-attempt token usage unknown.'}
    save(OUT/'trial.json', report)
    print(json.dumps({k: v for k, v in report.items() if k in
                     ('complete','macro_pair_f1','micro','new_input_tokens','new_estimated_usd','gate_pass','reused_pair_requests')}, indent=2))

if __name__ == '__main__':
    main()
