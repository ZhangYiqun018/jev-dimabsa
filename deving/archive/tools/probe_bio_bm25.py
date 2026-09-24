#!/usr/bin/env python3
"""Pure Jev BIO r3 with two retrieved train examples; trial/dev paired comparison."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'tools')]
from jev.client import JevClient, Answer, Response, DEFAULT_MODEL
from jev.extraction import SpanExtractor, tokenize
from jev.combined_extraction import Retriever
from jev.data import load_jsonl, _annotation_items
from iterate_extraction import DATA, CORPORA, evaluate
from diagnose_span_choice import selected
from combined_experiment import digest, save, BudgetReached

OUT = ROOT/'reports/bio_bm25_20260923'
LIMITS = {'trial': 194302, 'dev': 1424104}


def demonstration(row):
    return {'review': row['Text'],
            'tokens': ' '.join(f'{i}|{t.text}' for i, t in enumerate(tokenize(row['Text']))),
            'aspects': list(dict.fromkeys(x['Aspect'] for x in _annotation_items(row) if x['Aspect'].upper() != 'NULL')),
            'opinions': list(dict.fromkeys(x['Opinion'] for x in _annotation_items(row) if x['Opinion'].upper() != 'NULL'))}


class RetrievedExamples:
    """Change only the extraction example package; capture the actual sent state."""
    def __init__(self, client, examples):
        self.client, self.examples, self.trace = client, examples, []

    def ask(self, state, questions):
        actual = dict(state)
        if 'invented_examples' in actual:
            del actual['invented_examples']
            actual['examples'] = self.examples
        response = self.client.ask(actual, questions)
        self.trace.append({'state': actual, 'questions': questions,
                           'answers': {k:a.raw for k,a in response.answers.items()},
                           'usage': response.usage, 'model': response.model, 'attempts': response.attempts})
        return response


def summary(done):
    corpora = []
    for corpus in dict.fromkeys(c for c,r,p,path in done):
        group = [(r,p) for c,r,p,path in done if c == corpus]
        corpora.append({'corpus':corpus,'ids':[r['ID'] for r,p in group],
                        'metrics':evaluate([p for r,p in group],[r for r,p in group],.65)})
    return {'n':len(done),'corpora':corpora,
            'macro_pair_f1':sum(c['metrics']['pair']['f1'] for c in corpora)/len(corpora) if corpora else None,
            'micro':evaluate([p for c,r,p,path in done],[r for c,r,p,path in done],.65)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split',choices=['trial','dev'],required=True)
    args=parser.parse_args();split=args.split
    out=OUT/split;calls=out/'cache/calls';calls.mkdir(parents=True,exist_ok=True)
    n=6 if split=='trial' else 12
    rows=selected(split,n)
    corpora=list(dict.fromkeys(c for c,r in rows))
    groups={c:[r for cc,r in rows if cc==c] for c in corpora}
    jobs=[(c,groups[c][i]) for i in range(n) for c in corpora]
    banned=[r['Text'] for p in (DATA/'trial').glob('*_alltasks.jsonl') for r in load_jsonl(p)]
    for c in CORPORA:
        banned.extend(r['Text'] for r in load_jsonl(DATA/'track_a/subtask_2'/c[:3]/f'{c}_dev_task2.jsonl'))
    retrievers={c:Retriever(load_jsonl(DATA/'track_a/subtask_2'/c[:3]/f'{c}_train_alltasks.jsonl'),c,banned) for c in corpora}
    config={'split':split,'model':DEFAULT_MODEL,'shots':2,'threshold':.65,
            'retrieval':'BM25 k1=1.5 b=0.75; Chinese/Japanese character bigrams; other languages alphanumeric tokenizer units; ID tie-break',
            'dataset':json.loads((ROOT/'data-version.json').read_text())['upstream_commit'],
            'fingerprint':digest({p:(ROOT/p).read_text() for p in ['tools/probe_bio_bm25.py','jev/extraction.py','jev/combined_extraction.py','jev/fewshot.py','tools/diagnose_span_choice.py']})[:12],
            'max_new_input_tokens':LIMITS[split],'concurrency':3,
            'baseline':'BIO r3 41ea479bf861; same extraction/pair rules and questions; replace only example package',
            'example_field':'invented_examples renamed to examples to correctly describe retrieved training records',
            'train_pool_sizes':{c:len(v.rows) for c,v in retrievers.items()}}
    save(out/'protocol.json',config)
    previous={};prior_requests={}
    def request(state,questions):return {'model':DEFAULT_MODEL,'state':state,'questions':questions}
    for c,r in jobs:
        key=hashlib.sha256((c+r['ID']+r['Text']).encode()).hexdigest()[:16]
        path=ROOT/'reports/extraction_comparison_20260923'/f'{split}_bio/cache'/f'bio_r3_41ea479bf861_{key}.json'
        pred=json.loads(path.read_text());previous[c,r['ID']]=pred
        for trace in pred['trace']:
            if 'tokens' not in trace['state'] and trace['model']==DEFAULT_MODEL:
                prior_requests[digest(request(trace['state'],trace['questions']))]=trace
    client=JevClient(timeout=60);lock=threading.Lock();key_locks={};reused=set()
    spent=sum(json.loads(p.read_text())['usage']['input_tokens'] for p in calls.glob('*.json'))
    class Cached:
        def ask(self,state,questions):
            nonlocal spent
            payload=request(state,questions);key=digest(payload);path=calls/f'{key}.json'
            with lock:key_lock=key_locks.setdefault(key,threading.Lock())
            with key_lock:
                if path.exists():cached=json.loads(path.read_text())
                elif key in prior_requests:
                    cached=prior_requests[key]
                    with lock:reused.add(key)
                else:cached=None
                if cached is not None:
                    return Response(cached['model'],{k:Answer(k,questions[k]['type'],v) for k,v in cached['answers'].items()},cached['usage'],cached['attempts'])
                with lock:
                    if spent>=LIMITS[split]:raise BudgetReached()
                result=client.ask(state,questions,attempts=2)
                save(path,{**payload,'model':result.model,'usage':result.usage,'attempts':result.attempts,'answers':{k:a.raw for k,a in result.answers.items()}})
                with lock:spent+=result.usage['input_tokens']
                return result
    def run(job):
        c,r=job;chosen=retrievers[c].select(r['Text'],2);demos=[demonstration(x) for x in chosen]
        path=out/'cache'/f'{digest([config,c,r["ID"],r["Text"],demos])[:20]}.json'
        try:
            if path.exists():pred=json.loads(path.read_text())
            else:
                adapter=RetrievedExamples(Cached(),demos)
                pred=SpanExtractor('bio',revision=3)(adapter,{'ID':r['ID'],'Text':r['Text']})
                pred['trace']=adapter.trace;pred['example_ids']=[x['ID'] for x in chosen]
                save(path,pred)
            print(f'{split} {c} {r["ID"]} done; new input={spent}',flush=True)
            return c,r,pred,path,None
        except BudgetReached:return c,r,None,path,'budget'
        except Exception as exc:
            save(path.with_name(path.stem+'_error.json'),{'type':type(exc).__name__,'error':str(exc)})
            return c,r,None,path,type(exc).__name__
    with ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(run,jobs))
    done=[(c,r,p,path) for c,r,p,path,error in results if p is not None]
    result=summary(done);macro=result.pop('macro_pair_f1');complete=len(done)==len(jobs)
    ledger=[json.loads(p.read_text()) for p in calls.glob('*.json')];stages=Counter()
    for call in ledger:stages['extract' if 'tokens' in call['state'] else 'pair']+=call['usage']['input_tokens']
    for c,r,p,path in done:
        for trace in p['trace']:
            key=digest(request(trace['state'],trace['questions']))
            if key in prior_requests:reused.add(key)
    baseline=summary([(c,r,previous[c,r['ID']],None) for c,r,p,path in done])
    report={**result,'config':config,'complete':complete,'n_requested':len(jobs),
            'macro_pair_f1':macro if complete else None,'partial_macro_pair_f1':macro if not complete else None,
            'baseline_same_completed_rows':baseline,'new_input_tokens':sum(stages.values()),'stage_new_input_tokens':dict(stages),
            'new_calls':len(ledger),'new_http_attempts':sum(c['attempts'] for c in ledger),'reused_pair_requests':len(reused),
            'new_estimated_usd':sum(stages.values())*.042/1e6,
            'logical_input_tokens_completed':sum(t['usage']['input_tokens'] for c,r,p,path in done for t in p['trace']),
            'logical_baseline_input_tokens_same_rows':sum(t['usage']['input_tokens'] for c,r,p,path in done for t in previous[c,r['ID']]['trace']),
            'returned_models':sorted({c['model'] for c in ledger}),
            'examples':[{'corpus':c,'ID':r['ID'],'example_ids':p['example_ids']} for c,r,p,path in done],
            'cache_files':[str(path.relative_to(ROOT)) for c,r,p,path in done],
            'missing':[{'corpus':c,'ID':r['ID'],'reason':error} for c,r,p,path,error in results if p is None],
            'budget_note':'Stop new requests after successful input reaches limit; up to three in-flight calls may overrun. Failed-attempt token usage unknown.',
            'metric_note':'Exact case-insensitive AO surface-pair F1, no VA. Existing development samples, not independent validation.'}
    save(out/'summary.json',report)
    print(json.dumps({k:v for k,v in report.items() if k in ['complete','n','macro_pair_f1','micro','new_input_tokens','new_estimated_usd','missing']},indent=2))
if __name__=='__main__':main()
