#!/usr/bin/env python3
"""Frozen matched trial comparison, per-call caching and bounded input usage."""
import sys,json,hashlib,argparse
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from jev.span_choice import MatchedExtractor,CONFIG
from jev.client import JevClient,Answer,Response,DEFAULT_MODEL
from jev.data import load_jsonl,_annotation_items
from jev.extraction import tokenize
from iterate_extraction import DATA,CORPORA,evaluate
from diagnose_span_choice import align,selected
from combined_experiment import save,digest,BudgetReached


def fixed_examples():
    banned={r['Text'].strip().lower() for p in (DATA/'trial').glob('*_alltasks.jsonl') for r in load_jsonl(p)}
    for c in CORPORA:
        banned.update(r['Text'].strip().lower() for r in load_jsonl(DATA/'track_a/subtask_2'/c[:3]/f'{c}_dev_task2.jsonl'))
    result={}
    for c in CORPORA:
        pool=[];seen=set()
        for r in load_jsonl(DATA/'track_a/subtask_2'/c[:3]/f'{c}_train_alltasks.jsonl'):
            normalized=r['Text'].strip().lower()
            if normalized in banned or normalized in seen:continue
            seen.add(normalized)
            spans,amb=align(r);n=len(tokenize(r['Text']))
            if sum(amb.values()) or not 5<=n<=35 or not all(spans.values()):continue
            # Unambiguous, BIO-representable demonstrations for both arms.
            if any(any(s2<=e1 for (s1,e1),(s2,e2) in zip(sorted(ss),sorted(ss)[1:])) for ss in spans.values()):continue
            if any(e-s>=12 for ss in spans.values() for s,e in ss):continue
            pairs={(x['Aspect'],x['Opinion']) for x in _annotation_items(r)}
            null=any('NULL' in p for p in pairs)
            multi=sum(any(e>s for s,e in ss) for ss in spans.values())
            pool.append((r,spans,null,multi,len(pairs)>1))
        # First seek NULL + multi-pair + multi-token examples, then a distinct multi-token review.
        first=sorted(pool,key=lambda v:(-v[2],-v[3],-v[4],hashlib.sha256(v[0]['ID'].encode()).hexdigest()))[0]
        rest=[v for v in pool if v is not first]
        second=sorted(rest,key=lambda v:(-v[3],-v[4],hashlib.sha256(v[0]['ID'].encode()).hexdigest()))[0]
        result[c]=[{'ID':r['ID'],'Text':r['Text'],'aligned':ss} for r,ss,*_ in (first,second)]
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['bio','choice'],required=True)
    parser.add_argument('--out',type=Path,default=ROOT/'reports/span_choice_20260923')
    args=parser.parse_args();mode=args.mode
    out=args.out;calls=out/'cache'/mode/'calls';calls.mkdir(parents=True,exist_ok=True)
    examples=fixed_examples()
    config={**CONFIG,'mode':mode,'model':DEFAULT_MODEL,
            'dataset':json.loads((ROOT/'data-version.json').read_text())['upstream_commit'],
            'fingerprint':digest({p:(ROOT/p).read_text() for p in ['jev/span_choice.py','tools/probe_span_choice.py','tools/diagnose_span_choice.py']})[:12],
            'example_ids':{c:[r['ID'] for r in rs] for c,rs in examples.items()}}
    save(out/'cache'/f'{mode}_examples.json',examples)
    ledger=[json.loads(p.read_text()) for p in calls.glob('*.json')]
    spent=sum(r['usage'].get('input_tokens',0) for r in ledger)
    client=JevClient(timeout=60)
    class Cached:
        def ask(self,state,questions):
            nonlocal spent
            payload={'config':config,'state':state,'questions':questions};path=calls/f'{digest(payload)}.json'
            if path.exists():
                r=json.loads(path.read_text())
                return Response(r['model'],{k:Answer(k,questions[k]['type'],v) for k,v in r['answers'].items()},r['usage'],r['attempts'])
            if spent>=CONFIG['max_input_per_arm']:raise BudgetReached()
            response=client.ask(state,questions,attempts=2)
            save(path,{**payload,'usage':response.usage,'model':response.model,'attempts':response.attempts,'answers':{k:a.raw for k,a in response.answers.items()}})
            spent+=response.usage.get('input_tokens',0)
            return response
    rows=selected('trial',6);corpora=list(dict.fromkeys(c for c,r in rows))
    groups={c:[r for cc,r in rows if cc==c] for c in corpora}
    jobs=[(c,groups[c][i]) for i in range(6) for c in corpora]
    done=[];missing=[]
    for c,r in jobs:
        path=out/'cache'/mode/f'{digest([config,c,r["ID"],r["Text"]])[:20]}.json'
        try:
            if path.exists():pred=json.loads(path.read_text())
            else:
                pred=MatchedExtractor(mode,examples[c])(Cached(),{'ID':r['ID'],'Text':r['Text']})
                save(path,pred)
            done.append((c,r,pred,path))
            print(f'{mode} {c} {r["ID"]}: done {len(done)}/24, input={spent}',flush=True)
        except BudgetReached:
            missing.append({'corpus':c,'ID':r['ID'],'reason':'budget'})
        except Exception as exc:
            save(path.with_name(path.stem+'_error.json'),{'type':type(exc).__name__,'error':str(exc)})
            missing.append({'corpus':c,'ID':r['ID'],'reason':type(exc).__name__})
    results=[{'corpus':c,'ids':[r['ID'] for cc,r,p,path in done if cc==c],
              'metrics':evaluate([p for cc,r,p,path in done if cc==c],[r for cc,r,p,path in done if cc==c],.65)} for c in corpora]
    ledger=[json.loads(p.read_text()) for p in calls.glob('*.json')];stage=Counter()
    for call in ledger:stage[call['state']['stage']]+=call['usage'].get('input_tokens',0)
    preds=[p for c,r,p,path in done];gold=[r for c,r,p,path in done]
    without=[{**p,'pairs':[q for q in p['pairs'] if q['Opinion']!='NULL']} for p in preds]
    report={'config':config,'complete':len(done)==24,'n_completed':len(done),'n_requested':24,
            'corpora':results,'macro_pair_f1':sum(x['metrics']['pair']['f1'] for x in results)/len(results) if len(done)==24 else None,
            'micro':evaluate(preds,gold,.65),'without_null_opinion':evaluate(without,gold,.65),
            'input_tokens':sum(stage.values()),'stage_input_tokens':dict(stage),'estimated_usd':sum(stage.values())*.042/1e6,
            'calls':len(ledger),'questions':sum(len(c['questions']) for c in ledger),'http_attempts':sum(c['attempts'] for c in ledger),
            'returned_models':sorted({c['model'] for c in ledger}),
            'cache_files':[str(path.relative_to(ROOT)) for c,r,p,path in done],'missing':missing,
            'budget_note':'Sequential requests; stop before the next request once reported successful input reaches cap. Last request may overrun cap; failed attempts have unknown token cost.',
            'metric_note':'Exact case-insensitive AO surface-pair F1, no VA. Incomplete runs have no aggregate full-trial macro score.'}
    save(out/f'trial_{mode}.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('config','corpora','cache_files')},indent=2))
if __name__=='__main__':main()
