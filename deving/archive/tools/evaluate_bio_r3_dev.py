#!/usr/bin/env python3
"""Full dev BIO r3 + existing pair-conditioned Jev VA; official scorer unchanged."""
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import threading
from collections import Counter

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from jev.client import JevClient,Answer,Response,DEFAULT_MODEL,score_to_va,format_va
from jev.data import load_jsonl,write_jsonl
from jev.extraction import SpanExtractor
from jev.rubrics import valence_question,arousal_question
from iterate_extraction import DATA,CORPORA,evaluate
from combined_experiment import digest,save

OUT=ROOT/'reports/bio_r3_full_dev_20260923'
THRESHOLD=.65


def score_pairs(client,record,extracted,calibration):
    # Same case-insensitive surface identity as AO evaluation; one output per pair.
    pairs={}
    for pair in extracted['pairs']:
        if pair['probability']>=THRESHOLD:
            pairs.setdefault((pair['Aspect'].lower(),pair['Opinion'].lower()),(pair['Aspect'],pair['Opinion']))
    pairs=list(pairs.values());raw=[];calibrated=[];trace=[]
    for offset in range(0,len(pairs),16):
        batch=pairs[offset:offset+16];questions={}
        for i,(a,o) in enumerate(batch):
            questions[f'v{i}']=valence_question(a,o)
            questions[f'a{i}']=arousal_question(a,o)
        response=client.ask(record['Text'],questions)
        trace.append({'state':record['Text'],'questions':questions,'answers':{k:a.raw for k,a in response.answers.items()},
                      'model':response.model,'usage':response.usage,'attempts':response.attempts})
        for i,(a,o) in enumerate(batch):
            va=[score_to_va(response.answers[f'{d}{i}'].score) for d in ('v','a')]
            corrected=[min(9.,max(1.,s*x+b)) for x,s,b in zip(va,calibration['slope'],calibration['intercept'])]
            raw.append({'Aspect':a,'Opinion':o,'VA':format_va(*va)})
            calibrated.append({'Aspect':a,'Opinion':o,'VA':format_va(*corrected)})
    return {'ID':record['ID'],'raw':raw,'calibrated':calibrated,'trace':trace}


def official(corpus,source,pred,variant):
    result=subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'scoring/score.py'),
        '--task','2','--gold',str(source),'--pred',str(pred)],capture_output=True,text=True)
    log=OUT/'cache'/f'{corpus}_{variant}_scorer.txt'
    log.write_text(result.stdout+result.stderr)
    match=re.search(r'Final Results: (\{.*\})',result.stdout)
    if result.returncode or not match:raise RuntimeError(f'Official scorer failed; see {log}')
    metrics=ast.literal_eval(re.sub(r'np\.float64\(([^)]*)\)',r'\1',match.group(1)))
    metrics['categorical_TP']=float(re.search(r'True Positives \(TP\): ([\d.]+)',result.stdout).group(1))
    return metrics


def main():
    calls=OUT/'cache/calls';calls.mkdir(parents=True,exist_ok=True)
    calibration_path=ROOT/'reports/calibration_20260923/parameters.json'
    params=json.loads(calibration_path.read_text())['0']
    config={'model':DEFAULT_MODEL,'mode':'bio','revision':3,'pair_threshold':THRESHOLD,
        'dataset':json.loads((ROOT/'data-version.json').read_text())['upstream_commit'],
        'extractor_fingerprint':hashlib.sha256((ROOT/'jev/extraction.py').read_bytes()).hexdigest()[:12],
        'fingerprint':digest({p:(ROOT/p).read_text() for p in ['tools/evaluate_bio_r3_dev.py','jev/extraction.py','jev/rubrics.py','jev/fewshot.py']})[:12],
        'calibration':'Task1 zero-shot train-fitted shrink; previously selected on Task1 dev; frozen, no refit',
        'calibration_sha256':hashlib.sha256(calibration_path.read_bytes()).hexdigest(),
        'parameters':{c:params[c]['shrink'] for c in CORPORA},'va_batch_pairs':16,'concurrency':8,
        'primary':'calibrated cF1','secondary':'raw cF1 and exact AO F1',
        'no_new_null_opinion_support':True}
    save(OUT/'protocol.json',config)
    rows={c:load_jsonl(DATA/'track_a/subtask_2'/c[:3]/f'{c}_dev_task2.jsonl') for c in CORPORA}
    print('Full dev counts: '+json.dumps({c:len(rs) for c,rs in rows.items()}),flush=True)
    jobs=[(c,rs[i]) for i in range(max(map(len,rows.values()))) for c,rs in rows.items() if i<len(rs)]
    client=JevClient(timeout=90);lock=threading.Lock();key_locks={};finished=0
    class Cached:
        def __init__(self,stage):self.stage=stage
        def ask(self,state,questions):
            payload={'model':DEFAULT_MODEL,'state':state,'questions':questions};key=digest(payload);path=calls/f'{key}.json'
            with lock:key_lock=key_locks.setdefault(key,threading.Lock())
            with key_lock:
                if path.exists():
                    r=json.loads(path.read_text())
                    return Response(r['model'],{k:Answer(k,questions[k]['type'],v) for k,v in r['answers'].items()},r['usage'],r['attempts'])
                result=client.ask(state,questions,attempts=3)
                save(path,{**payload,'stage':self.stage,'model':result.model,'usage':result.usage,'attempts':result.attempts,
                           'answers':{k:a.raw for k,a in result.answers.items()}})
                return result
    def run(job):
        nonlocal finished
        c,r=job;key=hashlib.sha256((c+r['ID']+r['Text']).encode()).hexdigest()[:16]
        old=ROOT/'reports/extraction_comparison_20260923/dev_bio/cache'/f'bio_r3_{config["extractor_fingerprint"]}_{key}.json'
        ep=OUT/'cache'/f'extract_{config["extractor_fingerprint"]}_{key}.json'
        vp=OUT/'cache'/f'va_{digest([config,c,r["ID"],r["Text"]])[:20]}.json'
        try:
            if old.exists():extracted=json.loads(old.read_text());reused=True
            elif ep.exists():extracted=json.loads(ep.read_text());reused=False
            else:
                extracted=SpanExtractor('bio',revision=3)(Cached('extract_pair'),{'ID':r['ID'],'Text':r['Text']})
                save(ep,extracted);reused=False
            if vp.exists():va=json.loads(vp.read_text())
            else:
                va=score_pairs(Cached('va'),{'ID':r['ID'],'Text':r['Text']},extracted,params[c]['shrink'])
                save(vp,va)
            with lock:
                finished+=1
                if finished%20==0 or finished==len(jobs):print(f'Completed {finished}/{len(jobs)}',flush=True)
            return c,r,extracted,va,reused,None
        except Exception as exc:
            save(vp.with_name(vp.stem+'_error.json'),{'type':type(exc).__name__,'error':str(exc)})
            print(f'{c} {r["ID"]}: {type(exc).__name__}; retained cached calls',flush=True)
            return c,r,None,None,False,type(exc).__name__
    with ThreadPoolExecutor(max_workers=8) as pool:results=list(pool.map(run,jobs))
    done=[r for r in results if r[2] is not None];report_rows=[]
    for c,records in rows.items():
        group=[r for r in done if r[0]==c]
        row={'corpus':c,'requested':len(records),'completed':len(group),'complete':len(group)==len(records),
             'ao':evaluate([r[2] for r in group],[r[1] for r in group],THRESHOLD)}
        if row['complete']:
            by_id={r[1]['ID']:r for r in group}
            for variant in ('raw','calibrated'):
                path=OUT/'cache'/f'{c}_{variant}.jsonl'
                write_jsonl(path,[{'ID':r['ID'],'Triplet':by_id[r['ID']][3][variant]} for r in records])
                row[variant]=official(c,DATA/'track_a/subtask_2'/c[:3]/f'{c}_dev_task2.jsonl',path,variant)
        report_rows.append(row)
    ledger=[json.loads(p.read_text()) for p in calls.glob('*.json')];stages=Counter()
    for call in ledger:stages[call['stage']]+=call['usage']['input_tokens']
    complete=len(done)==len(jobs)
    report={'config':config,'complete':complete,'requested':len(jobs),'completed':len(done),'corpora':report_rows,
        'macro_ao_f1':sum(r['ao']['pair']['f1'] for r in report_rows)/len(report_rows) if complete else None,
        'macro_calibrated_cf1':sum(r['calibrated']['cF1'] for r in report_rows)/len(report_rows) if complete else None,
        'macro_raw_cf1':sum(r['raw']['cF1'] for r in report_rows)/len(report_rows) if complete else None,
        'micro_ao':evaluate([r[2] for r in done],[r[1] for r in done],THRESHOLD),
        'historical_extraction_records_reused':sum(r[4] for r in done),'new_calls':len(ledger),
        'new_http_attempts':sum(c['attempts'] for c in ledger),'new_input_tokens':sum(stages.values()),
        'new_stage_input_tokens':dict(stages),'new_estimated_usd':sum(stages.values())*.042/1e6,
        'logical_input_tokens_completed':sum(t['usage']['input_tokens'] for r in done for t in r[2]['trace']+r[3]['trace']),
        'returned_models':sorted({c['model'] for c in ledger}),
        'missing':[{'corpus':r[0],'ID':r[1]['ID'],'reason':r[5]} for r in results if r[2] is None],
        'note':'Full development evaluation, not official test rank. No refitting. AO uses surface sets; official cF1 uses unchanged organizer scorer.'}
    save(OUT/'summary.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('config','corpora','micro_ao')},indent=2))
if __name__=='__main__':main()
