#!/usr/bin/env python3
"""Offline paired review bootstrap for the frozen BIO BM25 experiment."""
import hashlib
import json
from pathlib import Path
import random
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from diagnose_span_choice import selected
from iterate_extraction import evaluate
from combined_experiment import save
OUT=ROOT/'reports/bio_bm25_20260923'


def f1(counts):
    tp,fp,fn=counts
    return 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.


def interval(values):
    v=sorted(values)
    return [v[int(.025*(len(v)-1))],v[int(.975*(len(v)-1))]]


def main():
    results={}
    for split,n in [('trial',6),('dev',12)]:
        report=json.loads((OUT/split/'summary.json').read_text())
        if not report['complete']:
            results[split]={'complete':False,'note':'No full-split bootstrap for incomplete execution.'}
            continue
        rows=selected(split,n);corpora=list(dict.fromkeys(c for c,r in rows))
        groups={c:[r for cc,r in rows if cc==c] for c in corpora}
        jobs=[(c,groups[c][i]) for i in range(n) for c in corpora]
        counts={c:[] for c in corpora}
        for (c,r),p in zip(jobs,report['cache_files']):
            pred=json.loads((ROOT/p).read_text())
            key=hashlib.sha256((c+r['ID']+r['Text']).encode()).hexdigest()[:16]
            baseline=json.loads((ROOT/'reports/extraction_comparison_20260923'/f'{split}_bio/cache'/f'bio_r3_41ea479bf861_{key}.json').read_text())
            pair_counts=[]
            for value in (baseline,pred):
                m=evaluate([value],[r],.65)['pair']
                pair_counts.append([m[k] for k in ('tp','fp','fn')])
            counts[c].append(pair_counts)
        rng=random.Random(20260923);macro_deltas=[];micro_deltas=[]
        for _ in range(5000):
            macros=[0.,0.];micros=[[0,0,0],[0,0,0]]
            for c in corpora:
                aggregate=[[0,0,0],[0,0,0]]
                for i in rng.choices(range(len(counts[c])),k=len(counts[c])):
                    for arm in (0,1):
                        for j in range(3):aggregate[arm][j]+=counts[c][i][arm][j]
                for arm in (0,1):
                    macros[arm]+=f1(aggregate[arm])/len(corpora)
                    for j in range(3):micros[arm][j]+=aggregate[arm][j]
            macro_deltas.append(macros[1]-macros[0]);micro_deltas.append(f1(micros[1])-f1(micros[0]))
        base=report['baseline_same_completed_rows']
        results[split]={'complete':True,'n':report['n'],
                       'macro_delta':report['macro_pair_f1']-base['macro_pair_f1'],
                       'micro_f1_delta':report['micro']['pair']['f1']-base['micro']['pair']['f1'],
                       'macro_delta_95_percentile_interval':interval(macro_deltas),
                       'micro_f1_delta_95_percentile_interval':interval(micro_deltas),
                       'logical_input_ratio':report['logical_input_tokens_completed']/report['logical_baseline_input_tokens_same_rows'],
                       'new_input_tokens':report['new_input_tokens'],'new_estimated_usd':report['new_estimated_usd']}
    save(OUT/'comparison.json',{'splits':results,'bootstrap':{'replicates':5000,'seed':20260923,
         'unit':'review, paired arms, stratified by corpus','note':'Descriptive uncertainty on reused development data, not independent validation; no model-repeat uncertainty.'}})
    print(json.dumps(results,indent=2))
if __name__=='__main__':main()
