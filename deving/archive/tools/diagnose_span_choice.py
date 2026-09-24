#!/usr/bin/env python3
"""Offline diagnostics only; public report contains counts, raw cases stay in cache."""
import sys, json, hashlib
from pathlib import Path
from collections import Counter, defaultdict
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from iterate_extraction import DATA,CORPORA
from jev.data import load_jsonl,_annotation_items
from jev.extraction import tokenize
from combined_experiment import save
OUT=ROOT/'reports/span_choice_20260923'

def align(row):
    text=row['Text']; ts=tokenize(text)
    spans={}; ambiguous=Counter()
    for role in ('aspect','opinion'):
        values={x[role.title()] for x in _annotation_items(row)}-{'NULL'}
        spans[role]=[]
        for value in sorted(values):
            # Exact case-sensitive text alignment, no fabricated occurrence supervision.
            hits=[(i,j) for i,t in enumerate(ts) if text.startswith(value,t.start)
                  for j in range(i,len(ts)) if ts[j].end==t.start+len(value)]
            if len(hits)==1: spans[role].append(hits[0])
            else: ambiguous[role]+=1
    return spans,ambiguous

def selected(split,n):
    banned={r['Text'].strip().lower() for p in (DATA/'trial').glob('*_alltasks.jsonl') for r in load_jsonl(p)}
    result=[]
    for c in CORPORA:
        p=DATA/'trial'/f'{c}_trial_alltasks.jsonl' if split=='trial' else DATA/'track_a/subtask_2'/c[:3]/f'{c}_dev_task2.jsonl'
        if not p.exists():continue
        rows=load_jsonl(p)
        if split=='dev': rows=[r for r in rows if r['Text'].strip().lower() not in banned]
        rows.sort(key=lambda r:hashlib.sha256(('20260923:'+c+':'+r['ID']).encode()).digest())
        result.extend((c,r) for r in rows[:n])
    return result

def old(c,r,mode):
    key=hashlib.sha256((c+r['ID']+r['Text']).encode()).hexdigest()[:16]
    p=ROOT/'reports/extraction_comparison_20260923'/f'trial_{mode}'/'cache'/f'{mode}_r3_41ea479bf861_{key}.json'
    return json.loads(p.read_text())

def main():
    train=[]
    for c in CORPORA:
        counts=Counter()
        for r in load_jsonl(DATA/'track_a/subtask_2'/c[:3]/f'{c}_train_alltasks.jsonl'):
            spans,amb=align(r);counts['reviews']+=1
            for role,ss in spans.items():
                groups=defaultdict(set)
                for s,e in ss:groups[s].add(e)
                counts[role+'_unique_aligned_spans']+=len(ss)
                counts[role+'_ambiguous_or_unaligned_surfaces']+=amb[role]
                counts[role+'_same_start_groups']+=sum(len(ends)>1 for ends in groups.values())
                counts[role+'_same_start_extra_spans']+=sum(len(ends)-1 for ends in groups.values())
        train.append({'corpus':c,**counts})
    cases=[];counts=Counter();cross=defaultdict(Counter)
    for c,r in selected('trial',6):
        bio,se=old(c,r,'bio'),old(c,r,'pointer')
        gold={(x['Aspect'].lower(),x['Opinion'].lower()) for x in _annotation_items(r)}
        ga={a for a,o in gold if a!='null'};sa={a.lower() for a in se['spans']['aspect']}
        for category,values,opposite in [('fp',sa-ga,ga),('fn',ga-sa,sa)]:
            for value in sorted(values):
                related=[x for x in opposite if value in x or x in value]
                counts[category]+=1;counts[category+'_substring_related' if related else category+'_other']+=1
                cases.append({'corpus':c,'ID':r['ID'],'text':r['Text'],'category':category,'span':value,'related':related,'trace':se['trace'] if category=='fn' else []})
        for name,ap,op in [('bio_bio',bio,bio),('se_bio',se,bio),('bio_se',bio,se),('se_se',se,se)]:
            aa={x.lower() for x in ap['spans']['aspect']}|{'null'};oo={x.lower() for x in op['spans']['opinion']}
            cp={(a,o) for a in aa for o in oo}
            cross[name]['covered_gold']+=len(cp&gold);cross[name]['candidate_pairs']+=len(cp);cross[name]['gold']+=len(gold)
    summary=json.loads((ROOT/'reports/extraction_combined_20260923/dev_combined_n12.json').read_text())
    # Join by corpus+ID, never ID alone.
    rawpred=[json.loads((ROOT/p).read_text()) for p in summary['cache_files']]
    rows=selected('dev',12)
    # The saved cache list is round-robin across corpora.
    groups={c:[r for cc,r in rows if cc==c] for c in CORPORA}
    jobs=[(c,groups[c][i]) for i in range(12) for c in CORPORA]
    rank=Counter(); scored=[]
    for (c,r),p in zip(jobs,rawpred):
        assert p['ID']==r['ID']
        gold={(x['Aspect'].lower(),x['Opinion'].lower()) for x in _annotation_items(r)}
        probs={(x['Aspect'].lower(),x['Opinion'].lower()):x['probability'] for x in p['pairs']}
        for g in gold&probs.keys():
            rank['covered_gold']+=1
            negatives=[v for q,v in probs.items() if q not in gold and q!=g and
                       all(x==y or (x!='null' and y!='null' and (x in y or y in x)) for x,y in zip(q,g))]
            if negatives:
                rank['gold_with_boundary_negative']+=1
                rank['gold_below_best_boundary_negative' if probs[g]<max(negatives) else 'gold_tied_best_boundary_negative' if probs[g]==max(negatives) else 'gold_above_best_boundary_negative']+=1
            rank['covered_gold_below_080']+=probs[g]<.8
        scored.append((probs,gold))
    grid=[]
    for t in [i/100 for i in range(101)]:
        tp=fp=fn=0
        for probs,gold in scored:
            pred={q for q,v in probs.items() if v>=t};tp+=len(pred&gold);fp+=len(pred-gold);fn+=len(gold-pred)
        grid.append({'threshold':t,'tp':tp,'fp':fp,'fn':fn,'micro_f1':2*tp/(2*tp+fp+fn)})
    save(OUT/'cache/se_aspect_errors.json',cases)
    save(OUT/'diagnostics.json',{'train_same_start':train,'se_aspect_errors':dict(counts),'component_pair_coverage':dict(cross),'combined_dev_rank':dict(rank),'combined_dev_oracle_threshold':max(grid,key=lambda x:x['micro_f1']),'note':'Substring matches are structural signals, not manually confirmed error causes; oracle threshold uses the development set itself.'})
    print((OUT/'diagnostics.json').read_text())
if __name__=='__main__':main()
