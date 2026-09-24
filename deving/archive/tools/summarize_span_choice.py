#!/usr/bin/env python3
"""Summarize completed intersection and cached decision diagnostics; no API calls."""
import sys,json
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from diagnose_span_choice import selected,old
from iterate_extraction import evaluate
from jev.data import _annotation_items
from jev.extraction import tokenize
from combined_experiment import save
OUT=ROOT/'reports/span_choice_20260923'

def load(mode):
    report=json.loads((OUT/f'trial_{mode}.json').read_text())
    # Reports serialize completed rows in round-robin corpus order.
    remaining={c['corpus']:list(c['ids']) for c in report['corpora']}
    order=[]
    for c,r in jobs:
        if r['ID'] in remaining[c]:order.append((c,r))
    return report,{(c,r['ID']):json.loads((ROOT/p).read_text()) for (c,r),p in zip(order,report['cache_files'])}


def metrics(preds,keys):
    corpora=[]
    for c in dict.fromkeys(k[0] for k in keys):
        ck=[k for k in keys if k[0]==c]
        corpora.append({'corpus':c,'n':len(ck),'metrics':evaluate([preds[k] for k in ck],[rows[k] for k in ck],.65)})
    return {'n':len(keys),'macro_pair_f1':sum(c['metrics']['pair']['f1'] for c in corpora)/len(corpora),
            'micro':evaluate([preds[k] for k in keys],[rows[k] for k in keys],.65),'corpora':corpora,
            'input_tokens_completed_records':sum(t['usage']['input_tokens'] for k in keys for t in preds[k]['trace'])}

if __name__=='__main__':
    selected_rows=selected('trial',6);corpora=list(dict.fromkeys(c for c,r in selected_rows))
    grouped={c:[r for cc,r in selected_rows if cc==c] for c in corpora}
    jobs=[(c,grouped[c][i]) for i in range(6) for c in corpora];rows={(c,r['ID']):r for c,r in jobs}
    br,bio=load('bio');cr,choice=load('choice');historical={(c,r['ID']):old(c,r,'bio') for c,r in jobs}
    keys=[k for k in rows if k in choice and k in bio];common={name:metrics(p,keys) for name,p in [('bio_r3',historical),('bio_fixed',bio),('choice',choice)]}
    repeat=Counter();errors=Counter();case_details=[]
    for key in keys:
        r=rows[key];p=choice[key];ts=tokenize(r['Text']);gold={(x['Aspect'].lower(),x['Opinion'].lower()) for x in _annotation_items(r)}
        first={k:set() for k in ('aspect','opinion')};extra={k:set() for k in first}
        for t in p['trace']:
            if t['state']['stage']=='pair':continue
            for name,q in t['questions'].items():
                info=q['instructions'];role=info['role'];again=bool(info['already_selected'])
                repeat['continuation_questions' if again else 'initial_questions']+=1
                answer=t['answers'][name]['choice']
                if answer=='none':continue
                value=r['Text'][ts[info['start_token']].start:ts[int(answer)].end].lower()
                (extra if again else first)[role].add(value)
                repeat['continuation_positive_decisions' if again else 'initial_positive_decisions']+=1
        for role,i in [('aspect',0),('opinion',1)]:
            gold_spans={x[i] for x in gold if x[i]!='null'}
            additions=extra[role]-first[role]
            repeat[role+'_extra_unique_tp']+=len(additions&gold_spans)
            repeat[role+'_extra_unique_fp']+=len(additions-gold_spans)
        for q in p['pairs']:
            pair=(q['Aspect'].lower(),q['Opinion'].lower())
            if q['probability']<.65 or pair in gold:continue
            related=any(all(x==y or (x!='null' and y!='null' and (x in y or y in x)) for x,y in zip(pair,g)) for g in gold)
            errors['substring_related_pair_fp' if related else 'other_pair_fp']+=1
            case_details.append({'corpus':key[0],'ID':key[1],'pair':pair,'substring_related':related})
    costs=Counter()
    for f in (OUT/'cache/choice/calls').glob('*.json'):
        t=json.loads(f.read_text());stage=t['state']['stage']
        if stage!='pair':stage='continuation' if any(q['instructions']['already_selected'] for q in t['questions'].values()) else 'initial'
        costs[stage]+=t['usage']['input_tokens']
    save(OUT/'cache/choice_pair_fp.json',case_details)
    save(OUT/'comparison.json',{'common_completed':common,'common_ids':[{'corpus':c,'ID':i} for c,i in keys],
        'historical_bio_full_trial':metrics(historical,list(rows)),'choice_repeat_diagnostic':dict(repeat),
        'choice_pair_errors':dict(errors),'choice_all_call_stage_cost':dict(costs),
        'total_input_tokens':br['input_tokens']+cr['input_tokens'],'total_estimated_usd':br['estimated_usd']+cr['estimated_usd'],
        'decision':{'advance_to_dev':False,'reason':'Choice did not complete trial within the frozen input budget; full-trial performance gate not evaluated.'},
        'limitations':['The completed intersection is budget-censored (3/3/3/2 reviews), not the planned full trial.',
                       'Old BIO r3 differs in demonstrations, rules, pairing wording and NULL-opinion support.',
                       'Repeated-anchor contributions are offline descriptive counts, not a separate ablation run.']})
    print(json.dumps({'common':{k:{'n':v['n'],'macro':v['macro_pair_f1'],'pair':v['micro']['pair'],'aspect':v['micro']['aspect'],'opinion':v['micro']['opinion'],'input':v['input_tokens_completed_records']} for k,v in common.items()},'repeat':dict(repeat),'cost':dict(costs),'errors':dict(errors)},indent=2))
