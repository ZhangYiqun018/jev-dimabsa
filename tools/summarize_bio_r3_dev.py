#!/usr/bin/env python3
"""Offline reporting of the full BIO dev run; no model requests."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/bio_r3_full_dev_20260923'


def main():
    result=json.loads((OUT/'summary.json').read_text())
    if not result['complete']:raise SystemExit('Full dev incomplete; no full-dataset comparison written.')
    old=json.loads((ROOT/'reports/st2_baseline_20260923/task2_dev_summary.json').read_text())
    old={r['corpus']:r for r in old['corpora']}
    rows=[]
    for row in result['corpora']:
        c=row['corpus'];metric=row['calibrated']
        tp,fp,fn=metric['categorical_TP'],metric['FP'],metric['FN']
        official_ao=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
        rows.append({'corpus':c,'n':row['completed'],'ao_set_f1':row['ao']['pair']['f1'],
                     'official_categorical_f1':official_ao,'raw_cf1':row['raw']['cF1'],
                     'calibrated_cf1':metric['cF1'],'old_lexicon_dev_cf1':old[c]['metrics']['cF1'],
                     'delta_vs_lexicon':metric['cF1']-old[c]['metrics']['cF1']})
    tp=sum(r['calibrated']['categorical_TP'] for r in result['corpora'])
    fp=sum(r['calibrated']['FP'] for r in result['corpora'])
    fn=sum(r['calibrated']['FN'] for r in result['corpora'])
    ctp=sum(r['calibrated']['TP'] for r in result['corpora'])
    summary={'corpora':rows,'macro_ao_set_f1':result['macro_ao_f1'],
             'macro_official_categorical_f1':sum(r['official_categorical_f1'] for r in rows)/len(rows),
             'macro_raw_cf1':result['macro_raw_cf1'],'macro_calibrated_cf1':result['macro_calibrated_cf1'],
             'macro_lexicon_cf1':sum(r['old_lexicon_dev_cf1'] for r in rows)/len(rows),
             'micro_official_categorical':{'tp':tp,'fp':fp,'fn':fn,'f1':2*tp/(2*tp+fp+fn)},
             'micro_official_calibrated':{'ctp':ctp,'precision':ctp/(tp+fp),'recall':ctp/(tp+fn),'cf1':2*ctp/(2*tp+fp+fn)},
             'note':'All new/lexicon comparisons above use the same full dev. Official competition results are test and do not constitute a ranking of this run.'}
    (OUT/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
