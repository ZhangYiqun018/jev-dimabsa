#!/usr/bin/env python3
"""Unified Task 1 / Task 2 inference and official scoring entry point."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jev.client import DEFAULT_MODEL, JevClient
from jev.data import load_jsonl, write_jsonl
from jev.triplets import TripletPredictor
from runners.execution import execute
from runners.run_all_st1 import CORPORA, RESULTS_RE

DATA = ROOT / 'vendor/DimABSA2026/task-dataset/track_a'
CALIBRATION = ROOT / 'reports/calibration_20260923/parameters.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', type=int, required=True, choices=[1, 2])
    parser.add_argument('--split', choices=['dev', 'test'], default='dev')
    parser.add_argument('--corpus', default='all', choices=['all'] + [f'{a}_{b}' for a,b in CORPORA])
    parser.add_argument('--out', type=Path, required=True, help='run directory; resumable')
    parser.add_argument('--concurrency', type=int, default=8)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--shots', type=int, default=0, help='Task 1 only; Task 2 uses frozen transfer calibration')
    parser.add_argument('--example-selection', choices=['first-k', 'stratified'], default='first-k')
    args = parser.parse_args()
    if args.task == 2 and args.shots:
        parser.error('Task 2 baseline uses --shots 0')
    corpora = CORPORA if args.task == 1 else [(l,d) for l,d in CORPORA if d != 'finance']
    if args.corpus != 'all':
        corpora = [(l,d) for l,d in corpora if f'{l}_{d}' == args.corpus]
        if not corpora:
            parser.error('Task 2 has no finance corpus')
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for lang, domain in corpora:
        corpus = f'{lang}_{domain}'
        base = DATA / f'subtask_{args.task}' / lang
        source = base / f'{corpus}_{args.split}_task{args.task}.jsonl'
        pred = args.out / 'cache' / f'{corpus}_{args.split}.jsonl'
        records = load_jsonl(source)
        if args.limit:
            records = records[:args.limit]
        print(f'{corpus} task={args.task} split={args.split} n={len(records)}', flush=True)
        if args.task == 1:
            cmd = [sys.executable, str(ROOT/'runners/run_st1.py'), '--data', str(source),
                   '--out', str(pred), '--concurrency', str(args.concurrency),
                   '--shots', str(args.shots), '--example-selection', args.example_selection]
            if args.limit:
                cmd += ['--limit', str(args.limit)]
            status = subprocess.call(cmd)
        else:
            params = json.loads(CALIBRATION.read_text())['0'][corpus]['shrink']
            predictor = TripletPredictor(base/f'{corpus}_train_alltasks.jsonl',
                [base/f'{corpus}_{s}_task2.jsonl' for s in ('dev','test')], params)
            from types import SimpleNamespace
            run_args = SimpleNamespace(out=pred, data=source, model=DEFAULT_MODEL,
                restart=False, concurrency=args.concurrency, quiet=False, shots=0)
            config = dict(predictor.config, model=DEFAULT_MODEL,
                          data_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
            fingerprint = hashlib.sha256((ROOT/'jev/triplets.py').read_bytes()
                + (ROOT/'jev/rubrics.py').read_bytes()).hexdigest()[:12]
            status = execute(run_args, records, JevClient(), predictor, config, corpus, fingerprint)
        if status:
            return status
        gold = source
        if args.limit:
            gold = args.out/'cache'/f'{corpus}_{args.split}_gold.jsonl'
            write_jsonl(gold, records)
        result = subprocess.run([sys.executable, str(ROOT/'scoring/score.py'),
            '--task',str(args.task),'--gold',str(gold),'--pred',str(pred)], capture_output=True,text=True)
        match = RESULTS_RE.search(result.stdout)
        if result.returncode or not match:
            print(result.stdout + result.stderr, file=sys.stderr)
            return 1
        metrics = ast.literal_eval(re.sub(r'np\.float64\(([^)]*)\)',r'\1',match.group(1)))
        meta = json.loads(Path(str(pred)+'.meta.json').read_text())
        row = {'corpus':corpus,'records':len(records),'metrics':metrics,
               'usage':meta['usage_total'],'model':meta['model_returned']}
        rows.append(row)
        summary = {'task':args.task,'split':args.split,'limit':args.limit,'corpora':rows}
        (args.out/f'task{args.task}_{args.split}_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(row), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
