#!/usr/bin/env python3
"""Offline design diagnostics; no API calls and no changes to baseline predictions.

Keeps every gold annotation, matching official task-1 weighting. Development
calibration is exploratory, within-corpus, text-grouped five-fold OOF; it is
not a new test result. Only aggregate numbers and file hashes are persisted.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jev.data import load_jsonl, write_jsonl
from runners.run_all_st1 import CORPORA

REPORTS = ROOT / "reports"
DATA = ROOT / "vendor/DimABSA2026/task-dataset/track_a/subtask_1"
SOURCES = {}


def read(path):
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return load_jsonl(path)


def aligned(corpus, split, suffix=""):
    gold_path = DATA / corpus.split("_")[0] / f"{corpus}_{split}_task1.jsonl"
    gold = read(gold_path)
    pred = read(REPORTS / f"pred_st1_{corpus}_{split}{suffix}.jsonl")
    by_id = {r["ID"]: {a["Aspect"].lower(): list(map(float, a["VA"].split("#")))
                        for a in r["Aspect_VA"]} for r in pred}
    assert len(by_id) == len(pred), "duplicate prediction IDs"
    x, y, folds, keys = [], [], [], []
    for record in gold:
        text = " ".join(record["Text"].lower().split())
        fold = int(hashlib.sha256(text.encode()).hexdigest(), 16) % 5
        for item in record["Aspect_VA"]:
            key = (record["ID"], item["Aspect"].lower())
            x.append(by_id[key[0]][key[1]])
            y.append(list(map(float, item["VA"].split("#"))))
            folds.append(fold)
            keys.append(key)
    return np.array(x), np.array(y), np.array(folds), keys, gold_path


def metrics(x, y):
    error = x - y
    mse = (error**2).mean(axis=0)
    bias = error.mean(axis=0)
    return dict(n=len(y), rmse_va=float(np.sqrt(mse.sum())),
                rmse_v=float(np.sqrt(mse[0])), rmse_a=float(np.sqrt(mse[1])),
                bias_v=float(bias[0]), bias_a=float(bias[1]),
                arousal_sse_share=float(mse[1] / mse.sum()))


def official_check(gold_path, keys, values):
    rows = {}
    for (record_id, aspect), value in zip(keys, values):
        rows.setdefault(record_id, {})[aspect] = f"{value[0]:.2f}#{value[1]:.2f}"
    with tempfile.TemporaryDirectory(prefix="jev-design-") as directory:
        path = Path(directory) / "pred.jsonl"
        write_jsonl(path, [{"ID": rid, "Aspect_VA": [
            {"Aspect": aspect, "VA": va} for aspect, va in aspects.items()]}
            for rid, aspects in rows.items()])
        result = subprocess.run([sys.executable, str(ROOT / "scoring/score.py"),
                                 "--task", "1", "--gold", str(gold_path),
                                 "--pred", str(path)], capture_output=True, text=True)
        match = re.search(r"Final Results: (\{.*\})", result.stdout)
        if result.returncode or not match:
            raise RuntimeError(result.stdout + result.stderr)
        literal = re.sub(r"np\.float64\(([^)]*)\)", r"\1", match[1])
        # Constant calibration outputs have undefined PCC, but valid RMSE.
        parsed = ast.literal_eval(re.sub(r"\bnan\b", "None", literal))
        return float(parsed["RMSE_VA"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--verify-official", action="store_true")
    args = parser.parse_args()
    result = {"method": "cached predictions only; gold entries not deduplicated; "
              "dev OOF folds = SHA256(normalized text) modulo 5 within each corpus; "
              "annotation-weighted OLS fits; clip to [1,9]; no test calibration",
              "test_diagnostics": {}, "dev_oof": {}}
    for suffix in ("", "_s3", "_s3_stratified", "_s5_stratified", "_s9_stratified"):
        rows, xs, ys = [], [], []
        for lang, domain in CORPORA:
            corpus = f"{lang}_{domain}"
            x, y, _, _, _ = aligned(corpus, "test", suffix)
            rows.append({"corpus": corpus, **metrics(x, y)})
            xs.append(x)
            ys.append(y)
        result["test_diagnostics"][suffix or "zero"] = {
            "corpora": rows, "aggregate": metrics(np.concatenate(xs), np.concatenate(ys))}
    predictions = {key: [] for key in ("raw", "mean", "offset", "affine")}
    ys, rows = [], []
    for lang, domain in CORPORA:
        corpus = f"{lang}_{domain}"
        x, y, fold, keys, gold_path = aligned(corpus, "dev")
        outputs = {key: np.zeros_like(x) for key in predictions}
        outputs["raw"] = x
        for index in range(5):
            train, valid = fold != index, fold == index
            assert train.any() and valid.any()
            outputs["mean"][valid] = y[train].mean(axis=0)
            outputs["offset"][valid] = np.clip(x[valid] + (y[train] - x[train]).mean(axis=0), 1, 9)
            for dimension in (0, 1):
                design = np.column_stack([np.ones(train.sum()), x[train, dimension]])
                intercept, slope = np.linalg.lstsq(design, y[train, dimension], rcond=None)[0]
                outputs["affine"][valid, dimension] = np.clip(
                    intercept + slope * x[valid, dimension], 1, 9)
        row = {"corpus": corpus}
        for name, values in outputs.items():
            row[name] = metrics(values, y)
            predictions[name].append(values)
        if args.verify_official:
            official = official_check(gold_path, keys, outputs["affine"])
            rounded = metrics(np.round(outputs["affine"], 2), y)["rmse_va"]
            # The official scorer prints four decimal places.
            assert abs(official - rounded) <= 0.000051, (corpus, official, rounded)
            row["affine_official_rmse_rounded_predictions"] = official
        ys.append(y)
        rows.append(row)
    result["dev_oof"] = {"corpora": rows, "aggregate": {
        name: metrics(np.concatenate(values), np.concatenate(ys))
        for name, values in predictions.items()}}
    result["input_sha256"] = SOURCES
    result["analysis_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["dev_oof"]["aggregate"], indent=2))
    print(f"Saved {args.out}; official verification: {args.verify_official}")


if __name__ == "__main__":
    main()
