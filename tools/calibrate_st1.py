#!/usr/bin/env python3
"""Small train-calibration / dev-selection / frozen-test experiment. No preflight.

Run `dev`, then `test`. Interrupted inference resumes through run_st1.py.
Only cache/ contains dataset text; parameters and reports contain aggregates/IDs.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jev.data import _annotation_items, load_jsonl, write_jsonl
from jev.fewshot import load_examples, normalise, train_path, split_path
from runners.run_all_st1 import CORPORA
from tools.analyze_st1_design import official_check

SEED = 20260923
ARM_SHOTS = (0, 9)
METHODS = ("raw", "mean", "offset", "shrink", "linear")
PARALLEL = {"rus_restaurant", "tat_restaurant", "ukr_restaurant"}
DEFAULT_OUT = ROOT / "reports/calibration_20260923"


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def prepare(out):
    manifest_path = out / "samples.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    records, examples, parent, banned = {}, {}, {}, set()

    def find(key):
        parent.setdefault(key, key)
        if parent[key] != key:
            parent[key] = find(parent[key])
        return parent[key]

    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)

    parallel_ids = {}
    for lang, domain in CORPORA:
        c = f"{lang}_{domain}"
        records[c] = load_jsonl(train_path(lang, domain))
        examples[c] = load_examples(lang, domain, 9, strategy="stratified")
        heldout = [r for split in ("dev", "test")
                   for r in load_jsonl(split_path(lang, domain, split))]
        excluded_ids = {r["ID"] for r in heldout} | set(examples[c].source_ids)
        excluded_text = {normalise(r["Text"]) for r in heldout} | {
            normalise(e.review) for e in examples[c].examples}
        by_text = {}
        for r in records[c]:
            key = f"{c}:{r['ID']}"
            find(key)
            text = normalise(r["Text"])
            if text in by_text:
                union(key, by_text[text])
            by_text[text] = key
            if c in PARALLEL:
                if r["ID"] in parallel_ids:
                    union(key, parallel_ids[r["ID"]])
                parallel_ids[r["ID"]] = key
            if r["ID"] in excluded_ids or text in excluded_text:
                banned.add(key)
    banned_groups = {find(key) for key in banned}
    manifest = {"seed": SEED, "groups_per_corpus": 256, "corpora": {}}
    for lang, domain in CORPORA:
        c = f"{lang}_{domain}"
        groups = {find(f"{c}:{r['ID']}") for r in records[c]} - banned_groups
        selected = set(sorted(groups, key=lambda g: digest([SEED, g]))[:256])
        if len(selected) != 256:
            raise ValueError(f"{c}: only {len(selected)} eligible groups")
        sampled = [r for r in records[c] if find(f"{c}:{r['ID']}") in selected]
        cache = out / "cache" / c
        write_jsonl(cache / f"{c}_train_alltasks.jsonl", sampled)
        frozen = [asdict(e) for e in examples[c].examples]
        save(cache / "examples.json", frozen)
        manifest["corpora"][c] = {
            "ids": [r["ID"] for r in sampled],
            "groups": {r["ID"]: digest(find(f"{c}:{r['ID']}")) for r in sampled},
            "examples": examples[c].source_ids,
            "examples_sha256": digest(frozen),
        }
    save(manifest_path, manifest)
    return manifest


def infer(out, c, source, stage, shots):
    path = out / "cache" / c / f"{stage}_s{shots}.jsonl"
    command = [sys.executable, str(ROOT / "runners/run_st1.py"),
               "--data", str(source), "--out", str(path), "--shots", str(shots),
               "--example-selection", "stratified", "--concurrency", "10", "--quiet",
               "--examples", str(out / "cache" / c / "examples.json")]
    for attempt in range(3):
        print(f"{c} {stage} s{shots} attempt {attempt + 1}", flush=True)
        proc = subprocess.run(command, capture_output=True, text=True)
        with open(out / "run.log", "a") as log:
            log.write(proc.stdout + proc.stderr)
        if proc.returncode == 0:
            break
        if proc.returncode != 1 or attempt == 2:
            raise RuntimeError(proc.stdout[-2000:] + proc.stderr[-2000:])
    return path


def align(source, pred, group_map=None, corpus=""):
    gold, predictions = load_jsonl(source), load_jsonl(pred)
    by_id = {r["ID"]: r for r in predictions}
    x, y, groups, keys = [], [], [], []
    text_ids = {}
    for r in gold:
        p = by_id[r["ID"]]
        by_aspect = {}
        for i, a in enumerate(p["Aspect_VA"]):
            raw = p.get("_jev", {}).get("answers", {})
            by_aspect[a["Aspect"].lower()] = [raw[f"{d}{i}"]["score"] + 1 for d in ("v", "a")]
        text = normalise(r["Text"])
        default_group = r["ID"] if corpus in PARALLEL else text
        group = group_map[r["ID"]] if group_map else text_ids.setdefault(text, default_group)
        for item in _annotation_items(r):
            x.append(by_aspect[item["Aspect"].lower()])
            y.append(list(map(float, item["VA"].split("#"))))
            groups.append(group)
            keys.append((r["ID"], item["Aspect"].lower()))
    return np.array(x), np.array(y), np.array(groups), keys


def coefficients(x, y, method):
    mx, my = x.mean(axis=0), y.mean(axis=0)
    if method == "raw":
        slope = np.ones(2); intercept = np.zeros(2)
    elif method == "mean":
        slope = np.zeros(2); intercept = my
    elif method == "offset":
        slope = np.ones(2); intercept = my - mx
    elif method == "linear":
        variance = ((x - mx)**2).mean(axis=0)
        covariance = ((x - mx) * (y - my)).mean(axis=0)
        slope = np.maximum(0, np.divide(covariance, variance, out=np.zeros(2), where=variance > 1e-12))
        intercept = my - slope * mx
    else:
        raise ValueError(method)
    return {"intercept": intercept.tolist(), "slope": slope.tolist()}


def apply(x, parameters):
    return np.clip(np.array(parameters["intercept"]) + np.array(parameters["slope"]) * x, 1, 9)


def fit(x, y, groups):
    result = {m: coefficients(x, y, m) for m in METHODS if m != "shrink"}
    unique = sorted(set(groups), key=lambda g: digest([SEED, str(g)]))
    fold_for = {g: i % 5 for i, g in enumerate(unique)}
    folds = np.array([fold_for[g] for g in groups])
    alphas = np.array([0, .25, .5, .75, 1])
    loss = np.zeros((len(alphas), 2))
    for fold in range(5):
        train, valid = folds != fold, folds == fold
        mx, my = x[train].mean(axis=0), y[train].mean(axis=0)
        for i, alpha in enumerate(alphas):
            loss[i] += ((np.clip(my + alpha * (x[valid] - mx), 1, 9) - y[valid])**2).sum(axis=0)
    slope = alphas[loss.argmin(axis=0)]
    result["shrink"] = {"slope": slope.tolist(),
                        "intercept": (y.mean(axis=0) - slope * x.mean(axis=0)).tolist()}
    return result


def metrics(pred, gold):
    error = pred - gold
    mse = (error**2).mean(axis=0)
    pcc = []
    for d in (0, 1):
        pcc.append(float(np.corrcoef(pred[:, d], gold[:, d])[0, 1])
                   if pred[:, d].std() > 1e-12 and gold[:, d].std() > 1e-12 else None)
    return {"n_gold": len(gold), "RMSE_VA": float(np.sqrt(mse.sum())),
            "RMSE_V": float(np.sqrt(mse[0])), "RMSE_A": float(np.sqrt(mse[1])),
            "bias_V": float(error[:, 0].mean()), "bias_A": float(error[:, 1].mean()),
            "PCC_V": pcc[0], "PCC_A": pcc[1]}


def bootstrap(rows, candidate):
    # Cluster within each corpus, with all three translated corpora sharing a stratum.
    strata = {}
    for c, x, y, groups, parameters in rows:
        before = ((apply(x, parameters["raw"]) - y)**2).sum(axis=1)
        after = ((apply(x, parameters[candidate]) - y)**2).sum(axis=1)
        by_group = strata.setdefault("parallel" if c in PARALLEL else c, {})
        for g, b, a in zip(groups, before, after):
            entry = by_group.setdefault(str(g), np.zeros(3))
            entry += (b, a, 1)
    rng = np.random.default_rng(SEED)
    totals = np.zeros((2000, 3))
    for clusters in strata.values():
        values = np.array(list(clusters.values()))
        for i in range(2000):
            totals[i] += values[rng.integers(0, len(values), len(values))].sum(axis=0)
    delta = np.sqrt(totals[:, 1] / totals[:, 2]) - np.sqrt(totals[:, 0] / totals[:, 2])
    return np.quantile(delta, [.025, .975]).tolist()


def aggregate(rows, method):
    x = np.concatenate([apply(x, p[method]) for _, x, y, g, p in rows])
    y = np.concatenate([y for _, x, y, g, p in rows])
    m = metrics(x, y)
    # Pooled PCC is explicitly labelled; per-corpus PCC stays in the report.
    m["pooled_PCC_V"] = m.pop("PCC_V")
    m["pooled_PCC_A"] = m.pop("PCC_A")
    return m


def usage(out):
    total, records, models = 0, 0, set()
    for path in (out / "cache").glob("*/*_s*.jsonl"):
        for record in load_jsonl(path):
            raw = record.get("_jev", {})
            if not raw:
                continue
            total += raw.get("usage", {}).get("input_tokens", 0)
            records += 1
            if raw.get("model"):
                models.add(raw["model"])
    return {"input_tokens": total, "completed_records": records, "models": sorted(models),
            "estimated_usd": total / 1e6 * .042,
            "note": "Known response tokens only; unreported failed-request charges are unknown."}


def run_dev(out, samples):
    if (out / "selection.json").exists():
        print("Selection already frozen; use a new output directory for another experiment.")
        return
    params, report, selection = {}, {}, {}
    for shots in ARM_SHOTS:
        rows, per_corpus = [], {}
        for lang, domain in CORPORA:
            c = f"{lang}_{domain}"
            source = out / "cache" / c / f"{c}_train_alltasks.jsonl"
            pred = infer(out, c, source, "calibration", shots)
            x, y, g, _ = align(source, pred, samples["corpora"][c]["groups"], c)
            fitted = fit(x, y, g)
            dev = split_path(lang, domain, "dev")
            pred = infer(out, c, dev, "dev", shots)
            x, y, g, _ = align(dev, pred, corpus=c)
            rows.append((c, x, y, g, fitted))
            per_corpus[c] = {m: metrics(apply(x, fitted[m]), y) for m in METHODS}
            params.setdefault(str(shots), {})[c] = fitted
            save(out / "parameters.json", params)
        totals = {m: aggregate(rows, m) for m in METHODS}
        best = min(totals[m]["RMSE_VA"] for m in METHODS)
        chosen = next(m for m in METHODS if totals[m]["RMSE_VA"] <= best + .02)
        interval = bootstrap(rows, chosen)
        improvement = totals["raw"]["RMSE_VA"] - totals[chosen]["RMSE_VA"]
        accepted = chosen if improvement >= .02 and interval[1] < 0 else "raw"
        selection[str(shots)] = {"method": accepted, "candidate": chosen,
                                  "candidate_delta_ci95": interval}
        report[str(shots)] = {"aggregate": totals, "corpora": per_corpus,
                              "selection": selection[str(shots)]}
        save(out / "dev_summary.json", report)
        print(f"s{shots}: dev selected {accepted}; {totals[accepted]}", flush=True)
    selection["parameters_sha256"] = digest(params)
    selection["samples_sha256"] = digest(samples)
    selection["model"] = "jev-1.13.0"
    selection["request_configs_sha256"] = {
        str(path.relative_to(out)): digest(json.loads(path.read_text())["config"])
        for path in (out / "cache").glob("*/dev_s*.jsonl.meta.json")}
    save(out / "selection.json", selection)


def run_test(out, samples):
    selection = json.loads((out / "selection.json").read_text())
    params = json.loads((out / "parameters.json").read_text())
    if digest(params) != selection["parameters_sha256"] or digest(samples) != selection["samples_sha256"]:
        raise ValueError("Frozen parameters or samples changed")
    report = {}
    for shots in ARM_SHOTS:
        chosen = selection[str(shots)]["method"]
        rows, per_corpus = [], {}
        for lang, domain in CORPORA:
            c = f"{lang}_{domain}"
            source = split_path(lang, domain, "test")
            pred = infer(out, c, source, "test", shots)
            meta = json.loads(Path(str(pred) + ".meta.json").read_text())
            dev_config = f"cache/{c}/dev_s{shots}.jsonl.meta.json"
            if digest(meta["config"]) != selection["request_configs_sha256"][dev_config]:
                raise ValueError("Test request differs from the frozen dev request")
            x, y, g, keys = align(source, pred, corpus=c)
            fitted = params[str(shots)][c]
            rows.append((c, x, y, g, fitted))
            per_corpus[c] = {}
            for method in dict.fromkeys(["raw", chosen]):
                values = apply(x, fitted[method])
                rounded = np.round(values, 2)
                local = metrics(rounded, y)
                official = official_check(source, keys, values)
                if abs(official - local["RMSE_VA"]) > .000051:
                    raise ValueError("Official scorer disagrees")
                local["official_RMSE_VA"] = official
                per_corpus[c][method] = local
                by_id = {}
                for (rid, aspect), value in zip(keys, values):
                    by_id.setdefault(rid, {})[aspect] = f"{value[0]:.2f}#{value[1]:.2f}"
                write_jsonl(out / "cache" / c / f"test_s{shots}_{method}_export.jsonl", [
                    {"ID": rid, "Aspect_VA": [{"Aspect": a, "VA": va} for a, va in items.items()]}
                    for rid, items in by_id.items()])
        # Aggregate the official-format rounded predictions, not mean corpus RMSE.
        totals = {}
        for method in dict.fromkeys(["raw", chosen]):
            total_n = sum(v[method]["n_gold"] for v in per_corpus.values())
            totals[method] = {"n_gold": total_n, "RMSE_VA": float(np.sqrt(sum(
                v[method]["n_gold"] * v[method]["RMSE_VA"]**2 for v in per_corpus.values()) / total_n))}
        report[str(shots)] = {"method": chosen, "aggregate": totals,
                              "delta_ci95_unrounded": bootstrap(rows, chosen), "corpora": per_corpus}
        save(out / "test_summary.json", report)
        print(f"s{shots}: test {totals}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("dev", "test"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    samples = prepare(out) if args.phase == "dev" else json.loads((out / "samples.json").read_text())
    try:
        (run_dev if args.phase == "dev" else run_test)(out, samples)
    finally:
        save(out / "usage.json", usage(out))


if __name__ == "__main__":
    main()
