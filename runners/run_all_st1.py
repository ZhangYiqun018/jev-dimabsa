#!/usr/bin/env python3
"""Run subtask 1 over every corpus of a split, score each, and tabulate against the
published baselines.

The official baseline figures are TEST-set numbers taken from the SemEval-2026
Task 3 overview paper (arXiv 2604.07066), so they are only directly comparable
when this runs with --split test.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "vendor" / "DimABSA2026" / "task-dataset"
REPORTS = ROOT / "reports"
PY = str(ROOT / ".venv" / "bin" / "python")

# Parallel translations of the same source items, so their gold is identical.
CORPORA = [
    ("eng", "restaurant"), ("eng", "laptop"),
    ("zho", "restaurant"), ("zho", "laptop"), ("zho", "finance"),
    ("jpn", "hotel"), ("jpn", "finance"),
    ("rus", "restaurant"), ("tat", "restaurant"), ("ukr", "restaurant"),
]

# Official baselines, SemEval-2026 Task 3 overview paper (TEST set).
# Kimi K2 Thinking one-shot / Qwen3-14B QLoRA.
OFFICIAL_TEST = {
    "eng_restaurant": (2.1461, 2.6427),
    "eng_laptop":     (2.1893, 2.8089),
    "jpn_hotel":      (1.7553, 2.2906),
    "jpn_finance":    (1.6396, 1.8964),
    "rus_restaurant": (1.7768, 2.1528),
    "tat_restaurant": (1.9380, 2.6367),
    "ukr_restaurant": (1.7805, 2.2121),
    "zho_restaurant": (1.8959, 2.0073),
    "zho_laptop":     (1.6440, 1.7706),
    "zho_finance":    (1.9652, 1.4707),
}

RESULTS_RE = re.compile(r"Final Results: (\{.*\})")


def source_file(language: str, domain: str, split: str) -> Path:
    return (DATA / "track_a" / "subtask_1" / language
            / f"{language}_{domain}_{split}_task1.jsonl")


def tag_for(shots: int, strategy: str) -> str:
    """Short suffix distinguishing runs, so configurations cannot clobber each other."""
    return f"s{shots}" if strategy == "first-k" else f"s{shots}_{strategy}"


def run_one(language: str, domain: str, split: str, concurrency: int,
            shots: int, strategy: str) -> dict:
    source = source_file(language, domain, split)
    corpus = f"{language}_{domain}"
    pred = REPORTS / f"pred_st1_{corpus}_{split}_{tag_for(shots, strategy)}.jsonl"

    # Exit 1 means some records failed after the client's own retries; every other
    # record is written and scoring them is valid, so this is not a failed corpus.
    # Resume (without --restart, which would delete what succeeded) to fill the
    # gap, then score whatever is on disk. Exit 2 is the runner refusing to resume
    # across an instrument change, which no retry can fix.
    base = [PY, str(ROOT / "runners" / "run_st1.py"),
            "--data", str(source), "--out", str(pred), "--quiet",
            "--concurrency", str(concurrency), "--shots", str(shots),
            "--example-selection", strategy]
    started = time.time()
    for attempt in range(3):
        run = subprocess.run(base + (["--restart"] if attempt == 0 else []),
                             capture_output=True, text=True)
        if run.returncode == 0:
            break
        if run.returncode != 1 or attempt == 2:
            # The runner prints its per-record errors to stdout, not stderr.
            detail = (run.stderr.strip() or run.stdout.strip())[-300:]
            return {"corpus": corpus, "error": f"exit {run.returncode}: {detail}"}
    elapsed = time.time() - started

    score = subprocess.run(
        [PY, str(ROOT / "scoring" / "score.py"),
         "--task", "1", "--gold", str(source), "--pred", str(pred)],
        capture_output=True, text=True,
    )
    match = RESULTS_RE.search(score.stdout)
    if not match:
        return {"corpus": corpus, "error": (score.stdout + score.stderr).strip()[-300:]}

    # The official script prints a Python dict repr, not JSON: single quotes and
    # np.float64(...) wrappers. Strip the wrappers and literal_eval it.
    literal = re.sub(r"np\.float64\(([^)]*)\)", r"\1", match.group(1))
    metrics = ast.literal_eval(literal)
    meta = json.loads(Path(str(pred) + ".meta.json").read_text())

    n_gold = 0
    for line in open(source, encoding="utf-8"):
        line = line.strip()
        if line:
            n_gold += len(json.loads(line).get("Aspect_VA", []))

    return {
        "corpus": corpus,
        "n_gold": n_gold,
        "RMSE_VA": round(float(metrics["RMSE_VA"]), 4),
        "PCC_V": round(float(metrics["PCC_V"]), 4),
        "PCC_A": round(float(metrics["PCC_A"]), 4),
        "seconds": round(elapsed, 1),
        # Records still absent after the retries. Summing each run's failures would
        # double-count a record that failed and then succeeded on a resume.
        "failed": meta.get("records", 0) - meta.get("records_in_file", 0),
        # usage_total accumulates across resumed runs; never read the last run's
        # fragment, which describes only the records that run happened to add.
        "input_tokens": meta.get("usage_total", {}).get("input_tokens"),
        "model": ",".join(meta.get("model_returned", [])),
        "rubric": meta.get("rubric_sha256_12"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--shots", type=int, default=3,
                        help="in-context calibration examples (0 = zero-shot control)")
    parser.add_argument("--example-selection", default="first-k",
                        choices=["first-k", "stratified"])
    args = parser.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)
    rows = []
    for language, domain in CORPORA:
        corpus = f"{language}_{domain}"
        print(f"\n{'=' * 70}\n{corpus}  [{args.split}, shots={args.shots}, "
              f"{args.example_selection}]\n{'=' * 70}", flush=True)
        row = run_one(language, domain, args.split, args.concurrency,
                      args.shots, args.example_selection)
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)
        (REPORTS / f"st1_{args.split}_summary_{tag_for(args.shots, args.example_selection)}.json").write_text(
            json.dumps(rows, indent=2) + "\n")

    show_official = args.split == "test"
    width = 92 if show_official else 62
    print("\n\n" + "=" * width)
    if show_official:
        print(f"{'corpus':<18}{'n':>6}{'Jev RMSE':>10}{'KimiK2':>9}{'Qwen3':>9}"
              f"{'vs Kimi':>9}{'PCC_V':>9}{'PCC_A':>9}")
    else:
        print(f"{'corpus':<18}{'n':>6}{'RMSE_VA':>11}{'PCC_V':>9}{'PCC_A':>9}")
    print("-" * width)

    tot_tok = 0
    for row in rows:
        if "error" in row:
            print(f"{row['corpus']:<18} ERROR {row['error'][:50]}")
            continue
        tot_tok += row.get("input_tokens") or 0
        if show_official:
            kimi, qwen = OFFICIAL_TEST[row["corpus"]]
            print(f"{row['corpus']:<18}{row['n_gold']:>6}{row['RMSE_VA']:>10.4f}"
                  f"{kimi:>9.4f}{qwen:>9.4f}{row['RMSE_VA'] - kimi:>+9.4f}"
                  f"{row['PCC_V']:>9.4f}{row['PCC_A']:>9.4f}")
        else:
            print(f"{row['corpus']:<18}{row['n_gold']:>6}{row['RMSE_VA']:>11.4f}"
                  f"{row['PCC_V']:>9.4f}{row['PCC_A']:>9.4f}")
    print("-" * width)
    print(f"'vs Kimi' = Jev RMSE - Kimi K2 one-shot RMSE. Negative means Jev is better.")
    if tot_tok:
        print(f"total input tokens: {tot_tok:,}  ->  ${tot_tok / 1e6 * 0.042:.4f} "
              f"(at $0.042/Mtok, input only)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
