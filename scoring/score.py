#!/usr/bin/env python3
"""Run the OFFICIAL DimABSA scorer, unmodified.

The official script is `vendor/DimABSA2026/evaluation_script/metrics_subtask_1_2_3.py`.
We never reimplement cF1 or RMSE here; this wrapper only prepares inputs and
invokes it.

Note on `--do_norm`: the official flag defaults to off, and off yields the raw
(un-normalised) RMSE_VA -- the number comparable to the shared-task leaderboard.
Passing `--do_norm` divides by sqrt(128). Any reported number must state which.

Subtask 1 gold is derived from the source file when needed, because the official
scorer reads `Aspect_VA` and the trial files only carry `Quadruplet`.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OFFICIAL = ROOT / "vendor" / "DimABSA2026" / "evaluation_script" / "metrics_subtask_1_2_3.py"

sys.path.insert(0, str(ROOT))
from jev.data import load_jsonl, st1_gold, write_jsonl  # noqa: E402


def derive_st1_gold(source: str | Path, destination: str | Path) -> Path:
    destination = Path(destination)
    write_jsonl(destination, st1_gold(load_jsonl(source)))
    return destination


def needs_st1_gold_derivation(source: str | Path) -> bool:
    """Trial files carry only `Quadruplet`; dev files already carry `Aspect_VA`.

    Dev gold must be used verbatim: it may repeat an aspect within a sentence
    (one aspect attracting two opinions), and the official scorer counts every
    gold entry while collapsing predictions to one per aspect.
    """
    records = load_jsonl(source)
    return not any(record.get("Aspect_VA") for record in records)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--gold", required=True, help="gold jsonl (any task file for task 1)")
    parser.add_argument("--pred", required=True)
    parser.add_argument("--do_norm", action="store_true",
                        help="pass through to the official scorer (divides RMSE by sqrt(128))")
    parser.add_argument("--workdir", default=str(ROOT / "reports"))
    args = parser.parse_args()

    gold_path = Path(args.gold)
    if args.task == 1 and needs_st1_gold_derivation(args.gold):
        gold_path = derive_st1_gold(
            args.gold, Path(args.workdir) / f"_gold_st1_{Path(args.gold).stem}.jsonl"
        )
        print(f"(derived subtask-1 gold -> {gold_path})")

    command = [
        sys.executable, str(OFFICIAL),
        "-t", str(args.task),
        "-g", str(gold_path),
        "-p", args.pred,
    ]
    if args.do_norm:
        command.append("--do_norm")

    print("$ " + " ".join(command), "\n")
    return subprocess.call(command)


if __name__ == "__main__":
    sys.exit(main())
