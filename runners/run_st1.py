#!/usr/bin/env python3
"""Subtask 1 (DimASR): predict valence/arousal for the aspects given in the data.

One Jev call per sentence; two Score questions per aspect, evaluated in parallel
inside that call. Sentences are issued concurrently (``--concurrency``), and each
result is appended as it completes, so an interrupted run resumes by ID instead of
restarting.

Output order follows completion order, not input order. The official scorer keys
everything by ID, so this does not affect scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from runners.execution import execute

from jev.client import DEFAULT_MODEL, JevClient, format_va, score_to_va  # noqa: E402
from jev.data import aspects_for_inference, load_jsonl  # noqa: E402
from jev.fewshot import Example, ExampleSet, build_state, load_examples  # noqa: E402
from jev.rubrics import arousal_question, valence_question  # noqa: E402

RUBRICS_PATH = Path(__file__).resolve().parent.parent / "jev" / "rubrics.py"
FEWSHOT_PATH = Path(__file__).resolve().parent.parent / "jev" / "fewshot.py"


def rubric_fingerprint() -> str:
    """Which rubric produced a prediction file, so results stay attributable."""
    payload = RUBRICS_PATH.read_bytes() + FEWSHOT_PATH.read_bytes()
    return hashlib.sha256(payload).hexdigest()[:12]


def corpus_of(data_path: str) -> tuple[str, str]:
    """`.../subtask_1/zho/zho_restaurant_test_task1.jsonl` -> ("zho", "restaurant")."""
    stem = Path(data_path).stem
    language, _, rest = stem.partition("_")
    for split in ("_train_alltasks", "_train_task1", "_dev_task1", "_test_task1"):
        rest = rest.replace(split, "")
    return language, rest


def build_questions(aspects: list[str], n_examples: int = 0) -> dict[str, dict]:
    questions: dict[str, dict] = {}
    for index, aspect in enumerate(aspects):
        questions[f"v{index}"] = valence_question(aspect, n_examples=n_examples)
        questions[f"a{index}"] = arousal_question(aspect, n_examples=n_examples)
    return questions


def predict(client: JevClient, record: dict, examples: ExampleSet) -> dict:
    aspects = aspects_for_inference(record)
    entry = {"ID": record["ID"], "Text": record["Text"], "Aspect_VA": [], "_response": None}
    if not aspects:
        return entry
    n_examples = len(examples.examples)
    response = client.ask(
        build_state(record["Text"], examples),
        build_questions(aspects, n_examples),
    )
    entry["_response"] = response
    entry["_jev"] = {
        "model": response.model, "usage": response.usage,
        "attempts": response.attempts,
        "answers": {name: answer.raw for name, answer in response.answers.items()},
    }
    for index, aspect in enumerate(aspects):
        entry["Aspect_VA"].append(
            {
                "Aspect": aspect,
                "VA": format_va(
                    score_to_va(response.answers[f"v{index}"].score),
                    score_to_va(response.answers[f"a{index}"].score),
                ),
            }
        )
    return entry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="input jsonl (trial or dev)")
    parser.add_argument("--out", required=True, help="predictions jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=8,
                        help="sentences in flight at once (1 = sequential)")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--restart", action="store_true", help="ignore existing output")
    parser.add_argument("--shots", type=int, default=3,
                        help="in-context calibration examples per request (0 = zero-shot)")
    parser.add_argument("--example-selection", default="first-k",
                        choices=["first-k", "stratified"],
                        help="first-k = official protocol; stratified = one example "
                             "per equal-width band of the 1-9 scale")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--examples", type=Path, help="frozen Example dataclasses as JSON")
    args = parser.parse_args()

    records = load_jsonl(args.data)
    if args.limit:
        records = records[: args.limit]

    language, domain = corpus_of(args.data)
    examples = load_examples(language, domain, args.shots,
                             strategy=args.example_selection)

    if args.examples:
        examples = ExampleSet(
            corpus=f"{language}_{domain}", n_requested=args.shots, strategy="frozen",
            examples=[Example(**item) for item in json.loads(args.examples.read_text())]
            if args.shots else [],
        )

    config = {
        "model": args.model, "shots": args.shots, "examples": examples.payload(),
        "questions": build_questions(["__aspect__"], len(examples.examples)),
    }
    return execute(args, records, JevClient(model=args.model),
                   lambda client, record: predict(client, record, examples), config,
                   f"{language}_{domain}", rubric_fingerprint(), examples.ids_for_meta)


if __name__ == "__main__":
    sys.exit(main())
