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
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                done.add(json.loads(line)["ID"])
    return done


class Collector:
    """Thread-safe accumulator for usage/model stats."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.usage: dict[str, int] = {}
        self.models: set[str] = set()
        self.errors: list[str] = []

    def add(self, response) -> None:
        with self._lock:
            self.models.add(response.model)
            for key, value in response.usage.items():
                self.usage[key] = self.usage.get(key, 0) + value

    def record_error(self, record_id: str, message: str) -> None:
        with self._lock:
            self.errors.append(f"{record_id}: {message}")


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

    out = Path(args.out)
    meta_path = Path(str(out) + ".meta.json")
    previous_meta: dict = {}

    # Store only what determines the request; do this before issuing any calls.
    config = {
        "model": args.model, "shots": args.shots, "examples": examples.payload(),
        "questions": build_questions(["__aspect__"], len(examples.examples)),
    }
    if args.restart:
        out.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)
    elif out.exists() or meta_path.exists():
        previous_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        if previous_meta.get("config") != config:
            print("refusing to resume: missing or different request configuration; "
                  "use a new output path or explicit --restart", file=sys.stderr)
            return 2
    out.parent.mkdir(parents=True, exist_ok=True)
    previous_meta["config"] = config
    meta_path.write_text(json.dumps(previous_meta, indent=2) + "\n")
    done = completed_ids(out)
    todo = [r for r in records if r["ID"] not in done]
    if done:
        print(f"resuming: {len(done)} already done, {len(todo)} to go")

    client = JevClient(model=args.model)
    collector = Collector()
    started = time.time()
    finished = 0

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8") as sink, \
            ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        futures = {pool.submit(predict, client, record, examples): record
                   for record in todo}
        for future in as_completed(futures):
            record = futures[future]
            try:
                entry = future.result()
            except Exception as exc:  # keep going; report at the end
                collector.record_error(record["ID"], f"{type(exc).__name__}: {exc}")
                continue
            response = entry.pop("_response")
            if response is not None:
                collector.add(response)
            # Single writer: only this loop touches the file.
            sink.write(json.dumps(entry, ensure_ascii=False) + "\n")
            sink.flush()
            finished += 1
            if not args.quiet and finished % 25 == 0:
                rate = finished / max(time.time() - started, 1e-6)
                print(f"  {finished}/{len(todo)}  ({rate:.1f}/s)", flush=True)

    # Each completed row carries its own usage, so interrupted runs need no journal.
    usage_total: dict[str, int] = {}
    models: set[str] = set()
    for entry in load_jsonl(out):
        raw = entry.get("_jev", {})
        if raw.get("model"):
            models.add(raw["model"])
        for key, value in raw.get("usage", {}).items():
            usage_total[key] = usage_total.get(key, 0) + value
    run = {
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "records_completed": finished,
        "concurrency": args.concurrency,
        "usage": collector.usage,
        "failed": collector.errors,
        "seconds": round(time.time() - started, 1),
    }
    runs = previous_meta.get("runs", []) + [run]

    meta = {
        "config": config,
        "data": str(args.data),
        "predictions": str(out),
        "records": len(records),
        "records_in_file": len(done) + finished,
        "model_requested": client.model,
        "model_returned": sorted(models),
        "rubric_sha256_12": rubric_fingerprint(),
        "shots": args.shots,
        "corpus": f"{language}_{domain}",
        # Frozen per corpus: recorded so a run is reproducible and so it is
        # auditable that the examples were never re-picked against results.
        "example_set": examples.ids_for_meta,
        "usage_total": usage_total,
        "usage_this_run": collector.usage,
        "usage_note": "Returned usage only; unreported failed-request charges are unknown.",
        "runs": runs,
    }
    Path(str(out) + ".meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"wrote {finished}/{len(todo)} records -> {out}")
    if collector.errors:
        print(f"{len(collector.errors)} FAILED (rerun to retry them):")
        for line in collector.errors[:10]:
            print("  " + line)
    print(json.dumps({k: meta[k] for k in
                      ("model_returned", "rubric_sha256_12", "usage_total")}, indent=2))
    return 1 if collector.errors else 0


if __name__ == "__main__":
    sys.exit(main())
