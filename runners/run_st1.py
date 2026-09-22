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

from jev.client import JevClient, format_va, score_to_va  # noqa: E402
from jev.data import aspects_for_inference, load_jsonl  # noqa: E402
from jev.fewshot import ExampleSet, build_state, load_examples  # noqa: E402
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
    args = parser.parse_args()

    records = load_jsonl(args.data)
    if args.limit:
        records = records[: args.limit]

    language, domain = corpus_of(args.data)
    examples = load_examples(language, domain, args.shots)

    out = Path(args.out)
    meta_path = Path(str(out) + ".meta.json")
    previous_meta: dict = {}

    if args.restart and out.exists():
        out.unlink()
        meta_path.unlink(missing_ok=True)
    elif out.exists():
        # Resuming across a rubric change or a different shot count would
        # silently mix two instruments in one prediction file, so refuse rather
        # than guess.
        if meta_path.exists():
            previous_meta = json.loads(meta_path.read_text())
            previous = previous_meta.get("rubric_sha256_12")
            if previous and previous != rubric_fingerprint():
                print(
                    f"refusing to resume: {out} was produced with rubric {previous}, "
                    f"current rubric is {rubric_fingerprint()}. Re-run with --restart.",
                    file=sys.stderr,
                )
                return 2
            previous_shots = previous_meta.get("shots")
            if previous_shots is not None and previous_shots != args.shots:
                print(
                    f"refusing to resume: {out} was produced with shots={previous_shots}, "
                    f"this run uses shots={args.shots}. Re-run with --restart.",
                    file=sys.stderr,
                )
                return 2
    done = completed_ids(out)
    todo = [r for r in records if r["ID"] not in done]
    if done:
        print(f"resuming: {len(done)} already done, {len(todo)} to go")

    client = JevClient()
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

    # Usage accumulates across runs. A resumed run only performs the records that
    # are missing, so overwriting would make the file's token total describe the
    # last fragment rather than everything that was actually spent producing it.
    # Retries inside a request are invisible here -- the API only reports usage on
    # a successful response -- so totals are a lower bound on real spend.
    previous_usage = previous_meta.get("usage_total", {})
    usage_total = {
        key: previous_usage.get(key, 0) + collector.usage.get(key, 0)
        for key in set(previous_usage) | set(collector.usage)
    }
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
        "data": str(args.data),
        "predictions": str(out),
        "records": len(records),
        "records_in_file": len(done) + finished,
        "model_requested": client.model,
        "model_returned": sorted(set(previous_meta.get("model_returned", []))
                                 | collector.models),
        "rubric_sha256_12": rubric_fingerprint(),
        "shots": args.shots,
        "corpus": f"{language}_{domain}",
        # Frozen per corpus: recorded so a run is reproducible and so it is
        # auditable that the examples were never re-picked against results.
        "example_set": examples.ids_for_meta,
        "usage_total": usage_total,
        "usage_this_run": collector.usage,
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
