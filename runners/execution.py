"""Shared concurrent execution, resumable JSONL output and usage accounting."""
from __future__ import annotations
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from jev.data import load_jsonl

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


def execute(args, records, client, predictor, config, corpus, fingerprint, example_meta=None):
    out = Path(args.out)
    meta_path = Path(str(out) + ".meta.json")
    previous_meta: dict = {}

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

    collector = Collector()
    started = time.time()
    finished = 0

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8") as sink, \
            ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        futures = {pool.submit(predictor, client, record): record
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
    for entry in load_jsonl(out) if out.exists() else []:
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
        "rubric_sha256_12": fingerprint,
        "shots": args.shots,
        "corpus": corpus,
        # Frozen per corpus: recorded so a run is reproducible and so it is
        # auditable that the examples were never re-picked against results.
        "example_set": example_meta,
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
