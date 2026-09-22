#!/usr/bin/env python3
"""One-shot leakage audit for the few-shot example source.

Few-shot examples are drawn from the train split. That is only safe if no
evaluation item also appears in train. ID disjointness is not sufficient: the
same sentence can be filed under different IDs in different splits, so the audit
also compares normalised Text.

Run once and read the report. This is deliberately NOT a per-run gate — it does
not execute in the hot path of any runner.

Writes reports/leakage_audit.json and prints a summary table.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "vendor/DimABSA2026/task-dataset"

CORPORA = [
    ("eng", "restaurant"), ("eng", "laptop"),
    ("zho", "restaurant"), ("zho", "laptop"), ("zho", "finance"),
    ("jpn", "hotel"), ("jpn", "finance"),
    ("rus", "restaurant"), ("tat", "restaurant"), ("ukr", "restaurant"),
]


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def texts(records: list[dict]) -> dict[str, str]:
    """normalised text -> first ID carrying it"""
    out: dict[str, str] = {}
    for r in records:
        out.setdefault(norm(r.get("Text", "")), r["ID"])
    return out


def _first_existing(*candidates: Path) -> Path:
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def st1_paths(lang: str, domain: str) -> dict[str, Path]:
    """Train file naming is inconsistent: most corpora use `_train_alltasks`,
    but the two finance corpora use `_train_task1`. Reading only the first
    pattern silently reports zero training data for finance."""
    base = DATA / "track_a" / "subtask_1" / lang
    stem = f"{lang}_{domain}"
    return {
        "train": _first_existing(
            base / f"{stem}_train_alltasks.jsonl",
            base / f"{stem}_train_task1.jsonl",
        ),
        "dev": base / f"{stem}_dev_task1.jsonl",
        "test": base / f"{stem}_test_task1.jsonl",
    }


def audit_corpus(lang: str, domain: str) -> dict:
    paths = st1_paths(lang, domain)
    recs = {split: load(p) for split, p in paths.items()}
    ids = {split: {r["ID"] for r in rs} for split, rs in recs.items()}
    txt = {split: texts(rs) for split, rs in recs.items()}

    def id_overlap(a: str, b: str) -> set[str]:
        return ids[a] & ids[b]

    def text_overlap(a: str, b: str) -> set[str]:
        return set(txt[a]) & set(txt[b])

    return {
        "corpus": f"{lang}_{domain}",
        "sizes": {s: len(r) for s, r in recs.items()},
        "train_dev_id": len(id_overlap("train", "dev")),
        "train_test_id": len(id_overlap("train", "test")),
        "dev_test_id": len(id_overlap("dev", "test")),
        "train_dev_text": len(text_overlap("train", "dev")),
        "train_test_text": len(text_overlap("train", "test")),
        "train_test_text_examples": [
            {"text": t[:70], "train_id": txt["train"][t], "test_id": txt["test"][t]}
            for t in list(text_overlap("train", "test"))[:3]
        ],
        "train_dev_text_examples": [
            {"text": t[:70], "train_id": txt["train"][t], "dev_id": txt["dev"][t]}
            for t in list(text_overlap("train", "dev"))[:3]
        ],
    }


def audit_trial() -> list[dict]:
    """What is the trial set a subset of? Its IDs look like rest16_quad_dev_*."""
    trial_dir = DATA / "trial"
    if not trial_dir.exists():
        return []
    results = []
    for path in sorted(trial_dir.glob("*.jsonl")):
        trial = load(path)
        trial_ids = {r["ID"] for r in trial}
        trial_text = {norm(r.get("Text", "")) for r in trial}
        where = []
        for lang, domain in CORPORA:
            for split, p in st1_paths(lang, domain).items():
                other = load(p)
                other_ids = {r["ID"] for r in other}
                other_text = {norm(r.get("Text", "")) for r in other}
                id_hits = len(trial_ids & other_ids)
                text_hits = len(trial_text & other_text)
                if id_hits or text_hits:
                    where.append({"corpus": f"{lang}_{domain}", "split": split,
                                  "id_hits": id_hits, "text_hits": text_hits})
        results.append({"file": path.name, "n": len(trial), "appears_in": where})
    return results


def main() -> int:
    rows = [audit_corpus(lang, dom) for lang, dom in CORPORA]

    print("=== train / dev / test overlap ===")
    print(f"{'corpus':<18}{'train':>7}{'dev':>6}{'test':>6}"
          f"{'td_id':>7}{'tt_id':>7}{'dt_id':>7}{'td_txt':>8}{'tt_txt':>8}")
    print("-" * 76)
    for r in rows:
        print(f"{r['corpus']:<18}{r['sizes']['train']:>7}{r['sizes']['dev']:>6}"
              f"{r['sizes']['test']:>6}{r['train_dev_id']:>7}{r['train_test_id']:>7}"
              f"{r['dev_test_id']:>7}{r['train_dev_text']:>8}{r['train_test_text']:>8}")
    print("td/tt/dt = train-dev / train-test / dev-test intersections")
    print("_id = by ID, _txt = by normalised Text\n")

    for r in rows:
        if r["train_test_text_examples"]:
            print(f"  !! {r['corpus']} train/test text collisions:")
            for e in r["train_test_text_examples"]:
                print(f"     train {e['train_id']} == test {e['test_id']}: {e['text']}")

    print("\n=== trial set provenance ===")
    for t in audit_trial():
        print(f"{t['file']}  n={t['n']}")
        if not t["appears_in"]:
            print("   appears in NO train/dev/test file")
        for w in t["appears_in"]:
            print(f"   -> {w['corpus']:<18} {w['split']:<6} "
                  f"id_hits={w['id_hits']} text_hits={w['text_hits']}")

    # The dataset terms forbid redistribution, so the committed report carries
    # counts and IDs only. Colliding sentence text is printed to the console for
    # debugging but never written to disk.
    def strip_text(rows: list[dict]) -> list[dict]:
        return [
            {k: ([{"text": "<redacted>", **{kk: vv for kk, vv in e.items() if kk != "text"}}
                 for e in v] if k.endswith("_examples") else v)
             for k, v in r.items()}
            for r in rows
        ]

    out = ROOT / "reports" / "leakage_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"corpora": strip_text(rows), "trial": audit_trial()}, indent=2) + "\n"
    )
    print(f"\nwrote {out} (sentence text redacted)")

    blocking = [r["corpus"] for r in rows if r["train_test_text"] or r["train_dev_text"]]
    print("\nVERDICT: " + ("clean — train shares no Text with dev or test"
                           if not blocking
                           else f"OVERLAP in {blocking} — example selection must exclude these"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
