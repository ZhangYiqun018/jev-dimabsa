#!/usr/bin/env python3
"""Settle how few-shot examples should be delivered to Jev.

Runs the same sentence through four requests:

  A. no examples            -- current behaviour, the control
  B. structured state       -- `labelled_examples` + `review_to_score` keys
  C. concatenated text      -- examples and target as one text block
  D. repeat of A            -- establishes the run-to-run noise floor

A variant counts as "moving the answer" only if it differs from A by more than
the A/D spread. Everything else is the model's own nondeterminism.

Implementation validation, not a consistency experiment.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jev.client import JevClient  # noqa: E402
from jev.rubrics import arousal_question, valence_question  # noqa: E402

TEST = ROOT / "vendor/DimABSA2026/task-dataset/track_a/subtask_1/eng/eng_restaurant_test_task1.jsonl"
TRAIN = ROOT / "vendor/DimABSA2026/task-dataset/track_a/subtask_1/eng/eng_restaurant_train_alltasks.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def pick_sentence(records: list[dict]) -> dict:
    """A sentence whose aspects disagree in polarity, so miscalibration is visible."""
    best, best_gap = None, 0.0
    for r in records:
        items = r.get("Aspect_VA") or r.get("Quadruplet") or []
        seen = {}
        for it in items:
            seen.setdefault(it["Aspect"].lower(), float(it["VA"].split("#")[0]))
        if len(seen) < 2:
            continue
        gap = max(seen.values()) - min(seen.values())
        if gap > best_gap:
            best, best_gap = r, gap
    return best


def examples_from(records: list[dict], n: int) -> list[dict]:
    """First n train records carrying at least one aspect. Frozen, in file order."""
    out = []
    for r in records:
        items = r.get("Aspect_VA") or r.get("Quadruplet") or []
        if not items:
            continue
        out.append(
            {
                "review": r["Text"],
                "aspect": items[0]["Aspect"],
                "valence": float(items[0]["VA"].split("#")[0]),
                "arousal": float(items[0]["VA"].split("#")[1]),
            }
        )
        if len(out) == n:
            break
    return out


def render_text_block(examples: list[dict], target: str) -> str:
    lines = ["Below are already-scored reference examples.", ""]
    for i, ex in enumerate(examples, 1):
        lines += [
            f"Example {i}:",
            f"Review: {ex['review']}",
            f"Aspect: {ex['aspect']}",
            f"Valence: {ex['valence']:.2f}",
            f"Arousal: {ex['arousal']:.2f}",
            "",
        ]
    lines += ["Below is the review to score.", "", f"Review: {target}"]
    return "\n".join(lines)


def questions(aspects: list[str], structured: bool) -> dict:
    qs = {}
    for i, aspect in enumerate(aspects):
        for key, builder in (("v", valence_question), ("a", arousal_question)):
            q = builder(aspect)
            if structured:
                q["instructions"]["question"] = q["instructions"]["question"].replace(
                    f'"{aspect}"', f'"{aspect}" in `review_to_score`'
                )
                q["instructions"]["focus"] = (
                    "`labelled_examples` are already-scored pairs, for calibration only "
                    "-- do not score them. " + q["instructions"]["focus"]
                )
            qs[f"{key}{i}"] = q
    return qs


def call(client, label, state, aspects, structured):
    r = client.ask(state, questions(aspects, structured))
    row = {}
    for i, aspect in enumerate(aspects):
        row[aspect] = (
            r.answers[f"v{i}"].score + 1.0,
            r.answers[f"a{i}"].score + 1.0,
        )
    print(f"  {label:<26} " + "   ".join(
        f"{a}: {v:5.2f}#{y:5.2f}" for a, (v, y) in row.items()))
    print(f"  {'':<26} in={r.usage['input_tokens']}")
    return row


def main() -> int:
    test = load_jsonl(TEST)
    train = load_jsonl(TRAIN)

    rec = pick_sentence(test)
    items = rec.get("Aspect_VA") or rec.get("Quadruplet") or []
    seen = {}
    for it in items:
        seen.setdefault(it["Aspect"].lower(), it)
    aspects = [it["Aspect"] for it in seen.values()]
    gold = {
        it["Aspect"]: tuple(float(x) for x in it["VA"].split("#"))
        for it in seen.values()
    }

    print(f"sentence : {rec['Text']}")
    print(f"aspects  : {aspects}")
    print(f"gold     : " + "   ".join(f"{a}: {v:.2f}#{y:.2f}" for a, (v, y) in gold.items()))
    print(f"train    : {TRAIN.name}  ({len(train)} records)")
    print()

    ex = examples_from(train, 3)
    print(f"frozen {len(ex)} calibration examples:")
    for e in ex:
        print(f"  - [{e['aspect']}] {e['valence']:.2f}#{e['arousal']:.2f}  {e['review'][:64]}")
    print()

    client = JevClient()
    print("=" * 92)

    print("\nA. no examples (control)")
    a1 = call(client, "A1", rec["Text"], aspects, structured=False)

    print("\nB. structured state")
    b = call(
        client,
        "B",
        {"labelled_examples": ex, "review_to_score": rec["Text"]},
        aspects,
        structured=True,
    )

    print("\nC. concatenated text block")
    c = call(client, "C", render_text_block(ex, rec["Text"]), aspects, structured=False)

    print("\nD. no examples again (noise floor)")
    a2 = call(client, "D", rec["Text"], aspects, structured=False)

    print("\n" + "=" * 92)
    print("delta vs control (mean over aspects and both dimensions):")
    def mean_delta(x):
        return sum(
            abs(x[a][k] - a1[a][k]) for a in aspects for k in (0, 1)
        ) / (2 * len(aspects))
    noise = mean_delta(a2)
    print(f"  D repeat of A  : {noise:.3f}   <- noise floor")
    for label, x in (("B structured", b), ("C text block", c)):
        d = mean_delta(x)
        verdict = "MOVED" if d > max(noise * 1.5, 0.02) else "within noise"
        print(f"  {label:<15}: {d:.3f}   {verdict}")
    print()
    for label, x in (("B", b), ("C", c)):
        print(f"  {label} vs gold: " + "  ".join(
            f"{a} dV{x[a][0]-gold[a][0]:+.2f} dA{x[a][1]-gold[a][1]:+.2f}" for a in aspects))
    return 0


if __name__ == "__main__":
    sys.exit(main())
