"""In-context calibration examples for the DimABSA subtask-1 prompts.

Examples come only from the **train** split of the same corpus. Selection is a
frozen instrument-calibration step, not a tunable: the same records are used for
every request in a run, computed once, and never re-picked because a particular
prediction came out badly. Re-selecting against evaluation results would be
tuning on the evaluation set.

Leakage. `tools/leakage_audit.py` found that ID sets are disjoint across splits
but **Text is not**: `jpn_hotel` reuses 23 sentences between train and test,
`tat_restaurant` 2, and `eng_restaurant` 1 between train and dev. So selecting
"the first n train records" naively can put an evaluation sentence into the
prompt. Every candidate is therefore dropped if its normalised Text (or its ID)
also occurs in the dev or test split being evaluated.

The official protocol is *"the first k samples in the training set"*; the only
departure here is skipping records that collide with the evaluation split.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "vendor" / "DimABSA2026" / "task-dataset"


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def load_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def train_path(lang: str, domain: str) -> Path:
    """Train filenames are inconsistent: `_train_alltasks` for most corpora,
    `_train_task1` for the two finance ones."""
    base = DATA / "track_a" / "subtask_1" / lang
    stem = f"{lang}_{domain}"
    for name in (f"{stem}_train_alltasks.jsonl", f"{stem}_train_task1.jsonl"):
        if (base / name).exists():
            return base / name
    return base / f"{stem}_train_alltasks.jsonl"


def split_path(lang: str, domain: str, split: str) -> Path:
    return (DATA / "track_a" / "subtask_1" / lang
            / f"{lang}_{domain}_{split}_task1.jsonl")


@dataclass
class Example:
    source_id: str
    review: str
    aspect: str
    valence: float
    arousal: float

    def as_payload(self) -> dict:
        return {
            "review": self.review,
            "aspect": self.aspect,
            "valence": round(self.valence, 2),
            "arousal": round(self.arousal, 2),
        }


@dataclass
class ExampleSet:
    corpus: str
    examples: list[Example] = field(default_factory=list)
    n_requested: int = 0
    skipped_leak: list[str] = field(default_factory=list)

    @property
    def source_ids(self) -> list[str]:
        return [e.source_id for e in self.examples]

    def payload(self) -> list[dict]:
        return [e.as_payload() for e in self.examples]

    @property
    def ids_for_meta(self) -> dict:
        return {
            "examples": self.source_ids,
            "n_requested": self.n_requested,
            "skipped_as_leaking": self.skipped_leak,
        }


def _aspects_of(record: dict) -> list[dict]:
    return record.get("Aspect_VA") or record.get("Quadruplet") or record.get("Triplet") or []


def load_examples(
    lang: str,
    domain: str,
    n: int,
    exclude_splits: tuple[str, ...] = ("dev", "test"),
) -> ExampleSet:
    """First `n` train records that carry an aspect and do not leak an eval item.

    Each record contributes one example per aspect it contains, so a record with
    two aspects yields two examples.
    """
    corpus = f"{lang}_{domain}"
    result = ExampleSet(corpus=corpus, n_requested=n)
    if n <= 0:
        return result

    banned_ids: set[str] = set()
    banned_text: set[str] = set()
    for split in exclude_splits:
        for record in load_jsonl(split_path(lang, domain, split)):
            banned_ids.add(record["ID"])
            banned_text.add(normalise(record.get("Text", "")))

    for record in load_jsonl(train_path(lang, domain)):
        if len(result.examples) >= n:
            break
        aspects = _aspects_of(record)
        if not aspects:
            continue
        if record["ID"] in banned_ids or normalise(record.get("Text", "")) in banned_text:
            result.skipped_leak.append(record["ID"])
            continue
        for item in aspects[:1]:  # one example per record, matching "first k samples"
            valence, arousal = (float(x) for x in item["VA"].split("#"))
            result.examples.append(
                Example(
                    source_id=record["ID"],
                    review=record["Text"],
                    aspect=item["Aspect"],
                    valence=valence,
                    arousal=arousal,
                )
            )
    return result


def calibration_clause(n_examples: int) -> str:
    """The extra instruction text. Empty at n=0 so the request is unchanged."""
    if n_examples <= 0:
        return ""
    return (
        f" The {n_examples} entries in `labelled_examples` are already-scored "
        "(review, aspect) pairs; use them only to calibrate the 1-9 scale."
    )


def focus_prefix(n_examples: int) -> str:
    if n_examples <= 0:
        return ""
    return (
        "`labelled_examples` are for calibration only -- do not score them. "
        "Score only the aspect named in this question, as it appears in "
        "`review_to_score`. "
    )


def build_state(sentence: str, examples: ExampleSet):
    """The `state` payload.

    At n=0 this returns the bare sentence, so a zero-shot request stays
    byte-identical to the one that produced the recorded numbers.
    """
    if not examples.examples:
        return sentence
    return {
        "labelled_examples": examples.payload(),
        "review_to_score": sentence,
    }
