"""Loading and writing DimABSA task data.

Two different kinds of de-duplication apply here, and conflating them is a bug:

* **Predictions** must be de-duplicated by aspect. The official scorer does
  ``{entry['Aspect']: entry for entry in pred_value}``, so at most one prediction
  per lowercased aspect survives anyway.
* **Gold** must NOT be de-duplicated. The scorer traverses every gold item, so
  dropping repeats silently reweights the metric. One aspect can legitimately
  carry several annotations (e.g. two opinions with different VA).

Other scorer behaviour worth restating:

* It lowercases ``Aspect``/``Opinion``/``Category`` before matching.
* For task 1 it calls ``exit()`` -- not "score 0" -- when a gold ID or a gold
  aspect is missing from the predictions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

# ---------------------------------------------------------------- reading


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _parse_va(value: str) -> tuple[float, float]:
    valence, arousal = value.split("#")
    return float(valence), float(arousal)


def _annotation_items(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Every aspect annotation in a record, from any of the task file shapes."""
    return record.get("Aspect_VA") or record.get("Quadruplet") or record.get("Triplet") or []


def aspects_for_inference(record: dict[str, Any]) -> list[str]:
    """Aspect surface strings to ask about, de-duplicated, order preserved.

    Deliberately does not touch ``VA``: the aspect list is the task input, and an
    unlabelled record should still run. Duplicates collapse because predictions
    are keyed on the lowercased aspect.
    """
    seen: set[str] = set()
    aspects: list[str] = []
    for item in _annotation_items(record):
        aspect = item["Aspect"]
        key = aspect.lower()
        if key in seen:
            continue
        seen.add(key)
        aspects.append(aspect)
    return aspects


def st1_gold(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build a subtask-1 gold file (``Aspect_VA``) from any task file.

    Preserves every annotation, including repeats of the same aspect, because
    the official scorer counts each gold item. Only used for files that do not
    already carry ``Aspect_VA`` (the trial files, which are quadruplet-only).
    """
    gold = []
    for record in records:
        entries = []
        for item in _annotation_items(record):
            valence, arousal = _parse_va(item["VA"])
            entries.append(
                {"Aspect": item["Aspect"], "VA": f"{valence:.2f}#{arousal:.2f}"}
            )
        gold.append({"ID": record["ID"], "Text": record["Text"], "Aspect_VA": entries})
    return gold


# ---------------------------------------------------------------- writing


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
