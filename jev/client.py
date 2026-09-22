"""Thin HTTP client for the TypeSafe System One (Jev) API.

No third-party dependencies. The API key is read from the environment, falling
back to the user's shell rc files, and is never logged or included in errors.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-1.13.0"

RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504, 529})


def _retry_after(exc: urllib.error.HTTPError) -> float | None:
    value = exc.headers.get("Retry-After") if exc.headers else None
    try:
        return float(value) if value else None
    except (TypeError, ValueError):
        return None

_KEY_NAMES = ("TYPESAFE_API_KEY", "TYPESAFE_KEY", "TYPESAFE_API_TOKEN", "TYPESAFE_TOKEN")
_RC_FILES = ("~/.zshenv", "~/.zprofile", "~/.zshrc", "~/.bashrc", "~/.profile")


class JevError(RuntimeError):
    pass


def load_api_key() -> str:
    """Environment first, then shell rc files.

    The rc fallback exists because ``.zshrc`` is only sourced by interactive
    shells, so the variable is invisible to anything run non-interactively.
    """
    for name in _KEY_NAMES:
        value = os.environ.get(name)
        if value:
            return value.strip().strip("'\"")

    pattern = re.compile(
        r"^\s*export\s+(" + "|".join(_KEY_NAMES) + r")\s*=\s*(.+?)\s*$", re.MULTILINE
    )
    for rc in _RC_FILES:
        path = Path(rc).expanduser()
        if not path.is_file():
            continue
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        for match in pattern.finditer(text):
            value = re.sub(r"\s+#.*$", "", match.group(2)).strip().strip("'\"")
            if value:
                return value

    raise JevError(
        "No TypeSafe API key found. Set TYPESAFE_API_KEY or export it in ~/.zshrc."
    )


@dataclass
class Answer:
    name: str
    type: str
    raw: dict[str, Any]

    @property
    def score(self) -> float:
        return float(self.raw["score"])

    @property
    def choice(self) -> str:
        return self.raw["choice"]

    @property
    def noul(self) -> float:
        return float(self.raw["noul"])

    @property
    def confidence(self) -> float | None:
        value = self.raw.get("confidence")
        return None if value is None else float(value)

    @property
    def probabilities(self) -> dict[str, float]:
        return {k: float(v) for k, v in (self.raw.get("probabilities") or {}).items()}


@dataclass
class Response:
    model: str
    answers: dict[str, Answer]
    usage: dict[str, int]
    attempts: int = 1


class JevClient:
    def __init__(self, model: str = DEFAULT_MODEL, timeout: int = 180) -> None:
        self._key = load_api_key()
        self.model = model
        self.timeout = timeout

    def ask(
        self,
        state: Any,
        questions: dict[str, dict],
        attempts: int = 7,
        backoff: float = 2.0,
    ) -> Response:
        """POST one batch of questions against a single state.

        Retries transient failures (rate limits, 5xx, connection errors) with
        exponential backoff. A long run is hundreds of sequential requests, so
        giving up on the first 429 would throw away the whole run.
        """
        payload = {"state": state, "model": self.model, "questions": questions}
        request = urllib.request.Request(
            ENDPOINT,
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {self._key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        last_error = "?"
        for attempt in range(attempts):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode())
                break
            except urllib.error.HTTPError as exc:  # keep the key out of the message
                detail = exc.read().decode(errors="replace")[:300]
                last_error = f"HTTP {exc.code}: {detail}"
                if exc.code not in RETRYABLE_STATUS or attempt == attempts - 1:
                    raise JevError(f"System One {last_error}") from None
                delay = _retry_after(exc) or backoff * (2**attempt)
            except (urllib.error.URLError, TimeoutError) as exc:
                reason = getattr(exc, "reason", str(exc))
                last_error = f"connection error: {reason}"
                if attempt == attempts - 1:
                    raise JevError(f"Could not reach System One: {reason}") from None
                delay = backoff * (2**attempt)
            time.sleep(min(delay, 60.0))
        else:  # pragma: no cover - loop always breaks or raises
            raise JevError(f"System One request failed: {last_error}")

        answers = {
            name: Answer(name=name, type=raw.get("type", "?"), raw=raw)
            for name, raw in body.get("answers", {}).items()
        }
        return Response(
            model=body.get("model", "?"),
            answers=answers,
            usage=body.get("usage", {}),
            attempts=attempt + 1,
        )


def score_to_va(score: float) -> float:
    """Map a 9-level Score answer (0..8) onto the official VA scale (1..9).

    The API returns ``sum(level_index * probability)``, so the value is already
    continuous; we shift it onto 1..9 and deliberately do not round.
    """
    return 1.0 + float(score)


def format_va(valence: float, arousal: float) -> str:
    """Official VA serialisation, e.g. ``"7.23#4.18"``."""
    return f"{valence:.2f}#{arousal:.2f}"
