"""Personalized baseline and temporal trend.

The Cognitive Load Score is most useful relative to the *individual's own
history*: a score of 70 means something different for someone whose personal
average is 45 vs. 60. This module computes a personal baseline, the deviation
of the current score from it, the trend direction, and a confidence level based
on how many days of history are available.

Scores are stored as a plain JSONL file of numbers; deletion is supported so
users can wipe their history at any time (privacy first).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PersonalBaseline:
    """Average and spread of the user's own load history."""

    mean: float
    std: float
    n: int
    window: str = "personal history"


@dataclass
class TrendInfo:
    """How the current score compares to the user's personal baseline."""

    deviation_pct: float  # signed % vs. personal mean
    zscore: float
    direction: str  # rising | falling | stable
    confidence: str  # high | medium | low
    summary: str


def compute_baseline(scores: list[float]) -> PersonalBaseline | None:
    """Return a baseline from at least 2 samples, else None."""
    if not scores:
        return None
    clean = []
    for s in scores:
        try:
            val = float(s)
            if not math.isnan(val) and not math.isinf(val) and 0.0 <= val <= 100.0:
                clean.append(val)
        except (ValueError, TypeError):
            continue
    if len(clean) < 2:
        return None
    mean = sum(clean) / len(clean)
    variance = sum((s - mean) ** 2 for s in clean) / len(clean)
    return PersonalBaseline(mean=round(mean, 1), std=round(math.sqrt(variance), 1), n=len(clean))


def trend(score: float, baseline: PersonalBaseline | None) -> TrendInfo | None:
    """Compare a score against the personal baseline."""
    if baseline is None:
        return None
    deviation = score - baseline.mean
    deviation_pct = (deviation / baseline.mean * 100.0) if baseline.mean else 0.0
    zscore = deviation / baseline.std if baseline.std > 0 else 0.0

    if baseline.std > 0:
        direction = "rising" if zscore >= 1.0 else "falling" if zscore <= -1.0 else "stable"
    else:
        direction = "rising" if deviation > 0 else "falling" if deviation < 0 else "stable"

    if baseline.n >= 5:
        confidence = "high"
    elif baseline.n >= 3:
        confidence = "medium"
    else:
        confidence = "low"

    summary = (
        f"Your current load is {score:.0f}/100, {deviation_pct:+.0f}% vs your personal "
        f"baseline ({baseline.mean:.0f}±{baseline.std:.0f}, {baseline.n} day(s) of history). "
        f"Trend: {direction} ({confidence} confidence)."
    )
    return TrendInfo(
        deviation_pct=round(deviation_pct, 1),
        zscore=round(zscore, 2),
        direction=direction,
        confidence=confidence,
        summary=summary,
    )


def load_history(path: str | Path) -> list[float]:
    """Load daily scores from a JSONL file (one float per line)."""
    scores: list[float] = []
    path = Path(path)
    if not path.exists():
        return scores
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            scores.append(float(json.loads(line)))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return scores


def append_score(path: str | Path, score: float) -> None:
    """Append a daily score to the history file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(f"{json.dumps(score)}\n")


def clear_history(path: str | Path) -> int:
    """Delete the history file; returns how many scores were removed."""
    path = Path(path)
    if not path.exists():
        return 0
    count = len(load_history(path))
    path.unlink()
    return count
