"""Explainable Cognitive Load Score.

The score is a weighted, normalized combination of behavioral proxies. Each
proxy is normalized to 0..1 against a transparent threshold, then combined with
fixed, documented weights. This makes every recommendation traceable back to the
underlying signals.
"""

from __future__ import annotations

from .models import HIGH, LOW, MODERATE, OVERLOAD, FeatureSet, LoadReport

# Weights (sum to 1.0). Chosen so that interruption frequency (context switches,
# notifications) dominates, since it is the strongest behavioral proxy for
# cognitive load.
WEIGHTS = {
    "context_switches_per_hour": 0.30,
    "meeting_ratio": 0.20,
    "notification_rate": 0.20,
    "focus_ratio": 0.15,  # inverted: less focus -> more load
    "multitasking_index": 0.15,
}

# Normalization thresholds: values at/above these map to a contribution of 1.0.
MAX_CONTEXT_SWITCHES_PER_HOUR = 12.0
MAX_NOTIFICATIONS_PER_HOUR = 30.0

# Level boundaries for the 0..100 score.
BOUNDARIES = ((25.0, LOW), (50.0, MODERATE), (75.0, HIGH), (float("inf"), OVERLOAD))


def _normalize(value: float, maximum: float) -> float:
    if maximum <= 0.0:
        return 0.0
    return max(0.0, min(value / maximum, 1.0))


def _level(score: float) -> str:
    for upper, name in BOUNDARIES:
        if score < upper:
            return name
    return OVERLOAD


def _contributions(factors: dict[str, float]) -> dict[str, float]:
    """Normalize each factor to its 0..1 contribution to the score.

    This is the single source of truth for how each factor contributes, so the
    numeric score and the "Main drivers" explanation can never disagree.
    """
    return {
        "context_switches_per_hour": _normalize(
            factors.get("context_switches_per_hour", 0.0), MAX_CONTEXT_SWITCHES_PER_HOUR
        ),
        "meeting_ratio": max(0.0, min(factors.get("meeting_ratio", 0.0), 1.0)),
        "notification_rate": _normalize(
            factors.get("notification_rate", 0.0), MAX_NOTIFICATIONS_PER_HOUR
        ),
        # Focus time is protective: invert it so more focus lowers the score.
        "focus_ratio": max(0.0, min(1.0 - factors.get("focus_ratio", 0.0), 1.0)),
        "multitasking_index": max(0.0, min(factors.get("multitasking_index", 0.0), 1.0)),
    }


def _explanation(factors: dict[str, float]) -> str:
    factor_names = {
        "context_switches_per_hour": "context switches per hour",
        "meeting_ratio": "meeting density",
        "notification_rate": "notification rate",
        "focus_ratio": "focus time",
        "multitasking_index": "multitasking",
    }
    # Rank by weighted contribution (descending), consistent with score().
    weighted = {k: c * WEIGHTS[k] for k, c in _contributions(factors).items()}
    top = sorted(weighted.items(), key=lambda kv: kv[1], reverse=True)[:2]
    drivers = ", ".join(factor_names[k] for k, _ in top)
    return f"Main drivers: {drivers}."


def score(features: FeatureSet) -> LoadReport:
    """Compute a Cognitive Load Score from a FeatureSet."""
    factors = {
        "context_switches_per_hour": round(features.context_switches_per_hour, 2),
        "meeting_ratio": round(features.meeting_ratio, 2),
        "notification_rate": round(features.notification_rate, 2),
        "focus_ratio": round(features.focus_ratio, 2),
        "multitasking_index": round(features.multitasking_index, 2),
    }
    contributions = _contributions(factors)
    total = sum(contributions[k] * WEIGHTS[k] for k in WEIGHTS)
    value = round(total * 100.0, 1)
    level = _level(value)

    return LoadReport(
        score=value,
        level=level,
        factors=factors,
        explanation=_explanation(factors),
    )
