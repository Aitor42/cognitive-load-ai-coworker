"""Explainable Cognitive Load Score.

The score is a weighted, normalized combination of behavioral proxies.  Each
proxy is normalized to 0..1 via a smooth **sigmoid (Hill function)** that
avoids the hard saturation of a linear clamp, then combined with documented
weights plus an **interaction term** that captures compounding stressors (e.g.
interruptions during dense meetings).

Role profiles and custom weights are supported, enabling role-specific
cognitive load calibration (e.g. deep-writing researchers vs. coordination
heavy managers).
"""

from __future__ import annotations

from typing import Any

from .models import HIGH, LOW, MODERATE, OVERLOAD, FeatureSet, LoadReport

# Default weights for individual factors (sum to 1.0). Interruption frequency
# (context switches, notifications) dominates by default.
DEFAULT_WEIGHTS = {
    "context_switches_per_hour": 0.30,
    "meeting_ratio": 0.20,
    "notification_rate": 0.20,
    "focus_ratio": 0.15,  # inverted: less focus -> more load
    "multitasking_index": 0.15,
}

# Alias for backwards compatibility with tests and callers
WEIGHTS = DEFAULT_WEIGHTS

# Role-specific profiles tailoring factor sensitivity to different work styles.
ROLE_PROFILES: dict[str, dict[str, float]] = {
    "default": DEFAULT_WEIGHTS,
    "developer": {
        "context_switches_per_hour": 0.35,
        "meeting_ratio": 0.15,
        "notification_rate": 0.20,
        "focus_ratio": 0.20,
        "multitasking_index": 0.10,
    },
    "researcher": {
        "context_switches_per_hour": 0.20,
        "meeting_ratio": 0.15,
        "notification_rate": 0.15,
        "focus_ratio": 0.35,  # high focus sensitivity for deep writing/synthesis
        "multitasking_index": 0.15,
    },
    "manager": {
        "context_switches_per_hour": 0.25,
        "meeting_ratio": 0.25,
        "notification_rate": 0.25,
        "focus_ratio": 0.10,
        "multitasking_index": 0.15,
    },
    "support": {
        "context_switches_per_hour": 0.30,
        "meeting_ratio": 0.15,
        "notification_rate": 0.15,  # Higher tolerance for inbound notification volume
        "focus_ratio": 0.15,
        "multitasking_index": 0.25,  # High sensitivity to rapid ticket/context thrashing
    },
}

# Interaction weight: when meetings and interruptions overlap, the combined
# cognitive cost is super-additive.  Applied on top of the base weighted sum.
INTERACTION_WEIGHT = 0.10

# Sigmoid midpoints (Hill function): the function returns 0.5 at the midpoint
# and approaches 1.0 asymptotically, giving diminishing returns instead of
# the hard saturation of a linear clamp.
MIDPOINT_CONTEXT_SWITCHES = 6.0  # 6 switches/h -> 0.5 contribution
MIDPOINT_NOTIFICATIONS = 10.0  # 10 notifs/h -> 0.5 contribution

# Level boundaries for the 0..100 score.
BOUNDARIES = ((25.0, LOW), (50.0, MODERATE), (75.0, HIGH), (float("inf"), OVERLOAD))


def _normalize(value: float, midpoint: float) -> float:
    """Smooth sigmoid normalization (Hill function).

    Returns 0.5 at the midpoint and approaches 1.0 asymptotically, avoiding
    the hard saturation of a linear clamp.
    """
    if midpoint <= 0.0 or value <= 0.0:
        return 0.0
    return value / (value + midpoint)


def _level(score: float) -> str:
    for upper, name in BOUNDARIES:
        if score < upper:
            return name
    return OVERLOAD


def _contributions(
    factors: dict[str, float],
    baseline: Any = None,
) -> dict[str, float]:
    """Normalize each factor to its 0..1 contribution to the score.

    Supports optional adaptive midpoints derived from the user's personal
    baseline mean, so high-communication workers are evaluated relative to their
    personal baseline rather than a static global threshold.
    """
    scale = 1.0
    if (
        baseline is not None
        and getattr(baseline, "n", 0) >= 2
        and getattr(baseline, "mean", 0.0) > 0
    ):
        scale = max(0.75, min(baseline.mean / 40.0, 1.5))

    switches_mid = MIDPOINT_CONTEXT_SWITCHES * scale
    notifs_mid = MIDPOINT_NOTIFICATIONS * scale

    return {
        "context_switches_per_hour": _normalize(
            factors.get("context_switches_per_hour", 0.0), switches_mid
        ),
        "meeting_ratio": max(0.0, min(factors.get("meeting_ratio", 0.0), 1.0)),
        "notification_rate": _normalize(factors.get("notification_rate", 0.0), notifs_mid),
        # Focus time is protective: invert it so more focus lowers the score.
        "focus_ratio": max(0.0, min(1.0 - factors.get("focus_ratio", 0.0), 1.0)),
        "multitasking_index": max(0.0, min(factors.get("multitasking_index", 0.0), 1.0)),
    }


def _interaction_bonus(contributions: dict[str, float]) -> float:
    """Compounding penalty when meetings and interruptions overlap.

    When meeting density is high AND interruption rate (context switches or
    notifications) is also high, the cognitive cost is worse than the sum of
    the parts.  The bonus is the product of meeting contribution and the
    highest interruption contribution, so it is only significant when *both*
    stressors are elevated simultaneously.
    """
    meeting = contributions.get("meeting_ratio", 0.0)
    interruption = max(
        contributions.get("context_switches_per_hour", 0.0),
        contributions.get("notification_rate", 0.0),
    )
    return meeting * interruption


def _resolve_weights(
    weights: dict[str, float] | None = None,
    role: str | None = None,
) -> dict[str, float]:
    """Resolve and normalize factor weights from custom dict or role profile."""
    if weights is not None:
        total = sum(weights.values())
        if total > 0:
            return {k: weights.get(k, 0.0) / total for k in DEFAULT_WEIGHTS}
    if role is not None and role in ROLE_PROFILES:
        return ROLE_PROFILES[role]
    return DEFAULT_WEIGHTS


def _explanation(
    factors: dict[str, float],
    weights: dict[str, float] | None = None,
    ai_interruption_ratio: float | None = None,
    baseline: Any = None,
) -> str:
    factor_names = {
        "context_switches_per_hour": "context switches per hour",
        "meeting_ratio": "meeting density",
        "notification_rate": "notification rate",
        "focus_ratio": "focus time",
        "multitasking_index": "multitasking",
    }
    active = weights if weights is not None else DEFAULT_WEIGHTS
    contribs = _contributions(factors, baseline=baseline)
    # Rank by weighted contribution (descending), consistent with score().
    weighted = {k: c * active.get(k, 0.0) for k, c in contribs.items()}
    top = sorted(weighted.items(), key=lambda kv: kv[1], reverse=True)[:2]
    drivers = ", ".join(factor_names[k] for k, _ in top)
    text = f"Main drivers: {drivers}."
    # The interaction term can dominate the score without showing up in the
    # top drivers, surface it whenever it is meaningful.
    interaction = _interaction_bonus(contribs) * INTERACTION_WEIGHT
    if interaction >= 0.02:
        text += " Compounding meetings and interruptions amplify the load."
    if ai_interruption_ratio is not None and ai_interruption_ratio >= 0.40:
        text += (
            f" {int(ai_interruption_ratio * 100)}% of interruptions originated from AI assistants."
        )
    return text


def score(
    features: FeatureSet,
    weights: dict[str, float] | None = None,
    role: str | None = None,
    baseline: Any = None,
) -> LoadReport:
    """Compute a Cognitive Load Score from a FeatureSet.

    Supports custom weights, role-specific profiles (e.g. "developer",
    "researcher", "manager"), and baseline-adaptive midpoints.
    """
    active_weights = _resolve_weights(weights, role)
    factors = {
        "context_switches_per_hour": round(features.context_switches_per_hour, 2),
        "meeting_ratio": round(features.meeting_ratio, 2),
        "notification_rate": round(features.notification_rate, 2),
        "focus_ratio": round(features.focus_ratio, 2),
        "multitasking_index": round(features.multitasking_index, 2),
    }
    contributions = _contributions(factors, baseline=baseline)
    base = sum(contributions[k] * active_weights.get(k, 0.0) for k in active_weights)
    interaction = _interaction_bonus(contributions) * INTERACTION_WEIGHT
    total = min(base + interaction, 1.0)
    value = round(total * 100.0, 1)
    level = _level(value)

    # Share of interruptions that came from AI tools (None when no notifications).
    ai_ratio: float | None = None
    if features.notification_rate > 0.0:
        ai_ratio = round(min(features.ai_notification_rate / features.notification_rate, 1.0), 2)

    return LoadReport(
        score=value,
        level=level,
        factors=factors,
        explanation=_explanation(factors, active_weights, ai_ratio, baseline=baseline),
        ai_interruption_ratio=ai_ratio,
    )
