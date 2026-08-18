"""Impact Estimator.

Projects the Cognitive Load Score *after* the plan is followed, so the project's
value is measured rather than asserted. Every assumption is conservative and
documented; the estimator deliberately never claims to eliminate load, only to
reduce the interruption-driven portion of it.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import FeatureSet, Plan
from .scoring import score

# Assumptions behind the projection (documented in docs/architecture.md).
NOTIFICATION_REDUCTION = 0.5  # batching halves inbound notifications
FOCUS_GAIN_PER_BLOCK = 0.10  # each focus block adds ~10% focus share
MULTITASK_REDUCTION = 0.30  # breaks reduce multitasking by 30%
CONTEXT_REDUCTION_PER_DELEGATION = 0.10  # each delegated task cuts switches 10%


@dataclass
class ImpactResult:
    before_score: float
    before_level: str
    after_score: float
    after_level: str
    delta: float  # positive = load reduced
    assumptions: dict[str, str]


def estimate_impact(features: FeatureSet, plan: Plan) -> ImpactResult:
    """Estimate the before/after Cognitive Load Score given a plan."""
    projected = FeatureSet(
        context_switches_per_hour=features.context_switches_per_hour,
        meeting_ratio=features.meeting_ratio,
        notification_rate=features.notification_rate,
        focus_ratio=features.focus_ratio,
        multitasking_index=features.multitasking_index,
    )

    if any(i.action == "batch" for i in plan.items):
        projected.notification_rate *= NOTIFICATION_REDUCTION

    focus_blocks = [i for i in plan.items if i.action == "focus_block"]
    for _ in focus_blocks:
        projected.focus_ratio = min(projected.focus_ratio + FOCUS_GAIN_PER_BLOCK, 1.0)

    if any(i.action == "break" for i in plan.items):
        projected.multitasking_index *= 1.0 - MULTITASK_REDUCTION

    delegated = [i for i in plan.items if i.action == "delegate"]
    for _ in delegated:
        projected.context_switches_per_hour *= 1.0 - CONTEXT_REDUCTION_PER_DELEGATION

    before = score(features)
    after = score(projected)

    return ImpactResult(
        before_score=before.score,
        before_level=before.level,
        after_score=after.score,
        after_level=after.level,
        delta=round(before.score - after.score, 1),
        assumptions={
            "batch": f"notification rate x{NOTIFICATION_REDUCTION}",
            "focus_block": f"+{FOCUS_GAIN_PER_BLOCK} focus share each",
            "break": f"multitasking x{1 - MULTITASK_REDUCTION}",
            "delegate": f"context switches x{1 - CONTEXT_REDUCTION_PER_DELEGATION} each",
        },
    )
