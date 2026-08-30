"""Impact Estimator.

Projects the Cognitive Load Score *after* the plan is followed, so the project's
value is measured rather than asserted. Every assumption is conservative and
documented; the estimator deliberately never claims to eliminate load, only to
reduce the interruption-driven portion of it.

Refinements:
- **Interaction and shielding reflection**: focus blocks shield deep work from
  context switching during meeting-dense days, reducing compound interaction load.
- **History-based calibration**: when past score history is available, reduction
  constants are scaled down if historical interventions under-delivered.
- **Position-aware focus blocks**: a focus block placed early in the plan
  protects more cognitive resources than one placed at the end.
- **Role/weights alignment**: calculates before/after with the active profile.
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
FOCUS_SHIELDING_REDUCTION = 0.10  # focus blocks shield from fragmentation

# Focus blocks placed at the end of the plan lose up to this fraction of their
# effectiveness compared to one placed at the very start.
POSITION_DECAY = 0.5


@dataclass
class ImpactResult:
    before_score: float
    before_level: str
    after_score: float
    after_level: str
    delta: float  # positive = load reduced
    assumptions: dict[str, str]


def _calibration_factor(history: list[float] | None) -> float:
    """Scale factor for impact reductions, learned from score history.

    A factor < 1.0 means past interventions under-delivered, so projections
    are made more conservative.  A factor > 1.0 is capped at 1.0 because
    LoadGuard deliberately never over-promises.
    """
    if not history or len(history) < 4:
        return 1.0
    recent = history[-6:]
    mid = len(recent) // 2
    first_half = sum(recent[:mid]) / mid
    second_half = sum(recent[mid:]) / (len(recent) - mid)
    if first_half <= 0:
        return 1.0
    improvement = max(first_half - second_half, 0.0) / first_half
    # Scale: 0% improvement -> factor 0.5; >=20% improvement -> factor 1.0.
    return min(0.5 + improvement * 2.5, 1.0)


def estimate_impact(
    features: FeatureSet,
    plan: Plan,
    history: list[float] | None = None,
    weights: dict[str, float] | None = None,
    role: str | None = None,
) -> ImpactResult:
    """Estimate the before/after Cognitive Load Score given a plan."""
    cal = _calibration_factor(history)

    projected = FeatureSet(
        context_switches_per_hour=features.context_switches_per_hour,
        meeting_ratio=features.meeting_ratio,
        notification_rate=features.notification_rate,
        focus_ratio=features.focus_ratio,
        multitasking_index=features.multitasking_index,
        ai_notification_rate=features.ai_notification_rate,
    )

    if any(i.action == "batch" for i in plan.items):
        effective = 1.0 - (1.0 - NOTIFICATION_REDUCTION) * cal
        projected.notification_rate *= effective

    # Position-aware focus blocks: earlier blocks protect more cognitive
    # resources than later ones, and shield against context-switching fragmentation.
    plan_len = max(len(plan.items) - 1, 1)
    has_focus_block = False
    for idx, item in enumerate(plan.items):
        if item.action == "focus_block":
            has_focus_block = True
            progress = idx / plan_len if len(plan.items) > 1 else 0.0
            position_factor = 1.0 - POSITION_DECAY * progress
            projected.focus_ratio = min(
                projected.focus_ratio + FOCUS_GAIN_PER_BLOCK * position_factor * cal,
                1.0,
            )

    if has_focus_block:
        # Protected focus shields against context switching fragmentation
        projected.context_switches_per_hour *= 1.0 - FOCUS_SHIELDING_REDUCTION * cal

    if any(i.action == "break" for i in plan.items):
        projected.multitasking_index *= 1.0 - MULTITASK_REDUCTION * cal

    delegated = [i for i in plan.items if i.action == "delegate"]
    for _ in delegated:
        projected.context_switches_per_hour *= 1.0 - CONTEXT_REDUCTION_PER_DELEGATION * cal

    before = score(features, weights=weights, role=role)
    after = score(projected, weights=weights, role=role)

    return ImpactResult(
        before_score=before.score,
        before_level=before.level,
        after_score=after.score,
        after_level=after.level,
        delta=round(before.score - after.score, 1),
        assumptions={
            "batch": f"notification rate x{NOTIFICATION_REDUCTION}",
            "focus_block": (
                f"+{FOCUS_GAIN_PER_BLOCK} focus share each "
                "(position-adjusted, fragmentation shielded)"
            ),
            "break": f"multitasking x{1 - MULTITASK_REDUCTION}",
            "delegate": f"context switches x{1 - CONTEXT_REDUCTION_PER_DELEGATION} each",
            "calibration": f"{cal:.2f} (from {len(history) if history else 0} history points)",
        },
    )
