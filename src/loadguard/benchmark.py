"""Benchmark: measurable metrics for the LoadGuard pipeline.

Produces objective, reportable numbers (before/after Cognitive Load Score,
reduction, plan composition, and estimated interruptions eliminated) so the
project's value is measured rather than asserted. Comparable in spirit to the
F1/precision/recall benchmarks the strongest challenge entries report.

``run_pilot_evaluation`` implements a three-phase measurement:

- **baseline** — the day without LoadGuard;
- **projected** — the plan LoadGuard proposes (Impact Estimator);
- **observed** — signals recorded *after* the plan was applied.

Observed metrics are only reported when outcome events are supplied; otherwise
the result is clearly labelled as a projection, never as evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .llm import ChatModel
from .models import (
    CONTEXT_SWITCH,
    DONE,
    FOCUS_BLOCK,
    MEETING,
    NOTIFICATION,
    Event,
    FeatureSet,
    Task,
)
from .signals import _window_minutes, compute_features
from .scoring import score
from .workflow import run_workflow


@dataclass
class BenchmarkResult:
    window_minutes: float
    n_events: int
    signal_counts: dict[str, int]
    before_score: float
    before_level: str
    after_score: float
    after_level: str
    reduction_points: float
    reduction_pct: float
    plan_counts: dict[str, int]
    interruptions_eliminated: int
    metrics: dict[str, float] = field(default_factory=dict)


def run_benchmark(
    events: list[Event],
    tasks: list[Task],
    model: ChatModel | None = None,
    window_minutes: float | None = None,
) -> BenchmarkResult:
    """Run the pipeline and summarize measurable outcomes."""
    features = compute_features(events, window_minutes)
    before = score(features)
    result = run_workflow(events, tasks, model, window_minutes)
    after = result.impact.after_score
    delta = round(before.score - after, 1)

    signal_counts = {k: 0 for k in (CONTEXT_SWITCH, MEETING, NOTIFICATION, FOCUS_BLOCK)}
    for e in events:
        if e.kind in signal_counts:
            signal_counts[e.kind] += 1

    plan_counts: dict[str, int] = {}
    for item in result.plan.items:
        plan_counts[item.action] = plan_counts.get(item.action, 0) + 1

    window_hours = features_window(events, window_minutes) / 60.0
    eliminated = _estimate_eliminated(features, plan_counts, window_hours)

    return BenchmarkResult(
        window_minutes=round(features_window(events, window_minutes), 1),
        n_events=len(events),
        signal_counts=signal_counts,
        before_score=before.score,
        before_level=before.level,
        after_score=after,
        after_level=result.impact.after_level,
        reduction_points=delta,
        reduction_pct=round((delta / before.score * 100) if before.score else 0.0, 1),
        plan_counts=plan_counts,
        interruptions_eliminated=eliminated,
        metrics={
            "context_switches_per_hour": round(features.context_switches_per_hour, 2),
            "notification_rate": round(features.notification_rate, 2),
            "meeting_ratio": round(features.meeting_ratio, 2),
            "focus_ratio": round(features.focus_ratio, 2),
            "multitasking_index": round(features.multitasking_index, 2),
        },
    )


def features_window(events: list[Event], window_minutes: float | None) -> float:
    if window_minutes is not None and window_minutes > 0.0:
        return window_minutes
    return _window_minutes(events)


def _estimate_eliminated(
    features: FeatureSet, plan_counts: dict[str, int], window_hours: float
) -> int:
    """Estimate the number of interruptions removed by following the plan."""
    eliminated = 0.0
    if plan_counts.get("batch", 0) > 0:
        # Batching halves inbound notifications.
        eliminated += features.notification_rate * 0.5 * window_hours
    delegated = plan_counts.get("delegate", 0)
    if delegated > 0:
        # Each delegated task reduces context switches by 10% (geometric).
        switches_total = features.context_switches_per_hour * window_hours
        eliminated += switches_total * (1.0 - (0.9**delegated))
    return int(round(eliminated))


# ---------------------------------------------------------------------------
# Pilot evaluation: baseline / projected / observed
# ---------------------------------------------------------------------------


@dataclass
class PhaseMetrics:
    """One phase of the pilot evaluation."""

    label: str  # baseline | projected | observed
    score: float
    level: str
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Three-phase evaluation with honest observed/projected labelling."""

    baseline: PhaseMetrics
    projected: PhaseMetrics
    observed: PhaseMetrics | None
    has_observed: bool
    notification_reduction_pct: float | None  # during focus blocks
    context_switch_reduction_pct: float | None
    focus_minutes_gained: float | None
    acceptance_rate: float | None
    tasks_completed: int
    priority_tasks_completed: int
    load_delta: float | None  # baseline - observed
    summary: str


def _focus_intervals(events: list[Event]) -> list[tuple[float, float]]:
    return [
        (e.timestamp, e.timestamp + e.duration_minutes * 60.0)
        for e in events
        if e.kind == FOCUS_BLOCK
    ]


def _notifications_in_focus(events: list[Event]) -> tuple[float, float]:
    """Return (focus_minutes, notifications landing inside focus blocks)."""
    intervals = _focus_intervals(events)
    focus_minutes = sum((end - start) / 60.0 for start, end in intervals)
    inside = sum(
        1
        for e in events
        if e.kind == NOTIFICATION and any(start <= e.timestamp <= end for start, end in intervals)
    )
    return focus_minutes, inside


def _per_hour(events: list[Event], kind: str) -> float | None:
    stamps = [e.timestamp for e in events if e.kind == kind]
    if not stamps:
        return 0.0
    span_hours = max((max(stamps) - min(stamps)) / 3600.0, 1.0)
    return len(stamps) / span_hours


def _pct_reduction(baseline: float | None, observed: float | None) -> float | None:
    if baseline is None or observed is None or baseline <= 0:
        return None
    return round((1.0 - observed / baseline) * 100.0, 1)


def run_pilot_evaluation(
    events: list[Event],
    tasks: list[Task],
    outcome_events: list[Event] | None = None,
    accepted_recommendations: float | None = None,
    model: ChatModel | None = None,
    window_minutes: float | None = None,
) -> EvaluationResult:
    """Evaluate a pilot: baseline vs. projected vs. observed (when available)."""
    features = compute_features(events, window_minutes)
    before = score(features)
    baseline = PhaseMetrics(
        label="baseline",
        score=before.score,
        level=before.level,
        metrics={
            "context_switches_per_hour": round(features.context_switches_per_hour, 2),
            "notification_rate": round(features.notification_rate, 2),
            "focus_ratio": round(features.focus_ratio, 2),
            "meeting_ratio": round(features.meeting_ratio, 2),
            "multitasking_index": round(features.multitasking_index, 2),
        },
    )

    result = run_workflow(events, tasks, model, window_minutes)
    projected = PhaseMetrics(
        label="projected",
        score=result.impact.after_score,
        level=result.impact.after_level,
        metrics={"delta": result.impact.delta},
    )

    observed: PhaseMetrics | None = None
    outcome = outcome_events
    has_observed = bool(outcome)
    if outcome:
        obs_features = compute_features(outcome, window_minutes)
        obs = score(obs_features)
        observed = PhaseMetrics(
            label="observed",
            score=obs.score,
            level=obs.level,
            metrics={
                "context_switches_per_hour": round(obs_features.context_switches_per_hour, 2),
                "notification_rate": round(obs_features.notification_rate, 2),
                "focus_ratio": round(obs_features.focus_ratio, 2),
                "meeting_ratio": round(obs_features.meeting_ratio, 2),
                "multitasking_index": round(obs_features.multitasking_index, 2),
            },
        )

    # Focus-window interruption metrics (baseline vs observed).
    b_focus_min, b_notif = _notifications_in_focus(events)
    o_focus_min, o_notif = (0.0, 0.0)
    if outcome:
        o_focus_min, o_notif = _notifications_in_focus(outcome)
    b_focus_rate = (b_notif / (b_focus_min / 60.0)) if b_focus_min > 0 else None
    o_focus_rate = (o_notif / (o_focus_min / 60.0)) if o_focus_min > 0 else None
    notification_reduction = _pct_reduction(b_focus_rate, o_focus_rate) if outcome else None

    context_reduction = (
        _pct_reduction(
            _per_hour(events, CONTEXT_SWITCH),
            _per_hour(outcome, CONTEXT_SWITCH),
        )
        if outcome
        else None
    )
    focus_gained = (o_focus_min - b_focus_min) if outcome else None

    done = [t for t in tasks if t.status == DONE]
    priority_done = [t for t in done if t.priority >= 4]
    load_delta = round(before.score - observed.score, 1) if observed else None

    if has_observed and observed is not None:
        summary = (
            f"Pilot: baseline {before.score:.0f} -> projected {result.impact.after_score:.0f} -> "
            f"observed {observed.score:.0f} ({load_delta:+.0f} points)."
        )
        if notification_reduction is not None:
            summary += f" Interruptions during focus blocks down {notification_reduction:.0f}%."
    else:
        summary = (
            "No observed outcome signals supplied — results are a reproducible "
            "projection, not real-world evidence."
        )

    return EvaluationResult(
        baseline=baseline,
        projected=projected,
        observed=observed,
        has_observed=has_observed,
        notification_reduction_pct=notification_reduction,
        context_switch_reduction_pct=context_reduction,
        focus_minutes_gained=focus_gained,
        acceptance_rate=accepted_recommendations,
        tasks_completed=len(done),
        priority_tasks_completed=len(priority_done),
        load_delta=load_delta,
        summary=summary,
    )
