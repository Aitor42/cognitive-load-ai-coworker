"""Project the end-of-day load from partial-day observations.

A morning analysis scores what has *already happened*; this module projects the
rest of the workday so LoadGuard can re-organize at midday. The only assumption
is documented and conservative: the remaining day continues at the same
per-hour rates and ratios already observed (or at explicitly supplied
remaining-day features). The result is always labelled a projection — never a
measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .availability import find_reassignment_alerts
from .guardian import guard_plan
from .llm import HeuristicModel
from .models import (
    HIGH,
    OVERLOAD,
    Event,
    FeatureSet,
    LoadReport,
    Plan,
    ReassignmentAlert,
    Task,
    Worker,
)
from .recommender import build_plan
from .scoring import score
from .signals import compute_features

# Projected levels that trigger a midday re-organization.
REPLAN_LEVELS = (HIGH, OVERLOAD)


@dataclass
class DayProjection:
    """Blended observed + remaining-day features and the projected load report."""

    elapsed_minutes: float
    remaining_minutes: float
    observed: FeatureSet
    projected_remaining: FeatureSet
    full_day: FeatureSet
    load_report: LoadReport


@dataclass
class MiddayReview:
    """Result of the midday re-score and (optional) re-organization."""

    observed_score: float
    observed_level: str
    projected_score: float
    projected_level: str
    remaining_minutes: float
    reorganized: bool
    plan: Plan | None = None
    reassignment_alerts: list[ReassignmentAlert] = field(default_factory=list)
    rationale: str = ""


def project_end_of_day(
    observed: FeatureSet,
    elapsed_minutes: float,
    total_minutes: float,
    remaining: FeatureSet | None = None,
) -> DayProjection:
    """Blend observed partial-day features with an assumed rest-of-day.

    Each feature is a time-weighted average of the observed and remaining-day
    values; when ``remaining`` is omitted, the remaining day is assumed to
    continue at the observed rate (the documented, conservative assumption).
    """
    remaining = remaining if remaining is not None else observed
    elapsed = max(elapsed_minutes, 0.0)
    total = max(total_minutes, elapsed)
    rem_minutes = max(total - elapsed, 0.0)

    def blend(obs: float, rem: float) -> float:
        if total <= 0.0:
            return obs
        return (obs * elapsed + rem * rem_minutes) / total

    full_day = FeatureSet(
        context_switches_per_hour=blend(
            observed.context_switches_per_hour, remaining.context_switches_per_hour
        ),
        meeting_ratio=blend(observed.meeting_ratio, remaining.meeting_ratio),
        notification_rate=blend(observed.notification_rate, remaining.notification_rate),
        focus_ratio=blend(observed.focus_ratio, remaining.focus_ratio),
        multitasking_index=blend(observed.multitasking_index, remaining.multitasking_index),
    )
    return DayProjection(
        elapsed_minutes=elapsed,
        remaining_minutes=rem_minutes,
        observed=observed,
        projected_remaining=remaining,
        full_day=full_day,
        load_report=score(full_day),
    )


def run_midday_review(
    events_so_far: list[Event],
    tasks: list[Task],
    elapsed_minutes: float,
    total_minutes: float = 480.0,
    workers: list[Worker] | None = None,
    now: float | None = None,
) -> MiddayReview:
    """Re-score the day so far, project the remainder, and re-organize if needed.

    The plan is rebuilt (deterministically, then guarded) only when the
    projected end-of-day load is ``high`` or ``overload``.
    """
    observed = compute_features(events_so_far, window_minutes=elapsed_minutes)
    observed_report = score(observed)
    projection = project_end_of_day(observed, elapsed_minutes, total_minutes)
    reorganized = projection.load_report.level in REPLAN_LEVELS

    plan: Plan | None = None
    if reorganized:
        plan = build_plan(tasks, projection.load_report)
        plan.note = HeuristicModel().generate_note(projection.load_report, plan, tasks)
        plan, _ = guard_plan(plan, tasks)

    alerts = find_reassignment_alerts(tasks, workers or [], now)
    if reorganized:
        rationale = (
            f"Projected end-of-day load is {projection.load_report.level}; "
            "re-organizing to protect attention."
        )
    else:
        rationale = "Projected load is manageable; the morning plan stands."
    return MiddayReview(
        observed_score=observed_report.score,
        observed_level=observed_report.level,
        projected_score=projection.load_report.score,
        projected_level=projection.load_report.level,
        remaining_minutes=projection.remaining_minutes,
        reorganized=reorganized,
        plan=plan,
        reassignment_alerts=alerts,
        rationale=rationale,
    )
