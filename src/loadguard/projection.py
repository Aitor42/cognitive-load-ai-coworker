"""Project the end-of-day load from partial-day observations.

A morning analysis scores what has *already happened*; this module projects the
rest of the workday so LoadGuard can re-organize at midday. The only assumption
is documented and conservative: the remaining day continues at the same
per-hour rates and ratios already observed (or at explicitly supplied
remaining-day features). The result is always labelled a projection — never a
measurement.

An additional **fatigue factor** amplifies stress-contributing features later
in the day, modelling the well-documented decline in cognitive resilience over
sustained work.
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

# Fatigue slope: stress features are amplified by up to this fraction by end of
# day.  A slope of 0.15 means 15 % higher perceived interruption load at the
# end vs. the beginning.
FATIGUE_SLOPE = 0.15


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


def _apply_fatigue(features: FeatureSet, elapsed: float, total: float) -> FeatureSet:
    """Amplify stress-contributing features by a time-of-day fatigue factor.

    The same interruption rate is more disruptive in the afternoon than in the
    morning.  Only rate-based / stress-contributing features are amplified;
    meeting_ratio and focus_ratio remain unchanged because they are direct
    observations, not perceived-load proxies.
    """
    if total <= 0.0 or elapsed <= 0.0:
        return features
    fatigue = 1.0 + FATIGUE_SLOPE * (elapsed / total)
    return FeatureSet(
        context_switches_per_hour=features.context_switches_per_hour * fatigue,
        meeting_ratio=features.meeting_ratio,
        notification_rate=features.notification_rate * fatigue,
        focus_ratio=features.focus_ratio,
        multitasking_index=min(features.multitasking_index * fatigue, 1.0),
        ai_notification_rate=features.ai_notification_rate,
    )


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

    A fatigue factor amplifies stress-contributing features based on how far
    into the workday the observation was taken.
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
        ai_notification_rate=blend(observed.ai_notification_rate, remaining.ai_notification_rate),
    )

    # Apply fatigue for scoring only; full_day retains the raw blended values.
    fatigued = _apply_fatigue(full_day, elapsed, total)

    return DayProjection(
        elapsed_minutes=elapsed,
        remaining_minutes=rem_minutes,
        observed=observed,
        projected_remaining=remaining,
        full_day=full_day,
        load_report=score(fatigued),
    )


def run_midday_review(
    events_so_far: list[Event],
    tasks: list[Task],
    elapsed_minutes: float,
    total_minutes: float = 480.0,
    workers: list[Worker] | None = None,
    now: float | None = None,
    completed_task_ids: set[str] | None = None,
) -> MiddayReview:
    """Re-score the day so far, project the remainder, and re-organize if needed.

    The plan is rebuilt (deterministically, then guarded) only when the
    projected end-of-day load is ``high`` or ``overload``.

    If *completed_task_ids* is provided, already-finished tasks are excluded
    from the re-plan so that midday re-organization respects morning progress.
    """
    observed = compute_features(events_so_far, window_minutes=elapsed_minutes)
    observed_report = score(observed)
    projection = project_end_of_day(observed, elapsed_minutes, total_minutes)
    reorganized = projection.load_report.level in REPLAN_LEVELS

    # Filter out completed tasks so the re-plan only covers remaining work.
    remaining_tasks = (
        [t for t in tasks if t.id not in completed_task_ids] if completed_task_ids else tasks
    )

    plan: Plan | None = None
    if reorganized:
        plan = build_plan(remaining_tasks, projection.load_report, workers=workers, now=now)
        plan.note = HeuristicModel().generate_note(projection.load_report, plan, remaining_tasks)
        plan, _ = guard_plan(plan, remaining_tasks)

    alerts = find_reassignment_alerts(remaining_tasks, workers or [], now)
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
