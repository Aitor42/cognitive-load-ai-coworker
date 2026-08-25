"""Daily cycle: morning analysis followed by a midday re-organization.

LoadGuard's schedule has two beats:

1. **Morning** — run the full sense -> plan -> validate loop (``run_workflow``).
2. **Midday** — re-score the day so far, project the remaining hours, and
   re-organize the plan when the projected end-of-day load is high/overload.

A **reactive trigger** (``should_recheck``) supplements the fixed schedule:
when the cognitive load score jumps more than ``REACTIVE_THRESHOLD`` points
between any two consecutive readings, an immediate re-evaluation is warranted.

Both steps are pure, deterministic functions so they can be driven by cron, an
in-process loop, or any orchestration layer — no new dependencies required.
"""

from __future__ import annotations

from dataclasses import dataclass

from .llm import ChatModel
from .models import Event, Task, Worker
from .projection import MiddayReview, run_midday_review
from .workflow import WorkflowResult, run_workflow

# Score jump (points on the 0..100 scale) that triggers an immediate
# re-evaluation outside the fixed morning/midday schedule.
REACTIVE_THRESHOLD = 15.0


@dataclass
class DailyCycle:
    """The two scheduled beats of a LoadGuard workday."""

    morning: WorkflowResult
    midday: MiddayReview


def should_recheck(previous_score: float, current_score: float) -> bool:
    """Return True if the score jumped enough to warrant immediate re-evaluation.

    This supplements the fixed morning/midday schedule: when the cognitive load
    score rises by more than ``REACTIVE_THRESHOLD`` between any two readings,
    the plan should be re-assessed without waiting for the next scheduled beat.
    """
    return current_score - previous_score >= REACTIVE_THRESHOLD


def run_daily_cycle(
    morning_events: list[Event],
    events_so_far: list[Event],
    tasks: list[Task],
    workers: list[Worker] | None = None,
    elapsed_minutes: float = 240.0,
    total_minutes: float = 480.0,
    model: ChatModel | None = None,
    now: float | None = None,
) -> DailyCycle:
    """Run the morning analysis and the midday re-organization."""
    morning = run_workflow(morning_events, tasks, model=model, workers=workers, now=now)
    midday = run_midday_review(
        events_so_far,
        tasks,
        elapsed_minutes,
        total_minutes,
        workers=workers,
        now=now,
    )
    return DailyCycle(morning=morning, midday=midday)
