"""Daily cycle: morning analysis followed by a midday re-organization.

LoadGuard's schedule has two beats:

1. **Morning** — run the full sense -> plan -> validate loop (``run_workflow``).
2. **Midday** — re-score the day so far, project the remaining hours, and
   re-organize the plan when the projected end-of-day load is high/overload.

Both steps are pure, deterministic functions so they can be driven by cron, an
in-process loop, or any orchestration layer — no new dependencies required.
"""

from __future__ import annotations

from dataclasses import dataclass

from .llm import ChatModel
from .models import Event, Task, Worker
from .projection import MiddayReview, run_midday_review
from .workflow import WorkflowResult, run_workflow


@dataclass
class DailyCycle:
    """The two scheduled beats of a LoadGuard workday."""

    morning: WorkflowResult
    midday: MiddayReview


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
