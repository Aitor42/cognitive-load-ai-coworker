"""Unit tests for the daily cycle (morning + midday) orchestration."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loadguard.models import Event, FOCUS_BLOCK, MEETING, Task, Worker  # noqa: E402
from loadguard.projection import MiddayReview  # noqa: E402
from loadguard.scheduler import DailyCycle, run_daily_cycle  # noqa: E402
from loadguard.workflow import WorkflowResult  # noqa: E402


def _events() -> list[Event]:
    return [
        Event(timestamp=0.0, kind=MEETING, duration_minutes=60.0),
        Event(timestamp=3600.0, kind=FOCUS_BLOCK, duration_minutes=60.0),
    ]


class TestDailyCycle(unittest.TestCase):
    def test_run_daily_cycle_returns_both_beats(self) -> None:
        tasks = [Task(id="a", title="Fix", priority=5)]
        workers = [Worker(id="w1", name="Ada")]
        cycle = run_daily_cycle(
            _events(),
            _events(),
            tasks,
            workers=workers,
            elapsed_minutes=240.0,
            total_minutes=480.0,
        )
        self.assertIsInstance(cycle, DailyCycle)
        self.assertIsInstance(cycle.morning, WorkflowResult)
        self.assertIsInstance(cycle.midday, MiddayReview)
        self.assertEqual(cycle.morning.plan.plan_id, cycle.morning.plan.plan_id)

    def test_run_daily_cycle_with_role_and_tz(self) -> None:
        tasks = [Task(id="a", title="Fix", priority=5)]
        workers = [Worker(id="w1", name="Ada")]
        cycle = run_daily_cycle(
            _events(),
            _events(),
            tasks,
            workers=workers,
            elapsed_minutes=240.0,
            total_minutes=480.0,
            tz_name="Europe/Madrid",
            role="developer",
        )
        self.assertIsInstance(cycle, DailyCycle)
        self.assertIsInstance(cycle.morning, WorkflowResult)
        self.assertIsInstance(cycle.midday, MiddayReview)


if __name__ == "__main__":
    unittest.main()
