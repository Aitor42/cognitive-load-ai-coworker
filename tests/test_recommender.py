"""Unit tests for the deterministic planner (``recommender.py``)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loadguard.models import DONE, FeatureSet, Task  # noqa: E402
from loadguard.recommender import build_plan  # noqa: E402
from loadguard.scoring import score  # noqa: E402


def _overload_report():
    return score(
        FeatureSet(
            context_switches_per_hour=15.0,
            meeting_ratio=0.7,
            notification_rate=40.0,
            focus_ratio=0.05,
            multitasking_index=0.8,
        )
    )


class TestBuildPlan(unittest.TestCase):
    def test_overload_delegates_low_priority(self) -> None:
        tasks = [
            Task(id="a", title="Critical fix", priority=5),
            Task(id="b", title="Expenses", priority=1, focus_required=False),
            Task(id="c", title="Backlog labels", priority=2, focus_required=False),
        ]
        plan = build_plan(tasks, _overload_report())
        delegated = [i for i in plan.items if i.action == "delegate"]
        self.assertEqual({i.task_id for i in delegated}, {"b", "c"})

    def test_low_load_delegates_nothing(self) -> None:
        report = score(FeatureSet(focus_ratio=0.9))
        tasks = [Task(id="a", title="Deep work", priority=5)]
        plan = build_plan(tasks, report)
        self.assertFalse([i for i in plan.items if i.action == "delegate"])

    def test_plan_is_ordered_by_priority(self) -> None:
        tasks = [
            Task(id="low", title="Low", priority=2),
            Task(id="high", title="High", priority=5),
            Task(id="mid", title="Mid", priority=4),
        ]
        plan = build_plan(tasks, score(FeatureSet(focus_ratio=0.9)))
        do_items = [i for i in plan.items if i.action == "do"]
        self.assertEqual([i.task_id for i in do_items], ["high", "mid", "low"])

    def test_deadline_breaks_priority_ties(self) -> None:
        tasks = [
            Task(id="later", title="Later", priority=4, deadline=2000.0),
            Task(id="sooner", title="Sooner", priority=4, deadline=1000.0),
        ]
        plan = build_plan(tasks, score(FeatureSet(focus_ratio=0.9)))
        do_items = [i for i in plan.items if i.action == "do"]
        self.assertEqual([i.task_id for i in do_items], ["sooner", "later"])

    def test_non_todo_tasks_are_skipped(self) -> None:
        tasks = [
            Task(id="a", title="Done task", priority=5, status=DONE),
            Task(id="b", title="Todo task", priority=4),
        ]
        plan = build_plan(tasks, score(FeatureSet(focus_ratio=0.9)))
        do_items = [i for i in plan.items if i.action == "do"]
        self.assertEqual([i.task_id for i in do_items], ["b"])

    def test_batch_inserted_for_high_notification(self) -> None:
        report = score(FeatureSet(notification_rate=30.0, focus_ratio=1.0))
        plan = build_plan([Task(id="a", title="Work", priority=3)], report)
        self.assertEqual(plan.items[0].action, "batch")

    def test_focus_block_inserted_before_first_do(self) -> None:
        report = score(FeatureSet(focus_ratio=0.05, notification_rate=0.0))
        plan = build_plan([Task(id="a", title="Work", priority=5)], report)
        self.assertEqual(plan.items[0].action, "focus_block")
        do_positions = [i.position for i in plan.items if i.action == "do"]
        self.assertGreater(min(do_positions), plan.items[0].position)

    def test_break_after_every_two_do_tasks(self) -> None:
        tasks = [Task(id=f"t{i}", title=f"Task {i}", priority=5) for i in range(4)]
        plan = build_plan(tasks, _overload_report())
        actions = [i.action for i in plan.items]
        self.assertIn("break", actions)
        self.assertEqual(actions.count("do"), 4)

    def test_focus_block_appended_when_no_tasks(self) -> None:
        report = score(FeatureSet(focus_ratio=0.0))
        plan = build_plan([], report)
        self.assertEqual([i.action for i in plan.items], ["focus_block"])


if __name__ == "__main__":
    unittest.main()
