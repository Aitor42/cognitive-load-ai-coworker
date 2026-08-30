"""Unit tests for the data models and their validation (``models.py``)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loadguard.models import (  # noqa: E402
    DONE,
    LEAVE,
    OVERLOAD,
    VACATION,
    Absence,
    Event,
    LoadReport,
    Plan,
    PlanItem,
    ReassignmentAlert,
    Task,
    Worker,
)


class TestEvent(unittest.TestCase):
    def test_negative_duration_clamped_to_zero(self) -> None:
        e = Event(timestamp=0.0, kind="meeting", duration_minutes=-10.0)
        self.assertEqual(e.duration_minutes, 0.0)

    def test_unknown_kind_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Event(timestamp=0.0, kind="bogus")

    def test_defaults(self) -> None:
        e = Event(timestamp=0.0, kind="notification")
        self.assertEqual(e.duration_minutes, 0.0)
        self.assertEqual(e.meta, {})


class TestTask(unittest.TestCase):
    def test_defaults(self) -> None:
        t = Task(id="a", title="Fix", priority=3)
        self.assertEqual(t.duration_minutes, 30.0)
        self.assertTrue(t.focus_required)
        self.assertIsNone(t.deadline)
        self.assertEqual(t.status, "todo")

    def test_custom_status(self) -> None:
        t = Task(id="a", title="Fix", priority=5, status=DONE)
        self.assertEqual(t.status, DONE)

    def test_assignee_defaults_to_none(self) -> None:
        t = Task(id="a", title="Fix", priority=3)
        self.assertIsNone(t.assignee)

    def test_assignee_custom(self) -> None:
        t = Task(id="a", title="Fix", priority=3, assignee="w1")
        self.assertEqual(t.assignee, "w1")

    def test_priority_and_duration_clamped(self) -> None:
        t_low = Task(id="a", title="Low", priority=0, duration_minutes=-10.0)
        self.assertEqual(t_low.priority, 1)
        self.assertEqual(t_low.duration_minutes, 0.0)

        t_high = Task(id="b", title="High", priority=99)
        self.assertEqual(t_high.priority, 5)


class TestAbsence(unittest.TestCase):
    def test_defaults(self) -> None:
        a = Absence(start=0.0, end=10.0)
        self.assertEqual(a.kind, LEAVE)
        self.assertEqual(a.note, "")

    def test_vacation_kind(self) -> None:
        a = Absence(start=0.0, end=10.0, kind=VACATION)
        self.assertEqual(a.kind, VACATION)

    def test_unknown_kind_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Absence(start=0.0, end=10.0, kind="bogus")

    def test_end_before_start_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Absence(start=10.0, end=0.0)


class TestWorker(unittest.TestCase):
    def test_defaults(self) -> None:
        w = Worker(id="w1")
        self.assertEqual(w.name, "")
        self.assertEqual(w.absences, [])

    def test_with_absences(self) -> None:
        w = Worker(id="w1", name="Ada", absences=[Absence(start=0.0, end=10.0)])
        self.assertEqual(w.name, "Ada")
        self.assertEqual(len(w.absences), 1)


class TestReassignmentAlert(unittest.TestCase):
    def test_defaults(self) -> None:
        alert = ReassignmentAlert(
            task_id="a", title="Fix", assignee="w1", deadline=10.0, reason="away"
        )
        self.assertEqual(alert.suggested_assignees, [])


class TestPlan(unittest.TestCase):
    def test_defaults(self) -> None:
        p = Plan(load_report=LoadReport(score=50.0, level=OVERLOAD))
        self.assertEqual(p.items, [])
        self.assertEqual(p.status, "pending")
        self.assertEqual(p.generated_by, "heuristic")
        self.assertEqual(p.proposed_by, "deterministic")

    def test_item_fields(self) -> None:
        item = PlanItem(position=1, action="do", task_id="a", title="Fix")
        self.assertEqual(item.action, "do")
        self.assertEqual(item.rationale, "")


if __name__ == "__main__":
    unittest.main()
