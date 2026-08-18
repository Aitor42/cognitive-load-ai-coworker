"""Unit tests for the data models and their validation (``models.py``)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loadguard.models import (  # noqa: E402
    DONE,
    OVERLOAD,
    Event,
    LoadReport,
    Plan,
    PlanItem,
    Task,
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
