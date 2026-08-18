"""Tests for the LangGraph flow (sequential fallback runs without langgraph)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loadguard.langgraph_flow import run_graph, run_sequential_graph  # noqa: E402
from loadguard.models import (  # noqa: E402
    CONTEXT_SWITCH,
    FOCUS_BLOCK,
    MEETING,
    NOTIFICATION,
    Event,
    Task,
)


def _events() -> list[Event]:
    events = [Event(timestamp=0.0, kind=MEETING, duration_minutes=60.0)]
    events += [Event(timestamp=60.0 * i, kind=CONTEXT_SWITCH) for i in range(1, 12)]
    events += [Event(timestamp=30.0 * i, kind=NOTIFICATION) for i in range(1, 30)]
    events.append(Event(timestamp=3600.0, kind=FOCUS_BLOCK, duration_minutes=10.0))
    return events


def _tasks() -> list[Task]:
    return [
        Task(id="a", title="Critical fix", priority=5),
        Task(id="b", title="Expenses", priority=1),
    ]


class TestLangGraphFlow(unittest.TestCase):
    def test_sequential_full_loop_with_approval(self):
        state = run_graph(_events(), _tasks(), approval={"decision": "accepted"}, sequential=True)
        self.assertEqual(state["status"], "measured")
        self.assertEqual(state["plan"].status, "accepted")
        self.assertIn("ics", state["exports"])
        self.assertIn("BEGIN:VCALENDAR", state["exports"]["ics"])
        self.assertIn("tasks_csv", state["exports"])
        self.assertIsNotNone(state["guardian"])
        self.assertIsNotNone(state["impact"])

    def test_waits_for_human_approval(self):
        state = run_graph(_events(), _tasks(), sequential=True)
        self.assertEqual(state["status"], "awaiting_approval")
        self.assertEqual(state["plan"].status, "pending")
        self.assertNotIn("exports", state)  # nothing applied without approval

    def test_rejected_plan_not_applied(self):
        state = run_graph(_events(), _tasks(), approval={"decision": "rejected"}, sequential=True)
        self.assertEqual(state["status"], "rejected")
        self.assertEqual(state["plan"].status, "rejected")
        self.assertNotIn("exports", state)

    def test_observed_outcome_measured(self):
        outcome = [Event(timestamp=0.0, kind=FOCUS_BLOCK, duration_minutes=60.0)]
        state = run_graph(
            _events(),
            _tasks(),
            approval={"decision": "accepted"},
            outcome_events=outcome,
            sequential=True,
        )
        self.assertIsNotNone(state["observed"])
        self.assertLess(state["observed"]["score"], 40)

    def test_run_sequential_graph_equivalent(self):
        state = run_sequential_graph(
            {"events": _events(), "tasks": _tasks(), "approval": {"decision": "accepted"}}
        )
        self.assertEqual(state["status"], "measured")


if __name__ == "__main__":
    unittest.main()
