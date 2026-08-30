"""Tests for the LangGraph flow (sequential fallback runs without langgraph)."""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loadguard import langgraph_flow as lg  # noqa: E402
from loadguard.langgraph_flow import (  # noqa: E402
    _approval_router,
    build_graph,
    run_graph,
    run_sequential_graph,
)
from loadguard.llm import ChatModel  # noqa: E402
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


class _FakeModel(ChatModel):
    """A model that always proposes a valid plan adjustment."""

    name = "fake"

    def generate_note(self, load_report, plan, tasks) -> str:
        return "A safe, respectful note."

    def propose_plan(self, features, load_report, tasks) -> str:
        return '{"inserts": [{"action": "break"}], "rationale": "Recover."}'


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

    def test_empty_approval_waits(self):
        state = run_graph(_events(), _tasks(), approval={}, sequential=True)
        self.assertEqual(state["status"], "awaiting_approval")

    def test_rejected_plan_not_applied(self):
        state = run_graph(_events(), _tasks(), approval={"decision": "rejected"}, sequential=True)
        self.assertEqual(state["status"], "rejected")
        self.assertEqual(state["plan"].status, "rejected")
        self.assertNotIn("exports", state)

    def test_invalid_decision_treated_as_rejected(self):
        state = run_graph(_events(), _tasks(), approval={"decision": "weird"}, sequential=True)
        self.assertEqual(state["status"], "rejected")
        self.assertEqual(state["plan"].status, "rejected")

    def test_edited_plan_applied_with_empty_exports(self):
        state = run_graph(_events(), _tasks(), approval={"decision": "edited"}, sequential=True)
        self.assertEqual(state["status"], "measured")
        self.assertEqual(state["exports"], {})

    def test_model_proposal_marks_proposed_by(self):
        state = run_graph(
            _events(),
            _tasks(),
            model=_FakeModel(),
            approval={"decision": "accepted"},
            sequential=True,
        )
        self.assertEqual(state["plan"].proposed_by, "fake")

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

    def test_approval_router(self):
        self.assertEqual(_approval_router({}), "awaiting_approval")
        self.assertEqual(_approval_router({"status": "accepted"}), "accepted")


class TestLanggraphDetection(unittest.TestCase):
    def tearDown(self) -> None:
        lg._langgraph_available = None

    def test_not_installed(self) -> None:
        lg._langgraph_available = None
        with mock.patch.dict(sys.modules, {"langgraph": None}):
            self.assertFalse(lg.langgraph_installed())

    def test_installed(self) -> None:
        lg._langgraph_available = None
        fake = types.ModuleType("langgraph")
        with mock.patch.dict(sys.modules, {"langgraph": fake}):
            self.assertTrue(lg.langgraph_installed())

    def test_run_graph_with_role_and_baseline_adaptation(self) -> None:
        from loadguard.models import Worker

        workers = [Worker(id="w1", name="Alice")]
        # Run with researcher profile, outcome events, and personal history
        state = run_graph(
            _events(),
            _tasks(),
            role="researcher",
            history=[80.0, 75.0, 70.0],
            workers=workers,
            outcome_events=_events(),
            sequential=True,
            approval={"decision": "accepted"},
        )
        self.assertEqual(state["status"], "measured")
        self.assertIsNotNone(state["load_report"])
        self.assertIsNotNone(state["plan"])
        self.assertIsNotNone(state["observed"])
        self.assertIn("score", state["observed"])

    def test_cached_value_returned_without_reimport(self) -> None:
        lg._langgraph_available = True
        self.assertTrue(lg.langgraph_installed())
        lg._langgraph_available = False
        self.assertFalse(lg.langgraph_installed())


class TestBuildGraph(unittest.TestCase):
    def test_build_graph_compiles(self) -> None:
        graph = mock.MagicMock()
        graph.compile.return_value = "compiled"
        state_graph_cls = mock.MagicMock(return_value=graph)

        graph_mod = types.ModuleType("langgraph.graph")
        graph_mod.END = "END"
        graph_mod.StateGraph = state_graph_cls

        langgraph_mod = types.ModuleType("langgraph")
        langgraph_mod.graph = graph_mod

        with mock.patch.dict(
            sys.modules, {"langgraph": langgraph_mod, "langgraph.graph": graph_mod}
        ):
            self.assertEqual(build_graph(), "compiled")

        self.assertEqual(graph.add_node.call_count, 8)
        graph.set_entry_point.assert_called_once_with("collect_signals")
        graph.compile.assert_called_once_with()

    def test_run_graph_uses_langgraph_when_installed(self) -> None:
        with (
            mock.patch.object(lg, "langgraph_installed", return_value=True),
            mock.patch.object(lg, "build_graph") as bg,
        ):
            bg.return_value.invoke.return_value = {"status": "done"}
            state = run_graph(_events(), _tasks())
            self.assertEqual(state, {"status": "done"})
            bg.return_value.invoke.assert_called_once()


if __name__ == "__main__":
    unittest.main()
