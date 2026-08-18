"""Unit tests for the guardian safety layer."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loadguard.guardian import (  # noqa: E402
    guard_plan,
    run_deterministic_checks,
    run_llm_guard,
    validate_plan,
)
from loadguard.llm import ChatModel, HeuristicModel  # noqa: E402
from loadguard.models import LoadReport, Plan, PlanItem, Task  # noqa: E402


def _report(score: float = 80.0) -> LoadReport:
    return LoadReport(score=score, level="overload")


def _plan(items: list[PlanItem], note: str = "A safe note.") -> Plan:
    return Plan(load_report=_report(), items=items, note=note)


def _task(pid: str, priority: int = 3) -> Task:
    return Task(id=pid, title=f"Task {pid}", priority=priority)


class TestGuardian(unittest.TestCase):
    def test_valid_plan_passes(self):
        tasks = [_task("a", 5), _task("b", 1)]
        plan = _plan(
            [
                PlanItem(position=1, action="do", task_id="a", title="Task a"),
                PlanItem(position=2, action="delegate", task_id="b", title="Task b"),
            ]
        )
        result = validate_plan(plan, tasks, plan.note)
        self.assertTrue(result.passed, result.summary())
        self.assertEqual(result.engine, "deterministic")

    def test_medical_claim_fails_and_is_sanitized(self):
        tasks = [_task("a", 5)]
        plan = _plan(
            [PlanItem(position=1, action="do", task_id="a", title="Task a")],
            note="You are suffering from burnout and should take a medical leave.",
        )
        plan, result = guard_plan(plan, tasks, HeuristicModel())
        self.assertTrue(result.sanitized)
        self.assertTrue(result.passed)
        self.assertNotIn("burnout", plan.note.lower())

    def test_critical_delegation_fails(self):
        tasks = [_task("a", 5)]
        plan = _plan([PlanItem(position=1, action="delegate", task_id="a", title="Task a")])
        result = validate_plan(plan, tasks, plan.note)
        self.assertFalse(result.passed)
        names = [c.name for c in result.checks if not c.passed]
        self.assertIn("critical_tasks_safe", names)

    def test_unknown_task_fails(self):
        tasks = [_task("a")]
        plan = _plan([PlanItem(position=1, action="do", task_id="ghost", title="Ghost task")])
        result = validate_plan(plan, tasks, plan.note)
        self.assertFalse(result.passed)
        self.assertIn("known_tasks", [c.name for c in result.checks if not c.passed])

    def test_invented_title_fails(self):
        tasks = [_task("a")]
        plan = _plan([PlanItem(position=1, action="do", task_id="a", title="Something else")])
        result = validate_plan(plan, tasks, plan.note)
        self.assertFalse(result.passed)
        self.assertIn("no_invented_data", [c.name for c in result.checks if not c.passed])

    def test_sensitive_data_fails(self):
        tasks = [_task("a")]
        plan = _plan(
            [PlanItem(position=1, action="do", task_id="a", title="Task a")],
            note="Ask john.doe@example.com about it.",
        )
        checks = run_deterministic_checks(plan, tasks, plan.note)
        self.assertFalse(next(c for c in checks if c.name == "no_sensitive_data").passed)

    def test_bad_action_fails(self):
        tasks = [_task("a")]
        plan = _plan([PlanItem(position=1, action="explode", task_id="a", title="x")])
        result = validate_plan(plan, tasks, plan.note)
        self.assertFalse(result.passed)

    def test_diagnostics_not_flagged_as_medical(self):
        """A technical title like 'fix diagnostics' is not a medical diagnosis."""
        tasks = [_task("a")]
        plan = _plan(
            [PlanItem(position=1, action="do", task_id="a", title="Task a")],
            note="Focus on fixing the diagnostics pipeline today.",
        )
        result = validate_plan(plan, tasks, plan.note)
        self.assertTrue(result.passed, result.summary())

    def test_diagnose_word_is_flagged(self):
        tasks = [_task("a")]
        plan = _plan(
            [PlanItem(position=1, action="do", task_id="a", title="Task a")],
            note="I can diagnose that you have a medical condition.",
        )
        result = validate_plan(plan, tasks, plan.note)
        self.assertFalse(result.passed)
        self.assertIn("no_medical_diagnosis", [c.name for c in result.checks if not c.passed])

    def test_negated_diagnose_not_flagged(self):
        tasks = [_task("a")]
        plan = _plan(
            [PlanItem(position=1, action="do", task_id="a", title="Task a")],
            note="LoadGuard does not diagnose medical conditions.",
        )
        result = validate_plan(plan, tasks, plan.note)
        self.assertTrue(result.passed, result.summary())


class _FakeGuardModel(ChatModel):
    name = "fake-guard"

    def __init__(self, verdict: str | None = "safe") -> None:
        # verdict: "safe" | "unsafe" | None (guard_text returns None)
        self._verdict = verdict

    def generate_note(self, load_report, plan, tasks) -> str:
        return "A safe note."

    def guard_text(self, text: str) -> dict | None:
        if self._verdict == "safe":
            return {"safe": True, "issues": []}
        if self._verdict == "unsafe":
            return {"safe": False, "issues": ["disrespectful tone"]}
        return None


class TestLlmGuard(unittest.TestCase):
    def test_heuristic_or_missing_model_skips_llm_guard(self) -> None:
        self.assertIsNone(run_llm_guard(HeuristicModel(), "note"))
        self.assertIsNone(run_llm_guard(None, "note"))

    def test_model_without_verdict_skips(self) -> None:
        self.assertIsNone(run_llm_guard(_FakeGuardModel(None), "note"))

    def test_unsafe_verdict_is_flagged(self) -> None:
        check = run_llm_guard(_FakeGuardModel("unsafe"), "note")
        self.assertIsNotNone(check)
        self.assertEqual(check.name, "granite_guardian")
        self.assertFalse(check.passed)

    def test_validate_plan_with_model_uses_granite_engine(self) -> None:
        tasks = [_task("a", 5)]
        plan = _plan([PlanItem(position=1, action="do", task_id="a", title="Task a")])
        result = validate_plan(plan, tasks, plan.note, _FakeGuardModel("safe"))
        self.assertEqual(result.engine, "granite-guardian")
        self.assertTrue(result.passed)

    def test_guard_plan_reports_structural_failure(self) -> None:
        tasks = [_task("a", 5)]
        plan = _plan([PlanItem(position=1, action="do", task_id="ghost", title="Ghost")])
        _, result = guard_plan(plan, tasks, HeuristicModel())
        self.assertFalse(result.passed)
        self.assertFalse(result.sanitized)


if __name__ == "__main__":
    unittest.main()
