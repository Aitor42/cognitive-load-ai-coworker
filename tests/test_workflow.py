"""Unit tests for the multi-agent workflow and impact estimator."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loadguard.impact import estimate_impact  # noqa: E402
from loadguard.llm import ChatModel  # noqa: E402
from loadguard.models import (  # noqa: E402
    CONTEXT_SWITCH,
    FOCUS_BLOCK,
    MEETING,
    NOTIFICATION,
    VACATION,
    Absence,
    Event,
    FeatureSet,
    Task,
    Worker,
)
from loadguard.scoring import score  # noqa: E402
from loadguard.workflow import run_workflow  # noqa: E402


def _overload_events() -> list[Event]:
    events = [Event(timestamp=0.0, kind=MEETING, duration_minutes=60.0)]
    events += [Event(timestamp=60.0 * i, kind=CONTEXT_SWITCH) for i in range(1, 15)]
    events += [Event(timestamp=30.0 * i, kind=NOTIFICATION) for i in range(1, 40)]
    events.append(Event(timestamp=3600.0, kind=FOCUS_BLOCK, duration_minutes=10.0))
    return events


class TestWorkflow(unittest.TestCase):
    def test_run_workflow_end_to_end(self):
        tasks = [
            Task(id="a", title="Critical fix", priority=5),
            Task(id="b", title="Expenses", priority=1, focus_required=False),
        ]
        result = run_workflow(_overload_events(), tasks)
        self.assertIsNotNone(result.load_report)
        self.assertGreater(len(result.plan.items), 0)
        self.assertTrue(result.plan.note)  # narrative present
        self.assertIn(result.plan.generated_by, {"heuristic", "watsonx"})

    def test_impact_reduces_score_on_overload(self):
        tasks = [Task(id="a", title="Deep work", priority=5)]
        result = run_workflow(_overload_events(), tasks)
        # Overload scenario should project a reduced (or equal) score.
        self.assertLessEqual(result.impact.after_score, result.impact.before_score)

    def test_impact_documents_assumptions(self):
        report = score(FeatureSet(notification_rate=30.0, focus_ratio=0.05))
        from loadguard.recommender import build_plan

        plan = build_plan([], report)
        impact = estimate_impact(FeatureSet(notification_rate=30.0, focus_ratio=0.05), plan)
        self.assertTrue(impact.assumptions)  # every assumption is documented

    def test_workflow_runs_guardian_and_approval_state(self):
        result = run_workflow(_overload_events(), [Task(id="a", title="Fix", priority=5)])
        self.assertIsNotNone(result.guardian)
        self.assertTrue(result.guardian.passed)
        self.assertEqual(result.plan.status, "pending")
        self.assertTrue(result.plan.plan_id)

    def test_workflow_accepts_approval(self):
        result = run_workflow(
            _overload_events(),
            [Task(id="a", title="Fix", priority=5)],
            approval="accepted",
        )
        self.assertEqual(result.plan.status, "accepted")
        self.assertEqual(result.approval, "accepted")

    def test_workflow_reports_personal_trend(self):
        result = run_workflow(
            _overload_events(),
            [Task(id="a", title="Fix", priority=5)],
            history=[40.0, 45.0, 42.0, 50.0, 55.0],
        )
        self.assertIsNotNone(result.baseline)
        self.assertIsNotNone(result.trend)
        self.assertIn("personal baseline", result.trend.summary)

    def test_workflow_plan_has_proposed_by(self):
        result = run_workflow(_overload_events(), [Task(id="a", title="Fix", priority=5)])
        self.assertEqual(result.plan.proposed_by, "deterministic")
        self.assertIsNone(result.proposal)  # no LLM configured -> deterministic fallback

    def test_impact_no_mitigation(self):
        """When no mitigations are in the plan, the score should not change."""
        from loadguard.models import Plan, PlanItem

        features = FeatureSet(
            context_switches_per_hour=5.0,
            notification_rate=10.0,
            focus_ratio=0.5,
        )
        report = score(features)
        # A plan with only 'do' actions provides no mitigation.
        plan = Plan(
            load_report=report,
            items=[
                PlanItem(position=1, action="do", task_id="a", title="Work on task"),
            ],
        )
        impact = estimate_impact(features, plan)
        self.assertEqual(impact.before_score, impact.after_score)
        self.assertEqual(impact.delta, 0.0)

    def test_workflow_detects_reassignment_alerts(self):
        workers = [Worker(id="w1", absences=[Absence(start=0.0, end=1000.0, kind=VACATION)])]
        tasks = [Task(id="a", title="Fix", priority=5, assignee="w1", deadline=500.0)]
        result = run_workflow(_overload_events(), tasks, workers=workers, now=0.0)
        self.assertEqual(len(result.reassignment_alerts), 1)
        self.assertEqual(result.reassignment_alerts[0].task_id, "a")

    def test_workflow_without_workers_has_no_alerts(self):
        result = run_workflow(_overload_events(), [Task(id="a", title="Fix", priority=5)])
        self.assertEqual(result.reassignment_alerts, [])

    def test_workflow_records_proposing_model(self):
        class _Proposer(ChatModel):
            name = "fake"

            def generate_note(self, load_report, plan, tasks):
                return "A safe note."

            def propose_plan(self, features, load_report, tasks):
                return (
                    '{"priority_task_id": "a", "delegate_task_ids": [], '
                    '"inserts": [], "rationale": "ok"}'
                )

        result = run_workflow(
            _overload_events(),
            [Task(id="a", title="Fix", priority=5)],
            model=_Proposer(),
        )
        self.assertEqual(result.plan.proposed_by, "fake")


if __name__ == "__main__":
    unittest.main()
