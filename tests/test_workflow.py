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


class TestImpactCalibration(unittest.TestCase):
    """Tests for history-based calibration and position-aware focus blocks."""

    def test_calibration_with_improving_history(self):
        """When scores are improving, calibration factor is close to 1.0."""
        from loadguard.impact import _calibration_factor

        # Steady drop: interventions are working
        self.assertAlmostEqual(_calibration_factor([80, 70, 60, 50]), 1.0, places=1)

    def test_calibration_with_flat_history(self):
        """When scores are flat, calibration factor is below 1.0."""
        from loadguard.impact import _calibration_factor

        # No improvement: interventions aren't helping
        result = _calibration_factor([60, 60, 60, 60])
        self.assertLess(result, 1.0)

    def test_calibration_with_short_history_returns_one(self):
        from loadguard.impact import _calibration_factor

        self.assertEqual(_calibration_factor([50, 60]), 1.0)
        self.assertEqual(_calibration_factor(None), 1.0)
        self.assertEqual(_calibration_factor([]), 1.0)
        self.assertEqual(_calibration_factor([0.0, 0.0, 0.0, 0.0]), 1.0)

    def test_impact_with_history_adjusts_projection(self):
        """History-calibrated impact uses scaled reduction constants."""
        features = FeatureSet(notification_rate=30.0, focus_ratio=0.05)
        report = score(features)
        from loadguard.recommender import build_plan

        plan = build_plan([], report)
        impact_no_hist = estimate_impact(features, plan)
        impact_flat = estimate_impact(features, plan, history=[60, 60, 60, 60, 60, 60])
        # Flat history -> smaller reductions -> smaller delta
        self.assertLessEqual(impact_flat.delta, impact_no_hist.delta)
        self.assertIn("calibration", impact_flat.assumptions)

    def test_focus_block_position_affects_gain(self):
        """A focus block at position 0 is more effective than one at the end."""
        from loadguard.models import Plan, PlanItem

        features = FeatureSet(focus_ratio=0.1)
        report = score(features)
        # Focus block at start
        plan_early = Plan(
            load_report=report,
            items=[
                PlanItem(position=1, action="focus_block", title="Focus"),
                PlanItem(position=2, action="do", task_id="a", title="Work"),
                PlanItem(position=3, action="do", task_id="b", title="Work2"),
            ],
        )
        # Focus block at end
        plan_late = Plan(
            load_report=report,
            items=[
                PlanItem(position=1, action="do", task_id="a", title="Work"),
                PlanItem(position=2, action="do", task_id="b", title="Work2"),
                PlanItem(position=3, action="focus_block", title="Focus"),
            ],
        )
        impact_early = estimate_impact(features, plan_early)
        impact_late = estimate_impact(features, plan_late)
        # Early focus block should reduce score more
        self.assertGreater(impact_early.delta, impact_late.delta)


class TestSchedulerReactiveCheck(unittest.TestCase):
    """Tests for the reactive trigger in scheduler."""

    def test_large_jump_triggers_recheck(self):
        from loadguard.scheduler import REACTIVE_THRESHOLD, should_recheck

        self.assertTrue(should_recheck(40.0, 40.0 + REACTIVE_THRESHOLD))
        self.assertTrue(should_recheck(30.0, 60.0))

    def test_small_change_does_not_trigger(self):
        from loadguard.scheduler import should_recheck

        self.assertFalse(should_recheck(50.0, 55.0))
        self.assertFalse(should_recheck(50.0, 50.0))

    def test_drop_does_not_trigger(self):
        from loadguard.scheduler import should_recheck

        self.assertFalse(should_recheck(70.0, 50.0))


if __name__ == "__main__":
    unittest.main()
