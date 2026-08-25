"""Unit tests for end-of-day projection and the midday review."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loadguard.models import (  # noqa: E402
    CONTEXT_SWITCH,
    FOCUS_BLOCK,
    LEAVE,
    MEETING,
    NOTIFICATION,
    Absence,
    Event,
    FeatureSet,
    Task,
    Worker,
)
from loadguard.projection import (  # noqa: E402
    REPLAN_LEVELS,
    DayProjection,
    MiddayReview,
    project_end_of_day,
    run_midday_review,
)


def _overload_events() -> list[Event]:
    events = [Event(timestamp=0.0, kind=MEETING, duration_minutes=120.0)]
    events += [Event(timestamp=60.0 * i, kind=CONTEXT_SWITCH) for i in range(1, 50)]
    events += [Event(timestamp=30.0 * i, kind=NOTIFICATION) for i in range(1, 100)]
    events.append(Event(timestamp=14400.0, kind=FOCUS_BLOCK, duration_minutes=10.0))
    return events


def _calm_events() -> list[Event]:
    return [Event(timestamp=0.0, kind=FOCUS_BLOCK, duration_minutes=200.0)]


class TestProjectEndOfDay(unittest.TestCase):
    def test_remaining_defaults_to_observed(self) -> None:
        observed = FeatureSet(context_switches_per_hour=6.0)
        projection = project_end_of_day(observed, 240.0, 480.0)
        self.assertEqual(projection.projected_remaining.context_switches_per_hour, 6.0)
        self.assertEqual(projection.full_day.context_switches_per_hour, 6.0)

    def test_time_weighted_blend(self) -> None:
        observed = FeatureSet(context_switches_per_hour=6.0)
        remaining = FeatureSet(context_switches_per_hour=2.0)
        projection = project_end_of_day(observed, 120.0, 480.0, remaining=remaining)
        # (6 * 120 + 2 * 360) / 480 = 3.0
        self.assertAlmostEqual(projection.full_day.context_switches_per_hour, 3.0)
        self.assertEqual(projection.remaining_minutes, 360.0)
        self.assertEqual(projection.elapsed_minutes, 120.0)

    def test_zero_elapsed_returns_remaining(self) -> None:
        observed = FeatureSet(notification_rate=10.0)
        remaining = FeatureSet(notification_rate=2.0)
        projection = project_end_of_day(observed, 0.0, 480.0, remaining=remaining)
        self.assertAlmostEqual(projection.full_day.notification_rate, 2.0)

    def test_zero_total_returns_observed(self) -> None:
        observed = FeatureSet(meeting_ratio=0.5)
        projection = project_end_of_day(observed, 0.0, 0.0)
        self.assertAlmostEqual(projection.full_day.meeting_ratio, 0.5)

    def test_full_day_elapsed_returns_observed(self) -> None:
        observed = FeatureSet(focus_ratio=0.5)
        projection = project_end_of_day(observed, 480.0, 480.0)
        self.assertEqual(projection.remaining_minutes, 0.0)
        self.assertAlmostEqual(projection.full_day.focus_ratio, 0.5)

    def test_load_report_attached(self) -> None:
        projection = project_end_of_day(FeatureSet(), 240.0, 480.0)
        self.assertIsNotNone(projection.load_report)
        self.assertIsInstance(projection, DayProjection)


class TestRunMiddayReview(unittest.TestCase):
    def test_reorganizes_on_overload(self) -> None:
        review = run_midday_review(
            _overload_events(), [Task(id="a", title="Fix", priority=5)], 240.0, 480.0
        )
        self.assertIsInstance(review, MiddayReview)
        self.assertTrue(review.reorganized)
        self.assertIsNotNone(review.plan)
        self.assertIn("re-organizing", review.rationale)
        self.assertIn(review.projected_level, REPLAN_LEVELS)
        self.assertEqual(review.reassignment_alerts, [])

    def test_does_not_reorganize_on_low_load(self) -> None:
        review = run_midday_review(
            _calm_events(), [Task(id="a", title="Fix", priority=5)], 240.0, 480.0
        )
        self.assertFalse(review.reorganized)
        self.assertIsNone(review.plan)
        self.assertIn("morning plan stands", review.rationale)

    def test_detects_reassignment_alerts(self) -> None:
        workers = [
            Worker(id="w1", absences=[Absence(start=0.0, end=1000.0, kind=LEAVE)]),
            Worker(id="w2", absences=[]),
        ]
        tasks = [Task(id="a", title="Fix", priority=5, assignee="w1", deadline=500.0)]
        review = run_midday_review(_calm_events(), tasks, 240.0, 480.0, workers=workers, now=0.0)
        self.assertEqual(len(review.reassignment_alerts), 1)
        self.assertEqual(review.reassignment_alerts[0].suggested_assignees, ["w2"])

    def test_observed_and_projected_scores_present(self) -> None:
        review = run_midday_review(_calm_events(), [], 240.0, 480.0)
        self.assertGreaterEqual(review.observed_score, 0.0)
        self.assertGreaterEqual(review.projected_score, 0.0)
        self.assertTrue(review.observed_level)
        self.assertTrue(review.projected_level)
        self.assertEqual(review.remaining_minutes, 240.0)

    def test_completed_tasks_excluded_from_midday_replan(self) -> None:
        """Midday review only replans tasks not in completed_task_ids."""
        tasks = [
            Task(id="a", title="Done", priority=5),
            Task(id="b", title="Remaining", priority=4),
        ]
        review = run_midday_review(
            _overload_events(),
            tasks,
            240.0,
            480.0,
            completed_task_ids={"a"},
        )
        if review.plan:
            plan_task_ids = {i.task_id for i in review.plan.items if i.task_id}
            self.assertNotIn("a", plan_task_ids)


class TestFatigueFactor(unittest.TestCase):
    """Tests for the time-of-day fatigue amplification."""

    def test_fatigue_amplifies_projected_score(self) -> None:
        """Projection with elapsed time should score higher than without."""
        from loadguard.projection import _apply_fatigue

        base = FeatureSet(context_switches_per_hour=6.0, notification_rate=10.0)
        fatigued = _apply_fatigue(base, elapsed=240.0, total=480.0)
        # 240/480 = 0.5 -> fatigue = 1 + 0.15*0.5 = 1.075
        self.assertAlmostEqual(fatigued.context_switches_per_hour, 6.0 * 1.075)
        self.assertAlmostEqual(fatigued.notification_rate, 10.0 * 1.075)

    def test_fatigue_clamps_multitasking_to_one(self) -> None:
        from loadguard.projection import _apply_fatigue

        base = FeatureSet(multitasking_index=0.95)
        fatigued = _apply_fatigue(base, elapsed=480.0, total=480.0)
        self.assertLessEqual(fatigued.multitasking_index, 1.0)

    def test_no_fatigue_when_zero_elapsed(self) -> None:
        from loadguard.projection import _apply_fatigue

        base = FeatureSet(context_switches_per_hour=6.0)
        result = _apply_fatigue(base, elapsed=0.0, total=480.0)
        self.assertEqual(result.context_switches_per_hour, 6.0)

    def test_blended_features_unchanged_by_fatigue(self) -> None:
        """full_day retains raw blended values; fatigue only affects load_report."""
        observed = FeatureSet(context_switches_per_hour=6.0)
        projection = project_end_of_day(observed, 240.0, 480.0)
        # Blended feature should be 6.0 (same rate assumed for rest of day)
        self.assertAlmostEqual(projection.full_day.context_switches_per_hour, 6.0)
        # But the load_report.score should reflect fatigue (higher than raw)


if __name__ == "__main__":
    unittest.main()
