"""Unit tests for the scoring engine, signal features, and planner."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loadguard.models import (  # noqa: E402
    CONTEXT_SWITCH,
    MEETING,
    MODERATE,
    HIGH,
    OVERLOAD,
    Event,
    FeatureSet,
    Task,
)
from loadguard.recommender import build_plan  # noqa: E402
from loadguard.scoring import score  # noqa: E402
from loadguard.signals import compute_features, parse_event  # noqa: E402


def events_between(
    count: int, kind: str, start: float = 0.0, gap: float = 60.0, duration: float = 0.0
):
    return [
        Event(timestamp=start + i * gap, kind=kind, duration_minutes=duration) for i in range(count)
    ]


class TestSignals(unittest.TestCase):
    def test_parse_iso_timestamp(self):
        e = parse_event({"timestamp": "2026-08-17T09:00:00Z", "kind": "meeting"})
        self.assertEqual(e.kind, MEETING)
        self.assertGreater(e.timestamp, 0)

    def test_parse_rejects_unknown_kind(self):
        with self.assertRaises(ValueError):
            parse_event({"timestamp": "2026-08-17T09:00:00Z", "kind": "bogus"})

    def test_empty_events_use_floor_window(self):
        f = compute_features([])
        self.assertEqual(f.context_switches_per_hour, 0.0)
        self.assertEqual(f.notification_rate, 0.0)

    def test_feature_rates_are_per_hour(self):
        # 6 context switches over exactly 1 hour -> 6 per hour.
        events = events_between(6, CONTEXT_SWITCH, start=0.0, gap=600.0)
        f = compute_features(events, window_minutes=60.0)
        self.assertAlmostEqual(f.context_switches_per_hour, 6.0)

    def test_multitasking_index_during_meeting(self):
        """Context switches during a meeting should increase multitasking index."""
        events = [
            Event(timestamp=0.0, kind=MEETING, duration_minutes=60.0),
            # These switches happen during the 60-min meeting (0..3600s)
            Event(timestamp=300.0, kind=CONTEXT_SWITCH),
            Event(timestamp=600.0, kind=CONTEXT_SWITCH),
            # This switch happens after the meeting
            Event(timestamp=4000.0, kind=CONTEXT_SWITCH),
        ]
        f = compute_features(events, window_minutes=120.0)
        # 2 out of 3 switches during meeting -> multitasking ≈ 0.667
        self.assertAlmostEqual(f.multitasking_index, 2 / 3, places=2)

    def test_parse_datetime_object(self):
        from datetime import datetime, timezone

        dt = datetime(2026, 8, 17, 9, 0, 0, tzinfo=timezone.utc)
        e = parse_event({"timestamp": dt, "kind": "meeting"})
        self.assertEqual(e.kind, MEETING)
        self.assertEqual(e.timestamp, dt.timestamp())

    def test_zero_or_negative_window_safely_falls_back(self):
        events = events_between(6, CONTEXT_SWITCH, start=0.0, gap=600.0)
        f_zero = compute_features(events, window_minutes=0.0)
        f_neg = compute_features(events, window_minutes=-30.0)
        self.assertGreater(f_zero.context_switches_per_hour, 0)
        self.assertGreater(f_neg.context_switches_per_hour, 0)


class TestScoring(unittest.TestCase):
    def test_empty_features_score_low(self):
        # No signals -> no interruption load; only a small baseline from the
        # inverted focus term (no focus blocks logged), still level "low".
        r = score(FeatureSet())
        self.assertLess(r.score, 25.0)
        self.assertEqual(r.level, "low")

    def test_heavy_load_scores_overload(self):
        f = FeatureSet(
            context_switches_per_hour=15.0,
            meeting_ratio=0.7,
            notification_rate=40.0,
            focus_ratio=0.05,
            multitasking_index=0.8,
        )
        r = score(f)
        self.assertGreaterEqual(r.score, 75.0)
        self.assertEqual(r.level, OVERLOAD)

    def test_focus_only_scores_low(self):
        f = FeatureSet(focus_ratio=0.9)
        r = score(f)
        self.assertLess(r.score, 25.0)

    def test_explanation_ranks_focus_as_protective(self):
        """High focus time must not be reported as a load driver."""
        f = FeatureSet(
            context_switches_per_hour=6.0,
            meeting_ratio=0.2,
            notification_rate=2.0,
            focus_ratio=0.95,
            multitasking_index=0.1,
        )
        r = score(f)
        # context switches (6/12*0.30=0.15) dominates; focus is protective
        # ((1-0.95)*0.15=0.0075), so it must not appear as a top driver.
        self.assertIn("context switches per hour", r.explanation)
        self.assertNotIn("focus time", r.explanation)

    def test_score_within_bounds(self):
        f = FeatureSet(
            context_switches_per_hour=999.0,
            meeting_ratio=1.0,
            notification_rate=999.0,
            focus_ratio=0.0,
            multitasking_index=1.0,
        )
        r = score(f)
        self.assertLessEqual(r.score, 100.0)

    def test_moderate_boundary(self):
        """A score of exactly 25 should be moderate, not low."""
        # Carefully craft features that produce a score around 25.
        f = FeatureSet(
            context_switches_per_hour=4.0,
            meeting_ratio=0.2,
            notification_rate=10.0,
            focus_ratio=0.4,
            multitasking_index=0.2,
        )
        r = score(f)
        self.assertGreaterEqual(r.score, 25.0)
        self.assertIn(r.level, (MODERATE, HIGH))

    def test_high_boundary(self):
        """Features that push score into the high range (50-75)."""
        f = FeatureSet(
            context_switches_per_hour=9.0,
            meeting_ratio=0.5,
            notification_rate=20.0,
            focus_ratio=0.1,
            multitasking_index=0.5,
        )
        r = score(f)
        self.assertGreaterEqual(r.score, 50.0)
        self.assertLess(r.score, 75.0)
        self.assertEqual(r.level, HIGH)


class TestRecommender(unittest.TestCase):
    def _overload_report(self):
        return score(
            FeatureSet(
                context_switches_per_hour=15.0,
                meeting_ratio=0.7,
                notification_rate=40.0,
                focus_ratio=0.05,
                multitasking_index=0.8,
            )
        )

    def test_overload_delegates_low_priority(self):
        tasks = [
            Task(id="a", title="Critical fix", priority=5),
            Task(id="b", title="Expenses", priority=1, focus_required=False),
            Task(id="c", title="Backlog labels", priority=2, focus_required=False),
        ]
        plan = build_plan(tasks, self._overload_report())
        delegated = [i for i in plan.items if i.action == "delegate"]
        self.assertEqual({i.task_id for i in delegated}, {"b", "c"})

    def test_low_load_delegates_nothing(self):
        report = score(FeatureSet(focus_ratio=0.9))
        tasks = [Task(id="a", title="Deep work", priority=5)]
        plan = build_plan(tasks, report)
        self.assertFalse([i for i in plan.items if i.action == "delegate"])

    def test_plan_is_ordered_by_priority(self):
        tasks = [
            Task(id="low", title="Low", priority=2),
            Task(id="high", title="High", priority=5),
            Task(id="mid", title="Mid", priority=4),
        ]
        plan = build_plan(tasks, score(FeatureSet(focus_ratio=0.9)))
        do_items = [i for i in plan.items if i.action == "do"]
        self.assertEqual(do_items[0].task_id, "high")
        self.assertEqual(do_items[1].task_id, "mid")
        self.assertEqual(do_items[2].task_id, "low")


if __name__ == "__main__":
    unittest.main()
