"""Unit tests for signal ingestion and feature extraction (``signals.py``)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime, UTC
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loadguard.models import (  # noqa: E402
    CONTEXT_SWITCH,
    FOCUS_BLOCK,
    MEETING,
    NOTIFICATION,
    Event,
)
from loadguard.signals import (  # noqa: E402
    _is_ai_source,
    _to_epoch,
    _window_minutes,
    compute_features,
    load_events,
    parse_event,
)


def events_between(
    count: int, kind: str, start: float = 0.0, gap: float = 60.0, duration: float = 0.0
) -> list[Event]:
    return [
        Event(timestamp=start + i * gap, kind=kind, duration_minutes=duration) for i in range(count)
    ]


class TestToEpoch(unittest.TestCase):
    def test_numeric(self) -> None:
        self.assertEqual(_to_epoch(1_700_000_000), 1_700_000_000.0)
        self.assertEqual(_to_epoch(1_700_000_000.5), 1_700_000_000.5)

    def test_naive_datetime_assumes_utc(self) -> None:
        naive = datetime(2026, 8, 17, 9, 0, 0)
        self.assertEqual(_to_epoch(naive), naive.replace(tzinfo=UTC).timestamp())

    def test_naive_iso_string_assumes_utc(self) -> None:
        self.assertEqual(
            _to_epoch("2026-08-17T09:00:00"),
            datetime(2026, 8, 17, 9, 0, 0, tzinfo=UTC).timestamp(),
        )

    def test_rejects_unparseable_value(self) -> None:
        with self.assertRaises(ValueError):
            _to_epoch([1, 2, 3])


class TestParseEvent(unittest.TestCase):
    def test_parse_iso_timestamp(self) -> None:
        e = parse_event({"timestamp": "2026-08-17T09:00:00Z", "kind": "meeting"})
        self.assertEqual(e.kind, MEETING)
        self.assertGreater(e.timestamp, 0)

    def test_parse_datetime_object(self) -> None:
        dt = datetime(2026, 8, 17, 9, 0, 0, tzinfo=UTC)
        e = parse_event({"timestamp": dt, "kind": "meeting"})
        self.assertEqual(e.kind, MEETING)
        self.assertEqual(e.timestamp, dt.timestamp())

    def test_parse_rejects_unknown_kind(self) -> None:
        with self.assertRaises(ValueError):
            parse_event({"timestamp": "2026-08-17T09:00:00Z", "kind": "bogus"})


class TestLoadEvents(unittest.TestCase):
    def test_skips_comments_and_blanks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            path.write_text(
                '# comment\n\n{"timestamp": "2026-08-17T09:00:00Z", "kind": "meeting", '
                '"duration_minutes": 30}\n',
                encoding="utf-8",
            )
            events = load_events(path)
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].kind, MEETING)
            self.assertEqual(events[0].duration_minutes, 30.0)

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_events(Path("does-not-exist.jsonl"))


class TestComputeFeatures(unittest.TestCase):
    def test_empty_events_use_floor_window(self) -> None:
        f = compute_features([])
        self.assertEqual(f.context_switches_per_hour, 0.0)
        self.assertEqual(f.notification_rate, 0.0)

    def test_feature_rates_are_per_hour(self) -> None:
        events = events_between(6, CONTEXT_SWITCH, start=0.0, gap=600.0)
        f = compute_features(events, window_minutes=60.0)
        self.assertAlmostEqual(f.context_switches_per_hour, 6.0)

    def test_meeting_focus_and_notification_features(self) -> None:
        events = [
            Event(timestamp=0.0, kind=MEETING, duration_minutes=30.0),
            Event(timestamp=0.0, kind=FOCUS_BLOCK, duration_minutes=15.0),
            Event(timestamp=100.0, kind=NOTIFICATION),
            Event(timestamp=200.0, kind=NOTIFICATION),
        ]
        f = compute_features(events, window_minutes=60.0)
        self.assertAlmostEqual(f.meeting_ratio, 0.5)
        self.assertAlmostEqual(f.focus_ratio, 0.25)
        self.assertAlmostEqual(f.notification_rate, 2.0)

    def test_multitasking_index_during_meeting(self) -> None:
        events = [
            Event(timestamp=0.0, kind=MEETING, duration_minutes=60.0),
            Event(timestamp=300.0, kind=CONTEXT_SWITCH),
            Event(timestamp=600.0, kind=CONTEXT_SWITCH),
            Event(timestamp=4000.0, kind=CONTEXT_SWITCH),
        ]
        f = compute_features(events, window_minutes=120.0)
        self.assertAlmostEqual(f.multitasking_index, 2 / 3, places=2)

    def test_implicit_window_uses_event_span(self) -> None:
        events = [
            Event(timestamp=0.0, kind=CONTEXT_SWITCH),
            Event(timestamp=7200.0, kind=CONTEXT_SWITCH),
        ]
        f = compute_features(events)
        self.assertAlmostEqual(f.context_switches_per_hour, 1.0)

    def test_window_floor_is_fifteen_minutes(self) -> None:
        events = [
            Event(timestamp=0.0, kind=CONTEXT_SWITCH),
            Event(timestamp=60.0, kind=CONTEXT_SWITCH),
        ]
        self.assertEqual(_window_minutes(events), 15.0)

    def test_multiday_window_calculation(self) -> None:
        # Events across 3 days (e.g. Day 1, Day 2, Day 3)
        events = [
            Event(timestamp=0.0, kind=CONTEXT_SWITCH),
            Event(timestamp=86400.0 * 1, kind=CONTEXT_SWITCH),
            Event(timestamp=86400.0 * 2, kind=CONTEXT_SWITCH),
        ]
        # 3 days * 480 min = 1440 min
        self.assertEqual(_window_minutes(events), 1440.0)

    def test_unknown_event_kind_is_ignored(self) -> None:
        from types import SimpleNamespace

        events = [SimpleNamespace(timestamp=0.0, kind="bogus", duration_minutes=0.0)]
        f = compute_features(events, window_minutes=60.0)
        self.assertEqual(f.context_switches_per_hour, 0.0)
        self.assertEqual(f.notification_rate, 0.0)
        self.assertEqual(f.meeting_ratio, 0.0)
        self.assertEqual(f.focus_ratio, 0.0)

    def test_zero_or_negative_window_safely_falls_back(self) -> None:
        events = events_between(6, CONTEXT_SWITCH, start=0.0, gap=600.0)
        f_zero = compute_features(events, window_minutes=0.0)
        f_neg = compute_features(events, window_minutes=-30.0)
        self.assertGreater(f_zero.context_switches_per_hour, 0)
        self.assertGreater(f_neg.context_switches_per_hour, 0)

    def test_zero_duration_meeting_ignored_in_intervals(self) -> None:
        events = [
            Event(timestamp=0.0, kind=MEETING, duration_minutes=0.0),
            Event(timestamp=10.0, kind=CONTEXT_SWITCH),
        ]
        f = compute_features(events, window_minutes=60.0)
        self.assertEqual(f.meeting_ratio, 0.0)
        self.assertEqual(f.multitasking_index, 0.0)

    def test_rapid_fire_switches_increase_multitasking(self) -> None:
        """Switches within 2 minutes of each other signal frantic switching."""
        events = [
            Event(timestamp=0.0, kind=CONTEXT_SWITCH),
            Event(timestamp=60.0, kind=CONTEXT_SWITCH),  # 60s -> rapid
            Event(timestamp=100.0, kind=CONTEXT_SWITCH),  # 40s -> rapid
            Event(timestamp=7200.0, kind=CONTEXT_SWITCH),  # 2h -> not rapid
        ]
        f = compute_features(events, window_minutes=120.0)
        # 2 rapid pairs out of 3 gaps -> rapid_ratio = 2/3
        # No meetings -> meeting_multitask = 0
        # multitasking = max(0, 2/3) = 2/3
        self.assertAlmostEqual(f.multitasking_index, 2 / 3, places=2)

    def test_rapid_fire_dominates_over_meeting_overlap(self) -> None:
        """When rapid-fire ratio exceeds meeting overlap, it determines the index."""
        events = [
            Event(timestamp=0.0, kind=MEETING, duration_minutes=1),  # ends at t=60
            Event(timestamp=100.0, kind=CONTEXT_SWITCH),  # outside meeting
            Event(timestamp=110.0, kind=CONTEXT_SWITCH),  # 10s -> rapid
            Event(timestamp=120.0, kind=CONTEXT_SWITCH),  # 10s -> rapid
        ]
        f = compute_features(events, window_minutes=60.0)
        # All switches outside meeting, rapid pairs yield ratio 1.0
        self.assertAlmostEqual(f.multitasking_index, 1.0, places=2)

    def test_ai_source_markers(self) -> None:
        self.assertTrue(_is_ai_source("ai_assistant"))
        self.assertTrue(_is_ai_source("GitHub Copilot"))
        self.assertTrue(_is_ai_source("agent"))
        self.assertFalse(_is_ai_source("slack"))
        self.assertFalse(_is_ai_source("email"))
        self.assertFalse(_is_ai_source(""))

    def test_ai_notification_rate(self) -> None:
        events = [
            Event(timestamp=0.0, kind=NOTIFICATION, meta={"source": "slack"}),
            Event(timestamp=60.0, kind=NOTIFICATION, meta={"source": "ai_assistant"}),
            Event(timestamp=120.0, kind=NOTIFICATION, meta={"source": "ai_assistant"}),
            Event(timestamp=180.0, kind=NOTIFICATION, meta={"source": "email"}),
        ]
        f = compute_features(events, window_minutes=60.0)
        self.assertAlmostEqual(f.notification_rate, 4.0)
        self.assertAlmostEqual(f.ai_notification_rate, 2.0)

    def test_ai_notification_rate_zero_without_notifications(self) -> None:
        f = compute_features([], window_minutes=60.0)
        self.assertEqual(f.ai_notification_rate, 0.0)

    def test_ai_notification_rate_ignores_non_notification_events(self) -> None:
        events = [
            Event(timestamp=0.0, kind=MEETING, duration_minutes=30.0, meta={"source": "ai"}),
            Event(timestamp=100.0, kind=CONTEXT_SWITCH, meta={"source": "copilot"}),
        ]
        f = compute_features(events, window_minutes=60.0)
        self.assertEqual(f.ai_notification_rate, 0.0)


if __name__ == "__main__":
    unittest.main()
