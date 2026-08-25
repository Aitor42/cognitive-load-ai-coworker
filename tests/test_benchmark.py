"""Tests for the benchmark module and the signal-capture script."""

from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import capture_signals  # noqa: E402
from loadguard.benchmark import (  # noqa: E402
    _estimate_eliminated,
    _pct_reduction,
    features_window,
    run_benchmark,
    run_pilot_evaluation,
)
from loadguard.models import (  # noqa: E402
    FOCUS_BLOCK,
    LEAVE,
    NOTIFICATION,
    VACATION,
    Absence,
    Event,
    FeatureSet,
    Worker,
)
from loadguard.sample_data import sample_tasks  # noqa: E402
from loadguard.signals import load_events  # noqa: E402

SAMPLE = ROOT / "demo" / "sample_events.jsonl"


class TestBenchmark(unittest.TestCase):
    def test_metrics_are_consistent(self):
        b = run_benchmark(load_events(SAMPLE), sample_tasks())
        self.assertGreater(b.n_events, 0)
        self.assertGreaterEqual(b.reduction_points, 0.0)
        self.assertLessEqual(b.after_score, b.before_score)
        self.assertTrue(b.plan_counts)
        self.assertGreaterEqual(b.reduction_pct, 0.0)
        # signal counts sum to total events
        self.assertEqual(sum(b.signal_counts.values()), b.n_events)

    def test_reduction_pct_formula(self):
        b = run_benchmark(load_events(SAMPLE), sample_tasks())
        expected = round((b.reduction_points / b.before_score * 100), 1)
        self.assertAlmostEqual(b.reduction_pct, expected, places=1)

    def test_unknown_event_kind_not_counted(self):
        from types import SimpleNamespace

        bogus = SimpleNamespace(timestamp=0.0, kind="bogus", duration_minutes=0.0)
        b = run_benchmark([bogus], sample_tasks())
        self.assertEqual(sum(b.signal_counts.values()), 0)


class TestPilotEvaluation(unittest.TestCase):
    def test_no_outcome_is_honest_projection(self):
        e = run_pilot_evaluation(load_events(SAMPLE), sample_tasks())
        self.assertFalse(e.has_observed)
        self.assertIsNone(e.observed)
        self.assertIsNone(e.notification_reduction_pct)
        self.assertIn("projection", e.summary)

    def test_observed_outcome_measured(self):
        baseline = load_events(SAMPLE)
        # Outcome: same meetings, but focus protected and notifications reduced.
        outcome = [
            Event(timestamp=0.0, kind="meeting", duration_minutes=60.0),
            Event(timestamp=3600.0, kind="meeting", duration_minutes=30.0),
            Event(timestamp=7200.0, kind="meeting", duration_minutes=45.0),
            Event(timestamp=9000.0, kind=FOCUS_BLOCK, duration_minutes=60.0),
        ]
        outcome += [Event(timestamp=9060.0 + i, kind=NOTIFICATION) for i in range(0, 600, 120)]
        e = run_pilot_evaluation(baseline, sample_tasks(), outcome_events=outcome)
        self.assertTrue(e.has_observed)
        self.assertIsNotNone(e.observed)
        self.assertIsNotNone(e.load_delta)
        self.assertIsNotNone(e.focus_minutes_gained)
        self.assertGreaterEqual(e.focus_minutes_gained, 0)
        self.assertGreaterEqual(e.observed.score, 0)

    def test_acceptance_rate_passthrough(self):
        e = run_pilot_evaluation(load_events(SAMPLE), sample_tasks(), accepted_recommendations=0.78)
        self.assertEqual(e.acceptance_rate, 0.78)

    def test_observed_notification_reduction_reported(self):
        baseline = [
            Event(timestamp=0.0, kind=FOCUS_BLOCK, duration_minutes=60.0),
            Event(timestamp=100.0, kind=NOTIFICATION),
            Event(timestamp=200.0, kind=NOTIFICATION),
        ]
        outcome = [
            Event(timestamp=0.0, kind=FOCUS_BLOCK, duration_minutes=60.0),
            Event(timestamp=300.0, kind=NOTIFICATION),
        ]
        e = run_pilot_evaluation(baseline, sample_tasks(), outcome_events=outcome)
        self.assertTrue(e.has_observed)
        self.assertIsNotNone(e.notification_reduction_pct)

    def test_observed_without_baseline_focus_blocks(self):
        baseline = [Event(timestamp=0.0, kind=NOTIFICATION)]
        outcome = [Event(timestamp=0.0, kind=FOCUS_BLOCK, duration_minutes=30.0)]
        e = run_pilot_evaluation(baseline, sample_tasks(), outcome_events=outcome)
        self.assertTrue(e.has_observed)
        self.assertIsNone(e.notification_reduction_pct)


class TestHelpers(unittest.TestCase):
    def test_features_window(self):
        self.assertEqual(features_window([], None), 60.0)
        self.assertEqual(features_window([], 30.0), 30.0)
        events = [Event(timestamp=0.0, kind="meeting"), Event(timestamp=7200.0, kind="meeting")]
        self.assertEqual(features_window(events, None), 120.0)

    def test_estimate_eliminated(self):
        f = FeatureSet(notification_rate=10.0, context_switches_per_hour=5.0)
        self.assertEqual(_estimate_eliminated(f, {"batch": 1, "delegate": 1}, 2.0), 11)
        self.assertEqual(_estimate_eliminated(f, {}, 2.0), 0)

    def test_pct_reduction(self):
        self.assertEqual(_pct_reduction(10.0, 5.0), 50.0)
        self.assertIsNone(_pct_reduction(None, 5.0))
        self.assertIsNone(_pct_reduction(10.0, None))
        self.assertIsNone(_pct_reduction(0.0, 5.0))


class TestCapture(unittest.TestCase):
    def test_parse_ics(self):
        events = capture_signals.parse_ics(ROOT / "scripts" / "sample_calendar.ics")
        self.assertEqual(len(events), 3)
        self.assertTrue(all(e.kind == "meeting" for e in events))
        # DTSTART + DTEND -> 60 minutes
        self.assertEqual(events[0].duration_minutes, 60.0)
        # DTSTART + DURATION:PT45M -> 45 minutes
        self.assertEqual(events[2].duration_minutes, 45.0)

    def test_parse_calendar_text(self) -> None:
        """parse_calendar_text parses ICS text without file I/O."""
        ics = (
            "BEGIN:VCALENDAR\nVERSION:2.0\n"
            "BEGIN:VEVENT\nUID:m1\nDTSTART:20260817T090000Z\nDTEND:20260817T100000Z\n"
            "SUMMARY:Standup\nEND:VEVENT\n"
            "BEGIN:VEVENT\nUID:a1\nDTSTART;VALUE=DATE:20260817\nDTEND;VALUE=DATE:20260818\n"
            "SUMMARY:Out of office\nEND:VEVENT\n"
            "END:VCALENDAR\n"
        )
        events, absences = capture_signals.parse_calendar_text(ics)
        self.assertEqual(len(events), 1)
        self.assertEqual(len(absences), 1)
        self.assertEqual(events[0].meta["title"], "Standup")
        self.assertEqual(absences[0].kind, LEAVE)

    def test_parse_calendar_matches_individual_parsers(self) -> None:
        """parse_calendar returns the same split as parse_ics + parse_absences."""
        ics = (
            "BEGIN:VCALENDAR\nVERSION:2.0\n"
            "BEGIN:VEVENT\nUID:m1\nDTSTART:20260817T090000Z\nDTEND:20260817T100000Z\n"
            "SUMMARY:Standup\nEND:VEVENT\n"
            "BEGIN:VEVENT\nUID:a1\nDTSTART;VALUE=DATE:20260817\nDTEND;VALUE=DATE:20260818\n"
            "SUMMARY:Out of office\nEND:VEVENT\n"
            "END:VCALENDAR\n"
        )
        path = self._write_ics(ics)
        try:
            events, absences = capture_signals.parse_calendar(path)
            self.assertEqual(events, capture_signals.parse_ics(path))
            self.assertEqual(absences, capture_signals.parse_absences(path))
            self.assertEqual(len(events), 1)
            self.assertEqual(len(absences), 1)
            self.assertEqual(events[0].meta["title"], "Standup")
            self.assertEqual(absences[0].kind, LEAVE)
        finally:
            path.unlink()

    def test_parse_calendar_reads_file_once(self) -> None:
        """parse_calendar must read the calendar a single time."""
        ics = (
            "BEGIN:VCALENDAR\nVERSION:2.0\n"
            "BEGIN:VEVENT\nUID:m1\nDTSTART:20260817T090000Z\nDTEND:20260817T100000Z\n"
            "SUMMARY:Standup\nEND:VEVENT\n"
            "BEGIN:VEVENT\nUID:a1\nDTSTART;VALUE=DATE:20260817\nDTEND;VALUE=DATE:20260818\n"
            "SUMMARY:Out of office\nEND:VEVENT\n"
            "END:VCALENDAR\n"
        )
        path = self._write_ics(ics)
        try:
            original = capture_signals._vevents
            calls = 0

            def counting_vevents(p: Path):
                nonlocal calls
                calls += 1
                return original(p)

            with unittest.mock.patch("capture_signals._vevents", counting_vevents):
                events, absences = capture_signals.parse_calendar(path)
            self.assertEqual(calls, 1)
            self.assertEqual(len(events), 1)
            self.assertEqual(len(absences), 1)
        finally:
            path.unlink()

    def test_parse_notifications(self):
        with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as fh:
            fh.write("2026-08-17T09:02:00Z slack\n")
            fh.write("2026-08-17T09:08:00Z ai_assistant\n")
            path = Path(fh.name)
        try:
            events = capture_signals.parse_notifications(path)
            self.assertEqual(len(events), 2)
            self.assertEqual(events[0].meta["source"], "slack")
            self.assertEqual(events[1].kind, "notification")
        finally:
            path.unlink()

    def test_parse_duration_full(self):
        """PT1H30M should be 90 minutes."""
        self.assertAlmostEqual(capture_signals._parse_duration("PT1H30M"), 90.0)

    def test_parse_duration_minutes_only(self):
        """PT45M should be 45 minutes."""
        self.assertAlmostEqual(capture_signals._parse_duration("PT45M"), 45.0)

    def test_parse_duration_with_days(self) -> None:
        """P1D and P1DT2H30M should expand days into minutes."""
        self.assertAlmostEqual(capture_signals._parse_duration("P1D"), 1440.0)
        self.assertAlmostEqual(capture_signals._parse_duration("P1DT2H30M"), 1590.0)

    def test_parse_duration_empty_pt(self):
        """PT with no components should fall back to 60 minutes."""
        self.assertAlmostEqual(capture_signals._parse_duration("PT"), 60.0)

    def test_parse_duration_invalid(self):
        """Non-duration string should fall back to 60 minutes."""
        self.assertAlmostEqual(capture_signals._parse_duration("invalid"), 60.0)

    def test_parse_focus(self):
        """Focus log parsing should produce focus_block events."""
        with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as fh:
            fh.write("2026-08-17T10:00:00Z 25 Deep work\n")
            fh.write("2026-08-17T11:00:00Z 30\n")
            path = Path(fh.name)
        try:
            events = capture_signals.parse_focus(path)
            self.assertEqual(len(events), 2)
            self.assertEqual(events[0].kind, "focus_block")
            self.assertAlmostEqual(events[0].duration_minutes, 25.0)
            self.assertAlmostEqual(events[1].duration_minutes, 30.0)
        finally:
            path.unlink()

    def _write_ics(self, text: str) -> Path:
        tmp = tempfile.NamedTemporaryFile("w", suffix=".ics", delete=False)
        tmp.write(text)
        tmp.close()
        return Path(tmp.name)

    def test_parse_ics_date(self) -> None:
        ts = capture_signals._parse_ics_date("20260817")
        self.assertEqual(ts, capture_signals._parse_ics_datetime("20260817T000000Z"))

    def test_parse_ics_date_rejects_invalid(self) -> None:
        # Extra characters are no longer silently sliced away.
        with self.assertRaises(ValueError):
            capture_signals._parse_ics_date("20260817extra")

    def test_parse_ics_datetime_with_tzid(self) -> None:
        from zoneinfo import ZoneInfo

        expected = datetime(2026, 8, 17, 9, 0, tzinfo=ZoneInfo("Europe/Madrid")).timestamp()
        self.assertEqual(
            capture_signals._parse_ics_datetime("20260817T090000", "Europe/Madrid"), expected
        )

    def test_parse_ics_datetime_with_numeric_offset(self) -> None:
        self.assertEqual(
            capture_signals._parse_ics_datetime("20260817T090000+0200"),
            datetime(2026, 8, 17, 7, 0, tzinfo=timezone.utc).timestamp(),
        )

    def test_parse_ics_datetime_unknown_tzid_falls_back_to_utc(self) -> None:
        self.assertEqual(
            capture_signals._parse_ics_datetime("20260817T090000", "Bogus Time Zone"),
            datetime(2026, 8, 17, 9, 0, tzinfo=timezone.utc).timestamp(),
        )

    def test_parse_ics_datetime_windows_tzid(self) -> None:
        from zoneinfo import ZoneInfo

        expected = datetime(2026, 8, 17, 9, 0, tzinfo=ZoneInfo("Europe/Berlin")).timestamp()
        self.assertEqual(
            capture_signals._parse_ics_datetime("20260817T090000", "W. Europe Standard Time"),
            expected,
        )

    def test_windows_tz_map_values_resolve(self) -> None:
        """Every mapped Windows timezone must resolve via zoneinfo."""
        from zoneinfo import ZoneInfo

        for windows_name, iana in capture_signals._WINDOWS_TZ_MAP.items():
            with self.subTest(windows_name=windows_name, iana=iana):
                self.assertIsNotNone(ZoneInfo(iana))

    def test_parse_ics_datetime_malformed_raises(self) -> None:
        with self.assertRaises(ValueError):
            capture_signals._parse_ics_datetime("2026-08-17")

    def test_parse_ics_respects_tzid(self) -> None:
        from zoneinfo import ZoneInfo

        ics = (
            "BEGIN:VCALENDAR\nVERSION:2.0\n"
            "BEGIN:VEVENT\nUID:t1\nDTSTART;TZID=Europe/Madrid:20260817T090000\n"
            "DTEND;TZID=Europe/Madrid:20260817T100000\nSUMMARY:Standup\nEND:VEVENT\n"
            "END:VCALENDAR\n"
        )
        path = self._write_ics(ics)
        try:
            events = capture_signals.parse_ics(path)
            self.assertEqual(len(events), 1)
            self.assertEqual(
                events[0].timestamp,
                datetime(2026, 8, 17, 9, 0, tzinfo=ZoneInfo("Europe/Madrid")).timestamp(),
            )
            self.assertEqual(events[0].duration_minutes, 60.0)
        finally:
            path.unlink()

    def test_parse_ics_respects_windows_tzid(self) -> None:
        from zoneinfo import ZoneInfo

        ics = (
            "BEGIN:VCALENDAR\nVERSION:2.0\n"
            "BEGIN:VEVENT\nUID:t1\nDTSTART;TZID=W. Europe Standard Time:20260817T090000\n"
            "DTEND;TZID=W. Europe Standard Time:20260817T100000\nSUMMARY:Standup\nEND:VEVENT\n"
            "END:VCALENDAR\n"
        )
        path = self._write_ics(ics)
        try:
            events = capture_signals.parse_ics(path)
            self.assertEqual(len(events), 1)
            self.assertEqual(
                events[0].timestamp,
                datetime(2026, 8, 17, 9, 0, tzinfo=ZoneInfo("Europe/Berlin")).timestamp(),
            )
            self.assertEqual(events[0].duration_minutes, 60.0)
        finally:
            path.unlink()

    def test_parse_ics_expands_daily_recurrence(self) -> None:
        ics = (
            "BEGIN:VCALENDAR\nVERSION:2.0\n"
            "BEGIN:VEVENT\nUID:r1\nDTSTART:20260817T090000Z\nDTEND:20260817T100000Z\n"
            "SUMMARY:Daily sync\nRRULE:FREQ=DAILY;COUNT=3\nEND:VEVENT\n"
            "END:VCALENDAR\n"
        )
        path = self._write_ics(ics)
        try:
            events = capture_signals.parse_ics(path)
            self.assertEqual(len(events), 3)
            days = [datetime.fromtimestamp(e.timestamp, tz=timezone.utc).day for e in events]
            self.assertEqual(days, [17, 18, 19])
            self.assertTrue(all(e.duration_minutes == 60.0 for e in events))
        finally:
            path.unlink()

    def test_expand_rrule_weekly_byday_skips_days_before_start(self) -> None:
        start = datetime(2026, 8, 19, 9, 0, tzinfo=timezone.utc).timestamp()  # Wednesday
        starts = capture_signals._expand_rrule(
            start, {"FREQ": "WEEKLY", "BYDAY": "MO,WE", "COUNT": "4"}
        )
        days = [datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat() for ts in starts]
        # The Monday before the Wednesday start is skipped; then MO/WE recur.
        self.assertEqual(days, ["2026-08-19", "2026-08-24", "2026-08-26", "2026-08-31"])

    def test_expand_rrule_weekly_interval(self) -> None:
        start = datetime(2026, 8, 17, 9, 0, tzinfo=timezone.utc).timestamp()  # Monday
        starts = capture_signals._expand_rrule(
            start, {"FREQ": "WEEKLY", "INTERVAL": "2", "COUNT": "3"}
        )
        days = [datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat() for ts in starts]
        self.assertEqual(days, ["2026-08-17", "2026-08-31", "2026-09-14"])

    def test_event_occurrences_until_date_is_inclusive(self) -> None:
        props = {
            "DTSTART": "20260817T090000Z",
            "DTEND": "20260817T100000Z",
            "RRULE": "FREQ=DAILY;UNTIL=20260820",
        }
        occurrences = capture_signals._event_occurrences(props, {}, False)
        self.assertEqual(len(occurrences), 4)  # 17, 18, 19, 20 Aug inclusive

    def test_event_occurrences_exdate_excludes_instances(self) -> None:
        props = {
            "DTSTART": "20260817T090000Z",
            "DTEND": "20260817T100000Z",
            "RRULE": "FREQ=DAILY;COUNT=3",
            "EXDATE": "20260818T090000Z",
        }
        occurrences = capture_signals._event_occurrences(props, {}, False)
        self.assertEqual(len(occurrences), 2)

    def test_expand_rrule_open_ended_is_capped(self) -> None:
        start = datetime(2026, 8, 17, 9, 0, tzinfo=timezone.utc).timestamp()
        starts = capture_signals._expand_rrule(start, {"FREQ": "DAILY"})
        self.assertEqual(len(starts), capture_signals.MAX_RRULE_OCCURRENCES)

    def test_expand_rrule_monthly_clamps_short_months(self) -> None:
        start = datetime(2026, 1, 31, 9, 0, tzinfo=timezone.utc).timestamp()
        starts = capture_signals._expand_rrule(start, {"FREQ": "MONTHLY", "COUNT": "3"})
        days = [datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat() for ts in starts]
        self.assertEqual(days, ["2026-01-31", "2026-02-28", "2026-03-31"])

    def test_expand_rrule_yearly_clamps_leap_day(self) -> None:
        start = datetime(2024, 2, 29, 9, 0, tzinfo=timezone.utc).timestamp()
        starts = capture_signals._expand_rrule(start, {"FREQ": "YEARLY", "COUNT": "2"})
        days = [datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat() for ts in starts]
        self.assertEqual(days, ["2024-02-29", "2025-02-28"])

    def test_expand_rrule_unknown_freq_returns_start_only(self) -> None:
        start = 1_700_000_000.0
        starts = capture_signals._expand_rrule(start, {"FREQ": "HOURLY", "COUNT": "5"})
        self.assertEqual(starts, [start])

    def test_parse_absences_expands_recurrence(self) -> None:
        ics = (
            "BEGIN:VCALENDAR\nVERSION:2.0\n"
            "BEGIN:VEVENT\nUID:a1\nDTSTART;VALUE=DATE:20260817\nDTEND;VALUE=DATE:20260818\n"
            "SUMMARY:Out of office\nRRULE:FREQ=WEEKLY;COUNT=2\nEND:VEVENT\n"
            "END:VCALENDAR\n"
        )
        path = self._write_ics(ics)
        try:
            absences = capture_signals.parse_absences(path)
            self.assertEqual(len(absences), 2)
            self.assertEqual(absences[1].start - absences[0].start, 7 * 86400.0)
        finally:
            path.unlink()

    def test_is_absence_by_busy_status_and_summary(self) -> None:
        self.assertTrue(capture_signals._is_absence({"X-MICROSOFT-CDO-BUSYSTATUS": "OOF"}))
        self.assertTrue(capture_signals._is_absence({"SUMMARY": "Out of office"}))
        self.assertFalse(capture_signals._is_absence({"SUMMARY": "Standup"}))

    def test_absence_kind_vacation_vs_leave(self) -> None:
        self.assertEqual(capture_signals._absence_kind({"SUMMARY": "Vacation"}), VACATION)
        self.assertEqual(capture_signals._absence_kind({"SUMMARY": "Out of office"}), LEAVE)

    def test_absence_end_all_day_exclusive(self) -> None:
        start = capture_signals._parse_ics_date("20260817")
        end = capture_signals._absence_end({"DTEND": "20260820"}, start, True)
        self.assertEqual(end, capture_signals._parse_ics_date("20260820") - 1.0)

    def test_absence_end_all_day_with_duration_is_full_day(self) -> None:
        """An all-day event spans the whole day even with a DURATION."""
        start = capture_signals._parse_ics_date("20260817")
        end = capture_signals._absence_end({"DURATION": "PT8H"}, start, True)
        self.assertEqual(end, start + 86400.0 - 1.0)

    def test_transparent_all_day_event_is_absence(self) -> None:
        self.assertTrue(capture_signals._is_absence({"TRANSP": "TRANSPARENT"}, all_day=True))

    def test_timed_transparent_event_is_not_absence(self) -> None:
        self.assertFalse(capture_signals._is_absence({"TRANSP": "TRANSPARENT"}))

    def test_parse_absences_detects_transparent_all_day_events(self) -> None:
        """Google-style OOO: all-day + TRANSPARENT with a generic summary."""
        ics = (
            "BEGIN:VCALENDAR\nVERSION:2.0\n"
            "BEGIN:VEVENT\nUID:a1\nDTSTART;VALUE=DATE:20260817\nDTEND;VALUE=DATE:20260818\n"
            "SUMMARY:Team offsite\nTRANSP:TRANSPARENT\nEND:VEVENT\n"
            "END:VCALENDAR\n"
        )
        path = self._write_ics(ics)
        try:
            absences = capture_signals.parse_absences(path)
            self.assertEqual(len(absences), 1)
            self.assertEqual(absences[0].kind, LEAVE)
        finally:
            path.unlink()

    def test_parse_ics_skips_malformed_events(self) -> None:
        """One broken VEVENT must not abort the rest of the calendar."""
        import contextlib
        import io

        ics = (
            "BEGIN:VCALENDAR\nVERSION:2.0\n"
            "BEGIN:VEVENT\nUID:g1\nDTSTART:20260817T090000Z\nDTEND:20260817T100000Z\n"
            "SUMMARY:Standup\nEND:VEVENT\n"
            "BEGIN:VEVENT\nUID:b1\nDTSTART:notadate\nDTEND:20260817T100000Z\n"
            "SUMMARY:Broken\nEND:VEVENT\n"
            "END:VCALENDAR\n"
        )
        path = self._write_ics(ics)
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                events = capture_signals.parse_ics(path)
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].meta["title"], "Standup")
        finally:
            path.unlink()

    def test_parse_absences_skips_malformed_events(self) -> None:
        import contextlib
        import io

        ics = (
            "BEGIN:VCALENDAR\nVERSION:2.0\n"
            "BEGIN:VEVENT\nUID:a1\nDTSTART;VALUE=DATE:20260817\nDTEND;VALUE=DATE:20260818\n"
            "SUMMARY:Out of office\nEND:VEVENT\n"
            "BEGIN:VEVENT\nUID:b1\nDTSTART;VALUE=DATE:bad\nDTEND;VALUE=DATE:20260818\n"
            "SUMMARY:Vacation\nEND:VEVENT\n"
            "END:VCALENDAR\n"
        )
        path = self._write_ics(ics)
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                absences = capture_signals.parse_absences(path)
            self.assertEqual(len(absences), 1)
        finally:
            path.unlink()

    def test_parse_absences_from_ics(self) -> None:
        ics = (
            "BEGIN:VCALENDAR\nVERSION:2.0\n"
            "BEGIN:VEVENT\nUID:m1\nDTSTART:20260817T090000Z\nDTEND:20260817T100000Z\n"
            "SUMMARY:Standup\nEND:VEVENT\n"
            "BEGIN:VEVENT\nUID:a1\nDTSTART;VALUE=DATE:20260817\nDTEND;VALUE=DATE:20260820\n"
            "SUMMARY:Out of office\nEND:VEVENT\n"
            "BEGIN:VEVENT\nUID:a2\nDTSTART:20260810T090000Z\nDTEND:20260810T170000Z\n"
            "SUMMARY:Vacation\nEND:VEVENT\n"
            "BEGIN:VEVENT\nUID:a3\nDTSTART:20260811T090000Z\nDTEND:20260811T170000Z\n"
            "SUMMARY:Busy\nX-MICROSOFT-CDO-BUSYSTATUS:OOF\nEND:VEVENT\n"
            "END:VCALENDAR\n"
        )
        path = self._write_ics(ics)
        try:
            absences = capture_signals.parse_absences(path)
            self.assertEqual(len(absences), 3)
            kinds = [a.kind for a in absences]
            self.assertEqual(kinds.count(VACATION), 1)
            self.assertEqual(kinds.count(LEAVE), 2)
            for a in absences:
                self.assertGreater(a.end, a.start)
            # Absences never capture the summary text (privacy).
            self.assertTrue(all(a.note == "" for a in absences))
            # The same calendar still yields exactly one real meeting.
            self.assertEqual(len(capture_signals.parse_ics(path)), 1)
        finally:
            path.unlink()

    def test_write_and_load_absences_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "absences.jsonl"
            absences = [
                Absence(start=3.0, end=4.0, kind=VACATION),
                Absence(start=1.0, end=2.0, kind=LEAVE),
            ]
            capture_signals.write_absences(absences, path)
            loaded = capture_signals.load_absences(path)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[0].start, 1.0)  # sorted by start
            self.assertEqual(loaded[1].kind, VACATION)

    def test_load_absences_ignores_corrupt_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "absences.jsonl"
            path.write_text(
                '{"start":1.0,"end":2.0,"kind":"leave"}\nnot_json\n{"bad":true}\n',
                encoding="utf-8",
            )
            self.assertEqual(len(capture_signals.load_absences(path)), 1)

    def test_write_and_load_workers_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "workers.jsonl"
            worker = Worker(id="me", name="Ada", absences=[Absence(start=1.0, end=2.0, kind=LEAVE)])
            capture_signals.write_workers([worker], path)
            loaded = capture_signals.load_workers(path)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].id, "me")
            self.assertEqual(loaded[0].name, "Ada")
            self.assertEqual(loaded[0].absences[0].kind, LEAVE)

    def test_load_workers_ignores_corrupt_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "workers.jsonl"
            path.write_text('{"id":"me","absences":[]}\nnot_json\n', encoding="utf-8")
            self.assertEqual(len(capture_signals.load_workers(path)), 1)


if __name__ == "__main__":
    unittest.main()
