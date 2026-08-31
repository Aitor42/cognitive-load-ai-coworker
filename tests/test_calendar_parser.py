"""Comprehensive unit tests for loadguard.calendar_parser."""

from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime, UTC
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loadguard.calendar_parser import (
    _absence_end,
    _classify_vevents,
    _event_occurrences,
    _expand_rrule,
    _is_absence,
    _parse_duration,
    _parse_ics_date_or_datetime,
    _parse_ics_datetime,
    _parse_iso,
    _unfold,
    parse_absences,
    parse_calendar,
    parse_calendar_text,
    parse_ics,
)
from loadguard.models import LEAVE, VACATION


class TestCalendarParser(unittest.TestCase):
    def test_parse_ics_datetime_variants(self):
        # UTC with Z
        ts = _parse_ics_datetime("20260817T090000Z")
        self.assertEqual(ts, datetime(2026, 8, 17, 9, 0, tzinfo=UTC).timestamp())

        # With explicit offset
        ts2 = _parse_ics_datetime("20260817T090000+0200")
        self.assertEqual(ts2, datetime(2026, 8, 17, 7, 0, tzinfo=UTC).timestamp())

        # With IANA timezone
        ts3 = _parse_ics_datetime("20260817T090000", "Europe/Madrid")
        self.assertEqual(
            ts3, datetime(2026, 8, 17, 9, 0, tzinfo=ZoneInfo("Europe/Madrid")).timestamp()
        )

        # With Windows timezone
        ts4 = _parse_ics_datetime("20260817T090000", "W. Europe Standard Time")
        self.assertEqual(
            ts4, datetime(2026, 8, 17, 9, 0, tzinfo=ZoneInfo("Europe/Berlin")).timestamp()
        )

        # With unresolvable timezone fallback
        ts5 = _parse_ics_datetime("20260817T090000", "NonExistentZone/Unknown")
        self.assertEqual(ts5, datetime(2026, 8, 17, 9, 0, tzinfo=UTC).timestamp())

        # Invalid datetime format raises ValueError
        with self.assertRaises(ValueError):
            _parse_ics_datetime("not-a-datetime")

        # Without tzid
        self.assertEqual(
            _parse_ics_datetime("20260817T090000", None),
            datetime(2026, 8, 17, 9, 0, tzinfo=UTC).timestamp(),
        )

    def test_parse_ics_date_or_datetime(self):
        self.assertEqual(
            _parse_ics_date_or_datetime("20260817"),
            datetime(2026, 8, 17, 0, 0, tzinfo=UTC).timestamp(),
        )
        self.assertEqual(
            _parse_ics_date_or_datetime("20260817T090000Z"),
            datetime(2026, 8, 17, 9, 0, tzinfo=UTC).timestamp(),
        )

    def test_parse_iso(self):
        self.assertEqual(
            _parse_iso("2026-08-17T09:00:00Z"),
            datetime(2026, 8, 17, 9, 0, tzinfo=UTC).timestamp(),
        )
        self.assertEqual(
            _parse_iso("2026-08-17T09:00:00+02:00"),
            datetime(2026, 8, 17, 7, 0, tzinfo=UTC).timestamp(),
        )
        self.assertEqual(
            _parse_iso("2026-08-17T09:00:00"),
            datetime(2026, 8, 17, 9, 0, tzinfo=UTC).timestamp(),
        )

    def test_parse_duration(self):
        self.assertEqual(_parse_duration("PT1H30M15S"), 90.25)
        self.assertEqual(_parse_duration("P2D"), 2880.0)
        self.assertEqual(_parse_duration("invalid"), 60.0)
        self.assertEqual(_parse_duration(""), 60.0)
        self.assertEqual(_parse_duration("PT"), 60.0)

    def test_unfold(self):
        text = "SUMMARY:Line one\n and two\n and three\nDESCRIPTION:Foo"
        unfolded = _unfold(text)
        self.assertIn("SUMMARY:Line oneand twoand three", unfolded)

    def test_vevents_and_classify(self):
        ics = (
            "BEGIN:VCALENDAR\n"
            "BEGIN:VEVENT\n"
            "\n"
            "INVALID_LINE_NO_COLON\n"
            "UID:1\n"
            "DTSTART:20260817T090000Z\n"
            "DTEND:20260817T100000Z\n"
            "SUMMARY:Team Standup\n"
            "END:VEVENT\n"
            "BEGIN:VEVENT\n"
            "UID:2\n"
            "DTSTART;VALUE=DATE:20260818\n"
            "DTEND;VALUE=DATE:20260819\n"
            "SUMMARY:All-Day Workshop\n"
            "END:VEVENT\n"
            "BEGIN:VEVENT\n"
            "UID:3\n"
            "DTSTART:20260819T090000Z\n"
            "DURATION:PT2H\n"
            "SUMMARY:OOO - Doctor Appointment\n"
            "END:VEVENT\n"
            "BEGIN:VEVENT\n"
            "UID:4\n"
            "SUMMARY:Event without DTSTART\n"
            "END:VEVENT\n"
            "BEGIN:VEVENT\n"
            "UID:5\n"
            "DTSTART:20260820T090000Z\n"
            "SUMMARY:Vacation\n"
            "END:VEVENT\n"
            "END:VCALENDAR\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".ics", delete=False) as f:
            f.write(ics)
            path = Path(f.name)
        try:
            events, absences = parse_calendar(path)
            self.assertEqual(len(events), 2)  # Standup + Workshop
            self.assertEqual(len(absences), 2)  # OOO + Vacation
            self.assertEqual(events[1].duration_minutes, 480.0)  # All-day workshop
            self.assertEqual(absences[1].kind, VACATION)
            self.assertEqual(absences[0].kind, LEAVE)

            # Test text parsing
            ev2, abs2 = parse_calendar_text(ics)
            self.assertEqual(len(ev2), 2)
            self.assertEqual(len(abs2), 2)

            ev3 = parse_ics(path)
            self.assertEqual(len(ev3), 2)

            abs3 = parse_absences(path)
            self.assertEqual(len(abs3), 2)
        finally:
            path.unlink()

    def test_absence_end_fallbacks(self):
        props = {"SUMMARY": "Leave"}
        start = 1000.0
        # No DTEND, not all-day, no DURATION -> default + 3600
        end = _absence_end(props, start, all_day=False)
        self.assertEqual(end, 4600.0)

        # With DURATION
        props_dur = {"SUMMARY": "Leave", "DURATION": "PT30M"}
        end_dur = _absence_end(props_dur, start, all_day=False)
        self.assertEqual(end_dur, 1000.0 + 1800.0)

        # All-day without DTEND
        end_allday = _absence_end(props, start, all_day=True)
        self.assertEqual(end_allday, start + 86400.0 - 1.0)

    def test_rrule_expansion_edge_cases(self):
        start = datetime(2026, 1, 1, 9, 0, tzinfo=UTC).timestamp()

        # Unknown freq
        self.assertEqual(_expand_rrule(start, {"FREQ": "MINUTELY"}), [start])

        # Invalid interval / count
        res = _expand_rrule(start, {"FREQ": "DAILY", "INTERVAL": "bad", "COUNT": "bad"})
        self.assertEqual(len(res), 366)

        # Invalid BYDAY falls back
        res_byday = _expand_rrule(start, {"FREQ": "WEEKLY", "BYDAY": "XX,YY", "COUNT": "2"})
        self.assertEqual(len(res_byday), 2)

        # DAILY with until
        until = start + 2 * 86400.0
        res_daily_until = _expand_rrule(start, {"FREQ": "DAILY"}, until_epoch=until)
        self.assertEqual(len(res_daily_until), 3)

        # WEEKLY without byday with until
        res_weekly_until = _expand_rrule(
            start, {"FREQ": "WEEKLY"}, until_epoch=start + 14 * 86400.0
        )
        self.assertEqual(len(res_weekly_until), 3)

        # WEEKLY with byday and until
        res_byday_until = _expand_rrule(
            start, {"FREQ": "WEEKLY", "BYDAY": "TH,FR"}, until_epoch=start + 1 * 86400.0
        )
        self.assertEqual(len(res_byday_until), 2)

        # MONTHLY with until
        res_monthly_until = _expand_rrule(
            start, {"FREQ": "MONTHLY"}, until_epoch=start + 40 * 86400.0
        )
        self.assertEqual(len(res_monthly_until), 2)

        # YEARLY with until
        res_yearly_until = _expand_rrule(
            start, {"FREQ": "YEARLY"}, until_epoch=start + 400 * 86400.0
        )
        self.assertEqual(len(res_yearly_until), 2)

        # YEARLY leap day Feb 29
        leap_start = datetime(2024, 2, 29, 9, 0, tzinfo=UTC).timestamp()
        res_leap = _expand_rrule(leap_start, {"FREQ": "YEARLY", "COUNT": "3"})
        self.assertEqual(len(res_leap), 3)

    def test_classify_vevents_malformed_event_skipped(self):
        # Malformed props that trigger exception in occurrences
        vevents = [
            ({"DTSTART": "invalid-datetime", "SUMMARY": "Crash"}, {}, False),
            ({"DTSTART": "20260817T090000Z", "SUMMARY": "Valid"}, {}, False),
        ]
        events, absences = _classify_vevents(vevents)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].meta["title"], "Valid")

    def test_event_occurrences_datetime_until(self):
        props = {
            "DTSTART": "20260817T090000Z",
            "DTEND": "20260817T100000Z",
            "RRULE": "FREQ=DAILY;UNTIL=20260819T180000Z",
        }
        occurrences = _event_occurrences(props, {}, False)
        self.assertEqual(len(occurrences), 3)

    def test_vevents_repeated_properties_and_is_absence(self):
        ics = (
            "BEGIN:VCALENDAR\n"
            "BEGIN:VEVENT\n"
            "UID:ex1\n"
            "DTSTART:20260817T090000Z\n"
            "DTEND:20260817T100000Z\n"
            "RRULE:FREQ=DAILY;COUNT=5\n"
            "EXDATE:20260818T090000Z\n"
            "EXDATE:20260819T090000Z\n"
            "SUMMARY:out of office - conference\n"
            "END:VEVENT\n"
            "END:VCALENDAR\n"
        )
        events, absences = parse_calendar_text(ics)
        self.assertEqual(len(absences), 3)

    def test_is_absence_phrases(self):
        self.assertTrue(_is_absence({"SUMMARY": "I am on leave today"}))
        self.assertTrue(_is_absence({"SUMMARY": "Status: [ooo]"}))
        self.assertTrue(_is_absence({"SUMMARY": "vacation - road trip"}))
        self.assertTrue(_is_absence({"SUMMARY": "baja médica"}))
        self.assertFalse(_is_absence({"SUMMARY": "Permiso de obras reunión"}))
        self.assertFalse(_is_absence({"SUMMARY": "vacationing with family meeting"}))

    def test_unfold_with_tabs_and_spaces(self):
        text = "SUMMARY:Team Sync \n\twith Architecture \n and Product"
        self.assertEqual(_unfold(text), "SUMMARY:Team Sync with Architecture and Product")

    def test_classify_vevents_unescapes_summary(self):
        ics = (
            "BEGIN:VCALENDAR\n"
            "BEGIN:VEVENT\n"
            "UID:esc1\n"
            "DTSTART:20260817T090000Z\n"
            "DTEND:20260817T100000Z\n"
            "SUMMARY:Review\\, Architecture\\; Planning\\nSession\n"
            "END:VEVENT\n"
            "END:VCALENDAR\n"
        )
        events, _ = parse_calendar_text(ics)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].meta["title"], "Review, Architecture; Planning Session")


if __name__ == "__main__":
    unittest.main()
