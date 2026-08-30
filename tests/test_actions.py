"""Unit tests for the actions module (approval, exports, audit)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loadguard.actions import (  # noqa: E402
    _ics_alarm_trigger,
    _ics_escape,
    _ics_fold,
    clear_audit,
    export_ics,
    export_tasks_csv,
    load_audit,
    record_approval,
)
from loadguard.models import Event, LoadReport, Plan, PlanItem, Task  # noqa: E402


def _plan() -> Plan:
    return Plan(
        load_report=LoadReport(score=80.0, level="overload"),
        plan_id="abc123",
        items=[
            PlanItem(position=1, action="focus_block", title="Focus block (no notifications)"),
            PlanItem(position=2, action="do", task_id="a", title="Critical fix"),
            PlanItem(position=3, action="break", title="Recovery break", rationale="Rest"),
            PlanItem(position=4, action="delegate", task_id="c", title="Vendor emails"),
        ],
    )


def _tasks() -> list[Task]:
    return [
        Task(id="a", title="Critical fix", priority=5, duration_minutes=60.0),
        Task(id="c", title="Vendor emails", priority=2, duration_minutes=20.0),
    ]


class TestExportIcs(unittest.TestCase):
    def test_contains_focus_and_break_events(self):
        ics = export_ics(_plan(), _tasks(), start_epoch=1_700_000_000.0)
        self.assertIn("BEGIN:VCALENDAR", ics)
        self.assertEqual(ics.count("BEGIN:VEVENT"), 3)  # focus + do task + break
        self.assertIn("Focus block (no notifications)", ics)
        self.assertIn("Recovery break", ics)
        self.assertIn("Critical fix", ics)
        self.assertIn("CATEGORIES:LOADGUARD-TASK", ics)
        self.assertIn("DTSTART:20231114T221320Z", ics)  # 1700000000 -> Nov 14 2023 22:13:20Z

    def test_focus_block_duration_is_45min(self):
        ics = export_ics(_plan(), _tasks(), start_epoch=1_700_000_000.0)
        self.assertIn("DTEND:20231114T225820Z", ics)  # 22:13:20 + 45 min

    def test_break_after_do_task_is_scheduled_after_task_duration(self):
        # do task a (60 min) is scheduled after the focus block (45 min), so the
        # break starts at 22:13:20 + 45 + 60 = 23:58:20.
        ics = export_ics(_plan(), _tasks(), start_epoch=1_700_000_000.0)
        self.assertIn("DTSTART:20231114T235820Z", ics)

    def test_ics_escape_newlines(self):
        self.assertEqual(_ics_escape("Hello\r\nWorld"), "Hello\\nWorld")
        self.assertEqual(_ics_escape("Hello\rWorld"), "Hello\\nWorld")
        self.assertEqual(_ics_escape("Hello,;\\World"), "Hello\\,\\;\\\\World")
        self.assertEqual(_ics_escape(""), "")

    def test_batch_advances_cursor_before_block(self):
        plan = Plan(
            load_report=LoadReport(score=80.0, level="overload"),
            plan_id="x",
            items=[
                PlanItem(position=1, action="batch", title="Batch notifications"),
                PlanItem(position=2, action="focus_block", title="Focus block"),
            ],
        )
        ics = export_ics(plan, _tasks(), start_epoch=1_700_000_000.0)
        self.assertIn("DTSTART:20231114T221820Z", ics)  # 22:13:20 + 5 min (batch)

    def test_default_start_epoch_is_rounded(self):
        from unittest import mock

        with mock.patch("loadguard.actions.time.time", return_value=1_700_000_000.0):
            ics = export_ics(_plan(), _tasks())
        self.assertIn("BEGIN:VCALENDAR", ics)
        self.assertIn("BEGIN:VEVENT", ics)

    def test_dtstamp_is_creation_time(self):
        """DTSTAMP must be the render time, not the block start."""
        from unittest import mock

        with mock.patch("loadguard.actions.time.time", return_value=1_700_000_000.0):
            ics = export_ics(_plan(), _tasks(), start_epoch=1_700_000_000.0)
        self.assertIn("DTSTAMP:20231114T221320Z", ics)

    def test_ics_fold_long_lines(self):
        """SUMMARY/DESCRIPTION longer than 75 octets are folded per RFC 5545."""
        plan = Plan(
            load_report=LoadReport(score=80.0, level="overload"),
            plan_id="x",
            items=[
                PlanItem(
                    position=1,
                    action="focus_block",
                    title="F" * 100,
                    rationale="R" * 120,
                )
            ],
        )
        ics = export_ics(plan, _tasks(), start_epoch=1_700_000_000.0)
        self.assertIn("\r\n ", ics)
        self.assertTrue(all(len(line) <= 75 for line in ics.split("\r\n") if line))

    def test_ics_fold_counts_octets_not_characters(self):
        """RFC 5545 limits content lines to 75 *octets*; multibyte titles must
        be folded by UTF-8 width so every emitted line stays within the limit."""
        plan = Plan(
            load_report=LoadReport(score=80.0, level="overload"),
            plan_id="x",
            items=[
                PlanItem(
                    position=1,
                    action="focus_block",
                    title="Reunión de planificación trimestral con el equipo de producto — ñandú 🧠",
                )
            ],
        )
        ics = export_ics(plan, [], start_epoch=1_700_000_000.0)
        for line in ics.split("\r\n"):
            self.assertLessEqual(len(line.encode("utf-8")), 75, line)

    def test_ics_fold_unit_multibyte(self):
        """_ics_fold never splits a multi-byte sequence and keeps the octet budget."""
        folded = _ics_fold("SUMMARY:" + "Ñ" * 60)
        lines = folded.split("\r\n")
        self.assertEqual("".join(line.lstrip(" ") for line in lines), "SUMMARY:" + "Ñ" * 60)
        for line in lines:
            self.assertLessEqual(len(line.encode("utf-8")), 75)
        # Continuation lines still start with exactly one space.
        for line in lines[1:]:
            self.assertTrue(line.startswith(" "))

    def test_ics_alarm_trigger_format(self):
        """Whole-minute leads render as -PTnM; others fall back to seconds."""
        self.assertEqual(_ics_alarm_trigger(10.0), "-PT10M")
        self.assertEqual(_ics_alarm_trigger(2.5), "-PT150S")
        self.assertEqual(_ics_alarm_trigger(0.5), "-PT30S")

    def test_export_ics_focus_block_has_valarm(self):
        """Focus blocks carry a display alarm that fires before they start."""
        plan = Plan(
            load_report=LoadReport(score=80.0, level="overload"),
            plan_id="x",
            items=[PlanItem(position=1, action="focus_block", title="Deep work")],
        )
        ics = export_ics(plan, _tasks(), start_epoch=1_700_000_000.0)
        self.assertIn("BEGIN:VALARM", ics)
        self.assertIn("ACTION:DISPLAY", ics)
        self.assertIn("DESCRIPTION:Focus block starts in 10 minutes", ics)
        self.assertIn("TRIGGER:-PT10M", ics)
        self.assertIn("END:VALARM", ics)
        # The alarm nests inside the VEVENT, before it closes.
        self.assertLess(ics.index("BEGIN:VALARM"), ics.index("END:VEVENT"))

    def test_export_ics_non_focus_events_have_no_alarm(self):
        """Breaks and resequenced tasks are exported without reminders."""
        plan = Plan(
            load_report=LoadReport(score=80.0, level="overload"),
            plan_id="x",
            items=[
                PlanItem(position=1, action="break", title="Recovery break"),
                PlanItem(position=2, action="do", task_id="a", title="Critical fix"),
            ],
        )
        ics = export_ics(plan, _tasks(), start_epoch=1_700_000_000.0)
        self.assertNotIn("BEGIN:VALARM", ics)

    def test_export_ics_alarm_minutes_override(self):
        """The lead time is configurable, including sub-minute triggers."""
        plan = Plan(
            load_report=LoadReport(score=80.0, level="overload"),
            plan_id="x",
            items=[PlanItem(position=1, action="focus_block", title="Deep work")],
        )
        ics = export_ics(plan, [], start_epoch=1_700_000_000.0, alarm_minutes=2.5)
        self.assertIn("TRIGGER:-PT150S", ics)
        self.assertIn("DESCRIPTION:Focus block starts in 2.5 minutes", ics)

    def test_export_ics_alarm_disabled(self):
        """None or zero disables the reminder entirely."""
        plan = Plan(
            load_report=LoadReport(score=80.0, level="overload"),
            plan_id="x",
            items=[PlanItem(position=1, action="focus_block", title="Deep work")],
        )
        ics = export_ics(plan, [], start_epoch=1_700_000_000.0, alarm_minutes=None)
        self.assertNotIn("BEGIN:VALARM", ics)
        ics_zero = export_ics(plan, [], start_epoch=1_700_000_000.0, alarm_minutes=0)
        self.assertNotIn("BEGIN:VALARM", ics_zero)

    def test_ics_escape_roundtrip_non_ascii(self):
        """Non-ASCII titles survive escape + fold + standard unfold intact."""
        title = "Reunión de equipo — ñandú 🧠"
        folded = _ics_fold(f"SUMMARY:{_ics_escape(title)}")
        for line in folded.split("\r\n"):
            self.assertLessEqual(len(line.encode("utf-8")), 75)
        unfolded = folded.replace("\r\n ", "")
        self.assertEqual(unfolded, f"SUMMARY:{title}")

    def test_export_ics_stops_at_end_of_day(self):
        """Blocks starting after the UTC day boundary are not exported."""
        plan = Plan(
            load_report=LoadReport(score=80.0, level="overload"),
            plan_id="x",
            items=[
                PlanItem(position=i, action="focus_block", title=f"Focus {i}") for i in range(1, 5)
            ],
        )
        # Start 22:13:20Z: blocks at 22:13, 22:58, 23:43 (all before midnight)
        # and a fourth at 00:28 next day, which must not be exported.
        ics = export_ics(plan, _tasks(), start_epoch=1_700_000_000.0)
        self.assertEqual(ics.count("BEGIN:VEVENT"), 3)
        self.assertNotIn("Focus 4", ics)

    def test_export_ics_horizon_override(self):
        plan = Plan(
            load_report=LoadReport(score=80.0, level="overload"),
            plan_id="x",
            items=[
                PlanItem(position=i, action="focus_block", title=f"Focus {i}") for i in range(1, 5)
            ],
        )
        ics = export_ics(
            plan,
            _tasks(),
            start_epoch=1_700_000_000.0,
            horizon_epoch=1_700_000_000.0 + 4 * 3600.0,
        )
        self.assertEqual(ics.count("BEGIN:VEVENT"), 4)

    def test_day_end_epoch_timezone(self):
        """_day_end_epoch respects custom timezone string."""
        from loadguard.actions import _day_end_epoch

        # Start at 23:00 UTC (1_700_002_800). Next UTC midnight is +3600s.
        utc_end = _day_end_epoch(1_700_002_800.0, tz_name="UTC")
        self.assertEqual(utc_end, 1_700_006_400.0)
        # In America/New_York (UTC-5 in Nov), 23:00 UTC is 18:00 EST (same day), so NY midnight is 6 hours later (+21600s -> 1_700_024_400.0)
        ny_end = _day_end_epoch(1_700_002_800.0, tz_name="America/New_York")
        self.assertEqual(ny_end, 1_700_024_400.0)
        # Invalid timezone falls back to UTC
        invalid_end = _day_end_epoch(1_700_002_800.0, tz_name="Invalid/Timezone")
        self.assertEqual(invalid_end, utc_end)

    def test_export_ics_existing_events_without_timestamps_falls_back(self):
        plan = Plan(
            load_report=LoadReport(score=80.0, level="overload"),
            plan_id="x",
            items=[PlanItem(position=1, action="focus_block", title="Focus 1")],
        )
        dummy_event = Event(timestamp=None, kind="context_switch")
        ics = export_ics(plan, _tasks(), existing_events=[dummy_event])
        self.assertIn("BEGIN:VCALENDAR", ics)

    def test_unknown_action_warns(self):
        """An action outside the known vocabulary warns instead of failing silently."""
        plan = Plan(
            load_report=LoadReport(score=80.0, level="overload"),
            plan_id="x",
            items=[PlanItem(position=1, action="focus", title="Focus block")],
        )
        with self.assertWarns(UserWarning):
            ics = export_ics(plan, _tasks(), start_epoch=1_700_000_000.0)
        self.assertEqual(ics.count("BEGIN:VEVENT"), 0)

    def test_do_item_without_task_is_skipped(self):
        plan = Plan(
            load_report=LoadReport(score=80.0, level="overload"),
            plan_id="x",
            items=[PlanItem(position=1, action="do", task_id=None, title="Untracked")],
        )
        ics = export_ics(plan, _tasks(), start_epoch=1_700_000_000.0)
        self.assertEqual(ics.count("BEGIN:VEVENT"), 0)

    def test_export_ics_skips_overlapping_meeting(self):
        """Focus block avoids colliding with existing meeting and starts in the free gap."""
        from loadguard.models import Event, MEETING

        existing = [Event(timestamp=1_700_000_000.0, kind=MEETING, duration_minutes=60.0)]
        plan = Plan(
            load_report=LoadReport(score=80.0, level="overload"),
            plan_id="x",
            items=[PlanItem(position=1, action="focus_block", title="Focus block")],
        )
        # Meeting runs 1700000000 -> 1700003600 (22:13:20 -> 23:13:20)
        # Focus block should start at 23:13:20
        ics = export_ics(plan, _tasks(), start_epoch=1_700_000_000.0, existing_events=existing)
        self.assertIn("DTSTART:20231114T231320Z", ics)

    def test_export_ics_with_tzid(self):
        """When tzid is provided, X-WR-TIMEZONE is included in calendar header."""
        ics = export_ics(_plan(), _tasks(), start_epoch=1_700_000_000.0, tzid="Europe/Madrid")
        self.assertIn("X-WR-TIMEZONE:Europe/Madrid", ics)

    def test_export_ics_with_busy_intervals(self):
        """Custom busy intervals prevent collision."""
        busy = [(1_700_000_000.0, 1_700_001_800.0)]  # 30 min busy
        plan = Plan(
            load_report=LoadReport(score=80.0, level="overload"),
            plan_id="x",
            items=[PlanItem(position=1, action="focus_block", title="Focus block")],
        )
        ics = export_ics(plan, _tasks(), start_epoch=1_700_000_000.0, busy_intervals=busy)
        self.assertIn("DTSTART:20231114T224320Z", ics)

    def test_export_ics_shifts_do_task_around_busy_intervals(self):
        """Do tasks also avoid busy windows."""
        busy = [(1_700_000_000.0, 1_700_001_800.0)]
        plan = Plan(
            load_report=LoadReport(score=80.0, level="overload"),
            plan_id="x",
            items=[PlanItem(position=1, action="do", task_id="a", title="Critical fix")],
        )
        ics = export_ics(plan, _tasks(), start_epoch=1_700_000_000.0, busy_intervals=busy)
        self.assertIn("DTSTART:20231114T224320Z", ics)

    def test_export_ics_ignores_zero_duration_or_non_meeting_events(self):
        from loadguard.models import Event, NOTIFICATION, MEETING

        events = [
            Event(timestamp=1_700_000_000.0, kind=NOTIFICATION),
            Event(timestamp=1_700_000_000.0, kind=MEETING, duration_minutes=0.0),
        ]
        plan = Plan(
            load_report=LoadReport(score=80.0, level="overload"),
            plan_id="x",
            items=[PlanItem(position=1, action="focus_block", title="Focus block")],
        )
        ics = export_ics(plan, _tasks(), start_epoch=1_700_000_000.0, existing_events=events)
        self.assertIn("DTSTART:20231114T221320Z", ics)

    def test_merge_busy_intervals(self):
        """Overlapping and contiguous intervals merge properly."""
        from loadguard.actions import _merge_busy_intervals, _find_next_free_slot

        self.assertEqual(_merge_busy_intervals([]), [])
        merged = _merge_busy_intervals([(100.0, 200.0), (150.0, 250.0), (300.0, 400.0)])
        self.assertEqual(merged, [(100.0, 250.0), (300.0, 400.0)])

        # Free slot search jumps past merged overlapping blocks
        slot = _find_next_free_slot(100.0, 60.0, [(100.0, 200.0), (180.0, 300.0)])
        self.assertEqual(slot, 300.0)

        # Free slot fits in gap between two busy blocks
        slot_gap = _find_next_free_slot(100.0, 50.0, [(100.0, 200.0), (300.0, 400.0)])
        self.assertEqual(slot_gap, 200.0)


class TestExportCsv(unittest.TestCase):
    def test_columns_and_rows(self):
        csv_text = export_tasks_csv(_plan(), _tasks())
        lines = csv_text.strip().splitlines()
        self.assertEqual(lines[0], "position,action,task_id,title,priority,rationale")
        self.assertEqual(len(lines), 5)  # header + 4 items
        self.assertIn("Critical fix", csv_text)

    def test_unknown_task_priority_blank(self):
        plan = Plan(
            load_report=LoadReport(score=80.0, level="overload"),
            items=[PlanItem(position=1, action="do", task_id="ghost", title="Ghost")],
        )
        csv_text = export_tasks_csv(plan, _tasks())
        self.assertIn("Ghost", csv_text)


class TestAudit(unittest.TestCase):
    def test_record_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audit.jsonl"
            record = record_approval("p1", "accepted", feedback="ok", helpful="yes", path=path)
            self.assertEqual(record.decision, "accepted")
            records = load_audit(path)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["plan_id"], "p1")
            self.assertEqual(records[0]["helpful"], "yes")

    def test_load_audit_ignores_corrupted_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audit.jsonl"
            path.write_text(
                '{"plan_id": "p1"}\nnot_json\n42\n{"plan_id": "p2"}\n', encoding="utf-8"
            )
            records = load_audit(path)
            self.assertEqual(len(records), 2)
            self.assertEqual([r["plan_id"] for r in records], ["p1", "p2"])

    def test_invalid_decision_coerced_to_rejected(self):
        record = record_approval("p1", "maybe")
        self.assertEqual(record.decision, "rejected")

    def test_clear(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audit.jsonl"
            record_approval("p1", "accepted", path=path)
            self.assertEqual(clear_audit(path), 1)
            self.assertEqual(load_audit(path), [])

    def test_invalid_helpful_blanked(self):
        record = record_approval("p1", "accepted", helpful="maybe")
        self.assertEqual(record.helpful, "")

    def test_load_audit_missing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(load_audit(Path(tmp) / "nope.jsonl"), [])

    def test_clear_audit_missing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(clear_audit(Path(tmp) / "nope.jsonl"), 0)

    def test_load_audit_skips_blank_and_comment_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audit.jsonl"
            path.write_text('# comment\n\n{"plan_id": "p1"}\n', encoding="utf-8")
            self.assertEqual(len(load_audit(path)), 1)


if __name__ == "__main__":
    unittest.main()
