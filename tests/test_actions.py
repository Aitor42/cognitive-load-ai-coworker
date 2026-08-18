"""Unit tests for the actions module (approval, exports, audit)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loadguard.actions import (  # noqa: E402
    clear_audit,
    export_ics,
    export_tasks_csv,
    load_audit,
    record_approval,
)
from loadguard.models import LoadReport, Plan, PlanItem, Task  # noqa: E402


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
        self.assertEqual(ics.count("BEGIN:VEVENT"), 2)  # focus + break only
        self.assertIn("Focus block (no notifications)", ics)
        self.assertIn("Recovery break", ics)
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
        from loadguard.actions import _ics_escape

        self.assertEqual(_ics_escape("Hello\r\nWorld"), "Hello\\nWorld")
        self.assertEqual(_ics_escape("Hello\rWorld"), "Hello\\nWorld")
        self.assertEqual(_ics_escape("Hello,;\\World"), "Hello\\,\\;\\\\World")
        self.assertEqual(_ics_escape(""), "")


class TestExportCsv(unittest.TestCase):
    def test_columns_and_rows(self):
        csv_text = export_tasks_csv(_plan(), _tasks())
        lines = csv_text.strip().splitlines()
        self.assertEqual(lines[0], "position,action,task_id,title,priority,rationale")
        self.assertEqual(len(lines), 5)  # header + 4 items
        self.assertIn("Critical fix", csv_text)


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
            path.write_text('{"plan_id": "p1"}\nnot_json\n42\n{"plan_id": "p2"}\n', encoding="utf-8")
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


if __name__ == "__main__":
    unittest.main()
