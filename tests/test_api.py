"""Tests for the FastAPI layer (approve / export / history / feedback / privacy).

These require ``fastapi`` and ``httpx`` (TestClient), which are optional
dependencies. The whole class is skipped automatically when they are not
installed, so the stdlib-only CI keeps passing:

    pip install fastapi httpx
    python -m unittest tests.test_api
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

try:
    import fastapi  # noqa: F401
    from fastapi.testclient import TestClient  # noqa: F401

    import loadguard.api as api_module
    from loadguard.api import PLANS, app  # noqa: F401

    HAS_FASTAPI = True
except (ImportError, RuntimeError):
    # TestClient raises ImportError when fastapi/httpx are absent, and RuntimeError
    # on newer starlette versions when the httpx backend is missing.
    HAS_FASTAPI = False


@unittest.skipUnless(HAS_FASTAPI, "fastapi + httpx not installed")
class TestApi(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        self.client = TestClient(app)
        # Point the history/audit stores at a temp dir so tests never touch the repo.
        self._old_history = api_module.HISTORY_PATH
        self._old_audit = api_module.AUDIT_PATH
        api_module.HISTORY_PATH = tmp / "history.jsonl"
        api_module.AUDIT_PATH = tmp / "audit.jsonl"
        PLANS.clear()

    def tearDown(self) -> None:
        api_module.HISTORY_PATH = self._old_history
        api_module.AUDIT_PATH = self._old_audit
        PLANS.clear()
        self._tmp.cleanup()

    def _analyze(self) -> dict:
        sample = self.client.get("/sample").json()
        resp = self.client.post(
            "/analyze",
            json={"events": sample["events"], "tasks": sample["tasks"]},
        )
        self.assertEqual(resp.status_code, 200)
        return resp.json()

    def test_health(self) -> None:
        self.assertEqual(self.client.get("/health").json(), {"status": "ok"})

    def test_analyze_returns_pending_guarded_plan(self) -> None:
        data = self._analyze()
        self.assertIn("plan_id", data)
        self.assertEqual(data["plan"]["status"], "pending")
        self.assertTrue(data["guardian"]["passed"])

    def test_accept_plan_and_export_ics(self) -> None:
        data = self._analyze()
        pid = data["plan_id"]
        resp = self.client.post("/approve", json={"plan_id": pid, "decision": "accepted"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "accepted")

        ics = self.client.get(f"/plan/{pid}/export.ics")
        self.assertEqual(ics.status_code, 200)
        self.assertIn("text/calendar", ics.headers["content-type"])
        self.assertIn("BEGIN:VCALENDAR", ics.text)
        self.assertIn("BEGIN:VEVENT", ics.text)

    def test_analyze_alarm_minutes_controls_ics_reminder(self) -> None:
        """The /analyze alarm option is honored at export time."""
        sample = self.client.get("/sample").json()
        payload = {"events": sample["events"], "tasks": sample["tasks"]}
        # Default (field omitted): the server-wide 10-minute lead.
        pid = self._analyze()["plan_id"]
        ics = self.client.get(f"/plan/{pid}/export.ics").text
        self.assertIn("TRIGGER:-PT10M", ics)
        # Custom lead time.
        resp = self.client.post("/analyze", json={**payload, "alarm_minutes": 5.0})
        self.assertEqual(resp.status_code, 200)
        ics = self.client.get(f"/plan/{resp.json()['plan_id']}/export.ics").text
        self.assertIn("TRIGGER:-PT5M", ics)
        # Zero disables the reminder.
        resp = self.client.post("/analyze", json={**payload, "alarm_minutes": 0.0})
        self.assertEqual(resp.status_code, 200)
        ics = self.client.get(f"/plan/{resp.json()['plan_id']}/export.ics").text
        self.assertNotIn("BEGIN:VALARM", ics)

    def test_analyze_rejects_negative_alarm_minutes(self) -> None:
        sample = self.client.get("/sample").json()
        resp = self.client.post(
            "/analyze",
            json={"events": sample["events"], "tasks": sample["tasks"], "alarm_minutes": -1.0},
        )
        self.assertEqual(resp.status_code, 422)

    def test_midday_export_keeps_default_alarm(self) -> None:
        """Midday plans are exported with the default reminder, not customized."""
        sample = self.client.get("/sample").json()
        resp = self.client.post(
            "/midday",
            json={"events": sample["events"], "tasks": sample["tasks"], "elapsed_minutes": 240},
        )
        self.assertEqual(resp.status_code, 200)
        pid = resp.json()["plan_id"]
        ics = self.client.get(f"/plan/{pid}/export.ics").text
        self.assertIn("TRIGGER:-PT10M", ics)

    def test_midday_filters_completed_tasks(self) -> None:
        """Midday reorganization excludes tasks specified in completed_task_ids."""
        sample = self.client.get("/sample").json()
        resp = self.client.post(
            "/midday",
            json={
                "events": sample["events"],
                "tasks": sample["tasks"],
                "elapsed_minutes": 240,
                "completed_task_ids": ["t1", "t2"],
                "alarm_minutes": 15.0,
                "tz_name": "Europe/Madrid",
            },
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        if body.get("plan") and body["plan"].get("items"):
            task_ids = [i.get("task_id") for i in body["plan"]["items"] if i.get("task_id")]
            self.assertNotIn("t1", task_ids)
            self.assertNotIn("t2", task_ids)
        pid = body.get("plan_id")
        if pid:
            ics = self.client.get(f"/plan/{pid}/export.ics").text
            self.assertIn("TRIGGER:-PT15M", ics)

    def test_export_ics_respects_existing_events(self) -> None:
        """Exported calendar blocks must fall on the same day as the events."""
        past_ts = 1_700_000_000.0  # 2023-11-14T22:13:20Z
        events = [{"timestamp": past_ts, "kind": "meeting", "duration_minutes": 30.0, "meta": {}}]
        tasks = [{"id": "t1", "title": "Focus Work", "priority": 5, "duration_minutes": 45.0}]
        resp = self.client.post("/analyze", json={"events": events, "tasks": tasks})
        self.assertEqual(resp.status_code, 200)
        pid = resp.json()["plan_id"]
        ics = self.client.get(f"/plan/{pid}/export.ics").text
        self.assertIn("BEGIN:VEVENT", ics, "ICS was empty — start_epoch anchoring is broken")
        import re
        from datetime import datetime, timezone as tz

        for ts_str in re.findall(r"DTSTART:(\d{8}T\d{6}Z)", ics):
            dt = datetime.strptime(ts_str, "%Y%m%dT%H%M%SZ").replace(tzinfo=tz.utc)
            self.assertEqual(dt.date().isoformat(), "2023-11-14")

    def test_export_ics_accepts_tzid_query_param(self) -> None:
        data = self._analyze()
        pid = data["plan_id"]
        ics = self.client.get(f"/plan/{pid}/export.ics?tzid=Europe/Madrid").text
        self.assertIn("X-WR-TIMEZONE:Europe/Madrid", ics)

    def test_export_ics_with_workday_start_and_end_query_params(self) -> None:
        events = [
            {"timestamp": 1700000000.0, "kind": "meeting", "duration_minutes": 30.0, "meta": {}}
        ]
        tasks = [{"id": "t1", "title": "Morning Task", "priority": 5, "duration_minutes": 45.0}]
        resp = self.client.post(
            "/analyze",
            json={
                "events": events,
                "tasks": tasks,
                "workday_start": "08:30",
                "workday_end": "17:00",
            },
        )
        self.assertEqual(resp.status_code, 200)
        pid = resp.json()["plan_id"]
        ics = self.client.get(f"/plan/{pid}/export.ics").text
        self.assertIn("DTSTART:20231114T083000Z", ics)

    def test_export_ics_with_start_time_and_end_time_query_params(self) -> None:
        events = [
            {"timestamp": 1700000000.0, "kind": "meeting", "duration_minutes": 30.0, "meta": {}}
        ]
        tasks = [{"id": "t1", "title": "Morning Task", "priority": 5, "duration_minutes": 45.0}]
        resp = self.client.post("/analyze", json={"events": events, "tasks": tasks})
        pid = resp.json()["plan_id"]
        ics = self.client.get(f"/plan/{pid}/export.ics?start_time=07:30&end_time=16:30").text
        self.assertIn("DTSTART:20231114T073000Z", ics)

    def test_midday_with_workday_hours(self) -> None:
        tasks = self.client.get("/sample").json()["tasks"]
        resp = self.client.post(
            "/midday",
            json={
                "events": self._busy_day_events(),
                "tasks": tasks,
                "elapsed_minutes": 480.0,
                "total_minutes": 480.0,
                "workday_start": "08:00",
                "workday_end": "17:00",
            },
        )
        self.assertEqual(resp.status_code, 200)
        pid = resp.json()["plan_id"]
        ics = self.client.get(f"/plan/{pid}/export.ics").text
        self.assertIn("BEGIN:VCALENDAR", ics)

    def test_export_csv(self) -> None:
        data = self._analyze()
        pid = data["plan_id"]
        resp = self.client.get(f"/plan/{pid}/export.csv")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("position,action,task_id", resp.text)

    def test_reject_plan(self) -> None:
        data = self._analyze()
        pid = data["plan_id"]
        resp = self.client.post("/approve", json={"plan_id": pid, "decision": "rejected"})
        self.assertEqual(resp.json()["status"], "rejected")

    def test_edit_plan_replaces_items(self) -> None:
        data = self._analyze()
        pid = data["plan_id"]
        kept = data["plan"]["items"][:2]
        resp = self.client.post(
            "/approve",
            json={"plan_id": pid, "decision": "edited", "items": kept},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(PLANS[pid]["payload"]["plan"]["items"]), 2)

    def test_approve_rejects_invalid_state_transition(self) -> None:
        data = self._analyze()
        pid = data["plan_id"]
        # Accept plan (terminal state)
        resp1 = self.client.post("/approve", json={"plan_id": pid, "decision": "accepted"})
        self.assertEqual(resp1.status_code, 200)
        # Attempt to reject an already accepted plan
        resp2 = self.client.post("/approve", json={"plan_id": pid, "decision": "rejected"})
        self.assertEqual(resp2.status_code, 400)
        self.assertIn("Cannot transition plan", resp2.json()["detail"])

    def test_analyze_with_role_profile(self) -> None:
        sample = self.client.get("/sample").json()
        resp = self.client.post(
            "/analyze",
            json={
                "events": sample["events"],
                "tasks": sample["tasks"],
                "role": "developer",
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json().get("role"), "developer")
        self.assertEqual(resp.json().get("weights_profile"), "developer")

    def test_analyze_with_invalid_role_returns_422(self) -> None:
        sample = self.client.get("/sample").json()
        resp = self.client.post(
            "/analyze",
            json={
                "events": sample["events"],
                "tasks": sample["tasks"],
                "role": "astronaut",
            },
        )
        self.assertEqual(resp.status_code, 422)
        self.assertIn("unknown role 'astronaut'", str(resp.json()))

    def test_midday_with_invalid_role_returns_422(self) -> None:
        sample = self.client.get("/sample").json()
        resp = self.client.post(
            "/midday",
            json={
                "events": sample["events"],
                "tasks": sample["tasks"],
                "role": "astronaut",
            },
        )
        self.assertEqual(resp.status_code, 422)
        self.assertIn("unknown role 'astronaut'", str(resp.json()))

    def test_edit_plan_without_title_auto_populates_from_tasks(self) -> None:
        data = self._analyze()
        pid = data["plan_id"]
        # Pass minimal item with task_id and action only
        items = [{"action": "do", "task_id": "t1"}]
        resp = self.client.post(
            "/approve",
            json={"plan_id": pid, "decision": "edited", "items": items},
        )
        self.assertEqual(resp.status_code, 200)
        plan_items = PLANS[pid]["payload"]["plan"]["items"]
        self.assertEqual(plan_items[0]["title"], "Write incident postmortem")

    def test_edit_plan_with_renamed_title_passes_safety_gate(self) -> None:
        data = self._analyze()
        pid = data["plan_id"]
        # Rename the task title during editing
        items = [
            {"action": "do", "task_id": "t1", "title": "Write incident postmortem - Urgent Draft"}
        ]
        resp = self.client.post(
            "/approve",
            json={"plan_id": pid, "decision": "edited", "items": items},
        )
        self.assertEqual(resp.status_code, 200)
        plan_items = PLANS[pid]["payload"]["plan"]["items"]
        self.assertEqual(plan_items[0]["title"], "Write incident postmortem - Urgent Draft")
        # Source task in stored['tasks'] must remain strictly immutable
        stored_task = [t for t in PLANS[pid]["tasks"] if t["id"] == "t1"][0]
        self.assertEqual(stored_task["title"], "Write incident postmortem")

    def test_edit_plan_recomputes_impact(self) -> None:
        """Saving an edited plan must return a fresh before/after projection."""
        data = self._analyze()
        pid = data["plan_id"]
        original_delta = data["impact"]["delta"]
        # Re-submitting the same items recomputes the same projection.
        resp = self.client.post(
            "/approve",
            json={"plan_id": pid, "decision": "edited", "items": data["plan"]["items"]},
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("impact", body)
        self.assertIsNotNone(body["impact"])
        imp = body["impact"]
        for key in ("before_score", "after_score", "delta"):
            self.assertIn(key, imp)
        self.assertAlmostEqual(imp["delta"], original_delta, places=1)
        self.assertGreaterEqual(imp["delta"], 0.0)

    def test_approve_without_edit_returns_no_impact(self) -> None:
        data = self._analyze()
        pid = data["plan_id"]
        resp = self.client.post("/approve", json={"plan_id": pid, "decision": "accepted"})
        self.assertEqual(resp.status_code, 200)
        self.assertIsNone(resp.json()["impact"])

    def test_unknown_plan_returns_404(self) -> None:
        resp = self.client.post("/approve", json={"plan_id": "nope", "decision": "accepted"})
        self.assertEqual(resp.status_code, 404)

    def test_edited_plan_cannot_delegate_critical_task(self) -> None:
        """The safety gate must be re-run on edited items, not just at analyze time."""
        data = self._analyze()
        pid = data["plan_id"]
        critical = {
            "position": 1,
            "action": "delegate",
            "task_id": "t1",  # t1 is priority 5 (critical) in sample_tasks
            "title": "Write incident postmortem",
            "rationale": "",
        }
        resp = self.client.post(
            "/approve",
            json={"plan_id": pid, "decision": "edited", "items": [critical]},
        )
        self.assertEqual(resp.status_code, 400)

    def test_history_rejects_out_of_range_score(self) -> None:
        resp = self.client.post("/history", json={"score": 999.0})
        self.assertEqual(resp.status_code, 422)

    def test_privacy_does_not_leak_absolute_path(self) -> None:
        p = self.client.get("/privacy").json()
        self.assertNotIn("data_dir", p)

    def test_history_roundtrip_and_delete(self) -> None:
        self.client.post("/history", json={"score": 42.0})
        self.client.post("/history", json={"score": 55.0})
        self.assertEqual(self.client.get("/history").json()["history"], [42.0, 55.0])
        self.client.delete("/history")
        self.assertEqual(self.client.get("/history").json()["history"], [])

    def test_feedback_records_audit(self) -> None:
        data = self._analyze()
        pid = data["plan_id"]
        self.client.post("/approve", json={"plan_id": pid, "decision": "accepted"})
        resp = self.client.post(
            "/feedback",
            json={"plan_id": pid, "helpful": "yes", "feedback": "good"},
        )
        self.assertEqual(resp.status_code, 200)
        records = self.client.get("/audit").json()["records"]
        self.assertTrue(any(r.get("helpful") == "yes" for r in records))

    def test_privacy_statement(self) -> None:
        p = self.client.get("/privacy").json()
        self.assertIn("never_captured", p)
        self.assertIn("does not diagnose", p["statement"])

    def test_parse_bool_and_to_tasks(self) -> None:
        tasks_payload = [
            {"id": "1", "title": "T1", "priority": "4", "focus_required": "false"},
            {"id": "2", "title": "T2", "priority": 2, "focus_required": True},
        ]
        tasks = api_module._to_tasks(tasks_payload)
        self.assertFalse(tasks[0].focus_required)
        self.assertEqual(tasks[0].priority, 4)
        self.assertTrue(tasks[1].focus_required)

    def test_parse_bool_variants(self) -> None:
        # None falls back to ``default``.
        self.assertTrue(api_module._parse_bool(None))
        self.assertFalse(api_module._parse_bool(None, default=False))
        # Bools pass through.
        self.assertTrue(api_module._parse_bool(True))
        self.assertFalse(api_module._parse_bool(False))
        # Truthy strings.
        self.assertTrue(api_module._parse_bool("yes"))
        self.assertTrue(api_module._parse_bool("1"))
        self.assertTrue(api_module._parse_bool("true"))
        # Falsy strings.
        self.assertFalse(api_module._parse_bool("no"))
        self.assertFalse(api_module._parse_bool("0"))
        self.assertFalse(api_module._parse_bool("false"))
        # A non-keyword string falls through to bool().
        self.assertTrue(api_module._parse_bool("maybe"))
        # Other values fall through to bool().
        self.assertTrue(api_module._parse_bool(42))
        self.assertFalse(api_module._parse_bool(0))

    def test_feedback_unknown_plan_returns_404(self) -> None:
        resp = self.client.post("/feedback", json={"plan_id": "nope"})
        self.assertEqual(resp.status_code, 404)

    def test_export_ics_unknown_plan_returns_404(self) -> None:
        resp = self.client.get("/plan/nope/export.ics")
        self.assertEqual(resp.status_code, 404)

    def test_export_csv_unknown_plan_returns_404(self) -> None:
        resp = self.client.get("/plan/nope/export.csv")
        self.assertEqual(resp.status_code, 404)

    def test_audit_roundtrip_and_delete(self) -> None:
        data = self._analyze()
        pid = data["plan_id"]
        self.client.post("/approve", json={"plan_id": pid, "decision": "accepted"})
        self.assertGreaterEqual(len(self.client.get("/audit").json()["records"]), 1)
        self.client.delete("/audit")
        self.assertEqual(self.client.get("/audit").json()["records"], [])

    def test_dashboard_serves_html(self) -> None:
        # First call loads from disk and caches; second call serves from cache
        resp1 = self.client.get("/")
        self.assertEqual(resp1.status_code, 200)
        resp2 = self.client.get("/")
        self.assertEqual(resp2.status_code, 200)

    def test_favicon_returns_svg(self) -> None:
        resp = self.client.get("/favicon.ico")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("image/svg+xml", resp.headers["content-type"])

    def test_dashboard_missing_index_returns_404(self) -> None:
        with (
            mock.patch.object(api_module, "_DASHBOARD_HTML", None),
            mock.patch.object(Path, "exists", return_value=False),
        ):
            resp = api_module.dashboard()
            self.assertEqual(resp.status_code, 404)

    def test_trim_plans_enforces_max_capacity(self) -> None:
        with mock.patch.object(api_module, "MAX_PERSISTED_PLANS", 2):
            api_module.PLANS.clear()
            api_module.PLANS["p1"] = {"payload": {}}
            api_module.PLANS["p2"] = {"payload": {}}
            api_module.PLANS["p3"] = {"payload": {}}
            api_module._trim_plans()
            self.assertEqual(len(api_module.PLANS), 2)
            self.assertNotIn("p1", api_module.PLANS)

    def test_sample_includes_workers(self) -> None:
        sample = self.client.get("/sample").json()
        self.assertIn("workers", sample)
        self.assertGreater(len(sample["workers"]), 0)
        self.assertIn("absences", sample["workers"][0])

    def test_to_tasks_parses_assignee(self) -> None:
        tasks = api_module._to_tasks([{"id": "1", "title": "T", "assignee": "w1"}])
        self.assertEqual(tasks[0].assignee, "w1")
        tasks_none = api_module._to_tasks([{"id": "2", "title": "T"}])
        self.assertIsNone(tasks_none[0].assignee)

    def test_to_tasks_accepts_iso_8601_deadline(self) -> None:
        tasks = api_module._to_tasks(
            [{"id": "1", "title": "T", "deadline": "2026-08-28T17:00:00+00:00"}]
        )
        self.assertAlmostEqual(tasks[0].deadline, 1787936400.0)
        # A trailing 'Z' and epoch numbers are accepted too.
        tasks_z = api_module._to_tasks(
            [{"id": "2", "title": "T", "deadline": "2026-08-28T17:00:00Z"}]
        )
        self.assertAlmostEqual(tasks_z[0].deadline, 1787936400.0)
        tasks_epoch = api_module._to_tasks([{"id": "3", "title": "T", "deadline": 5.0}])
        self.assertEqual(tasks_epoch[0].deadline, 5.0)

    def test_to_tasks_without_deadline(self) -> None:
        tasks = api_module._to_tasks([{"id": "1", "title": "T"}])
        self.assertIsNone(tasks[0].deadline)

    def test_to_workers_builds_absences(self) -> None:
        workers = api_module._to_workers(
            [{"id": "w1", "name": "Ada", "absences": [{"start": 1.0, "end": 2.0, "kind": "leave"}]}]
        )
        self.assertEqual(workers[0].name, "Ada")
        self.assertEqual(workers[0].absences[0].kind, "leave")
        self.assertEqual(workers[0].absences[0].start, 1.0)

    def test_to_workers_accepts_iso_8601_absences(self) -> None:
        workers = api_module._to_workers(
            [
                {
                    "id": "w1",
                    "name": "Ada",
                    "absences": [
                        {
                            "start": "2026-09-01T09:00:00+00:00",
                            "end": "2026-09-05T17:00:00Z",
                            "kind": "vacation",
                        }
                    ],
                }
            ]
        )
        absence = workers[0].absences[0]
        self.assertAlmostEqual(absence.start, 1788253200.0)
        self.assertAlmostEqual(absence.end, 1788627600.0)
        self.assertEqual(absence.kind, "vacation")

    def test_analyze_with_workers_produces_reassignment_alerts(self) -> None:
        sample = self.client.get("/sample").json()
        resp = self.client.post(
            "/analyze",
            json={
                "events": sample["events"],
                "tasks": sample["tasks"],
                "workers": sample["workers"],
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("reassignment_alerts", data)
        self.assertGreater(len(data["reassignment_alerts"]), 0)

    def test_midday_endpoint(self) -> None:
        sample = self.client.get("/sample").json()
        resp = self.client.post(
            "/midday",
            json={
                "events": sample["events"],
                "tasks": sample["tasks"],
                "workers": sample["workers"],
                "elapsed_minutes": 240.0,
                "total_minutes": 480.0,
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("observed_score", data)
        self.assertIn("projected_score", data)
        self.assertIn("reorganized", data)
        self.assertIn("reassignment_alerts", data)

    def test_midday_with_role_profile(self) -> None:
        sample = self.client.get("/sample").json()
        resp = self.client.post(
            "/midday",
            json={
                "events": sample["events"],
                "tasks": sample["tasks"],
                "workers": sample["workers"],
                "elapsed_minutes": 240.0,
                "total_minutes": 480.0,
                "role": "developer",
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json().get("role"), "developer")

    def _busy_day_events(self) -> list[dict]:
        """A dense day that reliably projects to overload and triggers a replan."""
        events: list[dict] = []
        for h in range(8):
            t = 8 * 3600 + h * 3600
            events.append({"timestamp": t, "kind": "meeting", "duration_minutes": 50.0})
            for i in range(20):
                events.append(
                    {
                        "timestamp": t + 120 + i * 150,
                        "kind": "notification",
                        "meta": {"source": "slack"},
                    }
                )
                events.append(
                    {
                        "timestamp": t + 180 + i * 150,
                        "kind": "context_switch",
                        "meta": {"from": "a", "to": "b"},
                    }
                )
        return events

    def test_midday_without_replan_stores_nothing(self) -> None:
        """A calm day does not re-organize, so no plan is stored."""
        calm = [
            {"timestamp": 8 * 3600 + h * 3600, "kind": "focus_block", "duration_minutes": 55.0}
            for h in range(8)
        ]
        resp = self.client.post(
            "/midday",
            json={
                "events": calm,
                "tasks": self.client.get("/sample").json()["tasks"],
                "elapsed_minutes": 480.0,
                "total_minutes": 480.0,
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertFalse(data["reorganized"])
        self.assertNotIn("plan_id", data)

    def test_midday_plan_can_be_approved_and_exported(self) -> None:
        """The afternoon plan must be stored so it can be accepted and exported."""
        tasks = self.client.get("/sample").json()["tasks"]
        resp = self.client.post(
            "/midday",
            json={
                "events": self._busy_day_events(),
                "tasks": tasks,
                "elapsed_minutes": 480.0,
                "total_minutes": 480.0,
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["reorganized"])
        self.assertIn("plan_id", data)
        pid = data["plan_id"]
        self.assertIn(pid, PLANS)

        appr = self.client.post("/approve", json={"plan_id": pid, "decision": "accepted"})
        self.assertEqual(appr.status_code, 200)
        self.assertEqual(appr.json()["status"], "accepted")

        ics = self.client.get(f"/plan/{pid}/export.ics")
        self.assertEqual(ics.status_code, 200)
        self.assertIn("BEGIN:VCALENDAR", ics.text)
        self.assertIn("BEGIN:VEVENT", ics.text)

    def test_privacy_mentions_absence_reason(self) -> None:
        p = self.client.get("/privacy").json()
        self.assertIn("the medical or personal reason for an absence", p["never_captured"])

    def test_ingest_jsonl(self) -> None:
        text = (
            "# comment\n\n"
            '{"timestamp": "2026-08-17T09:00:00Z", "kind": "meeting", "duration_minutes": 30}\n'
            '{"timestamp": "2026-08-17T09:05:00Z", "kind": "notification", "meta": {"source": "slack"}}\n'
        )
        resp = self.client.post("/ingest", json={"text": text, "format": "jsonl"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["format"], "jsonl")
        self.assertEqual(len(data["events"]), 2)
        self.assertEqual(data["events"][1]["meta"]["source"], "slack")

    def test_ingest_ics(self) -> None:
        text = (
            "BEGIN:VCALENDAR\nVERSION:2.0\n"
            "BEGIN:VEVENT\nUID:m1\nDTSTART:20260817T090000Z\nDTEND:20260817T100000Z\n"
            "SUMMARY:Standup\nEND:VEVENT\n"
            "BEGIN:VEVENT\nUID:a1\nDTSTART;VALUE=DATE:20260817\nDTEND;VALUE=DATE:20260818\n"
            "SUMMARY:Out of office\nEND:VEVENT\n"
            "END:VCALENDAR\n"
        )
        resp = self.client.post("/ingest", json={"text": text, "format": "ics"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["format"], "ics")
        # Only the timed meeting becomes an event; the absence is separated.
        self.assertEqual(len(data["events"]), 1)
        self.assertEqual(data["events"][0]["meta"]["title"], "Standup")

    def test_ingest_auto_detects_format(self) -> None:
        ics = "BEGIN:VCALENDAR\nVERSION:2.0\nBEGIN:VEVENT\nUID:m1\nDTSTART:20260817T090000Z\nEND:VEVENT\nEND:VCALENDAR\n"
        resp = self.client.post("/ingest", json={"text": ics})
        self.assertEqual(resp.json()["format"], "ics")
        jsonl = '{"timestamp": "2026-08-17T09:00:00Z", "kind": "meeting"}\n'
        resp = self.client.post("/ingest", json={"text": jsonl})
        self.assertEqual(resp.json()["format"], "jsonl")

    def test_ingest_empty_returns_400(self) -> None:
        resp = self.client.post("/ingest", json={"text": "   "})
        self.assertEqual(resp.status_code, 400)

    def test_ingest_invalid_jsonl_returns_400(self) -> None:
        resp = self.client.post(
            "/ingest",
            json={"text": '{"timestamp": "2026-08-17T09:00:00Z", "kind": "meeting"}\nnot_json\n'},
        )
        self.assertEqual(resp.status_code, 400)

    def test_ingest_ics_calendar_text(self) -> None:
        ics_text = (
            "BEGIN:VCALENDAR\nVERSION:2.0\n"
            "BEGIN:VEVENT\nUID:1\nDTSTART:20260817T090000Z\nDTEND:20260817T100000Z\n"
            "SUMMARY:Standup\nEND:VEVENT\nEND:VCALENDAR\n"
        )
        resp = self.client.post("/ingest", json={"text": ics_text, "format": "ics"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["format"], "ics")
        self.assertEqual(len(resp.json()["events"]), 1)

    def test_ingest_derives_tasks_from_calendar_events(self) -> None:
        """Uploaded calendar meetings become tasks so the plan uses real activities."""
        ics_text = (
            "BEGIN:VCALENDAR\nVERSION:2.0\n"
            # Long meeting (90 min) → priority 4
            "BEGIN:VEVENT\nUID:1\nDTSTART:20260817T090000Z\nDTEND:20260817T103000Z\n"
            "SUMMARY:Architecture Review\nEND:VEVENT\n"
            # Short meeting (10 min) → priority 2
            "BEGIN:VEVENT\nUID:2\nDTSTART:20260817T110000Z\nDTEND:20260817T111000Z\n"
            "SUMMARY:Quick Sync\nEND:VEVENT\n"
            # Normal meeting (30 min) → priority 3
            "BEGIN:VEVENT\nUID:3\nDTSTART:20260817T140000Z\nDTEND:20260817T143000Z\n"
            "SUMMARY:Sprint Planning\nEND:VEVENT\n"
            "END:VCALENDAR\n"
        )
        resp = self.client.post("/ingest", json={"text": ics_text, "format": "ics"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(len(data["events"]), 3)
        self.assertEqual(len(data["tasks"]), 3)
        # Long meeting → high priority
        long_task = next(t for t in data["tasks"] if t["title"] == "Architecture Review")
        self.assertEqual(long_task["priority"], 4)
        self.assertAlmostEqual(long_task["duration_minutes"], 90.0, places=0)
        # Short meeting → low priority
        short_task = next(t for t in data["tasks"] if t["title"] == "Quick Sync")
        self.assertEqual(short_task["priority"], 2)
        self.assertAlmostEqual(short_task["duration_minutes"], 10.0, places=0)
        # Normal meeting → medium priority
        normal_task = next(t for t in data["tasks"] if t["title"] == "Sprint Planning")
        self.assertEqual(normal_task["priority"], 3)

    def test_ingest_jsonl_derives_tasks(self) -> None:
        """JSONL ingest also derives tasks from meeting events."""
        text = (
            '{"timestamp": "2026-08-17T09:00:00Z", "kind": "meeting", "duration_minutes": 90, "meta": {"title": "Long"}}\n'
            '{"timestamp": "2026-08-17T10:00:00Z", "kind": "notification", "meta": {"source": "slack"}}\n'
        )
        resp = self.client.post("/ingest", json={"text": text, "format": "jsonl"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        # Only the meeting becomes a task; the notification does not.
        self.assertEqual(len(data["tasks"]), 1)
        self.assertEqual(data["tasks"][0]["title"], "Long")
        self.assertEqual(data["tasks"][0]["priority"], 4)

    def test_ingest_ics_date_filter_matches(self) -> None:
        """Supplying the correct date returns only that day's events."""
        ics_text = (
            "BEGIN:VCALENDAR\nVERSION:2.0\n"
            "BEGIN:VEVENT\nUID:1\nDTSTART:20260817T090000Z\nDTEND:20260817T100000Z\n"
            "SUMMARY:Standup\nEND:VEVENT\n"
            "BEGIN:VEVENT\nUID:2\nDTSTART:20260818T090000Z\nDTEND:20260818T100000Z\n"
            "SUMMARY:Other day\nEND:VEVENT\n"
            "END:VCALENDAR\n"
        )
        resp = self.client.post(
            "/ingest", json={"text": ics_text, "format": "ics", "date": "2026-08-17"}
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(len(data["events"]), 1)
        self.assertEqual(data["events"][0]["meta"]["title"], "Standup")
        self.assertEqual(data["analyzed_date"], "2026-08-17")
        self.assertEqual(data["total_events_in_file"], 2)

    def test_ingest_ics_date_filter_no_match(self) -> None:
        """A date with no events returns an empty list and metadata."""
        ics_text = (
            "BEGIN:VCALENDAR\nVERSION:2.0\n"
            "BEGIN:VEVENT\nUID:1\nDTSTART:20260817T090000Z\nDTEND:20260817T100000Z\n"
            "SUMMARY:Standup\nEND:VEVENT\n"
            "END:VCALENDAR\n"
        )
        resp = self.client.post(
            "/ingest", json={"text": ics_text, "format": "ics", "date": "2026-08-18"}
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["events"], [])
        self.assertEqual(data["analyzed_date"], "2026-08-18")
        self.assertEqual(data["total_events_in_file"], 1)

    def test_ingest_ics_default_date_is_today(self) -> None:
        """When no date is supplied, it defaults to the events date or today."""
        from datetime import datetime

        ics_text = (
            "BEGIN:VCALENDAR\nVERSION:2.0\n"
            "BEGIN:VEVENT\nUID:1\nDTSTART:20260817T090000Z\nDTEND:20260817T100000Z\n"
            "SUMMARY:Standup\nEND:VEVENT\n"
            "END:VCALENDAR\n"
        )
        resp = self.client.post("/ingest", json={"text": ics_text, "format": "ics"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["analyzed_date"], "2026-08-17")
        self.assertEqual(data["total_events_in_file"], 1)

        # Empty calendar defaults to today
        today = datetime.now().strftime("%Y-%m-%d")
        empty_ics = "BEGIN:VCALENDAR\nVERSION:2.0\nEND:VCALENDAR\n"
        resp_empty = self.client.post("/ingest", json={"text": empty_ics, "format": "ics"})
        self.assertEqual(resp_empty.status_code, 200)
        self.assertEqual(resp_empty.json()["analyzed_date"], today)

    def test_ingest_ics_multi_day_only_returns_requested_day(self) -> None:
        """A full multi-day calendar is filtered down to a single requested day."""
        ics_text = (
            "BEGIN:VCALENDAR\nVERSION:2.0\n"
            "BEGIN:VEVENT\nUID:1\nDTSTART:20260817T090000Z\nDTEND:20260817T100000Z\n"
            "SUMMARY:Day1\nEND:VEVENT\n"
            "BEGIN:VEVENT\nUID:2\nDTSTART:20260818T100000Z\nDTEND:20260818T110000Z\n"
            "SUMMARY:Day2\nEND:VEVENT\n"
            "BEGIN:VEVENT\nUID:3\nDTSTART:20260819T140000Z\nDTEND:20260819T150000Z\n"
            "SUMMARY:Day3\nEND:VEVENT\n"
            "END:VCALENDAR\n"
        )
        resp = self.client.post(
            "/ingest", json={"text": ics_text, "format": "ics", "date": "2026-08-18"}
        )
        data = resp.json()
        self.assertEqual(len(data["events"]), 1)
        self.assertEqual(data["events"][0]["meta"]["title"], "Day2")
        self.assertEqual(len(data["tasks"]), 1)
        self.assertEqual(data["tasks"][0]["title"], "Day2")
        self.assertEqual(data["total_events_in_file"], 3)

    def test_filter_events_to_date_invalid_date_returns_all(self) -> None:
        """_filter_events_to_date returns the original list on a bad date string."""
        from loadguard.api import _filter_events_to_date

        events = [{"timestamp": 1755820800.0}]  # 2026-08-17
        result = _filter_events_to_date(events, "not-a-date")
        self.assertEqual(result, events)

    def test_plans_persist_to_disk(self) -> None:
        data = self._analyze()
        pid = data["plan_id"]
        self.assertTrue(api_module.PLANS_PATH.exists())
        loaded = api_module._load_persisted_plans()
        self.assertIn(pid, loaded)

    def test_load_persisted_plans_corrupt_fallback(self) -> None:
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write("not-valid-json")
            f_path = Path(f.name)
        try:
            with mock.patch("loadguard.api.PLANS_PATH", f_path):
                self.assertEqual(api_module._load_persisted_plans(), {})
        finally:
            f_path.unlink()
        with mock.patch("loadguard.api.PLANS_PATH", Path("non_existent_file_xyz.json")):
            self.assertEqual(api_module._load_persisted_plans(), {})

    def test_persist_plans_exception_handling(self) -> None:
        with mock.patch("loadguard.api.json.dumps", side_effect=TypeError("disk error")):
            # Should catch and not raise
            api_module._persist_plans()

    def test_pilot_endpoint_reports_projection_without_outcome(self) -> None:
        resp = self.client.get("/pilot")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("baseline", data)
        self.assertIn("projected", data)
        self.assertIn("observed", data)
        self.assertFalse(data["has_observed"])
        self.assertIsNone(data["observed"])
        self.assertIn("projection", data["summary"])

    def test_pilot_with_outcome_events_reports_observed(self) -> None:
        sample = self.client.get("/sample").json()
        # Outcome: the same day but only meetings and focus blocks (no
        # interruptions) — real post-plan signals, so observed is honest.
        outcome = [e for e in sample["events"] if e["kind"] in ("meeting", "focus_block")]
        resp = self.client.post(
            "/pilot",
            json={
                "events": sample["events"],
                "tasks": sample["tasks"],
                "outcome_events": outcome,
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["has_observed"])
        self.assertIsNotNone(data["observed"])
        self.assertIn("observed", data["summary"])

    def test_api_key_authentication(self) -> None:
        """When LOADGUARD_API_KEY is configured, protected routes require valid X-API-Key."""
        with mock.patch.dict(os.environ, {"LOADGUARD_API_KEY": "secret-token-123"}):
            # Health is always public
            resp_health = self.client.get("/health")
            self.assertEqual(resp_health.status_code, 200)

            # Protected endpoint without header -> 401
            resp_unauth = self.client.get("/history")
            self.assertEqual(resp_unauth.status_code, 401)
            self.assertEqual(resp_unauth.json()["detail"], "Invalid or missing X-API-Key header")

            # Protected endpoint with wrong header -> 401
            resp_wrong = self.client.get("/history", headers={"X-API-Key": "wrong-token"})
            self.assertEqual(resp_wrong.status_code, 401)

            # Protected endpoint with valid header -> 200
            resp_auth = self.client.get("/history", headers={"X-API-Key": "secret-token-123"})
            self.assertEqual(resp_auth.status_code, 200)

    def test_observability_middleware_and_telemetry(self) -> None:
        """Observability middleware injects correlation headers and endpoints report telemetry."""
        # 1. Automatic Request ID and Response Time headers
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("X-Request-ID", resp.headers)
        self.assertIn("X-Response-Time-Ms", resp.headers)
        auto_req_id = resp.headers["X-Request-ID"]
        self.assertTrue(len(auto_req_id) > 0)

        # 2. Propagates client-supplied correlation ID
        custom_id = "trace-uuid-45678"
        resp_custom = self.client.get("/health", headers={"X-Request-ID": custom_id})
        self.assertEqual(resp_custom.headers.get("X-Request-ID"), custom_id)

        # 3. Analyze endpoint includes structured telemetry payload
        sample = self.client.get("/sample").json()
        resp_analyze = self.client.post(
            "/analyze",
            json={"events": sample["events"], "tasks": sample["tasks"]},
            headers={"X-Request-ID": custom_id},
        )
        self.assertEqual(resp_analyze.status_code, 200)
        data = resp_analyze.json()
        self.assertIn("telemetry", data)
        self.assertEqual(data["telemetry"]["request_id"], custom_id)
        self.assertIn("duration_ms", data["telemetry"])
        self.assertIn("llm_provider", data["telemetry"])
        self.assertIn("guardian_engine", data["telemetry"])
        self.assertIn("guardian_passed", data["telemetry"])

        # 4. Midday review includes structured telemetry
        resp_midday = self.client.post(
            "/midday",
            json={"events": sample["events"], "tasks": sample["tasks"]},
            headers={"X-Request-ID": custom_id},
        )
        self.assertEqual(resp_midday.status_code, 200)
        data_mid = resp_midday.json()
        self.assertIn("telemetry", data_mid)
        self.assertEqual(data_mid["telemetry"]["request_id"], custom_id)

    def test_destructive_delete_protection(self) -> None:
        """When LOADGUARD_ALLOW_DELETE=false, DELETE endpoints return 403 Forbidden."""
        with mock.patch.dict(os.environ, {"LOADGUARD_ALLOW_DELETE": "false"}):
            resp_del_hist = self.client.delete("/history")
            self.assertEqual(resp_del_hist.status_code, 403)
            self.assertIn(
                "Destructive DELETE operations are disabled", resp_del_hist.json()["detail"]
            )

            resp_del_audit = self.client.delete("/audit")
            self.assertEqual(resp_del_audit.status_code, 403)
            self.assertIn(
                "Destructive DELETE operations are disabled", resp_del_audit.json()["detail"]
            )

    def test_approve_persists_status_to_disk(self) -> None:
        """When a plan is approved, its new status is persisted in the plans file."""
        data = self._analyze()
        pid = data["plan_id"]
        resp = self.client.post("/approve", json={"plan_id": pid, "decision": "accepted"})
        self.assertEqual(resp.status_code, 200)

        from loadguard.api import _load_persisted_plans

        persisted = _load_persisted_plans()
        self.assertIn(pid, persisted)
        self.assertEqual(persisted[pid]["payload"]["plan"]["status"], "accepted")

    def test_filter_events_to_date_missing_or_corrupt_timestamps(self) -> None:
        from loadguard.api import _filter_events_to_date

        events = [
            {"timestamp": None},
            {"timestamp": "corrupt_timestamp"},
            {"timestamp": 1e25},  # overflow
            {"no_timestamp_key": True},
        ]
        result = _filter_events_to_date(events, "2026-08-17")
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
