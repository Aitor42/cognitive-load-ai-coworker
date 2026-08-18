"""Tests for the FastAPI layer (approve / export / history / feedback / privacy).

These require ``fastapi`` and ``httpx`` (TestClient), which are optional
dependencies. The whole class is skipped automatically when they are not
installed, so the stdlib-only CI keeps passing:

    pip install fastapi httpx
    python -m unittest tests.test_api
"""

from __future__ import annotations

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
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)

    def test_dashboard_missing_index_returns_404(self) -> None:
        with mock.patch.object(Path, "exists", return_value=False):
            resp = api_module.dashboard()
            self.assertEqual(resp.status_code, 404)


if __name__ == "__main__":
    unittest.main()
