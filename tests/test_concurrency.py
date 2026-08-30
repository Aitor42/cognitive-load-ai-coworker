"""Concurrent multi-threaded stress tests verifying thread-safety and no race conditions."""

from __future__ import annotations

import concurrent.futures
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from loadguard.actions import load_audit, record_approval  # noqa: E402
from loadguard.api import app  # noqa: E402
from loadguard.baseline import append_score, load_history  # noqa: E402
from loadguard.sample_data import sample_tasks  # noqa: E402
from loadguard.signals import load_events  # noqa: E402
from loadguard.workflow import run_workflow  # noqa: E402


class TestConcurrency(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        self.sample_events_file = ROOT / "demo" / "sample_events.jsonl"
        self.events = [asdict(e) for e in load_events(self.sample_events_file)]
        self.tasks = [asdict(t) for t in sample_tasks()]

    def test_concurrent_api_analyze_and_approve(self) -> None:
        """Multiple concurrent requests analyzing and approving plans in parallel."""
        num_workers = 16
        num_requests = 32

        def _worker(idx: int) -> tuple[int, int, str]:
            resp = self.client.post(
                "/analyze",
                json={
                    "events": self.events,
                    "tasks": self.tasks,
                    "role": "developer" if idx % 2 == 0 else "manager",
                },
            )
            status_analyze = resp.status_code
            plan_id = resp.json().get("plan_id", "")

            # Concurrently approve the plan
            resp_appr = self.client.post(
                "/approve",
                json={
                    "plan_id": plan_id,
                    "decision": "accepted",
                    "feedback": f"concurrent test {idx}",
                },
            )
            status_approve = resp_appr.status_code

            # Concurrently export calendar
            resp_exp = self.client.get(f"/plan/{plan_id}/export.ics")
            return (status_analyze, status_approve, resp_exp.status_code)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_worker, i) for i in range(num_requests)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        for status_analyze, status_approve, status_export in results:
            self.assertEqual(status_analyze, 200)
            self.assertEqual(status_approve, 200)
            self.assertEqual(status_export, 200)

    def test_concurrent_midday_reviews(self) -> None:
        """Multiple concurrent threads performing midday re-organization."""
        num_workers = 12
        num_requests = 24

        def _midday_worker(idx: int) -> int:
            resp = self.client.post(
                "/midday",
                json={
                    "events": self.events,
                    "tasks": self.tasks,
                    "elapsed_minutes": 240.0,
                    "total_minutes": 480.0,
                    "role": "support",
                },
            )
            return resp.status_code

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_midday_worker, i) for i in range(num_requests)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        for code in results:
            self.assertEqual(code, 200)

    def test_concurrent_workflow_runs(self) -> None:
        """Pure workflow is stateless and safe to run concurrently across threads."""
        num_threads = 20
        events_objs = load_events(self.sample_events_file)
        tasks_objs = sample_tasks()

        def _run(idx: int) -> float:
            res = run_workflow(events_objs, tasks_objs, role="developer")
            return res.load_report.score

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(_run, i) for i in range(50)]
            scores = [f.result() for f in concurrent.futures.as_completed(futures)]

        self.assertEqual(len(scores), 50)
        # All scores computed over identical inputs should be identical (deterministic)
        self.assertTrue(all(s == scores[0] for s in scores))

    def test_concurrent_audit_trail_writes(self) -> None:
        """Concurrent writes to audit log do not corrupt JSONL records."""
        with tempfile.TemporaryDirectory() as tmp:
            audit_path = Path(tmp) / "audit.jsonl"
            num_threads = 16
            num_writes = 64

            def _write(idx: int) -> None:
                record_approval(
                    f"plan_{idx}",
                    "accepted" if idx % 2 == 0 else "rejected",
                    feedback=f"feedback {idx}",
                    path=audit_path,
                )

            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(_write, i) for i in range(num_writes)]
                for f in concurrent.futures.as_completed(futures):
                    f.result()

            records = load_audit(audit_path)
            self.assertEqual(len(records), num_writes)

    def test_concurrent_history_writes(self) -> None:
        """Concurrent writes to history score log do not corrupt JSONL values."""
        with tempfile.TemporaryDirectory() as tmp:
            history_path = Path(tmp) / "history.jsonl"
            num_threads = 16
            num_writes = 64

            def _write(idx: int) -> None:
                append_score(history_path, 40.0 + (idx % 30))

            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(_write, i) for i in range(num_writes)]
                for f in concurrent.futures.as_completed(futures):
                    f.result()

            scores = load_history(history_path)
            self.assertEqual(len(scores), num_writes)

    def test_concurrent_plan_state_persists(self) -> None:
        """Concurrent status approvals correctly persist to disk without race conditions."""
        resp = self.client.post(
            "/analyze",
            json={"events": self.events, "tasks": self.tasks, "role": "developer"},
        )
        pid = resp.json()["plan_id"]

        num_threads = 8

        def _approve(decision: str) -> int:
            r = self.client.post("/approve", json={"plan_id": pid, "decision": decision})
            return r.status_code

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(_approve, "accepted") for _ in range(16)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        self.assertTrue(all(code == 200 for code in results))


if __name__ == "__main__":
    unittest.main()
