"""Unit tests for the IBM Bob MCP server tools."""

from __future__ import annotations

import sys
import unittest
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from loadguard.sample_data import sample_tasks  # noqa: E402
from loadguard.signals import load_events  # noqa: E402
from mcp_server.server import (  # noqa: E402
    _parse_bool,
    _to_events,
    _to_tasks,
    analyze_workload,
    approve_plan,
    benchmark_workload,
    compute_load_score,
    export_plan_csv,
    export_plan_ics,
    pilot_evaluation,
    propose_plan,
    review_midday_workload,
    self_test,
)


class TestMCPServer(unittest.TestCase):
    def setUp(self) -> None:
        self.sample_events_file = ROOT / "demo" / "sample_events.jsonl"
        self.events = [asdict(e) for e in load_events(self.sample_events_file)]
        self.tasks = [asdict(t) for t in sample_tasks()]

    def test_parse_bool(self) -> None:
        self.assertTrue(_parse_bool(True))
        self.assertFalse(_parse_bool(False))
        self.assertTrue(_parse_bool("true"))
        self.assertTrue(_parse_bool("yes"))
        self.assertTrue(_parse_bool("1"))
        self.assertFalse(_parse_bool("false"))
        self.assertFalse(_parse_bool("no"))
        self.assertFalse(_parse_bool("0"))
        self.assertTrue(_parse_bool(None, default=True))
        self.assertFalse(_parse_bool(None, default=False))

    def test_converters(self) -> None:
        evs = _to_events(self.events)
        self.assertTrue(len(evs) > 0)
        tsks = _to_tasks(self.tasks)
        self.assertTrue(len(tsks) > 0)

    def test_compute_load_score(self) -> None:
        res = compute_load_score(self.events)
        self.assertIn("score", res)
        self.assertIn("level", res)
        self.assertNotIn("error", res)

    def test_compute_load_score_with_role(self) -> None:
        res = compute_load_score(self.events, role="developer")
        self.assertIn("score", res)

    def test_compute_load_score_error_handling(self) -> None:
        res = compute_load_score([{"kind": "unknown_invalid_xyz"}])
        self.assertIn("error", res)

    def test_analyze_workload(self) -> None:
        res = analyze_workload(self.events, self.tasks)
        self.assertIn("load_report", res)
        self.assertIn("plan", res)
        self.assertIn("impact", res)

    def test_benchmark_workload(self) -> None:
        res = benchmark_workload(self.events, self.tasks)
        self.assertIn("reduction_points", res)
        self.assertIn("before_score", res)

    def test_propose_plan(self) -> None:
        res = propose_plan(self.events, self.tasks, history=[40.0, 45.0, 50.0], role="manager")
        self.assertIn("plan", res)
        self.assertIn("trend", res)

    def test_review_midday_workload(self) -> None:
        res = review_midday_workload(
            self.events, self.tasks, elapsed_minutes=240.0, role="developer"
        )
        self.assertIn("projected_score", res)
        self.assertIn("reorganized", res)

    def test_approve_plan(self) -> None:
        res = approve_plan(self.events, self.tasks, decision="accepted", feedback="Looks great")
        self.assertEqual(res.get("status"), "accepted")
        self.assertIn("audit", res)

    def test_export_plan_ics(self) -> None:
        res = export_plan_ics(self.events, self.tasks, alarm_minutes=10.0, tz_name="UTC")
        self.assertIn("ics", res)
        self.assertIn("BEGIN:VCALENDAR", res["ics"])

    def test_export_plan_csv(self) -> None:
        res = export_plan_csv(self.events, self.tasks)
        self.assertIn("tasks_csv", res)
        self.assertIn("position,action,task_id", res["tasks_csv"])

    def test_pilot_evaluation(self) -> None:
        res = pilot_evaluation(self.events, self.tasks, outcome_events=self.events[:5])
        self.assertIn("baseline", res)
        self.assertIn("projected", res)
        self.assertIn("observed", res)

    def test_self_test_runs_cleanly(self) -> None:
        # self_test() executes all assertions without raising
        self_test()


if __name__ == "__main__":
    unittest.main()
