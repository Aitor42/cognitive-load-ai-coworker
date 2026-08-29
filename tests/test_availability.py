"""Unit tests for worker availability and reassignment alerts."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loadguard.availability import (  # noqa: E402
    absent_during,
    find_reassignment_alerts,
    is_absent,
    worker_absences,
)
from loadguard.models import (  # noqa: E402
    DONE,
    LEAVE,
    VACATION,
    Absence,
    Task,
    Worker,
)


def _workers() -> list[Worker]:
    return [
        Worker(
            id="w1",
            name="Ada",
            absences=[Absence(start=100.0, end=200.0, kind=VACATION)],
        ),
        Worker(id="w2", name="Bo", absences=[]),
        Worker(
            id="w3",
            name="Cy",
            absences=[Absence(start=50.0, end=150.0, kind=LEAVE)],
        ),
    ]


class TestIsAbsent(unittest.TestCase):
    def test_absent_within_absence(self) -> None:
        absences = [Absence(start=100.0, end=200.0)]
        self.assertEqual(is_absent(absences, 150.0).start, 100.0)

    def test_available_outside_absence(self) -> None:
        self.assertIsNone(is_absent([Absence(start=100.0, end=200.0)], 50.0))

    def test_empty_absences(self) -> None:
        self.assertIsNone(is_absent([], 150.0))


class TestAbsentDuring(unittest.TestCase):
    def test_overlapping_window(self) -> None:
        absences = [Absence(start=100.0, end=200.0)]
        self.assertEqual(absent_during(absences, 150.0, 300.0).start, 100.0)

    def test_non_overlapping_window(self) -> None:
        absences = [Absence(start=100.0, end=200.0)]
        self.assertIsNone(absent_during(absences, 201.0, 300.0))

    def test_empty_absences(self) -> None:
        self.assertIsNone(absent_during([], 0.0, 300.0))


class TestWorkerAbsences(unittest.TestCase):
    def test_known_worker(self) -> None:
        self.assertEqual(len(worker_absences("w1", _workers())), 1)

    def test_unknown_worker(self) -> None:
        self.assertEqual(worker_absences("ghost", _workers()), [])


class TestFindReassignmentAlerts(unittest.TestCase):
    def test_no_tasks(self) -> None:
        self.assertEqual(find_reassignment_alerts([], _workers(), now=0.0), [])

    def test_skips_non_todo(self) -> None:
        tasks = [Task(id="a", title="Done", priority=5, status=DONE, assignee="w1", deadline=150.0)]
        self.assertEqual(find_reassignment_alerts(tasks, _workers(), now=0.0), [])

    def test_skips_unassigned(self) -> None:
        tasks = [Task(id="a", title="Unassigned", priority=5, deadline=150.0)]
        self.assertEqual(find_reassignment_alerts(tasks, _workers(), now=0.0), [])

    def test_skips_no_deadline(self) -> None:
        tasks = [Task(id="a", title="No deadline", priority=5, assignee="w1")]
        self.assertEqual(find_reassignment_alerts(tasks, _workers(), now=0.0), [])

    def test_skips_past_deadline(self) -> None:
        tasks = [Task(id="a", title="Overdue", priority=5, assignee="w1", deadline=10.0)]
        self.assertEqual(find_reassignment_alerts(tasks, _workers(), now=100.0), [])

    def test_skips_unknown_assignee(self) -> None:
        tasks = [Task(id="a", title="Ghost", priority=5, assignee="ghost", deadline=150.0)]
        self.assertEqual(find_reassignment_alerts(tasks, _workers(), now=0.0), [])

    def test_skips_available_assignee(self) -> None:
        tasks = [Task(id="a", title="Fine", priority=5, assignee="w2", deadline=150.0)]
        self.assertEqual(find_reassignment_alerts(tasks, _workers(), now=0.0), [])

    def test_alert_generated_with_suggestions(self) -> None:
        tasks = [Task(id="a", title="Postmortem", priority=5, assignee="w1", deadline=150.0)]
        alerts = find_reassignment_alerts(tasks, _workers(), now=0.0)
        self.assertEqual(len(alerts), 1)
        alert = alerts[0]
        self.assertEqual(alert.task_id, "a")
        self.assertEqual(alert.assignee, "w1")
        self.assertIn("vacation", alert.reason)
        # w3 is also absent over [0,150], so only w2 is a valid suggestion.
        self.assertEqual(alert.suggested_assignees, ["w2"])

    def test_leave_kind_in_reason(self) -> None:
        tasks = [Task(id="b", title="Emails", priority=2, assignee="w3", deadline=100.0)]
        alerts = find_reassignment_alerts(tasks, _workers(), now=0.0)
        self.assertEqual(len(alerts), 1)
        self.assertIn("leave", alerts[0].reason)

    def test_custom_absence_kind_fallback(self) -> None:
        abs_obj = Absence(start=50.0, end=150.0, kind=LEAVE)
        abs_obj.kind = "medical"
        workers = [Worker(id="w9", name="Doctor", absences=[abs_obj])]
        tasks = [Task(id="b", title="Emails", priority=2, assignee="w9", deadline=100.0)]
        alerts = find_reassignment_alerts(tasks, workers, now=0.0)
        self.assertEqual(len(alerts), 1)
        self.assertIn("medical", alerts[0].reason)

    def test_now_defaults_to_current_time(self) -> None:
        tasks = [Task(id="a", title="Any", priority=5, assignee="w1", deadline=150.0)]
        # Past deadline with the default (real) now -> no alert.
        self.assertEqual(find_reassignment_alerts(tasks, _workers()), [])


if __name__ == "__main__":
    unittest.main()
