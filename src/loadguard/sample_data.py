"""Shared sample data used by the demo script and the API dashboard."""

from __future__ import annotations

import time

from .models import LEAVE, VACATION, Absence, Task, Worker

_DAY = 86400.0


def sample_tasks() -> list[Task]:
    """Return fresh copies of the sample tasks.

    A couple of tasks carry an ``assignee`` and a relative ``deadline`` so the
    reassignment-alert path can be demonstrated against ``sample_workers``.
    """
    now = time.time()
    return [
        Task(
            id="t1",
            title="Write incident postmortem",
            priority=5,
            focus_required=True,
            assignee="w1",
            deadline=now + 2 * _DAY,
        ),
        Task(id="t2", title="Review teammate PR", priority=4, focus_required=True),
        Task(id="t3", title="Prepare Q3 roadmap slides", priority=4, focus_required=True),
        Task(
            id="t4",
            title="Reply to vendor emails",
            priority=2,
            focus_required=False,
            assignee="w3",
            deadline=now + 0.5 * _DAY,
        ),
        Task(id="t5", title="Update expense report", priority=1, focus_required=False),
        Task(id="t6", title="Triaging backlog labels", priority=1, focus_required=False),
    ]


def sample_workers() -> list[Worker]:
    """Return a small team with time-relative absences for the demo.

    Absences are generated relative to "now" so the dashboard always shows a
    current, self-consistent team availability picture.
    """
    now = time.time()
    return [
        Worker(
            id="w1",
            name="Ada",
            absences=[Absence(start=now - 2 * _DAY, end=now + 3 * _DAY, kind=VACATION)],
        ),
        Worker(id="w2", name="Bo", absences=[]),
        Worker(
            id="w3",
            name="Cy",
            absences=[Absence(start=now - _DAY, end=now + _DAY, kind=LEAVE)],
        ),
    ]
