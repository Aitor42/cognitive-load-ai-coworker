"""Shared sample data used by the demo script and the API dashboard."""

from __future__ import annotations

from .models import Task

_SAMPLE_TASKS = [
    Task(id="t1", title="Write incident postmortem", priority=5, focus_required=True),
    Task(id="t2", title="Review teammate PR", priority=4, focus_required=True),
    Task(id="t3", title="Prepare Q3 roadmap slides", priority=4, focus_required=True),
    Task(id="t4", title="Reply to vendor emails", priority=2, focus_required=False),
    Task(id="t5", title="Update expense report", priority=1, focus_required=False),
    Task(id="t6", title="Triaging backlog labels", priority=1, focus_required=False),
]


def sample_tasks() -> list[Task]:
    """Return fresh copies of the sample tasks."""
    return [
        Task(
            id=t.id,
            title=t.title,
            priority=t.priority,
            duration_minutes=t.duration_minutes,
            focus_required=t.focus_required,
            deadline=t.deadline,
            status=t.status,
        )
        for t in _SAMPLE_TASKS
    ]
