"""Worker availability and deadline-driven reassignment alerts.

When a task has a deadline and its assignee is away before that deadline,
LoadGuard raises a deterministic alert suggesting which teammates are available
to take it over. Like every other LoadGuard output it is a *suggestion*: the
human still decides whether to reassign.
"""

from __future__ import annotations

import time

from .models import Absence, ReassignmentAlert, Task, TODO, VACATION, Worker

KIND_LABELS = {VACATION: "vacation", "leave": "leave"}


def is_absent(absences: list[Absence], at: float) -> Absence | None:
    """Return the absence covering ``at``, or None if the worker is available."""
    for absence in absences:
        if absence.start <= at <= absence.end:
            return absence
    return None


def absent_during(absences: list[Absence], start: float, end: float) -> Absence | None:
    """Return an absence overlapping ``[start, end]``, or None."""
    for absence in absences:
        if absence.start <= end and absence.end >= start:
            return absence
    return None


def worker_absences(worker_id: str, workers: list[Worker]) -> list[Absence]:
    """Return the absences of a worker, or an empty list when unknown."""
    for worker in workers:
        if worker.id == worker_id:
            return worker.absences
    return []


def find_reassignment_alerts(
    tasks: list[Task], workers: list[Worker], now: float | None = None
) -> list[ReassignmentAlert]:
    """Flag todo tasks whose assignee is away between now and the deadline.

    A task is flagged when it is assigned, has a future deadline, and its
    assignee has an absence overlapping ``[now, deadline]``. Suggested
    assignees are teammates available for that whole window.
    """
    now = time.time() if now is None else now
    by_id = {w.id: w for w in workers}
    alerts: list[ReassignmentAlert] = []
    for task in tasks:
        if task.status != TODO or task.assignee is None or task.deadline is None:
            continue
        if task.deadline < now:
            continue
        assignee = by_id.get(task.assignee)
        absence = absent_during(assignee.absences if assignee else [], now, task.deadline)
        if absence is None:
            continue
        suggestions = [
            w.id
            for w in workers
            if w.id != task.assignee and absent_during(w.absences, now, task.deadline) is None
        ]
        alerts.append(
            ReassignmentAlert(
                task_id=task.id,
                title=task.title,
                assignee=task.assignee,
                deadline=task.deadline,
                reason=(
                    f"{task.assignee} is on {KIND_LABELS.get(absence.kind, absence.kind or 'leave')} and cannot "
                    f"complete '{task.title}' by its deadline"
                ),
                suggested_assignees=suggestions,
            )
        )
    return alerts
