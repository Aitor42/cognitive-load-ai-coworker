"""Deterministic planner: resequence, delegate, and insert recovery blocks.

The planner is intentionally rule-based and transparent.  It produces only the
structured plan; the narrative is added separately by the Narrator agent.

Integrates:
- **Audit trail feedback**: respects user rejection patterns so rejected
  actions are not repeatedly forced.
- **Time-of-day awareness**: late afternoon shifts thresholds to protect
  depleting cognitive reserves.
- **Ultradian break rhythm**: inserts recovery breaks based on work duration.
- **Energy-aware (quick-win) scheduling**: prioritizes short tasks under high load.
"""

from __future__ import annotations

from datetime import datetime, timezone

from .availability import is_absent
from .models import HIGH, OVERLOAD, TODO, LoadReport, Plan, PlanItem, Task, Worker

# Notification rate (per hour) above which we suggest batching.
BATCH_THRESHOLD = 20.0
# Focus ratio below which we schedule a focus block.
LOW_FOCUS_THRESHOLD = 0.2
# Accumulated work minutes before inserting a recovery break (ultradian rhythm).
BREAK_CADENCE_MINUTES = 90.0

# Max task priority (1..5) that gets delegated at each load level.
DELEGATE_MAX_PRIORITY = {OVERLOAD: 2, HIGH: 1}


def _sort_key(task: Task, level: str) -> tuple[float, float, float]:
    """Sort key: primary=priority, secondary=duration (quick-win), tertiary=deadline.

    In HIGH/OVERLOAD load, shorter tasks of the same priority are scheduled
    first (quick-win effect): completing them early frees cognitive slots and
    builds momentum.  Under lower load the duration tiebreaker is disabled.
    """
    deadline = task.deadline if task.deadline is not None else float("inf")
    duration_bonus = task.duration_minutes if level in (HIGH, OVERLOAD) else 0.0
    return (-float(task.priority), duration_bonus, deadline)


def _available_workers(
    workers: list[Worker],
    exclude_id: str | None,
    now: float | None,
) -> list[str]:
    """Return names (or IDs) of workers not currently absent, excluding *exclude_id*."""
    result: list[str] = []
    for w in workers:
        if w.id == exclude_id or (w.name and w.name == exclude_id):
            continue
        if now is not None and is_absent(w.absences, now):
            continue
        result.append(w.name if w.name else w.id)
    return result


def _parse_audit_preferences(audit_records: list[dict] | None) -> set[str]:
    """Extract user rejection preferences from past audit trail records."""
    rejected: set[str] = set()
    if not audit_records:
        return rejected
    for rec in audit_records:
        if rec.get("decision") == "rejected":
            fb = (rec.get("feedback") or "").lower()
            if "batch" in fb:
                rejected.add("batch")
            if "focus" in fb:
                rejected.add("focus_block")
            if "break" in fb:
                rejected.add("break")
            if "delegate" in fb:
                rejected.add("delegate")
    return rejected


def build_plan(
    tasks: list[Task],
    load_report: LoadReport,
    workers: list[Worker] | None = None,
    now: float | None = None,
    audit_history: list[dict] | None = None,
    hour_of_day: float | None = None,
) -> Plan:
    """Build a resequenced plan (structure only) from tasks and a load report.

    Parameters:
    - ``workers``: Optional list of team members for delegation suggestions.
    - ``now``: Current epoch timestamp (used for absence checks and hour determination).
    - ``audit_history``: Past approval/rejection records for closed-loop learning.
    - ``hour_of_day``: Float 0..24 representing time of day for fatigue/urgency adjustments.
    """
    level = load_report.level
    ordered = sorted(tasks, key=lambda t: _sort_key(t, level))

    # Time-of-day awareness: determine hour if not explicitly provided
    if hour_of_day is None and now is not None:
        hour_of_day = float(datetime.fromtimestamp(now, tz=timezone.utc).hour)

    is_late_day = hour_of_day is not None and hour_of_day >= 16.0
    delegate_max = DELEGATE_MAX_PRIORITY.get(level, 0)
    if is_late_day and level in (HIGH, OVERLOAD):
        # Late day fatigue: raise delegation threshold by +1 to protect depleted reserves
        delegate_max = min(delegate_max + 1, 3)

    rejected_actions = _parse_audit_preferences(audit_history)

    items: list[PlanItem] = []
    position = 0

    def add(action: str, task: Task | None = None, title: str = "", rationale: str = "") -> None:
        nonlocal position
        position += 1
        items.append(
            PlanItem(
                position=position,
                action=action,
                task_id=task.id if task else None,
                title=title or (task.title if task else action),
                rationale=rationale,
            )
        )

    # Suggest batching notifications first if they are a significant driver and not rejected.
    if (
        load_report.factors.get("notification_rate", 0.0) >= BATCH_THRESHOLD
        and "batch" not in rejected_actions
    ):
        add(
            "batch",
            title="Batch notifications",
            rationale="High notification rate; consolidate alerts into scheduled check-ins.",
        )

    accumulated_work = 0.0
    for task in ordered:
        if task.status != TODO:
            continue
        if task.priority <= delegate_max and "delegate" not in rejected_actions:
            if is_late_day:
                rationale = (
                    f"Priority {task.priority}/5 and load is {level} (late-day protection); "
                    "hand off to protect attention."
                )
            else:
                rationale = f"Priority {task.priority}/5 and load is {level}; hand off to protect attention."
            # Suggest available teammates when worker information is provided.
            if workers is not None:
                candidates = _available_workers(workers, task.assignee, now)
                if candidates:
                    rationale += f" Suggested: {', '.join(candidates[:3])}."
                else:
                    rationale += " Warning: no available teammates found."
            add("delegate", task, rationale=rationale)
        else:
            add(
                "do",
                task,
                rationale=f"Priority {task.priority}/5; keep in focus order (deadline-aware).",
            )
            accumulated_work += task.duration_minutes
            # Time-based breaks: insert after every BREAK_CADENCE_MINUTES of
            # accumulated work instead of a fixed task count.
            if (
                level in (HIGH, OVERLOAD)
                and accumulated_work >= BREAK_CADENCE_MINUTES
                and "break" not in rejected_actions
            ):
                add(
                    "break",
                    title="Recovery break",
                    rationale="Insert rest to avoid overload buildup.",
                )
                accumulated_work = 0.0

    # Schedule a focus block when focus time is scarce or load is elevated, unless rejected.
    if (
        load_report.factors.get("focus_ratio", 1.0) <= LOW_FOCUS_THRESHOLD
        or level in (HIGH, OVERLOAD)
    ) and "focus_block" not in rejected_actions:
        first_do = next((i for i, item in enumerate(items) if item.action == "do"), len(items))
        focus_item = PlanItem(
            position=0,  # will be renumbered below
            action="focus_block",
            title="Focus block (no notifications)",
            rationale="Low focus time detected; protect a distraction-free window for deep work.",
        )
        items.insert(first_do, focus_item)

    # Renumber positions sequentially after all insertions.
    for i, item in enumerate(items):
        item.position = i + 1

    return Plan(load_report=load_report, items=items)
