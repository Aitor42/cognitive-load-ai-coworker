"""Deterministic planner: resequence, delegate, and insert recovery blocks.

The planner is intentionally rule-based and transparent. It produces only the
structured plan; the narrative is added separately by the Narrator agent.
"""

from __future__ import annotations

from .models import HIGH, OVERLOAD, TODO, LoadReport, Plan, PlanItem, Task

# Notification rate (per hour) above which we suggest batching.
BATCH_THRESHOLD = 20.0
# Focus ratio below which we schedule a focus block.
LOW_FOCUS_THRESHOLD = 0.2

# Max task priority (1..5) that gets delegated at each load level.
DELEGATE_MAX_PRIORITY = {OVERLOAD: 2, HIGH: 1}


def _sort_key(task: Task) -> tuple[float, float]:
    deadline = task.deadline if task.deadline is not None else float("inf")
    return (-float(task.priority), deadline)


def build_plan(tasks: list[Task], load_report: LoadReport) -> Plan:
    """Build a resequenced plan (structure only) from tasks and a load report."""
    ordered = sorted(tasks, key=_sort_key)
    level = load_report.level
    delegate_max = DELEGATE_MAX_PRIORITY.get(level, 0)

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

    # Suggest batching notifications first if they are a significant driver.
    if load_report.factors.get("notification_rate", 0.0) >= BATCH_THRESHOLD:
        add(
            "batch",
            title="Batch notifications",
            rationale="High notification rate; consolidate alerts into scheduled check-ins.",
        )

    do_count = 0
    for task in ordered:
        if task.status != TODO:
            continue
        if task.priority <= delegate_max:
            add(
                "delegate",
                task,
                rationale=f"Priority {task.priority}/5 and load is {level}; hand off to protect attention.",
            )
        else:
            add(
                "do",
                task,
                rationale=f"Priority {task.priority}/5; keep in focus order (deadline-aware).",
            )
            do_count += 1
            # Protect attention during high-load windows with periodic breaks.
            if level in (HIGH, OVERLOAD) and do_count % 2 == 0:
                add(
                    "break",
                    title="Recovery break",
                    rationale="Insert rest to avoid overload buildup.",
                )

    # Schedule a focus block when focus time is scarce or load is elevated.
    # Insert it *before* the first deep-work task so the developer starts with
    # protected attention, rather than tacking it on at the end.
    if load_report.factors.get("focus_ratio", 1.0) <= LOW_FOCUS_THRESHOLD or level in (
        HIGH,
        OVERLOAD,
    ):
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
