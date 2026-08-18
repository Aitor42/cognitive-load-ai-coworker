"""Closing the loop: human approval, calendar export, task export, audit.

LoadGuard does not just recommend — once the human approves, it can *act*:

- ``export_ics`` renders the focus/recovery blocks of an approved plan as a
  real calendar file (``.ics``) the user can import anywhere.
- ``export_tasks_csv`` renders the resequenced task list.
- ``record_approval`` appends every accept / reject / edit + feedback to an
  audit trail, so recommendation quality can be measured honestly.

All exports are local files; nothing leaves the machine.
"""

from __future__ import annotations

import csv
import io
import json
import math
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .models import Plan, Task

# Default durations (minutes) for the blocks LoadGuard schedules.
BLOCK_DURATIONS = {"focus_block": 45.0, "break": 15.0}
# Time (minutes) a delegated hand-off / notification batching takes in the day.
MINOR_STEP_MINUTES = 5.0

APPROVAL_DECISIONS = ("accepted", "rejected", "edited")


def new_plan_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class ApprovalRecord:
    """One human decision about a plan, written to the audit trail."""

    plan_id: str
    decision: str  # accepted | rejected | edited
    timestamp: float
    feedback: str = ""
    helpful: str = ""  # yes | no | ""


def _ics_datetime(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _ics_escape(text: str) -> str:
    if not text:
        return ""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return (
        normalized.replace("\\", "\\\\")
        .replace(";", "\\;")
        .replace(",", "\\,")
        .replace("\n", "\\n")
    )


def export_ics(plan: Plan, tasks: list[Task], start_epoch: float | None = None) -> str:
    """Render the plan's focus/recovery blocks as an iCalendar (.ics) string.

    The day is walked in plan order: ``do`` items consume their task duration,
    minor steps (delegate/batch) a small hand-off time, and every
    focus/break block becomes a VEVENT at its scheduled position.
    """
    if start_epoch is None:
        start_epoch = float(math.ceil(time.time() / 900.0) * 900)
    cursor = start_epoch
    task_durations = {t.id: max(0.0, float(t.duration_minutes)) for t in tasks}

    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//LoadGuard//Cognitive Load Co-Worker//EN",
        "CALSCALE:GREGORIAN",
    ]
    for item in plan.items:
        if item.action in BLOCK_DURATIONS:
            duration = BLOCK_DURATIONS[item.action]
            lines += [
                "BEGIN:VEVENT",
                f"UID:loadguard-{plan.plan_id or 'plan'}-{item.position}@loadguard",
                f"DTSTAMP:{_ics_datetime(cursor)}",
                f"DTSTART:{_ics_datetime(cursor)}",
                f"DTEND:{_ics_datetime(cursor + duration * 60.0)}",
                f"SUMMARY:{_ics_escape(item.title)}",
            ]
            if item.rationale:
                lines.append(f"DESCRIPTION:{_ics_escape(item.rationale)}")
            lines.append("END:VEVENT")
            cursor += duration * 60.0
        elif item.action == "do" and item.task_id:
            cursor += task_durations.get(item.task_id, 30.0) * 60.0
        elif item.action in ("delegate", "batch"):
            cursor += MINOR_STEP_MINUTES * 60.0
    lines.append("END:VCALENDAR")
    return "\r\n".join(lines) + "\r\n"


def export_tasks_csv(plan: Plan, tasks: list[Task]) -> str:
    """Render the resequenced task list as CSV (action, order, priority)."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["position", "action", "task_id", "title", "priority", "rationale"])
    for item in plan.items:
        priority = next((t.priority for t in tasks if t.id == item.task_id), "")
        writer.writerow(
            [item.position, item.action, item.task_id or "", item.title, priority, item.rationale]
        )
    return buffer.getvalue()


def record_approval(
    plan_id: str,
    decision: str,
    feedback: str = "",
    helpful: str = "",
    path: str | Path | None = None,
) -> ApprovalRecord:
    """Append a human decision to the audit trail (JSONL)."""
    record = ApprovalRecord(
        plan_id=plan_id,
        decision=decision if decision in APPROVAL_DECISIONS else "rejected",
        timestamp=time.time(),
        feedback=feedback,
        helpful=helpful if helpful in ("yes", "no") else "",
    )
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record.__dict__) + "\n")
    return record


def load_audit(path: str | Path) -> list[dict]:
    """Load the audit trail as a list of records (oldest first)."""
    path = Path(path)
    if not path.exists():
        return []
    records: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            data = json.loads(line)
            if isinstance(data, dict):
                records.append(data)
        except (json.JSONDecodeError, ValueError):
            continue
    return records


def clear_audit(path: str | Path) -> int:
    """Delete the audit trail; returns how many records were removed."""
    path = Path(path)
    if not path.exists():
        return 0
    count = len(load_audit(path))
    path.unlink()
    return count
