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
import time
import uuid
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .models import MEETING, Event, Plan, Task

# Default durations (minutes) for the blocks LoadGuard schedules.
BLOCK_DURATIONS = {"focus_block": 45.0, "break": 15.0}
# Lead time (minutes) of the VALARM reminder attached to exported focus blocks.
FOCUS_ALARM_MINUTES = 10.0
# Time (minutes) a delegated hand-off / notification batching takes in the day.
MINOR_STEP_MINUTES = 5.0

APPROVAL_DECISIONS = ("accepted", "rejected", "edited")

# State machine for human plan decisions. Terminal states cannot transition to other decisions.
VALID_TRANSITIONS: dict[str, set[str]] = {
    "pending": {"accepted", "rejected", "edited"},
    "edited": {"accepted", "rejected", "edited"},
    "accepted": set(),
    "rejected": set(),
}


def is_valid_transition(current_status: str, target_decision: str) -> bool:
    """Validate whether transitioning from current_status to target_decision is allowed."""
    if current_status == target_decision:
        return True
    return target_decision in VALID_TRANSITIONS.get(current_status, set())


def new_plan_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class ApprovalRecord:
    """One human decision about a plan, written to the audit trail."""

    plan_id: str
    decision: str  # accepted | rejected | edited | pending
    timestamp: float
    feedback: str = ""
    helpful: str = ""  # yes | no | ""


def _ics_datetime(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _ics_alarm_trigger(minutes: float) -> str:
    """Render an RFC 5545 TRIGGER duration that fires *minutes* before the start."""
    seconds = int(round(minutes * 60))
    if seconds % 60 == 0:
        return f"-PT{seconds // 60}M"
    return f"-PT{seconds}S"


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


def _ics_fold(text: str) -> str:
    """Fold a content line longer than 75 octets per RFC 5545 (CRLF + space).

    The limit counts UTF-8 octets, not characters, so non-ASCII titles and
    rationales stay within the limit a strict parser enforces. A fold is
    never placed inside a multi-byte sequence, keeping every emitted line
    independently valid UTF-8.
    """
    if len(text.encode("utf-8")) <= 75:
        return text
    lines: list[str] = []
    current = ""
    used = 0
    for ch in text:
        width = len(ch.encode("utf-8"))
        if used + width > 75:
            lines.append(current)
            current = " "  # continuation lines start with exactly one space
            used = 1
        current += ch
        used += width
    lines.append(current)
    return "\r\n".join(lines)


def _day_end_epoch(epoch: float, tz_name: str | None = None) -> float:
    """Start of the next day after ``epoch`` (exclusive day bound).

    Accepts an optional IANA timezone string (e.g. "America/New_York", "Europe/Madrid")
    or falls back to UTC when unsupplied or invalid.
    """
    tz: Any = timezone.utc
    if tz_name:
        try:
            tz = ZoneInfo(tz_name)
        except (ZoneInfoNotFoundError, ValueError, OSError):
            tz = timezone.utc
    dt = datetime.fromtimestamp(epoch, tz=tz)
    next_day = datetime(dt.year, dt.month, dt.day, tzinfo=tz) + timedelta(days=1)
    return next_day.timestamp()


def _merge_busy_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge overlapping and contiguous busy intervals into disjoint sorted intervals."""
    if not intervals:
        return []
    sorted_intervals = sorted(intervals, key=lambda x: (x[0], x[1]))
    merged: list[tuple[float, float]] = [sorted_intervals[0]]
    for start, end in sorted_intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _find_next_free_slot(
    cursor: float, duration_seconds: float, busy_intervals: list[tuple[float, float]]
) -> float:
    """Advance cursor past any overlapping busy intervals until a contiguous free window fits."""
    merged = _merge_busy_intervals(busy_intervals)
    for start, end in merged:
        if end <= cursor:
            continue
        if cursor + duration_seconds <= start:
            return cursor
        cursor = end
    return cursor


def _parse_time_of_day(
    val: Any,
    default_hour: int = 9,
    default_minute: int = 0,
) -> tuple[int, int]:
    """Parse HH:MM string, decimal hour, or return (default_hour, default_minute)."""
    if val is None:
        return default_hour, default_minute
    if isinstance(val, (int, float)):
        hour = int(val)
        minute = int(round((val - hour) * 60))
        return max(0, min(23, hour)), max(0, min(59, minute))
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return default_hour, default_minute
        parts = s.split(":")
        try:
            h = int(parts[0])
            m = int(parts[1]) if len(parts) > 1 else 0
            return max(0, min(23, h)), max(0, min(59, m))
        except ValueError:
            try:
                flt = float(s)
                return _parse_time_of_day(flt, default_hour, default_minute)
            except ValueError:
                return default_hour, default_minute
    return default_hour, default_minute


def export_ics(
    plan: Plan,
    tasks: list[Task],
    start_epoch: float | None = None,
    existing_events: list[Event] | None = None,
    busy_intervals: list[tuple[float, float]] | None = None,
    horizon_epoch: float | None = None,
    alarm_minutes: float | None = FOCUS_ALARM_MINUTES,
    tzid: str | None = None,
    tz_name: str | None = None,
    workday_start: str | float | int | None = None,
    workday_end: str | float | int | None = None,
) -> str:
    """Render the plan's focus/recovery blocks as an iCalendar (.ics) string.

    The day is walked in plan order: ``do`` items consume their task duration,
    minor steps (delegate/batch) a small hand-off time, and every
    focus/break block becomes a VEVENT at its scheduled position.

    When *existing_events* or *busy_intervals* are provided, collision detection
    skips over existing meetings and scheduled commitments, placing blocks only
    in genuine free gaps.

    Blocks that would start at or after *horizon_epoch* (by default the end of
    the workday or day containing *start_epoch* in *tz_name*) are not exported.

    Parameters:
    - *workday_start*: Start time of the workday (e.g. "09:00", "08:30" or 9.0).
    - *workday_end*: End time of the workday (e.g. "18:00", "17:30" or 18.0).
    - *alarm_minutes*: Lead time of the VALARM reminder (None or 0 for no alarm).
    """
    tz: Any = timezone.utc
    tz_str = tz_name or tzid
    if tz_str:
        try:
            tz = ZoneInfo(tz_str)
        except (ZoneInfoNotFoundError, ValueError, OSError):
            tz = timezone.utc

    if start_epoch is None:
        base_ts = time.time()
        if existing_events:
            stamps = [
                e.timestamp for e in existing_events if getattr(e, "timestamp", None) is not None
            ]
            if stamps:
                base_ts = min(stamps)
        dt = datetime.fromtimestamp(base_ts, tz=tz)
        s_hour, s_minute = _parse_time_of_day(workday_start, default_hour=9, default_minute=0)
        start_epoch = datetime(
            dt.year, dt.month, dt.day, s_hour, s_minute, 0, tzinfo=tz
        ).timestamp()

    if horizon_epoch is None:
        if workday_end is not None:
            e_hour, e_minute = _parse_time_of_day(workday_end, default_hour=18, default_minute=0)
            dt_start = datetime.fromtimestamp(start_epoch, tz=tz)
            candidate_end = datetime(
                dt_start.year, dt_start.month, dt_start.day, e_hour, e_minute, 0, tzinfo=tz
            ).timestamp()
            if candidate_end > start_epoch:
                horizon_epoch = candidate_end
            else:
                horizon_epoch = _day_end_epoch(start_epoch, tz_name=tz_name or tzid)
        else:
            horizon_epoch = _day_end_epoch(start_epoch, tz_name=tz_name or tzid)
    cursor = start_epoch
    dtstamp = _ics_datetime(time.time())

    all_busy: list[tuple[float, float]] = list(busy_intervals or [])
    if existing_events:
        for e in existing_events:
            if e.kind in (MEETING, "meeting", "busy") and e.duration_minutes > 0:
                all_busy.append((e.timestamp, e.timestamp + e.duration_minutes * 60.0))
    all_busy = _merge_busy_intervals(all_busy)

    task_durations = {t.id: max(0.0, float(t.duration_minutes or 30.0)) for t in tasks}

    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//LoadGuard//Cognitive Load Co-Worker//EN",
        "CALSCALE:GREGORIAN",
    ]
    if tzid:
        lines.append(f"X-WR-TIMEZONE:{tzid}")
    for item in plan.items:
        if cursor >= horizon_epoch:
            break
        if item.action in BLOCK_DURATIONS:
            duration = BLOCK_DURATIONS[item.action]
            dur_seconds = duration * 60.0
            if all_busy:
                cursor = _find_next_free_slot(cursor, dur_seconds, all_busy)
            lines += [
                "BEGIN:VEVENT",
                f"UID:loadguard-{plan.plan_id or 'plan'}-{item.position}@loadguard",
                f"DTSTAMP:{dtstamp}",
                f"DTSTART:{_ics_datetime(cursor)}",
                f"DTEND:{_ics_datetime(cursor + dur_seconds)}",
                _ics_fold(f"SUMMARY:{_ics_escape(item.title)}"),
            ]
            if item.rationale:
                lines.append(_ics_fold(f"DESCRIPTION:{_ics_escape(item.rationale)}"))
            if item.action == "focus_block" and alarm_minutes is not None and alarm_minutes > 0:
                alarm_note = f"Focus block starts in {alarm_minutes:g} minutes"
                lines += [
                    "BEGIN:VALARM",
                    "ACTION:DISPLAY",
                    _ics_fold(f"DESCRIPTION:{_ics_escape(alarm_note)}"),
                    f"TRIGGER:{_ics_alarm_trigger(alarm_minutes)}",
                    "END:VALARM",
                ]
            lines.append("END:VEVENT")
            cursor += dur_seconds
        elif item.action == "do":
            if item.task_id:
                duration = task_durations.get(item.task_id, 30.0)
                dur_seconds = duration * 60.0
                if all_busy:
                    cursor = _find_next_free_slot(cursor, dur_seconds, all_busy)
                lines += [
                    "BEGIN:VEVENT",
                    f"UID:loadguard-{plan.plan_id or 'plan'}-{item.position}@loadguard",
                    f"DTSTAMP:{dtstamp}",
                    f"DTSTART:{_ics_datetime(cursor)}",
                    f"DTEND:{_ics_datetime(cursor + dur_seconds)}",
                    _ics_fold(f"SUMMARY:{_ics_escape(item.title)}"),
                    "CATEGORIES:LOADGUARD-TASK",
                ]
                if item.rationale:
                    lines.append(_ics_fold(f"DESCRIPTION:{_ics_escape(item.rationale)}"))
                lines.append("END:VEVENT")
                cursor += dur_seconds
        elif item.action in ("delegate", "batch"):
            dur_seconds = MINOR_STEP_MINUTES * 60.0
            if all_busy:
                cursor = _find_next_free_slot(cursor, dur_seconds, all_busy)
            lines += [
                "BEGIN:VEVENT",
                f"UID:loadguard-{plan.plan_id or 'plan'}-{item.position}@loadguard",
                f"DTSTAMP:{dtstamp}",
                f"DTSTART:{_ics_datetime(cursor)}",
                f"DTEND:{_ics_datetime(cursor + dur_seconds)}",
                _ics_fold(f"SUMMARY:{_ics_escape(item.title)}"),
                "CATEGORIES:LOADGUARD-HANDOFF",
            ]
            if item.rationale:
                lines.append(_ics_fold(f"DESCRIPTION:{_ics_escape(item.rationale)}"))
            lines.append("END:VEVENT")
            cursor += dur_seconds
        else:
            warnings.warn(
                f"unknown plan action {item.action!r}; skipped in calendar export",
                stacklevel=2,
            )
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
        decision=decision
        if decision in APPROVAL_DECISIONS or decision == "pending"
        else "rejected",
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
    try:
        path.unlink()
    except FileNotFoundError:
        return 0
    return count
