"""Capture real work signals into LoadGuard's JSONL format.

Replaces the synthetic sample with real, privacy-preserving signals so the
project demonstrates real-world impact on real data:

    python scripts/capture_signals.py \\
        --calendar calendar.ics \\
        --notifications notifications.log \\
        --out signals.jsonl
    python scripts/capture_signals.py \\
        --calendar calendar.ics \\
        --absences-out absences.jsonl \\
        --workers-out workers.jsonl --worker-id me --worker-name Ada

- ``--calendar``: an ICS file exported from Google Calendar / Outlook. Timed
  VEVENTs become ``meeting`` events (only start/end/duration and title are
  kept — no attendees or content).
- ``--notifications``: a text log, one per line: ``<ISO-timestamp> <source>``.
- ``--focus``: optional focus-block log, one per line:
  ``<ISO-timestamp> <minutes> <title>``.
- ``--absences-out``: write out-of-office / vacation events as ``Absence``
  records (JSONL). Only the fact and type (vacation/leave) are kept — never
  the event summary, which could contain a medical or personal reason.
- ``--workers-out``: write a single ``Worker`` record wrapping those absences,
  ready to pass to ``POST /analyze`` as ``workers``.

Only counts, durations, and opaque labels are captured — never message bodies.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from calendar import monthrange
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from loadguard.models import LEAVE, VACATION, Absence, Event, Worker  # noqa: E402

from loadguard import calendar_parser  # noqa: E402
from loadguard.calendar_parser import (  # noqa: E402
    ABSENCE_SUMMARY_KEYWORDS,
    MAX_RRULE_OCCURRENCES,
    VACATION_KEYWORDS,
    _WINDOWS_TZ_MAP,
    _WEEKDAY_CODES,
    _absence_end,
    _absence_kind,
    _classify_vevents,
    _event_occurrences,
    _expand_rrule,
    _is_absence,
    _parse_duration,
    _parse_ics_date,
    _parse_ics_date_or_datetime,
    _parse_ics_datetime,
    _parse_iso,
    _parse_rrule,
    _unfold,
    _vevents_text,
    parse_absences,
    parse_calendar_text,
    parse_ics,
)


def _vevents(path: Path) -> list[tuple[dict[str, str], dict[str, str | None], bool]]:
    return calendar_parser._vevents(path)


def parse_calendar(path: Path) -> tuple[list[Event], list[Absence]]:
    return calendar_parser._classify_vevents(_vevents(path))


def write_absences(absences: list[Absence], path: Path) -> None:
    """Write absence records as JSONL (start/end are epoch seconds)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(absences, key=lambda a: a.start)
    with open(path, "w", encoding="utf-8") as fh:
        for a in ordered:
            fh.write(
                json.dumps({"start": a.start, "end": a.end, "kind": a.kind, "note": a.note}) + "\n"
            )
    print(f"Wrote {len(ordered)} absences to {path}")


def load_absences(path: Path) -> list[Absence]:
    """Load absence records from a JSONL file, skipping corrupted lines."""
    absences: list[Absence] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            data = json.loads(line)
            absences.append(
                Absence(
                    start=float(data["start"]),
                    end=float(data["end"]),
                    kind=str(data.get("kind", "leave")),
                    note=str(data.get("note", "")),
                )
            )
        except (json.JSONDecodeError, ValueError, KeyError, TypeError):
            continue
    return absences


def write_workers(workers: list[Worker], path: Path) -> None:
    """Write worker records (with nested absences) as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for w in workers:
            fh.write(
                json.dumps(
                    {
                        "id": w.id,
                        "name": w.name,
                        "absences": [
                            {"start": a.start, "end": a.end, "kind": a.kind, "note": a.note}
                            for a in w.absences
                        ],
                    }
                )
                + "\n"
            )
    print(f"Wrote {len(workers)} worker(s) to {path}")


def load_workers(path: Path) -> list[Worker]:
    """Load worker records from a JSONL file, skipping corrupted lines."""
    workers: list[Worker] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            data = json.loads(line)
            workers.append(
                Worker(
                    id=str(data["id"]),
                    name=str(data.get("name", "")),
                    absences=[
                        Absence(
                            start=float(a["start"]),
                            end=float(a["end"]),
                            kind=str(a.get("kind", "leave")),
                            note=str(a.get("note", "")),
                        )
                        for a in data.get("absences", [])
                    ],
                )
            )
        except (json.JSONDecodeError, ValueError, KeyError, TypeError):
            continue
    return workers


def parse_notifications(path: Path) -> list[Event]:
    events: list[Event] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        events.append(
            Event(
                timestamp=_parse_iso(parts[0]),
                kind="notification",
                meta={"source": parts[1] if len(parts) > 1 else "unknown"},
            )
        )
    return events


def parse_focus(path: Path) -> list[Event]:
    events: list[Event] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 2)
        ts = _parse_iso(parts[0])
        minutes = float(parts[1]) if len(parts) > 1 else 30.0
        title = parts[2] if len(parts) > 2 else ""
        events.append(
            Event(timestamp=ts, kind="focus_block", duration_minutes=minutes, meta={"title": title})
        )
    return events


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime(ISO)


def write_jsonl(events: list[Event], path: Path) -> None:
    ordered = sorted(events, key=lambda e: e.timestamp)
    with open(path, "w", encoding="utf-8") as fh:
        for e in ordered:
            fh.write(
                json.dumps(
                    {
                        "timestamp": _iso(e.timestamp),
                        "kind": e.kind,
                        "duration_minutes": e.duration_minutes,
                        "meta": e.meta,
                    }
                )
                + "\n"
            )
    print(f"Wrote {len(ordered)} events to {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Capture real work signals for LoadGuard.")
    ap.add_argument("--calendar", type=Path, help="ICS file (meetings + absences)")
    ap.add_argument("--notifications", type=Path, help="notification log")
    ap.add_argument("--focus", type=Path, help="focus-block log")
    ap.add_argument("--out", type=Path, default=Path("signals.jsonl"))
    ap.add_argument("--absences-out", type=Path, help="write absences to a JSONL file")
    ap.add_argument(
        "--workers-out", type=Path, help="write a worker record (with absences) as JSONL"
    )
    ap.add_argument("--worker-id", default="me", help="worker id for --workers-out")
    ap.add_argument("--worker-name", default="", help="worker name for --workers-out")
    args = ap.parse_args()

    events: list[Event] = []
    absences: list[Absence] = []
    if args.calendar:
        # Single read of the ICS file: meetings and absences together.
        meetings, absences = parse_calendar(args.calendar)
        events += meetings
    if args.notifications:
        events += parse_notifications(args.notifications)
    if args.focus:
        events += parse_focus(args.focus)

    if not events and args.absences_out is None and args.workers_out is None:
        ap.error("provide at least one of --calendar, --notifications, --focus")

    if events:
        write_jsonl(events, args.out)
    if args.absences_out:
        write_absences(absences, args.absences_out)
    if args.workers_out:
        write_workers(
            [Worker(id=args.worker_id, name=args.worker_name, absences=absences)],
            args.workers_out,
        )


if __name__ == "__main__":
    main()
