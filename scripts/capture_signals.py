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
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from loadguard.models import LEAVE, VACATION, Absence, Event, Worker  # noqa: E402

ISO = "%Y-%m-%dT%H:%M:%SZ"

# Summary substrings (lowercased) that mark a calendar event as an absence.
ABSENCE_SUMMARY_KEYWORDS = (
    "out of office",
    "out-of-office",
    "ooo",
    "vacation",
    "vacaciones",
    "on leave",
    "annual leave",
    "baja",
    "permiso",
)
VACATION_KEYWORDS = ("vacation", "vacaciones")


def _parse_ics_datetime(value: str) -> float:
    value = value.strip()
    if value.endswith("Z"):
        dt = datetime.strptime(value[:-1], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    else:
        dt = datetime.strptime(value, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _parse_ics_date(value: str) -> float:
    """Parse an all-day ICS DATE value (YYYYMMDD) to epoch seconds (UTC midnight)."""
    value = value.strip()
    dt = datetime.strptime(value[:8], "%Y%m%d").replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _parse_iso(value: str) -> float:
    value = value.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _parse_duration(value: str) -> float:
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", value.strip())
    if not m or not any(m.groups()):
        return 60.0  # no numeric components matched; fall back to 1 hour
    h = int(m.group(1) or 0)
    minutes = int(m.group(2) or 0)
    s = int(m.group(3) or 0)
    return h * 60 + minutes + s / 60.0


def _unfold(text: str) -> str:
    lines = text.splitlines()
    unfolded: list[str] = []
    for line in lines:
        if line.startswith(" ") and unfolded:
            unfolded[-1] += line[1:]
        else:
            unfolded.append(line)
    return "\n".join(unfolded)


def _vevents(path: Path) -> list[tuple[dict[str, str], bool]]:
    """Return each VEVENT's properties plus an all-day flag (minimal parser)."""
    text = _unfold(path.read_text(encoding="utf-8"))
    out: list[tuple[dict[str, str], bool]] = []
    for block in re.findall(r"BEGIN:VEVENT(.*?)END:VEVENT", text, re.S):
        props: dict[str, str] = {}
        all_day = False
        for line in block.splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            name_part, _, value = line.partition(":")
            if "VALUE=DATE" in name_part.upper():
                all_day = True
            props[name_part.split(";")[0].upper()] = value.strip()
        if "DTSTART" not in props:
            continue
        out.append((props, all_day))
    return out


def _is_absence(props: dict[str, str]) -> bool:
    """Detect an out-of-office / vacation / leave event from its properties."""
    summary = props.get("SUMMARY", "").lower()
    if any(k in summary for k in ABSENCE_SUMMARY_KEYWORDS):
        return True
    return props.get("X-MICROSOFT-CDO-BUSYSTATUS", "").upper() == "OOF"


def _absence_kind(props: dict[str, str]) -> str:
    summary = props.get("SUMMARY", "").lower()
    if any(k in summary for k in VACATION_KEYWORDS):
        return VACATION
    return LEAVE


def _absence_end(props: dict[str, str], start: float, all_day: bool) -> float:
    """Resolve an absence's end time, treating all-day DTEND as exclusive."""
    if "DTEND" in props:
        if all_day:
            return _parse_ics_date(props["DTEND"]) - 1.0
        return _parse_ics_datetime(props["DTEND"])
    if "DURATION" in props:
        return start + _parse_duration(props["DURATION"]) * 60.0
    if all_day:
        return start + 86400.0 - 1.0
    return start + 3600.0


def parse_ics(path: Path) -> list[Event]:
    """Parse calendar VEVENTs into ``meeting`` events (minimal, dependency-free).

    All-day and absence (out-of-office / vacation) events are skipped here;
    absences are handled by ``parse_absences``.
    """
    events: list[Event] = []
    for props, all_day in _vevents(path):
        if all_day or _is_absence(props):
            continue
        start = _parse_ics_datetime(props["DTSTART"])
        if "DTEND" in props:
            duration = (_parse_ics_datetime(props["DTEND"]) - start) / 60.0
        elif "DURATION" in props:
            duration = _parse_duration(props["DURATION"])
        else:
            duration = 60.0
        events.append(
            Event(
                timestamp=start,
                kind="meeting",
                duration_minutes=max(duration, 1.0),
                meta={"title": props.get("SUMMARY", "")},
            )
        )
    return events


def parse_absences(path: Path) -> list[Absence]:
    """Parse out-of-office / vacation events into ``Absence`` records.

    Only the fact and type of the absence are captured — never the event's
    summary text, which could contain a medical or personal reason.
    """
    absences: list[Absence] = []
    for props, all_day in _vevents(path):
        if not _is_absence(props):
            continue
        start = (
            _parse_ics_date(props["DTSTART"]) if all_day else _parse_ics_datetime(props["DTSTART"])
        )
        absences.append(
            Absence(start=start, end=_absence_end(props, start, all_day), kind=_absence_kind(props))
        )
    return absences


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
    if args.calendar:
        events += parse_ics(args.calendar)
    if args.notifications:
        events += parse_notifications(args.notifications)
    if args.focus:
        events += parse_focus(args.focus)

    absences: list[Absence] = parse_absences(args.calendar) if args.calendar else []

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
