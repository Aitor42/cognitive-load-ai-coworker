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

# Safety cap for open-ended recurrence rules (no COUNT/UNTIL) so an unbounded
# RRULE cannot loop forever when capturing signals.
MAX_RRULE_OCCURRENCES = 366

# RFC 5545 weekday codes -> Monday-based ``date.weekday()`` values.
_WEEKDAY_CODES = {"MO": 0, "TU": 1, "WE": 2, "TH": 3, "FR": 4, "SA": 5, "SU": 6}


def _parse_ics_datetime(value: str, tzid: str | None = None) -> float:
    """Parse an ICS DATE-TIME value to epoch seconds (UTC).

    Handles the ``Z`` suffix, explicit numeric offsets (e.g. ``+0200``) and a
    ``TZID`` (IANA name, e.g. ``Europe/Madrid``) via the stdlib ``zoneinfo``.
    A ``TZID`` that cannot be resolved (e.g. a Windows timezone name) falls
    back to treating the wall time as UTC rather than crashing the parse.
    """
    value = value.strip()
    if value.endswith("Z"):
        dt = datetime.strptime(value[:-1], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
        return dt.timestamp()
    try:
        # Explicit numeric offset, e.g. 20260817T090000+0200.
        return datetime.strptime(value, "%Y%m%dT%H%M%S%z").timestamp()
    except ValueError:
        pass
    dt = datetime.strptime(value, "%Y%m%dT%H%M%S")
    if tzid:
        try:
            return dt.replace(tzinfo=ZoneInfo(tzid)).timestamp()
        except (ZoneInfoNotFoundError, ValueError, OSError):
            pass
    return dt.replace(tzinfo=timezone.utc).timestamp()


def _parse_ics_date(value: str) -> float:
    """Parse an all-day ICS DATE value (YYYYMMDD) to epoch seconds (UTC midnight)."""
    dt = datetime.strptime(value.strip(), "%Y%m%d").replace(tzinfo=timezone.utc)
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
    m = re.match(
        r"P(?:(?P<days>\d+)D)?(?:T(?:(?P<hours>\d+)H)?"
        r"(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?)?",
        value.strip(),
    )
    if not m or not any(m.groups()):
        return 60.0  # no numeric components matched; fall back to 1 hour
    days = int(m.group("days") or 0)
    h = int(m.group("hours") or 0)
    minutes = int(m.group("minutes") or 0)
    s = int(m.group("seconds") or 0)
    return days * 1440 + h * 60 + minutes + s / 60.0


def _unfold(text: str) -> str:
    lines = text.splitlines()
    unfolded: list[str] = []
    for line in lines:
        if line.startswith(" ") and unfolded:
            unfolded[-1] += line[1:]
        else:
            unfolded.append(line)
    return "\n".join(unfolded)


def _vevents(path: Path) -> list[tuple[dict[str, str], dict[str, str | None], bool]]:
    """Return each VEVENT's properties, per-property TZID, and all-day flag."""
    text = _unfold(path.read_text(encoding="utf-8"))
    out: list[tuple[dict[str, str], dict[str, str | None], bool]] = []
    for block in re.findall(r"BEGIN:VEVENT(.*?)END:VEVENT", text, re.S):
        props: dict[str, str] = {}
        tzids: dict[str, str | None] = {}
        all_day = False
        for line in block.splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            name_part, _, value = line.partition(":")
            if "VALUE=DATE" in name_part.upper():
                all_day = True
            name = name_part.split(";")[0].upper()
            tzid = next(
                (
                    p.split("=", 1)[1]
                    for p in name_part.split(";")[1:]
                    if p.upper().startswith("TZID=")
                ),
                None,
            )
            # Repeated properties (e.g. multiple EXDATE lines) are joined so
            # no exclusion or value is lost.
            if name in props:
                props[name] += "," + value.strip()
            else:
                props[name] = value.strip()
            tzids[name] = tzid
        if "DTSTART" not in props:
            continue
        out.append((props, tzids, all_day))
    return out


def _is_absence(props: dict[str, str], all_day: bool = False) -> bool:
    """Detect an out-of-office / vacation / leave event from its properties.

    A summary keyword always marks an absence. Microsoft exporters also flag
    out-of-office with ``X-MICROSOFT-CDO-BUSYSTATUS:OOF``, and Google marks
    all-day events as ``TRANSP:TRANSPARENT``. The transparent signal only
    counts when the event is all-day, so free/transparent reminders (e.g. a
    birthday) are not mistaken for leave.
    """
    summary = props.get("SUMMARY", "").lower()
    if any(k in summary for k in ABSENCE_SUMMARY_KEYWORDS):
        return True
    if props.get("X-MICROSOFT-CDO-BUSYSTATUS", "").upper() == "OOF":
        return True
    return all_day and props.get("TRANSP", "").upper() == "TRANSPARENT"


def _absence_kind(props: dict[str, str]) -> str:
    summary = props.get("SUMMARY", "").lower()
    if any(k in summary for k in VACATION_KEYWORDS):
        return VACATION
    return LEAVE


def _absence_end(
    props: dict[str, str],
    start: float,
    all_day: bool,
    tzids: dict[str, str | None] | None = None,
) -> float:
    """Resolve an event's end time, treating all-day DTEND as exclusive."""
    tzids = tzids or {}
    if "DTEND" in props:
        if all_day:
            return _parse_ics_date(props["DTEND"]) - 1.0
        return _parse_ics_datetime(props["DTEND"], tzids.get("DTEND"))
    if all_day:
        # All-day events span the whole day regardless of any DURATION.
        return start + 86400.0 - 1.0
    if "DURATION" in props:
        return start + _parse_duration(props["DURATION"]) * 60.0
    return start + 3600.0


def _parse_rrule(value: str) -> dict[str, str]:
    """Parse an RFC 5545 RRULE string into a {part: value} dict."""
    return {
        part.split("=", 1)[0]: part.split("=", 1)[1]
        for part in value.split(";")
        if "=" in part
    }


def _parse_ics_date_or_datetime(value: str, tzid: str | None = None) -> float:
    """Parse an ICS DATE (YYYYMMDD) or DATE-TIME value (for UNTIL / EXDATE)."""
    value = value.strip()
    if len(value) == 8 and value.isdigit():
        return _parse_ics_date(value)
    return _parse_ics_datetime(value, tzid)


def _expand_rrule(
    start_epoch: float,
    rrule: dict[str, str],
    until_epoch: float | None = None,
    exdates: set[float] | None = None,
) -> list[float]:
    """Expand an RRULE into occurrence start epochs (UTC), best-effort subset.

    Supports ``FREQ`` of DAILY / WEEKLY / MONTHLY / YEARLY with ``INTERVAL``,
    ``COUNT``, ``UNTIL`` (date or date-time), ``BYDAY`` for WEEKLY, and
    ``EXDATE`` exclusion. Occurrences before ``DTSTART`` are skipped and
    open-ended rules are capped at ``MAX_RRULE_OCCURRENCES``.
    """
    freq = rrule.get("FREQ", "").upper()
    if freq not in ("DAILY", "WEEKLY", "MONTHLY", "YEARLY"):
        return [start_epoch]
    try:
        interval = max(int(rrule.get("INTERVAL", "1")), 1)
    except ValueError:
        interval = 1
    try:
        count = int(rrule["COUNT"]) if "COUNT" in rrule else None
    except ValueError:
        count = None
    byday = [d.upper() for d in rrule.get("BYDAY", "").split(",") if d]
    if byday and not all(d in _WEEKDAY_CODES for d in byday):
        byday = []
    limit = count if count is not None else MAX_RRULE_OCCURRENCES

    start_dt = datetime.fromtimestamp(start_epoch, tz=timezone.utc)
    start_date = start_dt.date()

    def ts(d: date) -> float:
        return datetime(
            d.year,
            d.month,
            d.day,
            start_dt.hour,
            start_dt.minute,
            start_dt.second,
            tzinfo=timezone.utc,
        ).timestamp()

    days: list[date] = []
    if freq == "DAILY":
        for i in range(limit):
            d = start_date + timedelta(days=i * interval)
            if until_epoch is not None and ts(d) > until_epoch:
                break
            days.append(d)
    elif freq == "WEEKLY":
        week_start = start_date - timedelta(days=start_date.weekday())
        if byday:
            offsets = sorted(_WEEKDAY_CODES[b] for b in byday)
            week = 0
            while len(days) < limit:
                base = week_start + timedelta(weeks=week * interval)
                for off in offsets:
                    d = base + timedelta(days=off)
                    if d < start_date:
                        continue
                    if until_epoch is not None and ts(d) > until_epoch:
                        excluded = exdates or set()
                        return [ts(day) for day in days if ts(day) not in excluded]
                    days.append(d)
                    if len(days) >= limit:
                        break
                week += 1
        else:
            for i in range(limit):
                d = start_date + timedelta(weeks=i * interval)
                if until_epoch is not None and ts(d) > until_epoch:
                    break
                days.append(d)
    elif freq == "MONTHLY":
        i = 0
        while len(days) < limit:
            month_index = start_date.year * 12 + (start_date.month - 1) + i * interval
            year, month = divmod(month_index, 12)
            month += 1
            d = date(year, month, min(start_date.day, monthrange(year, month)[1]))
            if until_epoch is not None and ts(d) > until_epoch:
                break
            days.append(d)
            i += 1
    else:  # YEARLY
        for i in range(limit):
            year = start_date.year + i * interval
            day = start_date.day
            if start_date.month == 2 and day == 29:
                day = min(day, monthrange(year, 2)[1])
            d = date(year, start_date.month, day)
            if until_epoch is not None and ts(d) > until_epoch:
                break
            days.append(d)

    excluded = exdates or set()
    return [ts(d) for d in days if ts(d) not in excluded]


def _event_occurrences(
    props: dict[str, str],
    tzids: dict[str, str | None],
    all_day: bool,
) -> list[tuple[float, float]]:
    """Expand a VEVENT into (start, end) epoch pairs, applying RRULE / EXDATE."""
    start = (
        _parse_ics_date(props["DTSTART"])
        if all_day
        else _parse_ics_datetime(props["DTSTART"], tzids.get("DTSTART"))
    )
    end = _absence_end(props, start, all_day, tzids)
    rrule = props.get("RRULE")
    if not rrule:
        return [(start, end)]
    rule = _parse_rrule(rrule)
    until = None
    if "UNTIL" in rule:
        until_value = rule["UNTIL"].strip()
        if len(until_value) == 8 and until_value.isdigit():
            # A DATE-only UNTIL bounds the whole day inclusively.
            until = _parse_ics_date(until_value) + 86400.0 - 1.0
        else:
            until = _parse_ics_datetime(until_value, tzids.get("DTSTART"))
    exdates = {
        _parse_ics_date_or_datetime(v, tzids.get("EXDATE"))
        for v in props.get("EXDATE", "").split(",")
        if v
    }
    duration = end - start
    return [(s, s + duration) for s in _expand_rrule(start, rule, until, exdates)]


def parse_ics(path: Path) -> list[Event]:
    """Parse calendar VEVENTs into ``meeting`` events (minimal, dependency-free).

    All-day and absence (out-of-office / vacation) events are skipped here;
    absences are handled by ``parse_absences``.
    """
    events: list[Event] = []
    for props, tzids, all_day in _vevents(path):
        try:
            if all_day or _is_absence(props, all_day):
                continue
            for start, end in _event_occurrences(props, tzids, all_day):
                events.append(
                    Event(
                        timestamp=start,
                        kind="meeting",
                        duration_minutes=max((end - start) / 60.0, 1.0),
                        meta={"title": props.get("SUMMARY", "")},
                    )
                )
        except (ValueError, TypeError) as exc:
            # One malformed VEVENT must not abort the whole calendar.
            print(f"warning: skipping malformed calendar event: {exc}", file=sys.stderr)
    return events


def parse_absences(path: Path) -> list[Absence]:
    """Parse out-of-office / vacation events into ``Absence`` records.

    Only the fact and type of the absence are captured — never the event's
    summary text, which could contain a medical or personal reason.
    """
    absences: list[Absence] = []
    for props, tzids, all_day in _vevents(path):
        try:
            if not _is_absence(props, all_day):
                continue
            kind = _absence_kind(props)
            for start, end in _event_occurrences(props, tzids, all_day):
                absences.append(Absence(start=start, end=end, kind=kind))
        except (ValueError, TypeError) as exc:
            # One malformed VEVENT must not abort the whole calendar.
            print(f"warning: skipping malformed absence event: {exc}", file=sys.stderr)
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
