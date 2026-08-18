"""Capture real work signals into LoadGuard's JSONL format.

Replaces the synthetic sample with real, privacy-preserving signals so the
project demonstrates real-world impact on real data:

    python scripts/capture_signals.py \\
        --calendar calendar.ics \\
        --notifications notifications.log \\
        --out signals.jsonl

- ``--calendar``: an ICS file exported from Google Calendar / Outlook. Each
  VEVENT becomes a ``meeting`` event (only start/end/duration and title are
  kept — no attendees or content).
- ``--notifications``: a text log, one per line: ``<ISO-timestamp> <source>``.
- ``--focus``: optional focus-block log, one per line:
  ``<ISO-timestamp> <minutes> <title>``.

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

from loadguard.models import Event  # noqa: E402

ISO = "%Y-%m-%dT%H:%M:%SZ"


def _parse_ics_datetime(value: str) -> float:
    value = value.strip()
    if value.endswith("Z"):
        dt = datetime.strptime(value[:-1], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    else:
        dt = datetime.strptime(value, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
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


def parse_ics(path: Path) -> list[Event]:
    """Parse calendar VEVENTs into ``meeting`` events (minimal, dependency-free)."""
    text = _unfold(path.read_text(encoding="utf-8"))
    events: list[Event] = []
    for block in re.findall(r"BEGIN:VEVENT(.*?)END:VEVENT", text, re.S):
        props: dict[str, str] = {}
        for line in block.splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            name_part, _, value = line.partition(":")
            if "VALUE=DATE" in name_part.upper():
                continue  # skip all-day events
            props[name_part.split(";")[0].upper()] = value.strip()
        if "DTSTART" not in props:
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
    ap.add_argument("--calendar", type=Path, help="ICS file (meetings)")
    ap.add_argument("--notifications", type=Path, help="notification log")
    ap.add_argument("--focus", type=Path, help="focus-block log")
    ap.add_argument("--out", type=Path, default=Path("signals.jsonl"))
    args = ap.parse_args()

    events: list[Event] = []
    if args.calendar:
        events += parse_ics(args.calendar)
    if args.notifications:
        events += parse_notifications(args.notifications)
    if args.focus:
        events += parse_focus(args.focus)

    if not events:
        ap.error("provide at least one of --calendar, --notifications, --focus")

    write_jsonl(events, args.out)


if __name__ == "__main__":
    main()
