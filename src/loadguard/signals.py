"""Signal ingestion and feature extraction.

This module turns a stream of lightweight, privacy-preserving events into the
explanable FeatureSet consumed by the scoring engine.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .models import CONTEXT_SWITCH, FOCUS_BLOCK, MEETING, NOTIFICATION, Event, FeatureSet

# Rapid-fire threshold: context switches closer than this are considered
# frantic multitasking, even outside meetings.
RAPID_SWITCH_SECONDS = 120.0


def _to_epoch(value: Any) -> float:
    """Coerce an ISO-8601 string, numeric timestamp, or datetime to epoch seconds."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.timestamp()
    if isinstance(value, str):
        ts = value.strip()
        # Replace trailing 'Z' with '+00:00' for fromisoformat compatibility.
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    raise ValueError(f"cannot parse timestamp: {value!r}")


def parse_event(raw: dict[str, Any]) -> Event:
    """Build an Event from a dict (as read from JSONL or the REST API)."""
    return Event(
        timestamp=_to_epoch(raw["timestamp"]),
        kind=raw["kind"],
        duration_minutes=float(raw.get("duration_minutes", 0.0)),
        meta=raw.get("meta", {}),
    )


def load_events(path: str | Path) -> list[Event]:
    """Load events from a JSONL file (one JSON object per line)."""
    events: list[Event] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            events.append(parse_event(json.loads(line)))
    return events


def _window_minutes(events: Iterable[Event]) -> float:
    """Total time span covered by the events, with a 60-minute floor."""
    stamps = [e.timestamp for e in events]
    if not stamps:
        return 60.0
    span = max(stamps) - min(stamps)
    return max(span / 60.0, 60.0)  # at least one hour


def compute_features(events: Iterable[Event], window_minutes: float | None = None) -> FeatureSet:
    """Aggregate events into a FeatureSet.

    All rates are normalized per hour; ratios are expressed over the analysis
    window so the resulting features are window-size independent.
    """
    events = list(events)
    window = (
        window_minutes
        if (window_minutes is not None and window_minutes > 0.0)
        else _window_minutes(events)
    )
    window_hours = window / 60.0

    switches = 0
    notifications = 0
    meeting_minutes = 0.0
    focus_minutes = 0.0

    meeting_intervals: list[tuple[float, float]] = []
    switch_times: list[float] = []

    for e in events:
        if e.kind == CONTEXT_SWITCH:
            switches += 1
            switch_times.append(e.timestamp)
        elif e.kind == NOTIFICATION:
            notifications += 1
        elif e.kind == MEETING:
            meeting_minutes += e.duration_minutes
            meeting_intervals.append((e.timestamp, e.timestamp + e.duration_minutes * 60.0))
        elif e.kind == FOCUS_BLOCK:
            focus_minutes += e.duration_minutes

    # --- Multitasking proxy (two complementary signals) ---

    # 1. Meeting-overlap: share of context switches during a meeting.
    multitask_switches = 0
    for t in switch_times:
        if any(start <= t <= end for start, end in meeting_intervals):
            multitask_switches += 1
    meeting_multitask = (multitask_switches / switches) if switches else 0.0

    # 2. Rapid-fire: switches clustered within 2 minutes indicate frantic
    #    context-switching even outside meetings.
    sorted_times = sorted(switch_times)
    rapid_pairs = sum(
        1
        for i in range(1, len(sorted_times))
        if sorted_times[i] - sorted_times[i - 1] < RAPID_SWITCH_SECONDS
    )
    rapid_ratio = (rapid_pairs / max(switches - 1, 1)) if switches > 1 else 0.0

    # Combined: whichever signal is stronger.
    multitasking_index = max(meeting_multitask, rapid_ratio)

    return FeatureSet(
        context_switches_per_hour=switches / window_hours,
        meeting_ratio=max(0.0, min(meeting_minutes / window, 1.0)),
        notification_rate=notifications / window_hours,
        focus_ratio=max(0.0, min(focus_minutes / window, 1.0)),
        multitasking_index=max(0.0, min(multitasking_index, 1.0)),
    )
