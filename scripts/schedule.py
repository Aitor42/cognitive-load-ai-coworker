"""Run LoadGuard's daily cycle: morning analysis + midday re-organization.

Cron-friendly, zero-dependency entry point. Schedule it twice a day (or use an
in-process loop); both beats are pure functions so they are safe to run from
any scheduler:

    python scripts/schedule.py --events events.jsonl
    python scripts/schedule.py --events events.jsonl --elapsed-minutes 240
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from loadguard.sample_data import sample_tasks, sample_workers  # noqa: E402
from loadguard.scheduler import run_daily_cycle  # noqa: E402
from loadguard.signals import load_events  # noqa: E402

SAMPLE = Path(__file__).resolve().parents[1] / "demo" / "sample_events.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser(description="Run LoadGuard's morning + midday cycle.")
    ap.add_argument("--events", type=Path, default=SAMPLE, help="JSONL signal stream")
    ap.add_argument("--elapsed-minutes", type=float, default=240.0)
    ap.add_argument("--total-minutes", type=float, default=480.0)
    args = ap.parse_args()

    events = load_events(args.events)
    cycle = run_daily_cycle(
        morning_events=events,
        events_so_far=events,
        tasks=sample_tasks(),
        workers=sample_workers(),
        elapsed_minutes=args.elapsed_minutes,
        total_minutes=args.total_minutes,
    )

    morning = cycle.morning
    midday = cycle.midday
    print("=" * 60)
    print(" LoadGuard — Daily Cycle")
    print("=" * 60)
    print(
        f" Morning    : score {morning.load_report.score:.0f}/100 "
        f"[{morning.load_report.level}]  ({len(morning.plan.items)} plan items)"
    )
    print(
        f" Midday     : observed {midday.observed_score:.0f} -> "
        f"projected {midday.projected_score:.0f}/100 [{midday.projected_level}]"
    )
    print(f" Reorganized: {'yes' if midday.reorganized else 'no'}")
    if midday.plan is not None:
        print(f"   new plan items: {len(midday.plan.items)}")
    print(f" Reassignment alerts: {len(midday.reassignment_alerts)}")
    print(f" {midday.rationale}")


if __name__ == "__main__":
    main()
