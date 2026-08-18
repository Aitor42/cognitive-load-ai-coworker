"""Benchmark CLI.

Usage:
    python demo/benchmark.py [path/to/events.jsonl]
    python demo/benchmark.py --pilot events.jsonl --outcome outcome.jsonl

Prints objective, reportable metrics for the LoadGuard pipeline. The ``--pilot``
mode compares baseline vs. projected vs. observed (when outcome events are
supplied), so results are reported honestly as measured or as projection.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loadguard.benchmark import run_benchmark, run_pilot_evaluation  # noqa: E402
from loadguard.sample_data import sample_tasks  # noqa: E402
from loadguard.signals import load_events  # noqa: E402

SAMPLE = Path(__file__).resolve().parent / "sample_events.jsonl"

KIND_LABELS = {
    "context_switch": "context switches",
    "meeting": "meetings",
    "notification": "notifications",
    "focus_block": "focus blocks",
}


def _print_pilot(path: Path, outcome: Path | None) -> None:
    events = load_events(path)
    outcome_events = load_events(outcome) if outcome else None
    e = run_pilot_evaluation(events, sample_tasks(), outcome_events=outcome_events)

    print("=" * 60)
    print(" LoadGuard — Pilot Evaluation (baseline / projected / observed)")
    print("=" * 60)
    for phase in (e.baseline, e.projected):
        print(f" {phase.label:10s} score {phase.score:5.1f}  [{phase.level}]")
    if e.observed is not None:
        print(f" {'observed':10s} score {e.observed.score:5.1f}  [{e.observed.level}]")
        if e.load_delta is not None:
            print(f" load delta : {e.load_delta:+.1f} points (baseline -> observed)")
    else:
        print(" observed   : NOT MEASURED (no --outcome events supplied)")
    print()
    if e.has_observed:
        if e.notification_reduction_pct is not None:
            print(
                f" Interruptions during focus blocks : {e.notification_reduction_pct:.1f}% reduction"
            )
        if e.context_switch_reduction_pct is not None:
            print(
                f" Context switches                  : {e.context_switch_reduction_pct:.1f}% reduction"
            )
        if e.focus_minutes_gained is not None:
            print(f" Focus minutes gained              : {e.focus_minutes_gained:+.0f} min")
        print(
            f" Tasks completed                   : {e.tasks_completed} "
            f"({e.priority_tasks_completed} priority)"
        )
    if e.acceptance_rate is not None:
        print(f" Recommendations accepted          : {e.acceptance_rate:.0%}")
    print()
    print(f" {e.summary}")


def main() -> None:
    argv = sys.argv[1:]
    if "--pilot" in argv:
        i = argv.index("--pilot")
        path = Path(argv[i + 1]) if i + 1 < len(argv) else SAMPLE
        outcome = None
        if "--outcome" in argv:
            outcome = Path(argv[argv.index("--outcome") + 1])
        _print_pilot(path, outcome)
        return

    path = Path(argv[0]) if argv else SAMPLE
    b = run_benchmark(load_events(path), sample_tasks())

    print("=" * 60)
    print(" LoadGuard — Benchmark Report")
    print("=" * 60)
    print(f" Window          : {b.window_minutes} min  ({b.n_events} events)")
    for kind, label in KIND_LABELS.items():
        print(f"   - {label:16s} {b.signal_counts[kind]}")
    print()
    print(" Cognitive Load Score")
    print(f"   before : {b.before_score:5.1f}  [{b.before_level}]")
    print(f"   after  : {b.after_score:5.1f}  [{b.after_level}]")
    print(f"   delta  : -{b.reduction_points} points  ({b.reduction_pct}% reduction)")
    print()
    print(" Plan composition")
    for action, count in sorted(b.plan_counts.items()):
        print(f"   - {action:12s} {count}")
    print()
    print(f" Interruptions eliminated (estimated): {b.interruptions_eliminated}")
    print()
    print(" Signal metrics (per hour / ratios)")
    for k, v in b.metrics.items():
        print(f"   - {k:28s} {v}")


if __name__ == "__main__":
    main()
