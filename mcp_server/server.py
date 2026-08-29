"""MCP server exposing LoadGuard tools to IBM Bob.

IBM Bob (the development tool) can drive LoadGuard through this MCP server —
the same pattern the strongest challenge entries use to make IBM Bob a deep part
of the project, not just a code generator.

Usage:
    pip install mcp
    python mcp_server/server.py            # start the MCP server (stdio)
    python mcp_server/server.py --self-test   # run tools directly, no MCP SDK

Add it to Bob via ``.bob/mcp.json`` (see mcp_server/README.md).
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from loadguard.actions import (  # noqa: E402
    APPROVAL_DECISIONS,
    FOCUS_ALARM_MINUTES,
    export_ics,
    export_tasks_csv,
    record_approval,
)
from loadguard.benchmark import run_benchmark, run_pilot_evaluation  # noqa: E402
from loadguard.models import Event, Task  # noqa: E402
from loadguard.sample_data import sample_tasks  # noqa: E402
from loadguard.scoring import score  # noqa: E402
from loadguard.signals import _to_epoch, compute_features, load_events, parse_event  # noqa: E402
from loadguard.workflow import run_workflow  # noqa: E402


def _to_events(payload: list[Any]) -> list[Event]:
    return [parse_event(e) if isinstance(e, dict) else e for e in payload]


def _parse_bool(val: Any, default: bool = True) -> bool:
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        lowered = val.strip().lower()
        if lowered in ("true", "1", "yes"):
            return True
        if lowered in ("false", "0", "no"):
            return False
    return bool(val)


def _to_tasks(payload: list[dict[str, Any]]) -> list[Task]:
    return [
        Task(
            id=str(t.get("id", i)),
            title=str(t.get("title", f"Task {i}")),
            priority=int(t.get("priority", 3)),
            duration_minutes=float(t.get("duration_minutes", 30.0)),
            focus_required=_parse_bool(t.get("focus_required"), True),
            deadline=_to_epoch(t["deadline"]) if t.get("deadline") is not None else None,
            status=str(t.get("status", "todo")),
        )
        for i, t in enumerate(payload)
    ]


def compute_load_score(events: list[dict]) -> dict:
    """Compute the Cognitive Load Score (0-100) and level from events."""
    try:
        return asdict(score(compute_features(_to_events(events))))
    except Exception as exc:
        return {"error": str(exc)}


def analyze_workload(events: list[dict], tasks: list[dict]) -> dict:
    """Run the full pipeline: score + plan + impact."""
    try:
        result = run_workflow(_to_events(events), _to_tasks(tasks))
        return {
            "features": asdict(result.features),
            "load_report": asdict(result.load_report),
            "plan": asdict(result.plan),
            "impact": asdict(result.impact),
        }
    except Exception as exc:
        return {"error": str(exc)}


def benchmark_workload(events: list[dict], tasks: list[dict]) -> dict:
    """Return objective benchmark metrics for the pipeline."""
    try:
        return asdict(run_benchmark(_to_events(events), _to_tasks(tasks)))
    except Exception as exc:
        return {"error": str(exc)}


def propose_plan(events: list[dict], tasks: list[dict], history: list[float] | None = None) -> dict:
    """Run the full loop: Granite proposal (gated) + guardian + baseline + impact."""
    try:
        result = run_workflow(_to_events(events), _to_tasks(tasks), history=history or None)
        return {
            "features": asdict(result.features),
            "load_report": asdict(result.load_report),
            "proposal": asdict(result.proposal) if result.proposal else None,
            "guardian": asdict(result.guardian) if result.guardian else None,
            "trend": asdict(result.trend) if result.trend else None,
            "plan": asdict(result.plan),
            "impact": asdict(result.impact),
        }
    except Exception as exc:
        return {"error": str(exc)}


def approve_plan(events: list[dict], tasks: list[dict], decision: str, feedback: str = "") -> dict:
    """Approve/reject a plan and record the decision in the audit trail."""
    try:
        decision = decision if decision in APPROVAL_DECISIONS else "rejected"
        result = run_workflow(_to_events(events), _to_tasks(tasks), approval=decision)
        record = record_approval(result.plan.plan_id, decision, feedback=feedback)
        return {
            "plan_id": result.plan.plan_id,
            "status": result.plan.status,
            "audit": record.__dict__,
        }
    except Exception as exc:
        return {"error": str(exc)}


def export_plan_ics(
    events: list[dict],
    tasks: list[dict],
    start_epoch: float | None = None,
    alarm_minutes: float | None = None,
) -> dict:
    """Render the approved plan's focus/recovery blocks as an .ics calendar.

    *alarm_minutes* sets the VALARM lead time for focus blocks: omitted uses
    the FOCUS_ALARM_MINUTES default and 0 exports without reminders.
    """
    try:
        parsed_events = _to_events(events)
        parsed_tasks = _to_tasks(tasks)
        result = run_workflow(parsed_events, parsed_tasks, approval="accepted")
        # export_ics treats None as "no alarms"; resolve the omitted case here.
        alarm = FOCUS_ALARM_MINUTES if alarm_minutes is None else alarm_minutes
        return {
            "ics": export_ics(
                result.plan,
                parsed_tasks,
                start_epoch,
                existing_events=parsed_events,
                alarm_minutes=alarm,
            )
        }
    except Exception as exc:
        return {"error": str(exc)}


def export_plan_csv(events: list[dict], tasks: list[dict]) -> dict:
    """Render the resequenced task list as CSV."""
    try:
        result = run_workflow(_to_events(events), _to_tasks(tasks))
        return {"tasks_csv": export_tasks_csv(result.plan, _to_tasks(tasks))}
    except Exception as exc:
        return {"error": str(exc)}


def pilot_evaluation(
    events: list[dict], tasks: list[dict], outcome_events: list[dict] | None = None
) -> dict:
    """Baseline vs. projected vs. observed evaluation (honest labelling)."""
    try:
        outcome = _to_events(outcome_events) if outcome_events else None
        return asdict(
            run_pilot_evaluation(_to_events(events), _to_tasks(tasks), outcome_events=outcome)
        )
    except Exception as exc:
        return {"error": str(exc)}


def self_test() -> None:
    sample = Path(__file__).resolve().parents[1] / "demo" / "sample_events.jsonl"
    events = load_events(sample)
    tasks = sample_tasks()
    out = analyze_workload([asdict(e) for e in events], [asdict(t) for t in tasks])
    print(json.dumps(out, indent=2))

    # Export variants: default reminder, custom lead time, no reminder.
    payloads = [asdict(e) for e in events]
    task_payloads = [asdict(t) for t in tasks]
    default_ics = export_plan_ics(payloads, task_payloads, start_epoch=1_700_000_000.0)["ics"]
    assert "TRIGGER:-PT10M" in default_ics, "default VALARM missing"
    custom = export_plan_ics(
        payloads, task_payloads, start_epoch=1_700_000_000.0, alarm_minutes=5.0
    )["ics"]
    assert "TRIGGER:-PT5M" in custom, "custom VALARM lead time missing"
    unalarmed = export_plan_ics(
        payloads, task_payloads, start_epoch=1_700_000_000.0, alarm_minutes=0.0
    )["ics"]
    assert "BEGIN:VALARM" not in unalarmed, "alarm_minutes=0 must disable reminders"
    print("export_plan_ics alarm variants OK", file=sys.stderr)


def main() -> None:
    if "--self-test" in sys.argv:
        self_test()
        return
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        sys.exit("Missing dependency: pip install mcp  (or use --self-test)")
    mcp = FastMCP("loadguard")
    mcp.tool()(compute_load_score)
    mcp.tool()(analyze_workload)
    mcp.tool()(benchmark_workload)
    mcp.tool()(propose_plan)
    mcp.tool()(approve_plan)
    mcp.tool()(export_plan_ics)
    mcp.tool()(export_plan_csv)
    mcp.tool()(pilot_evaluation)
    mcp.run()


if __name__ == "__main__":
    main()
