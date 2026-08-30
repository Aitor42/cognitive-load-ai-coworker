"""FastAPI layer (optional).

Exposes the pipeline over HTTP and serves the interactive dashboard. The core
logic remains importable without FastAPI installed.

Endpoints close the loop: analyze -> approve/reject -> export (.ics, tasks) ->
feedback, plus a local history store for the personalized baseline and a
privacy endpoint describing exactly what is captured.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse, Response
from pydantic import BaseModel, Field

from .actions import (
    APPROVAL_DECISIONS,
    FOCUS_ALARM_MINUTES,
    clear_audit,
    export_ics,
    export_tasks_csv,
    is_valid_transition,
    load_audit,
    new_plan_id,
    record_approval,
)
from .baseline import append_score, clear_history, load_history
from .benchmark import run_pilot_evaluation
from .calendar_parser import parse_calendar_text
from .config import get_guardian_model, get_model
from .guardian import validate_plan
from .impact import estimate_impact
from .models import Absence, LoadReport, Plan, PlanItem, Task, Worker
from .projection import run_midday_review
from .sample_data import sample_tasks, sample_workers
from .signals import _to_epoch, compute_features, load_events, parse_event
from .workflow import WorkflowResult, run_workflow

app = FastAPI(
    title="LoadGuard",
    description="Cognitive-Load-Aware AI Co-Worker — IBM AI Builders Challenge 2026",
    version="0.3.0",
)

# Local data directory (surfaced to the user; deletable from the dashboard).
_DATA_DIR = Path(__file__).resolve().parents[2] / ".loadguard"
HISTORY_PATH = _DATA_DIR / "history.jsonl"
AUDIT_PATH = _DATA_DIR / "audit.jsonl"
PLANS_PATH = _DATA_DIR / "plans.json"


def _load_persisted_plans() -> dict[str, dict[str, Any]]:
    if PLANS_PATH.exists():
        try:
            return json.loads(PLANS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


# Plan store (persisted in .loadguard/plans.json): plan_id -> {"payload": ..., "tasks": [...], "events": [...]}
MAX_PERSISTED_PLANS = 100
PLANS: dict[str, dict[str, Any]] = _load_persisted_plans()
_PLANS_LOCK = threading.Lock()


def _trim_plans() -> None:
    while len(PLANS) > MAX_PERSISTED_PLANS:
        oldest_key = next(iter(PLANS))
        del PLANS[oldest_key]


def _persist_plans() -> None:
    try:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        tmp = PLANS_PATH.with_suffix(f".tmp.{threading.get_ident()}")
        tmp.write_text(json.dumps(PLANS), encoding="utf-8")
        tmp.replace(PLANS_PATH)
    except Exception:
        pass


class AnalyzeRequest(BaseModel):
    events: list[dict[str, Any]]
    tasks: list[dict[str, Any]]
    window_minutes: Optional[float] = None
    history: Optional[list[float]] = None
    approval: Optional[str] = None
    workers: Optional[list[dict[str, Any]]] = None
    # Lead time (minutes) of the VALARM reminder in the exported .ics calendar.
    # None uses the server default (FOCUS_ALARM_MINUTES); 0 exports without alarms.
    alarm_minutes: Optional[float] = Field(default=None, ge=0.0)
    tz_name: Optional[str] = None
    role: Optional[str] = None
    weights: Optional[dict[str, float]] = None


class MiddayRequest(BaseModel):
    events: list[dict[str, Any]]
    tasks: list[dict[str, Any]]
    workers: Optional[list[dict[str, Any]]] = None
    elapsed_minutes: float = 240.0
    total_minutes: float = 480.0
    completed_task_ids: Optional[list[str]] = None
    alarm_minutes: Optional[float] = Field(default=None, ge=0.0)
    tz_name: Optional[str] = None
    role: Optional[str] = None
    weights: Optional[dict[str, float]] = None


class ApproveRequest(BaseModel):
    plan_id: str
    decision: str  # accepted | rejected | edited
    feedback: str = ""
    helpful: Optional[str] = None  # yes | no
    items: Optional[list[dict[str, Any]]] = None  # edited plan items


class FeedbackRequest(BaseModel):
    plan_id: str
    helpful: Optional[str] = None  # yes | no
    feedback: str = ""


class HistoryRequest(BaseModel):
    score: float = Field(ge=0.0, le=100.0)


class IngestRequest(BaseModel):
    text: str
    format: Optional[str] = None  # "ics" | "jsonl"; auto-detected when omitted


class PilotRequest(BaseModel):
    events: Optional[list[dict[str, Any]]] = None
    tasks: Optional[list[dict[str, Any]]] = None
    outcome_events: Optional[list[dict[str, Any]]] = None  # real post-plan signals


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
    """Build Task objects from API payload dicts.

    Timestamps are accepted as epoch numbers or ISO-8601 strings (the adapter
    contract documented in ``models.py``).
    """
    return [
        Task(
            id=str(t.get("id", i)),
            title=str(t.get("title", f"Task {i}")),
            priority=int(t.get("priority", 3)),
            duration_minutes=float(t.get("duration_minutes", 30.0)),
            focus_required=_parse_bool(t.get("focus_required"), True),
            deadline=_to_epoch(t["deadline"]) if t.get("deadline") is not None else None,
            status=str(t.get("status", "todo")),
            assignee=str(t["assignee"]) if t.get("assignee") else None,
        )
        for i, t in enumerate(payload)
    ]


def _to_workers(payload: list[dict[str, Any]]) -> list[Worker]:
    """Build Worker objects (with nested absences) from API payload dicts.

    Absence bounds are accepted as epoch numbers or ISO-8601 strings (the
    adapter contract documented in ``models.py``).
    """
    return [
        Worker(
            id=str(w.get("id", i)),
            name=str(w.get("name", "")),
            absences=[
                Absence(
                    start=_to_epoch(a["start"]),
                    end=_to_epoch(a["end"]),
                    kind=str(a.get("kind", "leave")),
                    note=str(a.get("note", "")),
                )
                for a in w.get("absences", [])
            ],
        )
        for i, w in enumerate(payload)
    ]


def _store_plan(
    result: WorkflowResult,
    tasks: list[Task],
    events: list[Any],
    workers: list[Worker] | None = None,
    alarm_minutes: float | None = None,
    tz_name: str | None = None,
) -> dict[str, Any]:
    """Persist a workflow result for later approval and export.

    *alarm_minutes* is the VALARM lead time for the exported calendar; ``None``
    resolves to the ``FOCUS_ALARM_MINUTES`` default (``0`` exports unalarmed).
    """
    payload = asdict(result)
    plan_id = result.plan.plan_id
    with _PLANS_LOCK:
        PLANS[plan_id] = {
            "payload": payload,
            "tasks": [asdict(t) for t in tasks],
            "events": events,
            "workers": [asdict(w) for w in workers] if workers else [],
            "alarm_minutes": FOCUS_ALARM_MINUTES if alarm_minutes is None else alarm_minutes,
            "tz_name": tz_name,
        }
        _trim_plans()
        _persist_plans()
    payload["plan_id"] = plan_id
    return payload


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _sample_events() -> list[Any]:
    """The demo day's events, used by /sample and the pilot evaluation."""
    path = Path(__file__).resolve().parents[2] / "demo" / "sample_events.jsonl"
    return load_events(path)


@app.get("/sample")
def sample() -> dict[str, Any]:
    events = _sample_events()
    return {
        "events": [asdict(e) for e in events],
        "tasks": [asdict(t) for t in sample_tasks()],
        "workers": [asdict(w) for w in sample_workers()],
    }


@app.get("/pilot")
def pilot() -> dict[str, Any]:
    """Three-phase pilot evaluation on the demo day (baseline vs. projected).

    The observed phase is only ever reported when real outcome signals are
    supplied; without them the result is explicitly labelled a projection.
    """
    result = run_pilot_evaluation(_sample_events(), sample_tasks())
    return asdict(result)


@app.post("/pilot")
def pilot_custom(req: PilotRequest) -> dict[str, Any]:
    """Pilot evaluation over uploaded signals, with real outcome events when given."""
    events = [parse_event(e) for e in req.events] if req.events is not None else _sample_events()
    tasks = _to_tasks(req.tasks) if req.tasks is not None else sample_tasks()
    outcome = [parse_event(e) for e in req.outcome_events] if req.outcome_events else None
    result = run_pilot_evaluation(events, tasks, outcome_events=outcome)
    return asdict(result)


@app.post("/ingest")
def ingest(req: IngestRequest) -> dict[str, Any]:
    """Parse raw uploaded signals (.ics calendar or .jsonl event log) into events."""
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="empty payload")
    fmt = (req.format or "").lower()
    if fmt not in ("ics", "jsonl"):
        fmt = "ics" if text.upper().startswith("BEGIN:VCALENDAR") else "jsonl"
    if fmt == "ics":
        events, _ = parse_calendar_text(text)
        return {"format": "ics", "events": [asdict(e) for e in events]}
    parsed: list[Any] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            parsed.append(parse_event(json.loads(line)))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"invalid JSONL line: {exc}") from exc
    return {"format": "jsonl", "events": [asdict(e) for e in parsed]}


@app.post("/analyze")
def analyze(req: AnalyzeRequest) -> dict[str, Any]:
    events = [parse_event(e) for e in req.events]
    tasks = _to_tasks(req.tasks)
    workers = _to_workers(req.workers or [])
    result = run_workflow(
        events,
        tasks,
        get_model(),
        req.window_minutes,
        history=req.history,
        approval=req.approval,
        guardian_model=get_guardian_model(),
        workers=workers,
        tz_name=req.tz_name,
        role=req.role,
        weights=req.weights,
    )
    payload = _store_plan(
        result,
        tasks,
        [asdict(e) for e in events],
        workers,
        alarm_minutes=req.alarm_minutes,
        tz_name=req.tz_name,
    )
    if req.role:
        payload["role"] = req.role
    return payload


@app.post("/midday")
def midday(req: MiddayRequest) -> dict[str, Any]:
    events = [parse_event(e) for e in req.events]
    tasks = _to_tasks(req.tasks)
    workers = _to_workers(req.workers or [])
    completed_ids = set(req.completed_task_ids) if req.completed_task_ids else None
    review = run_midday_review(
        events,
        tasks,
        req.elapsed_minutes,
        req.total_minutes,
        workers=workers,
        completed_task_ids=completed_ids,
        tz_name=req.tz_name,
        role=req.role,
        weights=req.weights,
    )
    result = asdict(review)
    if req.role:
        result["role"] = req.role
    if review.plan is not None:
        # Store the afternoon plan so it can be approved and exported through
        # the same endpoints as the morning plan (accept -> export .ics/.csv).
        review.plan.plan_id = review.plan.plan_id or new_plan_id()
        plan_id = review.plan.plan_id
        alarm = FOCUS_ALARM_MINUTES if req.alarm_minutes is None else req.alarm_minutes
        with _PLANS_LOCK:
            PLANS[plan_id] = {
                "payload": {
                    "load_report": asdict(review.plan.load_report),
                    "plan": asdict(review.plan),
                },
                "tasks": [asdict(t) for t in tasks],
                "events": [asdict(e) for e in events],
                "workers": [asdict(w) for w in workers],
                "alarm_minutes": alarm,
                "tz_name": req.tz_name,
            }
            _trim_plans()
            _persist_plans()
        result["plan_id"] = plan_id
    return result


@app.post("/approve")
def approve(req: ApproveRequest) -> dict[str, Any]:
    with _PLANS_LOCK:
        stored = PLANS.get(req.plan_id)
        if stored is None:
            raise HTTPException(status_code=404, detail="unknown plan_id")
        current_status = stored["payload"]["plan"].get("status", "pending")
        decision = req.decision if req.decision in APPROVAL_DECISIONS else "rejected"
        if not is_valid_transition(current_status, decision):
            raise HTTPException(
                status_code=400,
                detail=f"Cannot transition plan {req.plan_id} from terminal status {current_status!r} to {decision!r}",
            )
        if req.items is not None:
            # An edited plan replaces the stored items, but it must pass the same
            # safety gate as an original plan (no invented tasks, no critical
            # delegation, valid actions). Stored tasks remain strictly immutable.
            task_title_map = {t.get("id"): t.get("title", "") for t in stored.get("tasks", [])}
            cleaned = [
                {
                    "position": i + 1,
                    "action": item["action"],
                    "task_id": item.get("task_id"),
                    "title": item.get("title")
                    or task_title_map.get(item.get("task_id"))
                    or item["action"],
                    "rationale": item.get("rationale", ""),
                    "suggested_assignees": item.get("suggested_assignees", []),
                    "delegate_to": item.get("delegate_to"),
                }
                for i, item in enumerate(req.items)
                if item.get("action") in ("do", "delegate", "focus_block", "break", "batch")
            ]
            plan = _plan_from_payload(stored["payload"])
            plan.items = [
                PlanItem(
                    position=c["position"],
                    action=c["action"],
                    task_id=c["task_id"],
                    title=c["title"],
                    rationale=c["rationale"],
                    suggested_assignees=c.get("suggested_assignees", []),
                    delegate_to=c.get("delegate_to"),
                )
                for c in cleaned
            ]
            guard = validate_plan(
                plan,
                _to_tasks(stored["tasks"]),
                plan.note,
                allow_custom_titles=True,
            )
            if not guard.passed:
                raise HTTPException(
                    status_code=400,
                    detail=f"edited plan failed the safety gate: {guard.summary()}",
                )
            stored["payload"]["plan"]["items"] = cleaned
            _persist_plans()
        impact: dict[str, Any] | None = None
        if req.items is not None:
            # The edited plan changes what will actually be applied, so the
            # projected before/after score must be recomputed from the stored
            # signals rather than showing the original (stale) estimate.
            features = compute_features([parse_event(e) for e in stored["events"]])
            impact = asdict(estimate_impact(features, plan))
        decision = req.decision if req.decision in APPROVAL_DECISIONS else "rejected"
        record = record_approval(
            req.plan_id,
            decision,
            feedback=req.feedback,
            helpful=req.helpful or "",
            path=AUDIT_PATH,
        )
        stored["payload"]["plan"]["status"] = decision
        return {
            "plan_id": req.plan_id,
            "status": decision,
            "audit": record.__dict__,
            "impact": impact,
        }


@app.post("/feedback")
def feedback(req: FeedbackRequest) -> dict[str, Any]:
    with _PLANS_LOCK:
        stored = PLANS.get(req.plan_id)
        if stored is None:
            raise HTTPException(status_code=404, detail="unknown plan_id")
        record = record_approval(
            req.plan_id,
            stored["payload"]["plan"].get("status", "pending"),
            feedback=req.feedback,
            helpful=req.helpful or "",
            path=AUDIT_PATH,
        )
        return {"plan_id": req.plan_id, "audit": record.__dict__}


@app.get("/plan/{plan_id}/export.ics")
def export_plan_ics(plan_id: str, tzid: Optional[str] = None) -> Response:
    with _PLANS_LOCK:
        stored = PLANS.get(plan_id)
        if stored is None:
            raise HTTPException(status_code=404, detail="unknown plan_id")
        plan = _plan_from_payload(stored["payload"])
        tasks = _to_tasks(stored["tasks"])
        existing_events = [parse_event(e) for e in stored.get("events", [])]
        alarm_minutes = stored["alarm_minutes"]
        tz_name = tzid or stored.get("tz_name")
    return Response(
        content=export_ics(
            plan,
            tasks,
            existing_events=existing_events,
            alarm_minutes=alarm_minutes,
            tzid=tzid,
            tz_name=tz_name,
        ),
        media_type="text/calendar",
        headers={"Content-Disposition": f'attachment; filename="loadguard-{plan_id}.ics"'},
    )


@app.get("/plan/{plan_id}/export.csv")
def export_plan_csv(plan_id: str) -> PlainTextResponse:
    with _PLANS_LOCK:
        stored = PLANS.get(plan_id)
        if stored is None:
            raise HTTPException(status_code=404, detail="unknown plan_id")
        plan = _plan_from_payload(stored["payload"])
        tasks = _to_tasks(stored["tasks"])
    return PlainTextResponse(
        content=export_tasks_csv(plan, tasks),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="loadguard-{plan_id}.csv"'},
    )


@app.get("/history")
def history() -> dict[str, Any]:
    scores = load_history(HISTORY_PATH)
    return {"history": scores}


@app.post("/history")
def record_history(req: HistoryRequest) -> dict[str, Any]:
    append_score(HISTORY_PATH, req.score)
    return {"history": load_history(HISTORY_PATH)}


@app.delete("/history")
def delete_history() -> dict[str, Any]:
    removed = clear_history(HISTORY_PATH)
    return {"removed": removed}


@app.get("/audit")
def audit() -> dict[str, Any]:
    return {"records": load_audit(AUDIT_PATH)}


@app.delete("/audit")
def delete_audit() -> dict[str, Any]:
    removed = clear_audit(AUDIT_PATH)
    return {"removed": removed}


@app.get("/privacy")
def privacy() -> dict[str, Any]:
    """Exactly what LoadGuard captures — and what it never captures."""
    return {
        "captured": [
            "context-switch counts",
            "meeting count and duration",
            "notification count and source label",
            "focus-block count and duration",
        ],
        "never_captured": [
            "screen content",
            "keystrokes",
            "message bodies",
            "audio / video",
            "physiological data",
            "health or mental-health data",
            "the medical or personal reason for an absence",
        ],
        "local_first": True,
        "statement": (
            "LoadGuard detects behavioral patterns associated with interruption "
            "overload; it does not diagnose stress, burnout, or any medical condition."
        ),
    }


def _plan_from_payload(payload: dict[str, Any]) -> Plan:
    """Rebuild a Plan object from a stored asdict payload."""
    lr = payload["load_report"]
    load_report = LoadReport(
        score=lr["score"],
        level=lr["level"],
        factors=lr.get("factors", {}),
        explanation=lr.get("explanation", ""),
        disclaimer=lr.get("disclaimer", ""),
    )
    items = [
        PlanItem(
            position=i["position"],
            action=i["action"],
            task_id=i.get("task_id"),
            title=i.get("title", ""),
            rationale=i.get("rationale", ""),
            suggested_assignees=i.get("suggested_assignees", []),
            delegate_to=i.get("delegate_to"),
        )
        for i in payload["plan"]["items"]
    ]
    return Plan(
        load_report=load_report,
        items=items,
        note=payload["plan"].get("note", ""),
        generated_by=payload["plan"].get("generated_by", "heuristic"),
        proposed_by=payload["plan"].get("proposed_by", "deterministic"),
        plan_id=payload["plan"].get("plan_id", payload.get("plan_id", "")),
        status=payload["plan"].get("status", "pending"),
    )


_FAVICON_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
    '<text y=".9em" font-size="90">🧠</text></svg>'
)


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(content=_FAVICON_SVG, media_type="image/svg+xml")


_DASHBOARD_HTML: str | None = None


def _get_dashboard_html() -> str | None:
    global _DASHBOARD_HTML
    if _DASHBOARD_HTML is None:
        path = Path(__file__).resolve().parents[2] / "web" / "index.html"
        if path.exists():
            _DASHBOARD_HTML = path.read_text(encoding="utf-8")
    return _DASHBOARD_HTML


@app.get("/", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    content = _get_dashboard_html()
    if content is not None:
        return HTMLResponse(content)
    return HTMLResponse("<h1>dashboard/index.html not found</h1>", status_code=404)
