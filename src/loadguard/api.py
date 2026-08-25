"""FastAPI layer (optional).

Exposes the pipeline over HTTP and serves the interactive dashboard. The core
logic remains importable without FastAPI installed.

Endpoints close the loop: analyze -> approve/reject -> export (.ics, tasks) ->
feedback, plus a local history store for the personalized baseline and a
privacy endpoint describing exactly what is captured.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse, Response
from pydantic import BaseModel, Field

from .actions import (
    APPROVAL_DECISIONS,
    clear_audit,
    export_ics,
    export_tasks_csv,
    load_audit,
    new_plan_id,
    record_approval,
)
from .baseline import append_score, clear_history, load_history
from .config import get_guardian_model, get_model
from .guardian import validate_plan
from .models import Absence, LoadReport, Plan, PlanItem, Task, Worker
from .projection import run_midday_review
from .sample_data import sample_tasks, sample_workers
from .signals import load_events, parse_event
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

# In-memory plan store: plan_id -> {"payload": ..., "tasks": [...], "events": [...]}
PLANS: dict[str, dict[str, Any]] = {}


class AnalyzeRequest(BaseModel):
    events: list[dict[str, Any]]
    tasks: list[dict[str, Any]]
    window_minutes: Optional[float] = None
    history: Optional[list[float]] = None
    approval: Optional[str] = None
    workers: Optional[list[dict[str, Any]]] = None


class MiddayRequest(BaseModel):
    events: list[dict[str, Any]]
    tasks: list[dict[str, Any]]
    workers: Optional[list[dict[str, Any]]] = None
    elapsed_minutes: float = 240.0
    total_minutes: float = 480.0


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
            deadline=float(t["deadline"]) if t.get("deadline") is not None else None,
            status=str(t.get("status", "todo")),
            assignee=str(t["assignee"]) if t.get("assignee") else None,
        )
        for i, t in enumerate(payload)
    ]


def _to_workers(payload: list[dict[str, Any]]) -> list[Worker]:
    """Build Worker objects (with nested absences) from API payload dicts."""
    return [
        Worker(
            id=str(w.get("id", i)),
            name=str(w.get("name", "")),
            absences=[
                Absence(
                    start=float(a["start"]),
                    end=float(a["end"]),
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
) -> dict[str, Any]:
    payload = asdict(result)
    plan_id = result.plan.plan_id
    PLANS[plan_id] = {
        "payload": payload,
        "tasks": [asdict(t) for t in tasks],
        "events": events,
        "workers": [asdict(w) for w in workers] if workers else [],
    }
    payload["plan_id"] = plan_id
    return payload


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/sample")
def sample() -> dict[str, Any]:
    path = Path(__file__).resolve().parents[2] / "demo" / "sample_events.jsonl"
    events = load_events(path)
    return {
        "events": [asdict(e) for e in events],
        "tasks": [asdict(t) for t in sample_tasks()],
        "workers": [asdict(w) for w in sample_workers()],
    }


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
    )
    return _store_plan(result, tasks, [asdict(e) for e in events], workers)


@app.post("/midday")
def midday(req: MiddayRequest) -> dict[str, Any]:
    events = [parse_event(e) for e in req.events]
    tasks = _to_tasks(req.tasks)
    workers = _to_workers(req.workers or [])
    review = run_midday_review(
        events,
        tasks,
        req.elapsed_minutes,
        req.total_minutes,
        workers=workers,
    )
    result = asdict(review)
    if review.plan is not None:
        # Store the afternoon plan so it can be approved and exported through
        # the same endpoints as the morning plan (accept -> export .ics/.csv).
        review.plan.plan_id = review.plan.plan_id or new_plan_id()
        plan_id = review.plan.plan_id
        PLANS[plan_id] = {
            "payload": {
                "load_report": asdict(review.plan.load_report),
                "plan": asdict(review.plan),
            },
            "tasks": [asdict(t) for t in tasks],
            "events": [asdict(e) for e in events],
            "workers": [asdict(w) for w in workers],
        }
        result["plan_id"] = plan_id
    return result


@app.post("/approve")
def approve(req: ApproveRequest) -> dict[str, Any]:
    stored = PLANS.get(req.plan_id)
    if stored is None:
        raise HTTPException(status_code=404, detail="unknown plan_id")
    if req.items is not None:
        # An edited plan replaces the stored items, but it must pass the same
        # safety gate as an original plan (no invented tasks, no critical
        # delegation, valid actions).
        cleaned = [
            {
                "position": i + 1,
                "action": item["action"],
                "task_id": item.get("task_id"),
                "title": item.get("title", ""),
                "rationale": item.get("rationale", ""),
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
            )
            for c in cleaned
        ]
        guard = validate_plan(plan, _to_tasks(stored["tasks"]), plan.note)
        if not guard.passed:
            raise HTTPException(
                status_code=400,
                detail=f"edited plan failed the safety gate: {guard.summary()}",
            )
        stored["payload"]["plan"]["items"] = cleaned
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
    }


@app.post("/feedback")
def feedback(req: FeedbackRequest) -> dict[str, Any]:
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
def export_plan_ics(plan_id: str) -> Response:
    stored = PLANS.get(plan_id)
    if stored is None:
        raise HTTPException(status_code=404, detail="unknown plan_id")
    plan = _plan_from_payload(stored["payload"])
    tasks = _to_tasks(stored["tasks"])
    return Response(
        content=export_ics(plan, tasks),
        media_type="text/calendar",
        headers={"Content-Disposition": f'attachment; filename="loadguard-{plan_id}.ics"'},
    )


@app.get("/plan/{plan_id}/export.csv")
def export_plan_csv(plan_id: str) -> PlainTextResponse:
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


@app.get("/", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    path = Path(__file__).resolve().parents[2] / "web" / "index.html"
    if path.exists():
        return HTMLResponse(path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>dashboard/index.html not found</h1>", status_code=404)
