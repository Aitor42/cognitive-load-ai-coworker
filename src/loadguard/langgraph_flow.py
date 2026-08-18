"""Optional real LangGraph orchestration.

The pipeline can be compiled into a LangGraph ``StateGraph`` when ``langgraph``
is installed (``pip install langgraph``):

    collect_signals -> compute_features -> diagnose_load -> granite_plan
        -> guardian_validation -> human_approval -> apply_plan -> measure_outcome

``human_approval`` is an explicit approval gate: without an approval decision
the flow stops in the ``awaiting_approval`` state. Re-invoking it with the
human's decision recomputes the flow deterministically from the (re-supplied)
inputs rather than resuming a persisted LangGraph checkpoint.

When ``langgraph`` is not installed, the *same* node functions run sequentially,
so the core pipeline keeps its zero-dependency guarantee.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, TypedDict

from .actions import export_ics, export_tasks_csv, new_plan_id
from .agents import (
    LoadDiagnosticianAgent,
    NarratorAgent,
    SignalAnalystAgent,
    WorkloadPlannerAgent,
)
from .decision import GraniteDecisionAgent, merge_proposal
from .guardian import guard_plan
from .impact import estimate_impact
from .llm import ChatModel
from .models import Event, Task
from .scoring import score

APPROVAL_DECISIONS = ("accepted", "rejected", "edited")


class WorkflowState(TypedDict, total=False):
    """Shared state passed between graph nodes."""

    events: list[Event]
    tasks: list[Task]
    model: ChatModel | None
    guardian_model: ChatModel | None
    window_minutes: float | None
    history: list[float] | None
    plan_id: str | None
    approval: dict[str, Any] | None
    outcome_events: list[Event] | None
    features: Any
    load_report: Any
    base_plan: Any
    proposal: Any
    plan: Any
    guardian: Any
    impact: Any
    observed: dict[str, Any] | None
    exports: dict[str, str]
    status: str


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def collect_signals(state: WorkflowState) -> dict[str, Any]:
    """Entry point: events and tasks are already parsed by the caller."""
    return {"status": "signals_collected"}


def compute_features(state: WorkflowState) -> dict[str, Any]:
    features = SignalAnalystAgent().run(state["events"], state.get("window_minutes"))
    return {"features": features}


def diagnose_load(state: WorkflowState) -> dict[str, Any]:
    load_report = LoadDiagnosticianAgent().run(state["features"])
    return {"load_report": load_report}


def granite_plan(state: WorkflowState) -> dict[str, Any]:
    """Deterministic baseline plan + Granite proposal (gated) + merge."""
    load_report = state["load_report"]
    tasks = state["tasks"]
    base_plan = WorkloadPlannerAgent().run(tasks, load_report)
    base_plan.plan_id = state.get("plan_id") or new_plan_id()
    model = state.get("model")
    proposal = GraniteDecisionAgent(model).run(state["features"], load_report, tasks)
    plan, used = merge_proposal(tasks, load_report, base_plan, proposal)
    if used and model is not None:
        plan.proposed_by = model.name
    return {"base_plan": base_plan, "proposal": proposal, "plan": plan}


def guardian_validation(state: WorkflowState) -> dict[str, Any]:
    """Narrative + Granite Guardian / deterministic safety gate."""
    plan = state["plan"]
    tasks = state["tasks"]
    model = state.get("model")
    narrator = NarratorAgent(model)
    plan.note = narrator.run(state["load_report"], plan, tasks)
    plan.generated_by = narrator.model.name
    plan, guardian = guard_plan(plan, tasks, state.get("guardian_model") or model)
    return {"plan": plan, "guardian": guardian}


def human_approval(state: WorkflowState) -> dict[str, Any]:
    """Explicit human approval gate; the flow stops here without a decision."""
    approval = state.get("approval")
    if not approval or not approval.get("decision"):
        return {"status": "awaiting_approval"}
    decision = approval["decision"] if approval["decision"] in APPROVAL_DECISIONS else "rejected"
    state["plan"].status = decision
    return {"plan": state["plan"], "status": decision}


def apply_plan(state: WorkflowState) -> dict[str, Any]:
    """Act: export the protected calendar blocks and the resequenced tasks."""
    if state.get("status") != "accepted":
        return {"exports": {}}
    plan = state["plan"]
    tasks = state["tasks"]
    return {
        "exports": {
            "ics": export_ics(plan, tasks),
            "tasks_csv": export_tasks_csv(plan, tasks),
        },
        "status": "applied",
    }


def measure_outcome(state: WorkflowState) -> dict[str, Any]:
    """Projected impact, plus observed metrics when outcome events exist."""
    features = state["features"]
    plan = state["plan"]
    impact = estimate_impact(features, plan)
    observed: dict[str, Any] | None = None
    outcome = state.get("outcome_events")
    if outcome:
        obs_features = SignalAnalystAgent().run(outcome, state.get("window_minutes"))
        obs = score(obs_features)
        observed = {
            "score": obs.score,
            "level": obs.level,
            "features": asdict(obs_features),
        }
    return {"impact": impact, "observed": observed, "status": "measured"}


# ---------------------------------------------------------------------------
# Graph wiring
# ---------------------------------------------------------------------------


def _approval_router(state: WorkflowState) -> str:
    return state.get("status", "awaiting_approval")


def build_graph():
    """Compile the LangGraph StateGraph (requires ``langgraph`` installed)."""
    from langgraph.graph import END, StateGraph

    graph = StateGraph(WorkflowState)
    graph.add_node("collect_signals", collect_signals)
    graph.add_node("compute_features", compute_features)
    graph.add_node("diagnose_load", diagnose_load)
    graph.add_node("granite_plan", granite_plan)
    graph.add_node("guardian_validation", guardian_validation)
    graph.add_node("human_approval", human_approval)
    graph.add_node("apply_plan", apply_plan)
    graph.add_node("measure_outcome", measure_outcome)

    graph.set_entry_point("collect_signals")
    graph.add_edge("collect_signals", "compute_features")
    graph.add_edge("compute_features", "diagnose_load")
    graph.add_edge("diagnose_load", "granite_plan")
    graph.add_edge("granite_plan", "guardian_validation")
    graph.add_edge("guardian_validation", "human_approval")
    graph.add_conditional_edges(
        "human_approval",
        _approval_router,
        {
            "awaiting_approval": END,
            "rejected": END,
            "accepted": "apply_plan",
            "edited": "apply_plan",
        },
    )
    graph.add_edge("apply_plan", "measure_outcome")
    graph.add_edge("measure_outcome", END)
    return graph.compile()


def run_sequential_graph(state: WorkflowState) -> dict[str, Any]:
    """Run the same nodes in sequence when langgraph is not installed."""
    merged: dict[str, Any] = dict(state)
    merged.update(collect_signals(merged))
    merged.update(compute_features(merged))
    merged.update(diagnose_load(merged))
    merged.update(granite_plan(merged))
    merged.update(guardian_validation(merged))
    merged.update(human_approval(merged))
    if merged.get("status") in ("accepted", "edited"):
        merged.update(apply_plan(merged))
        merged.update(measure_outcome(merged))
    return merged


_langgraph_available: bool | None = None


def langgraph_installed() -> bool:
    global _langgraph_available
    if _langgraph_available is None:
        try:
            import langgraph  # noqa: F401

            _langgraph_available = True
        except ImportError:
            _langgraph_available = False
    return _langgraph_available


def run_graph(
    events: list[Event],
    tasks: list[Task],
    model: ChatModel | None = None,
    window_minutes: float | None = None,
    history: list[float] | None = None,
    approval: dict[str, Any] | None = None,
    plan_id: str | None = None,
    outcome_events: list[Event] | None = None,
    guardian_model: ChatModel | None = None,
    sequential: bool | None = None,
) -> dict[str, Any]:
    """Run the graph, using LangGraph when available (or force sequential)."""
    state: WorkflowState = {
        "events": events,
        "tasks": tasks,
        "model": model,
        "guardian_model": guardian_model,
        "window_minutes": window_minutes,
        "history": history,
        "approval": approval,
        "plan_id": plan_id,
        "outcome_events": outcome_events,
    }
    if sequential or not langgraph_installed():
        return run_sequential_graph(state)
    return build_graph().invoke(state)
