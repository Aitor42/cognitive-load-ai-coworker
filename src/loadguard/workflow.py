"""End-to-end orchestration of the multi-agent pipeline.

The full loop: **Granite proposes → LoadGuard validates → the human approves →
LoadGuard acts → the result is measured.**

1. ``SignalAnalystAgent`` — aggregate raw events into features.
2. ``LoadDiagnosticianAgent`` — explainable 0-100 score + level + drivers.
3. ``WorkloadPlannerAgent`` — deterministic safety-baseline plan.
4. ``GraniteDecisionAgent`` — Granite proposes structured adjustments, gated by
   ``validate_proposal`` (deterministic).
5. ``guard_plan`` — Granite Guardian / deterministic guard validates the result.
6. Human approval — the plan is returned with ``status="pending"`` unless the
   caller already supplies an approval decision.
7. ``estimate_impact`` — projected before/after score (observed measurement is
   handled by ``benchmark.run_pilot_evaluation``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .actions import new_plan_id
from .availability import find_reassignment_alerts
from .agents import (
    LoadDiagnosticianAgent,
    NarratorAgent,
    SignalAnalystAgent,
    WorkloadPlannerAgent,
)
from .baseline import PersonalBaseline, TrendInfo, compute_baseline, trend
from .decision import DecisionProposal, GraniteDecisionAgent, merge_proposal
from .guardian import GuardianResult, guard_plan
from .impact import ImpactResult, estimate_impact
from .llm import ChatModel
from .models import Event, FeatureSet, LoadReport, Plan, ReassignmentAlert, Task, Worker

PENDING = "pending"


@dataclass
class WorkflowResult:
    features: FeatureSet
    load_report: LoadReport
    plan: Plan
    impact: ImpactResult
    proposal: DecisionProposal | None = None
    guardian: GuardianResult | None = None
    baseline: PersonalBaseline | None = None
    trend: TrendInfo | None = None
    approval: str = PENDING
    reassignment_alerts: list[ReassignmentAlert] = field(default_factory=list)


def run_workflow(
    events: list[Event],
    tasks: list[Task],
    model: ChatModel | None = None,
    window_minutes: float | None = None,
    history: list[float] | None = None,
    approval: str | None = None,
    plan_id: str | None = None,
    guardian_model: ChatModel | None = None,
    workers: list[Worker] | None = None,
    now: float | None = None,
    audit_history: list[dict] | None = None,
    role: str | None = None,
    weights: dict[str, float] | None = None,
    hour_of_day: float | None = None,
) -> WorkflowResult:
    """Run the full sense -> diagnose -> plan -> validate -> approve -> impact pipeline."""
    features = SignalAnalystAgent().run(events, window_minutes)
    load_report = LoadDiagnosticianAgent().run(features, role=role, weights=weights)

    # 1) Deterministic safety-baseline plan (respects past audit decisions & time of day).
    base_plan = WorkloadPlannerAgent().run(
        tasks,
        load_report,
        workers=workers,
        now=now,
        audit_history=audit_history,
        hour_of_day=hour_of_day,
    )
    base_plan.plan_id = plan_id or new_plan_id()

    # 2) Granite proposes structured adjustments; the gate decides if they apply.
    proposal = GraniteDecisionAgent(model).run(features, load_report, tasks)
    plan, used_proposal = merge_proposal(tasks, load_report, base_plan, proposal)
    if used_proposal:
        plan.proposed_by = model.name if model is not None else "deterministic"

    # 3) Narrative (LLM) + safety guard (Granite Guardian / deterministic).
    narrator = NarratorAgent(model)
    plan.note = narrator.run(load_report, plan, tasks)
    plan.generated_by = narrator.model.name
    plan, guardian = guard_plan(plan, tasks, guardian_model or model)

    # 4) Human approval state.
    plan.status = approval if approval in ("accepted", "rejected", "edited") else PENDING

    # 5) Impact estimation with history-calibrated effectiveness.
    impact = estimate_impact(features, plan, history=history)

    personal = compute_baseline(history or [])
    alerts = find_reassignment_alerts(tasks, workers or [], now)
    return WorkflowResult(
        features=features,
        load_report=load_report,
        plan=plan,
        impact=impact,
        proposal=proposal,
        guardian=guardian,
        baseline=personal,
        trend=trend(load_report.score, personal),
        approval=plan.status,
        reassignment_alerts=alerts,
    )
