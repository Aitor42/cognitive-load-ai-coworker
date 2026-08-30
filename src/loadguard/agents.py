"""Multi-agent pipeline.

Each agent has a single responsibility and no framework dependency, so the same
implementations can be orchestrated sequentially (as here) or wrapped in a
LangGraph StateGraph for orchestration and tool use without changes.
"""

from __future__ import annotations

from typing import Any

from .llm import ChatModel, HeuristicModel
from .models import Event, FeatureSet, LoadReport, Plan, Task, Worker
from .recommender import build_plan
from .scoring import score
from .signals import compute_features


class SignalAnalystAgent:
    """Ingest raw events and aggregate them into explainable features."""

    name = "SignalAnalyst"

    def run(self, events: list[Event], window_minutes: float | None = None) -> FeatureSet:
        return compute_features(events, window_minutes)


class LoadDiagnosticianAgent:
    """Turn features into a Cognitive Load Score (0-100) and level."""

    name = "LoadDiagnostician"

    def run(
        self,
        features: FeatureSet,
        role: str | None = None,
        weights: dict[str, float] | None = None,
        baseline: Any = None,
    ) -> LoadReport:
        return score(features, weights=weights, role=role, baseline=baseline)


class WorkloadPlannerAgent:
    """Resequence, delegate, and insert recovery/focus blocks."""

    name = "WorkloadPlanner"

    def run(
        self,
        tasks: list[Task],
        load_report: LoadReport,
        workers: list[Worker] | None = None,
        now: float | None = None,
        audit_history: list[dict] | None = None,
        hour_of_day: float | None = None,
        tz_name: str | None = None,
    ) -> Plan:
        return build_plan(
            tasks,
            load_report,
            workers=workers,
            now=now,
            audit_history=audit_history,
            hour_of_day=hour_of_day,
            tz_name=tz_name,
        )


class NarratorAgent:
    """Write the human-readable explanation of the plan (LLM)."""

    name = "Narrator"

    def __init__(self, model: ChatModel | None = None) -> None:
        self.model = model or HeuristicModel()

    def run(self, load_report: LoadReport, plan: Plan, tasks: list[Task]) -> str:
        return self.model.generate_note(load_report, plan, tasks)
