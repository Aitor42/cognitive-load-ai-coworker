"""Granite Decision Agent and its deterministic gate.

The decision agent makes Granite participate in a *real* decision — which task
to prioritize, which to delegate, and when to insert focus/recovery blocks —
rather than only explaining a plan computed elsewhere.

The proposal is always gated by ``validate_proposal`` before it can affect the
plan:

- only known, todo tasks may be referenced;
- critical tasks (priority >= 4) can never be delegated;
- no invented data (titles, priorities, deadlines) is allowed;
- insert actions are limited and well-formed.

An invalid or missing proposal leaves the deterministic plan untouched, so the
system degrades safely: **Granite proposes, LoadGuard validates, the human
decides.**
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .llm import ChatModel, HeuristicModel
from .models import FeatureSet, LoadReport, Plan, PlanItem, Task, TODO

ALLOWED_INSERT_ACTIONS = {"focus_block", "break"}
# Tasks with priority >= this value are critical and can never be delegated.
CRITICAL_PRIORITY = 4
MAX_INSERTS = 2
MAX_RATIONALE_CHARS = 600


@dataclass
class InsertAction:
    """A focus/break block the LLM wants to insert into the day."""

    action: str
    after_task_id: str | None = None


@dataclass
class DecisionProposal:
    """Structured plan adjustments proposed by Granite."""

    priority_task_id: str | None = None
    delegate_task_ids: list[str] = field(default_factory=list)
    inserts: list[InsertAction] = field(default_factory=list)
    rationale: str = ""
    raw: str = ""


@dataclass
class ProposalValidation:
    """Result of the deterministic gate on a proposal."""

    valid: bool
    reasons: list[str] = field(default_factory=list)


def parse_proposal(raw_text: str) -> DecisionProposal | None:
    """Parse the LLM's JSON proposal; return None if it is not valid JSON."""
    if not raw_text:
        return None
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(raw_text[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None

    priority = data.get("priority_task_id")
    delegates = data.get("delegate_task_ids", []) or []
    raw_inserts = data.get("inserts", []) or []
    if not isinstance(delegates, list):
        delegates = []
    if not isinstance(raw_inserts, list):
        raw_inserts = []

    inserts: list[InsertAction] = []
    for item in raw_inserts:
        if not isinstance(item, dict):
            continue
        action = str(item.get("action", ""))
        after = item.get("after_task_id")
        inserts.append(InsertAction(action=action, after_task_id=after if after else None))

    return DecisionProposal(
        priority_task_id=priority if isinstance(priority, str) and priority else None,
        delegate_task_ids=[str(t) for t in delegates if isinstance(t, str)],
        inserts=inserts,
        rationale=str(data.get("rationale", "")).strip(),
        raw=raw_text,
    )


def validate_proposal(
    proposal: DecisionProposal, tasks: list[Task], load_report: LoadReport
) -> ProposalValidation:
    """Deterministic safety gate: reject anything that is unsafe or invented."""
    reasons: list[str] = []
    known_ids = {t.id for t in tasks}
    todo_ids = {t.id for t in tasks if t.status == TODO}
    priority_map = {t.id: t.priority for t in tasks}

    if proposal.priority_task_id is not None:
        pid = proposal.priority_task_id
        if pid not in known_ids:
            reasons.append(f"priority_task_id {pid!r} does not exist")
        elif pid not in todo_ids:
            reasons.append(f"priority_task_id {pid!r} is not a todo task")
        elif pid in proposal.delegate_task_ids:
            reasons.append("priority_task_id cannot also be delegated")

    if len(set(proposal.delegate_task_ids)) != len(proposal.delegate_task_ids):
        reasons.append("duplicate delegate_task_ids")

    for tid in proposal.delegate_task_ids:
        if tid not in known_ids:
            reasons.append(f"delegate_task_id {tid!r} does not exist")
        elif tid not in todo_ids:
            reasons.append(f"delegate_task_id {tid!r} is not a todo task")
        elif priority_map.get(tid, 0) >= CRITICAL_PRIORITY:
            reasons.append(f"delegate_task_id {tid!r} is critical (priority {priority_map[tid]})")

    if len(proposal.inserts) > MAX_INSERTS:
        reasons.append(f"too many inserts ({len(proposal.inserts)} > {MAX_INSERTS})")
    for ins in proposal.inserts:
        if ins.action not in ALLOWED_INSERT_ACTIONS:
            reasons.append(f"invalid insert action {ins.action!r}")
        if ins.after_task_id is not None and ins.after_task_id not in known_ids:
            reasons.append(f"insert references unknown task {ins.after_task_id!r}")

    if len(proposal.rationale) > MAX_RATIONALE_CHARS:
        reasons.append("rationale too long")

    return ProposalValidation(valid=not reasons, reasons=reasons)


class GraniteDecisionAgent:
    """Ask Granite for a structured plan proposal, gated by the validator."""

    name = "GraniteDecision"

    def __init__(self, model: ChatModel | None = None) -> None:
        self.model = model or HeuristicModel()

    def run(
        self,
        features: FeatureSet,
        load_report: LoadReport,
        tasks: list[Task],
    ) -> DecisionProposal | None:
        """Return a validated proposal, or None (deterministic fallback)."""
        if self.model.name == "heuristic":
            return None
        raw = self.model.propose_plan(features, load_report, tasks)
        proposal = parse_proposal(raw)
        if proposal is None:
            return None
        validation = validate_proposal(proposal, tasks, load_report)
        if not validation.valid:
            import logging

            logging.getLogger(__name__).info(
                "Granite proposal rejected by deterministic gate: %s", validation.reasons
            )
            return None
        return proposal


def merge_proposal(
    tasks: list[Task],
    load_report: LoadReport,
    base_plan: Plan,
    proposal: DecisionProposal | None,
) -> tuple[Plan, bool]:
    """Apply a validated proposal on top of the deterministic plan.

    Returns ``(plan, used)`` where ``used`` is True when the proposal actually
    changed the plan. The deterministic plan is the safety baseline; the
    proposal can only reorder, delegate low-priority work, and insert blocks.
    """
    if proposal is None:
        return base_plan, False

    items = [
        PlanItem(
            position=i.position,
            action=i.action,
            task_id=i.task_id,
            title=i.title,
            rationale=i.rationale,
        )
        for i in base_plan.items
    ]
    used = False

    def _copy() -> Plan:
        return Plan(
            load_report=load_report,
            items=items,
            generated_by=base_plan.generated_by,
            proposed_by=base_plan.proposed_by,
            plan_id=base_plan.plan_id,
            status=base_plan.status,
        )

    # 1) Prioritize a task: move it to the front of the "do" queue.
    if proposal.priority_task_id:
        idx = next(
            (i for i, it in enumerate(items) if it.task_id == proposal.priority_task_id),
            None,
        )
        if idx is not None:
            item = items.pop(idx)
            item.action = "do"
            if proposal.rationale:
                item.rationale = proposal.rationale
            first_do = next((i for i, it in enumerate(items) if it.action == "do"), len(items))
            items.insert(first_do, item)
            used = True

    # 2) Delegate low-priority tasks (only ones the deterministic plan kept).
    for tid in proposal.delegate_task_ids:
        idx = next(
            (i for i, it in enumerate(items) if it.task_id == tid and it.action == "do"),
            None,
        )
        if idx is not None:
            items[idx].action = "delegate"
            if proposal.rationale:
                items[idx].rationale = proposal.rationale
            used = True

    # 3) Insert focus/break blocks (deduplicated, capped).
    focus_exists = any(it.action == "focus_block" for it in items)
    break_count = sum(1 for it in items if it.action == "break")
    for ins in proposal.inserts:
        if ins.action == "focus_block":
            if focus_exists:
                continue
            anchor = (
                next(
                    (i for i, it in enumerate(items) if it.task_id == ins.after_task_id),
                    None,
                )
                if ins.after_task_id
                else None
            )
            first_do = next((i for i, it in enumerate(items) if it.action == "do"), 0)
            insert_at = anchor + 1 if anchor is not None else first_do
            items.insert(
                insert_at,
                PlanItem(
                    position=0,
                    action="focus_block",
                    title="Focus block (no notifications)",
                    rationale=proposal.rationale,
                ),
            )
            focus_exists = True
            used = True
        elif ins.action == "break":
            if break_count >= 3:
                continue
            anchor = (
                next(
                    (i for i, it in enumerate(items) if it.task_id == ins.after_task_id),
                    None,
                )
                if ins.after_task_id
                else None
            )
            insert_at = anchor + 1 if anchor is not None else len(items)
            items.insert(
                insert_at,
                PlanItem(
                    position=0,
                    action="break",
                    title="Recovery break",
                    rationale=proposal.rationale,
                ),
            )
            break_count += 1
            used = True

    for i, item in enumerate(items):
        item.position = i + 1

    plan = _copy()
    return plan, used
