"""Granite Guardian safety layer with a deterministic fallback guard.

LoadGuard's guardrail runs on *every* plan, with or without a model:

1. **Deterministic checks** (always on): the structured plan is well-formed,
   references only known tasks, never delegates critical work, never invents
   data, and the narrative contains no medical diagnosis, no sensitive personal
   data, no demeaning language, and stays in scope.
2. **Granite Guardian check** (when a model is configured): the narrative is
   also validated by a Granite Guardian-style prompt. If it flags an issue, the
   narrative is replaced with the deterministic one.

The guard never blocks the human from deciding — it only guarantees that what
reaches the human is safe, grounded, and in scope.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .llm import ChatModel, HeuristicModel
from .models import Plan, Task

ALLOWED_ACTIONS = {"do", "delegate", "focus_block", "break", "batch"}

# Priority (1..5) above which a task is considered critical and may never be
# delegated by the LLM decision agent (deterministic plans already use 1-2).
CRITICAL_PRIORITY = 4

# Phrases that would constitute a medical / burnout diagnosis. LoadGuard
# explicitly does NOT diagnose; it only reports behavioral patterns.
MEDICAL_PHRASES = [
    "you are burning out",
    "you have burnout",
    "burnout diagnosis",
    "you are suffering from burnout",
    "you are depressed",
    "you have depression",
    "you have anxiety",
    "anxiety disorder",
    "mental health condition",
]

# Diagnosis verbs as whole words, so technical terms like "diagnostics" (e.g. a
# software task title) are NOT flagged as medical diagnoses.
MEDICAL_TERMS_RE = re.compile(r"\bdiagnos(?:e|is|ed|ing)\b", re.IGNORECASE)

# Negation words that mark a "diagnos*" mention as the system's own disclaimer
# (e.g. "LoadGuard does not diagnose medical conditions") rather than a diagnosis.
NEGATION_RE = re.compile(
    r"\b(?:not|no|never|without|doesn't|don't|cannot|can't|won't|shouldn't)\b",
    re.IGNORECASE,
)

# Demeaning / disrespectful language.
RESPECT_VIOLATIONS = [
    "idiot",
    "stupid",
    "lazy",
    "incompetent",
    "useless",
    "worthless",
]

# Sensitive personal data patterns that must not appear in the narrative.
SENSITIVE_PATTERNS = [
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),  # email addresses
    re.compile(r"\+?\d[\d ()-]{7,}\d"),  # phone-like numbers
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN-like
]

# Out-of-scope advice (financial / legal / medical recommendations).
OUT_OF_SCOPE_PHRASES = [
    "you should invest",
    "buy this stock",
    "legal advice",
    "medical advice",
    "prescribe",
    "guarantee you will",
]


@dataclass
class GuardCheck:
    """The result of a single guardrail check."""

    name: str
    passed: bool
    detail: str = ""


@dataclass
class GuardianResult:
    """Aggregate result of the safety gate."""

    passed: bool
    checks: list[GuardCheck] = field(default_factory=list)
    engine: str = "deterministic"  # "deterministic" | "granite-guardian"
    sanitized: bool = False

    def summary(self) -> str:
        failed = [c.name for c in self.checks if not c.passed]
        if failed:
            return f"guard blocked: {', '.join(failed)}"
        return f"all {len(self.checks)} checks passed ({self.engine})"


def _scan(text: str, patterns: list[str], name: str) -> GuardCheck:
    lowered = text.lower()
    hits = [p for p in patterns if p in lowered]
    return GuardCheck(
        name=name,
        passed=not hits,
        detail=f"matched: {', '.join(hits)}" if hits else "no matches",
    )


def _scan_regex(text: str, patterns: list[re.Pattern], name: str) -> GuardCheck:
    hits = [p.pattern for p in patterns if p.search(text)]
    return GuardCheck(
        name=name,
        passed=not hits,
        detail=f"matched: {', '.join(hits)}" if hits else "no matches",
    )


def _scan_medical(text: str) -> GuardCheck:
    """Check for medical/burnout *diagnosis*, not benign technical wording.

    A "diagnos*" term is ignored when it appears in a clearly negated context
    (e.g. "LoadGuard does not diagnose medical conditions"), which is the
    system's own disclaimer rather than a diagnosis of the user.
    """
    lowered = text.lower()
    hits = {p for p in MEDICAL_PHRASES if p in lowered}
    for match in MEDICAL_TERMS_RE.finditer(text):
        before = text[max(0, match.start() - 40) : match.start()]
        if not NEGATION_RE.search(before):
            hits.add("diagnos*")
            break
    return GuardCheck(
        name="no_medical_diagnosis",
        passed=not hits,
        detail=f"matched: {', '.join(sorted(hits))}" if hits else "no matches",
    )


def _structure_checks(plan: Plan, tasks: list[Task]) -> list[GuardCheck]:
    checks: list[GuardCheck] = []
    known_ids = {t.id for t in tasks}
    critical = {t.id for t in tasks if t.priority >= CRITICAL_PRIORITY}

    # Well-formed: sequential positions and allowed actions.
    actions = [i.action for i in plan.items]
    bad_actions = [a for a in actions if a not in ALLOWED_ACTIONS]
    positions_ok = all(i.position == idx + 1 for idx, i in enumerate(plan.items))
    checks.append(
        GuardCheck(
            "well_formed",
            passed=not bad_actions and positions_ok,
            detail=f"bad actions: {bad_actions}" if bad_actions else "positions sequential",
        )
    )

    # Every referenced task exists (no invented tasks).
    referenced = [i.task_id for i in plan.items if i.task_id is not None]
    unknown = sorted({t for t in referenced if t not in known_ids})
    checks.append(
        GuardCheck(
            "known_tasks",
            passed=not unknown,
            detail=f"unknown tasks: {unknown}" if unknown else "all tasks known",
        )
    )

    # No critical task may be delegated.
    delegated = [i.task_id for i in plan.items if i.action == "delegate" and i.task_id]
    critical_delegated = [t for t in delegated if t in critical]
    checks.append(
        GuardCheck(
            "critical_tasks_safe",
            passed=not critical_delegated,
            detail=f"critical delegated: {critical_delegated}"
            if critical_delegated
            else "no critical delegation",
        )
    )

    # Titles for task-bound items match the source task (no invented data).
    title_map = {t.id: t.title for t in tasks}
    mismatched = [
        i.task_id for i in plan.items if i.task_id in title_map and i.title != title_map[i.task_id]
    ]
    checks.append(
        GuardCheck(
            "no_invented_data",
            passed=not mismatched,
            detail=f"title mismatches: {mismatched}" if mismatched else "titles match source tasks",
        )
    )
    return checks


def _narrative_checks(note: str) -> list[GuardCheck]:
    checks = [
        _scan_medical(note),
        _scan(note, RESPECT_VIOLATIONS, "respectful_language"),
        _scan_regex(note, SENSITIVE_PATTERNS, "no_sensitive_data"),
        _scan(note, OUT_OF_SCOPE_PHRASES, "in_scope"),
    ]
    return checks


def run_deterministic_checks(plan: Plan, tasks: list[Task], note: str) -> list[GuardCheck]:
    """Run the always-on structural + narrative checks."""
    return _structure_checks(plan, tasks) + _narrative_checks(note)


def run_llm_guard(model: ChatModel | None, note: str) -> GuardCheck | None:
    """Run the Granite Guardian check when a model is available."""
    if model is None or model.name == "heuristic":
        return None
    result = model.guard_text(note)
    if result is None:
        return None
    issues = result.get("issues", [])
    return GuardCheck(
        name="granite_guardian",
        passed=bool(result.get("safe", False)),
        detail="; ".join(issues) if issues else "model reports safe",
    )


def validate_plan(
    plan: Plan, tasks: list[Task], note: str, model: ChatModel | None = None
) -> GuardianResult:
    """Validate a plan + narrative, combining deterministic and LLM checks."""
    checks = run_deterministic_checks(plan, tasks, note)
    engine = "deterministic"
    llm_check = run_llm_guard(model, note)
    if llm_check is not None:
        checks.append(llm_check)
        engine = "granite-guardian"
    return GuardianResult(
        passed=all(c.passed for c in checks),
        checks=checks,
        engine=engine,
    )


def guard_plan(
    plan: Plan, tasks: list[Task], model: ChatModel | None = None
) -> tuple[Plan, GuardianResult]:
    """Guard a plan: sanitize the narrative if needed, then validate.

    If only the narrative fails the checks, it is replaced with the deterministic
    narrative and re-validated. Structural failures (which should not happen for
    plans produced by LoadGuard) are reported as-is so the caller can fall back.
    """
    result = validate_plan(plan, tasks, plan.note, model)
    if result.passed:
        return plan, result

    narrative_failed = any(
        c.name
        in (
            "no_medical_diagnosis",
            "respectful_language",
            "no_sensitive_data",
            "in_scope",
            "granite_guardian",
        )
        and not c.passed
        for c in result.checks
    )
    if not narrative_failed:
        return plan, result

    sanitized = Plan(
        load_report=plan.load_report,
        items=list(plan.items),
        note=HeuristicModel().generate_note(plan.load_report, plan, tasks),
        generated_by=plan.generated_by,
        proposed_by=plan.proposed_by,
        plan_id=plan.plan_id,
        status=plan.status,
    )
    result2 = validate_plan(sanitized, tasks, sanitized.note, model)
    result2.sanitized = True
    return sanitized, result2
