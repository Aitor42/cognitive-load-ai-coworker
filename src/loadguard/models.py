"""Data models for LoadGuard.

All models are plain dataclasses so the core pipeline has zero third-party
dependencies. Timestamps are stored as epoch seconds (floats) internally;
adapters are responsible for converting ISO-8601 strings on input.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

# Event kinds ingested by the signal pipeline.
CONTEXT_SWITCH = "context_switch"
MEETING = "meeting"
NOTIFICATION = "notification"
FOCUS_BLOCK = "focus_block"

EVENT_KINDS = (CONTEXT_SWITCH, MEETING, NOTIFICATION, FOCUS_BLOCK)

# Cognitive load levels.
LOW = "low"
MODERATE = "moderate"
HIGH = "high"
OVERLOAD = "overload"

# Task statuses.
TODO = "todo"
DONE = "done"
DELEGATED = "delegated"

# Absence kinds. LoadGuard stores only the *fact* and type of an absence
# (vacation vs. leave) — never a medical or personal reason.
VACATION = "vacation"
LEAVE = "leave"
ABSENCE_KINDS = (VACATION, LEAVE)


@dataclass
class Event:
    """A single work signal.

    Only lightweight, privacy-respecting metadata is stored: counts, durations,
    and opaque labels. Raw content (screen, keystrokes, message bodies) is never
    captured.
    """

    timestamp: float  # epoch seconds
    kind: str  # one of EVENT_KINDS
    duration_minutes: float = 0.0  # used by MEETING and FOCUS_BLOCK
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in EVENT_KINDS:
            raise ValueError(f"unknown event kind: {self.kind!r}")
        if self.duration_minutes < 0:
            self.duration_minutes = 0.0


@dataclass
class FeatureSet:
    """Aggregated, windowed features derived from raw events.

    These are the *proxies* used to estimate cognitive load. They are
    deliberately behavioral and explainable; LoadGuard makes no physiological
    claims.
    """

    context_switches_per_hour: float = 0.0
    meeting_ratio: float = 0.0  # minutes in meetings / window minutes (0..1)
    notification_rate: float = 0.0  # notifications per hour
    focus_ratio: float = 0.0  # minutes in focus blocks / window minutes (0..1)
    multitasking_index: float = 0.0  # 0..1, share of time overlapping activity
    ai_notification_rate: float = 0.0  # notifications per hour attributed to AI tools


@dataclass
class Task:
    """A unit of work the co-worker can resequence or delegate."""

    id: str
    title: str
    priority: int  # 1 (low) .. 5 (high)
    duration_minutes: float = 30.0
    focus_required: bool = True
    deadline: Optional[float] = None  # epoch seconds, optional
    status: str = TODO
    assignee: Optional[str] = None  # worker id the task is assigned to, optional


@dataclass
class Absence:
    """A period during which a worker is unavailable (vacation or leave).

    Only the fact and the type of the absence are stored; the reason (medical
    or otherwise) is never captured, consistent with LoadGuard's privacy-first
    design.
    """

    start: float  # epoch seconds
    end: float  # epoch seconds
    kind: str = LEAVE  # one of ABSENCE_KINDS
    note: str = ""

    def __post_init__(self) -> None:
        if self.kind not in ABSENCE_KINDS:
            raise ValueError(f"unknown absence kind: {self.kind!r}")
        if self.end < self.start:
            raise ValueError("absence end precedes start")


@dataclass
class Worker:
    """A member of the team whose attention LoadGuard protects."""

    id: str
    name: str = ""
    absences: list[Absence] = field(default_factory=list)


@dataclass
class ReassignmentAlert:
    """A deadline-driven suggestion to reassign a task away from an absent worker."""

    task_id: str
    title: str
    assignee: str
    deadline: float
    reason: str
    suggested_assignees: list[str] = field(default_factory=list)


@dataclass
class LoadReport:
    """The explainable result of the scoring engine."""

    score: float  # 0..100
    level: str  # low | moderate | high | overload
    factors: dict[str, float] = field(default_factory=dict)
    explanation: str = ""
    ai_interruption_ratio: Optional[float] = None  # share of notifications from AI tools
    disclaimer: str = (
        "Cognitive Load Score is a behavioral proxy derived from counts and "
        "ratios of work signals, not a physiological measurement."
    )


@dataclass
class PlanItem:
    """One scheduled action in the generated plan."""

    position: int
    action: str  # do | delegate | focus_block | break | batch
    task_id: Optional[str] = None
    title: str = ""
    rationale: str = ""


@dataclass
class Plan:
    """A resequenced day: load report plus ordered actions.

    - ``generated_by`` is the model that wrote the natural-language note.
    - ``proposed_by`` is the model that proposed the structured plan
      ("deterministic" when the Granite Decision Agent was not used).
    - ``status`` tracks the human-approval state of the plan.
    """

    load_report: LoadReport
    items: list[PlanItem] = field(default_factory=list)
    note: str = ""
    generated_by: str = "heuristic"
    proposed_by: str = "deterministic"
    plan_id: str = ""
    status: str = "pending"  # pending | accepted | rejected | edited
