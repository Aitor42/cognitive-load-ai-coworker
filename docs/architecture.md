# LoadGuard Architecture

## Overview

LoadGuard is a **cognitive-load-aware AI co-worker**. It treats the human's attention as a
first-class, finite resource and proactively reshapes the workday to protect it.

The system follows a strict, reviewable loop:

```text
signals (raw events) -> features (explainable proxies) -> load score (0-100)
        -> deterministic baseline plan -> Granite decision proposal (gated)
        -> guardian validation -> human approval -> apply (.ics / tasks)
        -> measured outcome (or honest projection)
```

The deterministic layer is the source of truth for *safety*; Granite proposes *what to change*;
the human decides *whether it happens*. This hybrid design keeps the prototype grounded,
reproducible, and safe, while still making the AI central.

See [`architecture.mmd`](architecture.mmd) for the full Mermaid diagram. The same diagram is embedded in the root [`README.md`](../README.md); `tests/test_docs.py` keeps both copies synchronized.

## Design principles

1. **Privacy first.** Only counts, durations, and opaque labels are captured — never screen
   content, keystrokes, or message bodies. Features are computed locally; only derived aggregates
   and task titles are sent to the LLM (the minimum needed to plan and explain).
2. **Explainability.** Every score is a weighted combination of named proxies with documented
   thresholds, so any recommendation can be traced back to a signal.
3. **Honesty about limits.** The "Cognitive Load Score" is explicitly a *behavioral proxy*, not a
   physiological measurement (see [`references.md`](references.md)). Projected impact is labelled
   as *projected*; observed impact is only reported when outcome signals exist.
4. **Granite proposes, LoadGuard validates, the human decides.** The LLM never directly mutates a
   critical task or invents data — every proposal passes a deterministic gate (`decision.py`), a
   safety guard (`guardian.py`), and an explicit human approval step.
5. **Degradability.** Without API keys or optional dependencies, the full pipeline still runs via
   the heuristic engine (deterministic plan + deterministic guard).
6. **Human in control.** Every output is a suggestion the user can accept, edit, or reject.

## The full loop

| Stage | Module | Deterministic / LLM |
| --- | --- | --- |
| Sense | `signals.SignalAnalystAgent` | Deterministic |
| Diagnose | `scoring.LoadDiagnosticianAgent` | Deterministic |
| Baseline plan | `recommender.WorkloadPlannerAgent` | Deterministic |
| **Decide** | `decision.GraniteDecisionAgent` → `validate_proposal` | LLM proposes, deterministic gate validates |
| **Validate** | `guardian.guard_plan` (Granite Guardian + deterministic) | Deterministic + LLM guard |
| **Approve** | `workflow.run_workflow(approval=...)` / `api` / dashboard | Human |
| **Act** | `actions.export_ics` / `export_tasks_csv` | Deterministic |
| **Measure** | `impact.estimate_impact` + `benchmark.run_pilot_evaluation` | Deterministic |

## Modules

| Module | Responsibility |
| --- | --- |
| `models.py` | Dataclasses: `Event`, `FeatureSet`, `Task`, `Absence`, `Worker`, `ReassignmentAlert`, `LoadReport`, `Plan`. |
| `signals.py` | Parse JSONL/API events; aggregate into per-hour/ratio features. |
| `scoring.py` | Weighted normalization of features into a 0–100 score + level. |
| `recommender.py` | Deterministic planner: resequence, delegate, insert breaks/focus. |
| `decision.py` | Granite Decision Agent + deterministic proposal validator + merge. |
| `guardian.py` | Granite Guardian safety check + always-on deterministic guard. |
| `baseline.py` | Personal baseline, deviation, trend, confidence, JSONL history. |
| `actions.py` | Human approval record, `.ics` export, task CSV export, audit trail. |
| `agents.py` | Thin agent wrappers (single responsibility per stage). |
| `workflow.py` | End-to-end orchestration of the full loop. |
| `langgraph_flow.py` | Real LangGraph `StateGraph` (with sequential fallback). |
| `impact.py` | Projects the load score after following the plan. |
| `availability.py` | Worker absences + deadline-driven reassignment alerts. |
| `projection.py` | End-of-day projection + midday re-organization. |
| `scheduler.py` | Morning + midday daily-cycle orchestration. |
| `benchmark.py` | Objective metrics + three-phase pilot evaluation. |
| `llm.py` | `ChatModel` interface; heuristic + watsonx + ollama (Granite). |
| `config.py` | Select the model / guardian provider from environment variables. |
| `api.py` / `app.py` | FastAPI HTTP layer and dashboard; available when the `api` dependency group is installed. |

## Signal proxies

| Feature | What it measures | Normalization (0..1) |
| --- | --- | --- |
| `context_switches_per_hour` | Interruption frequency | Hill sigmoid, midpoint 6/h (12/h → 0.67) |
| `meeting_ratio` | Share of window in meetings | Linear clamp (already 0..1) |
| `notification_rate` | Inbound notifications per hour | Hill sigmoid, midpoint 10/h (30/h → 0.75) |
| `focus_ratio` (inverted) | Share of window in focus blocks | Linear, inverted (1 − ratio) |
| `multitasking_index` | Share of context switches during meetings | Linear clamp (already 0..1) |

## Scoring

Each factor is normalized to 0..1 via a smooth **Hill sigmoid** ``σ(x) = x / (x + midpoint)``
(returns 0.5 at the midpoint, approaches 1.0 asymptotically — no hard saturation), except the
ratios that already live in 0..1. The score combines the weighted base with an **interaction
term** that captures compounding stressors (interruptions during dense meetings):

```text
score = 100 * min(base + interaction, 1.0)

base       = 0.30*σ(switches) + 0.20*meetings + 0.20*σ(notifications)
             + 0.15*(1 − focus) + 0.15*multitasking
interaction = 0.10 * meetings * max(σ(switches), σ(notifications))
```

Interruption frequency (switches + notifications, combined weight 0.50) is the dominant term
because it is the strongest, most consistent behavioral proxy for cognitive load.

Levels: `low` (<25), `moderate` (25–50), `high` (50–75), `overload` (≥75).

A **personal baseline** (`baseline.py`) reframes the absolute score relative to the individual's
own history, with a trend direction and confidence level — so a score of 70 means something
different for someone whose personal average is 45 vs. 60.

### Role profiles & sensitivity tuning

Different knowledge work profiles experience interruption stressors differently. LoadGuard
provides pre-configured, calibrated role profiles (`scoring.py`, `models.py`):

- `developer`: Higher weight on context switching and focus protection to shield deep cognitive tasks.
- `manager`: Higher meeting tolerance with stricter threshold for notifications and context switching.
- `researcher`: Maximizes focus ratio weighting and prioritizes undisturbed study blocks.
- `support`: Tuned for high baseline inbound communication volume.

### Timezone awareness & late-day fatigue protection

Workload planning (`recommender.py`) dynamically adapts to the user's local timezone (`tz_name`):
- Converts epoch timestamps to local workday hours using standard IANA `ZoneInfo`.
- **Late-day fatigue protection**: Past 16:00 local time, cognitive resilience is lower; the planner
  dynamically adjusts delegation thresholds to protect the human from afternoon overload.

### Calendar collision avoidance & multi-calendar merging

When exporting protected focus blocks (`actions.py`):
- `_merge_busy_intervals` combines overlapping and adjacent meetings from merged calendars into disjoint blocks.
- `_find_next_free_slot` traverses busy periods to ensure focus blocks are placed only in genuine free gaps.
- Exported `.ics` calendars respect the local day-boundary horizon and include configurable `VALARM` notifications.

## Granite Decision Agent (proposes, never dictates)

`decision.py` asks Granite for a structured JSON proposal — which task to prioritize, which to
delegate, and when to insert focus/recovery blocks. `validate_proposal` then rejects anything
that:

- references an unknown or non-todo task;
- delegates a critical task (priority ≥ 4);
- invents titles, priorities, deadlines, or data;
- inserts invalid or excessive blocks.

An invalid or missing proposal leaves the deterministic plan untouched. `generated_by` /
`proposed_by` on the plan records exactly which engine produced it, so the demo can show
*"Granite via watsonx"* vs. *"Deterministic fallback"*.

## Granite Guardian (safety gate)

`guardian.py` runs two layers:

1. **Deterministic checks** (always on): well-formed plan, known tasks, no critical delegation,
   no invented data, and narrative checks for medical/burnout diagnosis, disrespectful language,
   sensitive personal data, and out-of-scope advice.
2. **Granite Guardian check** (when a model is configured): the narrative is validated by a
   Granite Guardian-style prompt. A flagged narrative is replaced with the deterministic one.

LoadGuard never diagnoses: *"LoadGuard detects behavioral patterns associated with interruption
overload; it does not diagnose stress, burnout, or any medical condition."*

## Human approval, state machine & actions (closing the loop)

The plan is returned with `status="pending"`. The user can **accept**, **edit**, or **reject** it
(dashboard, REST, or MCP).

### Strict State Machine Lifecycle

State transitions are formally guarded in `actions.py` and `api.py` via `is_valid_transition`:
- `pending` → `accepted` | `rejected` | `edited`
- `edited` → `accepted` | `rejected` | `edited`
- `accepted` (terminal) → *immutable*
- `rejected` (terminal) → *immutable*

Attempting to alter a terminal state returns `400 Bad Request`.

### Data Source Immutability

When a human customizes or renames plan items in `/approve`, the safety gate validates structural invariants against the original source tasks without mutating `stored["tasks"]` in place. Raw input events and tasks remain strictly immutable.

On acceptance, `actions.py`:
- renders the focus/recovery blocks as a real `.ics` calendar (with `VALARM` notifications);
- renders the resequenced task list as CSV;
- appends the decision + feedback to a local audit trail.

## Measurement: projected vs. observed

- **Projected** (`impact.py`): the before/after score under documented assumptions (see below).
- **Observed** (`benchmark.run_pilot_evaluation`): three phases — *baseline*, *projected*,
  *observed* — where observed metrics (interruption reduction during focus blocks, context-switch
  reduction, focus minutes gained, load delta) are only reported when outcome signals are
  supplied. Without outcome data the result is explicitly labelled a reproducible projection.

### Impact Estimator assumptions (projected)

| Plan action | Effect on features (assumption) |
| --- | --- |
| `batch` (notifications) | `notification_rate` halved (consolidated into check-ins) |
| `focus_block` | `focus_ratio` increases by the block's share of the window |
| `break` | `multitasking_index` reduced by 30% |
| `delegate` | `context_switches_per_hour` reduced 10% per delegated task |

## LangGraph orchestration

`langgraph_flow.py` compiles the loop into a real LangGraph `StateGraph` when `langgraph` is
installed:

```text
collect_signals -> compute_features -> diagnose_load -> granite_plan
    -> guardian_validation -> human_approval -> apply_plan -> measure_outcome
```

`human_approval` is an explicit approval gate: without a decision the flow stops in
`awaiting_approval`, and re-invoking it with the decision recomputes the flow
deterministically from the (re-supplied) inputs rather than resuming a persisted LangGraph
checkpoint. The same node functions run sequentially when LangGraph is absent.

## LLM runtime

`LLM_PROVIDER` accepts `heuristic` (default), `watsonx`, or `ollama`. The `heuristic` provider uses the dependency-free deterministic planner, heuristic note, and deterministic safety guard. The other providers add model-backed proposals, notes, and Guardian checks, while retaining deterministic fallbacks.

`ChatModel` exposes three capabilities, each with a deterministic fallback:

- `generate_note()` — narrative.
- `propose_plan()` — structured decision proposal (JSON).
- `guard_text()` — Granite Guardian-style safety check.

Implementations: `HeuristicModel` (stdlib), `WatsonxModel` (IBM Granite through `langchain_ibm.ChatWatsonx`), and `OllamaModel` (local Granite through Ollama's `/api/generate` endpoint).

Configuration defaults from `config.py`:

| Provider | Main model | Guardian model |
| --- | --- | --- |
| `heuristic` | No external model | Deterministic guard |
| `watsonx` | `ibm/granite-3-8b-instruct` | `ibm/granite-guardian-3-8b` |
| `ollama` | `granite3.1-dense:8b` | `ibm-granite/granite-guardian:3.1-8b` |

Override the model IDs with `WATSONX_MODEL_ID`, `WATSONX_GUARDIAN_MODEL_ID`, `OLLAMA_MODEL`, and `OLLAMA_GUARDIAN_MODEL`. `WATSONX_API_KEY` and `WATSONX_PROJECT_ID` are required by the watsonx integration; `WATSONX_URL` defaults to `https://us-south.ml.cloud.ibm.com`; `OLLAMA_URL` defaults to `http://localhost:11434`.

```bash
cp .env.example .env
# set LLM_PROVIDER=watsonx + WATSONX_API_KEY / WATSONX_PROJECT_ID
# or LLM_PROVIDER=ollama (local, no cloud keys)
```

## Team availability & structured delegation

`availability.py` and `recommender.py` model team workload distribution:
- An `Absence` stores only the window and its type (`vacation` or `leave`) — never personal/medical data.
- **Structured Delegation**: `PlanItem` carries `suggested_assignees: list[str]` and `delegate_to: Optional[str]`.
- Eligible low-priority tasks (`priority <= 2`) are delegated only to active teammates who are free of overlapping absences.
- **Task Locking Constraint**: Tasks with `locked=True` (and optional `locked_start_time`) are treated as fixed commitments and are strictly excluded from delegation even under `OVERLOAD` conditions.
- Reassignment suggestions and interactive assignment chips allow seamless human delegation.

`scripts/capture_signals.py` extracts absences from real calendars (all-day events, `OOF` status).

## Daily cycle: morning analysis + midday re-organization

`projection.py` projects the end-of-day load from partial-day observations:
1. **Morning** — `run_workflow` produces the morning strategy and initial plan.
2. **Midday** — `run_midday_review` tracks completed tasks, re-scores observed load, and re-plans for the afternoon if projected load is `high` or `overload`.

## Workday Hours & Calendar Schedule Anchoring

`actions.export_ics` translates the optimized plan into RFC 5545 `.ics` calendar events:
- **Configurable Workday Window**: `workday_start` (default `09:00`) anchors the start of the daily schedule; `workday_end` (default `18:00`) prevents focus blocks and tasks from spilling past working hours.
- **Collision Avoidance**: `_merge_busy_intervals` and `_find_next_free_slot` place focus blocks and rescheduled tasks exclusively in genuine free gaps between existing meetings.
- **Category Tagging**: Tasks and delegation hand-offs are tagged with `CATEGORIES:LOADGUARD-TASK` and `CATEGORIES:LOADGUARD-HANDOFF`.

## Calendar Ingestion & Date Filtering

The `POST /ingest` endpoint accepts raw `.ics` or `.jsonl` signals. For calendar files containing multi-day or monthly exports, the `date` parameter (`YYYY-MM-DD`) isolates analysis to the targeted day, defaulting to the date of the first event in the file or today's date.

## Enterprise Observability & Telemetry

Every request through the FastAPI layer passes through correlation tracking middleware:
- `X-Request-ID`: Trace identifier propagated across headers and logs.
- `X-Response-Time-Ms`: High-resolution execution latency in milliseconds.
- `telemetry` block: Returns `duration_ms`, `llm_provider`, and Granite Guardian validation status.
- Structured non-PII logging for enterprise monitoring.

## Persistence, Concurrency and Security Boundaries

- **State Directory**: `.loadguard/` stores runtime plans, history, and audit logs (ignored by Git).
- **Concurrency & Race Conditions**: Mutex `_PLANS_LOCK` and atomic per-thread temporary file writes (`.tmp.<tid>`) prevent corrupted writes under parallel load.
- **Authentication**: when `LOADGUARD_API_KEY` is set, analytical and mutating endpoints enforce the `X-API-Key` header; `/health` remains public. When unset, the API is intended for trusted local use.
- **Destructive Operation Guard**: `LOADGUARD_ALLOW_DELETE=false` disables `DELETE /history` and `DELETE /audit`; enable this setting for network-exposed deployments.
- **Input Validation**: Pydantic `@field_validator("role")` returns `422 Unprocessable Entity` for unrecognized roles; `Task` defends against out-of-range priority and negative duration.

## Extensibility

New signal sources (calendar, Slack, OS window focus) plug into `signals.py` by emitting `Event`
objects — no changes to scoring or planning are required.
