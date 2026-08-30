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
| `api.py` / `app.py` | Optional FastAPI HTTP layer + dashboard. |

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

## Human approval & actions (closing the loop)

The plan is returned with `status="pending"`. The user can **accept**, **edit**, or **reject** it
(dashboard, REST, or MCP). On acceptance, `actions.py`:

- renders the focus/recovery blocks as a real `.ics` calendar;
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

`ChatModel` exposes three capabilities, each with a deterministic fallback:

- `generate_note()` — narrative.
- `propose_plan()` — structured decision proposal (JSON).
- `guard_text()` — Granite Guardian-style safety check.

Implementations: `HeuristicModel` (stdlib), `WatsonxModel` (IBM Granite via `langchain-ibm`),
`OllamaModel` (local Granite via Ollama).

```bash
cp .env.example .env
# set LLM_PROVIDER=watsonx + WATSONX_API_KEY / WATSONX_PROJECT_ID
# or LLM_PROVIDER=ollama (local, no cloud keys)
```

## Team availability & reassignment alerts

`availability.py` models when a worker is unavailable. An `Absence` stores only the window
and its type (`vacation` or `leave`) — never a medical or personal reason.
`find_reassignment_alerts` walks todo tasks and, when a task has an assignee, a future
deadline, and that assignee has an absence overlapping `[now, deadline]`, emits a
`ReassignmentAlert` listing the teammates available for the whole window.

The alert is a *suggestion*: the human decides whether to reassign. It never mutates the
plan automatically, consistent with "Granite proposes, LoadGuard validates, the human
decides".

`scripts/capture_signals.py` extracts absences from a real calendar ICS: out-of-office /
vacation events (all-day `VALUE=DATE` events, an `X-MICROSOFT-CDO-BUSYSTATUS:OOF` flag, or
a matching summary) become `Absence` records. Only the fact and type are kept — the event
summary is never captured, so no medical or personal reason leaks into the data.

## Daily cycle: morning analysis + midday re-organization

`projection.py` projects the end-of-day load from partial-day observations. The only
assumption is conservative and documented: the remaining day continues at the observed
per-hour rates and ratios (or at explicitly supplied remaining-day features). Features are
blended as a time-weighted average, then re-scored.

`scheduler.py` ties the two scheduled beats together:

1. **Morning** — the full `run_workflow` loop produces the initial plan.
2. **Midday** — `run_midday_review` re-scores the day so far, projects the remainder, and
   re-plans (deterministically, then guarded) when the projected end-of-day level is
   `high` or `overload`.

`scripts/schedule.py` is a cron-friendly CLI for both beats; the functions themselves are
pure, so any scheduler (cron, an in-process loop, APScheduler) can drive them.

## Persistence and deployment boundaries

The optional API stores plans, score history, and approval audit records under `.loadguard/` in the repository root. This directory is local runtime state and is ignored by Git. The API has no authentication and exposes deletion endpoints; bind it to `127.0.0.1` for local use or add an authentication/reverse-proxy layer before exposing it to a network. Docker binds to `0.0.0.0` inside the container so port publishing works.

## Extensibility

New signal sources (calendar, Slack, OS window focus) plug into `signals.py` by emitting `Event`
objects — no changes to scoring or planning are required.
