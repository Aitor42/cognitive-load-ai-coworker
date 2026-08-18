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

See [`architecture.mmd`](architecture.mmd) for the full Mermaid diagram.

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
| `models.py` | Dataclasses: `Event`, `FeatureSet`, `Task`, `LoadReport`, `Plan`. |
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
| `benchmark.py` | Objective metrics + three-phase pilot evaluation. |
| `llm.py` | `ChatModel` interface; heuristic + watsonx + ollama (Granite). |
| `config.py` | Select the model / guardian provider from environment variables. |
| `api.py` / `app.py` | Optional FastAPI HTTP layer + dashboard. |

## Signal proxies

| Feature | What it measures | Normalization threshold |
| --- | --- | --- |
| `context_switches_per_hour` | Interruption frequency | 12/h → 1.0 |
| `meeting_ratio` | Share of window in meetings | 1.0 (already 0..1) |
| `notification_rate` | Inbound notifications per hour | 30/h → 1.0 |
| `focus_ratio` (inverted) | Share of window in focus blocks | 1.0 (already 0..1) |
| `multitasking_index` | Share of context switches during meetings | 1.0 (already 0..1) |

## Scoring

```text
score = 100 * (0.30*switches + 0.20*meetings + 0.20*notifications
               + 0.15*(1 - focus) + 0.15*multitasking)
```

Interruption frequency (switches + notifications, combined weight 0.50) is the dominant term
because it is the strongest, most consistent behavioral proxy for cognitive load.

Levels: `low` (<25), `moderate` (25–50), `high` (50–75), `overload` (≥75).

A **personal baseline** (`baseline.py`) reframes the absolute score relative to the individual's
own history, with a trend direction and confidence level — so a score of 70 means something
different for someone whose personal average is 45 vs. 60.

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

## Extensibility

New signal sources (calendar, Slack, OS window focus) plug into `signals.py` by emitting `Event`
objects — no changes to scoring or planning are required.
