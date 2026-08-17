# LoadGuard Architecture

## Overview

LoadGuard is a **cognitive-load-aware AI co-worker**. It treats the human's
attention as a first-class, finite resource and proactively reshapes the workday
to protect it.

The system follows a strict, reviewable pipeline:

```
signals (raw events) -> features (explainable proxies) -> load score (0-100)
        -> deterministic plan (resequence / delegate / recover) -> narrative (LLM)
```

The deterministic layer is the source of truth for *what* to do; the LLM only
adds a human-readable explanation of *why*. This hybrid design keeps the
prototype grounded, reproducible, and safe, while still showcasing generative AI.

## Design principles

1. **Privacy first.** Only counts, durations, and opaque labels are captured —
   never screen content, keystrokes, or message bodies. Features are computed
   locally; only derived aggregates are sent to the LLM.
2. **Explainability.** Every score is a weighted combination of named proxies
   with documented thresholds, so any recommendation can be traced back to a
   signal.
3. **Honesty about limits.** The "Cognitive Load Score" is explicitly a
   *behavioral proxy*, not a physiological measurement. Overclaiming would be
   both scientifically wrong and penalized by technical judges.
4. **Degradability.** Without API keys or optional dependencies, the full
   pipeline still runs via the heuristic engine.

## Modules

| Module | Responsibility |
| --- | --- |
| `models.py` | Dataclasses: `Event`, `FeatureSet`, `Task`, `LoadReport`, `Plan`. |
| `signals.py` | Parse JSONL/API events; aggregate into per-hour/ratio features. |
| `scoring.py` | Weighted normalization of features into a 0–100 score + level. |
| `recommender.py` | Deterministic planner: resequence, delegate, insert breaks/focus. |
| `llm.py` | `ChatModel` interface; heuristic + watsonx (IBM Granite) providers. |
| `config.py` | Select the model provider from environment variables. |
| `api.py` / `app.py` | Optional FastAPI HTTP layer. |

## Signal proxies

| Feature | What it measures | Normalization threshold |
| --- | --- | --- |
| `context_switches_per_hour` | Interruption frequency | 12/h → 1.0 |
| `meeting_ratio` | Share of window in meetings | 1.0 (already 0..1) |
| `notification_rate` | Inbound notifications per hour | 30/h → 1.0 |
| `focus_ratio` (inverted) | Share of window in focus blocks | 1.0 (already 0..1) |
| `multitasking_index` | Share of context switches during meetings | 1.0 (already 0..1) |

## Scoring

```
score = 100 * (0.30*switches + 0.20*meetings + 0.20*notifications
               + 0.15*(1 - focus) + 0.15*multitasking)
```

Interruption frequency (switches + notifications, combined weight 0.50) is the
dominant term because it is the strongest, most consistent behavioral proxy for
cognitive load in the literature.

Levels: `low` (<25), `moderate` (25–50), `high` (50–75), `overload` (≥75).

## Planner rules

- Tasks are ordered by priority (desc), then deadline (asc).
- Under `high`/`overload`, low-priority tasks are marked for delegation.
- During `high`/`overload`, a recovery break is inserted every two tasks.
- When notification rate or focus time crosses a threshold, a "batch
  notifications" and/or "focus block" action is added.

## LLM runtime

`ChatModel.generate_note()` returns the plan narrative. Two implementations:

- `HeuristicModel` — templated text, zero dependencies (default).
- `WatsonxModel` — IBM Granite via `langchain-ibm`; falls back to heuristic on
  any error.

To enable watsonx:

```bash
cp .env.example .env
# set LLM_PROVIDER=watsonx and add WATSONX_API_KEY / WATSONX_PROJECT_ID
```

## Extensibility

New signal sources (calendar, Slack, OS window focus) plug into `signals.py` by
emitting `Event` objects — no changes to scoring or planning are required.
