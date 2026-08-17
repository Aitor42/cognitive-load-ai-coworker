# LoadGuard — Cognitive-Load-Aware AI Co-Worker

> **IBM AI Builders Challenge · August 2026 · Wildcard: "Build Intelligent Systems for the Future of Work"**
>
> An AI co-worker that protects knowledge workers from **"AI brain fry"** — the cognitive
> overload caused by too many AI-driven interruptions — by sensing overload and proactively
> resequencing or delegating tasks.

---

## Problem statement

Knowledge workers now juggle a growing number of AI assistants: each one adds notifications,
context switches, and micro-decisions. The paradox of 2026's AI-first workplace is that more
"productivity" tools often produce *less* productive, more exhausted humans.

Existing tools optimize **output**. Almost none protect the human's **attention budget**.
The result is a measurable, real-world problem: reduced focus, higher error rates, and burnout.

**LoadGuard answers one question:** *when does the AI need to slow down, so the human can keep up?*

## Solution

LoadGuard is a cognitive-load-aware AI co-worker that:

1. **Ingests lightweight, privacy-respecting signals** — context switches, meeting density,
   notification rate, and focus blocks (all proxies, never raw screen/keystroke content).
2. **Computes an explainable Cognitive Load Score (0–100)** from transparent, weighted
   proxies. We are explicit that this is a *behavioral proxy*, not a physiological measurement.
3. **Reasons with an LLM** (IBM Granite via watsonx / LangChain) over the load report plus the
   user's task list to **proactively resequence the day, delegate low-priority work, and insert
   recovery/focus blocks** before overload happens.

LoadGuard is an **AI co-worker whose job is to protect you from the other AI co-workers**.

## Key features

- **Signal ingestion** — JSONL event stream plus adapter-ready interfaces for calendar and
  notification sources.
- **Explainable Cognitive Load Score** — 0–100 with per-factor contributions and a
  low / moderate / high / overload level, so users can *see why* a recommendation was made.
- **Proactive task resequencing** — reorders the task list against load, priority, and deadlines.
- **Delegation suggestions** — flags low-priority, low-focus tasks that can be handed off.
- **Recovery & focus blocks** — schedules focus time and breaks into the plan.
- **Local-first & privacy-preserving** — signals are processed on-device; only derived features
  (counts/ratios) are sent to the LLM.
- **Deterministic fallback engine** — the full pipeline runs with zero API keys, making the
  prototype reproducible by judges.

## How it works

```
         ┌──────────────────────────── signals (JSONL / adapters) ────────────────────────────┐
         │  context_switch · meeting · notification · focus_block                              │
         ▼                                                                                     │
  ┌─────────────────────┐    features    ┌──────────────────────┐   load report   ┌──────────┐│
  │  signals.py         │ ────────────► │  scoring.py           │ ──────────────► │ LLM      ││
  │  ingest & aggregate │               │  weighted load score  │                 │ (Granite/││
  └─────────────────────┘               └──────────────────────┘                 │  watsonx)││
                                                                                  └────┬─────┘│
                                                         + task list                    │       │
                                                                                        ▼       │
                                                              ┌────────────────────────────────┐ │
                                                              │  recommender.py                │ │
                                                              │  resequence · delegate ·      │ │
                                                              │  insert focus/recovery blocks │ │
                                                              └────────────────────────────────┘ │
                                                                           │                     │
                                                                           ▼                     │
                                                              plan (JSON) + human-readable note  │
```

1. **Ingest** — events are read from a JSONL file (or pushed via the REST API).
2. **Score** — features are computed per time window and combined into a 0–100 score.
3. **Reason** — the load report + tasks are sent to the LLM (or the deterministic engine) which
   produces a resequenced plan with a plain-language explanation.
4. **Act** — the plan is returned as structured JSON plus a human-readable note.

## AI approach

- **Hybrid architecture**: a deterministic, explainable scoring layer acts as the *guardrail*,
  while the LLM contributes *planning and natural-language reasoning*. This keeps recommendations
  grounded and reproducible even when the model is swapped out.
- **Structured reasoning**: the LLM receives a compact, typed context (features, task list,
  constraints) and returns structured output (JSON), not free text.
- **Model-agnostic**: a `ChatModel` interface makes the runtime swappable between IBM Granite
  (watsonx), other LangChain-compatible models, and the built-in heuristic engine.

## IBM Bob usage

IBM Bob was used as the **primary development tool** across the full software lifecycle:

- **Planning & design** — Bob helped draft the solution design, module boundaries, and the
  implementation plan (Specification-Driven Development).
- **Implementation** — Bob generated and iterated on the Python modules (`signals`, `scoring`,
  `recommender`), the FastAPI layer, and the demo script.
- **Testing** — Bob generated unit tests for the scoring engine and edge cases.
- **Debugging & review** — Bob was used to troubleshoot and review changes during development.

The runtime AI component uses **IBM Granite via watsonx** (recommended technology) through a
LangChain-compatible interface, with a deterministic fallback so the prototype is reproducible
without credentials.

## Tech stack

- **Python 3.11+** (core pipeline is dependency-free: `dataclasses` + stdlib)
- **FastAPI + Uvicorn** (optional REST API)
- **LangChain / LangChain-IBM** (optional LLM runtime, IBM Granite via watsonx)
- **IBM Bob** (primary development tool)

## Getting started

### 1. Run the demo (no dependencies, no API keys)

```bash
python demo/demo.py
```

This ingests `demo/sample_events.jsonl`, computes the Cognitive Load Score, and prints the
resequenced plan. It works out of the box with only the Python standard library.

### 2. Run the unit tests

```bash
python -m unittest discover -s tests
```

### 3. Run the REST API (optional)

```bash
pip install -r requirements.txt
python app.py
# POST /analyze  with {"events": [...], "tasks": [...]}
# GET  /health
```

### 4. Enable the IBM Granite runtime (optional)

```bash
cp .env.example .env   # add your watsonx credentials
```

Set `LLM_PROVIDER=watsonx` in `.env`. Without it, the pipeline uses the deterministic engine.

## Project structure

```
cognitive-load-ai-coworker/
├── app.py                    # FastAPI entrypoint
├── src/loadguard/
│   ├── models.py             # dataclasses: Event, Task, LoadReport, Plan
│   ├── signals.py            # ingest events, compute features
│   ├── scoring.py            # weighted Cognitive Load Score (0–100)
│   ├── llm.py                # ChatModel interface + heuristic/watsonx providers
│   ├── recommender.py        # resequence / delegate / focus-block planning
│   └── api.py                # FastAPI routes
├── demo/
│   ├── sample_events.jsonl   # a realistic half-day of signals
│   └── demo.py               # zero-dependency CLI demo
├── tests/
│   └── test_scoring.py       # unit tests for the scoring engine
└── docs/
    └── architecture.md       # detailed architecture + scoring rationale
```

## Demo

A 3-minute demo walkthrough is linked here (to be added before submission). It shows a simulated
"overload morning" — back-to-back meetings, a notification storm, and rapid context switching —
and LoadGuard detecting the overload, resequencing the afternoon, delegating two low-priority
tasks, and inserting a focus block.

## Team

- *(add team member names, universities, and roles here)*

## License

MIT — see [LICENSE](LICENSE).
