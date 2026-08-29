# Planning Session — IBM Bob

**Mode:** Cognitive Load Engineer (`cognitive-load-engineering`)

## What Bob contributed

- Drafted the solution design: a **sense → diagnose → propose → validate → approve → act → measure** pipeline.
- Proposed the **hybrid deterministic + LLM** architecture (deterministic scoring as guardrail,
  LLM as narrator), so recommendations stay grounded and reproducible.
- Defined the five behavioral proxies and their normalization thresholds
  (`context_switches_per_hour`, `meeting_ratio`, `notification_rate`, `focus_ratio`,
  `multitasking_index`).
- Designed the multi-agent decomposition (SignalAnalyst → LoadDiagnostician → WorkloadPlanner →
  Decision Agent → Guardian → Narrator) and the Impact Estimator.
- Scoped the MVP for a two-week build: dependency-free core, optional FastAPI + watsonx layers.

## Key decisions locked in

1. Privacy-first: only counts/ratios reach the model, never raw content.
2. Honest framing: "Cognitive Load Score" is a *behavioral proxy*, not a physiological measurement.
3. Degradability: the full pipeline must run without API keys (heuristic fallback).
