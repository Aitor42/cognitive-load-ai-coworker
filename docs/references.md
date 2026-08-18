# References

Sources supporting the problem statement and design rationale.

## Interruption & context switching

- Mark, G., Gudith, D., & Klocke, U. (2008). *The Cost of Interrupted Work: More Speed and Stress.*
  Proceedings of CHI 2008. Found it takes an average of **~23 minutes** to return to a task after
  an interruption, and that interrupted work is performed faster with higher stress.
- American Psychological Association. *Multitasking: Switching costs.* Estimates that task
  switching can consume **up to 40% of productive time**.

## Burnout & cognitive load

- World Health Organization (ICD-11, 2019). Classifies **burnout** as an occupational phenomenon
  resulting from chronic workplace stress that has not been successfully managed.
- Sweller, J. (1988). *Cognitive load during problem solving.* Cognitive Science. Foundational work
  on cognitive load theory: limited working-memory capacity constrains performance.

## Why LoadGuard uses proxies, not physiological measurement

- LoadGuard's "Cognitive Load Score" is a **behavioral proxy** (counts and ratios of interruption
  signals), not a physiological measurement. This is deliberate: proxies are cheap, private, and
  explainable, and they avoid the validity problems of consumer-grade physiological claims.

## Related alarm-fatigue literature (methodological inspiration)

- Hundman, K., et al. (2018). *Detecting Spacecraft Anomalies Using LSTMs and Nonparametric
  Dynamic Thresholding.* KDD 2018. Documents how static-threshold alerting floods operators with
  false alarms and misses contextual anomalies — the same *noise vs. signal* tension LoadGuard
  addresses for human attention (alert fatigue ↔ interruption fatigue).
