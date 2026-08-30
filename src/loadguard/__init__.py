"""LoadGuard core package.

A cognitive-load-aware AI co-worker that ingests lightweight, privacy-preserving
work signals, computes an explainable Cognitive Load Score, and proactively
resequences or delegates tasks to prevent "AI brain fry".

The core pipeline (models, signals, scoring, recommender, agents, workflow,
impact) is dependency-free and uses only the Python standard library so the
prototype is reproducible by judges without installing anything.
"""

from .config import get_model
from .impact import ImpactResult, estimate_impact
from .models import Event, FeatureSet, LoadReport, Plan, PlanItem, Task
from .recommender import build_plan
from .scoring import score
from .signals import compute_features, load_events
from .workflow import WorkflowResult, run_workflow

__all__ = [
    "Event",
    "FeatureSet",
    "LoadReport",
    "Plan",
    "PlanItem",
    "Task",
    "compute_features",
    "load_events",
    "score",
    "build_plan",
    "ImpactResult",
    "estimate_impact",
    "WorkflowResult",
    "run_workflow",
    "get_model",
]

__version__ = "0.2.0"
