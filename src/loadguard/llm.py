"""ChatModel interface and providers.

The structured plan is always produced by the deterministic recommender, so the
prototype is reproducible and grounded. The LLM layer contributes three things,
all optional and all falling back to a deterministic engine when unavailable:

- **Narrative** (``generate_note``): plain-language explanation of the plan.
- **Decision proposal** (``propose_plan``): a structured JSON proposal of *which
  task to prioritize, which to delegate, and when to insert focus/breaks*. It is
  always validated by a deterministic gate (``decision.validate_proposal``)
  before it can change the plan.
- **Guard check** (``guard_text``): a Granite Guardian-style safety check of the
  narrative (respect, no medical diagnosis, no sensitive data, in-scope).

- ``HeuristicModel``: dependency-free templated narrative (default).
- ``WatsonxModel``: optional IBM Granite via langchain-ibm/watsonx.
- ``OllamaModel``: local IBM Granite via Ollama (no cloud keys, stdlib HTTP).
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any

from .models import FeatureSet, LoadReport, Plan, Task

logger = logging.getLogger(__name__)

PROPOSE_PLAN_PROMPT = """\
You are LoadGuard's Decision Agent, part of an AI co-worker that protects \
knowledge workers from cognitive overload. Propose *adjustments* to a day plan \
to protect the user's attention budget.

Current load score: {score:.0f}/100 ({level}).
Tasks (JSON): {tasks}

Respond with JSON ONLY. No markdown, no commentary:

{{
  "priority_task_id": "<id of the task to do first, or null>",
  "delegate_task_ids": ["<ids of low-priority tasks to delegate>"],
  "inserts": [{{"action": "focus_block" or "break", "after_task_id": "<task id, or null for the start>"}}],
  "rationale": "<2-3 sentence plain explanation>"
}}

Constraints:
- Only reference task ids from the provided list.
- Never delegate tasks with priority >= 4 (critical work).
- Never invent tasks, priorities, deadlines, or any data.
- Do not diagnose stress, burnout, or any medical condition.
- Insert at most 2 blocks.
"""

GUARD_PROMPT = """\
You are Granite Guardian, a safety guardrail. Validate the following LoadGuard \
note. Return JSON ONLY: {{"safe": true or false, "issues": ["<reason>", ...]}}.

Rules:
1. Language is respectful; no insults or demeaning tone.
2. No medical or burnout diagnosis of the user (no "you have burnout", "you are \
stressed/depressed", etc.).
3. No sensitive personal data (third-party names, emails, phone numbers).
4. Recommendations stay in scope: only work schedule / tasks / attention.
5. No invented facts or numbers.

Note: {text}
"""


class ChatModel(ABC):
    """Interface for generating the human-readable plan narrative."""

    name = "base"

    @abstractmethod
    def generate_note(self, load_report: LoadReport, plan: Plan, tasks: list[Task]) -> str:
        """Return a short, plain-language explanation of the plan."""

    def propose_plan(
        self, features: FeatureSet, load_report: LoadReport, tasks: list[Task]
    ) -> str | None:
        """Return a JSON proposal string, or None to use the deterministic plan."""
        return None

    def guard_text(self, text: str) -> dict[str, Any] | None:
        """Validate text; return {'safe': bool, 'issues': [...]} or None."""
        return None


class HeuristicModel(ChatModel):
    """Deterministic narrative based on load level (no API required)."""

    name = "heuristic"

    def generate_note(self, load_report: LoadReport, plan: Plan, tasks: list[Task]) -> str:
        delegated = [i for i in plan.items if i.action == "delegate"]
        blocks = [i for i in plan.items if i.action in ("focus_block", "break")]
        note = f"Your Cognitive Load Score is {load_report.score:.0f}/100 ({load_report.level}). "
        if delegated:
            titles = ", ".join(f'"{i.title}"' for i in delegated)
            note += f"I suggest delegating {len(delegated)} low-priority task(s): {titles}. "
        if blocks:
            note += f"I inserted {len(blocks)} recovery/focus block(s) to protect your attention. "
        if load_report.level in ("high", "overload"):
            note += "Consider batching notifications and blocking focus time before deep work."
        else:
            note += "Load looks manageable; use the extra headroom for deep, high-focus work."
        return note


class RemoteModel(ChatModel):
    """Shared behaviour for model-backed providers.

    Subclasses implement ``_invoke(prompt)``; this base layers the narrative,
    the structured decision proposal, and the Granite Guardian check on top, all
    with safe fallbacks so the pipeline never fails for want of a model.
    """

    def __init__(self) -> None:
        self._fallback = HeuristicModel()

    @abstractmethod
    def _invoke(self, prompt: str) -> str:
        """Send a prompt and return the raw text response."""

    def generate_note(self, load_report: LoadReport, plan: Plan, tasks: list[Task]) -> str:
        try:
            task_list = ", ".join(t.title for t in tasks[:10]) or "(none)"
            prompt = (
                "You are LoadGuard, an AI co-worker that prevents cognitive overload. "
                f"Load score: {load_report.score:.0f}/100 ({load_report.level}). "
                f"Planned actions: {[i.title for i in plan.items]}. "
                f"Tasks: {task_list}. "
                "Write a concise, empathetic 2-3 sentence note explaining the plan "
                "and protecting the user's attention. No markdown."
            )
            text = self._invoke(prompt)
            return text.strip() or self._fallback.generate_note(load_report, plan, tasks)
        except Exception:  # pragma: no cover - depends on optional deps/credentials
            logger.warning(
                "%s call failed; falling back to heuristic", type(self).__name__, exc_info=True
            )
            return self._fallback.generate_note(load_report, plan, tasks)

    def propose_plan(
        self, features: FeatureSet, load_report: LoadReport, tasks: list[Task]
    ) -> str | None:
        try:
            task_json = json.dumps(
                [
                    {
                        "id": t.id,
                        "title": t.title,
                        "priority": t.priority,
                        "status": t.status,
                    }
                    for t in tasks[:15]
                ]
            )
            prompt = PROPOSE_PLAN_PROMPT.format(
                score=load_report.score, level=load_report.level, tasks=task_json
            )
            text = self._invoke(prompt)
            return text.strip() or None
        except Exception:  # pragma: no cover - depends on optional deps/credentials
            logger.warning(
                "%s propose_plan failed; using deterministic plan",
                type(self).__name__,
                exc_info=True,
            )
            return None

    def guard_text(self, text: str) -> dict[str, Any] | None:
        try:
            out = self._invoke(GUARD_PROMPT.format(text=text))
            return _parse_guard_json(out)
        except Exception:  # pragma: no cover - depends on optional deps/credentials
            logger.warning(
                "%s guard check failed; using deterministic guard",
                type(self).__name__,
                exc_info=True,
            )
            return None


def _parse_guard_json(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from a model response."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    if "safe" not in data:
        return None
    issues = data.get("issues")
    if not isinstance(issues, list):
        issues = []
    return {"safe": bool(data["safe"]), "issues": [str(i) for i in issues]}


class WatsonxModel(RemoteModel):
    """Optional IBM Granite via watsonx (langchain-ibm).

    Requires ``pip install langchain langchain-ibm`` and valid watsonx
    credentials. Falls back to the heuristic narrative on any error so the
    pipeline never fails for want of a model.
    """

    name = "watsonx"

    def __init__(
        self,
        api_key: str,
        project_id: str,
        url: str,
        model_id: str,
    ) -> None:
        super().__init__()
        self._api_key = api_key
        self._project_id = project_id
        self._url = url
        self._model_id = model_id

    def _invoke(self, prompt: str) -> str:
        from langchain_ibm import ChatWatsonx  # type: ignore

        model = ChatWatsonx(
            model_id=self._model_id,
            url=self._url,
            project_id=self._project_id,
            apikey=self._api_key,
        )
        result = model.invoke(prompt)
        return str(getattr(result, "content", result))


class OllamaModel(RemoteModel):
    """Local IBM Granite via Ollama (no cloud credentials).

    Uses the Ollama HTTP API with the standard library only, so the AI component
    runs locally and out of the box once ``ollama serve`` is running. Falls back
    to the heuristic narrative on any error.

    Setup: ``ollama pull granite3.1-dense:8b``
    """

    name = "ollama"

    def __init__(
        self,
        model_id: str = "granite3.1-dense:8b",
        url: str = "http://localhost:11434",
    ) -> None:
        super().__init__()
        self._model_id = model_id
        self._url = url.rstrip("/")

    def _invoke(self, prompt: str) -> str:
        from urllib import request

        payload = json.dumps({"model": self._model_id, "prompt": prompt, "stream": False}).encode(
            "utf-8"
        )
        req = request.Request(
            f"{self._url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("response", "").strip()
