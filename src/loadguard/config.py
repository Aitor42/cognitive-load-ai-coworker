"""Runtime configuration: select the ChatModel provider from environment."""

from __future__ import annotations

import os

from .llm import ChatModel, HeuristicModel, OllamaModel, WatsonxModel

# Granite Guardian model ids used for the safety check (prompt-based guard).
WATSONX_GUARDIAN_MODEL = "ibm/granite-guardian-3-8b"
OLLAMA_GUARDIAN_MODEL = "ibm-granite/granite-guardian:3.1-8b"


def _watsonx(api_key: str, project_id: str, url: str, model_id: str) -> WatsonxModel:
    return WatsonxModel(
        api_key=api_key,
        project_id=project_id,
        url=url,
        model_id=model_id,
    )


def _ollama(model_id: str, url: str) -> OllamaModel:
    return OllamaModel(model_id=model_id, url=url)


def get_model() -> ChatModel:
    """Return a ChatModel based on ``LLM_PROVIDER`` (default: heuristic)."""
    provider = os.environ.get("LLM_PROVIDER", "heuristic").strip().lower()
    if provider == "watsonx":
        return _watsonx(
            api_key=os.environ.get("WATSONX_API_KEY", ""),
            project_id=os.environ.get("WATSONX_PROJECT_ID", ""),
            url=os.environ.get("WATSONX_URL", "https://us-south.ml.cloud.ibm.com"),
            model_id=os.environ.get("WATSONX_MODEL_ID", "ibm/granite-3-8b-instruct"),
        )
    if provider == "ollama":
        return _ollama(
            model_id=os.environ.get("OLLAMA_MODEL", "granite3.1-dense:8b"),
            url=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
        )
    return HeuristicModel()


def get_guardian_model() -> ChatModel | None:
    """Return a Granite Guardian model for the safety check, or None.

    None is returned when the provider is heuristic, meaning the deterministic
    guard (``guardian.run_deterministic_checks``) is used instead.
    """
    provider = os.environ.get("LLM_PROVIDER", "heuristic").strip().lower()
    if provider == "watsonx":
        return _watsonx(
            api_key=os.environ.get("WATSONX_API_KEY", ""),
            project_id=os.environ.get("WATSONX_PROJECT_ID", ""),
            url=os.environ.get("WATSONX_URL", "https://us-south.ml.cloud.ibm.com"),
            model_id=os.environ.get("WATSONX_GUARDIAN_MODEL_ID", WATSONX_GUARDIAN_MODEL),
        )
    if provider == "ollama":
        return _ollama(
            model_id=os.environ.get("OLLAMA_GUARDIAN_MODEL", OLLAMA_GUARDIAN_MODEL),
            url=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
        )
    return None
