"""Unit tests for runtime provider selection (``config.py``)."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loadguard import config  # noqa: E402
from loadguard.llm import HeuristicModel, OllamaModel, WatsonxModel  # noqa: E402


class TestGetModel(unittest.TestCase):
    def test_default_is_heuristic(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsInstance(config.get_model(), HeuristicModel)

    def test_watsonx_provider(self) -> None:
        env = {"LLM_PROVIDER": "watsonx", "WATSONX_API_KEY": "k", "WATSONX_PROJECT_ID": "p"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertIsInstance(config.get_model(), WatsonxModel)

    def test_ollama_provider(self) -> None:
        with mock.patch.dict(os.environ, {"LLM_PROVIDER": "ollama"}, clear=True):
            self.assertIsInstance(config.get_model(), OllamaModel)

    def test_unknown_provider_falls_back_to_heuristic(self) -> None:
        with mock.patch.dict(os.environ, {"LLM_PROVIDER": "bogus"}, clear=True):
            self.assertIsInstance(config.get_model(), HeuristicModel)


class TestGetGuardianModel(unittest.TestCase):
    def test_default_is_none(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(config.get_guardian_model())

    def test_watsonx_guardian(self) -> None:
        env = {"LLM_PROVIDER": "watsonx", "WATSONX_API_KEY": "k", "WATSONX_PROJECT_ID": "p"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertIsInstance(config.get_guardian_model(), WatsonxModel)

    def test_ollama_guardian(self) -> None:
        with mock.patch.dict(os.environ, {"LLM_PROVIDER": "ollama"}, clear=True):
            self.assertIsInstance(config.get_guardian_model(), OllamaModel)


if __name__ == "__main__":
    unittest.main()
