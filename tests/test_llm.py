"""Unit tests for the ChatModel layer (``llm.py``)."""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loadguard.llm import (  # noqa: E402
    HeuristicModel,
    OllamaModel,
    RemoteModel,
    WatsonxModel,
    _parse_guard_json,
)
from loadguard.models import FeatureSet, LoadReport, Plan, PlanItem  # noqa: E402


def _report(score: float = 80.0, level: str = "overload") -> LoadReport:
    return LoadReport(score=score, level=level)


def _plan(items: list[PlanItem] | None = None, level: str = "overload") -> Plan:
    return Plan(load_report=_report(level=level), items=items or [])


class _FakeRemote(RemoteModel):
    def __init__(self, response: str = "", fail: bool = False) -> None:
        super().__init__()
        self._response = response
        self._fail = fail

    def _invoke(self, prompt: str) -> str:
        if self._fail:
            raise RuntimeError("boom")
        return self._response


class TestHeuristicModel(unittest.TestCase):
    def test_note_mentions_delegation_and_blocks(self) -> None:
        items = [
            PlanItem(position=1, action="delegate", task_id="c", title="Emails"),
            PlanItem(position=2, action="focus_block", title="Focus"),
        ]
        note = HeuristicModel().generate_note(_report(), _plan(items), [])
        self.assertIn("delegating", note)
        self.assertIn("recovery/focus", note)

    def test_note_low_load_is_manageable(self) -> None:
        note = HeuristicModel().generate_note(_report(10.0, "low"), _plan(level="low"), [])
        self.assertIn("manageable", note)

    def test_note_high_load_suggests_batching(self) -> None:
        note = HeuristicModel().generate_note(_report(90.0, "overload"), _plan(), [])
        self.assertIn("batching", note)

    def test_base_propose_plan_defaults_to_none(self) -> None:
        # HeuristicModel does not override propose_plan; the base class returns None.
        self.assertIsNone(HeuristicModel().propose_plan(FeatureSet(), _report(), []))


class TestRemoteModel(unittest.TestCase):
    def test_generate_note_uses_model_response(self) -> None:
        note = _FakeRemote(response="Hello there").generate_note(_report(), _plan(), [])
        self.assertEqual(note, "Hello there")

    def test_generate_note_falls_back_on_failure(self) -> None:
        note = _FakeRemote(fail=True).generate_note(_report(), _plan(), [])
        self.assertIn("Cognitive Load Score", note)

    def test_propose_plan_returns_response(self) -> None:
        raw = _FakeRemote(response='{"a": 1}').propose_plan(FeatureSet(), _report(), [])
        self.assertEqual(raw, '{"a": 1}')

    def test_propose_plan_falls_back_on_failure(self) -> None:
        self.assertIsNone(_FakeRemote(fail=True).propose_plan(FeatureSet(), _report(), []))

    def test_guard_text_parses_response(self) -> None:
        out = _FakeRemote(response='{"safe": true, "issues": []}').guard_text("note")
        self.assertEqual(out, {"safe": True, "issues": []})

    def test_guard_text_falls_back_on_failure(self) -> None:
        self.assertIsNone(_FakeRemote(fail=True).guard_text("note"))


class TestParseGuardJson(unittest.TestCase):
    def test_no_braces(self) -> None:
        self.assertIsNone(_parse_guard_json("no json"))

    def test_invalid_json(self) -> None:
        self.assertIsNone(_parse_guard_json('{"safe": }'))

    def test_valid(self) -> None:
        self.assertEqual(
            _parse_guard_json('{"safe": true, "issues": ["a", 2]}'),
            {"safe": True, "issues": ["a", "2"]},
        )

    def test_missing_safe_key(self) -> None:
        self.assertIsNone(_parse_guard_json('{"issues": []}'))

    def test_issues_not_a_list(self) -> None:
        self.assertEqual(
            _parse_guard_json('{"safe": false, "issues": "oops"}'),
            {"safe": False, "issues": []},
        )


class TestOllamaModel(unittest.TestCase):
    @mock.patch("urllib.request.urlopen")
    def test_invoke(self, urlopen) -> None:
        cm = mock.MagicMock()
        cm.__enter__.return_value.read.return_value = b'{"response": "hi"}'
        urlopen.return_value = cm
        model = OllamaModel(model_id="x", url="http://localhost:11434/")
        self.assertEqual(model._invoke("prompt"), "hi")

    def test_url_trailing_slash_trimmed(self) -> None:
        model = OllamaModel(model_id="x", url="http://localhost:11434/")
        self.assertEqual(model._url, "http://localhost:11434")


class TestWatsonxModel(unittest.TestCase):
    def test_invoke(self) -> None:
        fake_ibm = types.ModuleType("langchain_ibm")
        fake_chat = mock.MagicMock()
        fake_chat.invoke.return_value = "result"
        fake_ibm.ChatWatsonx = mock.MagicMock(return_value=fake_chat)
        with mock.patch.dict(sys.modules, {"langchain_ibm": fake_ibm}):
            model = WatsonxModel(api_key="k", project_id="p", url="u", model_id="m")
            self.assertEqual(model._invoke("prompt"), "result")


if __name__ == "__main__":
    unittest.main()
