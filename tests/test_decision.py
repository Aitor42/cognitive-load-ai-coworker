"""Unit tests for the Granite Decision Agent gate and plan merge (``decision.py``)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loadguard.decision import (  # noqa: E402
    DecisionProposal,
    GraniteDecisionAgent,
    InsertAction,
    merge_proposal,
    parse_proposal,
    validate_proposal,
)
from loadguard.llm import ChatModel, HeuristicModel  # noqa: E402
from loadguard.models import DONE, LoadReport, Plan, PlanItem, Task, Worker  # noqa: E402
from loadguard.recommender import build_plan  # noqa: E402


class _FakeModel(ChatModel):
    """A model that returns a configurable proposal string."""

    name = "fake"

    def __init__(self, proposal: str | None = None) -> None:
        self._proposal = proposal

    def generate_note(self, load_report, plan, tasks) -> str:
        return "A safe note."

    def propose_plan(self, features, load_report, tasks) -> str | None:
        return self._proposal


def _report() -> LoadReport:
    return LoadReport(score=80.0, level="overload")


def _low_report() -> LoadReport:
    return LoadReport(score=10.0, level="low")


def _tasks() -> list[Task]:
    return [
        Task(id="a", title="Critical fix", priority=5),
        Task(id="b", title="Roadmap slides", priority=4),
        Task(id="c", title="Vendor emails", priority=2),
        Task(id="d", title="Expenses", priority=1),
    ]


VALID_PROPOSAL = (
    '{"priority_task_id": "a", "delegate_task_ids": ["c"], '
    '"inserts": [{"action": "break", "after_task_id": "b"}], "rationale": "ok"}'
)


class TestParseProposal(unittest.TestCase):
    def test_parse_valid_json(self) -> None:
        p = parse_proposal(
            '{"priority_task_id": "a", "delegate_task_ids": ["c"], '
            '"inserts": [{"action": "break", "after_task_id": "b"}], '
            '"rationale": "Protect deep work."}'
        )
        self.assertIsNotNone(p)
        self.assertEqual(p.priority_task_id, "a")
        self.assertEqual(p.delegate_task_ids, ["c"])
        self.assertEqual(p.inserts[0].action, "break")

    def test_parse_garbage_returns_none(self) -> None:
        self.assertIsNone(parse_proposal("no json here"))
        self.assertIsNone(parse_proposal(""))
        self.assertIsNone(parse_proposal('{"broken": '))

    def test_parse_invalid_json_raises_gracefully(self) -> None:
        self.assertIsNone(parse_proposal('{"a": }'))

    def test_parse_non_dict_json_returns_none(self) -> None:
        self.assertIsNone(parse_proposal("[1, 2, 3]"))

    def test_parse_non_list_fields_ignored(self) -> None:
        p = parse_proposal('{"delegate_task_ids": "c", "inserts": {"action": "break"}}')
        self.assertIsNotNone(p)
        self.assertEqual(p.delegate_task_ids, [])
        self.assertEqual(p.inserts, [])

    def test_parse_skips_non_dict_inserts(self) -> None:
        p = parse_proposal('{"inserts": ["nope", {"action": "break"}]}')
        self.assertIsNotNone(p)
        self.assertEqual(len(p.inserts), 1)
        self.assertEqual(p.inserts[0].action, "break")

    def test_parse_non_string_priority_ignored(self) -> None:
        p = parse_proposal('{"priority_task_id": 42}')
        self.assertIsNotNone(p)
        self.assertIsNone(p.priority_task_id)

    def test_parse_filters_non_string_delegates(self) -> None:
        p = parse_proposal('{"delegate_task_ids": ["a", 2, null]}')
        self.assertIsNotNone(p)
        self.assertEqual(p.delegate_task_ids, ["a"])


class TestValidateProposal(unittest.TestCase):
    def setUp(self) -> None:
        self.tasks = _tasks()

    def test_valid_proposal(self) -> None:
        p = DecisionProposal(
            priority_task_id="a",
            delegate_task_ids=["c", "d"],
            inserts=[InsertAction(action="break", after_task_id="b")],
            rationale="Short rationale.",
        )
        v = validate_proposal(p, self.tasks, _report())
        self.assertTrue(v.valid, v.reasons)

    def test_critical_task_cannot_be_delegated(self) -> None:
        p = DecisionProposal(delegate_task_ids=["b"])  # priority 4 is critical
        v = validate_proposal(p, self.tasks, _report())
        self.assertFalse(v.valid)
        self.assertTrue(any("critical" in r for r in v.reasons))

    def test_unknown_task_rejected(self) -> None:
        p = DecisionProposal(priority_task_id="ghost")
        v = validate_proposal(p, self.tasks, _report())
        self.assertFalse(v.valid)

    def test_too_many_inserts_rejected(self) -> None:
        p = DecisionProposal(inserts=[InsertAction("break")] * 3)
        v = validate_proposal(p, self.tasks, _report())
        self.assertFalse(v.valid)

    def test_duplicate_delegates_rejected(self) -> None:
        p = DecisionProposal(delegate_task_ids=["c", "c"])
        v = validate_proposal(p, self.tasks, _report())
        self.assertFalse(v.valid)

    def test_priority_task_also_delegated_rejected(self) -> None:
        p = DecisionProposal(priority_task_id="c", delegate_task_ids=["c"])
        v = validate_proposal(p, self.tasks, _report())
        self.assertFalse(v.valid)

    def test_priority_3_task_can_be_delegated(self) -> None:
        tasks = [Task(id="p3", title="Medium task", priority=3)]
        p = DecisionProposal(delegate_task_ids=["p3"])
        v = validate_proposal(p, tasks, _report())
        self.assertTrue(v.valid, v.reasons)

    def test_priority_task_not_todo_rejected(self) -> None:
        tasks = self.tasks + [Task(id="x", title="Done", priority=3, status=DONE)]
        p = DecisionProposal(priority_task_id="x")
        v = validate_proposal(p, tasks, _report())
        self.assertFalse(v.valid)

    def test_unknown_delegate_rejected(self) -> None:
        p = DecisionProposal(delegate_task_ids=["ghost"])
        v = validate_proposal(p, self.tasks, _report())
        self.assertFalse(v.valid)
        self.assertTrue(any("does not exist" in r for r in v.reasons))

    def test_delegate_not_todo_rejected(self) -> None:
        tasks = self.tasks + [Task(id="x", title="Done", priority=1, status=DONE)]
        p = DecisionProposal(delegate_task_ids=["x"])
        v = validate_proposal(p, tasks, _report())
        self.assertFalse(v.valid)
        self.assertTrue(any("not a todo" in r for r in v.reasons))

    def test_invalid_insert_action_rejected(self) -> None:
        p = DecisionProposal(inserts=[InsertAction(action="focus")])
        v = validate_proposal(p, self.tasks, _report())
        self.assertFalse(v.valid)
        self.assertTrue(any("invalid insert action" in r for r in v.reasons))

    def test_insert_unknown_after_task_rejected(self) -> None:
        p = DecisionProposal(inserts=[InsertAction(action="break", after_task_id="ghost")])
        v = validate_proposal(p, self.tasks, _report())
        self.assertFalse(v.valid)

    def test_rationale_too_long_rejected(self) -> None:
        p = DecisionProposal(rationale="x" * 601)
        v = validate_proposal(p, self.tasks, _report())
        self.assertFalse(v.valid)
        self.assertTrue(any("rationale too long" in r for r in v.reasons))


class TestGraniteDecisionAgent(unittest.TestCase):
    def test_heuristic_model_produces_no_proposal(self) -> None:
        agent = GraniteDecisionAgent(HeuristicModel())
        self.assertIsNone(agent.run(None, _report(), _tasks()))

    def test_model_without_proposal_returns_none(self) -> None:
        agent = GraniteDecisionAgent(_FakeModel(proposal=None))
        self.assertIsNone(agent.run(None, _report(), _tasks()))

    def test_model_with_garbage_returns_none(self) -> None:
        agent = GraniteDecisionAgent(_FakeModel(proposal="no json here"))
        self.assertIsNone(agent.run(None, _report(), _tasks()))

    def test_model_with_invalid_proposal_returns_none(self) -> None:
        agent = GraniteDecisionAgent(_FakeModel(proposal='{"priority_task_id": "ghost"}'))
        self.assertIsNone(agent.run(None, _report(), _tasks()))

    def test_model_with_valid_proposal_is_returned(self) -> None:
        agent = GraniteDecisionAgent(_FakeModel(proposal=VALID_PROPOSAL))
        p = agent.run(None, _report(), _tasks())
        self.assertIsNotNone(p)
        self.assertEqual(p.priority_task_id, "a")


class TestMergeProposal(unittest.TestCase):
    def test_proposal_applies(self) -> None:
        tasks = _tasks()
        base = build_plan(tasks, _report())
        p = DecisionProposal(
            priority_task_id="a",
            delegate_task_ids=["c"],
            inserts=[InsertAction(action="focus_block")],
            rationale="Granite rationale",
        )
        plan, used = merge_proposal(tasks, _report(), base, p)
        self.assertTrue(used)
        self.assertEqual(plan.items[0].action, "focus_block")
        do_items = [i for i in plan.items if i.action == "do"]
        self.assertEqual(do_items[0].task_id, "a")
        self.assertTrue(any(i.action == "delegate" and i.task_id == "c" for i in plan.items))

    def test_prioritize_previously_delegated_task(self) -> None:
        tasks = _tasks()
        base = build_plan(tasks, _report())
        self.assertTrue(any(i.task_id == "c" and i.action == "delegate" for i in base.items))
        p = DecisionProposal(priority_task_id="c")
        plan, used = merge_proposal(tasks, _report(), base, p)
        self.assertTrue(used)
        do_items = [i for i in plan.items if i.action == "do"]
        self.assertEqual(do_items[0].task_id, "c")

    def test_no_proposal_keeps_base(self) -> None:
        tasks = _tasks()
        base = build_plan(tasks, _report())
        plan, used = merge_proposal(tasks, _report(), base, None)
        self.assertFalse(used)
        self.assertEqual(plan, base)

    def test_positions_renumbered(self) -> None:
        tasks = _tasks()
        base = build_plan(tasks, _report())
        p = DecisionProposal(inserts=[InsertAction(action="break", after_task_id="a")])
        plan, _ = merge_proposal(tasks, _report(), base, p)
        self.assertEqual([i.position for i in plan.items], list(range(1, len(plan.items) + 1)))

    def test_priority_task_not_found_is_ignored(self) -> None:
        base = build_plan(_tasks(), _report())
        plan, used = merge_proposal(
            _tasks(), _report(), base, DecisionProposal(priority_task_id="ghost")
        )
        self.assertFalse(used)

    def test_focus_block_deduplicated(self) -> None:
        base = build_plan(_tasks(), _low_report())
        p = DecisionProposal(inserts=[InsertAction("focus_block"), InsertAction("focus_block")])
        plan, used = merge_proposal(_tasks(), _report(), base, p)
        self.assertTrue(used)
        self.assertEqual(sum(1 for i in plan.items if i.action == "focus_block"), 1)

    def test_focus_block_anchored_after_task(self) -> None:
        base = build_plan(_tasks(), _low_report())
        p = DecisionProposal(inserts=[InsertAction("focus_block", after_task_id="b")])
        plan, used = merge_proposal(_tasks(), _report(), base, p)
        self.assertTrue(used)
        idx = next(i for i, it in enumerate(plan.items) if it.action == "focus_block")
        self.assertEqual(plan.items[idx - 1].task_id, "b")

    def test_break_without_anchor_appended(self) -> None:
        base = build_plan(_tasks(), _low_report())
        p = DecisionProposal(inserts=[InsertAction("break")])
        plan, used = merge_proposal(_tasks(), _report(), base, p)
        self.assertTrue(used)
        self.assertEqual(plan.items[-1].action, "break")

    def test_break_capped_at_three(self) -> None:
        base = Plan(
            load_report=_report(),
            items=[PlanItem(position=i, action="break", title="b") for i in range(1, 4)],
        )
        plan, used = merge_proposal(
            _tasks(), _report(), base, DecisionProposal(inserts=[InsertAction("break")])
        )
        self.assertFalse(used)
        self.assertEqual(sum(1 for i in plan.items if i.action == "break"), 3)

    def test_delegate_task_already_delegated_is_skipped(self) -> None:
        base = build_plan(_tasks(), _report())  # 'c' is already delegated
        plan, used = merge_proposal(
            _tasks(), _report(), base, DecisionProposal(delegate_task_ids=["c"])
        )
        self.assertFalse(used)

    def test_delegate_do_task_with_rationale(self) -> None:
        base = build_plan(_tasks(), _low_report())  # every task is "do" here
        p = DecisionProposal(delegate_task_ids=["d"], rationale="Delegate it.")
        plan, used = merge_proposal(_tasks(), _report(), base, p)
        self.assertTrue(used)
        item = next(i for i in plan.items if i.task_id == "d")
        self.assertEqual(item.action, "delegate")
        self.assertEqual(item.rationale, "Delegate it.")

    def test_delegate_do_task_without_rationale(self) -> None:
        base = build_plan(_tasks(), _low_report())
        original = next(i.rationale for i in base.items if i.task_id == "d")
        p = DecisionProposal(delegate_task_ids=["d"])
        plan, used = merge_proposal(_tasks(), _report(), base, p)
        self.assertTrue(used)
        item = next(i for i in plan.items if i.task_id == "d")
        self.assertEqual(item.action, "delegate")
        # Without an explicit rationale, the deterministic rationale is kept.
        self.assertEqual(item.rationale, original)

    def test_unknown_insert_action_is_ignored(self) -> None:
        base = build_plan(_tasks(), _low_report())
        p = DecisionProposal(inserts=[InsertAction(action="weird")])
        plan, used = merge_proposal(_tasks(), _report(), base, p)
        self.assertFalse(used)

    def test_delegate_populates_suggested_assignees_from_workers(self) -> None:
        base = build_plan(_tasks(), _low_report())
        workers = [Worker(id="w1", name="Ada"), Worker(id="w2", name="Bob")]
        p = DecisionProposal(delegate_task_ids=["d"])
        plan, used = merge_proposal(_tasks(), _report(), base, p, workers=workers, now=100.0)
        self.assertTrue(used)
        item = next(i for i in plan.items if i.task_id == "d")
        self.assertEqual(item.action, "delegate")
        self.assertEqual(item.suggested_assignees, ["Ada", "Bob"])


if __name__ == "__main__":
    unittest.main()
