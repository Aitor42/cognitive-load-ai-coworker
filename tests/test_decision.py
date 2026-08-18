"""Unit tests for the Granite Decision Agent gate and plan merge."""

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
from loadguard.llm import HeuristicModel  # noqa: E402
from loadguard.models import LoadReport, Task  # noqa: E402
from loadguard.recommender import build_plan  # noqa: E402


def _report() -> LoadReport:
    return LoadReport(score=80.0, level="overload")


def _tasks() -> list[Task]:
    return [
        Task(id="a", title="Critical fix", priority=5),
        Task(id="b", title="Roadmap slides", priority=4),
        Task(id="c", title="Vendor emails", priority=2),
        Task(id="d", title="Expenses", priority=1),
    ]


class TestParseProposal(unittest.TestCase):
    def test_parse_valid_json(self):
        p = parse_proposal(
            '{"priority_task_id": "a", "delegate_task_ids": ["c"], '
            '"inserts": [{"action": "break", "after_task_id": "b"}], '
            '"rationale": "Protect deep work."}'
        )
        self.assertIsNotNone(p)
        self.assertEqual(p.priority_task_id, "a")
        self.assertEqual(p.delegate_task_ids, ["c"])
        self.assertEqual(p.inserts[0].action, "break")

    def test_parse_garbage_returns_none(self):
        self.assertIsNone(parse_proposal("no json here"))
        self.assertIsNone(parse_proposal(""))
        self.assertIsNone(parse_proposal('{"broken": '))


class TestValidateProposal(unittest.TestCase):
    def setUp(self):
        self.tasks = _tasks()

    def test_valid_proposal(self):
        p = DecisionProposal(
            priority_task_id="a",
            delegate_task_ids=["c", "d"],
            inserts=[InsertAction(action="break", after_task_id="b")],
            rationale="Short rationale.",
        )
        v = validate_proposal(p, self.tasks, _report())
        self.assertTrue(v.valid, v.reasons)

    def test_critical_task_cannot_be_delegated(self):
        p = DecisionProposal(delegate_task_ids=["b"])  # priority 4 is critical
        v = validate_proposal(p, self.tasks, _report())
        self.assertFalse(v.valid)
        self.assertTrue(any("critical" in r for r in v.reasons))

    def test_unknown_task_rejected(self):
        p = DecisionProposal(priority_task_id="ghost")
        v = validate_proposal(p, self.tasks, _report())
        self.assertFalse(v.valid)

    def test_too_many_inserts_rejected(self):
        p = DecisionProposal(inserts=[InsertAction("break")] * 3)
        v = validate_proposal(p, self.tasks, _report())
        self.assertFalse(v.valid)

    def test_duplicate_delegates_rejected(self):
        p = DecisionProposal(delegate_task_ids=["c", "c"])
        v = validate_proposal(p, self.tasks, _report())
        self.assertFalse(v.valid)

    def test_priority_task_also_delegated_rejected(self):
        p = DecisionProposal(priority_task_id="c", delegate_task_ids=["c"])
        v = validate_proposal(p, self.tasks, _report())
        self.assertFalse(v.valid)


    def test_priority_3_task_can_be_delegated(self):
        tasks = [Task(id="p3", title="Medium task", priority=3)]
        p = DecisionProposal(delegate_task_ids=["p3"])
        v = validate_proposal(p, tasks, _report())
        self.assertTrue(v.valid, v.reasons)


class TestMergeProposal(unittest.TestCase):
    def test_proposal_applies(self):
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
        self.assertEqual(plan.items[0].action, "focus_block")  # focus block at start
        do_items = [i for i in plan.items if i.action == "do"]
        self.assertEqual(do_items[0].task_id, "a")  # prioritized first
        self.assertTrue(any(i.action == "delegate" and i.task_id == "c" for i in plan.items))

    def test_prioritize_previously_delegated_task(self):
        # In overload, priority 2 task 'c' is delegated by default.
        tasks = _tasks()
        base = build_plan(tasks, _report())
        self.assertTrue(any(i.task_id == "c" and i.action == "delegate" for i in base.items))
        p = DecisionProposal(priority_task_id="c")
        plan, used = merge_proposal(tasks, _report(), base, p)
        self.assertTrue(used)
        do_items = [i for i in plan.items if i.action == "do"]
        self.assertEqual(do_items[0].task_id, "c")

    def test_no_proposal_keeps_base(self):
        tasks = _tasks()
        base = build_plan(tasks, _report())
        plan, used = merge_proposal(tasks, _report(), base, None)
        self.assertFalse(used)
        self.assertEqual(plan, base)

    def test_positions_renumbered(self):
        tasks = _tasks()
        base = build_plan(tasks, _report())
        p = DecisionProposal(inserts=[InsertAction(action="break", after_task_id="a")])
        plan, _ = merge_proposal(tasks, _report(), base, p)
        self.assertEqual([i.position for i in plan.items], list(range(1, len(plan.items) + 1)))

    def test_heuristic_model_produces_no_proposal(self):
        agent = GraniteDecisionAgent(HeuristicModel())
        self.assertIsNone(agent.run(None, _report(), _tasks()))


if __name__ == "__main__":
    unittest.main()
