"""Parametrized unit tests for the Cognitive Load Score engine (``scoring.py``).

The score is the core business deliverable: a weighted, normalized 0-100 value
computed from five behavioral proxies.  Because it is pure logic with no I/O or
external dependencies, it is tested here with full white-box coverage using two
complementary strategies.

Branch coverage
---------------
Every decision outcome in the module is exercised:

* ``_normalize`` — the ``midpoint <= 0`` and ``value <= 0`` guards (both
  outcomes) and the Hill function for positive inputs.
* ``_level`` — all four bands and their exact boundaries (25 / 50 / 75) so that
  every ``score < upper`` outcome is taken.
* ``_contributions`` — the focus inversion (``1 - focus``) and the per-factor
  clamp into ``[0, 1]``.
* ``_interaction_bonus`` — non-zero only when meetings AND interruptions are
  both elevated.

Pairwise (All-Pairs) testing
----------------------------
``score`` is a weighted sum of five independent factors plus an interaction
term, so the full cross-product is redundant.  An All-Pairs covering array is
built greedily from ``SCORE_PARAMETER_SPACE`` so that every *pair* of factor
levels is exercised at least once, then verified for completeness before scoring.
``PAIRWISE_CASES`` additionally pins a handful of exact score/level anchors at
the boundary values.
"""

from __future__ import annotations

import itertools
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loadguard.models import FeatureSet  # noqa: E402
from loadguard.scoring import (  # noqa: E402
    _contributions,
    _explanation,
    _interaction_bonus,
    _level,
    _normalize,
    score,
)

# ---------------------------------------------------------------------------
# Parametrization tables
# ---------------------------------------------------------------------------

# (description, value, midpoint, expected) — branch coverage of _normalize.
# The Hill function returns value / (value + midpoint) for positive inputs.
NORMALIZE_CASES = [
    ("zero midpoint short-circuits", 5.0, 0.0, 0.0),
    ("negative midpoint short-circuits", 5.0, -1.0, 0.0),
    ("negative value returns zero", -3.0, 10.0, 0.0),
    ("value equals midpoint gives half", 6.0, 6.0, 0.5),
    ("double midpoint gives two-thirds", 12.0, 6.0, 2 / 3),
    ("large value approaches one", 99.0, 1.0, 0.99),
]

# (score, expected_level) — every band plus the exact boundary values.
LEVEL_CASES = [
    (0.0, "low"),
    (24.9, "low"),
    (25.0, "moderate"),
    (49.9, "moderate"),
    (50.0, "high"),
    (74.9, "high"),
    (75.0, "overload"),
    (100.0, "overload"),
]

# (description, features, expected_level, expected_score)
# Expected scores are derived from the documented formula:
# base = 0.30*cs_norm + 0.20*meeting + 0.20*notif_norm + 0.15*(1-focus)
#        + 0.15*multitask
# interaction = meeting * max(cs_norm, notif_norm) * 0.10
# total = min(base + interaction, 1.0) * 100
PAIRWISE_CASES = [
    ("all calm", FeatureSet(focus_ratio=1.0), "low", 0.0),
    ("no focus logged", FeatureSet(), "low", 15.0),
    (
        "context switches at sigmoid midpoint x2",
        FeatureSet(context_switches_per_hour=12.0, focus_ratio=1.0),
        "low",
        20.0,
    ),
    (
        "notifications at high rate",
        FeatureSet(notification_rate=30.0, focus_ratio=1.0),
        "low",
        15.0,
    ),
    ("meetings saturated", FeatureSet(meeting_ratio=1.0, focus_ratio=1.0), "low", 20.0),
    ("multitasking saturated", FeatureSet(multitasking_index=1.0, focus_ratio=1.0), "low", 15.0),
    (
        "switches + notifications both high",
        FeatureSet(context_switches_per_hour=12.0, notification_rate=30.0, focus_ratio=1.0),
        "moderate",
        35.0,
    ),
    (
        "switches + meetings + no focus (interaction kicks in)",
        FeatureSet(context_switches_per_hour=12.0, meeting_ratio=1.0, focus_ratio=0.0),
        "high",
        61.7,
    ),
    (
        "everything maxed",
        FeatureSet(
            context_switches_per_hour=100.0,
            meeting_ratio=1.0,
            notification_rate=100.0,
            focus_ratio=0.0,
            multitasking_index=1.0,
        ),
        "overload",
        100.0,
    ),
    (
        "high interruptions + half focus/multitasking, no meetings",
        FeatureSet(
            context_switches_per_hour=24.0,
            notification_rate=30.0,
            focus_ratio=0.5,
            multitasking_index=0.5,
        ),
        "high",
        54.0,
    ),
    (
        "midpoint across all factors",
        FeatureSet(
            context_switches_per_hour=6.0,
            meeting_ratio=0.5,
            notification_rate=15.0,
            focus_ratio=0.5,
            multitasking_index=0.5,
        ),
        "high",
        55.0,
    ),
    (
        "switches beyond midpoint (sigmoid gives diminishing returns)",
        FeatureSet(context_switches_per_hour=24.0, focus_ratio=1.0),
        "low",
        24.0,
    ),
    (
        "notifications beyond midpoint (sigmoid gives diminishing returns)",
        FeatureSet(notification_rate=60.0, focus_ratio=1.0),
        "low",
        17.1,
    ),
]

# All-Pairs parameter space: factor -> boundary levels (calm / mid / high /
# extreme). The full cross-product is 4x3x4x3x3 = 432 combinations; the
# All-Pairs covering array below exercises every factor *pair* in far fewer rows.
SCORE_PARAMETER_SPACE: dict[str, list[float]] = {
    "context_switches_per_hour": [0.0, 6.0, 12.0, 24.0],
    "meeting_ratio": [0.0, 0.5, 1.0],
    "notification_rate": [0.0, 15.0, 30.0, 60.0],
    "focus_ratio": [0.0, 0.5, 1.0],
    "multitasking_index": [0.0, 0.5, 1.0],
}


def _all_pairs(parameter_space: dict[str, list[float]]) -> list[dict[str, float]]:
    """Greedy All-Pairs covering array.

    Every unordered pair of factor values ``(factor_i, value_i, factor_j,
    value_j)`` is covered by at least one returned row. The construction is
    deterministic because ``itertools.product`` yields combinations in a fixed
    order, and each round keeps the row covering the most still-uncovered pairs.
    """
    factors = list(parameter_space)
    all_pairs = {
        (factors[i], vi, factors[j], vj)
        for i in range(len(factors))
        for j in range(i + 1, len(factors))
        for vi in parameter_space[factors[i]]
        for vj in parameter_space[factors[j]]
    }

    def covered(row: dict[str, float]) -> set:
        return {
            (factors[a], row[factors[a]], factors[b], row[factors[b]])
            for a in range(len(factors))
            for b in range(a + 1, len(factors))
        }

    uncovered = set(all_pairs)
    rows: list[dict[str, float]] = []
    while uncovered:
        best_row: dict[str, float] | None = None
        best_hit: set = set()
        for combo in itertools.product(*(parameter_space[f] for f in factors)):
            row = dict(zip(factors, combo))
            hit = covered(row) & uncovered
            if len(hit) > len(best_hit):
                best_hit = hit
                best_row = row
        if best_row is None or not best_hit:
            raise AssertionError("All-Pairs construction stalled; pair space not coverable")
        rows.append(best_row)
        uncovered -= best_hit
    return rows


def _uncovered_pairs(parameter_space: dict[str, list[float]], rows: list[dict[str, float]]) -> set:
    """Pairs not covered by ``rows`` (empty set means All-Pairs is satisfied)."""
    factors = list(parameter_space)
    all_pairs = {
        (factors[i], vi, factors[j], vj)
        for i in range(len(factors))
        for j in range(i + 1, len(factors))
        for vi in parameter_space[factors[i]]
        for vj in parameter_space[factors[j]]
    }
    covered = {
        (factors[a], row[factors[a]], factors[b], row[factors[b]])
        for row in rows
        for a in range(len(factors))
        for b in range(a + 1, len(factors))
    }
    return all_pairs - covered


class TestNormalizeBranchCoverage(unittest.TestCase):
    """Branch coverage of ``_normalize``: guard and Hill function regions."""

    def test_guard_and_hill_function(self) -> None:
        for desc, value, midpoint, expected in NORMALIZE_CASES:
            with self.subTest(case=desc):
                self.assertAlmostEqual(_normalize(value, midpoint), expected, places=6)


class TestLevelBranchCoverage(unittest.TestCase):
    """Branch coverage of ``_level``: every band and its exact boundaries."""

    def test_all_bands_and_boundaries(self) -> None:
        for value, expected in LEVEL_CASES:
            with self.subTest(score=value):
                self.assertEqual(_level(value), expected)

    def test_defensive_fallback_for_non_finite_scores(self) -> None:
        # inf/NaN are never < any boundary, so the loop falls through to the
        # defensive `return OVERLOAD` at the end of ``_level``.
        self.assertEqual(_level(float("inf")), "overload")
        self.assertEqual(_level(float("nan")), "overload")


class TestContributions(unittest.TestCase):
    """White-box checks of the per-factor normalization and focus inversion."""

    def test_focus_is_inverted(self) -> None:
        # No focus logged -> maximum load contribution; full focus -> none.
        self.assertEqual(_contributions({"focus_ratio": 0.0})["focus_ratio"], 1.0)
        self.assertEqual(_contributions({"focus_ratio": 1.0})["focus_ratio"], 0.0)

    def test_ratios_clamp_to_unit_range(self) -> None:
        cases = [
            ("meeting_ratio", {"meeting_ratio": 1.5}, 1.0),
            ("meeting_ratio", {"meeting_ratio": -0.5}, 0.0),
            ("multitasking_index", {"multitasking_index": 2.0}, 1.0),
            ("focus_ratio", {"focus_ratio": 1.5}, 0.0),  # 1 - 1.5 -> clamped to 0
        ]
        for name, factors, expected in cases:
            with self.subTest(case=name):
                self.assertEqual(_contributions(factors)[name], expected)


class TestInteractionBonus(unittest.TestCase):
    """White-box checks of the interaction bonus."""

    def test_zero_meetings_gives_zero_interaction(self) -> None:
        self.assertEqual(
            _interaction_bonus({"meeting_ratio": 0.0, "context_switches_per_hour": 0.8}), 0.0
        )

    def test_zero_interruptions_gives_zero_interaction(self) -> None:
        self.assertEqual(
            _interaction_bonus(
                {"meeting_ratio": 1.0, "context_switches_per_hour": 0.0, "notification_rate": 0.0}
            ),
            0.0,
        )

    def test_both_elevated_gives_positive_interaction(self) -> None:
        result = _interaction_bonus(
            {"meeting_ratio": 0.5, "context_switches_per_hour": 0.8, "notification_rate": 0.3}
        )
        self.assertAlmostEqual(result, 0.5 * 0.8, places=6)


class TestScorePairwise(unittest.TestCase):
    """Pairwise matrix: score + level for boundary factor combinations."""

    def test_pairwise_matrix(self) -> None:
        for desc, features, expected_level, expected_score in PAIRWISE_CASES:
            with self.subTest(case=desc):
                report = score(features)
                self.assertEqual(report.level, expected_level)
                self.assertAlmostEqual(report.score, expected_score, places=1)


class TestAllPairsScore(unittest.TestCase):
    """All-Pairs coverage of ``score`` across the five-factor parameter space."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = _all_pairs(SCORE_PARAMETER_SPACE)

    def test_array_is_complete(self) -> None:
        """The generated array must cover every factor pair at least once."""
        self.assertGreater(len(self.rows), 0)
        self.assertEqual(_uncovered_pairs(SCORE_PARAMETER_SPACE, self.rows), set())

    def test_score_invariants_over_all_pairs(self) -> None:
        for row in self.rows:
            with self.subTest(**row):
                report = score(FeatureSet(**row))
                self.assertGreaterEqual(report.score, 0.0)
                self.assertLessEqual(report.score, 100.0)
                self.assertEqual(report.level, _level(report.score))


class TestMonotonicity(unittest.TestCase):
    """The score must move monotonically with each individual factor."""

    def test_load_factors_are_monotonic_non_decreasing(self) -> None:
        load_levels = {
            "context_switches_per_hour": [0.0, 6.0, 12.0, 24.0],
            "meeting_ratio": [0.0, 0.5, 1.0],
            "notification_rate": [0.0, 15.0, 30.0, 60.0],
            "multitasking_index": [0.0, 0.5, 1.0],
        }
        for factor, levels in load_levels.items():
            previous = score(FeatureSet(**{factor: levels[0]})).score
            for level in levels[1:]:
                with self.subTest(factor=factor, level=level):
                    current = score(FeatureSet(**{factor: level})).score
                    self.assertGreaterEqual(current, previous)
                    previous = current

    def test_focus_is_monotonic_non_increasing(self) -> None:
        previous = score(FeatureSet(focus_ratio=0.0)).score
        for level in (0.5, 1.0):
            with self.subTest(focus_ratio=level):
                current = score(FeatureSet(focus_ratio=level)).score
                self.assertLessEqual(current, previous)
                previous = current


class TestScoreContract(unittest.TestCase):
    """Invariants that must hold for any input."""

    def test_score_always_within_bounds(self) -> None:
        extremes = [
            FeatureSet(),
            FeatureSet(
                context_switches_per_hour=999.0,
                meeting_ratio=1.0,
                notification_rate=999.0,
                focus_ratio=0.0,
                multitasking_index=1.0,
            ),
            FeatureSet(focus_ratio=1.0),
        ]
        for features in extremes:
            with self.subTest(features=features):
                report = score(features)
                self.assertGreaterEqual(report.score, 0.0)
                self.assertLessEqual(report.score, 100.0)

    def test_level_is_consistent_with_score(self) -> None:
        for desc, features, _level_expected, _score_expected in PAIRWISE_CASES:
            with self.subTest(case=desc):
                report = score(features)
                self.assertEqual(report.level, _level(report.score))

    def test_explanation_is_grounded_in_contributions(self) -> None:
        # Focus time is protective: when it is high it must not be a top driver.
        report = score(
            FeatureSet(
                context_switches_per_hour=6.0,
                meeting_ratio=0.2,
                notification_rate=2.0,
                focus_ratio=0.95,
                multitasking_index=0.1,
            )
        )
        self.assertTrue(report.explanation.startswith("Main drivers: "))
        self.assertIn("context switches per hour", report.explanation)
        self.assertNotIn("focus time", report.explanation)

    def test_explanation_names_two_drivers(self) -> None:
        report = score(
            FeatureSet(context_switches_per_hour=12.0, notification_rate=30.0, focus_ratio=1.0)
        )
        drivers = _explanation(
            {
                "context_switches_per_hour": 12.0,
                "meeting_ratio": 0.0,
                "notification_rate": 30.0,
                "focus_ratio": 1.0,
                "multitasking_index": 0.0,
            }
        )
        self.assertEqual(drivers.count(", "), 1)  # exactly two drivers, comma-separated
        self.assertEqual(report.explanation, drivers)

    def test_interaction_amplifies_combined_stressors(self) -> None:
        """Meeting + interruptions scores higher than the sum of their parts."""
        cs_only = score(FeatureSet(context_switches_per_hour=12.0, focus_ratio=1.0))
        mtg_only = score(FeatureSet(meeting_ratio=1.0, focus_ratio=1.0))
        combined = score(
            FeatureSet(context_switches_per_hour=12.0, meeting_ratio=1.0, focus_ratio=1.0)
        )
        self.assertGreater(combined.score, cs_only.score + mtg_only.score)


class TestRoleProfilesAndCustomWeights(unittest.TestCase):
    """Tests for role-specific profiles and custom scoring weights."""

    def test_researcher_profile_sensitizes_focus_loss(self) -> None:
        """Researcher profile assigns higher weight to focus loss than default."""
        f = FeatureSet(focus_ratio=0.0)  # zero focus
        rep_default = score(f)
        rep_researcher = score(f, role="researcher")
        # In researcher profile, focus loss has weight 0.35 vs 0.15 in default
        self.assertGreater(rep_researcher.score, rep_default.score)

    def test_manager_profile_sensitizes_meetings_and_notifications(self) -> None:
        """Manager profile weighs meetings/notifications higher."""
        f = FeatureSet(meeting_ratio=0.8, notification_rate=20.0, focus_ratio=1.0)
        rep_default = score(f)
        rep_manager = score(f, role="manager")
        self.assertGreater(rep_manager.score, rep_default.score)

    def test_custom_weights_normalized(self) -> None:
        """Custom weights are normalized to sum to 1.0."""
        f = FeatureSet(context_switches_per_hour=12.0, focus_ratio=1.0)
        custom = {"context_switches_per_hour": 100.0}
        rep = score(f, weights=custom)
        # All weight on context switches (midpoint 6 -> cs_norm = 2/3 ≈ 0.667 -> score 66.7)
        self.assertAlmostEqual(rep.score, 66.7, places=1)

    def test_unknown_role_falls_back_to_default(self) -> None:
        f = FeatureSet(context_switches_per_hour=6.0, focus_ratio=0.5)
        rep_default = score(f)
        rep_unknown = score(f, role="astronaut")
        self.assertEqual(rep_default.score, rep_unknown.score)

    def test_zero_sum_custom_weights_falls_back(self) -> None:
        f = FeatureSet(context_switches_per_hour=6.0, focus_ratio=0.5)
        rep_default = score(f)
        rep_zero = score(f, weights={"context_switches_per_hour": 0.0})
        self.assertEqual(rep_default.score, rep_zero.score)


if __name__ == "__main__":
    unittest.main()
