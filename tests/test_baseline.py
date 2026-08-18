"""Unit tests for the personalized baseline and trend module."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loadguard.baseline import (  # noqa: E402
    append_score,
    clear_history,
    compute_baseline,
    load_history,
    trend,
)


class TestBaseline(unittest.TestCase):
    def test_requires_two_samples(self):
        self.assertIsNone(compute_baseline([]))
        self.assertIsNone(compute_baseline([50.0]))

    def test_mean_and_std(self):
        b = compute_baseline([40.0, 60.0])
        self.assertIsNotNone(b)
        self.assertEqual(b.mean, 50.0)
        self.assertEqual(b.std, 10.0)
        self.assertEqual(b.n, 2)

    def test_trend_above_baseline_rising(self):
        b = compute_baseline([40.0, 60.0])  # mean 50, std 10
        t = trend(70.0, b)
        self.assertIsNotNone(t)
        self.assertEqual(t.direction, "rising")
        self.assertEqual(t.deviation_pct, 40.0)  # 20 above 50

    def test_trend_below_baseline_falling(self):
        b = compute_baseline([40.0, 60.0])
        t = trend(30.0, b)
        self.assertEqual(t.direction, "falling")

    def test_trend_stable(self):
        b = compute_baseline([50.0, 50.0, 50.0])
        t = trend(50.0, b)
        self.assertEqual(t.direction, "stable")

    def test_confidence_scales_with_samples(self):
        low = compute_baseline([50.0, 51.0])
        med = compute_baseline([50.0, 51.0, 49.0])
        high = compute_baseline([50.0, 51.0, 49.0, 52.0, 48.0])
        self.assertEqual(trend(60.0, low).confidence, "low")
        self.assertEqual(trend(60.0, med).confidence, "medium")
        self.assertEqual(trend(60.0, high).confidence, "high")

    def test_no_baseline_no_trend(self):
        self.assertIsNone(trend(50.0, None))

    def test_compute_baseline_filters_none_and_nan(self):
        self.assertIsNone(compute_baseline([None, 50.0]))
        b = compute_baseline([40.0, None, float("nan"), 60.0])
        self.assertIsNotNone(b)
        self.assertEqual(b.mean, 50.0)
        self.assertEqual(b.n, 2)

    def test_compute_baseline_ignores_non_numeric_strings(self):
        b = compute_baseline([40.0, "oops", 60.0])
        self.assertIsNotNone(b)
        self.assertEqual(b.mean, 50.0)
        self.assertEqual(b.n, 2)

    def test_trend_zero_std_uses_deviation(self):
        b = compute_baseline([50.0, 50.0, 50.0])
        self.assertEqual(b.std, 0.0)
        self.assertEqual(trend(60.0, b).direction, "rising")
        self.assertEqual(trend(40.0, b).direction, "falling")
        self.assertEqual(trend(50.0, b).direction, "stable")


class TestHistoryPersistence(unittest.TestCase):
    def test_roundtrip_and_clear(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.jsonl"
            append_score(path, 42.0)
            append_score(path, 55.0)
            self.assertEqual(load_history(path), [42.0, 55.0])
            self.assertEqual(clear_history(path), 2)
            self.assertEqual(load_history(path), [])

    def test_missing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing.jsonl"
            self.assertEqual(load_history(path), [])
            self.assertEqual(clear_history(path), 0)

    def test_load_history_ignores_invalid_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.jsonl"
            path.write_text("not_a_number\n# comment\n42\n", encoding="utf-8")
            self.assertEqual(load_history(path), [42.0])


if __name__ == "__main__":
    unittest.main()
