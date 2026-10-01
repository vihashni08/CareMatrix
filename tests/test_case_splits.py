"""Tests for case-level data partitioning and zero-leakage guarantee.

Verifies:
  - case_splits.json exists and adheres to schema.
  - Zero data leakage between Train, Validation, and Test sets (pairwise disjoint).
  - High-risk outcome definition is documented: (death_inhosp == 1) | (icu_days > 0).
  - Train/Val/Test proportions match design (70% / 15% / 15%).
"""

from __future__ import annotations

import json
from pathlib import Path
import unittest


class TestCaseSplitsRigor(unittest.TestCase):
    """Unit tests for dataset split integrity and leakage prevention."""

    def setUp(self):
        self.splits_path = Path(__file__).resolve().parent.parent / "data" / "case_splits.json"
        self.assertTrue(self.splits_path.exists(), f"Missing split file: {self.splits_path}")
        with open(self.splits_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def test_split_sections_exist(self):
        """All three partition sets must be present in splits file."""
        self.assertIn("train_cases", self.data)
        self.assertIn("validation_cases", self.data)
        self.assertIn("test_cases", self.data)
        self.assertIn("metadata", self.data)

    def test_zero_leakage_between_splits(self):
        """Crucial ML invariant: no case ID may appear in more than one partition."""
        train_set = set(self.data["train_cases"])
        val_set = set(self.data["validation_cases"])
        test_set = set(self.data["test_cases"])

        # Check disjointness
        train_val_overlap = train_set & val_set
        train_test_overlap = train_set & test_set
        val_test_overlap = val_set & test_set

        self.assertEqual(len(train_val_overlap), 0, f"Train-Val leakage detected: {train_val_overlap}")
        self.assertEqual(len(train_test_overlap), 0, f"Train-Test leakage detected: {train_test_overlap}")
        self.assertEqual(len(val_test_overlap), 0, f"Val-Test leakage detected: {val_test_overlap}")

    def test_partition_sizes_and_proportions(self):
        """Case splits must have non-empty sets matching 70/15/15 ratio."""
        n_train = len(self.data["train_cases"])
        n_val = len(self.data["validation_cases"])
        n_test = len(self.data["test_cases"])
        n_total = n_train + n_val + n_test

        self.assertGreaterEqual(n_train, 50)
        self.assertGreaterEqual(n_val, 15)
        self.assertGreaterEqual(n_test, 15)

        # Proportions: ~70% train, ~15% val, ~15% test
        self.assertAlmostEqual(n_train / n_total, 0.70, delta=0.05)
        self.assertAlmostEqual(n_val / n_total, 0.15, delta=0.05)
        self.assertAlmostEqual(n_test / n_total, 0.15, delta=0.05)

    def test_metadata_outcome_definition(self):
        """Metadata must document the clinical outcome definition."""
        meta = self.data.get("metadata", {})
        outcome_def = meta.get("outcome_definition", "")
        self.assertIn("death_inhosp", outcome_def)
        self.assertIn("icu_days", outcome_def)


if __name__ == "__main__":
    unittest.main()
