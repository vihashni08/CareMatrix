"""Unit tests for evaluation/metrics.py.

Verifies:
  - Confusion matrix (TP, FP, FN, TN) calculation correctness.
  - Precision, Recall, Specificity, F1, and AUROC calculations.
  - GROUND_TRUTH_UNAVAILABLE path when y_true is None or lacks valid binary labels.
  - ClassificationMetricsResult.to_dict() serialization including specificity.
"""

from __future__ import annotations

import unittest
import numpy as np

from evaluation.metrics import (
    ClassificationMetricsResult,
    GROUND_TRUTH_UNAVAILABLE,
    compute_classification_metrics,
)


class TestEvaluationMetrics(unittest.TestCase):
    """Unit tests for classification metrics calculations and safety checks."""

    def test_ground_truth_unavailable_when_y_true_none(self):
        """When y_true is None, status must be GROUND_TRUTH_UNAVAILABLE."""
        res = compute_classification_metrics(y_true=None, y_pred=[1, 0, 1])
        self.assertEqual(res.status, GROUND_TRUTH_UNAVAILABLE)
        self.assertIsNone(res.accuracy)
        self.assertIn("not available", res.message)

    def test_ground_truth_unavailable_when_empty_or_invalid(self):
        """When y_true contains no valid labels, status must be GROUND_TRUTH_UNAVAILABLE."""
        res = compute_classification_metrics(y_true=[], y_pred=[])
        self.assertEqual(res.status, GROUND_TRUTH_UNAVAILABLE)

        res2 = compute_classification_metrics(y_true=[None, None], y_pred=[1, 0])
        self.assertEqual(res2.status, GROUND_TRUTH_UNAVAILABLE)

    def test_confusion_matrix_and_metrics_calculation(self):
        """Verify TP, FP, FN, TN and derived rates: accuracy, precision, recall, specificity, F1."""
        # Ground truth: 2 positive (indices 0, 1), 2 negative (indices 2, 3)
        # Predictions:  TP at 0, FN at 1, FP at 2, TN at 3
        y_true = [1, 1, 0, 0]
        y_pred = [1, 0, 1, 0]
        y_prob = [0.9, 0.4, 0.7, 0.2]

        res = compute_classification_metrics(y_true=y_true, y_pred=y_pred, y_prob=y_prob)

        self.assertEqual(res.status, "SUCCESS")
        self.assertEqual(res.sample_count, 4)
        cm = res.confusion_matrix
        self.assertEqual(cm["TP"], 1)
        self.assertEqual(cm["FN"], 1)
        self.assertEqual(cm["FP"], 1)
        self.assertEqual(cm["TN"], 1)

        # TP=1, FP=1, FN=1, TN=1
        # Accuracy: (1 + 1) / 4 = 0.5
        self.assertAlmostEqual(res.accuracy, 0.5)
        # Precision: 1 / (1 + 1) = 0.5
        self.assertAlmostEqual(res.precision, 0.5)
        # Recall (Sensitivity): 1 / (1 + 1) = 0.5
        self.assertAlmostEqual(res.recall, 0.5)
        # Specificity: 1 / (1 + 1) = 0.5
        self.assertAlmostEqual(res.specificity, 0.5)
        # F1: 2 * (0.5 * 0.5) / (0.5 + 0.5) = 0.5
        self.assertAlmostEqual(res.f1, 0.5)
        # AUROC should be computable with valid probabilities
        self.assertIsNotNone(res.roc_auc)

    def test_perfect_prediction_metrics(self):
        """Verify perfect prediction yields 1.0 across accuracy, precision, recall, specificity, F1, AUROC."""
        y_true = [1, 1, 0, 0]
        y_pred = [1, 1, 0, 0]
        y_prob = [0.95, 0.85, 0.15, 0.05]

        res = compute_classification_metrics(y_true=y_true, y_pred=y_pred, y_prob=y_prob)

        self.assertEqual(res.status, "SUCCESS")
        self.assertEqual(res.accuracy, 1.0)
        self.assertEqual(res.precision, 1.0)
        self.assertEqual(res.recall, 1.0)
        self.assertEqual(res.specificity, 1.0)
        self.assertEqual(res.f1, 1.0)
        self.assertEqual(res.roc_auc, 1.0)
        self.assertEqual(res.confusion_matrix, {"TP": 2, "FP": 0, "FN": 0, "TN": 2})

    def test_string_labels_conversion(self):
        """String risk labels (HIGH_RISK / LOW_RISK) must correctly map to binary 1/0."""
        y_true = ["HIGH_RISK", "LOW_RISK", "HIGH RISK", "NORMAL"]
        y_pred = ["HIGH_RISK", "LOW_RISK", "LOW_RISK", "NORMAL"]

        res = compute_classification_metrics(y_true=y_true, y_pred=y_pred)
        self.assertEqual(res.status, "SUCCESS")
        self.assertEqual(res.sample_count, 4)
        self.assertEqual(res.confusion_matrix["TP"], 1)
        self.assertEqual(res.confusion_matrix["TN"], 2)
        self.assertEqual(res.confusion_matrix["FN"], 1)
        self.assertEqual(res.confusion_matrix["FP"], 0)

    def test_to_dict_includes_specificity(self):
        """to_dict() must serialize all fields including specificity."""
        res = ClassificationMetricsResult(
            status="SUCCESS",
            sample_count=10,
            accuracy=0.9,
            precision=0.85,
            recall=0.8,
            specificity=0.95,
            f1=0.824,
            roc_auc=0.92,
            confusion_matrix={"TP": 4, "FP": 1, "FN": 1, "TN": 4},
        )
        d = res.to_dict()
        self.assertEqual(d["status"], "SUCCESS")
        self.assertEqual(d["specificity"], 0.95)
        self.assertEqual(d["accuracy"], 0.9)


if __name__ == "__main__":
    unittest.main()
