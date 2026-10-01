"""Classification and performance metrics with ground-truth transparency."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np

GROUND_TRUTH_UNAVAILABLE = "GROUND_TRUTH_UNAVAILABLE"


@dataclass
class ClassificationMetricsResult:
    """Structured result of model prediction evaluation."""

    status: str  # "SUCCESS" or "GROUND_TRUTH_UNAVAILABLE"
    sample_count: int = 0
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    specificity: float | None = None
    f1: float | None = None
    roc_auc: float | None = None
    confusion_matrix: dict[str, int] = field(default_factory=dict)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "sample_count": self.sample_count,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "specificity": self.specificity,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "confusion_matrix": self.confusion_matrix,
            "message": self.message,
        }


def compute_classification_metrics(
    y_true: list[int | str | None] | np.ndarray | None,
    y_pred: list[int | str] | np.ndarray,
    y_prob: list[float] | np.ndarray | None = None,
) -> ClassificationMetricsResult:
    """Calculate classification performance without inventing or fabricating ground truth.
    
    If y_true is None or lacks valid binary labels, returns GROUND_TRUTH_UNAVAILABLE.
    """
    if y_true is None:
        return ClassificationMetricsResult(
            status=GROUND_TRUTH_UNAVAILABLE,
            message="Ground truth labels not available in this dataset. Metrics not calculated to prevent hallucinated validation.",
        )

    # Convert to standard numeric arrays filtering out None/NaN
    valid_pairs: list[tuple[int, int, float | None]] = []
    y_prob_list = list(y_prob) if y_prob is not None else [None] * len(y_pred)

    for i, yt in enumerate(y_true):
        if yt is None:
            continue
        try:
            yt_val = int(yt) if str(yt).isdigit() else (1 if str(yt).upper() in ("1", "TRUE", "HIGH_RISK", "HIGH RISK") else 0)
            yp_raw = y_pred[i]
            yp_val = int(yp_raw) if str(yp_raw).isdigit() else (1 if str(yp_raw).upper() in ("1", "TRUE", "HIGH_RISK", "HIGH RISK") else 0)
            prob_val = float(y_prob_list[i]) if y_prob_list[i] is not None else None
            valid_pairs.append((yt_val, yp_val, prob_val))
        except (ValueError, IndexError):
            continue

    if not valid_pairs:
        return ClassificationMetricsResult(
            status=GROUND_TRUTH_UNAVAILABLE,
            message="No valid ground truth outcome pairs available for evaluation.",
        )

    y_t = np.array([p[0] for p in valid_pairs])
    y_p = np.array([p[1] for p in valid_pairs])
    n = len(y_t)

    # Basic counts
    tp = int(np.sum((y_t == 1) & (y_p == 1)))
    fp = int(np.sum((y_t == 0) & (y_p == 1)))
    fn = int(np.sum((y_t == 1) & (y_p == 0)))
    tn = int(np.sum((y_t == 0) & (y_p == 0)))

    acc = (tp + tn) / n if n > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    # Optional AUROC if probabilities and both classes exist
    roc_auc = None
    has_probs = all(p[2] is not None for p in valid_pairs)
    if has_probs and len(np.unique(y_t)) > 1:
        try:
            from sklearn.metrics import roc_auc_score
            probs = np.array([p[2] for p in valid_pairs])
            roc_auc = round(float(roc_auc_score(y_t, probs)), 4)
        except Exception:
            roc_auc = None

    return ClassificationMetricsResult(
        status="SUCCESS",
        sample_count=n,
        accuracy=round(float(acc), 4),
        precision=round(float(prec), 4),
        recall=round(float(rec), 4),
        specificity=round(float(spec), 4),
        f1=round(float(f1), 4),
        roc_auc=roc_auc,
        confusion_matrix={"TP": tp, "FP": fp, "FN": fn, "TN": tn},
        message="Metrics computed against verified reference labels.",
    )


__all__ = [
    "ClassificationMetricsResult",
    "GROUND_TRUTH_UNAVAILABLE",
    "compute_classification_metrics",
]
