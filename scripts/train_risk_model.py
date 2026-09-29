#!/usr/bin/env python3
"""CareMatrix: Reproducible Risk Agent Model Training & Evaluation.

Trains 7 candidate ML models using stratified case-level splits from data/case_splits.json,
selects the optimal model and threshold strictly on the validation split,
and evaluates once on the untouched test split.

Outputs:
  - risk_agent/models/best_model.pkl
  - risk_agent/models/preprocessor.pkl
  - risk_agent/models/feature_columns.json
  - risk_agent/models/risk_threshold.json
  - risk_agent/models/model_metadata.json
  - risk_agent/models/model_metrics.json
  - risk_agent/models/model_comparison.csv
  - docs/figures/roc_curve.png
  - docs/figures/pr_curve.png
  - docs/figures/confusion_matrix.png
  - docs/figures/feature_importance.png
  - docs/figures/validation_model_comparison.png
"""

from __future__ import annotations

import datetime
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from xgboost import XGBClassifier

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Project imports
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from data.adapters.vitaldb_adapter import VitalDBAdapter

WINDOW_SECONDS = 300
SAMPLE_INTERVAL = 5
MAX_TRAINING_MINUTES = 60
RANDOM_STATE = 42

VITAL_COLUMNS = ["HR", "SpO2", "RR", "SBP", "DBP", "MAP", "BT"]
DEMOGRAPHIC_COLUMNS = ["age", "sex", "bmi", "asa", "emop"]


def extract_window_features(window_df: pd.DataFrame) -> dict[str, float]:
    """Extract 7-vital statistical features from a time window."""
    features: dict[str, float] = {}
    for vital in VITAL_COLUMNS:
        if vital not in window_df.columns:
            continue
        values = pd.to_numeric(window_df[vital], errors="coerce").dropna()
        if len(values) == 0:
            continue

        features[f"{vital}_mean"] = float(values.mean())
        features[f"{vital}_min"] = float(values.min())
        features[f"{vital}_max"] = float(values.max())
        features[f"{vital}_std"] = float(values.std()) if len(values) > 1 else 0.0
        features[f"{vital}_latest"] = float(values.iloc[-1])
        features[f"{vital}_change"] = float(values.iloc[-1] - values.iloc[0])

        if len(values) > 1:
            x = np.arange(len(values))
            try:
                slope = float(np.polyfit(x, values.values, 1)[0])
            except Exception:
                slope = 0.0
        else:
            slope = 0.0
        features[f"{vital}_slope"] = slope

    return features


def create_patient_windows(
    case_id: int,
    adapter: VitalDBAdapter,
) -> list[dict[str, Any]]:
    """Load case and split into non-overlapping 5-minute windows."""
    try:
        data = adapter.load_case_dataframe(case_id, interval=SAMPLE_INTERVAL)
    except Exception as exc:
        print(f"  [Warning] Failed loading case {case_id}: {exc}")
        return []

    if data is None or data.empty or "Time" not in data.columns:
        return []

    # Filter to first MAX_TRAINING_MINUTES
    data = data[data["Time"] <= MAX_TRAINING_MINUTES * 60].copy()
    if len(data) == 0:
        return []

    max_time = float(data["Time"].max())
    windows: list[dict[str, Any]] = []
    start_time = 0.0

    while start_time + WINDOW_SECONDS <= max_time:
        end_time = start_time + WINDOW_SECONDS
        window = data[(data["Time"] >= start_time) & (data["Time"] < end_time)]
        if len(window) >= 5:  # require at least 5 samples in 5-min window
            feats = extract_window_features(window)
            if len(feats) > 0:
                feats["caseid"] = case_id
                feats["window_start"] = start_time
                feats["window_end"] = end_time
                windows.append(feats)
        start_time += WINDOW_SECONDS

    return windows


def find_best_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, float]:
    """Search optimal decision threshold on validation set maximizing F1."""
    best_thresh = 0.50
    best_f1 = -1.0

    for thresh in np.arange(0.05, 0.91, 0.01):
        preds = (probabilities >= thresh).astype(int)
        f1 = float(f1_score(y_true, preds, zero_division=0))
        rec = float(recall_score(y_true, preds, zero_division=0))
        # Prioritize F1, use recall as tiebreaker
        score = f1 + 0.01 * rec
        if score > best_f1:
            best_f1 = score
            best_thresh = float(thresh)

    actual_f1 = float(f1_score(y_true, (probabilities >= best_thresh).astype(int), zero_division=0))
    return best_thresh, actual_f1


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Compute comprehensive classification metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
    except Exception:
        roc_auc = float("nan")

    try:
        pr_auc = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
    except Exception:
        pr_auc = float("nan")

    brier = float(brier_score_loss(y_true, y_prob))

    return {
        "accuracy": acc,
        "precision": prec,
        "sensitivity": sensitivity,
        "recall": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "threshold": threshold,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "support": {
            "total": int(len(y_true)),
            "positive": int(np.sum(y_true == 1)),
            "negative": int(np.sum(y_true == 0)),
            "prevalence": float(np.mean(y_true)),
        },
    }


def main():
    print("=" * 70)
    print("CAREMATRIX: RISK PREDICTION MODEL TRAINING & BENCHMARKING")
    print("=" * 70)
    start_time_all = time.time()

    # 1. Load Case Splits
    splits_file = REPO_ROOT / "data" / "case_splits.json"
    if not splits_file.exists():
        raise FileNotFoundError(f"Case splits file not found: {splits_file}")

    with open(splits_file, "r", encoding="utf-8") as f:
        splits = json.load(f)

    train_cases = splits["train_cases"]
    val_cases = splits["validation_cases"]
    test_cases = splits["test_cases"]

    # Leakage check
    set_train = set(train_cases)
    set_val = set(val_cases)
    set_test = set(test_cases)
    assert not (set_train & set_val), "Data leakage: overlap between train and val splits!"
    assert not (set_train & set_test), "Data leakage: overlap between train and test splits!"
    assert not (set_val & set_test), "Data leakage: overlap between val and test splits!"
    print(f"Loaded splits: {len(train_cases)} Train, {len(val_cases)} Val, {len(test_cases)} Test (zero leakage verified).")

    # 2. Load Clinical Metadata & Demographics
    adapter = VitalDBAdapter()
    cases_df = None
    local_cases_path = REPO_ROOT / "data" / "vitaldb_cache" / "cases.csv"
    if local_cases_path.exists():
        cases_df = pd.read_csv(local_cases_path)
    else:
        cases_df = pd.read_csv("https://api.vitaldb.net/cases")
        cases_df.to_csv(local_cases_path, index=False)

    cases_df["death_inhosp"] = pd.to_numeric(cases_df.get("death_inhosp", 0), errors="coerce").fillna(0)
    cases_df["icu_days"] = pd.to_numeric(cases_df.get("icu_days", 0), errors="coerce").fillna(0)
    cases_df["High Risk"] = ((cases_df["death_inhosp"] == 1) | (cases_df["icu_days"] > 0)).astype(int)

    case_metadata: dict[int, dict[str, Any]] = {}
    for _, row in cases_df.iterrows():
        cid = int(row["caseid"])
        sex_str = str(row.get("sex", "")).upper()
        case_metadata[cid] = {
            "age": float(pd.to_numeric(row.get("age"), errors="coerce")),
            "sex": 1.0 if sex_str == "M" else 0.0 if sex_str == "F" else np.nan,
            "bmi": float(pd.to_numeric(row.get("bmi"), errors="coerce")),
            "asa": float(pd.to_numeric(row.get("asa"), errors="coerce")),
            "emop": float(pd.to_numeric(row.get("emop"), errors="coerce")),
            "high_risk": int(row["High Risk"]),
        }

    # 3. Parallel Window Extraction
    all_case_ids = train_cases + val_cases + test_cases
    print(f"\nExtracting 5-minute telemetry windows across {len(all_case_ids)} cases...")

    def _process_single_case(cid: int) -> list[dict[str, Any]]:
        meta = case_metadata.get(cid, {})
        windows = create_patient_windows(cid, adapter)
        for w in windows:
            w["age"] = meta.get("age", np.nan)
            w["sex"] = meta.get("sex", np.nan)
            w["bmi"] = meta.get("bmi", np.nan)
            w["asa"] = meta.get("asa", np.nan)
            w["emop"] = meta.get("emop", np.nan)
            w["High Risk"] = meta.get("high_risk", 0)
        return windows

    all_windows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for idx, res in enumerate(ex.map(_process_single_case, all_case_ids)):
            all_windows.extend(res)
            if (idx + 1) % 20 == 0 or (idx + 1) == len(all_case_ids):
                print(f"  Processed {idx + 1}/{len(all_case_ids)} cases -> {len(all_windows)} windows")

    df_windows = pd.DataFrame(all_windows)
    print(f"\nExtracted total {len(df_windows)} windows across {df_windows['caseid'].nunique()} cases.")
    print("Class distribution across all windows:")
    print(df_windows["High Risk"].value_counts(normalize=True).rename("Prevalence"))

    # 4. Partition Windows into Train / Val / Test
    train_df = df_windows[df_windows["caseid"].isin(train_cases)].copy()
    val_df = df_windows[df_windows["caseid"].isin(val_cases)].copy()
    test_df = df_windows[df_windows["caseid"].isin(test_cases)].copy()

    print(f"\nSplit Windows:")
    print(f"  Train:      {len(train_df)} windows ({train_df['caseid'].nunique()} cases, {train_df['High Risk'].sum()} high-risk windows)")
    print(f"  Validation: {len(val_df)} windows ({val_df['caseid'].nunique()} cases, {val_df['High Risk'].sum()} high-risk windows)")
    print(f"  Test:       {len(test_df)} windows ({test_df['caseid'].nunique()} cases, {test_df['High Risk'].sum()} high-risk windows)")

    # 5. Define Feature Columns & Preprocessor
    feature_cols_file = REPO_ROOT / "risk_agent" / "models" / "feature_columns.json"
    if feature_cols_file.exists():
        with open(feature_cols_file, "r") as f:
            ML_FEATURES = json.load(f)
    else:
        # Build standard 54 features
        stat_names = ["mean", "min", "max", "std", "latest", "change", "slope"]
        vital_features = [f"{v}_{s}" for v in VITAL_COLUMNS for s in stat_names]
        ML_FEATURES = vital_features + DEMOGRAPHIC_COLUMNS

    # Ensure all features exist in dataframes
    for col in ML_FEATURES:
        for d in (train_df, val_df, test_df):
            if col not in d.columns:
                d[col] = np.nan

    X_train = train_df[ML_FEATURES]
    y_train = train_df["High Risk"].astype(int).values

    X_val = val_df[ML_FEATURES]
    y_val = val_df["High Risk"].astype(int).values

    X_test = test_df[ML_FEATURES]
    y_test = test_df["High Risk"].astype(int).values

    # Preprocessing Pipeline (median imputer + standard scaler)
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                ML_FEATURES,
            )
        ]
    )

    print("\nFitting preprocessor on training split...")
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)

    # 6. Candidate Models
    candidates: dict[str, Any] = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300,
            max_depth=6,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            max_depth=4,
            random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=float((y_train == 0).sum() / max(1, (y_train == 1).sum())),
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        ),
    }

    # 7. Model Training & Validation Selection
    print("\n" + "=" * 70)
    print("TRAINING & VALIDATING 7 CANDIDATE MODELS")
    print("=" * 70)

    val_results: dict[str, dict[str, Any]] = {}
    fitted_models: dict[str, Any] = {}
    val_probs: dict[str, np.ndarray] = {}

    for name, model in candidates.items():
        t0 = time.time()
        model.fit(X_train_proc, y_train)
        train_time = time.time() - t0
        fitted_models[name] = model

        # Predict probabilities on validation split
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_val_proc)[:, 1]
        else:
            probs = model.decision_function(X_val_proc)
            probs = 1.0 / (1.0 + np.exp(-probs))

        val_probs[name] = probs

        # Threshold search on validation set only
        best_thresh, _ = find_best_threshold(y_val, probs)
        metrics = compute_metrics(y_val, probs, best_thresh)
        metrics["training_time_seconds"] = train_time
        val_results[name] = metrics

        print(f"[{name:22s}] Val F1: {metrics['f1']:.4f} | Sens: {metrics['sensitivity']:.4f} | Spec: {metrics['specificity']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f} | PR-AUC: {metrics['pr_auc']:.4f} | Thresh: {best_thresh:.2f} ({train_time:.1f}s)")

    # 8. Model Selection (highest Validation F1, with ROC-AUC tiebreaker)
    best_model_name = max(
        val_results.keys(),
        key=lambda k: (val_results[k]["f1"], val_results[k]["roc_auc"]),
    )
    best_model = fitted_models[best_model_name]
    best_threshold = float(val_results[best_model_name]["threshold"])

    print("\n" + "=" * 70)
    print(f"SELECTED OPTIMAL MODEL: {best_model_name} (Threshold: {best_threshold:.4f})")
    print("=" * 70)

    # 9. Evaluate Once on Untouched Test Split
    print("\nEvaluating selected model on UNTOUCHED TEST SPLIT...")
    test_probs = best_model.predict_proba(X_test_proc)[:, 1]
    test_window_metrics = compute_metrics(y_test, test_probs, best_threshold)

    # Case-level evaluation: aggregate window probabilities per case
    test_df_eval = test_df.copy()
    test_df_eval["pred_prob"] = test_probs
    case_summary = test_df_eval.groupby("caseid").agg(
        max_prob=("pred_prob", "max"),
        mean_prob=("pred_prob", "mean"),
        actual=("High Risk", "first"),
    ).reset_index()

    y_test_case = case_summary["actual"].values
    case_max_probs = case_summary["max_prob"].values
    case_mean_probs = case_summary["mean_prob"].values

    test_case_max_metrics = compute_metrics(y_test_case, case_max_probs, best_threshold)
    test_case_mean_metrics = compute_metrics(y_test_case, case_mean_probs, best_threshold)

    print(f"\n--- TEST SET RESULTS ({best_model_name}) ---")
    print(f"Window-Level  : F1={test_window_metrics['f1']:.4f}, Sens={test_window_metrics['sensitivity']:.4f}, Spec={test_window_metrics['specificity']:.4f}, ROC-AUC={test_window_metrics['roc_auc']:.4f}, PR-AUC={test_window_metrics['pr_auc']:.4f}")
    print(f"Case-Level Max : Sens={test_case_max_metrics['sensitivity']:.4f}, Spec={test_case_max_metrics['specificity']:.4f}, Acc={test_case_max_metrics['accuracy']:.4f}, F1={test_case_max_metrics['f1']:.4f}, ROC-AUC={test_case_max_metrics['roc_auc']:.4f}")
    print(f"Case-Level Mean: Sens={test_case_mean_metrics['sensitivity']:.4f}, Spec={test_case_mean_metrics['specificity']:.4f}, Acc={test_case_mean_metrics['accuracy']:.4f}, F1={test_case_mean_metrics['f1']:.4f}, ROC-AUC={test_case_mean_metrics['roc_auc']:.4f}")

    # 10. Save Model Artifacts
    models_dir = REPO_ROOT / "risk_agent" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, models_dir / "best_model.pkl")
    joblib.dump(preprocessor, models_dir / "preprocessor.pkl")

    with open(models_dir / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(ML_FEATURES, f, indent=4)

    with open(models_dir / "risk_threshold.json", "w", encoding="utf-8") as f:
        json.dump({"threshold": best_threshold}, f, indent=4)

    metadata = {
        "training_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "selected_model": best_model_name,
        "selected_threshold": best_threshold,
        "primary_case_aggregation": "max",
        "random_state": RANDOM_STATE,
        "window_seconds": WINDOW_SECONDS,
        "sample_interval": SAMPLE_INTERVAL,
        "max_training_minutes": MAX_TRAINING_MINUTES,
        "num_features": len(ML_FEATURES),
        "split_summary": {
            "train_cases": len(train_cases),
            "train_windows": len(train_df),
            "validation_cases": len(val_cases),
            "validation_windows": len(val_df),
            "test_cases": len(test_cases),
            "test_windows": len(test_df),
        },
        "library_versions": {
            "python": sys.version.split()[0],
            "scikit_learn": sklearn.__version__,
            "xgboost": xgb.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "label_definition": "High Risk = (death_inhosp == 1) | (icu_days > 0) [case-level outcome mapped to window data]",
    }
    with open(models_dir / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    full_metrics_export = {
        "metadata": metadata,
        "candidate_validation_comparison": val_results,
        "selected_model": best_model_name,
        "test_metrics_window_level": test_window_metrics,
        "test_metrics_case_level_max": test_case_max_metrics,
        "test_metrics_case_level_mean": test_case_mean_metrics,
    }
    with open(models_dir / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(full_metrics_export, f, indent=4)

    # 11. Save Comparison CSV
    comparison_rows = []
    for name, m in val_results.items():
        comparison_rows.append({
            "model": name,
            "split": "validation",
            "threshold": m["threshold"],
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "sensitivity": m["sensitivity"],
            "specificity": m["specificity"],
            "f1": m["f1"],
            "roc_auc": m["roc_auc"],
            "pr_auc": m["pr_auc"],
            "brier_score": m["brier_score"],
            "training_time_sec": m["training_time_seconds"],
        })
    comparison_rows.append({
        "model": f"{best_model_name} (Selected)",
        "split": "test_window",
        "threshold": test_window_metrics["threshold"],
        "accuracy": test_window_metrics["accuracy"],
        "precision": test_window_metrics["precision"],
        "sensitivity": test_window_metrics["sensitivity"],
        "specificity": test_window_metrics["specificity"],
        "f1": test_window_metrics["f1"],
        "roc_auc": test_window_metrics["roc_auc"],
        "pr_auc": test_window_metrics["pr_auc"],
        "brier_score": test_window_metrics["brier_score"],
        "training_time_sec": val_results[best_model_name]["training_time_seconds"],
    })
    comparison_rows.append({
        "model": f"{best_model_name} (Selected Case-Max)",
        "split": "test_case_max",
        "threshold": test_case_max_metrics["threshold"],
        "accuracy": test_case_max_metrics["accuracy"],
        "precision": test_case_max_metrics["precision"],
        "sensitivity": test_case_max_metrics["sensitivity"],
        "specificity": test_case_max_metrics["specificity"],
        "f1": test_case_max_metrics["f1"],
        "roc_auc": test_case_max_metrics["roc_auc"],
        "pr_auc": test_case_max_metrics["pr_auc"],
        "brier_score": test_case_max_metrics["brier_score"],
        "training_time_sec": val_results[best_model_name]["training_time_seconds"],
    })
    pd.DataFrame(comparison_rows).to_csv(models_dir / "model_comparison.csv", index=False)

    # 12. Generate Benchmark Figures
    figs_dir = REPO_ROOT / "docs" / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    # A. ROC Curve
    plt.figure(figsize=(7, 6))
    for name, probs in val_probs.items():
        fpr, tpr, _ = roc_curve(y_val, probs)
        auc_val = roc_auc_score(y_val, probs)
        plt.plot(fpr, tpr, label=f"{name} (Val AUC={auc_val:.3f})")
    fpr_test, tpr_test, _ = roc_curve(y_test, test_probs)
    plt.plot(fpr_test, tpr_test, "k--", linewidth=2.5, label=f"{best_model_name} (Test AUC={test_window_metrics['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "r:", label="Chance")
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title("Receiver Operating Characteristic (ROC) Benchmark")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figs_dir / "roc_curve.png", dpi=300)
    plt.close()

    # B. PR Curve
    plt.figure(figsize=(7, 6))
    for name, probs in val_probs.items():
        prec, rec, _ = precision_recall_curve(y_val, probs)
        ap = average_precision_score(y_val, probs)
        plt.plot(rec, prec, label=f"{name} (Val AP={ap:.3f})")
    prec_t, rec_t, _ = precision_recall_curve(y_test, test_probs)
    plt.plot(rec_t, prec_t, "k--", linewidth=2.5, label=f"{best_model_name} (Test AP={test_window_metrics['pr_auc']:.3f})")
    plt.xlabel("Recall (Sensitivity)")
    plt.ylabel("Precision")
    plt.title("Precision-Recall (PR) Curve Benchmark")
    plt.legend(loc="lower left", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figs_dir / "pr_curve.png", dpi=300)
    plt.close()

    # C. Confusion Matrix Heatmap
    plt.figure(figsize=(5, 4))
    cm_test = confusion_matrix(y_test, (test_probs >= best_threshold).astype(int))
    plt.imshow(cm_test, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Test Confusion Matrix\n({best_model_name} @ {best_threshold:.2f})")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Low Risk (0)", "High Risk (1)"])
    plt.yticks(tick_marks, ["Low Risk (0)", "High Risk (1)"])
    thresh_val = cm_test.max() / 2.0
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{cm_test[i, j]:,}", horizontalalignment="center",
                     color="white" if cm_test[i, j] > thresh_val else "black", fontsize=12)
    plt.ylabel("True Clinical Label")
    plt.xlabel("Predicted Risk Class")
    plt.tight_layout()
    plt.savefig(figs_dir / "confusion_matrix.png", dpi=300)
    plt.close()

    # D. Feature Importance (Top 15)
    importances = None
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    elif hasattr(best_model, "coef_"):
        importances = np.abs(best_model.coef_[0])

    if importances is not None:
        idx_sorted = np.argsort(importances)[-15:]
        top_feats = [ML_FEATURES[i] for i in idx_sorted]
        top_scores = importances[idx_sorted]

        plt.figure(figsize=(8, 6))
        plt.barh(range(len(top_feats)), top_scores, align="center", color="#2563eb")
        plt.yticks(range(len(top_feats)), top_feats)
        plt.xlabel("Relative Feature Importance")
        plt.title(f"Top 15 Predictive Features ({best_model_name})")
        plt.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(figs_dir / "feature_importance.png", dpi=300)
        plt.close()

    # E. Validation Model Comparison Bar Chart
    plt.figure(figsize=(10, 5))
    mod_names = list(val_results.keys())
    f1_scores = [val_results[m]["f1"] for m in mod_names]
    roc_scores = [val_results[m]["roc_auc"] for m in mod_names]
    x = np.arange(len(mod_names))
    width = 0.35

    plt.bar(x - width/2, f1_scores, width, label="F1 Score", color="#0284c7")
    plt.bar(x + width/2, roc_scores, width, label="ROC-AUC", color="#10b981")
    plt.xticks(x, mod_names, rotation=25, ha="right")
    plt.ylabel("Score")
    plt.title("Candidate Model Validation Performance Comparison")
    plt.legend(loc="upper left")
    plt.ylim(0, 1.05)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(figs_dir / "validation_model_comparison.png", dpi=300)
    plt.close()

    total_time = time.time() - start_time_all
    print("\n" + "=" * 70)
    print(f"TRAINING COMPLETE IN {total_time:.1f}s. ALL ARTIFACTS AND FIGURES SAVED.")
    print("=" * 70)


if __name__ == "__main__":
    main()
