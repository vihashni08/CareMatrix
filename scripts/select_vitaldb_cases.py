#!/usr/bin/env python3
"""Reproducible case selection and stratification script for VitalDB.

Filters cases having all 7 mandatory physiological tracks:
- Solar8000/HR
- Solar8000/PLETH_SPO2
- Solar8000/RR
- Solar8000/NIBP_SBP
- Solar8000/NIBP_DBP
- Solar8000/NIBP_MBP
- Solar8000/BT

Defines ground truth outcome label:
  High Risk = (death_inhosp == 1) | (icu_days > 0)

Splits by CASE ID (never by window) into:
- Train (70%)
- Validation (15%)
- Test (15%)
Ensures stratified class distribution across all splits without data leakage.
Outputs to: data/case_splits.json
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "vitaldb_cache"

REQUIRED_TRACKS = [
    "Solar8000/HR",
    "Solar8000/PLETH_SPO2",
    "Solar8000/RR",
    "Solar8000/NIBP_SBP",
    "Solar8000/NIBP_DBP",
    "Solar8000/NIBP_MBP",
    "Solar8000/BT",
]


def load_vitaldb_metadata() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load cases and tracks tables from local cache or online VitalDB API."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cases_file = CACHE_DIR / "cases.csv"
    trks_file = CACHE_DIR / "trks.csv"

    if cases_file.exists():
        cases_df = pd.read_csv(cases_file)
    else:
        print("Downloading cases table from VitalDB API...")
        cases_df = pd.read_csv("https://api.vitaldb.net/cases")
        cases_df.to_csv(cases_file, index=False)

    if trks_file.exists():
        trks_df = pd.read_csv(trks_file)
    else:
        print("Downloading trks table from VitalDB API...")
        trks_df = pd.read_csv("https://api.vitaldb.net/trks")
        trks_df.to_csv(trks_file, index=False)

    return cases_df, trks_df


def select_cases(
    n_cases: int = 160,
    random_state: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> dict[str, Any]:
    """Select stratified case cohorts with all 7 tracks and partition without leakage."""
    cases_df, trks_df = load_vitaldb_metadata()

    # Find cases containing all 7 tracks
    matching_trks = trks_df[trks_df["tname"].isin(REQUIRED_TRACKS)]
    cases_with_all_tracks = (
        matching_trks.groupby("caseid")["tname"]
        .nunique()
        .loc[lambda s: s == len(REQUIRED_TRACKS)]
        .index.tolist()
    )

    available = cases_df[cases_df["caseid"].isin(cases_with_all_tracks)].copy()
    available["death_inhosp"] = pd.to_numeric(available["death_inhosp"], errors="coerce").fillna(0)
    available["icu_days"] = pd.to_numeric(available["icu_days"], errors="coerce").fillna(0)
    available["high_risk"] = ((available["death_inhosp"] == 1) | (available["icu_days"] > 0)).astype(int)

    # Subsample to target cohort size if specified, maintaining stratification
    if n_cases and len(available) > n_cases:
        available, _ = train_test_split(
            available,
            train_size=n_cases,
            random_state=random_state,
            stratify=available["high_risk"],
        )

    available = available.reset_index(drop=True)

    # Partition into Train vs (Val + Test)
    val_test_ratio = val_ratio + test_ratio
    train_df, val_test_df = train_test_split(
        available,
        test_size=val_test_ratio,
        random_state=random_state,
        stratify=available["high_risk"],
    )

    # Partition (Val + Test) into Validation vs Test
    rel_test_ratio = test_ratio / val_test_ratio
    val_df, test_df = train_test_split(
        val_test_df,
        test_size=rel_test_ratio,
        random_state=random_state,
        stratify=val_test_df["high_risk"],
    )

    train_cases = sorted([int(x) for x in train_df["caseid"].tolist()])
    val_cases = sorted([int(x) for x in val_df["caseid"].tolist()])
    test_cases = sorted([int(x) for x in test_df["caseid"].tolist()])

    # Verification: zero leakage
    assert len(set(train_cases) & set(val_cases)) == 0, "Leakage between train and validation!"
    assert len(set(train_cases) & set(test_cases)) == 0, "Leakage between train and test!"
    assert len(set(val_cases) & set(test_cases)) == 0, "Leakage between validation and test!"

    splits = {
        "metadata": {
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "random_seed": random_state,
            "outcome_definition": "High Risk = (death_inhosp == 1) | (icu_days > 0)",
            "required_tracks": REQUIRED_TRACKS,
            "total_selected_cases": len(available),
            "train_count": len(train_cases),
            "validation_count": len(val_cases),
            "test_count": len(test_cases),
            "train_prevalence": round(float(train_df["high_risk"].mean()), 4),
            "validation_prevalence": round(float(val_df["high_risk"].mean()), 4),
            "test_prevalence": round(float(test_df["high_risk"].mean()), 4),
        },
        "train_cases": train_cases,
        "validation_cases": val_cases,
        "test_cases": test_cases,
    }

    output_path = DATA_DIR / "case_splits.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    print(f"Wrote case splits to: {output_path}")
    print(f"Total: {len(available)} cases (Train: {len(train_cases)}, Val: {len(val_cases)}, Test: {len(test_cases)})")
    print(f"Prevalence -> Train: {splits['metadata']['train_prevalence'] * 100:.1f}%, "
          f"Val: {splits['metadata']['validation_prevalence'] * 100:.1f}%, "
          f"Test: {splits['metadata']['test_prevalence'] * 100:.1f}%")

    return splits


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select and partition VitalDB cases.")
    parser.add_argument("--n-cases", type=int, default=160, help="Total number of cases to include in cohort")
    parser.add_argument("--seed", type=int, default=42, help="Random state for reproducibility")
    args = parser.parse_args()

    select_cases(n_cases=args.n_cases, random_state=args.seed)
