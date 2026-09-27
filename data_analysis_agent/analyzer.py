"""Analytical tools for trend metrics, data quality assessment, and pattern detection."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from communication.events import RiskDecisionEvent
from data_analysis_agent.data_loader import SAMPLE_INTERVAL, VITAL_COLUMNS, WINDOW_SECONDS


def calculate_trend(values: pd.Series) -> tuple[float, str]:
    """Calculate linear slope and classify robust descriptive trend label."""
    if len(values) < 2:
        return 0.0, "insufficient_data"
    slope = float(np.polyfit(np.arange(len(values)), values.to_numpy(), 1)[0])
    scale = max(float(values.std(ddof=0)), abs(float(values.mean())) * 0.005, 1e-6)
    if abs(slope) <= scale * 0.05:
        return slope, "stable"
    return slope, "increasing" if slope > 0 else "decreasing"


def assess_window_quality(
    window: pd.DataFrame,
    expected_samples: int | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Assess data quality, missingness ratio, sample sufficiency, and possible abrupt artifacts."""
    expected = expected_samples or max(2, int(WINDOW_SECONDS / SAMPLE_INTERVAL))
    details: dict[str, Any] = {
        "samples": int(len(window)),
        "expected_samples": expected,
        "vitals": {},
    }
    flagged = len(window) < 2

    for vital in VITAL_COLUMNS:
        values = window[vital]
        missing_ratio = float(values.isna().mean()) if len(values) else 1.0
        clean = values.dropna()
        artifact = False
        if len(clean) >= 3:
            jumps = clean.diff().abs().dropna()
            typical = float(jumps.median())
            artifact = bool((jumps > max(typical * 8, 1e-9)).any() and typical > 0)
        details["vitals"][vital] = {
            "missing_ratio": round(missing_ratio, 4),
            "valid_samples": int(len(clean)),
            "possible_artifact": artifact,
        }
        flagged = flagged or missing_ratio > 0.5 or artifact

    details["insufficient_samples"] = len(window) < expected * 0.25
    return bool(flagged), details


def extract_metrics_and_patterns(
    window: pd.DataFrame,
    risk: RiskDecisionEvent,
    requested_checks: list[str] | None = None,
) -> tuple[dict[str, Any], list[str], list[str], list[str]]:
    """Extract temporal statistical metrics and observable physiological patterns for risk decision context.

    Adapts execution to requested checks (trends, variability, patterns, verification).

    Returns:
        (metrics, patterns, changes, executed_checks)
    """
    metrics: dict[str, Any] = {}
    patterns: list[str] = []
    changes: list[str] = []
    executed_checks: list[str] = ["trends", "data_quality"]

    checks = set(requested_checks or ["trends", "variability", "patterns"])

    # 1. Base Trends & Statistical Metrics
    for vital in VITAL_COLUMNS:
        values = window[vital].dropna()
        if values.empty:
            continue
        slope, direction = calculate_trend(values)
        change = float(values.iloc[-1] - values.iloc[0])
        std = float(values.std(ddof=0))
        metrics[vital] = {
            "mean": float(values.mean()),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "standard_deviation": std,
            "latest_value": float(values.iloc[-1]),
            "change_over_window": change,
            "slope_per_sample": slope,
            "trend": direction,
            "sample_count": int(len(values)),
        }
        if vital in risk.affected_vitals or direction != "stable":
            changes.append(f"{vital}: {direction}; change over window {change:.3f}.")
        if direction != "stable" and vital in risk.affected_vitals:
            patterns.append(f"{vital} shows a {direction} trend in the assessment window.")

        # 2. Variability Analysis (if requested)
        if "variability" in checks:
            if len(values) >= 3 and std > max(abs(float(values.mean())) * 0.10, 1e-6):
                patterns.append(f"{vital} has high variability in the assessment window.")

    if "variability" in checks:
        executed_checks.append("variability")

    # 3. Multi-Vital Pattern Analysis (if requested)
    changed_affected = [
        v for v in risk.affected_vitals
        if metrics.get(v, {}).get("trend") not in (None, "stable", "insufficient_data")
    ]
    if "patterns" in checks:
        executed_checks.append("patterns")
        if len(changed_affected) >= 2:
            patterns.append("Simultaneous directional changes are present across multiple affected vitals.")

        if risk.risk_level == "LOW RISK":
            if changed_affected:
                patterns.append("LOW RISK classification retained; trends are reported as supporting evidence only.")
            else:
                patterns.append("LOW RISK classification retained; available affected-vital data is stable over this window.")
        elif risk.risk_level == "HIGH RISK":
            patterns.append("HIGH RISK classification retained; trend and data-quality evidence is provided for follow-up reasoning.")

    # 4. Cross-Vital Verification (if requested)
    if "verification" in checks:
        executed_checks.append("verification")
        hr_trend = metrics.get("HR", {}).get("trend")
        map_trend = metrics.get("MAP", {}).get("trend")
        if hr_trend == "increasing" and map_trend == "decreasing":
            patterns.append("Hemodynamic discordance verified: compensatory tachycardia with concurrent arterial hypotension.")
        elif hr_trend == "decreasing" and map_trend == "decreasing":
            patterns.append("Hemodynamic depression verified: concurrent bradycardia and hypotension.")

    return metrics, list(dict.fromkeys(patterns)), changes, executed_checks


__all__ = [
    "assess_window_quality",
    "calculate_trend",
    "extract_metrics_and_patterns",
]
