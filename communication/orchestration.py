"""Adaptive orchestration and cross-agent verification rules for CareMatrix.

Provides lightweight, evidence-aware decision support for adjusting analysis depth
and verifying cross-agent consistency between model predictions and physiological trends.
"""

from __future__ import annotations

from typing import Any


def is_high_risk_case(
    decision: str | None = None,
    probability: float | None = None,
    severity: str | None = None,
) -> bool:
    """Evaluate whether a case meets the shared high-risk definition.

    Shared between DataAnalysis requirements and ClinicalReasoning evidence retrieval gating:
    True if decision is HIGH_RISK, OR probability >= 0.16, OR severity in (severe, critical).
    Fails safe to True if decision is missing, empty, or unparseable.
    """
    if decision is None and probability is None and severity is None:
        return True

    if decision is not None:
        d_str = str(decision).strip().upper().replace(" ", "_")
        if not d_str:
            return True
        if d_str in ("HIGH_RISK", "HIGH"):
            return True
        if d_str not in ("LOW_RISK", "LOW", "MODERATE", "MEDIUM", "INDETERMINATE"):
            # Unparseable / unrecognized risk string -> fail safe to True
            return True

    if probability is not None:
        try:
            if float(probability) >= 0.16:
                return True
        except (ValueError, TypeError):
            pass

    if severity is not None:
        s_str = str(severity).strip().lower()
        if s_str in ("severe", "critical"):
            return True

    if decision is not None:
        d_str = str(decision).strip().upper().replace(" ", "_")
        if d_str in ("LOW_RISK", "LOW"):
            return False

    return False


def determine_analysis_requirements(
    decision: str,
    probability: float,
    severity: str,
    affected_vitals: list[str],
) -> tuple[str, list[str], bool]:
    """Determine required analysis depth and checks based on initial risk assessment.

    Returns:
        (analysis_level, requested_checks, verification_required)
    """
    if is_high_risk_case(decision, probability, severity):
        analysis_level = "detailed"
        requested_checks = ["trends", "variability", "patterns", "verification"]
        verification_required = True
    elif len(affected_vitals) >= 2:
        analysis_level = "standard"
        requested_checks = ["trends", "data_quality", "patterns"]
        verification_required = False
    else:
        analysis_level = "standard"
        requested_checks = ["trends", "data_quality"]
        verification_required = False

    return analysis_level, requested_checks, verification_required


def verify_cross_agent_consistency(
    risk_level: str,
    metrics: dict[str, Any],
    patterns: list[str],
    data_quality_flag: bool,
    analysis_status: str,
) -> tuple[str, list[str], list[str], list[str], bool, float]:
    """Cross-verify Risk Agent output against Data Analysis temporal findings.

    Compares the model risk prediction with observed vital sign trajectories,
    sensor quality, and multi-vital patterns to detect evidence alignment or conflicts.

    Returns:
        (evidence_consistency, supporting_evidence, conflicting_evidence, conflict_flags, verification_required, confidence)
    """
    supporting_evidence: list[str] = []
    conflicting_evidence: list[str] = []
    conflict_flags: list[str] = []

    # 1. Identify active trend trajectories
    worsening_vitals = [
        vital
        for vital, m in metrics.items()
        if m.get("trend") in ("increasing", "decreasing")
    ]
    stable_vitals = [
        vital
        for vital, m in metrics.items()
        if m.get("trend") == "stable"
    ]

    # 2. Check Data Quality limitations
    if data_quality_flag or analysis_status == "partial_analysis":
        evidence_consistency = "UNCERTAIN"
        conflict_flags.append("DATA_QUALITY_COMPROMISED")
        conflicting_evidence.append(
            "Data quality alert: High missingness, sparsity, or possible sensor artifacts detected."
        )
        verification_required = True
        confidence = 0.55
        return (
            evidence_consistency,
            supporting_evidence,
            conflicting_evidence,
            conflict_flags,
            verification_required,
            confidence,
        )

    # 3. Cross-verify HIGH RISK
    if risk_level == "HIGH RISK":
        if worsening_vitals:
            evidence_consistency = "SUPPORTING"
            supporting_evidence.append(
                f"Model HIGH RISK prediction substantiated by active deterioration in: {', '.join(worsening_vitals)}."
            )
            verification_required = False
            confidence = 0.95
        else:
            evidence_consistency = "CONFLICTING"
            conflict_flags.append("HIGH_RISK_WITH_STABLE_TRENDS")
            conflicting_evidence.append(
                f"Model predicted HIGH RISK, but all monitored vitals ({', '.join(stable_vitals) or 'monitored vitals'}) remain stable over the assessment window."
            )
            verification_required = True
            confidence = 0.70

    # 4. Cross-verify LOW RISK
    else:
        rapid_worsening = len(worsening_vitals) >= 2 or any("simultaneous directional changes" in p.lower() for p in patterns)
        if rapid_worsening:
            evidence_consistency = "CONFLICTING"
            conflict_flags.append("LOW_RISK_WITH_WORSENING_TRENDS")
            conflicting_evidence.append(
                f"Model predicted LOW RISK, but multiple vitals exhibit worsening trajectories ({', '.join(worsening_vitals)})."
            )
            verification_required = True
            confidence = 0.65
        elif worsening_vitals:
            evidence_consistency = "SUPPORTING"
            supporting_evidence.append(
                f"Model LOW RISK prediction with isolated trajectory change in: {', '.join(worsening_vitals)}."
            )
            verification_required = False
            confidence = 0.85
        else:
            evidence_consistency = "SUPPORTING"
            supporting_evidence.append(
                "Model LOW RISK prediction corroborated by stable physiological trajectories."
            )
            verification_required = False
            confidence = 0.95

    return (
        evidence_consistency,
        supporting_evidence,
        conflicting_evidence,
        conflict_flags,
        verification_required,
        confidence,
    )


__all__ = [
    "is_high_risk_case",
    "determine_analysis_requirements",
    "verify_cross_agent_consistency",
]

