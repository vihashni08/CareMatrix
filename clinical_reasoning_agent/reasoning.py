"""Non-diagnostic clinical reasoning engine synthesizing risk predictions and analytical evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from communication.events import DataAnalysisEvent
from communication.orchestration import verify_cross_agent_consistency
from clinical_reasoning_agent.schemas import (
    ClinicalReasoningPriority,
    LLMReasoningResult,
    ReasoningMode,
)
from clinical_reasoning_agent.state import ClinicalReasoningState


@dataclass
class ClinicalReasoningOutput:
    """Structured, explainable clinical reasoning result produced by the engine."""

    clinical_status: str
    priority: str
    escalation_required: bool
    clinical_summary: str
    findings: list[str]
    recommended_actions: list[str]
    data_reliability: str
    confidence: float
    evidence_consistency: str = "SUPPORTING"
    supporting_evidence: list[str] = field(default_factory=list)
    conflicting_evidence: list[str] = field(default_factory=list)
    verification_required: bool = False
    reasoning_mode: str = ReasoningMode.DETERMINISTIC.value
    llm_reasoning_conflict: bool = False
    knowledge_sources: list[dict[str, Any]] = field(default_factory=list)
    retrieved_evidence: list[str] = field(default_factory=list)
    retrieval_status: str = "NO_RELEVANT_EVIDENCE"
    metadata: dict[str, Any] = field(default_factory=dict)


class ClinicalReasoningEngine:
    """Evaluates analytical evidence to synthesize structured clinical recommendations.

    Deliberately non-diagnostic: provides clinical decision support focusing on
    physiological trend interpretation, cross-agent consistency verification,
    urgency prioritization, sensor reliability, and actionable bedside protocols.
    """

    def evaluate(
        self,
        event: DataAnalysisEvent,
        state: ClinicalReasoningState | None = None,
    ) -> ClinicalReasoningOutput:
        """Synthesize risk assessment, vital trends, patterns, and data quality into an actionable decision."""
        risk_level = getattr(event, "risk_level", "LOW RISK")
        patterns = getattr(event, "pattern_identified", []) or []
        metrics = getattr(event, "trend_metrics", {}) or {}
        quality_flag = bool(getattr(event, "data_quality_flag", False))
        status = getattr(event, "analysis_status", "complete")
        case_id = getattr(event, "case_id", 0)
        event_id = getattr(event, "event_id", "unknown_event")

        # 1. Execute Cross-Agent Evidence Consistency Verification
        (
            evidence_consistency,
            supporting_evidence,
            conflicting_evidence,
            conflict_flags,
            verification_required,
            confidence,
        ) = verify_cross_agent_consistency(
            risk_level=risk_level,
            metrics=metrics,
            patterns=patterns,
            data_quality_flag=quality_flag,
            analysis_status=status,
        )

        # Merge any conflict flags from the upstream event
        upstream_flags = getattr(event, "conflict_flags", []) or []
        for f in upstream_flags:
            if f not in conflict_flags:
                conflict_flags.append(f)

        findings: list[str] = []
        recommended_actions: list[str] = []

        # 2. Extract Contributing Physiological Findings from Trend Metrics
        worsening_vitals: list[str] = []
        stable_vitals: list[str] = []
        trend_summaries: list[str] = []

        for vital, m in metrics.items():
            trend = m.get("trend", "stable")
            latest = m.get("latest_value")
            change = m.get("change_over_window")
            val_str = f"latest={latest:.1f}" if latest is not None else ""
            chg_str = f"Δ={change:+.1f}" if change is not None else ""

            if trend in ("increasing", "decreasing"):
                worsening_vitals.append(vital)
                trend_summaries.append(f"{vital} {trend} ({val_str}, {chg_str})".strip())
            elif trend == "stable":
                stable_vitals.append(vital)

        if trend_summaries:
            findings.append("Active vital trajectories: " + "; ".join(trend_summaries) + ".")
        elif stable_vitals:
            findings.append(f"Monitored vitals ({', '.join(stable_vitals)}) remain stable over the assessment window.")

        for p in patterns:
            findings.append(p)

        # Add cross-agent verification notes to findings
        if supporting_evidence:
            findings.append("Supporting evidence: " + " ".join(supporting_evidence))
        if conflicting_evidence:
            findings.append("Conflicting evidence: " + " ".join(conflicting_evidence))

        # 3. Patient State & History Context Integration
        if state is not None and state.previous_events:
            prior = state.previous_events[-1]
            findings.append(
                f"Historical context: Prior assessment was {prior['priority']} at t={prior['timestamp']:.1f}s."
            )
            if state.total_evaluations > 1:
                findings.append(f"Cumulative active alerts evaluated for case: {len(state.active_alerts)}.")

        # 4. Synthesize Clinical Status, Priority, and Actionable Pathways
        if evidence_consistency == "UNCERTAIN":
            priority = ClinicalReasoningPriority.ELEVATED.value
            escalation_required = False
            data_reliability = "COMPROMISED"
            clinical_status = "DATA_QUALITY_COMPROMISED"
            clinical_summary = (
                f"ELEVATED surveillance advised for Case {case_id}: Signal quality limitations "
                f"({', '.join(conflict_flags) or 'sensor noise'}) require manual verification."
            )
            recommended_actions.append(
                "Inspect sensor attachment, probe placement, and transducer calibration prior to clinical intervention."
            )
            recommended_actions.append("Verify vital sign fidelity with manual measurement.")
            recommended_actions.append("Continue close physiological monitoring.")

        elif risk_level == "HIGH RISK":
            data_reliability = "HIGH"
            if evidence_consistency == "SUPPORTING":
                priority = ClinicalReasoningPriority.URGENT.value
                escalation_required = True
                clinical_status = "CRITICAL_PHYSIOLOGICAL_DETERIORATION"
                clinical_summary = (
                    f"HIGH RISK assessment confirmed for Case {case_id} ({event_id}). "
                    f"Active deterioration detected in {', '.join(worsening_vitals) or 'monitored vitals'} requiring immediate clinical attention."
                )
                recommended_actions.insert(
                    0,
                    "Immediate bedside clinical evaluation of patient hemodynamic and respiratory status.",
                )
                recommended_actions.append(
                    "Notify attending care team and prepare targeted stabilization protocol if deviation persists.",
                )
                recommended_actions.append(
                    "Initiate high-frequency physiological re-assessment.",
                )
            else:
                # HIGH RISK but conflicting evidence (e.g. vitals stable)
                priority = ClinicalReasoningPriority.ELEVATED.value
                escalation_required = False
                clinical_status = "DISCORDANT_RISK_AND_TRENDS"
                clinical_summary = (
                    f"ELEVATED surveillance advised for Case {case_id}: Model predicted HIGH RISK, "
                    "but analytical trends remain stable over the assessment window. Verification required."
                )
                recommended_actions.append(
                    "Verify risk prediction against stable clinical signs before aggressive intervention."
                )
                recommended_actions.append(
                    "Re-evaluate patient vitals and ensure sensor calibration."
                )
                recommended_actions.append("Maintain close observation.")

        else:
            # LOW RISK
            data_reliability = "HIGH"
            if evidence_consistency == "CONFLICTING":
                priority = ClinicalReasoningPriority.ELEVATED.value
                escalation_required = False
                clinical_status = "UNEXPECTED_PHYSIOLOGICAL_DETERIORATION"
                clinical_summary = (
                    f"ELEVATED surveillance advised for Case {case_id}: Model predicted LOW RISK, "
                    f"but multiple vitals exhibit worsening trajectories ({', '.join(worsening_vitals)})."
                )
                recommended_actions.append(
                    "Urgent bedside review advised: multiple vitals deteriorating despite LOW RISK model baseline."
                )
                recommended_actions.append("Continue close physiological observation and track trajectory trends.")
                recommended_actions.append("Shorten re-evaluation window to monitor for sustained deterioration.")
            elif worsening_vitals or any("multiple affected vitals" in p.lower() for p in patterns):
                priority = ClinicalReasoningPriority.ELEVATED.value
                escalation_required = False
                clinical_status = "POTENTIAL_INSTABILITY_UNDER_OBSERVATION"
                clinical_summary = (
                    f"ELEVATED surveillance advised for Case {case_id}: Vitals exhibit directional changes "
                    f"({', '.join(worsening_vitals)}) within safe model risk tolerance."
                )
                recommended_actions.append("Continue close physiological observation and track trajectory trends.")
                recommended_actions.append("Shorten re-evaluation window to monitor for sustained deterioration.")
            else:
                priority = ClinicalReasoningPriority.ROUTINE.value
                escalation_required = False
                clinical_status = "PHYSIOLOGICALLY_STABLE"
                clinical_summary = (
                    f"ROUTINE monitoring indicated for Case {case_id}: Physiological parameters are stable or "
                    "within expected baseline tolerance."
                )
                recommended_actions.append("Continue standard automated surveillance protocol.")

        return ClinicalReasoningOutput(
            clinical_status=clinical_status,
            priority=priority,
            escalation_required=escalation_required,
            clinical_summary=clinical_summary,
            findings=findings,
            recommended_actions=recommended_actions,
            data_reliability=data_reliability,
            confidence=confidence,
            evidence_consistency=evidence_consistency,
            supporting_evidence=supporting_evidence,
            conflicting_evidence=conflicting_evidence,
            verification_required=verification_required,
            reasoning_mode=ReasoningMode.DETERMINISTIC.value,
            llm_reasoning_conflict=False,
            metadata={
                "analysis_status": status,
                "data_quality_flag": quality_flag,
                "conflict_flags": conflict_flags,
                "decision_support_only": True,
            },
        )

    analyze_case = evaluate

    def arbitrate(
        self,
        deterministic: ClinicalReasoningOutput,
        llm_result: LLMReasoningResult | None,
        risk_level: str,
        data_quality_flag: bool = False,
        fallback_error: str | None = None,
    ) -> ClinicalReasoningOutput:
        """Arbitrate between deterministic safety baseline and LLM-assisted interpretation.

        Deterministic safety rules always serve as the final safety boundary.
        If the LLM conflicts with deterministic safety requirements (e.g. proposes ROUTINE
        for a HIGH RISK case, or ignores data quality degradation), deterministic safety overrides
        the decision and logs llm_reasoning_conflict = True.
        """
        if llm_result is None:
            # Deterministic fallback or standalone mode
            mode = (
                ReasoningMode.LLM_FALLBACK.value
                if fallback_error
                else ReasoningMode.DETERMINISTIC.value
            )
            meta = dict(deterministic.metadata)
            meta["reasoning_mode"] = mode
            if fallback_error:
                meta["llm_fallback_reason"] = fallback_error

            return ClinicalReasoningOutput(
                clinical_status=deterministic.clinical_status,
                priority=deterministic.priority,
                escalation_required=deterministic.escalation_required,
                clinical_summary=deterministic.clinical_summary,
                findings=deterministic.findings,
                recommended_actions=deterministic.recommended_actions,
                data_reliability=deterministic.data_reliability,
                confidence=deterministic.confidence,
                evidence_consistency=deterministic.evidence_consistency,
                supporting_evidence=deterministic.supporting_evidence,
                conflicting_evidence=deterministic.conflicting_evidence,
                verification_required=deterministic.verification_required,
                reasoning_mode=mode,
                llm_reasoning_conflict=False,
                metadata=meta,
            )

        # 1. Check for Critical Risk vs Priority Conflict
        is_high_risk = risk_level in ("HIGH RISK", "HIGH_RISK")
        llm_priority = llm_result.priority.upper()
        conflict = False

        if is_high_risk and llm_priority == ClinicalReasoningPriority.ROUTINE.value:
            conflict = True

        # 2. Preserve Deterministic Safety Boundaries
        final_verification_required = (
            deterministic.verification_required or is_high_risk or data_quality_flag
        )
        final_data_reliability = (
            "COMPROMISED"
            if (data_quality_flag or deterministic.data_reliability == "COMPROMISED")
            else "HIGH"
        )

        # 3. Determine Final Priority & Summary Under Safety Rules
        if conflict:
            # LLM downgraded HIGH RISK to ROUTINE -> reject LLM priority, enforce safety baseline
            final_priority = deterministic.priority
            final_escalation = deterministic.escalation_required
            final_clinical_status = deterministic.clinical_status
            final_summary = (
                f"[SAFETY ARBITRATION] {deterministic.clinical_summary} "
                f"(Safety notice: LLM suggested {llm_priority} priority, overridden by deterministic "
                f"safety rules due to model {risk_level} prediction.)"
            )
            final_conflicting = list(deterministic.conflicting_evidence)
            final_conflicting.append(
                f"LLM proposed {llm_priority} priority conflicting with model {risk_level} prediction."
            )
            final_supporting = list(deterministic.supporting_evidence)
            final_findings = list(deterministic.findings)
            final_findings.append(
                f"LLM Reasoning conflict flagged: LLM interpretation contradicted safety baseline."
            )
            final_actions = list(deterministic.recommended_actions)
            final_confidence = min(deterministic.confidence, 0.70)
        else:
            # LLM reasoning aligns with safety bounds
            final_priority = llm_priority
            final_escalation = (
                True if final_priority == ClinicalReasoningPriority.URGENT.value else deterministic.escalation_required
            )
            final_clinical_status = (
                deterministic.clinical_status
                if final_priority == deterministic.priority
                else f"LLM_ASSISTED_{final_priority}"
            )
            final_summary = llm_result.clinical_summary
            final_supporting = llm_result.supporting_evidence or deterministic.supporting_evidence
            final_conflicting = llm_result.conflicting_evidence or deterministic.conflicting_evidence
            final_actions = llm_result.recommended_actions or deterministic.recommended_actions
            final_findings = list(llm_result.key_findings) if llm_result.key_findings else list(deterministic.findings)

            # Bound confidence: if data is compromised, confidence cannot exceed 0.60
            if final_data_reliability == "COMPROMISED":
                final_confidence = min(llm_result.confidence, 0.60)
            else:
                final_confidence = llm_result.confidence

        # Extract RAG knowledge sources and retrieval status
        knowledge_sources = getattr(llm_result, "knowledge_sources", []) or []
        retrieved_evidence = getattr(llm_result, "retrieved_evidence", []) or []
        retrieval_status = getattr(llm_result, "retrieval_status", "NO_RELEVANT_EVIDENCE")

        # Determine mode: LLM_RAG if retrieval succeeded and was used, else LLM_ASSISTED
        if retrieval_status == "SUCCESS" and knowledge_sources:
            reasoning_mode = ReasoningMode.LLM_RAG.value
        else:
            reasoning_mode = ReasoningMode.LLM_ASSISTED.value

        meta = dict(deterministic.metadata)
        meta["reasoning_mode"] = reasoning_mode
        meta["llm_reasoning_conflict"] = conflict
        meta["risk_interpretation"] = llm_result.risk_interpretation
        meta["uncertainties"] = llm_result.uncertainties
        meta["knowledge_sources"] = knowledge_sources
        meta["retrieved_evidence"] = retrieved_evidence
        meta["retrieval_status"] = retrieval_status

        return ClinicalReasoningOutput(
            clinical_status=final_clinical_status,
            priority=final_priority,
            escalation_required=final_escalation,
            clinical_summary=final_summary,
            findings=final_findings,
            recommended_actions=final_actions,
            data_reliability=final_data_reliability,
            confidence=final_confidence,
            evidence_consistency=deterministic.evidence_consistency,
            supporting_evidence=final_supporting,
            conflicting_evidence=final_conflicting,
            verification_required=final_verification_required,
            reasoning_mode=reasoning_mode,
            llm_reasoning_conflict=conflict,
            knowledge_sources=knowledge_sources,
            retrieved_evidence=retrieved_evidence,
            retrieval_status=retrieval_status,
            metadata=meta,
        )


__all__ = [
    "ClinicalReasoningEngine",
    "ClinicalReasoningOutput",
]
