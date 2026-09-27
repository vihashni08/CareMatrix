"""Structured evidence package and prompt builder for LLM-assisted clinical reasoning.

Ensures that the LLM receives strictly pre-computed, grounded evidence from upstream
agents without raw file access, unrestricted queries, or hallucination vectors.
"""

from __future__ import annotations

import json
from typing import Any

from clinical_reasoning_agent.rag.schemas import RetrievalResult
from clinical_reasoning_agent.state import ClinicalReasoningState


CLINICAL_REASONING_SYSTEM_INSTRUCTION = """You are the Clinical Decision Support synthesizer in the CareMatrix autonomous multi-agent intensive care monitoring architecture.

Your objective is to provide evidence-grounded, non-diagnostic clinical decision support by interpreting and synthesizing analytical evidence already produced by the CareMatrix agent pipeline.

MANDATORY CLINICAL SAFETY AND INTEGRITY GUIDELINES:
1. NON-DIAGNOSTIC: You provide bedside monitoring and risk surveillance decision support, NOT definitive clinical diagnoses.
2. NO HALLUCINATIONS: Do NOT invent vital signs, patient history, physiological measurements, or lab values.
3. GROUNDED REASONING: Base your synthesis strictly and solely on the supplied patient evidence package.
4. EVIDENCE INTEGRITY: Do NOT override the Machine Learning model risk prediction or alter verification requirements.
5. CONFLICT TRANSPARENCY: Clearly distinguish verified analytical evidence from interpretation. If evidence is conflicting, uncertain, or compromised by sensor quality, explicitly state this in conflicting_evidence and uncertainties.
6. MEDICAL LITERATURE DISTINCTION: Any retrieved medical literature provided under RETRIEVED MEDICAL KNOWLEDGE represents general clinical practice guidelines, NOT patient facts. Use it only to provide physiological context and practical bedside recommendations.
7. CITATION INTEGRITY: In knowledge_sources, cite only document_id, title, and section present in the provided medical knowledge. If no medical knowledge was provided or relevant, set retrieval_status to 'NO_RELEVANT_EVIDENCE' and leave knowledge_sources empty. Never fabricate citations.
8. EPISODIC FEEDBACK INTEGRITY: Any episodic memory or clinician feedback provided represents historical bedside context only. You must NEVER suppress, delay, or downgrade a fresh physiological deterioration based on historical false positives or prior resolutions.
9. STRUCTURED OUTPUT: You MUST respond ONLY with a single valid JSON object strictly matching the required schema. Do not enclose in markdown code fences or add extraneous conversational text.
"""

STRUCTURED_OUTPUT_SCHEMA_DESCRIPTION = {
    "clinical_summary": "Concise (1-3 sentences) executive clinical synthesis of current patient state.",
    "supporting_evidence": ["List of specific findings directly supporting the risk assessment."],
    "conflicting_evidence": ["List of any findings contradicting the risk assessment or indicating discordance."],
    "key_findings": ["List of notable physiological trajectories, vital sign changes, or pattern observations."],
    "risk_interpretation": "Brief clinical interpretation of why this patient is at low, elevated, or high risk.",
    "priority": "Exact priority level: 'ROUTINE', 'ELEVATED', or 'URGENT'.",
    "recommended_actions": ["Ordered list of practical, bedside monitoring and assessment protocols."],
    "confidence": 0.95,  # Float between 0.0 and 1.0
    "uncertainties": ["Any data quality, sensor, or physiological limitations identified."],
    "knowledge_sources": [
        {
            "document_id": "ID of retrieved document actually used",
            "title": "Document title",
            "section": "Relevant section",
        }
    ],
    "retrieved_evidence": ["Key physiological threshold or guideline applied from literature."],
    "retrieval_status": "SUCCESS or NO_RELEVANT_EVIDENCE",
}


def build_evidence_package(
    event: DataAnalysisEvent,
    state: ClinicalReasoningState | None = None,
    patient_memory: Any | None = None,
) -> dict[str, Any]:
    """Construct a safe, structured evidence package from upstream agent findings."""
    case_id = getattr(event, "case_id", 0)
    event_id = getattr(event, "event_id", "unknown_event")
    timestamp = getattr(event, "timestamp", 0.0)

    # 1. Model Risk Assessment
    risk_info = {
        "risk_level": getattr(event, "risk_level", "LOW RISK"),
        "analysis_status": getattr(event, "analysis_status", "complete"),
    }

    # 2. Physiological Trend Metrics & Identified Patterns
    metrics = getattr(event, "trend_metrics", {}) or {}
    patterns = getattr(event, "pattern_identified", []) or []
    executed_checks = getattr(event, "executed_checks", []) or []
    quality_flag = bool(getattr(event, "data_quality_flag", False))
    quality_details = getattr(event, "data_quality_details", {}) or {}

    analysis_info = {
        "trend_metrics": metrics,
        "patterns_identified": patterns,
        "executed_checks": executed_checks,
        "data_quality_flag": quality_flag,
        "data_quality_details": quality_details,
    }

    # 3. Cross-Agent Consistency Verification
    consistency_info = {
        "evidence_consistency": getattr(event, "evidence_consistency", "SUPPORTING"),
        "conflict_flags": getattr(event, "conflict_flags", []) or [],
        "verification_required": bool(getattr(event, "verification_required", False)),
    }

    # 4. Patient Historical Context
    patient_context: dict[str, Any] = {
        "case_id": case_id,
        "timestamp_seconds": timestamp,
        "total_prior_evaluations": state.total_evaluations if state else 0,
        "prior_priority": state.last_priority if state else None,
        "active_alerts_count": len(state.active_alerts) if state else 0,
    }

    # 5. Episodic Patient Memory & Clinician Feedback (Additive Context Only)
    episodic_memory: list[dict[str, Any]] = []
    episodic_memory_summary: str = ""
    if patient_memory is not None:
        if hasattr(patient_memory, "get_recent_summaries"):
            episodic_memory = patient_memory.get_recent_summaries(case_id, limit=5)
        if hasattr(patient_memory, "format_memory_for_prompt"):
            episodic_memory_summary = patient_memory.format_memory_for_prompt(case_id, limit=5)
    elif state is not None and hasattr(state, "episodic_memory"):
        episodic_memory = getattr(state, "episodic_memory", [])
        episodic_memory_summary = getattr(state, "episodic_memory_summary", "")

    return {
        "patient_context": patient_context,
        "event_id": event_id,
        "risk_assessment": risk_info,
        "data_analysis": analysis_info,
        "cross_agent_verification": consistency_info,
        "episodic_memory": episodic_memory,
        "episodic_memory_summary": episodic_memory_summary,
    }


def build_reasoning_prompt(
    evidence_package: dict[str, Any],
    retrieval_result: RetrievalResult | None = None,
) -> str:
    """Format patient evidence, episodic clinician feedback, and medical literature into a partitioned prompt."""
    evidence_json = json.dumps(evidence_package, indent=2)
    schema_json = json.dumps(STRUCTURED_OUTPUT_SCHEMA_DESCRIPTION, indent=2)

    # Format retrieved medical knowledge block
    if retrieval_result and retrieval_result.has_evidence:
        passages_text_list = []
        for i, p in enumerate(retrieval_result.passages, start=1):
            passages_text_list.append(
                f"[{i}] SOURCE: {p.source} | DOC_ID: {p.document_id}\n"
                f"TITLE: {p.title}\n"
                f"SECTION: {p.section}\n"
                f"EXCERPT: {p.text}\n"
                f"RELEVANCE SCORE: {p.relevance_score:.3f}"
            )
        knowledge_block = "\n\n".join(passages_text_list)
    else:
        knowledge_block = "NO RELEVANT MEDICAL GUIDANCE RETRIEVED (Base synthesis purely on patient evidence)."

    memory_summary = evidence_package.get("episodic_memory_summary", "").strip()
    memory_section = ""
    if memory_summary and memory_summary != "No prior alert or clinician feedback history for this patient.":
        memory_section = f"""
=== SECTION 1B: EPISODIC PATIENT MEMORY & CLINICIAN FEEDBACK (Historical Context Only) ===
{memory_summary}
"""

    prompt = f"""=== SECTION 1: PATIENT EVIDENCE (Pre-computed CareMatrix Findings) ===
Case: {evidence_package.get('patient_context', {}).get('case_id', 'N/A')}
Event: {evidence_package.get('event_id', 'N/A')}
{evidence_json}
{memory_section}
=== SECTION 2: RETRIEVED MEDICAL KNOWLEDGE (For Clinical Decision Support Context Only — NOT Patient Facts) ===
{knowledge_block}

=== SECTION 3: REQUIRED STRUCTURED JSON OUTPUT ===
Respond ONLY with a valid JSON object strictly matching this schema:
{schema_json}

Synthesize the patient evidence and medical knowledge now:"""
    return prompt


__all__ = [
    "CLINICAL_REASONING_SYSTEM_INSTRUCTION",
    "STRUCTURED_OUTPUT_SCHEMA_DESCRIPTION",
    "build_evidence_package",
    "build_reasoning_prompt",
]

