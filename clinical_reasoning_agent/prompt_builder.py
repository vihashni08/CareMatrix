"""Structured evidence package and prompt builder for LLM-assisted clinical reasoning.

Ensures that the LLM receives strictly pre-computed, grounded evidence from upstream
agents without raw file access, unrestricted queries, or hallucination vectors.
"""

from __future__ import annotations

import json
from typing import Any

from clinical_reasoning_agent.rag.schemas import RetrievalResult
from clinical_reasoning_agent.state import ClinicalReasoningState


CLINICAL_REASONING_SYSTEM_INSTRUCTION = """You are the Clinical Intelligence Decision Support synthesizer in the CareMatrix autonomous multi-agent intensive care monitoring architecture.

Your objective is to produce a rigorous, structured, evidence-grounded CLINICAL INTELLIGENCE REPORT by synthesizing analytical evidence already produced by the CareMatrix agent pipeline and relevant retrieved medical literature.

MANDATORY CLINICAL SAFETY AND GROUNDING GUIDELINES:
1. NON-DIAGNOSTIC SURVEILLANCE & CLINICAL DECISION SUPPORT BOUNDARY:
   - You provide bedside monitoring, trajectory surveillance, and risk decision support, NOT definitive disease diagnoses.
   - You are a Clinical Decision Support System, NOT an autonomous treatment-ordering or prescribing system.
   - All recommendations must be framed as clinician-facing decision-support guidance: bedside clinical evaluation, physician assessment, diagnostic verification, monitoring priorities, and clinical protocol review.
   - You must NEVER issue direct autonomous treatment orders, drug administrations, dosing instructions, or invasive intervention commands (e.g., NEVER say "Administer 500ml saline", "Start norepinephrine", "Prescribe medication X", "Give bolus Y"). Frame as: "Consider clinician evaluation for appropriate fluid/hemodynamic management as clinically indicated."
2. STRICT GROUNDING & NO HALLUCINATION: Only use information directly supplied in the prompt:
   - Patient context and baseline vital signs from the Monitoring Agent
   - Machine Learning deterioration risk prediction from the Risk Agent
   - Temporal window trends, rates of change, and data quality metrics from the Data Analysis Agent
   - Cross-agent verification flags and consistency findings
   - Explicitly retrieved PubMed literature passages in SECTION 2
   NEVER invent or hallucinate physiological measurements, laboratory values, diagnoses, medications, patient medical history, or unretrieved literature citations.
3. RIGOROUS EVIDENCE SEPARATION: Clearly separate and distinguish:
   - Patient Evidence: Verified vital signs and deviations observed at the bedside.
   - Model Evidence: Predictive ML model risk probability and classification.
   - Temporal Evidence: Rates of change (slopes), window changes (Δ), and multi-vital trends over time.
   - Medical Literature Source Metadata & Excerpt: External clinical guidelines and studies (contextual evidence, NOT patient facts).
   - Evidence Extracted from Source: Specific thresholds and findings directly stated in the retrieved abstract.
   - Case-Specific Interpretation: How the external guidance applies to this patient's observed trajectory.
4. MEDICAL LITERATURE REASONING & CLAIM-LEVEL GROUNDING:
   - Treat retrieved literature as contextual guidance on physiological mechanisms and clinical protocols, never as proof of an unmeasured patient condition.
   - For every relevant retrieved PubMed source, explicitly explain:
     * Why was this source retrieved based on the patient's presentation?
     * What key biomedical evidence or guideline threshold does it offer? (MUST be directly stated in the retrieved text).
     * How does that evidence apply directly to this patient's case? (Do NOT introduce unmeasured patient facts or autonomous treatment orders).
   - If a claim cannot be verified directly from the retrieved abstract, DO NOT fabricate or exaggerate. State what the text actually says or mark uncertainty.
   - Never cite unprovided PMIDs, authors, or papers. If no medical knowledge was provided or relevant, set retrieval_status to 'NO_RELEVANT_EVIDENCE' and leave medical_evidence empty.
5. UNCERTAINTY & CONFLICT IDENTIFICATION: Explicitly report any sensor noise, missing or sparse channels, conflicting cross-agent findings (e.g. high model risk with stable vitals, or low model risk with deteriorating vitals), or data quality limitations under 'conflicting_evidence' and 'uncertainties'.
6. DETERMINISTIC SAFETY BOUNDARY: Do NOT downplay severe physiological instability or contradict the deterministic safety baseline.
7. HISTORICAL CONTEXT: Any episodic memory or clinician feedback provided represents historical bedside context only. Never suppress or downgrade a fresh acute physiological deterioration.
8. STRUCTURED JSON OUTPUT: You MUST respond ONLY with a single valid JSON object strictly adhering to the schema below. Do NOT wrap in markdown fences (```json ... ```) or include extra conversational text.
"""

STRUCTURED_OUTPUT_SCHEMA_DESCRIPTION = {
    "executive_summary": "Comprehensive 2-4 sentence executive clinical synthesis of current patient state, primary risk drivers, and urgency.",
    "clinical_summary": "Concise executive clinical summary (aligned with executive_summary for backward compatibility).",
    "clinical_status": {
        "risk_level": "LOW RISK or HIGH RISK as predicted by the model",
        "priority": "Exact priority level: 'ROUTINE', 'ELEVATED', or 'URGENT'",
        "confidence": 0.95,  # Float between 0.0 and 1.0 reflecting evidence certainty
        "data_reliability": "HIGH or COMPROMISED",
        "evidence_consistency": "SUPPORTING, CONFLICTING, or UNCERTAIN",
    },
    "key_findings": ["List of notable physiological findings, trajectory changes, or cross-agent verification flags."],
    "physiological_analysis": [
        "System-by-system physiological breakdown (e.g. Cardiovascular: HR/MAP trajectories; Respiratory: SpO2/RR stability; Hemodynamics: perfusion signs)."
    ],
    "temporal_analysis": [
        "Detailed temporal breakdown of vital trajectories: rates of change, magnitude of drift over the window, and stability vs acceleration."
    ],
    "risk_interpretation": "In-depth clinical interpretation of the predictive risk model output in the context of observed physiology.",
    "supporting_evidence": ["Specific verified findings directly corroborating the clinical assessment."],
    "conflicting_evidence": ["Any findings contradicting the risk classification or showing cross-agent discordance."],
    "evidence_synthesis": "Coherent narrative synthesis bridging patient observations, temporal trajectory, and risk predictions.",
    "medical_evidence": [
        {
            "pmid": "PubMed PMID if available",
            "title": "Title of retrieved document",
            "authors": "Author list if available",
            "journal": "Journal name if available",
            "year": "Publication year if available",
            "doi": "DOI if available",
            "relevance_score": 0.85,
            "abstract": "Verified abstract or source passage text",
            "why_retrieved": "Clinical rationale explaining why this literature source was retrieved for this case",
            "key_evidence": "Core physiological threshold or clinical guidance directly stated in the paper",
            "grounding_status": "GROUNDED or UNCERTAIN",
            "application_to_case": "Specific clinical application of this evidence to the patient's current trajectory (decision support)",
        }
    ],
    "clinical_interpretation": "Comprehensive clinical reasoning evaluating potential underlying physiological stress and bedside implications.",
    "uncertainties": ["Data quality limitations, sensor artifacts, missing channels, or physiological ambiguities."],
    "recommended_actions": [
        "Prioritized clinician-facing decision-support recommendations: bedside clinical evaluation, diagnostic verification, monitoring priorities, and protocol review (NO direct prescriptive medication/treatment orders)."
    ],
    "monitoring_priorities": ["Specific vitals, assessment frequencies, and telemetry alerts to prioritize at the bedside."],
    "escalation_rationale": "Clear clinical justification for the assigned urgency priority and specific thresholds that warrant escalation.",
    "priority": "Exact priority level: 'ROUTINE', 'ELEVATED', or 'URGENT'",
    "confidence": 0.95,  # Float between 0.0 and 1.0
    "knowledge_sources": [
        {
            "document_id": "Document or passage ID actually used",
            "title": "Title of source",
            "section": "Section or abstract",
        }
    ],
    "retrieved_evidence": ["Key excerpt or guideline threshold applied from literature."],
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
            pmid = getattr(p, "pmid", "") or (p.metadata.get("pmid", "") if hasattr(p, "metadata") else "")
            authors = getattr(p, "authors", "") or (p.metadata.get("authors", "") if hasattr(p, "metadata") else "")
            journal = getattr(p, "journal", "") or (p.metadata.get("journal", "") if hasattr(p, "metadata") else "")
            year = getattr(p, "year", "") or (p.metadata.get("year", "") if hasattr(p, "metadata") else "")
            doi = getattr(p, "doi", "") or (p.metadata.get("doi", "") if hasattr(p, "metadata") else "")

            meta_lines = []
            if pmid:
                meta_lines.append(f"PMID: {pmid}")
            if authors:
                meta_lines.append(f"AUTHORS: {authors}")
            if journal:
                meta_lines.append(f"JOURNAL: {journal}")
            if year:
                meta_lines.append(f"YEAR: {year}")
            if doi:
                meta_lines.append(f"DOI: {doi}")
            meta_str = ("\n" + "\n".join(meta_lines)) if meta_lines else ""

            passages_text_list.append(
                f"[{i}] SOURCE: {p.source} | DOC_ID: {p.document_id}\n"
                f"TITLE: {p.title}"
                f"{meta_str}\n"
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

