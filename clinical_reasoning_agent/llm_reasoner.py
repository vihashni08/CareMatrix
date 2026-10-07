"""LLM reasoning tool for the Clinical Reasoning Agent using Google Gemini.

Interprets and synthesizes structured analytical evidence packages into validated,
explainable clinical decision support outputs using the Gemini API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import re
from pathlib import Path
from typing import Any

from clinical_reasoning_agent.prompt_builder import (
    CLINICAL_REASONING_SYSTEM_INSTRUCTION,
    build_reasoning_prompt,
)
from clinical_reasoning_agent.rag.schemas import RetrievalResult
from clinical_reasoning_agent.schemas import (
    ClinicalReasoningPriority,
    LLMReasoningResult,
)


from communication.llm_client import (
    BaseGeminiReasoner,
    LLMReasonerError,
    LLMSchemaValidationError,
    LLMTimeoutError,
    LLMUnavailableError,
    clean_json_text as _clean_json_text,
    find_api_key as _find_api_key,
)


def _validate_claim_grounding(claim: str, source_text: str) -> tuple[bool, str]:
    """Verify that a literature claim is genuinely grounded in the source text.

    Checks:
    1. Non-empty claim and source.
    2. Direct containment or high lexical/concept alignment.
    3. Absence of hallucinated prescriptive orders or numbers not in source.

    Returns:
        (is_grounded, grounding_status)
    """
    if not claim or not claim.strip():
        return False, "UNVERIFIED_EMPTY_CLAIM"
    if not source_text or not source_text.strip():
        return False, "UNVERIFIED_NO_SOURCE"

    claim_norm = claim.strip().lower()
    source_norm = source_text.strip().lower()

    # Exact substring
    if claim_norm in source_norm:
        return True, "GROUNDED"

    # Stopwords filter
    stopwords = {
        "the", "and", "for", "that", "this", "with", "from", "are", "was", "were",
        "been", "being", "have", "has", "had", "does", "did", "can", "could",
        "should", "would", "will", "shall", "may", "might", "must", "about",
        "above", "after", "again", "against", "all", "any", "both", "each",
        "few", "more", "most", "other", "some", "such", "only", "own", "same",
        "than", "too", "very", "into", "through", "during", "before", "under",
        "between", "out", "off", "over", "down", "our", "their", "its", "which",
        "who", "whom", "whose", "what", "when", "where", "why", "how", "not",
        "also", "than", "then", "into", "per", "via", "well", "such",
    }

    claim_tokens = [
        w for w in re.findall(r"\b[a-z0-9\-\.\%]+\b", claim_norm)
        if len(w) >= 3 and w not in stopwords
    ]
    if not claim_tokens:
        return True, "GROUNDED"

    source_tokens = set(
        w for w in re.findall(r"\b[a-z0-9\-\.\%]+\b", source_norm)
        if len(w) >= 2
    )

    # Check for unsupported prescriptive commands in key_evidence
    prescriptive_keywords = {"administer", "prescribe", "infuse", "bolus", "inject"}
    if any(k in claim_norm for k in prescriptive_keywords) and not any(k in source_norm for k in prescriptive_keywords):
        return False, "UNVERIFIED_PRESCRIPTIVE_CLAIM_REPLACED"

    # Check numbers/thresholds in claim
    claim_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", claim_norm))
    source_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", source_norm))
    unsupported_numbers = claim_numbers - source_numbers
    if unsupported_numbers:
        return False, "UNVERIFIED_NUMERICAL_CLAIM_REPLACED"

    # Token overlap check (with basic prefix matching for plurals/tenses)
    matched_count = 0
    for ct in claim_tokens:
        if ct in source_tokens:
            matched_count += 1
        elif any(ct.startswith(st) or st.startswith(ct) for st in source_tokens if min(len(st), len(ct)) >= 4):
            matched_count += 1

    overlap_ratio = matched_count / len(claim_tokens)
    if overlap_ratio < 0.35:
        return False, "UNVERIFIED_LOW_OVERLAP_REPLACED"

    return True, "GROUNDED"


def sanitize_single_recommendation(action: str) -> str:
    """Reframe autonomous treatment directives into decision support guidance."""
    cleaned = action.strip()
    if not cleaned:
        return cleaned

    # Direct prescriptive medication or procedural commands:
    # "Administer drug X", "Give 500ml saline", "Infuse dopamine", "Inject epinephrine", "Prescribe antibiotic"
    direct_rx = re.match(
        r"^\s*(administer|give|infuse|inject|prescribe)\s+(.+)",
        cleaned,
        re.IGNORECASE,
    )
    if direct_rx:
        target = direct_rx.group(2).rstrip(".")
        return f"Consider clinician evaluation for {target} as clinically indicated."

    # "Start..." or "Initiate..." or "Begin..." of medications/treatments/protocols
    start_rx = re.match(
        r"^\s*(start|initiate|begin)\s+(.+)",
        cleaned,
        re.IGNORECASE,
    )
    if start_rx:
        rest = start_rx.group(2).strip()
        rest_lower = rest.lower()
        safe_words = (
            "monitoring",
            "surveillance",
            "assessment",
            "evaluation",
            "re-assessment",
            "observation",
            "telemetry",
            "check",
            "review",
            "verification",
            "investigation",
            "tracking",
        )
        if any(w in rest_lower for w in safe_words):
            return cleaned
        target = rest.rstrip(".")
        return f"Consider clinician evaluation for initiating {target} as clinically indicated."

    return cleaned


def sanitize_clinical_recommendations(actions: list[str]) -> list[str]:
    """Sanitize and reframe clinical recommendations into decision support.

    Ensures the Clinical Decision Support System never issues autonomous
    medication or prescriptive treatment orders.
    """
    sanitized: list[str] = []
    for a in actions:
        if not isinstance(a, str):
            continue
        cleaned = sanitize_single_recommendation(a)
        if cleaned and cleaned not in sanitized:
            sanitized.append(cleaned)
    return sanitized


class GeminiClinicalReasoner(BaseGeminiReasoner):
    """Interprets structured evidence using Google Gemini with strict schema validation."""

    def reason(
        self,
        evidence_package: dict[str, Any],
        retrieval_result: RetrievalResult | None = None,
    ) -> LLMReasoningResult:
        """Invoke Gemini on the evidence package and validate structured output."""
        prompt = build_reasoning_prompt(evidence_package, retrieval_result=retrieval_result)
        raw_text = self.call_gemini(
            prompt=prompt,
            system_instruction=CLINICAL_REASONING_SYSTEM_INSTRUCTION,
        )
        return self.validate_and_parse(raw_text, retrieval_result=retrieval_result)


    def validate_and_parse(
        self,
        raw_text: str,
        retrieval_result: RetrievalResult | None = None,
    ) -> LLMReasoningResult:
        """Validate that raw text conforms to structured output requirements."""
        cleaned = _clean_json_text(raw_text)
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise LLMSchemaValidationError(f"Malformed JSON in Gemini response: {exc}") from exc

        if not isinstance(payload, dict):
            raise LLMSchemaValidationError("Expected JSON object at root of Gemini response.")

        # Compatibility bridging between clinical_summary and executive_summary
        if "clinical_summary" not in payload and "executive_summary" in payload:
            payload["clinical_summary"] = payload["executive_summary"]
        if "executive_summary" not in payload and "clinical_summary" in payload:
            payload["executive_summary"] = payload["clinical_summary"]

        # Required fields check
        required_keys = [
            "clinical_summary",
            "supporting_evidence",
            "conflicting_evidence",
            "key_findings",
            "risk_interpretation",
            "priority",
            "recommended_actions",
            "confidence",
        ]
        missing = [k for k in required_keys if k not in payload]
        if missing:
            raise LLMSchemaValidationError(f"Missing required fields in Gemini response: {missing}")

        # Field types and bounds verification
        summary = str(payload["clinical_summary"]).strip()
        if not summary:
            raise LLMSchemaValidationError("clinical_summary cannot be empty.")

        exec_summary = str(payload.get("executive_summary", summary)).strip() or summary

        priority_raw = str(payload["priority"]).strip().upper()
        valid_priorities = {p.value for p in ClinicalReasoningPriority}
        if priority_raw not in valid_priorities:
            raise LLMSchemaValidationError(
                f"Invalid priority '{priority_raw}'. Must be one of: {sorted(valid_priorities)}"
            )

        try:
            confidence = float(payload["confidence"])
            if not (0.0 <= confidence <= 1.0):
                raise ValueError()
        except (TypeError, ValueError):
            raise LLMSchemaValidationError("confidence must be a float between 0.0 and 1.0.")

        def _to_list_of_str(val: Any, field_name: str) -> list[str]:
            if not isinstance(val, list):
                raise LLMSchemaValidationError(f"'{field_name}' must be a list of strings.")
            return [str(item).strip() for item in val if str(item).strip()]

        supporting_evidence = _to_list_of_str(payload["supporting_evidence"], "supporting_evidence")
        conflicting_evidence = _to_list_of_str(payload["conflicting_evidence"], "conflicting_evidence")
        key_findings = _to_list_of_str(payload["key_findings"], "key_findings")
        raw_recommended_actions = _to_list_of_str(payload["recommended_actions"], "recommended_actions")
        # Sanitize recommended actions to strictly preserve decision-support boundaries
        recommended_actions = sanitize_clinical_recommendations(raw_recommended_actions)
        uncertainties = _to_list_of_str(payload.get("uncertainties", []), "uncertainties")
        risk_interpretation = str(payload["risk_interpretation"]).strip()

        # Rich Clinical Intelligence Report optional/structured fields
        raw_phys = payload.get("physiological_analysis", [])
        physiological_analysis = (
            _to_list_of_str(raw_phys, "physiological_analysis") if isinstance(raw_phys, list) else []
        )

        raw_temp = payload.get("temporal_analysis", [])
        temporal_analysis = (
            _to_list_of_str(raw_temp, "temporal_analysis") if isinstance(raw_temp, list) else []
        )

        evidence_synthesis = str(payload.get("evidence_synthesis", "")).strip() or summary
        clinical_interpretation = str(payload.get("clinical_interpretation", "")).strip() or summary

        raw_mon_pri = payload.get("monitoring_priorities", [])
        monitoring_priorities = (
            _to_list_of_str(raw_mon_pri, "monitoring_priorities") if isinstance(raw_mon_pri, list) else []
        )

        escalation_rationale = str(payload.get("escalation_rationale", "")).strip()

        raw_status = payload.get("clinical_status")
        if isinstance(raw_status, dict):
            clinical_status = {
                "risk_level": str(raw_status.get("risk_level", "LOW RISK")),
                "priority": str(raw_status.get("priority", priority_raw)),
                "confidence": float(raw_status.get("confidence", confidence)),
                "data_reliability": str(raw_status.get("data_reliability", "HIGH")),
                "evidence_consistency": str(raw_status.get("evidence_consistency", "SUPPORTING")),
            }
        else:
            clinical_status = {
                "risk_level": "HIGH RISK" if priority_raw == "URGENT" else "LOW RISK",
                "priority": priority_raw,
                "confidence": confidence,
                "data_reliability": "HIGH",
                "evidence_consistency": "SUPPORTING" if not conflicting_evidence else "CONFLICTING",
            }

        # Handle knowledge sources, medical evidence, and citations with strict grounding
        knowledge_sources: list[dict[str, Any]] = []
        retrieved_evidence: list[str] = []
        medical_evidence: list[dict[str, Any]] = []

        if retrieval_result and retrieval_result.has_evidence:
            retrieval_status = "SUCCESS"

            # Build lookup of verified retrieved passages
            passage_by_id: dict[str, Any] = {}
            for p in retrieval_result.passages:
                pmid = getattr(p, "pmid", "") or (p.metadata.get("pmid", "") if hasattr(p, "metadata") else "")
                if pmid:
                    passage_by_id[pmid.lower()] = p
                passage_by_id[p.document_id.lower()] = p
                passage_by_id[p.title.lower()] = p

            raw_med_evidence = payload.get("medical_evidence", [])
            if isinstance(raw_med_evidence, list) and raw_med_evidence:
                for item in raw_med_evidence:
                    if not isinstance(item, dict):
                        continue
                    pmid_cand = str(item.get("pmid", "")).strip()
                    title_cand = str(item.get("title", "")).strip()

                    # Find matching retrieved passage for strict grounding
                    matched = None
                    if pmid_cand and pmid_cand.lower() in passage_by_id:
                        matched = passage_by_id[pmid_cand.lower()]
                    elif title_cand and title_cand.lower() in passage_by_id:
                        matched = passage_by_id[title_cand.lower()]

                    if matched is not None:
                        pmid_val = getattr(matched, "pmid", "") or matched.metadata.get("pmid", pmid_cand)
                        authors_val = getattr(matched, "authors", "") or matched.metadata.get("authors", item.get("authors", ""))
                        journal_val = getattr(matched, "journal", "") or matched.metadata.get("journal", item.get("journal", ""))
                        year_val = getattr(matched, "year", "") or matched.metadata.get("year", item.get("year", ""))
                        doi_val = getattr(matched, "doi", "") or matched.metadata.get("doi", item.get("doi", ""))
                        rel_score = float(item.get("relevance_score", matched.relevance_score))

                        raw_key_evidence = str(item.get("key_evidence", "")).strip()
                        is_grounded, grounding_status = _validate_claim_grounding(raw_key_evidence, matched.text)
                        if is_grounded:
                            safe_key_evidence = raw_key_evidence
                        else:
                            # Replace unsupported claim with direct excerpt from retrieved source
                            safe_key_evidence = matched.text[:200] + ("..." if len(matched.text) > 200 else "")

                        raw_app = str(item.get("application_to_case", "Applied to contextual bedside monitoring.")).strip()
                        safe_app = sanitize_single_recommendation(raw_app)

                        medical_evidence.append({
                            "pmid": pmid_val,
                            "title": matched.title,
                            "authors": authors_val,
                            "journal": journal_val,
                            "year": year_val,
                            "doi": doi_val,
                            "relevance_score": round(rel_score, 3),
                            "abstract": matched.text,
                            "why_retrieved": str(item.get("why_retrieved", "Retrieved based on physiological trajectory.")),
                            "key_evidence": safe_key_evidence,
                            "grounding_status": grounding_status,
                            "application_to_case": safe_app,
                        })

            # If medical_evidence wasn't populated by LLM or had ungrounded entries, populate from passages
            if not medical_evidence:
                for p in retrieval_result.passages:
                    pmid_val = getattr(p, "pmid", "") or (p.metadata.get("pmid", "") if hasattr(p, "metadata") else "")
                    authors_val = getattr(p, "authors", "") or (p.metadata.get("authors", "") if hasattr(p, "metadata") else "")
                    journal_val = getattr(p, "journal", "") or (p.metadata.get("journal", "") if hasattr(p, "metadata") else "")
                    year_val = getattr(p, "year", "") or (p.metadata.get("year", "") if hasattr(p, "metadata") else "")
                    doi_val = getattr(p, "doi", "") or (p.metadata.get("doi", "") if hasattr(p, "metadata") else "")

                    medical_evidence.append({
                        "pmid": pmid_val,
                        "title": p.title,
                        "authors": authors_val,
                        "journal": journal_val,
                        "year": year_val,
                        "doi": doi_val,
                        "relevance_score": round(p.relevance_score, 3),
                        "abstract": p.text,
                        "why_retrieved": "Authoritative guidance retrieved for case presentation.",
                        "key_evidence": p.text[:200] + ("..." if len(p.text) > 200 else ""),
                        "grounding_status": "GROUNDED",
                        "application_to_case": "Informs bedside vital sign monitoring and threshold evaluation.",
                    })

            # Populate backward-compatible knowledge_sources and retrieved_evidence
            raw_sources = payload.get("knowledge_sources", [])
            if isinstance(raw_sources, list) and raw_sources:
                for s in raw_sources:
                    if isinstance(s, dict) and s.get("document_id"):
                        knowledge_sources.append({
                            "document_id": str(s.get("document_id")),
                            "title": str(s.get("title", "Clinical Guideline")),
                            "section": str(s.get("section", "General")),
                        })
            if not knowledge_sources:
                knowledge_sources = [p.to_citation_dict() for p in retrieval_result.passages]

            retrieved_evidence = _to_list_of_str(payload.get("retrieved_evidence", []), "retrieved_evidence")
            if not retrieved_evidence:
                retrieved_evidence = [p.text[:120] + "..." for p in retrieval_result.passages]
        else:
            retrieval_status = retrieval_result.retrieval_status if retrieval_result else "NO_RELEVANT_EVIDENCE"
            knowledge_sources = []
            retrieved_evidence = []
            medical_evidence = []

        return LLMReasoningResult(
            clinical_summary=summary,
            supporting_evidence=supporting_evidence,
            conflicting_evidence=conflicting_evidence,
            key_findings=key_findings,
            risk_interpretation=risk_interpretation,
            priority=priority_raw,
            recommended_actions=recommended_actions,
            confidence=confidence,
            uncertainties=uncertainties,
            knowledge_sources=knowledge_sources,
            retrieved_evidence=retrieved_evidence,
            retrieval_status=retrieval_status,
            raw_response=raw_text,
            executive_summary=exec_summary,
            clinical_status=clinical_status,
            physiological_analysis=physiological_analysis,
            temporal_analysis=temporal_analysis,
            evidence_synthesis=evidence_synthesis,
            medical_evidence=medical_evidence,
            clinical_interpretation=clinical_interpretation,
            monitoring_priorities=monitoring_priorities,
            escalation_rationale=escalation_rationale,
        )

    def select_tool(
        self,
        case_id: int,
        event_id: str,
        risk_level: str,
        affected_vitals: list[str],
        executed_tools: list[str],
        state_summary: dict[str, Any],
        iteration: int,
        max_iterations: int = 5,
    ) -> ClinicalReasoningToolDecision:
        """Ask Gemini which clinical reasoning tool to execute next or whether information is sufficient."""
        prompt = build_clinical_tool_selection_prompt(
            case_id=case_id,
            event_id=event_id,
            risk_level=risk_level,
            affected_vitals=affected_vitals,
            executed_tools=executed_tools,
            state_summary=state_summary,
            iteration=iteration,
            max_iterations=max_iterations,
        )

        raw_text = self.call_gemini(
            prompt=prompt,
            system_instruction=CLINICAL_TOOL_SYSTEM_INSTRUCTION,
        )
        return self.validate_and_parse_tool_selection(raw_text)

    def validate_and_parse_tool_selection(self, raw_text: str) -> ClinicalReasoningToolDecision:
        """Validate and parse Gemini response into ClinicalReasoningToolDecision."""
        cleaned = _clean_json_text(raw_text)
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise LLMSchemaValidationError(f"Malformed JSON in Gemini clinical tool selection response: {exc}") from exc

        if not isinstance(payload, dict):
            raise LLMSchemaValidationError("Expected JSON object at root of Gemini tool selection response.")

        # If model returned a full clinical reasoning report directly (e.g. from an existing test or prompt),
        # accept it as an immediate finish decision
        if "clinical_summary" in payload or "executive_summary" in payload:
            thought = str(payload.get("clinical_summary", payload.get("thought", ""))).strip()
            return ClinicalReasoningToolDecision(
                thought=thought,
                action="finish",
                tool_name=None,
                tool_args={},
                rationale="Clinical assessment complete.",
                confidence=float(payload.get("confidence", 1.0)),
                raw_response=raw_text,
            )

        thought = str(payload.get("thought", "")).strip()
        action = str(payload.get("action", "finish")).strip().lower()
        if action not in ("tool", "finish"):
            action = "finish"

        tool_name = payload.get("tool_name")
        if tool_name:
            tool_name = str(tool_name).strip()
        tool_args = payload.get("tool_args", {})
        if not isinstance(tool_args, dict):
            tool_args = {}

        rationale = str(payload.get("rationale", "")).strip()
        try:
            confidence = float(payload.get("confidence", 1.0))
            if not (0.0 <= confidence <= 1.0):
                confidence = 1.0
        except (TypeError, ValueError):
            confidence = 1.0

        return ClinicalReasoningToolDecision(
            thought=thought,
            action=action,
            tool_name=tool_name,
            tool_args=tool_args,
            rationale=rationale,
            confidence=confidence,
            raw_response=raw_text,
        )


CLINICAL_TOOL_SYSTEM_INSTRUCTION = """You are an autonomous Clinical Reasoning Agent for CareMatrix.
Your goal is to inspect incoming patient analytical evidence, identify what clinical knowledge, patient history, or evidence synthesis is required, and decide which clinical tools to invoke.

Available Tools:
- get_patient_context: Retrieve historical episodic memory and past clinician feedback (acknowledgments, resolutions, overrides) for the patient.
- build_focused_medical_query: Construct a de-identified clinical query focusing on vital trends, risk status, and patterns for literature search.
- retrieve_medical_evidence: Retrieve authoritative evidence-based medical literature passages from PubMed and local curated knowledge store using RAG.
- evaluate_clinical_evidence: Synthesize risk score, vital trends, patterns, and data quality into explainable non-diagnostic clinical recommendations.
- arbitrate_clinical_reasoning: Arbitrate between deterministic safety baseline and LLM proposal ensuring strict safety invariants cannot be downgraded.

You must select EXACTLY ONE tool to run next, OR decide that sufficient information has been gathered to reason and finish ("finish").
DO NOT repeatedly call the same tool unless necessary.

You must respond ONLY with a valid JSON object matching this schema:
{
  "thought": "Clinical reasoning explaining what clinical information is needed next and why",
  "action": "tool" | "finish",
  "tool_name": "get_patient_context" | "build_focused_medical_query" | "retrieve_medical_evidence" | "evaluate_clinical_evidence" | "arbitrate_clinical_reasoning" | null,
  "tool_args": {},
  "rationale": "Summary rationale explaining clinical findings or readiness to finish",
  "confidence": 0.0 to 1.0
}
"""


@dataclass
class ClinicalReasoningToolDecision:
    """Structured decision returned by Gemini in the clinical reasoning tool-selection loop."""

    thought: str
    action: str  # "tool" | "finish"
    tool_name: str | None = None
    tool_args: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    confidence: float = 1.0
    raw_response: str = ""


def build_clinical_tool_selection_prompt(
    case_id: int,
    event_id: str,
    risk_level: str,
    affected_vitals: list[str],
    executed_tools: list[str],
    state_summary: dict[str, Any],
    iteration: int,
    max_iterations: int,
) -> str:
    """Build prompt providing clinical event context and history for LLM tool selection."""
    executed_desc = ", ".join(executed_tools) if executed_tools else "None"

    return f"""CLINICAL REASONING AGENT CYCLE (Iteration {iteration + 1} of max {max_iterations}):
Patient / Case ID: {case_id}
Target Event ID: {event_id}
Risk Assessment: {risk_level}
Affected Vitals: {affected_vitals}

Current Clinical State Summary:
- Tools Already Executed: {executed_desc}
- Patient Episodic Memory Retrieved: {state_summary.get('patient_context_retrieved', False)}
- Patient Prior Assessments: {state_summary.get('prior_assessments_count', 0)}
- Medical Query Formulated: {state_summary.get('medical_query_built', False)}
- Current Medical Query: {json.dumps(state_summary.get('current_query', ''))}
- Medical Literature Retrieved: {state_summary.get('evidence_retrieved', False)}
- Retrieved Passages Count: {state_summary.get('retrieved_passages_count', 0)}
- Clinical Evidence Evaluated: {state_summary.get('evidence_evaluated', False)}
- Deterministic Evaluation Baseline Available: {state_summary.get('has_deterministic_baseline', False)}
- Vitals and Patterns Glance: {json.dumps(state_summary.get('findings_glance', []))}

TASK:
Determine what clinical tool should be executed next to gather needed patient context, medical literature, or synthesis,
OR choose "finish" if sufficient clinical evidence has been gathered to complete the report.
Respond strictly in the specified JSON format.
"""


__all__ = [
    "CLINICAL_TOOL_SYSTEM_INSTRUCTION",
    "ClinicalReasoningToolDecision",
    "GeminiClinicalReasoner",
    "LLMReasonerError",
    "LLMSchemaValidationError",
    "LLMTimeoutError",
    "LLMUnavailableError",
    "build_clinical_tool_selection_prompt",
    "sanitize_single_recommendation",
    "sanitize_clinical_recommendations",
    "_validate_claim_grounding",
]

