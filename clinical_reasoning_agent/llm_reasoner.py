"""LLM reasoning tool for the Clinical Reasoning Agent using Google Gemini.

Interprets and synthesizes structured analytical evidence packages into validated,
explainable clinical decision support outputs using the Gemini API.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
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
        recommended_actions = _to_list_of_str(payload["recommended_actions"], "recommended_actions")
        uncertainties = _to_list_of_str(payload.get("uncertainties", []), "uncertainties")
        risk_interpretation = str(payload["risk_interpretation"]).strip()

        # Handle knowledge sources and citations
        knowledge_sources: list[dict[str, Any]] = []
        retrieved_evidence: list[str] = []

        if retrieval_result and retrieval_result.has_evidence:
            retrieval_status = "SUCCESS"
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
                # Populate authoritative citations directly from retrieval passages
                knowledge_sources = [p.to_citation_dict() for p in retrieval_result.passages]

            retrieved_evidence = _to_list_of_str(payload.get("retrieved_evidence", []), "retrieved_evidence")
            if not retrieved_evidence:
                retrieved_evidence = [p.text[:120] + "..." for p in retrieval_result.passages]
        else:
            retrieval_status = retrieval_result.retrieval_status if retrieval_result else "NO_RELEVANT_EVIDENCE"
            knowledge_sources = []
            retrieved_evidence = []

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
        )


__all__ = [
    "GeminiClinicalReasoner",
    "LLMReasonerError",
    "LLMSchemaValidationError",
    "LLMTimeoutError",
    "LLMUnavailableError",
]

