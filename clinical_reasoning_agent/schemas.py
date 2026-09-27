"""Schemas and event definitions for the Clinical Reasoning Agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from communication.events import ClinicalReasoningEvent, DataAnalysisEvent


class ClinicalReasoningPriority(str, Enum):
    """Clinical urgency and prioritization levels for downstream care teams."""

    ROUTINE = "ROUTINE"
    ELEVATED = "ELEVATED"
    URGENT = "URGENT"


class ReasoningMode(str, Enum):
    """Execution mode of the clinical reasoning synthesis."""

    DETERMINISTIC = "DETERMINISTIC"
    LLM_ASSISTED = "LLM_ASSISTED"
    LLM_RAG = "LLM_RAG"
    LLM_FALLBACK = "LLM_FALLBACK"


@dataclass
class LLMReasoningResult:
    """Validated structured output produced by the LLM reasoning capability."""

    clinical_summary: str
    supporting_evidence: list[str]
    conflicting_evidence: list[str]
    key_findings: list[str]
    risk_interpretation: str
    priority: str
    recommended_actions: list[str]
    confidence: float
    uncertainties: list[str] = field(default_factory=list)
    knowledge_sources: list[dict[str, Any]] = field(default_factory=list)
    retrieved_evidence: list[str] = field(default_factory=list)
    retrieval_status: str = "NO_RELEVANT_EVIDENCE"
    raw_response: str | None = None


__all__ = [
    "ClinicalReasoningEvent",
    "ClinicalReasoningPriority",
    "DataAnalysisEvent",
    "LLMReasoningResult",
    "ReasoningMode",
]


