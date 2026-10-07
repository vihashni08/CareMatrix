"""Standardized tools for CareMatrix Clinical Reasoning Agent.

Wraps existing functions:
- clinical evidence retrieval (MedicalRetriever / PubMed + local curated RAG)
- patient-context retrieval (PatientMemory episodic summaries and history)
- focused medical query builder
- deterministic clinical reasoning engine synthesis & safety arbitration
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from carematrix_runtime.patient_memory import EpisodeSummary, PatientMemory
from carematrix_runtime.tools.base import Tool, ToolRegistry
from clinical_reasoning_agent.rag.retriever import MedicalRetriever, build_focused_medical_query
from clinical_reasoning_agent.rag.schemas import RetrievalResult
from clinical_reasoning_agent.reasoning import ClinicalReasoningEngine, ClinicalReasoningOutput
from clinical_reasoning_agent.schemas import LLMReasoningResult
from clinical_reasoning_agent.state import ClinicalReasoningState
from communication.events import DataAnalysisEvent


def create_clinical_reasoning_tools(
    retriever: Optional[MedicalRetriever] = None,
    patient_memory: Optional[PatientMemory] = None,
    engine: Optional[ClinicalReasoningEngine] = None,
) -> ToolRegistry:
    """Create and register all clinical reasoning agent tools."""
    registry = ToolRegistry(name="clinical_reasoning")

    active_retriever = retriever or MedicalRetriever()
    active_engine = engine or ClinicalReasoningEngine()

    # 1. Focused medical query construction tool
    def _build_focused_query(event: DataAnalysisEvent) -> str:
        return build_focused_medical_query(event)

    registry.register(
        Tool(
            name="build_focused_medical_query",
            description="Construct de-identified clinical query focusing on vital trends, risk status, and patterns for literature search.",
            func=_build_focused_query,
            parameters={
                "type": "object",
                "properties": {
                    "event": {
                        "type": "object",
                        "description": "DataAnalysisEvent containing risk level, trend metrics, and patterns",
                    }
                },
                "required": ["event"],
            },
            output_schema={
                "type": "string",
                "description": "Clinical search query string",
            },
            tags=["clinical_reasoning", "rag"],
        )
    )

    # 2. Clinical evidence retrieval (RAG retrieval)
    def _retrieve_medical_evidence(
        event_or_query: Any,
        top_k: Optional[int] = None,
    ) -> RetrievalResult:
        return active_retriever.retrieve(event_or_query, top_k=top_k)

    registry.register(
        Tool(
            name="retrieve_medical_evidence",
            description="Retrieve authoritative evidence-based medical literature passages from PubMed and local curated knowledge store.",
            func=_retrieve_medical_evidence,
            parameters={
                "type": "object",
                "properties": {
                    "event_or_query": {
                        "type": "string",
                        "description": "Clinical query string or DataAnalysisEvent",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Maximum number of literature passages to retrieve (default: 2)",
                    },
                },
                "required": ["event_or_query"],
            },
            output_schema={
                "type": "object",
                "description": "RetrievalResult containing ranked RetrievedPassage objects",
            },
            tags=["clinical_reasoning", "rag"],
        )
    )

    # 3. Patient-context retrieval from PatientMemory
    def _get_patient_context(patient_id: int, limit: int = 5) -> Dict[str, Any]:
        if patient_memory is None:
            return {
                "patient_id": patient_id,
                "episodes": [],
                "formatted_prompt": "No episodic memory configured for patient.",
            }
        episodes: List[EpisodeSummary] = patient_memory.get_episodes(patient_id, limit=limit)
        summaries = [e.to_dict() for e in episodes]
        formatted = patient_memory.format_memory_for_prompt(patient_id, limit=limit)
        return {
            "patient_id": patient_id,
            "episodes": summaries,
            "formatted_prompt": formatted,
        }

    registry.register(
        Tool(
            name="get_patient_context",
            description="Retrieve historical episodic memory and past clinician feedback (acknowledgments, resolutions, overrides) for a patient.",
            func=_get_patient_context,
            parameters={
                "type": "object",
                "properties": {
                    "patient_id": {
                        "type": "integer",
                        "description": "Patient identifier",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of recent episodes to retrieve (default: 5)",
                    },
                },
                "required": ["patient_id"],
            },
            output_schema={
                "type": "object",
                "description": "Dictionary containing episode records and formatted clinical context",
            },
            tags=["clinical_reasoning", "memory"],
        )
    )

    # 4. Clinical reasoning evaluation (synthesis)
    def _evaluate_clinical_evidence(
        event: DataAnalysisEvent,
        state: Optional[ClinicalReasoningState] = None,
    ) -> ClinicalReasoningOutput:
        return active_engine.evaluate(event, state=state)

    registry.register(
        Tool(
            name="evaluate_clinical_evidence",
            description="Synthesize risk score, vital trends, patterns, and data quality into explainable non-diagnostic clinical recommendations.",
            func=_evaluate_clinical_evidence,
            parameters={
                "type": "object",
                "properties": {
                    "event": {
                        "type": "object",
                        "description": "DataAnalysisEvent containing analytical evidence",
                    },
                    "state": {
                        "type": "object",
                        "description": "Optional patient ClinicalReasoningState tracking episode histories",
                    },
                },
                "required": ["event"],
            },
            output_schema={
                "type": "object",
                "description": "Structured ClinicalReasoningOutput",
            },
            tags=["clinical_reasoning", "synthesis"],
        )
    )

    # 5. Deterministic safety arbitration
    def _arbitrate_clinical_reasoning(
        deterministic: ClinicalReasoningOutput,
        llm_result: Optional[LLMReasoningResult],
        risk_level: str,
        data_quality_flag: bool = False,
        fallback_error: Optional[str] = None,
    ) -> ClinicalReasoningOutput:
        return active_engine.arbitrate(
            deterministic=deterministic,
            llm_result=llm_result,
            risk_level=risk_level,
            data_quality_flag=data_quality_flag,
            fallback_error=fallback_error,
        )

    registry.register(
        Tool(
            name="arbitrate_clinical_reasoning",
            description="Arbitrate between deterministic safety baseline and LLM proposal ensuring strict safety invariants cannot be downgraded.",
            func=_arbitrate_clinical_reasoning,
            parameters={
                "type": "object",
                "properties": {
                    "deterministic": {
                        "type": "object",
                        "description": "ClinicalReasoningOutput from deterministic evaluation",
                    },
                    "llm_result": {
                        "type": "object",
                        "description": "LLMReasoningResult from LLM generation, or None",
                    },
                    "risk_level": {
                        "type": "string",
                        "description": "Risk classification ('HIGH RISK' or 'LOW RISK')",
                    },
                    "data_quality_flag": {
                        "type": "boolean",
                        "description": "True if data quality issues are flagged",
                    },
                    "fallback_error": {
                        "type": "string",
                        "description": "Optional error string if LLM failed",
                    },
                },
                "required": ["deterministic", "risk_level"],
            },
            output_schema={
                "type": "object",
                "description": "Arbitrated final ClinicalReasoningOutput",
            },
            tags=["clinical_reasoning", "arbitration"],
        )
    )

    return registry

