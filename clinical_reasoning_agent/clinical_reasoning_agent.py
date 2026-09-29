"""CareMatrix Clinical Reasoning Agent controller with autonomous synthesis cycle.

Lifecycle:
RECEIVE DATA ANALYSIS EVIDENCE → REASON (SYNTHESIS) → DECIDE → ACT (UPDATE STATE) → COMMUNICATE
"""

from __future__ import annotations

import datetime
import threading
import time
from typing import Any

from communication.event_queue import EventQueue
from communication.events import (
    AgentHeartbeatEvent,
    ClinicalReasoningEvent,
    DataAnalysisEvent,
    PerformativeType,
)
from communication.orchestration import is_high_risk_case
from clinical_reasoning_agent.llm_reasoner import (
    GeminiClinicalReasoner,
    LLMReasonerError,
)
from clinical_reasoning_agent.prompt_builder import (
    build_evidence_package,
    build_reasoning_prompt,
)
from clinical_reasoning_agent.rag import MedicalRetriever
from clinical_reasoning_agent.reasoning import ClinicalReasoningEngine
from clinical_reasoning_agent.schemas import (
    ClinicalReasoningPriority,
    ReasoningMode,
)
from clinical_reasoning_agent.state import ClinicalReasoningState

RETRIEVAL_CONFIDENCE_THRESHOLD: float = 0.80


def _format_log(name: str, action: str, details: str) -> str:
    now = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    return f"[{now}] {name:<24} | {action:<10} | {details}"


class ClinicalReasoningAgent:
    """Autonomous Clinical Reasoning Agent controller.

    Consumes analytical evidence from DataAnalysisAgent, retrieves evidence-grounded
    medical guidance, executes LLM-assisted clinical reasoning with deterministic safety
    arbitration, prioritizes bedside monitoring actions, and publishes validated decisions.
    """

    def __init__(
        self,
        event_queue: EventQueue | None = None,
        name: str = "ClinicalReasoningAgent",
        verbose: bool = False,
        llm_reasoner: GeminiClinicalReasoner | None = None,
        retriever: MedicalRetriever | None = None,
        enable_llm: bool = True,
        enable_rag: bool = True,
        patient_memory: Any | None = None,
        retrieval_confidence_threshold: float = RETRIEVAL_CONFIDENCE_THRESHOLD,
    ):
        self.name = name
        self.event_queue = event_queue
        self.verbose = verbose
        self.enable_llm = enable_llm
        self.enable_rag = enable_rag
        self.patient_memory = patient_memory
        self.retrieval_confidence_threshold = float(retrieval_confidence_threshold)
        self.llm_reasoner = llm_reasoner or GeminiClinicalReasoner()
        self.retriever = retriever if retriever is not None else MedicalRetriever()
        self.reasoning_engine = ClinicalReasoningEngine()
        self.patient_states: dict[int, ClinicalReasoningState] = {}
        self.is_healthy: bool = True

        # Independent worker thread management
        self._worker_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._is_running: bool = False

    def emit_heartbeat(self) -> AgentHeartbeatEvent:
        """Publish heartbeat to supervisor."""
        hb = AgentHeartbeatEvent(
            agent_name=self.name,
            status="healthy" if self.is_healthy else "degraded",
            metrics={
                "monitored_patients": len(self.patient_states),
                "llm_available": self.llm_reasoner.is_available,
                "rag_available": getattr(self.retriever, "_is_initialized", False),
            },
        )
        if self.event_queue is not None:
            self.event_queue.publish("heartbeats", hb)
        return hb

    # ------------------------------------------------------------------------
    # Autonomous Event Processing: receive → gather evidence → retrieve RAG → reason → validate → act → communicate
    # ------------------------------------------------------------------------
    def process_event(self, event: DataAnalysisEvent) -> ClinicalReasoningEvent:
        """Execute one autonomous clinical reasoning cycle."""
        t_start = time.perf_counter()
        case_id = int(getattr(event, "case_id", 0) or 0)
        event_id = str(getattr(event, "event_id", "unknown_analysis_event"))
        timestamp = float(getattr(event, "timestamp", 0.0) or 0.0)
        risk_level = str(getattr(event, "risk_level", "INDETERMINATE"))
        quality_flag = bool(getattr(event, "data_quality_flag", False))

        prev_cumulative = (
            float((event.metadata or {}).get("cumulative_latency_ms", 0.0) or 0.0)
            if hasattr(event, "metadata") and isinstance(event.metadata, dict)
            else 0.0
        )

        # 1. RECEIVE & HEARTBEAT
        state = self.patient_states.setdefault(case_id, ClinicalReasoningState(case_id=case_id))
        self.emit_heartbeat()
        if self.verbose:
            print(_format_log(self.name, "RECEIVE", f"event={event_id} (Case: {case_id}, Risk: {risk_level})"))

        try:
            if not isinstance(event, DataAnalysisEvent):
                raise TypeError("Expected DataAnalysisEvent")

            # 2. GATHER EVIDENCE & REASON (DETERMINISTIC SAFETY BASELINE)
            deterministic_out = self.reasoning_engine.evaluate(event=event, state=state)

            # 3. MEDICAL RAG: Retrieve Relevant Authoritative Guidance (Risk-Gated)
            retrieval_result = None
            evidence_retrieval_status = "SKIPPED_LOW_RISK_CONFIDENT"
            evidence_retrieval_reason = ""

            # Fail-safe defaulting for missing/empty fields:
            # - Missing/empty risk_level -> is_high_risk_case treats as True
            # - Missing evidence_consistency -> "UNCERTAIN"
            # - Missing confidence -> 0.0
            consistency_raw = getattr(event, "evidence_consistency", None)
            if consistency_raw is None and hasattr(event, "metadata") and isinstance(event.metadata, dict):
                consistency_raw = event.metadata.get("evidence_consistency")
            consistency = str(consistency_raw).strip().upper() if consistency_raw is not None and str(consistency_raw).strip() != "" else "UNCERTAIN"

            confidence_raw = getattr(event, "confidence", None)
            if confidence_raw is None and hasattr(event, "metadata") and isinstance(event.metadata, dict):
                confidence_raw = event.metadata.get("confidence")
            try:
                confidence = float(confidence_raw) if confidence_raw is not None else 0.0
            except (ValueError, TypeError):
                confidence = 0.0

            prob_raw = getattr(event, "probability", None)
            if prob_raw is None and hasattr(event, "metadata") and isinstance(event.metadata, dict):
                prob_raw = event.metadata.get("probability")
            try:
                prob = float(prob_raw) if prob_raw is not None else None
            except (ValueError, TypeError):
                prob = None

            sev_raw = getattr(event, "severity", None)
            if sev_raw is None and hasattr(event, "metadata") and isinstance(event.metadata, dict):
                sev_raw = event.metadata.get("severity")

            high_risk = is_high_risk_case(decision=risk_level, probability=prob, severity=sev_raw)

            if high_risk:
                should_retrieve = True
                evidence_retrieval_status = "RUN_HIGH_RISK"
                evidence_retrieval_reason = f"High-risk clinical trajectory (risk={risk_level}): authoritative guidance required."
            elif consistency in ("CONFLICTING", "UNCERTAIN"):
                should_retrieve = True
                evidence_retrieval_status = "RUN_CONFIDENCE_ESCALATION"
                evidence_retrieval_reason = f"Low-risk case escalated due to {consistency} cross-agent evidence alignment."
            elif confidence < self.retrieval_confidence_threshold:
                should_retrieve = True
                evidence_retrieval_status = "RUN_CONFIDENCE_ESCALATION"
                evidence_retrieval_reason = (
                    f"Low-risk case escalated due to low cross-agent confidence "
                    f"({confidence:.2f} < {self.retrieval_confidence_threshold:.2f})."
                )
            else:
                should_retrieve = False
                evidence_retrieval_status = "SKIPPED_LOW_RISK_CONFIDENT"
                evidence_retrieval_reason = (
                    f"Low-risk case with confident, supporting consistency ({confidence:.2f}): RAG skipped."
                )

            if should_retrieve and self.enable_rag and self.retriever is not None:
                try:
                    retrieval_result = self.retriever.retrieve(event)
                    if self.verbose and retrieval_result.has_evidence:
                        print(_format_log(self.name, "RAG_RETRIEVE", f"Found {len(retrieval_result.passages)} guidelines (score: {retrieval_result.top_score:.2f}) [{evidence_retrieval_status}]"))
                except Exception as rag_exc:
                    if self.verbose:
                        print(_format_log(self.name, "RAG_ERROR", f"Retrieval failed: {rag_exc}"))
                    retrieval_result = None
            elif not should_retrieve:
                if self.verbose:
                    print(_format_log(self.name, "RAG_SKIPPED", evidence_retrieval_reason))

            # 4. INVOKE REASONING CAPABILITY (LLM TOOL WITH RAG CONTEXT)
            llm_result = None
            fallback_error = None

            evidence_package = build_evidence_package(event, state, patient_memory=self.patient_memory)
            prompt_len = len(build_reasoning_prompt(evidence_package, retrieval_result))
            if retrieval_result and retrieval_result.has_evidence:
                prompt_diff = prompt_len - len(build_reasoning_prompt(evidence_package, None))
            else:
                prompt_diff = 0

            if self.enable_llm and self.llm_reasoner.is_available:
                try:
                    if self.verbose:
                        print(_format_log(self.name, "LLM_INVOKE", f"Invoking Gemini reasoner for {event_id}"))
                    llm_result = self.llm_reasoner.reason(evidence_package, retrieval_result=retrieval_result)
                    if self.verbose:
                        print(_format_log(self.name, "LLM_SUCCESS", f"Gemini reasoning completed: priority={llm_result.priority}"))
                except Exception as exc:
                    llm_result = None
                    fallback_error = str(exc)
                    if self.verbose:
                        print(_format_log(self.name, "FALLBACK", f"LLM error: {exc}. Using deterministic safety baseline."))

            # 5. VALIDATE & ARBITRATE (DETERMINISTIC SAFETY ARBITER)
            out = self.reasoning_engine.arbitrate(
                deterministic=deterministic_out,
                llm_result=llm_result,
                risk_level=risk_level,
                data_quality_flag=quality_flag,
                fallback_error=fallback_error,
            )

            # 5b. Extract cross-agent negotiation trace
            input_meta = getattr(event, "metadata", {}) or {}
            negotiation_trace = input_meta.get("negotiation_trace")
            if not negotiation_trace and self.event_queue is not None:
                # Discover negotiation history across event queue
                history = self.event_queue.get_history()
                exchange_events = [
                    ev for ev in history
                    if getattr(ev, "event_id", "") == event_id
                    and getattr(ev, "performative", "") in (
                        PerformativeType.CHALLENGE.value,
                        PerformativeType.AGREE.value,
                        PerformativeType.PROPOSE.value,
                    )
                ]
                if exchange_events:
                    dialogue = []
                    for ev in exchange_events:
                        perf = getattr(ev, "performative", "")
                        source = getattr(ev, "source", None) or ("RiskAgent" if hasattr(ev, "decision") else "DataAnalysisAgent")
                        stmt = getattr(ev, "reason", "") or (getattr(ev, "metadata", {}).get("conflicting_evidence", [""])[0] if getattr(ev, "metadata", {}) else "")
                        dialogue.append({
                            "agent": source,
                            "performative": perf,
                            "round": getattr(ev, "challenge_round", 1),
                            "statement": stmt,
                            "decision": getattr(ev, "decision", getattr(ev, "risk_level", "")),
                        })
                    last_resp = exchange_events[-1]
                    res_summary = (
                        f"Resolved after {len(exchange_events)} negotiation steps: "
                        f"{getattr(last_resp, 'performative', '')} by {getattr(last_resp, 'source', 'RiskAgent')} "
                        f"({getattr(last_resp, 'decision', '')})."
                    )
                    negotiation_trace = {
                        "challenge_round": max([getattr(ev, "challenge_round", 1) for ev in exchange_events], default=1),
                        "status": "RESOLVED",
                        "dialogue": dialogue,
                        "resolution_summary": res_summary,
                    }

            findings = list(out.findings)
            if negotiation_trace:
                findings.append(
                    f"Cross-Agent Challenge Resolution: {negotiation_trace.get('resolution_summary', '')}"
                )

            result = ClinicalReasoningEvent(
                case_id=case_id,
                event_id=event_id,
                timestamp=timestamp,
                risk_level=risk_level,
                priority=out.priority,
                clinical_summary=out.clinical_summary,
                findings=findings,
                recommended_actions=out.recommended_actions,
                escalation_required=out.escalation_required,
                data_reliability=out.data_reliability,
                confidence=out.confidence,
                evidence_consistency=out.evidence_consistency,
                supporting_evidence=out.supporting_evidence,
                conflicting_evidence=out.conflicting_evidence,
                verification_required=out.verification_required,
                metadata={
                    **out.metadata,
                    "clinical_status": out.clinical_status,
                    "reasoning_mode": out.reasoning_mode,
                    "llm_reasoning_conflict": out.llm_reasoning_conflict,
                    "knowledge_sources": out.knowledge_sources,
                    "retrieved_evidence": out.retrieved_evidence,
                    "retrieval_status": out.retrieval_status,
                    "evidence_retrieval": evidence_retrieval_status,
                    "evidence_retrieval_reason": evidence_retrieval_reason,
                    "prompt_length_chars": prompt_len,
                    "prompt_diff_chars": prompt_diff,
                    **({"negotiation_trace": negotiation_trace} if negotiation_trace else {}),
                },
                performative=PerformativeType.INFORM.value,
                challenge_round=getattr(event, "challenge_round", 0),
            )

            # 5. ACT: Update Patient Clinical State
            state.record_assessment(
                event_id=event_id,
                timestamp=timestamp,
                priority=out.priority,
                risk_level=risk_level,
                summary=out.clinical_summary,
                findings=out.findings,
                recommendations=out.recommended_actions,
            )

        except Exception as exc:
            # ERROR RECOVERY: Safe fallback event
            priority = ClinicalReasoningPriority.ELEVATED.value
            summary = f"Degraded clinical evaluation for Case {case_id}: Analysis input error ({exc})."
            actions = [
                "Verify automated agent pipeline communication and sensor integrity.",
                "Conduct manual bedside vital signs assessment.",
            ]
            result = ClinicalReasoningEvent(
                case_id=case_id,
                event_id=event_id,
                timestamp=timestamp,
                risk_level=risk_level,
                priority=priority,
                clinical_summary=summary,
                findings=[f"Input error during clinical reasoning: {exc}"],
                recommended_actions=actions,
                escalation_required=False,
                data_reliability="COMPROMISED",
                confidence=0.30,
                evidence_consistency="UNCERTAIN",
                supporting_evidence=[],
                conflicting_evidence=[f"Input error during clinical reasoning: {exc}"],
                verification_required=True,
                metadata={
                    "error": str(exc),
                    "reasoning_mode": ReasoningMode.LLM_FALLBACK.value,
                    "evidence_retrieval": evidence_retrieval_status if "evidence_retrieval_status" in locals() else "RUN_HIGH_RISK",
                    "evidence_retrieval_reason": evidence_retrieval_reason if "evidence_retrieval_reason" in locals() else str(exc),
                },
            )
            state.record_assessment(event_id, timestamp, priority, risk_level, summary, result.findings, actions)

        # Record stage latency and cumulative latency
        reasoning_latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
        meta = dict(result.metadata) if hasattr(result, "metadata") and isinstance(result.metadata, dict) else {}
        meta["stage_latency_ms"] = reasoning_latency_ms
        meta["reasoning_latency_ms"] = reasoning_latency_ms
        meta["cumulative_latency_ms"] = round(prev_cumulative + reasoning_latency_ms, 2)
        result.metadata = meta

        if self.verbose:
            mode = result.metadata.get("reasoning_mode", "DETERMINISTIC")
            print(_format_log(self.name, "REASON", f"Mode: {mode} | Priority: {result.priority} | Consistency: {result.evidence_consistency}"))

        # 6. COMMUNICATE: Publish downstream
        self._publish(result)
        return result

    def _publish(self, event: ClinicalReasoningEvent) -> None:
        """Publish clinical reasoning event to downstream topics."""
        if self.event_queue is not None:
            if self.verbose:
                print(_format_log(self.name, "PUBLISH", f"decision={event.priority} (Event: {event.event_id})"))
            self.event_queue.publish("clinical_decisions", event)
            self.event_queue.publish("clinical_reasoning_outputs", event)
            self.event_queue.publish("all_events", event)

    # ------------------------------------------------------------------------
    # Independent Worker Thread Execution
    # ------------------------------------------------------------------------
    def start(self) -> None:
        """Start the Clinical Reasoning Agent as an independent background worker thread."""
        if self._is_running or self.event_queue is None:
            return
        self._stop_event.clear()
        self._is_running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name=f"{self.name}-Worker",
            daemon=True,
        )
        self._worker_thread.start()
        if self.verbose:
            print(_format_log(self.name, "STARTED", f"Worker thread online ({self._worker_thread.name})"))

    def _worker_loop(self) -> None:
        """Continuous event consumption loop for analytical evidence."""
        self.emit_heartbeat()
        last_hb = time.time()
        while not self._stop_event.is_set():
            event = self.event_queue.consume("clinical_reasoning_events", timeout=0.2) if self.event_queue else None
            if isinstance(event, DataAnalysisEvent):
                self.process_event(event)
                self.emit_heartbeat()

            if not self.event_queue:
                time.sleep(0.1)

            if time.time() - last_hb >= 1.0:
                self.emit_heartbeat()
                last_hb = time.time()

        self._is_running = False
        if self.verbose:
            print(_format_log(self.name, "STOPPED", "Worker loop finished."))

    def stop(self) -> None:
        """Signal worker thread to stop."""
        self._stop_event.set()
        self._is_running = False

    def join(self, timeout: float | None = 5.0) -> None:
        """Wait for worker thread to finish."""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)


__all__ = [
    "ClinicalReasoningAgent",
    "ClinicalReasoningEngine",
    "ClinicalReasoningEvent",
    "ClinicalReasoningPriority",
    "ClinicalReasoningState",
    "RETRIEVAL_CONFIDENCE_THRESHOLD",
]
