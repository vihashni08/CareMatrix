"""CareMatrix Risk Agent controller with genuine independent worker execution loop.

Lifecycle:
WAIT FOR EVENT → RECEIVE → VALIDATE → REASON (FEATURES & ML TOOL) → DECIDE → RESPOND
"""

from __future__ import annotations

import datetime
import threading
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from communication.event_queue import EventQueue
from communication.events import (
    AgentFailureEvent,
    AgentHeartbeatEvent,
    MonitoringEvent,
    PerformativeType,
    RiskDecision,
    RiskDecisionEvent,
)
from risk_agent.decision_engine import RiskDecisionEngine
from risk_agent.model import RiskModelTool
from risk_agent.preprocessing import (
    MIN_WINDOW_SAMPLES,
    VITAL_COLUMNS_7,
    construct_features as _construct_features,
    gather_patient_context as _gather_patient_context,
)
from risk_agent.risk_llm_reasoner import GeminiRiskReasoner, RiskChallengeResponse
from risk_agent.state import RiskAgentState


def _format_log(agent_name: str, action: str, details: str) -> str:
    now_str = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    return f"[{now_str}] {agent_name:<16} | {action:<10} | {details}"


class RiskAgent:
    """Autonomous Risk Agent controller utilizing ML model artifacts as inference tools.

    Runs as an independent background worker thread consuming from the EventQueue asynchronously.
    """

    def __init__(
        self,
        event_queue: EventQueue | None = None,
        model_dir: Path | str | None = None,
        max_retries: int = 3,
        name: str = "RiskAgent",
        verbose: bool = False,
        enable_llm: bool = False,
        llm_reasoner: Any | None = None,
        window_provider: Callable[[int], pd.DataFrame] | None = None,
    ):
        self.name = name
        self.event_queue = event_queue
        self.max_retries = max_retries
        self.verbose = verbose
        self.enable_llm = enable_llm
        self.llm_reasoner = llm_reasoner or (GeminiRiskReasoner() if enable_llm else None)
        self.window_provider = window_provider

        # ML Model Tool loading
        self.model_tool = RiskModelTool(model_dir=model_dir)
        self.model = self.model_tool.model
        self.preprocessor = self.model_tool.preprocessor
        self.feature_columns = self.model_tool.feature_columns
        self.risk_threshold = self.model_tool.risk_threshold
        self.model_name = self.model_tool.model_name

        # Standardized agent tools registry
        from carematrix_runtime.tools.risk_tools import create_risk_tools
        self.tools = create_risk_tools(model_tool=self.model_tool)

        # Internal state & decision engine
        self.state = RiskAgentState()
        self.decision_engine = RiskDecisionEngine(
            default_threshold=self.risk_threshold,
            max_retries=self.max_retries,
        )

        self.is_healthy: bool = True

        # Independent worker thread management
        self._worker_thread: threading.Thread | None = None
        self._is_running: bool = False
        self._stop_event = threading.Event()

        # Failure demonstration hook
        self.inject_failure: bool = False

    def attach_to_queue(self, event_queue: EventQueue) -> None:
        """Attach to event queue."""
        self.event_queue = event_queue

    def emit_heartbeat(self) -> AgentHeartbeatEvent:
        """Publish heartbeat to supervisor."""
        self.state.last_heartbeat = time.time()
        hb = AgentHeartbeatEvent(
            agent_name=self.name,
            timestamp=self.state.last_heartbeat,
            status="healthy" if self.is_healthy else "degraded",
            metrics={
                "total_processed": self.state.total_processed,
                "failure_count": self.state.failure_count,
                "active_evaluations": len(self.state.active_evaluations),
                "model_name": self.model_name,
                "threshold": self.risk_threshold,
            },
        )
        if self.event_queue is not None:
            self.event_queue.publish("heartbeats", hb)
        return hb

    def gather_patient_context(self, case_id: int) -> dict[str, float]:
        """Gather demographic context (age, sex, bmi, asa, emop) with offline fallback."""
        return _gather_patient_context(case_id, cache=self.state.patient_context_cache)

    def construct_features(
        self,
        event: MonitoringEvent,
        patient_context: dict[str, float],
        window_df: pd.DataFrame | None = None,
        return_source: bool = False,
    ) -> dict[str, float] | tuple[dict[str, float], str]:
        """Construct feature vector aligned with trained model feature columns."""
        if window_df is None and self.window_provider is not None:
            try:
                window_df = self.window_provider(event.patient_id)
            except Exception:
                window_df = None

        return _construct_features(
            event=event,
            patient_context=patient_context,
            feature_columns=self.feature_columns,
            window_df=window_df,
            return_source=return_source,
        )

    def use_ml_model_tool(self, features: dict[str, float]) -> float:
        """Invoke preprocessor and ML model tool to compute risk probability."""
        if self.inject_failure:
            raise RuntimeError("Simulated ML model tool inference failure for demo.")
        return self.model_tool.predict_proba(features)

    # ------------------------------------------------------------------------
    # Autonomous Receive-Reason-Decide-Act Cycle
    # ------------------------------------------------------------------------
    def process_event(
        self,
        event: MonitoringEvent,
        window_df: pd.DataFrame | None = None,
    ) -> RiskDecisionEvent:
        """Execute full autonomous Risk Agent cycle:

        RECEIVE EVENT -> VALIDATE -> REASON (CONTEXT & ML TOOL) -> DECIDE -> RESPOND
        """
        # 1. RECEIVE EVENT
        t_start = time.perf_counter()
        self.state.record_received_event(event)
        self.emit_heartbeat()

        prev_cumulative = float((event.metadata or {}).get("cumulative_latency_ms", 0.0) or 0.0) if hasattr(event, "metadata") and event.metadata else 0.0

        if window_df is None and self.window_provider is not None:
            try:
                window_df = self.window_provider(event.patient_id)
            except Exception:
                window_df = None

        if self.verbose:
            print(_format_log(self.name, "RECEIVE", f"event={event.event_id} (Severity: {event.severity})"))

        # 2. VALIDATE EVENT
        is_valid, validation_msg = self.decision_engine.validate_event(event)
        if not is_valid:
            err = ValueError(f"Invalid monitoring event: {validation_msg}")
            self.state.record_failure(event.event_id, str(err))
            risk_latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
            decision_event = self.decision_engine.decide_from_failure(
                event=event,
                error=err,
                retry_count=self.max_retries,
                model_name=self.model_name,
                metadata={
                    "feature_source": "approximated_fallback",
                    "stage_latency_ms": risk_latency_ms,
                    "risk_latency_ms": risk_latency_ms,
                    "cumulative_latency_ms": round(prev_cumulative + risk_latency_ms, 2),
                },
            )
            self._respond(decision_event)
            return decision_event

        # 3. REASON: GATHER CONTEXT, EXTRACT FEATURES & INVOKE ML TOOL (with Retries)
        retry_count = 0
        last_exception: Exception | None = None

        while retry_count <= self.max_retries:
            try:
                patient_context = self.gather_patient_context(event.patient_id)
                features, feature_source = self.construct_features(
                    event, patient_context, window_df=window_df, return_source=True
                )

                if self.verbose:
                    print(_format_log(self.name, "ML_TOOL", f"Invoking {self.model_name} on patient {event.patient_id} (source: {feature_source})"))

                probability = self.use_ml_model_tool(features)

                # 4. DECIDE
                risk_latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
                decision_event = self.decision_engine.decide_from_prediction(
                    event=event,
                    probability=probability,
                    threshold=self.risk_threshold,
                    model_name=self.model_name,
                    features=features,
                    metadata={
                        "feature_source": feature_source,
                        "stage_latency_ms": risk_latency_ms,
                        "risk_latency_ms": risk_latency_ms,
                        "cumulative_latency_ms": round(prev_cumulative + risk_latency_ms, 2),
                    },
                )

                if self.verbose:
                    print(_format_log(self.name, "DECIDE", f"{decision_event.decision} (Prob: {probability*100:.2f}% | Thresh: {self.risk_threshold:.2f})"))

                # 5. RESPOND & COMMUNICATE
                self.state.record_decision(decision_event, features=features)
                self._respond(decision_event)
                return decision_event

            except Exception as exc:
                last_exception = exc
                retry_count = self.state.increment_retry(event.event_id)
                if self.verbose:
                    print(_format_log(self.name, "RETRY", f"Attempt {retry_count}/{self.max_retries} failed: {exc}"))
                if retry_count > self.max_retries:
                    break
                time.sleep(0.05)

        # Fallback when retries are exhausted
        self.is_healthy = False
        self.state.record_failure(event.event_id, str(last_exception))
        risk_latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
        fail_event = self.decision_engine.decide_from_failure(
            event=event,
            error=last_exception if last_exception is not None else RuntimeError("Unknown error"),
            retry_count=retry_count,
            model_name=self.model_name,
            metadata={
                "feature_source": "approximated_fallback",
                "stage_latency_ms": risk_latency_ms,
                "risk_latency_ms": risk_latency_ms,
                "cumulative_latency_ms": round(prev_cumulative + risk_latency_ms, 2),
            },
        )
        self._respond(fail_event)

        if self.verbose:
            print(_format_log(self.name, "DECIDE", f"ESCALATE (Safety fallback after {self.max_retries} retries)"))

        # Notify queue of agent failure without crashing
        if self.event_queue is not None:
            self.event_queue.publish(
                "agent_failures",
                AgentFailureEvent(
                    agent_name=self.name,
                    error_message=str(last_exception),
                    recoverable=True,
                    context={"event_id": event.event_id, "retries": retry_count},
                ),
            )

        return fail_event

    def handle_challenge(self, challenge_event: Any) -> RiskDecisionEvent | None:
        """Handle a cross-agent challenge from DataAnalysisAgent.

        Evaluates the challenging evidence against the original risk assessment using
        the LLM reasoning core (or deterministic fallback) and responds with either:
        - An AGREE event confirming the original decision with added justification.
        - A revised RiskDecisionEvent with PROPOSE performative and revised risk level.

        Hard real-time safety bound: Cap challenge-response to at most ONE round per event chain.
        Refuse any challenge past round 1 to guarantee bounded latency in a real-time clinical system.
        """
        round_num = getattr(challenge_event, "challenge_round", 1)
        # Cap this at ONE round of challenge-response per event chain.
        # This is a hard real-time safety bound, not optional: bounded latency for a real-time clinical system.
        # Unbounded agent-to-agent debate loops could delay critical alerting or exhaust compute during patient deterioration.
        if round_num > 1:
            if self.verbose:
                print(_format_log(
                    self.name,
                    "CHALLENGE_CAP",
                    f"Refusing challenge for {getattr(challenge_event, 'event_id', '')}: exceeded round cap (round={round_num}). Bounded real-time latency enforced."
                ))
            return None

        event_id = getattr(challenge_event, "event_id", "")
        if self.verbose:
            print(_format_log(self.name, "CHALLENGE", f"Received challenge for event {event_id} (Round {round_num})"))

        # Match recent decision by event_id
        original = self.state.decisions_by_id.get(event_id)
        features = self.state.features_by_id.get(event_id, {})
        if not features and hasattr(challenge_event, "trend_metrics"):
            features = {f"{k}_mean": v.get("mean", 0.0) for k, v in challenge_event.trend_metrics.items()}

        challenge_meta = getattr(challenge_event, "metadata", {}) or {}
        conflicting_evidence = (
            challenge_meta.get("conflicting_evidence")
            or getattr(challenge_event, "conflicting_evidence", [])
            or [f"Evidence conflict: {', '.join(getattr(challenge_event, 'conflict_flags', []))}"]
        )
        conflict_flags = getattr(challenge_event, "conflict_flags", [])

        # Re-evaluate with LLM reasoning core or deterministic baseline
        re_eval_decision = "AGREE"
        revised_risk_level = original.risk_level if original else getattr(challenge_event, "risk_level", "LOW RISK")
        rationale = ""
        confidence = 0.85

        if self.enable_llm and self.llm_reasoner is not None and getattr(self.llm_reasoner, "is_available", False):
            try:
                if self.verbose:
                    print(_format_log(self.name, "LLM_REEVAL", f"Invoking Gemini risk reasoner with challenge evidence for {event_id}"))
                llm_response = self.llm_reasoner.re_evaluate_challenge(
                    features=features,
                    model_prob=original.risk_probability if original else 0.5,
                    original_decision=original.decision if original else getattr(challenge_event, "risk_level", "LOW_RISK"),
                    challenge_evidence=conflicting_evidence,
                    conflict_flags=conflict_flags,
                )
                re_eval_decision = llm_response.decision
                revised_risk_level = llm_response.revised_risk_level
                rationale = llm_response.rationale
                confidence = llm_response.confidence
            except Exception as exc:
                if self.verbose:
                    print(_format_log(self.name, "FALLBACK", f"LLM challenge reasoning failed: {exc}, using deterministic fallback."))
                re_eval_decision, revised_risk_level, rationale = self._deterministic_challenge_arbitration(
                    original, challenge_event, conflict_flags
                )
        else:
            re_eval_decision, revised_risk_level, rationale = self._deterministic_challenge_arbitration(
                original, challenge_event, conflict_flags
            )

        # Formulate AGREE or PROPOSE response event
        performative = PerformativeType.AGREE.value if re_eval_decision == "AGREE" else PerformativeType.PROPOSE.value
        decision_str = "HIGH_RISK" if "HIGH" in revised_risk_level.upper() else "LOW_RISK"

        response_event = RiskDecisionEvent(
            patient_id=original.patient_id if original else getattr(challenge_event, "case_id", 0),
            event_id=event_id,
            timestamp=time.time(),
            decision=decision_str,
            risk_probability=original.risk_probability if original else (0.75 if decision_str == "HIGH_RISK" else 0.10),
            risk_level=revised_risk_level,
            threshold=original.threshold if original else self.risk_threshold,
            model_name=original.model_name if original else self.model_name,
            features_used=features,
            reason=f"Post-challenge {re_eval_decision}: {rationale}",
            severity=original.severity if original else "moderate",
            affected_vitals=original.affected_vitals if original else [],
            window_start=original.window_start if original else None,
            window_end=original.window_end if original else None,
            analysis_level="verification",
            requested_checks=original.requested_checks if original else [],
            verification_required=False,
            metadata={
                "challenge_round": 1,
                "original_decision": original.decision if original else "UNKNOWN",
                "response_performative": performative,
                "rationale": rationale,
                "confidence": confidence,
                "conflict_flags": conflict_flags,
            },
            performative=performative,
            challenge_round=1,
        )

        self.state.record_decision(response_event, features=features)
        if self.verbose:
            print(_format_log(self.name, "RESPOND", f"Challenge response ({performative}) for {event_id}: {response_event.reason}"))

        self._respond(response_event)
        return response_event

    def _deterministic_challenge_arbitration(
        self,
        original: RiskDecisionEvent | None,
        challenge: Any,
        conflict_flags: list[str],
    ) -> tuple[str, str, str]:
        """Deterministic safety arbitration for cross-agent challenges."""
        orig_dec = original.decision if original else getattr(challenge, "risk_level", "LOW_RISK")
        orig_prob = original.risk_probability if original else 0.5

        if "LOW_RISK_WITH_WORSENING_TRENDS" in conflict_flags:
            # Worsening multi-vital trajectories override model low-risk prediction: safety escalation
            return (
                "PROPOSE",
                "HIGH RISK",
                "Model LOW_RISK revised to HIGH_RISK: conceding to observed multi-vital temporal deterioration.",
            )
        elif "HIGH_RISK_WITH_STABLE_TRENDS" in conflict_flags:
            if orig_prob < 0.20:
                # Borderline probability with completely flat vitals -> revise down
                return (
                    "PROPOSE",
                    "LOW RISK",
                    "Borderline model risk revised to LOW_RISK: vitals remain robustly stable across window.",
                )
            else:
                # Defensible baseline risk -> maintain with added justification
                return (
                    "AGREE",
                    "HIGH RISK",
                    f"Model HIGH_RISK confirmed (prob={orig_prob:.2f}): baseline vital elevation indicates underlying instability despite flat short-term slope.",
                )
        return ("AGREE", original.risk_level if original else "LOW RISK", "Assessment confirmed post-challenge.")

    def _respond(self, decision_event: RiskDecisionEvent) -> None:
        """Publish decision back to event queue."""
        if self.event_queue is not None:
            if self.verbose:
                print(_format_log(self.name, "PUBLISH", f"decision={decision_event.decision} (Event: {decision_event.event_id})"))
            self.event_queue.publish("risk_decisions", decision_event)
            self.event_queue.publish("risk_challenges", decision_event)
            self.event_queue.publish("all_events", decision_event)

    # ------------------------------------------------------------------------
    # Independent Worker Thread Execution
    # ------------------------------------------------------------------------
    def start(self) -> None:
        """Start the Risk Agent as an independent background worker thread."""
        if self._is_running:
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
            print(_format_log(self.name, "STARTED", f"Independent worker thread online (TID={self._worker_thread.name})"))

    def _worker_loop(self) -> None:
        """Continuous event consumption loop of the independent Risk worker thread."""
        while not self._stop_event.is_set():
            if self.event_queue is None:
                time.sleep(0.1)
                continue

            # 1. First check for incoming risk challenges
            challenge = self.event_queue.consume("risk_challenges", timeout=0.05)
            if challenge is not None and getattr(challenge, "performative", "") == PerformativeType.CHALLENGE.value:
                self.handle_challenge(challenge)
                continue

            # 2. Check for monitoring escalation events
            event = self.event_queue.consume("monitoring_events", timeout=0.15)
            if event is None:
                self.emit_heartbeat()
                continue

            # Process escalation events
            if isinstance(event, MonitoringEvent) and event.event_type == "alert_started":
                window_df = None
                if self.window_provider is not None:
                    try:
                        window_df = self.window_provider(event.patient_id)
                    except Exception:
                        window_df = None
                self.process_event(event, window_df=window_df)

        self._is_running = False
        if self.verbose:
            print(_format_log(self.name, "STOPPED", f"Worker loop finished ({self.state.total_processed} events processed)."))

    def stop(self) -> None:
        """Signal the worker thread to stop."""
        self._stop_event.set()
        self._is_running = False

    def join(self, timeout: float | None = 5.0) -> None:
        """Wait for the background worker thread to finish."""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)

    def get_state_snapshot(self) -> dict[str, Any]:
        """Export snapshot for supervisor state recovery."""
        return self.state.get_snapshot()

    def restore_state_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Restore state from snapshot."""
        self.state.restore_snapshot(snapshot)

    def reset(self) -> None:
        """Reset internal state."""
        self.state = RiskAgentState()
        self.is_healthy = True

    def restart_worker(self, snapshot: dict[str, Any] | None = None) -> bool:
        """Safely stop dead/crashed worker thread, restore state snapshot, and start a new live worker thread."""
        self._stop_event.set()
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=0.5)

        if snapshot is not None:
            self.restore_state_snapshot(snapshot)

        self._stop_event.clear()
        self._is_running = True
        self.is_healthy = True

        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name=f"{self.name}-Worker",
            daemon=True,
        )
        self._worker_thread.start()
        time.sleep(0.02)
        return self._worker_thread.is_alive()


__all__ = [
    "RiskAgent",
    "VITAL_COLUMNS_7",
    "_format_log",
]
