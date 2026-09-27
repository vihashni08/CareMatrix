"""CareMatrix Data Analysis Agent controller with autonomous analysis cycle.

Lifecycle:
RECEIVE RISK DECISION → RETRIEVE WINDOW DATA → ASSESS QUALITY → ANALYZE PATTERNS → PUBLISH EVIDENCE
"""

from __future__ import annotations

import datetime
import threading
from typing import Any, Callable

import pandas as pd

from communication.event_queue import EventQueue
from communication.events import DataAnalysisEvent, PerformativeType, RiskDecisionEvent
from communication.orchestration import verify_cross_agent_consistency
from data_analysis_agent.analyzer import (
    assess_window_quality,
    extract_metrics_and_patterns,
)
from data_analysis_agent.data_loader import (
    SAMPLE_INTERVAL,
    VITAL_COLUMNS,
    VITAL_TRACKS,
    WINDOW_SECONDS,
    load_case_data as _default_load_case_data,
)
from data_analysis_agent.state import PatientAnalysisState


def _format_log(name: str, action: str, details: str) -> str:
    now = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    return f"[{now}] {name:<24} | {action:<10} | {details}"


class DataAnalysisAgent:
    """Autonomous Data Analysis Agent controller.

    Consumes risk decisions, extracts temporal vital sign behavior, assesses
    data quality, and emits structured analytical evidence for clinical reasoning.
    """

    def __init__(
        self,
        event_queue: EventQueue | None = None,
        data_loader: Callable[[int], pd.DataFrame] | None = None,
        name: str = "DataAnalysisAgent",
        verbose: bool = False,
    ):
        self.name = name
        self.event_queue = event_queue
        self.data_loader = data_loader or _default_load_case_data
        self.patient_states: dict[int, PatientAnalysisState] = {}
        self.verbose = bool(verbose)
        self._worker_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._is_running: bool = False

    # ------------------------------------------------------------------------
    # Autonomous Event Processing Cycle
    # ------------------------------------------------------------------------
    def process_event(self, risk_event: RiskDecisionEvent) -> DataAnalysisEvent:
        """Perform one autonomous analysis cycle and always return a traceable event."""
        # Extract defensively before validation so a malformed event cannot crash worker
        case_id = int(getattr(risk_event, "patient_id", 0) or 0)
        event_id = str(getattr(risk_event, "event_id", "unknown_risk_event"))
        timestamp = float(getattr(risk_event, "timestamp", 0.0) or 0.0)
        risk_level = str(getattr(risk_event, "risk_level", "INDETERMINATE"))
        state = self.patient_states.setdefault(case_id, PatientAnalysisState())

        try:
            # 1. RECEIVE & VALIDATE
            if self.verbose:
                print(_format_log(self.name, "RECEIVE", f"event={event_id} (Risk: {risk_level})"))
            if not isinstance(risk_event, RiskDecisionEvent):
                raise TypeError("Expected RiskDecisionEvent")
            if not risk_event.event_id:
                raise ValueError("Risk assessment has no event_id")

            # 2. RETRIEVE WINDOW DATA
            frame = self.data_loader(case_id)
            start = (
                risk_event.window_start
                if risk_event.window_start is not None
                else max(0.0, risk_event.timestamp - WINDOW_SECONDS)
            )
            end = risk_event.window_end if risk_event.window_end is not None else risk_event.timestamp
            window = frame.loc[(frame["Time"] >= start) & (frame["Time"] <= end), ["Time", *VITAL_COLUMNS]].copy()
            if window.empty:
                raise ValueError(f"No samples available in analysis window {start}–{end}.")

            # 3. REASON: QUALITY ASSESSMENT & PATTERN EXTRACTION (ADAPTIVE)
            requested_checks = getattr(risk_event, "requested_checks", None)
            quality_flag, quality = assess_window_quality(window)
            metrics, patterns, changes, executed_checks = extract_metrics_and_patterns(
                window=window,
                risk=risk_event,
                requested_checks=requested_checks,
            )
            if self.verbose:
                tier = getattr(risk_event, "analysis_level", "DETAILED")
                print(_format_log(self.name, "ANALYZE", f"Tier: {tier} | Vitals: {len(metrics)} | Quality: {'COMPROMISED' if quality_flag else 'NOMINAL'}"))

            # 4. CONTEXT: COMPARE WITH PATIENT HISTORICAL STATE
            current_trends = {v: m["trend"] for v, m in metrics.items()}
            repeated = [
                vital for vital, trend in current_trends.items()
                if trend not in ("stable", "insufficient_data")
                and state.previous_trends.get(vital) == trend
            ]
            if repeated:
                patterns.append(
                    "Recent behaviour is consistent with the prior analysis for: "
                    + ", ".join(repeated) + "."
                )
            elif state.analysed_events:
                patterns.append("Recent directional behaviour differs from the prior analysis window.")

            status = "partial_analysis" if quality["insufficient_samples"] or not metrics else "complete"

            # 5. CROSS-AGENT CONSISTENCY VERIFICATION & NEGOTIATION
            consistency, supporting, conflicting, conflict_flags, verify_req, _ = verify_cross_agent_consistency(
                risk_level=risk_level,
                metrics=metrics,
                patterns=patterns,
                data_quality_flag=quality_flag,
                analysis_status=status,
            )
            if self.verbose:
                print(_format_log(self.name, "VERIFY", f"Consistency: {consistency.upper()} | Verification Req: {verify_req}"))

            current_round = getattr(risk_event, "challenge_round", 0)
            risk_performative = getattr(risk_event, "performative", PerformativeType.INFORM.value)
            negotiation_trace: dict[str, Any] | None = None

            # Hard real-time safety bound: cap challenge-response to at most ONE round per event chain.
            # Bounded latency is required for a real-time clinical monitoring system because
            # unbounded negotiation loops could delay critical alerting or exhaust compute during patient deterioration.
            if consistency.upper() == "CONFLICTING" and current_round < 1:
                # 5a. ISSUE CHALLENGE TO RISK AGENT
                challenge_event = DataAnalysisEvent(
                    case_id=case_id,
                    event_id=event_id,
                    timestamp=timestamp,
                    risk_level=risk_level,
                    trend_metrics=metrics,
                    pattern_identified=patterns,
                    important_changes=changes,
                    data_quality_flag=quality_flag,
                    analysis_status=status,
                    data_quality_details=quality,
                    executed_checks=executed_checks,
                    evidence_consistency=consistency,
                    conflict_flags=conflict_flags,
                    verification_required=True,
                    metadata={
                        "conflicting_evidence": conflicting,
                        "supporting_evidence": supporting,
                        "challenge_reason": f"Observed trends conflict with risk assessment {risk_level}: {conflict_flags}",
                        "target_agent": "RiskAgent",
                        "target_event_id": event_id,
                    },
                    performative=PerformativeType.CHALLENGE.value,
                    challenge_round=current_round + 1,
                )
                if self.verbose:
                    reason_msg = conflicting[0] if conflicting else str(conflict_flags)
                    print(_format_log(
                        self.name,
                        "CHALLENGE",
                        f"Round 1 Challenge to RiskAgent on {event_id}: {reason_msg}"
                    ))
                if self.event_queue is not None:
                    self.event_queue.publish("risk_challenges", challenge_event)

                # Update patient historical state
                state.previous_trends = current_trends
                state.previous_data_quality_flag = quality_flag
                state.previous_patterns = patterns
                state.last_analysed_timestamp = timestamp
                state.analysed_events += 1

                self._publish(challenge_event)
                return challenge_event

            # If this is already a response from RiskAgent (Round 1 response: AGREE or PROPOSE)
            if current_round >= 1 or risk_performative in (PerformativeType.AGREE.value, PerformativeType.PROPOSE.value):
                statement = (
                    conflicting[0] if conflicting
                    else f"Discrepancy between model risk ({risk_level}) and trend trajectories."
                )
                negotiation_trace = {
                    "challenge_round": current_round,
                    "status": "RESOLVED",
                    "dialogue": [
                        {
                            "agent": self.name,
                            "performative": PerformativeType.CHALLENGE.value,
                            "round": 1,
                            "statement": statement,
                            "conflict_flags": conflict_flags,
                        },
                        {
                            "agent": "RiskAgent",
                            "performative": risk_performative,
                            "round": 1,
                            "statement": getattr(risk_event, "reason", ""),
                            "decision": getattr(risk_event, "decision", ""),
                            "risk_level": risk_level,
                        },
                    ],
                    "resolution_summary": (
                        f"RiskAgent {risk_performative}D with DataAnalysisAgent challenge. "
                        f"Final decision: {getattr(risk_event, 'decision', risk_level)}. Rationale: {getattr(risk_event, 'reason', '')}"
                    ),
                }
                if self.verbose:
                    print(_format_log(
                        self.name,
                        "NEGOTIATE",
                        f"Round 1 negotiation resolved with RiskAgent ({risk_performative}) for {event_id}"
                    ))

            result = DataAnalysisEvent(
                case_id=case_id,
                event_id=event_id,
                timestamp=timestamp,
                risk_level=risk_level,
                trend_metrics=metrics,
                pattern_identified=patterns,
                important_changes=changes,
                data_quality_flag=quality_flag,
                analysis_status=status,
                data_quality_details=quality,
                executed_checks=executed_checks,
                evidence_consistency=consistency,
                conflict_flags=conflict_flags,
                verification_required=verify_req,
                metadata={
                    "conflicting_evidence": conflicting,
                    "supporting_evidence": supporting,
                    **({"negotiation_trace": negotiation_trace} if negotiation_trace else {}),
                },
                performative=PerformativeType.INFORM.value,
                challenge_round=current_round,
            )

            # Update patient historical state
            state.previous_trends = current_trends
            state.previous_data_quality_flag = quality_flag
            state.previous_patterns = patterns

        except Exception as exc:
            # ERROR RECOVERY: Formulate safe partial analysis event
            result = DataAnalysisEvent(
                case_id=case_id,
                event_id=event_id,
                timestamp=timestamp,
                risk_level=risk_level,
                trend_metrics={},
                pattern_identified=[],
                important_changes=[],
                data_quality_flag=True,
                analysis_status="partial_analysis",
                error_message=str(exc),
                data_quality_details={"error": str(exc)},
                executed_checks=["error_recovery"],
                evidence_consistency="uncertain",
                conflict_flags=["ANALYSIS_EXCEPTION"],
                verification_required=True,
            )

        state.last_analysed_timestamp = timestamp
        state.analysed_events += 1

        # 5. ACT & COMMUNICATE
        self._publish(result)
        return result

    def _publish(self, event: DataAnalysisEvent) -> None:
        """Publish analysis event to downstream topics."""
        if self.verbose:
            print(_format_log(self.name, "PUBLISH", f"decision={event.analysis_status} (Event: {event.event_id})"))
        if self.event_queue is not None:
            self.event_queue.publish("data_analysis_events", event)
            self.event_queue.publish("clinical_reasoning_events", event)
            self.event_queue.publish("all_events", event)

    # ------------------------------------------------------------------------
    # Independent Worker Thread Execution
    # ------------------------------------------------------------------------
    def start(self) -> None:
        """Start the Data Analysis Agent as an independent background worker thread."""
        if self._is_running or self.event_queue is None:
            return
        self._stop_event.clear()
        self._is_running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name=f"{self.name}-Worker",
            daemon=True,
        )
        if self.verbose:
            print(_format_log(self.name, "STARTED", f"Independent worker thread online ({self.name}-Worker)"))
        self._worker_thread.start()

    def _worker_loop(self) -> None:
        """Continuous event consumption loop for risk decision assessments."""
        while not self._stop_event.is_set():
            event = self.event_queue.consume("risk_decisions", timeout=0.2) if self.event_queue else None
            if isinstance(event, RiskDecisionEvent):
                self.process_event(event)
        if self.verbose:
            print(_format_log(self.name, "STOPPED", f"Worker loop finished ({sum(s.analysed_events for s in self.patient_states.values())} events processed)."))
        self._is_running = False

    def stop(self) -> None:
        """Signal the worker thread to stop."""
        self._stop_event.set()
        self._is_running = False

    def join(self, timeout: float | None = 5.0) -> None:
        """Wait for the worker thread to finish."""
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=timeout)


__all__ = [
    "DataAnalysisAgent",
    "PatientAnalysisState",
    "SAMPLE_INTERVAL",
    "VITAL_COLUMNS",
    "VITAL_TRACKS",
    "WINDOW_SECONDS",
]
