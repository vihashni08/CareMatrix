"""CareMatrix Care Coordination Agent.

Consumes validated ClinicalReasoningEvent instances and coordinates actionable
clinical workflows (notifications, clinician review scheduling, data verification requests,
and order sets) without fabricating diagnoses or overriding clinical reasoning.
"""

from __future__ import annotations

import datetime
import threading
import time
from typing import Any

from care_coordination_agent.action_policy import ActionPolicy
from care_coordination_agent.decision_engine import CareCoordinationEngine
from care_coordination_agent.schemas import CareActionType, CoordinationPriority, CoordinationStatus
from care_coordination_agent.state import PatientCoordinationState
from communication.event_queue import EventQueue
from communication.events import AgentHeartbeatEvent, CareCoordinationEvent, ClinicalReasoningEvent


def _format_log(name: str, action: str, details: str) -> str:
    now_str = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    return f"[{now_str}] {name:<24} | {action:<10} | {details}"


class CareCoordinationAgent:
    """Autonomous Care Coordination Agent controller.

    Coordinates clinical care workflows, orders, and escalation pathways from
    validated clinical reasoning evidence.
    """

    def __init__(
        self,
        event_queue: EventQueue | None = None,
        name: str = "CareCoordinationAgent",
        action_policy: ActionPolicy | None = None,
        verbose: bool = False,
    ):
        self.name = name
        self.event_queue = event_queue
        self.verbose = bool(verbose)
        self.engine = CareCoordinationEngine(action_policy=action_policy)
        self.patient_states: dict[int, PatientCoordinationState] = {}
        self.is_healthy: bool = True

        # Independent worker thread management
        self._worker_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._is_running: bool = False
        self._last_heartbeat: float = 0.0

    def emit_heartbeat(self) -> AgentHeartbeatEvent:
        """Publish heartbeat to supervisor."""
        total_actions = sum(len(s.action_history) for s in self.patient_states.values())
        active_actions = sum(len(s.active_actions) for s in self.patient_states.values())
        hb = AgentHeartbeatEvent(
            agent_name=self.name,
            status="healthy" if self.is_healthy else "degraded",
            metrics={
                "monitored_patients": len(self.patient_states),
                "total_actions": total_actions,
                "active_actions": active_actions,
                "is_running": self._is_running,
            },
        )
        if self.event_queue is not None:
            self.event_queue.publish("heartbeats", hb)
        self._last_heartbeat = time.time()
        return hb

    def get_state_snapshot(self) -> dict[str, Any]:
        """Capture serializable state snapshot for supervisor recovery."""
        return {
            "patient_states": {
                pid: s.to_dict()
                for pid, s in self.patient_states.items()
            },
            "is_healthy": self.is_healthy,
        }

    def restore_state_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Restore state from snapshot."""
        if not snapshot:
            return
        self.is_healthy = snapshot.get("is_healthy", True)
        raw_states = snapshot.get("patient_states", {})
        for pid_str, data in raw_states.items():
            pid = int(pid_str)
            st = self.patient_states.setdefault(pid, PatientCoordinationState(patient_id=pid))
            st.last_action_type = data.get("last_action_type")
            st.last_priority = data.get("last_priority")
            st.unresolved_condition = data.get("unresolved_condition", False)

    def restart_worker(self, snapshot: dict[str, Any] | None = None) -> bool:
        """Restart worker thread after a failure or crash."""
        self.stop()
        if snapshot:
            self.restore_state_snapshot(snapshot)
        self.is_healthy = True
        self.start()
        return self._is_running

    # ------------------------------------------------------------------------
    # Autonomous Event Processing Cycle
    # ------------------------------------------------------------------------
    def process_event(self, event: ClinicalReasoningEvent) -> CareCoordinationEvent:
        """Process one clinical reasoning event and emit an actionable care plan."""
        t_start = time.perf_counter()
        case_id = int(getattr(event, "case_id", 0) or getattr(event, "patient_id", 0) or 0)
        event_id = str(getattr(event, "event_id", "unknown_event"))
        priority = str(getattr(event, "priority", "ROUTINE"))

        prev_cumulative = (
            float((event.metadata or {}).get("cumulative_latency_ms", 0.0) or 0.0)
            if hasattr(event, "metadata") and isinstance(event.metadata, dict)
            else 0.0
        )

        state = self.patient_states.setdefault(case_id, PatientCoordinationState(patient_id=case_id))
        self.emit_heartbeat()

        if self.verbose:
            print(_format_log(self.name, "RECEIVE", f"event={event_id} (Patient: {case_id}, Priority: {priority})"))

        try:
            if not isinstance(event, ClinicalReasoningEvent):
                # Attempt defensive duck typing
                if not hasattr(event, "priority") or not hasattr(event, "event_id"):
                    raise TypeError("Expected ClinicalReasoningEvent or compatible structured event")

            action_event = self.engine.coordinate(event, state)

        except Exception as exc:
            self.is_healthy = False
            if self.verbose:
                print(_format_log(self.name, "ERROR", f"Coordination failed: {exc}. Fallback to safety protocol."))

            # Deterministic fallback action
            action_event = CareCoordinationEvent(
                patient_id=case_id,
                event_id=event_id,
                timestamp=time.time(),
                action_type=CareActionType.SCHEDULE_CLINICIAN_REVIEW.value,
                priority=CoordinationPriority.ELEVATED.value,
                status=CoordinationStatus.NEW.value,
                reason=f"Care coordination fallback triggered due to internal processing error: {exc}",
                clinician_review_required=True,
                suggested_orders=["Conduct urgent manual bedside vital signs assessment.", "Verify telemetry network connectivity."],
                escalation_pathway="Fallback Inpatient Safety Protocol",
                clinical_summary=f"Degraded care coordination for Patient {case_id}: {exc}",
                evidence_consistency="UNCERTAIN",
                data_reliability="COMPROMISED",
                confidence=0.30,
                source=self.name,
                correlation_id=event_id,
                metadata={"error": str(exc), "fallback": True},
            )
            state.record_action(action_event)

        # Record stage latency and cumulative pipeline latency
        coord_latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
        meta = dict(action_event.metadata) if hasattr(action_event, "metadata") and isinstance(action_event.metadata, dict) else {}
        meta["stage_latency_ms"] = coord_latency_ms
        meta["coordination_latency_ms"] = coord_latency_ms
        meta["cumulative_latency_ms"] = round(prev_cumulative + coord_latency_ms, 2)
        action_event.metadata = meta

        if self.verbose:
            print(_format_log(
                self.name,
                "COORDINATE",
                f"Action: {action_event.action_type} | Priority: {action_event.priority} | ReviewReq: {action_event.clinician_review_required}"
            ))

        self._publish(action_event)
        return action_event

    def handle_recovery(self, patient_id: int, reason: str = "Physiological stabilization detected") -> list[CareCoordinationEvent]:
        """Handle patient recovery and resolve active alerts."""
        if patient_id in self.patient_states:
            resolved = self.patient_states[patient_id].resolve_condition(resolution_reason=reason)
            for res_event in resolved:
                if self.verbose:
                    print(_format_log(self.name, "RESOLVE", f"Resolved action {res_event.action_id} for Patient {patient_id}"))
                self._publish(res_event)
            return resolved
        return []

    def _publish(self, event: CareCoordinationEvent) -> None:
        """Publish care coordination event to downstream queues."""
        if self.event_queue is not None:
            if self.verbose:
                print(_format_log(self.name, "PUBLISH", f"action={event.action_type} (ActionID: {event.action_id})"))
            self.event_queue.publish("care_coordination_events", event)
            self.event_queue.publish("care_actions", event)
            self.event_queue.publish("all_events", event)

    # ------------------------------------------------------------------------
    # Independent Worker Thread Execution
    # ------------------------------------------------------------------------
    def start(self) -> None:
        """Start the Care Coordination Agent as an independent background worker thread."""
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
        """Continuous event consumption loop for clinical reasoning decisions."""
        self.emit_heartbeat()
        last_hb = time.time()
        while not self._stop_event.is_set():
            # Check clinical_decisions topic first, fallback to care_coordination_inputs
            event = self.event_queue.consume("clinical_decisions", timeout=0.2) if self.event_queue else None
            if event is None and self.event_queue:
                event = self.event_queue.consume("care_coordination_inputs", timeout=0.0)

            if isinstance(event, ClinicalReasoningEvent):
                self.process_event(event)
                self.emit_heartbeat()

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
    "ActionPolicy",
    "CareActionType",
    "CareCoordinationAgent",
    "CareCoordinationEngine",
    "CareCoordinationEvent",
    "CoordinationPriority",
    "CoordinationStatus",
    "PatientCoordinationState",
]
