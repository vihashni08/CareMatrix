"""CareMatrix Continuous Multi-Agent Monitoring Runtime.

Integrates all 5 autonomous agents:
1. Monitoring Agent
2. Risk Agent
3. Data Analysis Agent
4. Clinical Reasoning Agent
5. Care Coordination Agent

With independent Supervisor, Centralized State Manager, Alert Manager,
and Live Continuous Telemetry Stream Simulator.
"""

from __future__ import annotations

import datetime
import os
import threading
import time
from typing import Any, Callable

from care_coordination_agent import CareCoordinationAgent
from carematrix_runtime.alert_manager import AlertManager
from carematrix_runtime.metrics import SystemMetricsTracker
from carematrix_runtime.patient_memory import PatientMemory
from carematrix_runtime.patient_stream import PatientScenario, PatientStreamSimulator
from carematrix_runtime.state_manager import PatientStateManager
from clinical_reasoning_agent import ClinicalReasoningAgent
from communication.event_log import EventLogWriter
from communication.event_queue import EventQueue
from communication.events import CareCoordinationEvent, ClinicalReasoningEvent, MonitoringDecision, MonitoringEvent
from data_analysis_agent import DataAnalysisAgent
from monitoring_agent.monitoring_agent import MonitoringAgent
from risk_agent.risk_agent import RiskAgent
from supervisor.supervisor import AgentSupervisor


def _format_log(name: str, action: str, details: str) -> str:
    now_str = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    return f"[{now_str}] {name:<24} | {action:<10} | {details}"


class CareMatrixRuntime:
    """Continuous Multi-Agent Runtime Coordinator."""

    def __init__(
        self,
        stream_interval_seconds: float = 1.0,
        enable_llm_reasoning: bool = False,  # Default False for fast deterministic demo/testing unless enabled
        verbose: bool = True,
        event_log_path: str | None = None,
    ):
        self.stream_interval = stream_interval_seconds
        self.verbose = verbose
        self.enable_llm_reasoning = enable_llm_reasoning
        self.event_log_path = event_log_path

        # 1. Event Broker
        self.event_queue = EventQueue()

        # 2. Centralized State & Alert Managers
        self.alert_manager = AlertManager()
        self.state_manager = PatientStateManager(alert_manager=self.alert_manager)

        # Wire up event queue listener to automatically update state manager
        self.event_queue.add_diagnostic_listener(self._on_event_published)

        # 2b. Durable Event Log Writer & Startup Replay
        self.event_log_writer: EventLogWriter | None = None
        if self.event_log_path:
            if os.path.exists(self.event_log_path):
                self.event_log_writer = EventLogWriter(self.event_log_path)
                replayed_count = self.event_log_writer.replay_into(self.state_manager)
                if self.verbose:
                    print(_format_log("Runtime", "REPLAY", f"Replayed {replayed_count} events from {self.event_log_path}"))
            else:
                self.event_log_writer = EventLogWriter(self.event_log_path)

            self.event_queue.add_diagnostic_listener(self.event_log_writer.log_event)


        # 2c. Episodic Patient Memory (backed by Phase 1 SQLite log)
        self.patient_memory = PatientMemory(
            db_path=self.event_log_path,
            event_log_writer=self.event_log_writer,
            event_queue=self.event_queue,
            max_episodes_per_patient=20,
        )

        # 3. Patient Stream Simulator
        self.simulator = PatientStreamSimulator()

        # 4. Multi-Agent System Instantiation
        self.monitoring_agent = MonitoringAgent(
            case_id=101,
            event_queue=self.event_queue,
            baseline_window=20,
            persistence_duration=3,
            verbose=self.verbose,
            enable_llm=self.enable_llm_reasoning,
        )

        self.risk_agent = RiskAgent(
            event_queue=self.event_queue,
            verbose=self.verbose,
            enable_llm=self.enable_llm_reasoning,
        )
        self.data_analysis_agent = DataAnalysisAgent(
            event_queue=self.event_queue,
            verbose=self.verbose,
        )
        self.clinical_reasoning_agent = ClinicalReasoningAgent(
            event_queue=self.event_queue,
            enable_llm=self.enable_llm_reasoning,
            verbose=self.verbose,
            patient_memory=self.patient_memory,
        )
        self.care_coordination_agent = CareCoordinationAgent(
            event_queue=self.event_queue,
            verbose=self.verbose,
        )

        # 5. Agent Supervisor (tracks all 5 agents)
        self.supervisor = AgentSupervisor(
            event_queue=self.event_queue,
            heartbeat_timeout_seconds=4.0,
            check_interval_seconds=0.5,
            auto_recover=True,
            verbose=self.verbose,
        )
        self.supervisor.register_agent(self.monitoring_agent)
        self.supervisor.register_agent(self.risk_agent)
        self.supervisor.register_agent(self.data_analysis_agent)
        self.supervisor.register_agent(self.clinical_reasoning_agent)
        self.supervisor.register_agent(self.care_coordination_agent)

        # System & Agent Metrics Tracker
        self.metrics_tracker = SystemMetricsTracker()

        # Adaptive Execution Metrics
        self.metrics: dict[str, int] = {
            "total_observations_evaluated": 0,
            "routine_bypassed_cycles": 0,
            "escalated_cycles": 0,
            "recoveries_detected": 0,
        }

        # Background stream thread management
        self._stream_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._is_running = False

        # Live subscriber callbacks (e.g. for SSE or WebSockets)
        self._live_listeners: list[Callable[[str, Any], None]] = []
        self._listeners_lock = threading.RLock()

    def add_live_listener(self, listener: Callable[[str, Any], None]) -> None:
        """Register live listener for real-time dashboard push updates."""
        with self._listeners_lock:
            if listener not in self._live_listeners:
                self._live_listeners.append(listener)

    def remove_live_listener(self, listener: Callable[[str, Any], None]) -> None:
        with self._listeners_lock:
            if listener in self._live_listeners:
                self._live_listeners.remove(listener)

    def _on_event_published(self, topic: str, event: Any) -> None:
        """Internal callback invoked whenever any agent publishes to the queue."""
        self.state_manager.handle_event(topic, event)

        # Record into episodic patient memory
        if hasattr(self, "patient_memory") and self.patient_memory is not None:
            if topic in ("clinical_decisions", "clinical_reasoning_outputs") and isinstance(event, ClinicalReasoningEvent):
                self.patient_memory.record_clinical_reasoning(event)
            elif topic == "care_coordination_events" and isinstance(event, CareCoordinationEvent):
                self.patient_memory.record_care_coordination(event)

        # Update metrics tracker based on agent event
        if topic == "monitoring_events" and isinstance(event, MonitoringEvent):
            self.metrics_tracker.record_agent_execution("MonitoringAgent", 0.005, success=True)
            self.metrics_tracker.record_monitoring_event(
                event.event_type,
                candidate=False,
                persistent=("started" in event.event_type or "active" in event.event_type),
            )
        elif topic in ("risk_predictions", "risk_decisions"):
            r_level = getattr(event, "risk_level", "LOW_RISK")
            self.metrics_tracker.record_agent_execution("RiskAgent", 0.015, success=True)
            self.metrics_tracker.record_risk_decision(r_level)
        elif topic in ("data_analysis_inputs", "analytical_evidence", "data_analysis_events"):
            q_flag = getattr(event, "data_quality_flag", False)
            v_req = getattr(event, "verification_required", False)
            self.metrics_tracker.record_agent_execution("DataAnalysisAgent", 0.010, success=True)
            self.metrics_tracker.record_data_analysis(q_flag, v_req)
        elif topic == "clinical_decisions":
            meta = getattr(event, "metadata", {}) or {}
            mode = meta.get("reasoning_mode", "DETERMINISTIC")
            rag_used = bool(meta.get("knowledge_sources"))
            override = bool(meta.get("llm_reasoning_conflict"))
            self.metrics_tracker.record_agent_execution("ClinicalReasoningAgent", 0.025, success=True)
            self.metrics_tracker.record_clinical_reasoning(mode, rag_used, override)
        elif topic == "care_coordination_events":
            a_type = getattr(event, "action_type", "CONTINUE_ROUTINE_MONITORING")
            prio = getattr(event, "priority", "ROUTINE")
            rev = getattr(event, "clinician_review_required", False)
            dup = bool(getattr(event, "metadata", {}).get("is_duplicate_suppressed", False))
            self.metrics_tracker.record_agent_execution("CareCoordinationAgent", 0.008, success=True)
            self.metrics_tracker.record_care_action(a_type, prio, rev, dup)

        # Notify SSE/WebSocket subscribers
        with self._listeners_lock:
            listeners = list(self._live_listeners)
        for cb in listeners:
            try:
                cb(topic, event)
            except Exception:
                pass

    def trigger_scenario(self, patient_id: int, scenario: PatientScenario | str) -> bool:
        """Switch simulation scenario for a monitored bed."""
        success = self.simulator.set_scenario(patient_id, scenario)
        if success and self.verbose:
            print(_format_log("Runtime", "SCENARIO", f"Patient {patient_id} switched to {scenario}"))
        return success

    def step(self) -> None:
        """Execute one simulation and monitoring evaluation cycle across all monitored beds."""
        for patient_id in list(self.simulator.patients.keys()):
            # 1. Generate live observation
            obs = self.simulator.generate_observation(patient_id)
            self.state_manager.record_observation(obs)
            self.metrics_tracker.record_observation(patient_id)
            self.metrics["total_observations_evaluated"] += 1

            # 2. Gatekeeper: Monitoring Agent evaluates observation
            t0 = time.time()
            decision, transition = self.monitoring_agent.observe(obs)
            self.metrics_tracker.record_agent_execution("MonitoringAgent", max(0.0001, time.time() - t0), success=True)

            # 3. Adaptive Execution Logic
            if decision == MonitoringDecision.CONTINUE_MONITORING:
                # Routine stable observation: bypass downstream ML, Analysis, Reasoning, Coordination
                self.metrics["routine_bypassed_cycles"] += 1
                self.metrics_tracker.record_adaptive_decision(bypassed=True)

            elif decision == MonitoringDecision.ESCALATE_TO_RISK:
                # Meaningful abnormality detected: execute downstream pipeline
                self.metrics["escalated_cycles"] += 1
                self.metrics_tracker.record_adaptive_decision(bypassed=False)
                if self.verbose:
                    print(_format_log(
                        "Runtime",
                        "ADAPTIVE",
                        f"Patient {patient_id} triggered escalation -> downstream pipeline activated."
                    ))

            elif decision == MonitoringDecision.RECOVERY:
                self.metrics["recoveries_detected"] += 1
                self.metrics_tracker.record_monitoring_event("recovery", persistent=False)
                self.metrics_tracker.record_alert_resolution()
                self.alert_manager.resolve_patient_alerts(patient_id, reason="Patient physiological stabilization")
                self.care_coordination_agent.handle_recovery(patient_id)
                if self.verbose:
                    print(_format_log("Runtime", "RECOVERY", f"Patient {patient_id} stabilized -> alerts resolved."))

            # Broadcast observation event to live listeners
            with self._listeners_lock:
                listeners = list(self._live_listeners)
            for cb in listeners:
                try:
                    cb("observation", obs)
                except Exception:
                    pass

    def get_metrics(self) -> dict[str, Any]:
        """Return comprehensive system and agent metrics snapshot."""
        return self.metrics_tracker.get_summary()

    def start(self) -> None:
        """Start all agent worker threads, supervisor monitor, and continuous stream generator."""
        if self._is_running:
            return
        self._stop_event.clear()
        self._is_running = True

        if self.verbose:
            print("\n" + "=" * 70)
            print("STARTING CAREMATRIX AUTONOMOUS RUNTIME & 5-AGENT ARCHITECTURE")
            print("=" * 70)

        # Start Supervisor
        self.supervisor.start()

        # Start all 5 Agent worker threads
        self.monitoring_agent.start()
        self.risk_agent.start()
        self.data_analysis_agent.start()
        self.clinical_reasoning_agent.start()
        self.care_coordination_agent.start()

        # Start continuous patient telemetry stream generator thread
        self._stream_thread = threading.Thread(
            target=self._stream_loop,
            name="CareMatrix-TelemetryStream",
            daemon=True,
        )
        self._stream_thread.start()

        if self.verbose:
            print(_format_log("Runtime", "STARTED", "All 5 agents + stream simulator online."))

    def _stream_loop(self) -> None:
        """Continuous stream evaluation loop."""
        while not self._stop_event.is_set():
            self.step()
            time.sleep(self.stream_interval)

    def stop(self) -> None:
        """Cleanly stop runtime, stream thread, supervisor, and all agents."""
        if not self._is_running:
            return
        self._stop_event.set()
        self._is_running = False

        if self.verbose:
            print(_format_log("Runtime", "STOPPING", "Initiating graceful shutdown..."))

        # Stop stream
        if self._stream_thread is not None:
            self._stream_thread.join(timeout=1.0)

        # Stop agents and supervisor
        self.monitoring_agent.stop()
        self.risk_agent.stop()
        self.data_analysis_agent.stop()
        self.clinical_reasoning_agent.stop()
        self.care_coordination_agent.stop()
        self.supervisor.stop()

        self.monitoring_agent.join(timeout=1.0)
        self.risk_agent.join(timeout=1.0)
        self.data_analysis_agent.join(timeout=1.0)
        self.clinical_reasoning_agent.join(timeout=1.0)
        self.care_coordination_agent.join(timeout=1.0)
        self.supervisor.join(timeout=1.0)
        self.event_queue.shutdown()

        if self.event_log_writer is not None:
            self.event_log_writer.close()

        if self.verbose:
            print(_format_log("Runtime", "STOPPED", "CareMatrix runtime safely shutdown."))


    def get_system_status(self) -> dict[str, Any]:
        """Return comprehensive system health, pipeline metrics, and bed summaries."""
        health = self.supervisor.get_system_health_snapshot()
        return {
            "runtime_running": self._is_running,
            "metrics": dict(self.metrics),
            "supervisor": health,
            "patients": self.state_manager.get_patient_summary_list(),
            "active_alerts": self.alert_manager.get_active_alerts(),
            "alert_history_count": len(self.alert_manager._alert_history),
        }


__all__ = [
    "CareMatrixRuntime",
    "PatientScenario",
    "PatientStreamSimulator",
]
