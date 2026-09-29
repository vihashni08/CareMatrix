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

import numpy as np
import pandas as pd

from care_coordination_agent import CareCoordinationAgent
from carematrix_runtime.alert_manager import AlertManager
from carematrix_runtime.metrics import SystemMetricsTracker
from carematrix_runtime.patient_memory import PatientMemory
from carematrix_runtime.patient_stream import PatientScenario, PatientStreamSimulator
from carematrix_runtime.replay_stream import PatientStreamReplayer
from carematrix_runtime.state_manager import PatientStateManager
from clinical_reasoning_agent import ClinicalReasoningAgent
from communication.event_log import EventLogWriter
from communication.event_queue import EventQueue
from communication.events import CareCoordinationEvent, ClinicalReasoningEvent, MonitoringDecision, MonitoringEvent
from data.adapters.mimic_adapter import MIMICIVAdapter
from data.adapters.vitaldb_adapter import VitalDBAdapter
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
            window_provider=self.provide_analysis_window,
        )
        self.data_analysis_agent = DataAnalysisAgent(
            event_queue=self.event_queue,
            data_loader=self.provide_analysis_window,
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

        # Dataset-backed patient tracking:
        #   { patient_id: {"replayer": PatientStreamReplayer, "data_source": str, "thread": Thread} }
        self._dataset_sources: dict[int, dict[str, Any]] = {}
        self._dataset_lock = threading.RLock()

        # Lock protecting self.metrics counter dict from concurrent dataset-patient threads
        self._metrics_lock = threading.RLock()

        # Live subscriber callbacks (e.g. for SSE or WebSockets)
        self._live_listeners: list[Callable[[str, Any], None]] = []
        self._listeners_lock = threading.RLock()

    def provide_analysis_window(self, case_id: int | str) -> pd.DataFrame:
        """Provide rolling buffered window dataframe for DataAnalysisAgent and RiskAgent.

        Retrieves recent observations from the MonitoringAgent's rolling buffer for the patient.
        Returns a DataFrame with Time and standard physiological columns (last 300s / 60 samples).
        """
        pid = int(case_id) if str(case_id).isdigit() else case_id
        target_pid = None

        with self.monitoring_agent._patient_state_lock:
            if pid in self.monitoring_agent.patient_states:
                target_pid = pid
            else:
                with self._dataset_lock:
                    for d_pid, entry in self._dataset_sources.items():
                        if entry.get("case_id") == pid:
                            target_pid = d_pid
                            break
                if target_pid is None and getattr(self.monitoring_agent, "case_id", None) == pid:
                    target_pid = pid

            p_state = self.monitoring_agent.patient_states.get(target_pid) if target_pid is not None else None
            if p_state is None:
                p_state = getattr(self.monitoring_agent, "state", None)

            if p_state is None or not p_state.raw_buffer:
                return pd.DataFrame(columns=["Time", "HR", "SpO2", "RR", "SBP", "DBP", "MAP", "BT"])

            records = list(p_state.raw_buffer)

        df = pd.DataFrame(records)
        if "Time" not in df.columns:
            if "timestamp" in df.columns:
                df["Time"] = df["timestamp"]
            else:
                df.insert(0, "Time", np.arange(len(df)) * 5.0)

        for vital in ["HR", "SpO2", "RR", "SBP", "DBP", "MAP", "BT"]:
            if vital not in df.columns:
                df[vital] = np.nan
            else:
                df[vital] = pd.to_numeric(df[vital], errors="coerce")

        df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
        df = df.dropna(subset=["Time"])

        if not df.empty and len(df) > 60:
            latest_t = df["Time"].iloc[-1]
            df = df[df["Time"] >= latest_t - 300.0]
            if len(df) > 60:
                df = df.iloc[-60:]
        return df

    def add_live_listener(self, listener: Callable[[str, Any], None]) -> None:
        """Register live listener for real-time dashboard push updates."""
        with self._listeners_lock:
            if listener not in self._live_listeners:
                self._live_listeners.append(listener)

    def remove_live_listener(self, listener: Callable[[str, Any], None]) -> None:
        with self._listeners_lock:
            if listener in self._live_listeners:
                self._live_listeners.remove(listener)

    def add_dataset_patient(self, case_id: int | str, adapter_name: str = "vitaldb") -> dict[str, Any]:
        """Load a real dataset case and stream its observations through the agent pipeline.

        If a patient with the same adapter+case_id is already running, returns the existing
        patient_id rather than spawning a duplicate thread.

        The new patient is assigned a unique patient_id >= 200 (to avoid colliding with
        simulated beds 101-103), registered with the state manager, and streamed in a
        dedicated background thread paced at self.stream_interval seconds per observation
        (matching the simulated patient loop).

        Args:
            case_id: Case identifier understood by the selected adapter.
            adapter_name: "vitaldb" (default) or "mimic".

        Returns:
            A dict with patient_id, case_id, adapter, data_source, and name — the same shape
            returned by GET /api/patients/<id>, augmented with data_source.
        """
        adapter_name_lower = adapter_name.lower()
        case_id_int = int(case_id) if str(case_id).isdigit() else case_id

        # Duplicate guard: return existing bed if same (adapter, case_id) already running
        with self._dataset_lock:
            for pid, entry in self._dataset_sources.items():
                if entry["adapter_name"] == adapter_name_lower and entry["case_id"] == case_id_int:
                    if self.verbose:
                        print(_format_log("Runtime", "DATASET", f"Case {case_id} already running as patient {pid} — reusing."))
                    return {
                        "patient_id": pid,
                        "case_id": entry["case_id"],
                        "adapter": adapter_name_lower,
                        "data_source": entry["data_source"],
                        "name": self.state_manager.get_or_create(pid).name,
                    }

        if adapter_name_lower in ("mimic", "mimic-iv", "mimic_iv"):
            adapter = MIMICIVAdapter()
            source_label = f"MIMIC-IV:case_{case_id}"
        else:
            adapter = VitalDBAdapter()
            source_label = f"VitalDB:case_{case_id}"

        # Pace observations at the same interval as the simulated patient loop
        replayer = PatientStreamReplayer(
            case_id=case_id,
            adapter=adapter,
            loop=True,
            delay_seconds=self.stream_interval,  # one observation per stream_interval seconds
        )

        # Assign a unique patient_id in the 200+ range, avoiding collisions
        with self._dataset_lock:
            existing_ids = set(self._dataset_sources.keys())
            new_pid = max(existing_ids, default=199) + 1

            bed_name = f"Bed {new_pid} ({source_label})"
            self.state_manager.get_or_create(new_pid, name=bed_name)

            entry: dict[str, Any] = {
                "replayer": replayer,
                "data_source": source_label,
                "case_id": case_id_int,
                "adapter_name": adapter_name_lower,
                "thread": None,
            }
            self._dataset_sources[new_pid] = entry

        if self.verbose:
            print(_format_log("Runtime", "DATASET", f"Patient {new_pid} loaded from {source_label}"))

        # Start background streaming thread if runtime is already running
        if self._is_running:
            t = threading.Thread(
                target=self._dataset_stream_loop,
                args=(new_pid,),
                name=f"CareMatrix-Dataset-{new_pid}",
                daemon=True,
            )
            with self._dataset_lock:
                self._dataset_sources[new_pid]["thread"] = t
            t.start()

        return {
            "patient_id": new_pid,
            "case_id": entry["case_id"],
            "adapter": adapter_name_lower,
            "data_source": source_label,
            "name": bed_name,
        }

    def _dataset_stream_loop(self, patient_id: int) -> None:
        """Background thread: feeds one dataset observation per stream_interval into the pipeline.

        The replayer's delay_seconds already paces one row per stream_interval; this loop
        does not add an additional sleep. Exceptions are caught, logged, and reflected in the
        patient's data_source label so the dashboard shows the failure rather than silently
        going stale.
        """
        with self._dataset_lock:
            entry = self._dataset_sources.get(patient_id)
        if entry is None:
            return
        replayer: PatientStreamReplayer = entry["replayer"]
        data_source: str = entry["data_source"]
        case_id = entry["case_id"]

        try:
            for sample in replayer.stream():
                if self._stop_event.is_set():
                    break
                # Augment sample with patient_id, case_id, and data_source so downstream agents
                # treat it identically to a simulated observation
                obs: dict[str, Any] = dict(sample)
                obs["patient_id"] = patient_id
                obs["case_id"] = case_id
                obs["data_source"] = data_source

                self.state_manager.record_observation(obs)
                self.metrics_tracker.record_observation(patient_id)
                with self._metrics_lock:
                    self.metrics["total_observations_evaluated"] += 1

                t0 = time.time()
                decision, transition = self.monitoring_agent.observe(obs)
                self.metrics_tracker.record_agent_execution("MonitoringAgent", max(0.0001, time.time() - t0), success=True)

                if decision == MonitoringDecision.CONTINUE_MONITORING:
                    with self._metrics_lock:
                        self.metrics["routine_bypassed_cycles"] += 1
                    self.metrics_tracker.record_adaptive_decision(bypassed=True)
                elif decision == MonitoringDecision.ESCALATE_TO_RISK:
                    with self._metrics_lock:
                        self.metrics["escalated_cycles"] += 1
                    self.metrics_tracker.record_adaptive_decision(bypassed=False)
                    if self.verbose:
                        print(_format_log(
                            "Runtime", "ADAPTIVE",
                            f"Dataset patient {patient_id} triggered escalation -> downstream pipeline activated.",
                        ))
                elif decision == MonitoringDecision.RECOVERY:
                    with self._metrics_lock:
                        self.metrics["recoveries_detected"] += 1
                    self.metrics_tracker.record_monitoring_event("recovery", persistent=False)
                    self.metrics_tracker.record_alert_resolution()
                    self.alert_manager.resolve_patient_alerts(patient_id, reason="Patient physiological stabilization")
                    self.care_coordination_agent.handle_recovery(patient_id)

                # Broadcast observation to live (SSE) listeners
                with self._listeners_lock:
                    listeners = list(self._live_listeners)
                for cb in listeners:
                    try:
                        cb("observation", obs)
                    except Exception:
                        pass

                # Pacing is handled by replayer.delay_seconds — no additional sleep here.

        except Exception as exc:
            # Log failure visibly rather than silently dying
            error_label = f"{data_source} (stream error)"
            print(_format_log("Runtime", "DATASET_ERR", f"Patient {patient_id} stream failed: {exc!r} — marking as '{error_label}'"))
            with self._dataset_lock:
                if patient_id in self._dataset_sources:
                    self._dataset_sources[patient_id]["data_source"] = error_label
            # Update state manager name so the dashboard reflects the error
            try:
                record = self.state_manager.get_or_create(patient_id)
                with record._lock:
                    record.status = "STABLE"  # don't escalate; just stop streaming
            except Exception:
                pass

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
            with self._metrics_lock:
                self.metrics["total_observations_evaluated"] += 1

            # 2. Gatekeeper: Monitoring Agent evaluates observation
            t0 = time.time()
            decision, transition = self.monitoring_agent.observe(obs)
            self.metrics_tracker.record_agent_execution("MonitoringAgent", max(0.0001, time.time() - t0), success=True)

            # 3. Adaptive Execution Logic
            if decision == MonitoringDecision.CONTINUE_MONITORING:
                # Routine stable observation: bypass downstream ML, Analysis, Reasoning, Coordination
                with self._metrics_lock:
                    self.metrics["routine_bypassed_cycles"] += 1
                self.metrics_tracker.record_adaptive_decision(bypassed=True)

            elif decision == MonitoringDecision.ESCALATE_TO_RISK:
                # Meaningful abnormality detected: execute downstream pipeline
                with self._metrics_lock:
                    self.metrics["escalated_cycles"] += 1
                self.metrics_tracker.record_adaptive_decision(bypassed=False)
                if self.verbose:
                    print(_format_log(
                        "Runtime",
                        "ADAPTIVE",
                        f"Patient {patient_id} triggered escalation -> downstream pipeline activated."
                    ))

            elif decision == MonitoringDecision.RECOVERY:
                with self._metrics_lock:
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

        # Start any dataset patient streaming threads registered before start()
        with self._dataset_lock:
            pending = [(pid, e) for pid, e in self._dataset_sources.items() if e.get("thread") is None]
        for pid, entry in pending:
            t = threading.Thread(
                target=self._dataset_stream_loop,
                args=(pid,),
                name=f"CareMatrix-Dataset-{pid}",
                daemon=True,
            )
            with self._dataset_lock:
                self._dataset_sources[pid]["thread"] = t
            t.start()

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

        # Stop all dataset-backed replayers (their threads check _stop_event)
        with self._dataset_lock:
            for entry in self._dataset_sources.values():
                entry["replayer"].stop()

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
        with self._metrics_lock:
            metrics_snapshot = dict(self.metrics)
        return {
            "runtime_running": self._is_running,
            "metrics": metrics_snapshot,
            "supervisor": health,
            "patients": self.state_manager.get_patient_summary_list(),
            "active_alerts": self.alert_manager.get_active_alerts(),
            "alert_history_count": len(self.alert_manager._alert_history),
        }

    def get_patient_data_source(self, patient_id: int) -> str:
        """Return the data_source label for a patient, or 'simulated' if simulator-backed."""
        with self._dataset_lock:
            entry = self._dataset_sources.get(patient_id)
        if entry is not None:
            return entry["data_source"]
        return "simulated"


__all__ = [
    "CareMatrixRuntime",
    "PatientScenario",
    "PatientStreamSimulator",
]
