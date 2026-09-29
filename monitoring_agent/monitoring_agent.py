"""CareMatrix Monitoring Agent controller with autonomous observe-analyze-decide-act loop.

Supports both synchronous stepping and independent concurrent background worker execution.
"""

from __future__ import annotations

import datetime
import threading
import time
from typing import Any, Generator, Iterable

import numpy as np
import pandas as pd

from communication.event_queue import EventQueue
from communication.events import AgentHeartbeatEvent, MonitoringDecision, MonitoringEvent
from monitoring_agent.analyzer import (
    _apply_severity_aware_persistence,
    monitor_dataframe,
    run_monitoring,
)
from monitoring_agent.baseline import calculate_baseline
from monitoring_agent.config import (
    ALERT_COOLDOWN_SECONDS,
    BASELINE_WINDOW_SECONDS,
    PERSISTENCE_SECONDS,
    RECOVERY_DURATION_SECONDS,
)
from monitoring_agent.data_loader import (
    TRACK_MAPPING,
    find_suitable_cases,
    load_case_data,
)
from monitoring_agent.decision_engine import MonitoringDecisionEngine
from monitoring_agent.deviation_detector import calculate_deviation
from monitoring_agent.preprocessing import preprocess_data
from monitoring_agent.schemas import AgentLifecycleState
from monitoring_agent.signal_quality import assess_signal_quality, invalid_measurement_mask
from monitoring_agent.state import PatientMonitoringState, VITAL_COLUMNS
from monitoring_agent.trend_detector import calculate_trends


def _format_log(agent_name: str, action: str, details: str) -> str:
    now_str = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    return f"[{now_str}] {agent_name:<16} | {action:<10} | {details}"


class MonitoringAgent:
    """Autonomous Monitoring Agent controller.

    Operates continuously on streaming observations:
    OBSERVE -> UPDATE STATE -> ANALYZE -> DECIDE -> ACT
    """

    def __init__(
        self,
        case_id: int,
        event_queue: EventQueue | None = None,
        baseline_window: int = BASELINE_WINDOW_SECONDS,
        persistence_duration: int = PERSISTENCE_SECONDS,
        recovery_duration: int = RECOVERY_DURATION_SECONDS,
        cooldown_seconds: int = ALERT_COOLDOWN_SECONDS,
        name: str = "MonitoringAgent",
        verbose: bool = False,
        enable_llm: bool = False,
        llm_reasoner: Any | None = None,
    ):
        self.name = name
        self.case_id = int(case_id)
        self.event_queue = event_queue
        self.baseline_window = baseline_window
        self.persistence_duration = persistence_duration
        self.recovery_duration = recovery_duration
        self.cooldown_seconds = cooldown_seconds
        self.verbose = verbose
        self.enable_llm = enable_llm
        self.llm_reasoner = llm_reasoner


        # Lifecycle & state tracking
        self.lifecycle_state = AgentLifecycleState.STARTING
        self.state = PatientMonitoringState(case_id=self.case_id)

        # Autonomous decision engine
        self.decision_engine = MonitoringDecisionEngine(
            cooldown_seconds=self.cooldown_seconds,
            recovery_duration_seconds=self.recovery_duration,
        )

        # Analysis tools used by agent
        self.tools = {
            "preprocess": preprocess_data,
            "calculate_baseline": calculate_baseline,
            "calculate_deviation": calculate_deviation,
            "calculate_trends": calculate_trends,
            "assess_signal_quality": assess_signal_quality,
            "invalid_measurement_mask": invalid_measurement_mask,
        }

        self.last_decision: MonitoringDecision = MonitoringDecision.CONTINUE_MONITORING
        self.last_event: MonitoringEvent | None = None
        self.is_healthy: bool = True

        # Thread management for independent background worker execution
        self._worker_thread: threading.Thread | None = None
        self._is_running: bool = False
        self._stop_event = threading.Event()
        self._stream_source: Iterable[dict[str, Any] | pd.Series] | None = None
        self._sample_delay: float = 0.0

        # Fault injection demonstration hooks
        self._hang_at_sample: int | None = None
        self._crash_at_sample: int | None = None

        # Multi-patient state and decision engine registry
        self.patient_states: dict[int, PatientMonitoringState] = {
            self.case_id: self.state
        }
        self.decision_engines: dict[int, MonitoringDecisionEngine] = {
            self.case_id: self.decision_engine
        }
        # RLock protecting patient_states and decision_engines dict mutations so that
        # concurrent dataset-patient threads don't race on get_or_create_patient().
        self._patient_state_lock = threading.RLock()

        self.lifecycle_state = AgentLifecycleState.RUNNING

    def get_or_create_patient(self, pid: int) -> tuple[PatientMonitoringState, MonitoringDecisionEngine]:
        """Retrieve or create an isolated monitoring state and decision engine for a patient."""
        with self._patient_state_lock:
            if pid not in self.patient_states:
                self.patient_states[pid] = PatientMonitoringState(case_id=pid)
                self.decision_engines[pid] = MonitoringDecisionEngine(
                    cooldown_seconds=self.cooldown_seconds,
                    recovery_duration_seconds=self.recovery_duration,
                )
            return self.patient_states[pid], self.decision_engines[pid]

    # ------------------------------------------------------------------------
    # State & Fault Tolerance
    # ------------------------------------------------------------------------
    def reset(self, new_case_id: int | None = None) -> None:
        """Reset internal state, optionally for a new case."""
        if new_case_id is not None:
            self.case_id = int(new_case_id)
        self.state = PatientMonitoringState(case_id=self.case_id)
        self.last_decision = MonitoringDecision.CONTINUE_MONITORING
        self.last_event = None
        self.is_healthy = True
        self.lifecycle_state = AgentLifecycleState.RUNNING
        self._stop_event.clear()

    def get_state_snapshot(self) -> dict[str, Any]:
        """Export current state for fault tolerance checkpointing."""
        return self.state.get_snapshot()

    def restore_state_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Restore state from supervisor checkpoint."""
        self.lifecycle_state = AgentLifecycleState.RECOVERING
        self.state.restore_snapshot(snapshot)
        self.case_id = self.state.case_id
        self.is_healthy = True
        self.lifecycle_state = AgentLifecycleState.RUNNING

    def emit_heartbeat(self) -> AgentHeartbeatEvent:
        """Generate and emit a heartbeat event to supervisor."""
        self.state.last_heartbeat = time.time()
        hb = AgentHeartbeatEvent(
            agent_name=self.name,
            timestamp=self.state.last_heartbeat,
            status="healthy" if self.is_healthy else "degraded",
            metrics={
                "case_id": self.case_id,
                "lifecycle_state": self.lifecycle_state.value,
                "total_observations": self.state.total_observations,
                "total_escalations": self.state.total_escalations,
                "total_recoveries": self.state.total_recoveries,
                "active_event_id": self.state.active_event_id,
                "latest_decision": self.last_decision.value,
            },
        )
        if self.event_queue is not None:
            self.event_queue.publish("heartbeats", hb)
        return hb

    # ------------------------------------------------------------------------
    # Autonomous Observe-Analyze-Decide-Act Step
    # ------------------------------------------------------------------------
    def step(self, sample: dict[str, Any] | pd.Series) -> tuple[MonitoringDecision, MonitoringEvent | None]:
        """Execute one autonomous cycle: OBSERVE -> UPDATE STATE -> ANALYZE -> DECIDE -> ACT."""
        if not self.is_healthy or self.lifecycle_state == AgentLifecycleState.DEGRADED:
            raise RuntimeError(f"{self.name} is in a degraded/failed state and must be restarted.")

        # Check failure injection hooks for demonstration
        curr_obs = self.state.total_observations
        if self._hang_at_sample is not None and curr_obs >= self._hang_at_sample:
            if self.verbose:
                print(_format_log(self.name, "CRASH/HANG", f"Intentionally hanging at sample {curr_obs} for demo."))
            self.lifecycle_state = AgentLifecycleState.DEGRADED
            self.is_healthy = False
            while not self._stop_event.is_set() and not self.is_healthy:
                time.sleep(0.1)
            if self._stop_event.is_set():
                return self.last_decision, None

        if self._crash_at_sample is not None and curr_obs >= self._crash_at_sample:
            self.lifecycle_state = AgentLifecycleState.DEGRADED
            self.is_healthy = False
            raise RuntimeError(f"Simulated unhandled MonitoringAgent exception at sample {curr_obs}")

        # 1. OBSERVE & UPDATE STATE
        t_start = time.perf_counter()
        if isinstance(sample, pd.Series):
            sample_dict = sample.to_dict()
            if sample.name is not None and "timestamp" not in sample_dict:
                sample_dict["timestamp"] = sample.name
        else:
            sample_dict = dict(sample)

        pid = int(sample_dict.get("patient_id", sample_dict.get("case_id", self.case_id)))
        target_state, target_engine = self.get_or_create_patient(pid)

        target_state.add_observation(sample_dict)
        self.emit_heartbeat()

        if self.verbose:
            print(_format_log(self.name, "OBSERVE", f"patient={pid} sample={target_state.total_observations} (t={target_state.latest_timestamp})"))

        # 2. ANALYZE (Invoke tool capabilities on progressive observation buffer)
        raw_df = target_state.buffer_dataframe()
        signal_quality_df = self.tools["assess_signal_quality"](raw_df)
        invalid_mask_df = self.tools["invalid_measurement_mask"](raw_df)

        try:
            cleaned_df = self.tools["preprocess"](raw_df)
            baseline_df = self.tools["calculate_baseline"](cleaned_df, window=self.baseline_window)
            deviation_df = self.tools["calculate_deviation"](cleaned_df, baseline_df)
            trends_df = self.tools["calculate_trends"](cleaned_df)
        except Exception:
            cleaned_df = raw_df.copy()
            baseline_df = pd.DataFrame(np.nan, index=raw_df.index, columns=VITAL_COLUMNS)
            deviation_df = pd.DataFrame(np.nan, index=raw_df.index, columns=VITAL_COLUMNS)
            trends_df = pd.DataFrame("insufficient_data", index=raw_df.index, columns=VITAL_COLUMNS)

        latest_clean = {v: float(cleaned_df[v].iloc[-1]) if pd.notna(cleaned_df[v].iloc[-1]) else np.nan for v in VITAL_COLUMNS}
        latest_base = {v: float(baseline_df[v].iloc[-1]) if pd.notna(baseline_df[v].iloc[-1]) else np.nan for v in VITAL_COLUMNS}
        latest_dev = {v: float(deviation_df[v].iloc[-1]) if pd.notna(deviation_df[v].iloc[-1]) else np.nan for v in VITAL_COLUMNS}
        latest_tr = {v: str(trends_df[v].iloc[-1]) for v in VITAL_COLUMNS}
        latest_sq = {v: str(signal_quality_df[v].iloc[-1]) for v in VITAL_COLUMNS}
        latest_inv = {v: bool(invalid_mask_df[v].iloc[-1]) for v in VITAL_COLUMNS}

        # 3. DECIDE (Autonomous evaluation by Decision Engine)
        decision, event, context = target_engine.evaluate(
            state=target_state,
            latest_clean_values=latest_clean,
            latest_baselines=latest_base,
            latest_deviations=latest_dev,
            latest_trends=latest_tr,
            latest_signal_quality=latest_sq,
            invalid_mask=latest_inv,
        )

        # 3b. LLM PROPOSAL & SAFETY ARBITRATION (if enabled)
        if self.enable_llm:
            llm_proposal = None
            fallback_err = None
            try:
                if self.llm_reasoner is None:
                    from monitoring_agent.monitoring_llm_reasoner import GeminiMonitoringReasoner
                    self.llm_reasoner = GeminiMonitoringReasoner()
                llm_proposal = self.llm_reasoner.propose(context)
            except Exception as exc:
                fallback_err = str(exc)

            decision, event, context = target_engine.arbitrate(
                decision=decision,
                event=event,
                context=context,
                llm_proposal=llm_proposal,
                fallback_error=fallback_err,
            )

        if event is not None:
            stage_latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
            meta = dict(event.metadata) if hasattr(event, "metadata") and event.metadata else {}
            meta["stage_latency_ms"] = stage_latency_ms
            meta["monitoring_latency_ms"] = stage_latency_ms
            meta["cumulative_latency_ms"] = stage_latency_ms
            event.metadata = meta

        if pid == self.case_id:
            self.last_decision = decision
            self.last_event = event
            self.state = target_state


        # Update state cache
        target_state.latest_decision = decision.value
        target_state.latest_clean_values = latest_clean
        target_state.latest_baselines = latest_base
        target_state.latest_deviations = latest_dev
        target_state.latest_trends = latest_tr
        target_state.latest_signal_quality = latest_sq
        target_state.latest_vital_severity = context.get("vital_severity", {})
        target_state.latest_overall_severity = context.get("overall_severity", "normal")
        target_state.latest_candidate_alert = context.get("is_candidate", False)
        target_state.latest_persistent_alert = context.get("is_persistent", False)

        if self.verbose:
            if decision == MonitoringDecision.ESCALATE_TO_RISK:
                print(_format_log(self.name, "DECIDE", f"ESCALATE_TO_RISK | Severity: {context.get('overall_severity')} | Vitals: {event.affected_vitals if event else []}"))
            elif decision == MonitoringDecision.RECOVERY:
                print(_format_log(self.name, "DECIDE", f"RECOVERY | Closed event: {event.event_id if event else 'N/A'}"))

        # 4. ACT (Publish event to asynchronous event queue)
        if event is not None and self.event_queue is not None:
            if decision in (MonitoringDecision.ESCALATE_TO_RISK, MonitoringDecision.RECOVERY):
                if self.verbose:
                    print(_format_log(self.name, "PUBLISH", f"event={event.event_id} (Topic: 'monitoring_events')"))
                self.event_queue.publish("monitoring_events", event)
                self.event_queue.publish("all_events", event)

        return decision, event

    observe = step

    # ------------------------------------------------------------------------
    # Independent Worker Thread Execution
    # ------------------------------------------------------------------------
    def start(self, stream_source: Iterable[dict[str, Any] | pd.Series] | None = None, sample_delay: float = 0.0) -> None:
        """Start the Monitoring Agent as an independent long-running background worker."""
        if self._is_running:
            return

        self._stream_source = stream_source
        self._sample_delay = sample_delay
        self._stop_event.clear()
        self._is_running = True
        self.lifecycle_state = AgentLifecycleState.RUNNING

        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name=f"{self.name}-Worker",
            daemon=True,
        )
        self._worker_thread.start()
        if self.verbose:
            print(_format_log(self.name, "STARTED", f"Independent worker thread online (TID={self._worker_thread.name})"))

    def _worker_loop(self) -> None:
        """Continuous execution loop of the independent worker thread."""
        if self._stream_source is None:
            self.emit_heartbeat()
            last_hb = time.time()
            while not self._stop_event.is_set():
                if self.event_queue:
                    obs = self.event_queue.consume("patient_observations", timeout=0.2)
                    if obs is not None:
                        try:
                            self.observe(obs)
                            self.emit_heartbeat()
                        except Exception as exc:
                            if self.verbose:
                                print(_format_log(self.name, "ERROR", f"Exception in worker observe: {exc}"))
                            self.is_healthy = False
                            self.lifecycle_state = AgentLifecycleState.DEGRADED
                else:
                    time.sleep(0.1)
                if time.time() - last_hb >= 1.0:
                    self.emit_heartbeat()
                    last_hb = time.time()
            if self.is_healthy:
                self.lifecycle_state = AgentLifecycleState.STOPPED
            self._is_running = False
            return

        source = self._stream_source
        if isinstance(source, (list, tuple)):
            current_pos = self.state.total_observations
            source = source[current_pos:]
        elif isinstance(source, pd.DataFrame):
            current_pos = self.state.total_observations
            source = (row for _, row in source.iloc[current_pos:].iterrows())

        for sample in source:
            if self._stop_event.is_set():
                break

            try:
                self.step(sample)
            except Exception as exc:
                if self.verbose:
                    print(_format_log(self.name, "ERROR", f"Exception in worker loop: {exc}"))
                self.is_healthy = False
                self.lifecycle_state = AgentLifecycleState.DEGRADED
                break

            if self._sample_delay > 0:
                time.sleep(self._sample_delay)

        if self.is_healthy:
            self.lifecycle_state = AgentLifecycleState.STOPPED
        self._is_running = False
        if self.verbose:
            print(_format_log(self.name, "STOPPED", f"Worker loop finished ({self.state.total_observations} observations)."))

    def stop(self) -> None:
        """Signal the worker thread to stop."""
        self._stop_event.set()
        self.lifecycle_state = AgentLifecycleState.STOPPED
        self._is_running = False

    def join(self, timeout: float | None = 5.0) -> None:
        """Wait for the background worker thread to finish."""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)

    def restart_worker(self, snapshot: dict[str, Any] | None = None) -> bool:
        """Safely stop dead/crashed worker thread, restore state snapshot, and start a new live worker thread."""
        if self._worker_thread is not None and self._worker_thread.is_alive() and self.is_healthy:
            return True

        self._stop_event.set()
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=0.5)

        if snapshot is not None:
            self.restore_state_snapshot(snapshot)

        self._stop_event.clear()
        self.is_healthy = True
        self._is_running = True
        self.lifecycle_state = AgentLifecycleState.RUNNING

        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name=f"{self.name}-Worker",
            daemon=True,
        )
        self._worker_thread.start()
        time.sleep(0.02)
        self.emit_heartbeat()
        return self._worker_thread.is_alive()

    def stream_observations(
        self,
        samples: Iterable[dict[str, Any] | pd.Series],
    ) -> Generator[tuple[MonitoringDecision, MonitoringEvent | None], None, None]:
        """Progressively stream patient observations through the agent loop."""
        self._stop_event.clear()
        for sample in samples:
            if self._stop_event.is_set():
                break
            yield self.step(sample)


__all__ = [
    "AgentLifecycleState",
    "MonitoringAgent",
    "TRACK_MAPPING",
    "VITAL_COLUMNS",
    "_apply_severity_aware_persistence",
    "find_suitable_cases",
    "load_case_data",
    "monitor_dataframe",
    "run_monitoring",
]
