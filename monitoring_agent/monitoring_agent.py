"""CareMatrix Monitoring Agent controller with autonomous observe-state-analyze-decide-act loop.

Supports both synchronous stepping and independent concurrent background worker execution.
"""

from __future__ import annotations

import datetime
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Generator, Iterable

import numpy as np
import pandas as pd

from communication.event_queue import EventQueue
from communication.events import AgentHeartbeatEvent, MonitoringDecision, MonitoringEvent
from monitoring_agent.alert import create_alert
from monitoring_agent.alert_state import PhysiologicalEventTracker, manage_alert_lifecycle
from monitoring_agent.baseline import calculate_baseline
from monitoring_agent.config import (
    ALERT_COOLDOWN_SECONDS,
    BASELINE_WINDOW_SECONDS,
    MIN_BASELINE_SAMPLES,
    PERSISTENCE_SECONDS,
    RECOVERY_DURATION_SECONDS,
    SEVERITY_MIN_VITALS,
    SEVERITY_MULTIPLIERS,
    SEVERITY_PERSISTENCE,
)
from monitoring_agent.decision_engine import MonitoringDecisionEngine
from monitoring_agent.deviation_detector import (
    DEVIATION_THRESHOLDS,
    apply_persistence,
    calculate_deviation,
)
from monitoring_agent.preprocessing import preprocess_data
from monitoring_agent.signal_quality import assess_signal_quality, invalid_measurement_mask
from monitoring_agent.state import (
    PatientMonitoringState,
    SEVERITY_LEVELS,
    VITAL_COLUMNS,
    _NUM_SEVERITY,
    _SEVERITY_NUM,
)
from monitoring_agent.trend_detector import calculate_trends

TRACK_MAPPING = {
    "Solar8000/HR": "HR",
    "Solar8000/ART_MBP": "MAP",
    "Solar8000/PLETH_SPO2": "SpO2",
    "Solar8000/RR": "RR",
}


class AgentLifecycleState(str, Enum):
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    DEGRADED = "DEGRADED"
    RECOVERING = "RECOVERING"
    STOPPED = "STOPPED"


def _format_log(agent_name: str, action: str, details: str) -> str:
    now_str = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    return f"[{now_str}] {agent_name:<16} | {action:<10} | {details}"


def _apply_severity_aware_persistence(
    candidate: pd.Series,
    severity: pd.Series,
    severity_persistence: dict[str, int] = SEVERITY_PERSISTENCE,
) -> pd.Series:
    """Mark candidate samples as persistent using severity-dependent durations."""
    candidates = candidate.fillna(False).astype(bool)
    result: list[bool] = []
    run_length = 0
    is_persistent = False

    for i in range(len(candidates)):
        if candidates.iloc[i]:
            run_length += 1
            if not is_persistent:
                sev = severity.iloc[i]
                required = severity_persistence.get(
                    sev, max(severity_persistence.values()),
                )
                if run_length >= required:
                    is_persistent = True
            result.append(is_persistent)
        else:
            run_length = 0
            is_persistent = False
            result.append(False)

    return pd.Series(result, index=candidates.index, dtype=bool, name="persistent_alert")


def _get_vitaldb() -> Any:
    """Import VitalDB only when live dataset access is requested."""
    try:
        import vitaldb
    except ImportError as exc:
        raise RuntimeError(
            "VitalDB is not installed. Activate your virtual environment and run "
            "'pip install -r requirements.txt'."
        ) from exc
    return vitaldb


def find_suitable_cases(limit: int | None = None) -> list[int]:
    """Find VitalDB cases containing all four required numeric vital tracks."""
    vitaldb = _get_vitaldb()
    try:
        case_ids = list(vitaldb.find_cases(list(TRACK_MAPPING)))
    except Exception as exc:
        raise RuntimeError(
            "Unable to query VitalDB for cases. Check your network connection and retry."
        ) from exc

    if not case_ids:
        raise RuntimeError("VitalDB returned no cases with HR, MAP, SpO2, and RR tracks.")
    return [int(case_id) for case_id in case_ids[:limit]] if limit else [int(case_id) for case_id in case_ids]


def load_case_data(case_id: int) -> pd.DataFrame:
    """Load one VitalDB case as one-second HR, MAP, SpO2, and RR samples with local cache fallback."""
    try:
        vitaldb = _get_vitaldb()
        values = vitaldb.load_case(int(case_id), list(TRACK_MAPPING), interval=1)
        if values is not None and np.size(values) > 0:
            values = np.asarray(values)
            if values.ndim == 2 and values.shape[1] == len(VITAL_COLUMNS):
                df = pd.DataFrame(values, columns=VITAL_COLUMNS)
                df.index = pd.RangeIndex(start=0, stop=len(df), step=1, name="timestamp")
                return df
    except Exception:
        pass

    # Fallback to locally cached processed file if present
    local_file = Path(__file__).resolve().parent.parent / "data" / "processed" / f"patient_{case_id}.csv"
    if local_file.exists():
        df_local = pd.read_csv(local_file)
        if "timestamp" in df_local.columns:
            df_local = df_local.set_index("timestamp")
        return df_local[VITAL_COLUMNS]

    raise RuntimeError(
        f"Unable to load VitalDB case {case_id}. Check the case ID and network connection."
    )


class MonitoringAgent:
    """Genuine autonomous Monitoring Agent controller.

    Supports both step-by-step processing and independent background worker execution:
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
    ):
        self.name = name
        self.case_id = int(case_id)
        self.event_queue = event_queue
        self.baseline_window = baseline_window
        self.persistence_duration = persistence_duration
        self.recovery_duration = recovery_duration
        self.cooldown_seconds = cooldown_seconds
        self.verbose = verbose

        # Lifecycle state
        self.lifecycle_state = AgentLifecycleState.STARTING

        # Agent state
        self.state = PatientMonitoringState(case_id=self.case_id)

        # Autonomous decision engine
        self.decision_engine = MonitoringDecisionEngine(
            cooldown_seconds=self.cooldown_seconds,
            recovery_duration_seconds=self.recovery_duration,
        )

        # Tools used by agent
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

        # Thread management for independent worker execution
        self._worker_thread: threading.Thread | None = None
        self._is_running: bool = False
        self._stop_event = threading.Event()
        self._stream_source: Iterable[dict[str, Any] | pd.Series] | None = None

        # Failure demonstration hooks
        self._hang_at_sample: int | None = None
        self._crash_at_sample: int | None = None

        self.lifecycle_state = AgentLifecycleState.RUNNING

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

    def step(self, sample: dict[str, Any] | pd.Series) -> tuple[MonitoringDecision, MonitoringEvent | None]:
        """Execute one continuous observation cycle: OBSERVE -> UPDATE STATE -> ANALYZE -> DECIDE -> ACT."""
        if not self.is_healthy or self.lifecycle_state == AgentLifecycleState.DEGRADED:
            raise RuntimeError(f"{self.name} is in a degraded/failed state and must be restarted.")

        # Check failure injection hooks for demonstration
        curr_obs = self.state.total_observations
        if self._hang_at_sample is not None and curr_obs >= self._hang_at_sample:
            # Simulate an unhandled freeze / hang: do not emit heartbeat, sleep to trigger supervisor
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

        # 1. OBSERVE & 2. UPDATE STATE
        if isinstance(sample, pd.Series):
            sample_dict = sample.to_dict()
            if sample.name is not None and "timestamp" not in sample_dict:
                sample_dict["timestamp"] = sample.name
        else:
            sample_dict = dict(sample)

        self.state.add_observation(sample_dict)
        self.emit_heartbeat()

        if self.verbose:
            print(_format_log(self.name, "OBSERVE", f"sample={self.state.total_observations} (t={self.state.latest_timestamp})"))

        # 3. ANALYZE (Invoke tool functions on progressive observation buffer)
        raw_df = self.state.buffer_dataframe()
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

        # 4. DECIDE (Autonomous evaluation by Decision Engine)
        decision, event, context = self.decision_engine.evaluate(
            state=self.state,
            latest_clean_values=latest_clean,
            latest_baselines=latest_base,
            latest_deviations=latest_dev,
            latest_trends=latest_tr,
            latest_signal_quality=latest_sq,
            invalid_mask=latest_inv,
        )

        self.last_decision = decision
        self.last_event = event

        # Update state cache
        self.state.latest_decision = decision.value
        self.state.latest_clean_values = latest_clean
        self.state.latest_baselines = latest_base
        self.state.latest_deviations = latest_dev
        self.state.latest_trends = latest_tr
        self.state.latest_signal_quality = latest_sq
        self.state.latest_vital_severity = context.get("vital_severity", {})
        self.state.latest_overall_severity = context.get("overall_severity", "normal")
        self.state.latest_candidate_alert = context.get("is_candidate", False)
        self.state.latest_persistent_alert = context.get("is_persistent", False)

        if self.verbose:
            if decision == MonitoringDecision.ESCALATE_TO_RISK:
                print(_format_log(self.name, "DECIDE", f"ESCALATE_TO_RISK | Severity: {context.get('overall_severity')} | Vitals: {event.affected_vitals if event else []}"))
            elif decision == MonitoringDecision.RECOVERY:
                print(_format_log(self.name, "DECIDE", f"RECOVERY | Closed event: {event.event_id if event else 'N/A'}"))

        # 5. ACT (Publish event to asynchronous FIFO queue without synchronously invoking Risk)
        if event is not None and self.event_queue is not None:
            if decision in (MonitoringDecision.ESCALATE_TO_RISK, MonitoringDecision.RECOVERY):
                if self.verbose:
                    print(_format_log(self.name, "PUBLISH", f"event={event.event_id} (Topic: 'monitoring_events')"))
                self.event_queue.publish("monitoring_events", event)
                self.event_queue.publish("all_events", event)

        return decision, event

    # ------------------------------------------------------------------------
    # Independent Worker Thread Execution
    # ------------------------------------------------------------------------
    def start(self, stream_source: Iterable[dict[str, Any] | pd.Series], sample_delay: float = 0.0) -> None:
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
        # Do not create duplicate workers if the existing worker is still alive and healthy
        if self._worker_thread is not None and self._worker_thread.is_alive() and self.is_healthy:
            return True

        # Safely handle the old/dead worker
        self._stop_event.set()
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=0.5)

        # Preserve/restore the provided state snapshot
        if snapshot is not None:
            self.restore_state_snapshot(snapshot)

        # Clear stop/failure events and reset health/lifecycle state
        self._stop_event.clear()
        self.is_healthy = True
        self._is_running = True
        self.lifecycle_state = AgentLifecycleState.RUNNING

        # Create a new threading.Thread targeting the existing worker loop
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name=f"{self.name}-Worker",
            daemon=True,
        )
        self._worker_thread.start()
        time.sleep(0.02)

        # Emit immediate heartbeat so supervisor / queue sees healthy status
        self.emit_heartbeat()

        # Verify that the new thread is alive
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


def monitor_dataframe(
    case_id: int,
    df: pd.DataFrame,
    baseline_window: int = BASELINE_WINDOW_SECONDS,
    persistence_duration: int = PERSISTENCE_SECONDS,
) -> dict[str, Any]:
    """Run full monitoring on a dataframe, preserving exact backward compatibility."""
    missing = [column for column in VITAL_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError("Input data is missing required columns: " + ", ".join(missing))

    raw_df = df.loc[:, VITAL_COLUMNS].copy()
    if raw_df.empty:
        raise ValueError("Cannot monitor empty case data.")
    if raw_df.index.name is None:
        raw_df.index.name = "timestamp"

    signal_quality = assess_signal_quality(raw_df)
    invalid_measurements = invalid_measurement_mask(raw_df)
    processed_df = preprocess_data(raw_df)
    baseline = calculate_baseline(processed_df, window=baseline_window)
    deviation = calculate_deviation(processed_df, baseline)
    trends = calculate_trends(processed_df)

    baseline_ready = baseline.notna().all(axis=1)

    vital_severity = pd.DataFrame(
        "normal", index=processed_df.index, columns=VITAL_COLUMNS, dtype="object",
    )
    for sev_name in SEVERITY_LEVELS:
        multiplier = SEVERITY_MULTIPLIERS[sev_name]
        for vital in VITAL_COLUMNS:
            threshold = DEVIATION_THRESHOLDS[vital] * multiplier
            exceeds = (
                deviation[vital].ge(threshold).fillna(False)
                & ~invalid_measurements[vital]
                & baseline_ready
            )
            vital_severity.loc[exceeds, vital] = sev_name

    vital_severity_numeric = vital_severity.apply(lambda col: col.map(_SEVERITY_NUM))
    overall_severity = (
        vital_severity_numeric.max(axis=1).map(_NUM_SEVERITY).rename("overall_severity")
    )

    alert_eligible = vital_severity.ne("normal")
    deviation_count = alert_eligible.sum(axis=1).astype(int)

    candidate_alert = pd.Series(False, index=processed_df.index, dtype=bool)
    for sev_name in SEVERITY_LEVELS:
        mask = overall_severity == sev_name
        min_v = SEVERITY_MIN_VITALS[sev_name]
        candidate_alert.loc[mask] = (
            (deviation_count[mask] >= min_v) & baseline_ready[mask]
        )
    candidate_alert = candidate_alert.rename("candidate_alert")

    persistent_alert = _apply_severity_aware_persistence(
        candidate_alert, overall_severity, SEVERITY_PERSISTENCE,
    )
    lifecycle = manage_alert_lifecycle(
        candidate_alert,
        persistent_alert,
        recovery_duration=RECOVERY_DURATION_SECONDS,
    )

    monitoring_results = processed_df.copy()
    for vital in VITAL_COLUMNS:
        monitoring_results[f"{vital}_baseline"] = baseline[vital]
        monitoring_results[f"{vital}_deviation"] = deviation[vital]
        monitoring_results[f"{vital}_trend"] = trends[vital]
        monitoring_results[f"{vital}_signal_quality"] = signal_quality[vital]
        monitoring_results[f"{vital}_severity"] = vital_severity[vital]
        monitoring_results[f"{vital}_invalid"] = invalid_measurements[vital]
    monitoring_results["deviation_count"] = deviation_count
    monitoring_results["overall_severity"] = overall_severity
    monitoring_results["candidate_alert"] = candidate_alert
    monitoring_results["persistent_alert"] = persistent_alert
    monitoring_results["alert_state"] = lifecycle["alert_state"]
    monitoring_results["active_alert_id"] = lifecycle["active_alert_id"]
    monitoring_results["event_id"] = lifecycle["active_alert_id"].map(
        lambda value: f"case_{case_id}_event_{int(value):03d}" if pd.notna(value) else None
    )

    candidate_starts: dict[Any, tuple[Any, int]] = {}
    run_start_position: int | None = None
    for position, timestamp in enumerate(monitoring_results.index):
        if bool(candidate_alert.iloc[position]):
            if run_start_position is None:
                run_start_position = position
        else:
            run_start_position = None
        if monitoring_results.at[timestamp, "alert_state"] == "alert_started":
            candidate_starts[timestamp] = (
                monitoring_results.index[run_start_position],
                run_start_position,
            )

    def vital_details_at(timestamp: Any, vitals: list[str]) -> dict[str, dict[str, Any]]:
        details: dict[str, dict[str, Any]] = {}
        for vital in vitals:
            current = monitoring_results.at[timestamp, vital]
            baseline_value = monitoring_results.at[timestamp, f"{vital}_baseline"]
            relative_deviation = monitoring_results.at[timestamp, f"{vital}_deviation"]
            direction = "stable"
            if pd.notna(current) and pd.notna(baseline_value):
                direction = "increasing" if current > baseline_value else "decreasing" if current < baseline_value else "stable"
            details[vital] = {
                "current": float(current),
                "baseline": float(baseline_value),
                "relative_deviation": float(relative_deviation),
                "severity": str(monitoring_results.at[timestamp, f"{vital}_severity"]),
                "direction": direction,
                "trend": str(monitoring_results.at[timestamp, f"{vital}_trend"]),
                "signal_quality": str(monitoring_results.at[timestamp, f"{vital}_signal_quality"]),
            }
        return details

    alerts: list[dict[str, Any]] = []
    lifecycle_events: list[dict[str, Any]] = []
    event_tracker = PhysiologicalEventTracker(case_id)
    for timestamp in monitoring_results.index:
        state = monitoring_results.at[timestamp, "alert_state"]
        alert_id = monitoring_results.at[timestamp, "active_alert_id"]
        if state == "alert_started":
            affected_vitals = [
                vital for vital in VITAL_COLUMNS if bool(alert_eligible.at[timestamp, vital])
            ]
            details = vital_details_at(timestamp, affected_vitals)
            severity = str(monitoring_results.at[timestamp, "overall_severity"])
            start_timestamp, start_position = candidate_starts[timestamp]
            initial_details = vital_details_at(start_timestamp, affected_vitals)
            event_tracker.start(
                int(alert_id), start_timestamp, affected_vitals,
                initial_details, severity=severity,
            )
            current_position = monitoring_results.index.get_loc(timestamp)
            for position in range(start_position + 1, current_position + 1):
                historical_timestamp = monitoring_results.index[position]
                event_tracker.update(
                    int(alert_id),
                    historical_timestamp,
                    vital_details_at(historical_timestamp, affected_vitals),
                )
            snapshot = event_tracker.snapshot(int(alert_id), timestamp, "alert_started")
            n_vitals = len(affected_vitals)
            reason = (
                f"Persistent {'multi-vital' if n_vitals > 1 else 'single-vital'} "
                f"physiological deviation detected (severity: {severity})"
            )
            event = create_alert(
                case_id=case_id,
                timestamp=timestamp,
                affected_vitals=affected_vitals,
                deviation_values={vital: details[vital]["relative_deviation"] for vital in affected_vitals},
                duration=snapshot["duration_seconds"],
                severity=severity,
                vital_details=details,
                reason=reason,
            )
            event.update(snapshot)
            event["event_duration_seconds"] = snapshot["duration_seconds"]
            alerts.append(event)
            lifecycle_events.append(event)
        elif state in {"alert_active", "recovering", "alert_recovered"} and pd.notna(alert_id):
            active_event = event_tracker.active_events.get(int(alert_id))
            if active_event is None:
                continue
            affected_vitals = active_event["affected_vitals"]
            details = vital_details_at(timestamp, affected_vitals)
            if state == "alert_recovered":
                snapshot = event_tracker.recover(int(alert_id), timestamp, details)
                if snapshot is not None:
                    n_vitals = len(affected_vitals)
                    recovery_event = create_alert(
                        case_id=case_id,
                        timestamp=timestamp,
                        affected_vitals=affected_vitals,
                        deviation_values={vital: details[vital]["relative_deviation"] for vital in affected_vitals},
                        duration=snapshot["duration_seconds"],
                        alert_state="alert_recovered",
                        severity=snapshot.get("severity", "moderate"),
                        vital_details=details,
                        reason=(
                            f"{'Multi-vital' if n_vitals > 1 else 'Single-vital'} "
                            f"physiological deviation recovered toward baseline"
                        ),
                    )
                    recovery_event.update(snapshot)
                    recovery_event["end_timestamp"] = timestamp
                    lifecycle_events.append(recovery_event)
            else:
                event_tracker.update(int(alert_id), timestamp, details, state)

    def count_short_candidate_runs(series: pd.Series) -> int:
        count = 0
        run_length = 0
        for value in series.astype(bool):
            if value:
                run_length += 1
            elif run_length:
                if run_length < persistence_duration:
                    count += 1
                run_length = 0
        if run_length and run_length < persistence_duration:
            count += 1
        return count

    any_deviation = alert_eligible.any(axis=1)
    failed_severity_gate = int(
        (baseline_ready & any_deviation & ~candidate_alert).sum()
    )
    monitoring_summary = {
        "threshold_violation_samples": int((baseline_ready & any_deviation).sum()),
        "candidate_deviation_samples": int(candidate_alert.sum()),
        "failed_severity_gate_samples": failed_severity_gate,
        "candidate_runs_failed_persistence": count_short_candidate_runs(candidate_alert),
        "candidate_rejected_insufficient_data": 0,
        "invalid_measurements": int(invalid_measurements.to_numpy().sum()),
        "suspicious_measurements": int(signal_quality.eq("suspicious").to_numpy().sum()),
        "insufficient_data_measurements": int(signal_quality.eq("insufficient_data").to_numpy().sum()),
        "alert_started_events": len(alerts),
        "alert_recovered_events": sum(
            event["alert_state"] == "alert_recovered" for event in lifecycle_events
        ),
        "currently_active_events": len(event_tracker.active_events),
        "independent_monitoring_events": len(alerts),
    }

    return {
        "raw_df": raw_df,
        "df": processed_df,
        "baseline": baseline,
        "deviation": deviation,
        "trends": trends,
        "signal_quality": signal_quality,
        "invalid_measurements": invalid_measurements,
        "monitoring_results": monitoring_results,
        "alerts": alerts,
        "lifecycle_events": lifecycle_events,
        "completed_events": event_tracker.completed_events,
        "active_events": event_tracker.current_events(monitoring_results.index[-1]),
        "monitoring_summary": monitoring_summary,
    }


def run_monitoring(case_id: int) -> dict[str, Any]:
    """Load one VitalDB case and return all Phase 1 monitoring outputs."""
    df = load_case_data(case_id)
    return monitor_dataframe(case_id=case_id, df=df)
