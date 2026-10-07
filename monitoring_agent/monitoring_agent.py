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
        max_tool_iterations: int = 5,
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
        self.max_tool_iterations = max_tool_iterations


        # Lifecycle & state tracking
        self.lifecycle_state = AgentLifecycleState.STARTING
        self.state = PatientMonitoringState(case_id=self.case_id)

        # Autonomous decision engine
        self.decision_engine = MonitoringDecisionEngine(
            cooldown_seconds=self.cooldown_seconds,
            recovery_duration_seconds=self.recovery_duration,
        )

        # Standardized agent tools registry (transparently supports dict access and Tool execution)
        from carematrix_runtime.tools.monitoring_tools import create_monitoring_tools
        self.tools = create_monitoring_tools(baseline_window=self.baseline_window)

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

        # 2. ANALYZE (Genuine, bounded LLM tool-selection loop if LLM enabled; else deterministic)
        tool_trace: list[dict[str, Any]] = []
        agentic_mode = "DETERMINISTIC"
        llm_proposal = None
        fallback_err = None

        if self.enable_llm:
            try:
                (
                    cleaned_df,
                    baseline_df,
                    deviation_df,
                    trends_df,
                    signal_quality_df,
                    invalid_mask_df,
                    tool_trace,
                    llm_proposal,
                ) = self._run_agentic_tool_loop(
                    target_state=target_state,
                    latest_sample=sample_dict,
                )
                agentic_mode = "LLM_AGENTIC_TOOLS"
            except Exception as exc:
                fallback_err = str(exc)
                agentic_mode = "LLM_FALLBACK_DETERMINISTIC"
                if self.verbose:
                    print(_format_log(self.name, "LLM_LOOP_FALLBACK", f"Tool loop failed: {exc}. Executing deterministic fallback."))
                cleaned_df, baseline_df, deviation_df, trends_df, signal_quality_df, invalid_mask_df = self._run_deterministic_pipeline(target_state)
        else:
            cleaned_df, baseline_df, deviation_df, trends_df, signal_quality_df, invalid_mask_df = self._run_deterministic_pipeline(target_state)

        target_state.latest_tool_trace = tool_trace
        target_state.last_agentic_mode = agentic_mode

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

        # 3b. SAFETY ARBITRATION (Deterministic safety layer arbitrates final decision)
        if self.enable_llm:
            # If agentic tool loop did not produce proposal or ran into error, attempt propose or record fallback
            if llm_proposal is None and not fallback_err:
                try:
                    if self.llm_reasoner is None:
                        from monitoring_agent.monitoring_llm_reasoner import GeminiMonitoringReasoner
                        self.llm_reasoner = GeminiMonitoringReasoner()
                    if hasattr(self.llm_reasoner, "propose"):
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
            if tool_trace:
                meta["tool_trace"] = tool_trace
            meta["agentic_mode"] = agentic_mode
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

    def _run_deterministic_pipeline(
        self,
        target_state: PatientMonitoringState,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Execute the standard deterministic analytical pipeline using registered tools."""
        df_raw = target_state.buffer_dataframe()

        # Assess signal quality & invalid mask
        sq_tool = self.tools.get("assess_signal_quality")
        inv_tool = self.tools.get("invalid_measurement_mask")
        signal_quality_df = sq_tool(df_raw) if sq_tool else assess_signal_quality(df_raw)
        invalid_mask_df = inv_tool(df_raw) if inv_tool else invalid_measurement_mask(df_raw)

        # Preprocessing, baseline, deviation, trends with defensive fallback for small buffer sizes
        try:
            prep_tool = self.tools.get("preprocess")
            cleaned_df = prep_tool(df_raw) if prep_tool else preprocess_data(df_raw)

            base_tool = self.tools.get("calculate_baseline")
            baseline_df = base_tool(cleaned_df, window=self.baseline_window) if base_tool else calculate_baseline(cleaned_df, window=self.baseline_window)

            dev_tool = self.tools.get("calculate_deviation")
            deviation_df = dev_tool(cleaned_df, baseline_df) if dev_tool else calculate_deviation(cleaned_df, baseline_df)

            trend_tool = self.tools.get("calculate_trends")
            trends_df = trend_tool(cleaned_df) if trend_tool else calculate_trends(cleaned_df)
        except Exception:
            cleaned_df = df_raw.copy()
            baseline_df = pd.DataFrame(np.nan, index=df_raw.index, columns=VITAL_COLUMNS)
            deviation_df = pd.DataFrame(np.nan, index=df_raw.index, columns=VITAL_COLUMNS)
            trends_df = pd.DataFrame("insufficient_data", index=df_raw.index, columns=VITAL_COLUMNS)

        return cleaned_df, baseline_df, deviation_df, trends_df, signal_quality_df, invalid_mask_df

    def _run_agentic_tool_loop(
        self,
        target_state: PatientMonitoringState,
        latest_sample: dict[str, Any],
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]], Any]:
        """Execute genuine bounded LLM tool-selection loop.

        Gemini determines which analytical tool to run next based on observation and current state.
        Executes chosen tool, records step trace, passes result back to agent state, and iterates
        until 'finish' decision or max_tool_iterations is reached.
        """
        if self.llm_reasoner is None:
            from monitoring_agent.monitoring_llm_reasoner import GeminiMonitoringReasoner
            self.llm_reasoner = GeminiMonitoringReasoner()

        raw_df = target_state.buffer_dataframe()
        cleaned_df: pd.DataFrame | None = None
        baseline_df: pd.DataFrame | None = None
        deviation_df: pd.DataFrame | None = None
        trends_df: pd.DataFrame | None = None
        signal_quality_df: pd.DataFrame | None = None
        invalid_mask_df: pd.DataFrame | None = None

        executed_tools: list[str] = []
        tool_trace: list[dict[str, Any]] = []
        final_proposal = None

        for iteration in range(self.max_tool_iterations):
            state_summary = {
                "has_clean_values": cleaned_df is not None,
                "baseline_established": baseline_df is not None,
                "deviations_calculated": deviation_df is not None,
                "trends_calculated": trends_df is not None,
                "signal_quality_assessed": signal_quality_df is not None,
                "vitals_glance": {
                    v: float(latest_sample.get(v, np.nan))
                    for v in VITAL_COLUMNS
                    if v in latest_sample and pd.notna(latest_sample.get(v))
                },
            }

            # Gemini selects next analytical tool or decides to finish
            decision = self.llm_reasoner.select_tool(
                patient_id=target_state.case_id,
                observations_count=target_state.total_observations,
                latest_observation=latest_sample,
                executed_tools=executed_tools,
                state_summary=state_summary,
                iteration=iteration,
                max_iterations=self.max_tool_iterations,
            )

            # Record thought & chosen action
            step_record: dict[str, Any] = {
                "iteration": iteration + 1,
                "thought": decision.thought,
                "action": decision.action,
                "tool_name": decision.tool_name,
                "tool_args": decision.tool_args,
            }

            if decision.action == "finish":
                step_record["status"] = "finished"
                tool_trace.append(step_record)
                if decision.classification and decision.rationale:
                    from monitoring_agent.monitoring_llm_reasoner import MonitoringLLMProposal
                    final_proposal = MonitoringLLMProposal(
                        classification=decision.classification,
                        rationale=decision.rationale,
                        confidence=decision.confidence,
                        recommended_action=decision.final_decision or "CONTINUE_MONITORING",
                        raw_response=decision.raw_response,
                    )
                break

            tool_name = (decision.tool_name or "").strip()
            if not tool_name or tool_name not in self.tools:
                step_record["status"] = "error"
                step_record["error"] = f"Unknown tool: '{tool_name}'"
                tool_trace.append(step_record)
                break

            # Execute selected tool
            try:
                if tool_name == "assess_signal_quality":
                    window = decision.tool_args.get("window", 10) if isinstance(decision.tool_args, dict) else 10
                    signal_quality_df = self.tools[tool_name](df=raw_df, window=window)
                    output_summary = {v: str(signal_quality_df[v].iloc[-1]) for v in VITAL_COLUMNS if v in signal_quality_df}
                elif tool_name == "invalid_measurement_mask":
                    invalid_mask_df = self.tools[tool_name](df=raw_df)
                    output_summary = {v: bool(invalid_mask_df[v].iloc[-1]) for v in VITAL_COLUMNS if v in invalid_mask_df}
                elif tool_name == "preprocess":
                    cleaned_df = self.tools[tool_name](df=raw_df)
                    output_summary = {v: float(cleaned_df[v].iloc[-1]) for v in VITAL_COLUMNS if v in cleaned_df and pd.notna(cleaned_df[v].iloc[-1])}
                elif tool_name == "calculate_baseline":
                    input_df = cleaned_df if cleaned_df is not None else raw_df
                    baseline_df = self.tools[tool_name](df=input_df, window=self.baseline_window)
                    output_summary = {v: float(baseline_df[v].iloc[-1]) for v in VITAL_COLUMNS if v in baseline_df and pd.notna(baseline_df[v].iloc[-1])}
                elif tool_name == "calculate_deviation":
                    input_df = cleaned_df if cleaned_df is not None else raw_df
                    base_df = baseline_df if baseline_df is not None else (self.tools["calculate_baseline"](input_df, window=self.baseline_window))
                    baseline_df = base_df
                    deviation_df = self.tools[tool_name](df=input_df, baseline=base_df)
                    output_summary = {v: float(deviation_df[v].iloc[-1]) for v in VITAL_COLUMNS if v in deviation_df and pd.notna(deviation_df[v].iloc[-1])}
                elif tool_name == "calculate_trends":
                    input_df = cleaned_df if cleaned_df is not None else raw_df
                    trends_df = self.tools[tool_name](df=input_df)
                    output_summary = {v: str(trends_df[v].iloc[-1]) for v in VITAL_COLUMNS if v in trends_df}
                else:
                    output_summary = "Tool executed successfully"

                step_record["status"] = "success"
                step_record["output"] = output_summary
                executed_tools.append(tool_name)
            except Exception as err:
                step_record["status"] = "error"
                step_record["error"] = str(err)
                tool_trace.append(step_record)
                break

            tool_trace.append(step_record)

        # Fallback completion: Ensure any analytical steps not invoked by LLM are populated deterministically
        # to guarantee the decision engine receives complete and valid data structures.
        try:
            if cleaned_df is None:
                prep_tool = self.tools.get("preprocess")
                cleaned_df = prep_tool(raw_df) if prep_tool else preprocess_data(raw_df)
            if baseline_df is None:
                base_tool = self.tools.get("calculate_baseline")
                baseline_df = base_tool(cleaned_df, window=self.baseline_window) if base_tool else calculate_baseline(cleaned_df, window=self.baseline_window)
            if deviation_df is None:
                dev_tool = self.tools.get("calculate_deviation")
                deviation_df = dev_tool(cleaned_df, baseline_df) if dev_tool else calculate_deviation(cleaned_df, baseline_df)
            if trends_df is None:
                tr_tool = self.tools.get("calculate_trends")
                trends_df = tr_tool(cleaned_df) if tr_tool else calculate_trends(cleaned_df)
        except Exception:
            if cleaned_df is None:
                cleaned_df = raw_df.copy()
            if baseline_df is None:
                baseline_df = pd.DataFrame(np.nan, index=raw_df.index, columns=VITAL_COLUMNS)
            if deviation_df is None:
                deviation_df = pd.DataFrame(np.nan, index=raw_df.index, columns=VITAL_COLUMNS)
            if trends_df is None:
                trends_df = pd.DataFrame("insufficient_data", index=raw_df.index, columns=VITAL_COLUMNS)

        try:
            if signal_quality_df is None:
                sq_tool = self.tools.get("assess_signal_quality")
                signal_quality_df = sq_tool(raw_df) if sq_tool else assess_signal_quality(raw_df)
            if invalid_mask_df is None:
                inv_tool = self.tools.get("invalid_measurement_mask")
                invalid_mask_df = inv_tool(raw_df) if inv_tool else invalid_measurement_mask(raw_df)
        except Exception:
            if signal_quality_df is None:
                signal_quality_df = pd.DataFrame("good", index=raw_df.index, columns=VITAL_COLUMNS)
            if invalid_mask_df is None:
                invalid_mask_df = pd.DataFrame(False, index=raw_df.index, columns=VITAL_COLUMNS)

        return cleaned_df, baseline_df, deviation_df, trends_df, signal_quality_df, invalid_mask_df, tool_trace, final_proposal

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
