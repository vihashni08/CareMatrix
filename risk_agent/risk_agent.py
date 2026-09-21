"""CareMatrix Risk Agent controller with genuine independent worker execution loop.

Lifecycle:
WAIT FOR EVENT → RECEIVE → VALIDATE → GATHER CONTEXT → INVOKE ML TOOL → EVALUATE → DECIDE → RESPOND → WAIT FOR NEXT EVENT
"""

from __future__ import annotations

import datetime
import json
import threading
import time
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd

from communication.event_queue import EventQueue
from communication.events import (
    AgentFailureEvent,
    AgentHeartbeatEvent,
    MonitoringEvent,
    RiskDecision,
    RiskDecisionEvent,
)
from risk_agent.decision_engine import RiskDecisionEngine
from risk_agent.state import RiskAgentState

VITAL_COLUMNS_7 = ["HR", "SpO2", "RR", "SBP", "DBP", "MAP", "BT"]


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
    ):
        self.name = name
        self.event_queue = event_queue
        self.max_retries = max_retries
        self.verbose = verbose

        # Model artifact loading
        if model_dir is None:
            self.model_dir = Path(__file__).resolve().parent / "models"
        else:
            self.model_dir = Path(model_dir)

        self._load_artifacts()

        # Dynamic model identification
        self.model_name = type(self.model).__name__

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

    def _load_artifacts(self) -> None:
        """Load trained ML model, preprocessor, and metadata artifacts."""
        model_path = self.model_dir / "best_model.pkl"
        preprocessor_path = self.model_dir / "preprocessor.pkl"
        features_path = self.model_dir / "feature_columns.json"
        threshold_path = self.model_dir / "risk_threshold.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Model file missing at: {model_path}")
        if not preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor file missing at: {preprocessor_path}")

        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)

        with open(features_path, "r", encoding="utf-8") as f:
            self.feature_columns = json.load(f)

        with open(threshold_path, "r", encoding="utf-8") as f:
            self.risk_threshold = float(json.load(f)["threshold"])

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
        if case_id in self.state.patient_context_cache:
            return self.state.patient_context_cache[case_id]

        context = {
            "age": np.nan,
            "sex": np.nan,
            "bmi": np.nan,
            "asa": np.nan,
            "emop": np.nan,
        }

        try:
            cases_df = pd.read_csv("https://api.vitaldb.net/cases", timeout=2.0)
            patient = cases_df.loc[cases_df["caseid"] == int(case_id)]
            if not patient.empty:
                row = patient.iloc[0]
                sex_str = str(row.get("sex", "")).upper()
                context["age"] = pd.to_numeric(row.get("age"), errors="coerce")
                context["sex"] = 1.0 if sex_str == "M" else 0.0 if sex_str == "F" else np.nan
                context["bmi"] = pd.to_numeric(row.get("bmi"), errors="coerce")
                context["asa"] = pd.to_numeric(row.get("asa"), errors="coerce")
                context["emop"] = pd.to_numeric(row.get("emop"), errors="coerce")
        except Exception:
            context["age"] = 60.0
            context["sex"] = 1.0
            context["bmi"] = 24.5
            context["asa"] = 2.0
            context["emop"] = 0.0

        self.state.patient_context_cache[case_id] = context
        return context

    def construct_features(
        self,
        event: MonitoringEvent,
        patient_context: dict[str, float],
        window_df: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        """Construct feature vector aligned with feature_columns.json."""
        features: dict[str, float] = {col: np.nan for col in self.feature_columns}

        if window_df is not None and not window_df.empty:
            for vital in VITAL_COLUMNS_7:
                if vital in window_df.columns:
                    vals = pd.to_numeric(window_df[vital], errors="coerce").dropna()
                    if not vals.empty:
                        features[f"{vital}_mean"] = float(vals.mean())
                        features[f"{vital}_min"] = float(vals.min())
                        features[f"{vital}_max"] = float(vals.max())
                        features[f"{vital}_std"] = float(vals.std()) if len(vals) > 1 else 0.0
                        features[f"{vital}_latest"] = float(vals.iloc[-1])
                        features[f"{vital}_change"] = float(vals.iloc[-1] - vals.iloc[0])
                        features[f"{vital}_slope"] = (
                            float(np.polyfit(np.arange(len(vals)), vals.to_numpy(), 1)[0])
                            if len(vals) > 1
                            else 0.0
                        )
        else:
            for vital in event.affected_vitals:
                curr = event.current_values.get(vital, np.nan)
                summary = event.vital_summary.get(vital, {})
                init = summary.get("initial_value", curr)
                mn = summary.get("minimum_value", curr)
                mx = summary.get("maximum_value", curr)

                features[f"{vital}_latest"] = float(curr)
                features[f"{vital}_min"] = float(mn)
                features[f"{vital}_max"] = float(mx)
                features[f"{vital}_mean"] = float((mn + mx + curr) / 3.0) if pd.notna(curr) else np.nan
                features[f"{vital}_std"] = float(abs(mx - mn) / 2.0) if pd.notna(mx) and pd.notna(mn) else 0.0
                features[f"{vital}_change"] = float(curr - init) if pd.notna(curr) and pd.notna(init) else 0.0

                trend = event.trends.get(vital, "stable")
                features[f"{vital}_slope"] = 1.0 if trend == "increasing" else -1.0 if trend == "decreasing" else 0.0

        features.update(patient_context)
        return features

    def use_ml_model_tool(self, features: dict[str, float]) -> float:
        """Invoke preprocessor and ML model tool to compute risk probability."""
        if self.inject_failure:
            raise RuntimeError("Simulated ML model tool inference failure for demo.")

        feature_frame = pd.DataFrame(
            [{feat: features.get(feat, np.nan) for feat in self.feature_columns}],
            columns=self.feature_columns,
        )
        processed = self.preprocessor.transform(feature_frame)
        probabilities = self.model.predict_proba(processed)
        return float(probabilities[0, 1])

    def process_event(
        self,
        event: MonitoringEvent,
        window_df: pd.DataFrame | None = None,
    ) -> RiskDecisionEvent:
        """Execute full Risk Agent cycle:

        RECEIVE EVENT -> VALIDATE -> GATHER CONTEXT -> USE ML TOOL -> EVALUATE -> DECIDE -> RESPOND
        """
        # 1. RECEIVE EVENT
        self.state.record_received_event(event)
        self.emit_heartbeat()

        if self.verbose:
            print(_format_log(self.name, "RECEIVE", f"event={event.event_id} (Severity: {event.severity})"))

        # 2. VALIDATE
        is_valid, validation_msg = self.decision_engine.validate_event(event)
        if not is_valid:
            err = ValueError(f"Invalid monitoring event: {validation_msg}")
            self.state.record_failure(event.event_id, str(err))
            decision_event = self.decision_engine.decide_from_failure(
                event=event,
                error=err,
                retry_count=self.max_retries,
                model_name=self.model_name,
            )
            self._respond(decision_event)
            return decision_event

        # 3. GATHER CONTEXT & 4. CONSTRUCT FEATURES & 5. USE ML TOOL (with Retries)
        retry_count = 0
        last_exception: Exception | None = None

        while retry_count <= self.max_retries:
            try:
                patient_context = self.gather_patient_context(event.patient_id)
                features = self.construct_features(event, patient_context, window_df=window_df)

                if self.verbose:
                    print(_format_log(self.name, "ML_TOOL", f"Invoking {self.model_name} on patient {event.patient_id}"))

                probability = self.use_ml_model_tool(features)

                # 6. EVALUATE & DECIDE
                decision_event = self.decision_engine.decide_from_prediction(
                    event=event,
                    probability=probability,
                    threshold=self.risk_threshold,
                    model_name=self.model_name,
                    features=features,
                )

                if self.verbose:
                    print(_format_log(self.name, "DECIDE", f"{decision_event.decision} (Prob: {probability*100:.2f}% | Thresh: {self.risk_threshold:.2f})"))

                # 7. RESPOND
                self.state.record_decision(decision_event)
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
        fail_event = self.decision_engine.decide_from_failure(
            event=event,
            error=last_exception if last_exception is not None else RuntimeError("Unknown error"),
            retry_count=retry_count,
            model_name=self.model_name,
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

    def _respond(self, decision_event: RiskDecisionEvent) -> None:
        """Publish decision back to event queue."""
        if self.event_queue is not None:
            if self.verbose:
                print(_format_log(self.name, "PUBLISH", f"decision={decision_event.decision} (Event: {decision_event.event_id})"))
            self.event_queue.publish("risk_decisions", decision_event)
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

            event = self.event_queue.consume("monitoring_events", timeout=0.2)
            if event is None:
                self.emit_heartbeat()
                continue

            # Process only escalation events
            if isinstance(event, MonitoringEvent) and event.event_type == "alert_started":
                self.process_event(event)

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
