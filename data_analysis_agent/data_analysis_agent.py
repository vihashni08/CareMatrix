"""Event-driven, non-diagnostic physiological analysis for CareMatrix."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import threading
import time
from typing import Any, Callable

import numpy as np
import pandas as pd

from communication.event_queue import EventQueue
from communication.events import DataAnalysisEvent, RiskDecisionEvent


WINDOW_SECONDS = 300
SAMPLE_INTERVAL = 5
VITAL_TRACKS = [
    "Solar8000/HR", "Solar8000/PLETH_SPO2", "Solar8000/RR",
    "Solar8000/NIBP_SBP", "Solar8000/NIBP_DBP", "Solar8000/NIBP_MBP",
    "Solar8000/BT",
]
VITAL_COLUMNS = ["HR", "SpO2", "RR", "SBP", "DBP", "MAP", "BT"]


@dataclass
class PatientAnalysisState:
    last_analysed_timestamp: float | None = None
    previous_trends: dict[str, str] = field(default_factory=dict)
    previous_data_quality_flag: bool | None = None
    previous_patterns: list[str] = field(default_factory=list)
    analysed_events: int = 0


class DataAnalysisAgent:
    """Consumes risk decisions, selects applicable analysis, and emits evidence.

    It deliberately reports measured temporal behaviour only; it does not assign
    diagnoses or replace the Random Forest risk classification.
    """

    def __init__(
        self,
        event_queue: EventQueue | None = None,
        data_loader: Callable[[int], pd.DataFrame] | None = None,
        name: str = "DataAnalysisAgent",
    ):
        self.name = name
        self.event_queue = event_queue
        self.data_loader = data_loader or self._load_case_data
        self.patient_states: dict[int, PatientAnalysisState] = {}
        self._worker_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._is_running = False

    @staticmethod
    def _normalise_frame(data: Any) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            frame = data.copy()
            if "Time" not in frame.columns:
                frame.insert(0, "Time", np.arange(len(frame)) * SAMPLE_INTERVAL)
        else:
            frame = pd.DataFrame(np.asarray(data), columns=VITAL_COLUMNS)
            frame.insert(0, "Time", np.arange(len(frame)) * SAMPLE_INTERVAL)
        for vital in VITAL_COLUMNS:
            if vital not in frame:
                frame[vital] = np.nan
            frame[vital] = pd.to_numeric(frame[vital], errors="coerce")
        frame["Time"] = pd.to_numeric(frame["Time"], errors="coerce")
        return frame.dropna(subset=["Time"])

    def _load_case_data(self, case_id: int) -> pd.DataFrame:
        """Use the existing VitalDB tracks, then its existing local-file fallback."""
        try:
            import vitaldb
            data = vitaldb.load_case(int(case_id), VITAL_TRACKS, interval=SAMPLE_INTERVAL)
            if data is not None and np.size(data):
                return self._normalise_frame(data)
        except Exception:
            pass
        local_path = Path(__file__).resolve().parent.parent / "data" / "processed" / f"patient_{case_id}.csv"
        if local_path.exists():
            return self._normalise_frame(pd.read_csv(local_path))
        raise ValueError(f"No VitalDB or local physiological data is available for case {case_id}.")

    @staticmethod
    def _trend(values: pd.Series) -> tuple[float, str]:
        if len(values) < 2:
            return 0.0, "insufficient_data"
        slope = float(np.polyfit(np.arange(len(values)), values.to_numpy(), 1)[0])
        scale = max(float(values.std(ddof=0)), abs(float(values.mean())) * 0.005, 1e-6)
        if abs(slope) <= scale * 0.05:
            return slope, "stable"
        return slope, "increasing" if slope > 0 else "decreasing"

    def _assess_quality(self, window: pd.DataFrame) -> tuple[bool, dict[str, Any]]:
        expected = max(2, int(WINDOW_SECONDS / SAMPLE_INTERVAL))
        details: dict[str, Any] = {"samples": int(len(window)), "expected_samples": expected, "vitals": {}}
        flagged = len(window) < 2
        for vital in VITAL_COLUMNS:
            values = window[vital]
            missing_ratio = float(values.isna().mean()) if len(values) else 1.0
            clean = values.dropna()
            artifact = False
            if len(clean) >= 3:
                jumps = clean.diff().abs().dropna()
                typical = float(jumps.median())
                artifact = bool((jumps > max(typical * 8, 1e-9)).any() and typical > 0)
            details["vitals"][vital] = {
                "missing_ratio": round(missing_ratio, 4), "valid_samples": int(len(clean)),
                "possible_artifact": artifact,
            }
            flagged = flagged or missing_ratio > 0.5 or artifact
        details["insufficient_samples"] = len(window) < expected * 0.25
        return bool(flagged), details

    def _metrics_and_patterns(self, window: pd.DataFrame, risk: RiskDecisionEvent) -> tuple[dict[str, Any], list[str], list[str]]:
        metrics: dict[str, Any] = {}
        patterns: list[str] = []
        changes: list[str] = []
        directions: list[str] = []
        for vital in VITAL_COLUMNS:
            values = window[vital].dropna()
            if values.empty:
                continue
            slope, direction = self._trend(values)
            change = float(values.iloc[-1] - values.iloc[0])
            std = float(values.std(ddof=0))
            metrics[vital] = {
                "mean": float(values.mean()), "minimum": float(values.min()), "maximum": float(values.max()),
                "standard_deviation": std, "latest_value": float(values.iloc[-1]),
                "change_over_window": change, "slope_per_sample": slope, "trend": direction,
                "sample_count": int(len(values)),
            }
            directions.append(direction)
            if vital in risk.affected_vitals or direction != "stable":
                changes.append(f"{vital}: {direction}; change over window {change:.3f}.")
            if direction != "stable" and vital in risk.affected_vitals:
                patterns.append(f"{vital} shows a {direction} trend in the assessment window.")
            if len(values) >= 3 and std > max(abs(float(values.mean())) * 0.10, 1e-6):
                patterns.append(f"{vital} has high variability in the assessment window.")
        changed_affected = [v for v in risk.affected_vitals if metrics.get(v, {}).get("trend") not in (None, "stable", "insufficient_data")]
        if len(changed_affected) >= 2:
            patterns.append("Simultaneous directional changes are present across multiple affected vitals.")
        if risk.risk_level == "LOW RISK":
            if changed_affected:
                patterns.append("LOW RISK classification retained; trends are reported as supporting evidence only.")
            else:
                patterns.append("LOW RISK classification retained; available affected-vital data is stable over this window.")
        elif risk.risk_level == "HIGH RISK":
            patterns.append("HIGH RISK classification retained; trend and data-quality evidence is provided for follow-up reasoning.")
        return metrics, list(dict.fromkeys(patterns)), changes

    def process_event(self, risk_event: RiskDecisionEvent) -> DataAnalysisEvent:
        """Perform one autonomous analysis cycle and always return a traceable event."""
        # Extract defensively before validation so a malformed queue item cannot
        # crash the worker before it has a chance to emit a partial result.
        case_id = int(getattr(risk_event, "patient_id", 0) or 0)
        event_id = str(getattr(risk_event, "event_id", "unknown_risk_event"))
        timestamp = float(getattr(risk_event, "timestamp", 0.0) or 0.0)
        risk_level = str(getattr(risk_event, "risk_level", "INDETERMINATE"))
        state = self.patient_states.setdefault(case_id, PatientAnalysisState())
        try:
            if not isinstance(risk_event, RiskDecisionEvent):
                raise TypeError("Expected RiskDecisionEvent")
            if not risk_event.event_id:
                raise ValueError("Risk assessment has no event_id")
            frame = self.data_loader(case_id)
            start = risk_event.window_start if risk_event.window_start is not None else max(0.0, risk_event.timestamp - WINDOW_SECONDS)
            end = risk_event.window_end if risk_event.window_end is not None else risk_event.timestamp
            window = frame.loc[(frame["Time"] >= start) & (frame["Time"] <= end), ["Time", *VITAL_COLUMNS]].copy()
            if window.empty:
                raise ValueError(f"No samples available in analysis window {start}–{end}.")
            quality_flag, quality = self._assess_quality(window)
            metrics, patterns, changes = self._metrics_and_patterns(window, risk_event)
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
            result = DataAnalysisEvent(case_id, event_id, timestamp, risk_level,
                                       metrics, patterns, changes, quality_flag, status, data_quality_details=quality)
            state.previous_trends = current_trends
            state.previous_data_quality_flag = quality_flag
            state.previous_patterns = patterns
        except Exception as exc:
            result = DataAnalysisEvent(case_id, event_id, timestamp, risk_level,
                                       {}, [], [], True, "partial_analysis", error_message=str(exc),
                                       data_quality_details={"error": str(exc)})
        state.last_analysed_timestamp = timestamp
        state.analysed_events += 1
        self._publish(result)
        return result

    def _publish(self, event: DataAnalysisEvent) -> None:
        if self.event_queue is not None:
            self.event_queue.publish("data_analysis_events", event)
            # This topic is deliberately the next agent's input boundary.
            self.event_queue.publish("clinical_reasoning_events", event)
            self.event_queue.publish("all_events", event)

    def start(self) -> None:
        """Start only when the existing asynchronous broker workflow is in use."""
        if self._is_running or self.event_queue is None:
            return
        self._stop_event.clear(); self._is_running = True
        self._worker_thread = threading.Thread(target=self._worker_loop, name=f"{self.name}-Worker", daemon=True)
        self._worker_thread.start()

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            event = self.event_queue.consume("risk_decisions", timeout=0.2) if self.event_queue else None
            if isinstance(event, RiskDecisionEvent):
                self.process_event(event)
        self._is_running = False

    def stop(self) -> None:
        self._stop_event.set(); self._is_running = False

    def join(self, timeout: float | None = 5.0) -> None:
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=timeout)
