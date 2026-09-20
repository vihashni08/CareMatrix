"""Risk Prediction Agent for CareMatrix (Backward compatibility wrapper around RiskAgent)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from communication.events import MonitoringEvent, RiskDecisionEvent
from risk_agent.risk_agent import RiskAgent

WINDOW_SECONDS = 300
SAMPLE_INTERVAL = 5
MAX_PREDICTION_MINUTES = 60

VITAL_TRACKS = [
    "Solar8000/HR",
    "Solar8000/PLETH_SPO2",
    "Solar8000/RR",
    "Solar8000/NIBP_SBP",
    "Solar8000/NIBP_DBP",
    "Solar8000/NIBP_MBP",
    "Solar8000/BT",
]

VITAL_COLUMNS = ["HR", "SpO2", "RR", "SBP", "DBP", "MAP", "BT"]


class RiskPredictionAgent:
    """Risk Prediction Agent wrapper integrating the genuine RiskAgent controller."""

    def __init__(self):
        self._model_dir = Path(__file__).parent / "models"
        self.risk_agent = RiskAgent(model_dir=self._model_dir)

        # Expose legacy attributes for compatibility
        self.model = self.risk_agent.model
        self.preprocessor = self.risk_agent.preprocessor
        self.feature_columns = self.risk_agent.feature_columns
        self.risk_threshold = self.risk_agent.risk_threshold
        self.model_name = self.risk_agent.model_name

        print(f"Risk Prediction Agent initialized (Tool: {self.model_name})")
        print("Ready to classify monitoring results.")

    def predict_risk_probability(self, features: dict[str, Any]) -> float:
        """Return high-risk probability using the actual ML tool."""
        return self.risk_agent.use_ml_model_tool(features)

    def is_model_high_risk(self, features: dict[str, Any]) -> bool:
        """Classify a complete feature row using the saved training threshold."""
        return self.predict_risk_probability(features) >= self.risk_threshold

    @staticmethod
    def _extract_window_features(window: pd.DataFrame) -> dict[str, float]:
        """Create the same vital-sign summary features used during training."""
        features: dict[str, float] = {}
        for vital in VITAL_COLUMNS:
            if vital not in window.columns:
                continue
            values = pd.to_numeric(window[vital], errors="coerce").dropna()
            if values.empty:
                continue

            features[f"{vital}_mean"] = float(values.mean())
            features[f"{vital}_min"] = float(values.min())
            features[f"{vital}_max"] = float(values.max())
            features[f"{vital}_std"] = float(values.std()) if len(values) > 1 else 0.0
            features[f"{vital}_latest"] = float(values.iloc[-1])
            features[f"{vital}_change"] = float(values.iloc[-1] - values.iloc[0])
            features[f"{vital}_slope"] = (
                float(np.polyfit(np.arange(len(values)), values.to_numpy(), 1)[0])
                if len(values) > 1
                else 0.0
            )
        return features

    def _patient_metadata(self, case_id: int) -> dict[str, float]:
        """Load demographic features with offline resilience."""
        return self.risk_agent.gather_patient_context(case_id)

    def _case_windows(self, case_id: int) -> list[dict[str, Any]]:
        """Load seven-vital data and produce training-compatible five-minute rows."""
        data = None
        try:
            import vitaldb
            data = vitaldb.load_case(int(case_id), VITAL_TRACKS, interval=SAMPLE_INTERVAL)
        except Exception:
            pass

        if data is None or np.size(data) == 0:
            # Fallback to local processed patient file if available
            local_path = Path(__file__).resolve().parent.parent / "data" / "processed" / f"patient_{case_id}.csv"
            if local_path.exists():
                local_df = pd.read_csv(local_path)
                frame = pd.DataFrame(index=local_df.index)
                for col in VITAL_COLUMNS:
                    frame[col] = local_df[col] if col in local_df.columns else np.nan
                frame["Time"] = np.arange(len(frame))
                data = frame
            else:
                raise ValueError(f"No vital data is available for case {case_id}.")

        if isinstance(data, pd.DataFrame):
            frame = data.copy()
            if "Time" not in frame.columns:
                frame.insert(0, "Time", np.arange(len(frame)) * SAMPLE_INTERVAL)
        else:
            frame = pd.DataFrame(np.asarray(data), columns=VITAL_COLUMNS)
            frame.insert(0, "Time", np.arange(len(frame)) * SAMPLE_INTERVAL)

        frame = frame.loc[frame["Time"] <= MAX_PREDICTION_MINUTES * 60].copy()
        metadata = self._patient_metadata(case_id)
        windows: list[dict[str, Any]] = []

        for start in range(0, int(frame["Time"].max() if not frame.empty else 0) + 1, WINDOW_SECONDS):
            end = start + WINDOW_SECONDS
            window = frame.loc[(frame["Time"] >= start) & (frame["Time"] < end)]
            if window.empty:
                continue
            features = self._extract_window_features(window)
            features.update(metadata)
            windows.append({
                "features": features,
                "window_start": start,
                "window_end": end,
            })
        return windows

    def _predict_case(self, case_id: int, monitoring_output: dict[str, Any]) -> list[dict[str, Any]]:
        """Return predictions using the actual trained model tool on patient windows."""
        alerts = monitoring_output.get("alerts", [])
        results: list[dict[str, Any]] = []

        try:
            windows = self._case_windows(case_id)
        except Exception:
            windows = []

        if not windows and alerts:
            # Evaluate alerts directly with RiskAgent
            for index, alert in enumerate(alerts, start=1):
                res = self.classify_alert(alert, case_id=case_id)
                res["alert_number"] = index
                results.append(res)
            return results

        for index, window in enumerate(windows, start=1):
            probability = self.predict_risk_probability(window["features"])
            window_alerts = [
                alert
                for alert in alerts
                if window["window_start"]
                <= alert.get("timestamp", -1)
                < window["window_end"]
            ]
            affected_vitals = sorted({
                vital
                for alert in window_alerts
                for vital in alert.get("affected_vitals", [])
            })
            severity_order = {"moderate": 1, "severe": 2, "critical": 3}
            severity = max(
                (alert.get("severity", "unknown") for alert in window_alerts),
                key=lambda value: severity_order.get(value, 0),
                default="none",
            )
            risk_level = "HIGH RISK" if probability >= self.risk_threshold else "LOW RISK"
            results.append({
                "alert_number": index,
                "event_id": f"case_{case_id}_window_{index:03d}",
                "deviation_detected": bool(window_alerts),
                "risk_probability": probability,
                "risk_level": risk_level,
                "model": self.model_name,
                "threshold": self.risk_threshold,
                "window_start": window["window_start"],
                "window_end": window["window_end"],
                "window_samples": WINDOW_SECONDS // SAMPLE_INTERVAL,
                "severity": severity,
                "affected_vitals": affected_vitals,
                "reason": f"{self.model_name} inference from 5-minute patient vital window.",
                "source": "Risk Agent",
            })
        return results

    def classify_alert(self, alert: dict[str, Any], case_id: int | None = None) -> dict[str, Any]:
        """Classify one monitoring alert using the actual trained model tool."""
        patient_id = case_id if case_id is not None else alert.get("case_id", 0)

        # Construct MonitoringEvent from alert dictionary
        monitoring_event = MonitoringEvent(
            patient_id=int(patient_id),
            event_id=alert.get("event_id", "alert_unknown"),
            timestamp=float(alert.get("timestamp", 0)),
            event_type=alert.get("alert_state", "alert_started"),
            severity=alert.get("severity", "moderate"),
            affected_vitals=list(alert.get("affected_vitals", [])),
            current_values={
                v: alert["vital_details"][v]["current"]
                for v in alert.get("affected_vitals", [])
                if "vital_details" in alert and v in alert["vital_details"]
            },
            baseline_values={
                v: alert["vital_details"][v]["baseline"]
                for v in alert.get("affected_vitals", [])
                if "vital_details" in alert and v in alert["vital_details"]
            },
            deviation_values=dict(alert.get("deviation_values", {})),
            trends={
                v: alert["vital_details"][v]["trend"]
                for v in alert.get("affected_vitals", [])
                if "vital_details" in alert and v in alert["vital_details"]
            },
            signal_quality={
                v: alert["vital_details"][v]["signal_quality"]
                for v in alert.get("affected_vitals", [])
                if "vital_details" in alert and v in alert["vital_details"]
            },
            persistence_duration=int(alert.get("duration_seconds", 0)),
            recommended_action="assess_patient_risk",
            vital_details=alert.get("vital_details", {}),
            vital_summary=alert.get("vital_summary", {}),
            metadata={"source_alert": alert},
        )

        decision_event = self.risk_agent.process_event(monitoring_event)

        return {
            "event_id": decision_event.event_id,
            "deviation_detected": True,
            "risk_probability": decision_event.risk_probability,
            "risk_level": decision_event.risk_level,
            "decision": decision_event.decision,
            "model": self.model_name,
            "threshold": self.risk_threshold,
            "reason": decision_event.reason,
            "source": "Risk Agent",
            "recommended_action": decision_event.recommended_action,
        }

    def process_monitoring_output(
        self,
        monitoring_output: dict[str, Any],
        patient_data: Any | None = None,
        *,
        case_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """Classify monitoring output using the genuine RiskAgent controller."""
        if case_id is not None:
            return self._predict_case(case_id, monitoring_output)

        alerts = monitoring_output.get("alerts", [])
        results = []

        if not alerts:
            results.append({
                "alert_number": None,
                "event_id": None,
                "deviation_detected": False,
                "risk_probability": 0.0,
                "risk_level": "LOW RISK",
                "model": self.model_name,
                "threshold": self.risk_threshold,
                "reason": "No persistent physiological deviation detected by Monitoring Agent.",
                "source": "Risk Agent",
            })
            return results

        for index, alert in enumerate(alerts, start=1):
            res = self.classify_alert(alert)
            res["alert_number"] = index
            results.append(res)

        return results
