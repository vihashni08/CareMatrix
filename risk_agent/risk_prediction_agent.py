from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


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
    """
    Risk Prediction Agent for CareMatrix.

    Current implementation:
        Persistent physiological deviation detected -> HIGH RISK
        No persistent physiological deviation -> LOW RISK
    """

    def __init__(self):
        self._model_dir = Path(__file__).parent / "models"
        self.model = joblib.load(self._model_dir / "best_model.pkl")
        self.preprocessor = joblib.load(self._model_dir / "preprocessor.pkl")
        self.feature_columns = json.loads(
            (self._model_dir / "feature_columns.json").read_text(encoding="utf-8")
        )
        self.risk_threshold = json.loads(
            (self._model_dir / "risk_threshold.json").read_text(encoding="utf-8")
        )["threshold"]

        print("Risk Prediction Agent initialized")
        print("Ready to classify monitoring results.")

    def predict_risk_probability(self, features: dict[str, Any]) -> float:
        """Return the Random Forest high-risk probability for one feature row.

        Missing vital features are represented as ``NaN`` so the saved median
        imputer handles them exactly as it did during training.
        """
        feature_frame = pd.DataFrame(
            [{feature: features.get(feature, np.nan) for feature in self.feature_columns}],
            columns=self.feature_columns,
        )
        processed_features = self.preprocessor.transform(feature_frame)
        return float(self.model.predict_proba(processed_features)[0, 1])

    def is_model_high_risk(self, features: dict[str, Any]) -> bool:
        """Classify a complete feature row using the saved training threshold."""
        return self.predict_risk_probability(features) >= self.risk_threshold

    @staticmethod
    def _extract_window_features(window: pd.DataFrame) -> dict[str, float]:
        """Create the same vital-sign summary features used during training."""
        features: dict[str, float] = {}
        for vital in VITAL_COLUMNS:
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

    @staticmethod
    def _patient_metadata(case_id: int) -> dict[str, float]:
        """Load the demographic features included when the model was trained."""
        cases = pd.read_csv("https://api.vitaldb.net/cases")
        patient = cases.loc[cases["caseid"] == int(case_id)]
        if patient.empty:
            raise ValueError(f"Patient case {case_id} was not found in VitalDB.")

        row = patient.iloc[0]
        sex = str(row.get("sex", "")).upper()
        return {
            "age": pd.to_numeric(row.get("age"), errors="coerce"),
            "sex": 1.0 if sex == "M" else 0.0 if sex == "F" else np.nan,
            "bmi": pd.to_numeric(row.get("bmi"), errors="coerce"),
            "asa": pd.to_numeric(row.get("asa"), errors="coerce"),
            "emop": pd.to_numeric(row.get("emop"), errors="coerce"),
        }

    def _case_windows(self, case_id: int) -> list[dict[str, Any]]:
        """Load seven-vital data and produce training-compatible five-minute rows."""
        try:
            import vitaldb
        except ImportError as exc:
            raise RuntimeError("VitalDB is required for Random Forest prediction.") from exc

        data = vitaldb.load_case(int(case_id), VITAL_TRACKS, interval=SAMPLE_INTERVAL)
        if data is None or np.size(data) == 0:
            raise ValueError(f"No seven-vital data is available for case {case_id}.")

        frame = pd.DataFrame(np.asarray(data), columns=VITAL_COLUMNS)
        frame.insert(0, "Time", np.arange(len(frame)) * SAMPLE_INTERVAL)
        frame = frame.loc[frame["Time"] <= MAX_PREDICTION_MINUTES * 60].copy()
        metadata = self._patient_metadata(case_id)
        windows: list[dict[str, Any]] = []

        for start in range(0, MAX_PREDICTION_MINUTES * 60, WINDOW_SECONDS):
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
        """Return one Random Forest prediction for each available five-minute window."""
        alerts = monitoring_output.get("alerts", [])
        results: list[dict[str, Any]] = []
        for index, window in enumerate(self._case_windows(case_id), start=1):
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
            results.append({
                "alert_number": index,
                "event_id": f"case_{case_id}_window_{index:03d}",
                "deviation_detected": bool(window_alerts),
                "risk_probability": probability,
                "risk_level": "HIGH RISK" if probability >= self.risk_threshold else "LOW RISK",
                "model": "Random Forest",
                "threshold": self.risk_threshold,
                "window_start": window["window_start"],
                "window_end": window["window_end"],
                "window_samples": WINDOW_SECONDS // SAMPLE_INTERVAL,
                "severity": severity,
                "affected_vitals": affected_vitals,
                "reason": "Random Forest prediction from a 5-minute seven-vital window.",
                "source": "Risk Prediction Agent",
            })
        return results

    def classify_alert(self, alert: dict[str, Any]) -> dict[str, Any]:
        """
        Classify one monitoring alert.

        In the current implementation, every alert generated by
        the Monitoring Agent represents a persistent deviation,
        so it is classified as HIGH RISK.
        """

        return {
            "deviation_detected": True,
            "risk_level": "HIGH RISK",
            "reason": (
                "A persistent physiological deviation was detected "
                "by the Monitoring Agent."
            ),
            "source": "Risk Prediction Agent",
        }

    def process_monitoring_output(
        self,
        monitoring_output: dict[str, Any],
        patient_data: Any | None = None,
        *,
        case_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """Classify monitoring alerts.

        With ``case_id``, the saved Random Forest predicts on seven-vital,
        five-minute patient windows. Without it, the original alert-only
        behavior is retained for backward compatibility.
        """

        if case_id is not None:
            return self._predict_case(case_id, monitoring_output)

        alerts = monitoring_output.get("alerts", [])

        results = []

        # No alerts -> LOW RISK
        if not alerts:
            results.append({
                "alert_number": None,
                "event_id": None,
                "deviation_detected": False,
                "risk_level": "LOW RISK",
                "reason": (
                    "No persistent physiological deviation "
                    "was detected by the Monitoring Agent."
                ),
                "source": "Risk Prediction Agent",
            })

            return results

        # Alerts present -> classify each alert
        for index, alert in enumerate(alerts, start=1):

            result = self.classify_alert(alert)

            result["alert_number"] = index
            result["event_id"] = alert.get(
                "event_id",
                f"alert_{index}"
            )

            results.append(result)

        return results


def print_results(title, results):

    print("\n")
    print("=" * 70)
    print(title)
    print("=" * 70)

    for result in results:

        if result["alert_number"] is None:
            print("\nCase-Level Result")
        else:
            print(f"\nAlert {result['alert_number']}")

        if result["event_id"]:
            print(f"Event ID            : {result['event_id']}")

        print(
            "Deviation Detected  : "
            f"{'YES' if result['deviation_detected'] else 'NO'}"
        )

        print(f"Risk Level          : {result['risk_level']}")
        print(f"Reason              : {result['reason']}")
        print(f"Source              : {result['source']}")


if __name__ == "__main__":

    agent = RiskPredictionAgent()

    # ============================================================
    # EXAMPLE 1: NO ALERT
    # ============================================================

    no_alerts = {
        "alerts": []
    }

    results = agent.process_monitoring_output(no_alerts)

    print_results(
        "EXAMPLE 1 - NO PERSISTENT DEVIATION",
        results
    )


    # ============================================================
    # EXAMPLE 2: ONE ALERT
    # ============================================================

    one_alert = {
        "alerts": [
            {
                "event_id": "case_4_event_001",
                "event": "physiological_deviation",
                "alert_type": "physiological_deviation"
            }
        ]
    }

    results = agent.process_monitoring_output(one_alert)

    print_results(
        "EXAMPLE 2 - SINGLE DEVIATION",
        results
    )


    # ============================================================
    # EXAMPLE 3: MULTIPLE ALERTS
    # ============================================================

    multiple_alerts = {
        "alerts": [
            {
                "event_id": "case_4_event_001",
                "event": "physiological_deviation"
            },
            {
                "event_id": "case_4_event_002",
                "event": "physiological_deviation"
            },
            {
                "event_id": "case_4_event_003",
                "event": "physiological_deviation"
            },
            {
                "event_id": "case_4_event_004",
                "event": "physiological_deviation"
            },
            {
                "event_id": "case_4_event_005",
                "event": "physiological_deviation"
            }
        ]
    }

    results = agent.process_monitoring_output(multiple_alerts)

    print_results(
        "EXAMPLE 3 - MULTIPLE DEVIATIONS",
        results
    )


    # ============================================================
    # EXAMPLE 4: DESIRED MIXED OUTPUT FORMAT
    # ============================================================
    # This is ONLY a demonstration of how mixed risk results
    # would look in an ML-based implementation.
    # It does NOT pretend to calculate actual risk.

    print("\n")
    print("=" * 70)
    print("EXAMPLE 4 - MIXED LOW / HIGH RISK OUTPUT FORMAT")
    print("=" * 70)

    demo_results = [
        {
            "alert_number": 1,
            "event_id": "case_4_event_001",
            "deviation_detected": True,
            "risk_probability": 0.18,
            "risk_level": "LOW RISK",
            "reason": "Model classified this alert as Low Risk."
        },
        {
            "alert_number": 2,
            "event_id": "case_4_event_002",
            "deviation_detected": True,
            "risk_probability": 0.31,
            "risk_level": "LOW RISK",
            "reason": "Model classified this alert as Low Risk."
        },
        {
            "alert_number": 3,
            "event_id": "case_4_event_003",
            "deviation_detected": True,
            "risk_probability": 0.76,
            "risk_level": "HIGH RISK",
            "reason": "Model classified this alert as High Risk."
        },
        {
            "alert_number": 4,
            "event_id": "case_4_event_004",
            "deviation_detected": True,
            "risk_probability": 0.84,
            "risk_level": "HIGH RISK",
            "reason": "Model classified this alert as High Risk."
        },
        {
            "alert_number": 5,
            "event_id": "case_4_event_005",
            "deviation_detected": True,
            "risk_probability": 0.91,
            "risk_level": "HIGH RISK",
            "reason": "Model classified this alert as High Risk."
        }
    ]

    for result in demo_results:

        print(f"\nAlert {result['alert_number']}")
        print(f"Event ID            : {result['event_id']}")
        print(
            "Deviation Detected  : "
            f"{'YES' if result['deviation_detected'] else 'NO'}"
        )
        print(
            f"Risk Probability    : "
            f"{result['risk_probability'] * 100:.2f}%"
        )
        print(f"Risk Level          : {result['risk_level']}")
        print(f"Reason              : {result['reason']}")

    print("\n")
    print("=" * 70)
    print("END OF RISK PREDICTION AGENT EXAMPLES")
    print("=" * 70)
