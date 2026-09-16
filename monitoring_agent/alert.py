"""Structured, non-diagnostic Monitoring Agent alert events."""

from __future__ import annotations

from typing import Any, Iterable, Mapping


def create_alert(
    case_id: int,
    timestamp: int | float,
    affected_vitals: Iterable[str],
    deviation_values: Mapping[str, float],
    duration: int,
    *,
    alert_state: str = "alert_started",
    vital_details: Mapping[str, Mapping[str, Any]] | None = None,
    reason: str = "Persistent multi-vital physiological deviation detected",
) -> dict:
    """Create a backward-compatible event for a future Risk Prediction Agent."""
    vitals = list(affected_vitals)
    event = {
        "case_id": int(case_id),
        "timestamp": int(timestamp) if float(timestamp).is_integer() else float(timestamp),
        "event": "physiological_deviation",
        "alert_type": "physiological_deviation",
        "alert_state": alert_state,
        "affected_vitals": vitals,
        "deviation_values": {vital: float(deviation_values[vital]) for vital in vitals},
        "duration_seconds": int(duration),
        "source": "Monitoring Agent",
        "data_source": "VitalDB",
        "reason": reason,
    }
    if vital_details is not None:
        event["vital_details"] = {vital: dict(vital_details[vital]) for vital in vitals}
    return event
