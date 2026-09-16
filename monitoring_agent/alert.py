"""Structured, non-diagnostic Monitoring Agent alert events."""

from __future__ import annotations

from typing import Iterable, Mapping


def create_alert(
    case_id: int,
    timestamp: int | float,
    affected_vitals: Iterable[str],
    deviation_values: Mapping[str, float],
    duration: int,
) -> dict:
    """Create an event for a future Risk Prediction Agent to consume."""
    vitals = list(affected_vitals)
    return {
        "case_id": int(case_id),
        "timestamp": int(timestamp) if float(timestamp).is_integer() else float(timestamp),
        "event": "physiological_deviation",
        "affected_vitals": vitals,
        "deviation_values": {vital: float(deviation_values[vital]) for vital in vitals},
        "duration_seconds": int(duration),
        "source": "Monitoring Agent",
    }
