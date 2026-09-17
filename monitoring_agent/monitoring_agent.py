"""Simple patient-specific physiological deviation Monitoring Agent."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:  # Supports both package imports and direct script execution.
    from .alert import create_alert
    from .baseline import calculate_baseline
    from .deviation_detector import (
        DEVIATION_THRESHOLDS,
        apply_persistence,
        calculate_deviation,
    )
    from .alert_state import PhysiologicalEventTracker, manage_alert_lifecycle
    from .config import (
        BASELINE_WINDOW_SECONDS,
        PERSISTENCE_SECONDS,
        RECOVERY_DURATION_SECONDS,
        SEVERITY_MIN_VITALS,
        SEVERITY_MULTIPLIERS,
        SEVERITY_PERSISTENCE,
    )
    from .preprocessing import preprocess_data
    from .signal_quality import assess_signal_quality, invalid_measurement_mask
    from .trend_detector import calculate_trends
except ImportError:  # pragma: no cover - convenience for direct execution
    from alert import create_alert
    from baseline import calculate_baseline
    from deviation_detector import DEVIATION_THRESHOLDS, apply_persistence, calculate_deviation
    from alert_state import PhysiologicalEventTracker, manage_alert_lifecycle
    from config import (
        BASELINE_WINDOW_SECONDS, PERSISTENCE_SECONDS, RECOVERY_DURATION_SECONDS,
        SEVERITY_MIN_VITALS, SEVERITY_MULTIPLIERS, SEVERITY_PERSISTENCE,
    )
    from preprocessing import preprocess_data
    from signal_quality import assess_signal_quality, invalid_measurement_mask
    from trend_detector import calculate_trends


TRACK_MAPPING = {
    "Solar8000/HR": "HR",
    "Solar8000/ART_MBP": "MAP",
    "Solar8000/PLETH_SPO2": "SpO2",
    "Solar8000/RR": "RR",
}
VITAL_COLUMNS = list(TRACK_MAPPING.values())

# Ordered severity levels for comparison operations.
SEVERITY_LEVELS = ("mild", "moderate", "severe", "critical")
_SEVERITY_RANK = {level: rank for rank, level in enumerate(SEVERITY_LEVELS)}
_SEVERITY_NUM = {"normal": 0, "mild": 1, "moderate": 2, "severe": 3, "critical": 4}
_NUM_SEVERITY = {v: k for k, v in _SEVERITY_NUM.items()}


def _apply_severity_aware_persistence(
    candidate: pd.Series,
    severity: pd.Series,
    severity_persistence: dict[str, int] = SEVERITY_PERSISTENCE,
) -> pd.Series:
    """Mark candidate samples as persistent using severity-dependent durations.

    Within a consecutive candidate run, persistence is achieved when the run
    length reaches the persistence requirement of the *current* severity at
    that sample.  Once achieved, persistence remains true for the rest of the
    run.  This is a prototype implementation suitable for second-resolution data.
    """
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
    """Load one VitalDB case as one-second HR, MAP, SpO2, and RR samples."""
    vitaldb = _get_vitaldb()
    try:
        values = vitaldb.load_case(int(case_id), list(TRACK_MAPPING), interval=1)
    except Exception as exc:
        raise RuntimeError(
            f"Unable to load VitalDB case {case_id}. Check the case ID and network connection."
        ) from exc

    if values is None or np.size(values) == 0:
        raise ValueError(f"VitalDB case {case_id} returned no data for the required tracks.")

    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] != len(VITAL_COLUMNS):
        raise ValueError(
            f"VitalDB case {case_id} returned an unexpected shape {values.shape}; "
            f"expected N rows by {len(VITAL_COLUMNS)} tracks."
        )
    df = pd.DataFrame(values, columns=VITAL_COLUMNS)
    df.index = pd.RangeIndex(start=0, stop=len(df), step=1, name="timestamp")
    return df


def monitor_dataframe(
    case_id: int,
    df: pd.DataFrame,
    baseline_window: int = BASELINE_WINDOW_SECONDS,
    persistence_duration: int = PERSISTENCE_SECONDS,
) -> dict[str, Any]:
    """Run the full monitoring pipeline on a prepared patient dataframe.

    This function enables offline/synthetic tests without changing the live
    VitalDB workflow used by :func:`run_monitoring`.
    """
    missing = [column for column in VITAL_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError("Input data is missing required columns: " + ", ".join(missing))

    raw_df = df.loc[:, VITAL_COLUMNS].copy()
    if raw_df.empty:
        raise ValueError("Cannot monitor empty case data.")
    if raw_df.index.name is None:
        raw_df.index.name = "timestamp"

    # Quality uses raw values so invalid readings and isolated spikes remain
    # visible as metadata; preprocessing retains its existing clean data output.
    signal_quality = assess_signal_quality(raw_df)
    invalid_measurements = invalid_measurement_mask(raw_df)
    processed_df = preprocess_data(raw_df)
    baseline = calculate_baseline(processed_df, window=baseline_window)
    deviation = calculate_deviation(processed_df, baseline)
    trends = calculate_trends(processed_df)

    baseline_ready = baseline.notna().all(axis=1)

    # --- Per-vital severity classification ---
    # Each vital's deviation is compared against tiered thresholds built from
    # the base DEVIATION_THRESHOLDS and configurable SEVERITY_MULTIPLIERS.
    # Invalid measurements and samples without an established baseline are
    # excluded so that severity metadata in the result is always meaningful.
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

    # Overall severity = maximum severity across all vitals at each timestamp.
    vital_severity_numeric = vital_severity.apply(lambda col: col.map(_SEVERITY_NUM))
    overall_severity = (
        vital_severity_numeric.max(axis=1).map(_NUM_SEVERITY).rename("overall_severity")
    )

    # A vital is alert-eligible if its severity is above "normal" (it exceeds
    # its base deviation threshold and is not an invalid measurement).
    alert_eligible = vital_severity.ne("normal")
    deviation_count = alert_eligible.sum(axis=1).astype(int)

    # Candidate: the sample has enough deviating vitals for its severity level.
    # Severe/critical allow a single vital; mild/moderate require at least two.
    candidate_alert = pd.Series(False, index=processed_df.index, dtype=bool)
    for sev_name in SEVERITY_LEVELS:
        mask = overall_severity == sev_name
        min_v = SEVERITY_MIN_VITALS[sev_name]
        candidate_alert.loc[mask] = (
            (deviation_count[mask] >= min_v) & baseline_ready[mask]
        )
    candidate_alert = candidate_alert.rename("candidate_alert")

    # Persistence uses severity-dependent durations: critical deviations trigger
    # with minimal delay while mild deviations must sustain for longer.
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

    # Remember the first sample of each candidate run. When persistence later
    # confirms an event, its compact history begins at the true deviation onset
    # rather than only at the tenth confirmation sample.
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

    # Preserve ``alerts`` as started events for compatibility. Lifecycle events
    # additionally include recoveries, while the result dataframe shows active
    # state at every timestamp without emitting duplicate notifications.
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
        """Count candidate runs that never meet the existing persistence rule."""
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
