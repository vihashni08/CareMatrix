"""Deterministic checks for Phase 1 monitoring enhancements (no test framework)."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from monitoring_agent.monitoring_agent import monitor_dataframe
from monitoring_agent.signal_quality import assess_signal_quality
from monitoring_agent.trend_detector import classify_trend


def patient_frame(samples: int = 160) -> pd.DataFrame:
    """Build stable one-second baseline data for focused monitoring tests."""
    return pd.DataFrame(
        {
            "HR": np.full(samples, 70.0),
            "MAP": np.full(samples, 85.0),
            "SpO2": np.full(samples, 98.0),
            "RR": np.full(samples, 14.0),
        },
        index=pd.RangeIndex(samples, name="timestamp"),
    )


def test_trends() -> None:
    assert classify_trend([70, 72, 75, 78, 82, 86, 90]) == "increasing"
    assert classify_trend([25, 23, 21, 20, 18, 17, 16]) == "decreasing"
    assert classify_trend([70, 71, 70, 69, 70, 71, 70]) == "stable"
    assert classify_trend([70, 71]) == "insufficient_data"
    print("Trend detection: PASS")


def test_existing_stable_data() -> None:
    results = monitor_dataframe(case_id=100, df=patient_frame(100))
    assert results["alerts"] == []
    print("Existing stable synthetic data: PASS (0 alert_started events)")


def test_existing_persistence_data() -> None:
    values = patient_frame(100)
    values.loc[70:80, ["HR", "MAP", "SpO2"]] = [95.0, 60.0, 90.0]
    results = monitor_dataframe(case_id=101, df=values)
    assert len(results["alerts"]) == 1
    assert results["alerts"][0]["alert_state"] == "alert_started"
    assert "severity" in results["alerts"][0]
    print("Existing persistence synthetic data: PASS (exactly 1 alert_started event)")


def test_artifact_detection() -> None:
    values = patient_frame(7)
    values["MAP"] = [75, 76, 75, 250, 76, 75, 76]
    quality = assess_signal_quality(values)
    assert quality.at[3, "MAP"] == "suspicious"
    print("Artifact detection: PASS")


def test_lifecycle() -> None:
    values = patient_frame()
    # First sustained event: two vitals deviate for 15 seconds, then recover.
    values.loc[70:84, ["HR", "MAP"]] = [95.0, 60.0]
    # A second event after recovery must be treated as new, not a duplicate.
    values.loc[110:124, ["HR", "MAP"]] = [95.0, 60.0]
    results = monitor_dataframe(case_id=123, df=values)
    started = [event for event in results["lifecycle_events"] if event["alert_state"] == "alert_started"]
    recovered = [event for event in results["lifecycle_events"] if event["alert_state"] == "alert_recovered"]
    assert len(results["alerts"]) == 2
    assert len(started) == 2
    assert len(recovered) == 2
    # HR moderate (0.357) + MAP mild (0.294) → overall moderate → persistence 8
    assert results["monitoring_results"].at[77, "alert_state"] == "alert_started"
    assert results["monitoring_results"].at[85, "alert_state"] == "recovering"
    assert results["monitoring_results"].at[89, "alert_state"] == "alert_recovered"
    assert results["monitoring_results"].at[117, "alert_state"] == "alert_started"
    assert started[0]["event_id"] == recovered[0]["event_id"]
    print("Persistence, deduplication, recovery, and new-event lifecycle: PASS")


def test_event_evolution() -> None:
    values = patient_frame(100)
    values.loc[70:79, "HR"] = [90, 98, 105, 115, 110, 100, 100, 100, 100, 100]
    values.loc[70:79, "MAP"] = 60.0
    results = monitor_dataframe(case_id=129, df=values)
    assert len(results["alerts"]) == 1
    summary = results["alerts"][0]["vital_summary"]["HR"]
    assert summary["initial_value"] == 90.0
    assert summary["maximum_value"] == 115.0
    # Severity-aware persistence triggers at 5s (position 74), where HR is 110.0,
    # while the earlier peak (115.0) remains preserved in maximum_value.
    assert summary["latest_value"] == 110.0
    assert summary["peak_absolute_deviation"] > summary["initial_relative_deviation"]
    print("Event evolution: PASS (initial, peak, and latest values preserved)")


def test_active_event_snapshot() -> None:
    values = patient_frame(96)
    values.loc[70:95, "HR"] = np.linspace(90.0, 115.0, 26)
    values.loc[70:95, "MAP"] = 60.0
    results = monitor_dataframe(case_id=130, df=values)
    assert len(results["alerts"]) == 1
    assert len(results["active_events"]) == 1
    active = results["active_events"][0]
    assert active["event_state"] == "alert_active"
    assert active["vital_summary"]["HR"]["latest_value"] == 115.0
    assert active["vital_summary"]["HR"]["current_trend"] == "increasing"
    print("Active event snapshot: PASS (latest value and current trend exposed)")


def test_decreasing_recovery_summary() -> None:
    values = patient_frame(100)
    values.loc[70:79, ["HR", "MAP"]] = [115.0, 60.0]
    values.loc[80:88, "HR"] = [110.0, 100.0, 90.0, 80.0, 72.0, 70.0, 70.0, 70.0, 70.0]
    values.loc[80:88, "MAP"] = [65.0, 70.0, 75.0, 80.0, 85.0, 85.0, 85.0, 85.0, 85.0]
    results = monitor_dataframe(case_id=131, df=values)
    recovered = [event for event in results["lifecycle_events"] if event["alert_state"] == "alert_recovered"]
    assert len(recovered) == 1
    summary = recovered[0]["vital_summary"]["HR"]
    assert summary["latest_value"] == 70.0
    assert summary["current_trend"] == "decreasing"
    print("Decreasing recovery summary: PASS")


def test_long_continuous_event() -> None:
    values = patient_frame()
    # Longer than persistence but shorter than the fixed 60-second baseline
    # window, so the deliberately unchanged rolling baseline does not adapt.
    values.loc[70:95, ["HR", "MAP"]] = [95.0, 60.0]
    results = monitor_dataframe(case_id=124, df=values)
    assert len(results["alerts"]) == 1
    assert sum(event["alert_state"] == "alert_recovered" for event in results["lifecycle_events"]) == 1
    print("Long continuous event deduplication: PASS (1 alert_started event)")


def test_multiple_independent_events() -> None:
    values = patient_frame(300)
    # Separate events by more than one baseline window of stable data so each
    # is independently assessed by the unchanged rolling baseline.
    for start in (70, 150, 230):
        values.loc[start : start + 14, ["HR", "MAP"]] = [95.0, 60.0]
    results = monitor_dataframe(case_id=125, df=values)
    assert len(results["alerts"]) == 3
    print("Multiple independent events: PASS (3 alert_started events, no alert cap)")


def test_suspicious_persistent_event() -> None:
    values = patient_frame(120)
    values.loc[70:80, ["HR", "MAP"]] = [95.0, 60.0]
    # MAP is a valid but isolated possible artifact exactly when persistence is
    # met; it must remain alert evidence with transparent quality metadata.
    # With moderate persistence = 8, the alert fires at position 77.
    values.at[77, "MAP"] = 250.0
    results = monitor_dataframe(case_id=126, df=values)
    assert len(results["alerts"]) == 1
    assert results["alerts"][0]["vital_details"]["MAP"]["signal_quality"] == "suspicious"
    print("Suspicious persistent multi-vital event: PASS (alert retained with metadata)")


def test_invalid_data() -> None:
    values = patient_frame(100)
    values.loc[70:80, "HR"] = 0.0
    values.loc[70:80, "MAP"] = -5.0
    results = monitor_dataframe(case_id=127, df=values)
    assert int(results["invalid_measurements"].to_numpy().sum()) == 22
    assert results["alerts"] == []
    print("Invalid-data exclusion: PASS (invalid measurements excluded without crash)")


def test_missing_data() -> None:
    values = patient_frame(40)
    values.loc[10:20, "RR"] = np.nan
    results = monitor_dataframe(case_id=456, df=values)
    assert "insufficient_data" in set(results["signal_quality"]["RR"])
    assert results["alerts"] == []
    print("Missing-data signal quality and zero-alert handling: PASS")


def test_sparse_persistent_data() -> None:
    values = patient_frame(120)
    values.loc[70:80, ["HR", "MAP"]] = [95.0, 60.0]
    # Alternate missing samples are conservatively interpolated by the existing
    # preprocessing path. They may carry insufficient-data metadata but must
    # not block a genuinely persistent multi-vital event by themselves.
    values.loc[71:79:2, ["HR", "MAP"]] = np.nan
    results = monitor_dataframe(case_id=128, df=values)
    assert len(results["alerts"]) == 1
    assert results["alerts"][0]["vital_details"]["HR"]["signal_quality"] == "insufficient_data"
    print("Sparse/fill-derived persistent data: PASS (alert retained with quality metadata)")


# ---------------------------------------------------------------------------
# New severity-specific tests
# ---------------------------------------------------------------------------


def test_single_severe_vital() -> None:
    """A single vital with severe deviation triggers an alert on its own."""
    values = patient_frame()
    # HR=105 → deviation = |105-70|/70 = 0.50 → severe (≥ 0.20×2.0 = 0.40)
    values.loc[70:80, "HR"] = 105.0
    results = monitor_dataframe(case_id=200, df=values)
    assert len(results["alerts"]) == 1
    alert = results["alerts"][0]
    assert "HR" in alert["affected_vitals"]
    assert alert["severity"] in ("severe", "critical")
    print("Single severe vital: PASS (1 alert from a single vital)")


def test_single_critical_vital() -> None:
    """A critical single vital triggers with minimal persistence."""
    values = patient_frame()
    # HR=115 → deviation = |115-70|/70 = 0.643 → critical (≥ 0.20×3.0 = 0.60)
    # 5 seconds duration matches calibrated critical persistence = 5.
    values.loc[70:74, "HR"] = 115.0
    results = monitor_dataframe(case_id=201, df=values)
    assert len(results["alerts"]) == 1
    alert = results["alerts"][0]
    assert alert["severity"] == "critical"
    assert "HR" in alert["affected_vitals"]
    print("Single critical vital: PASS (rapid alert with critical severity)")


def test_single_mild_vital_no_alert() -> None:
    """A single mild vital does NOT trigger an alert (requires 2 vitals)."""
    values = patient_frame()
    # HR=90 → deviation = |90-70|/70 = 0.286 → mild (≥ 0.20 but < 0.30)
    values.loc[70:90, "HR"] = 90.0
    results = monitor_dataframe(case_id=202, df=values)
    assert results["alerts"] == []
    print("Single mild vital no alert: PASS (mild requires 2 vitals)")


def test_severity_classification() -> None:
    """Verify severity label appears in alert output."""
    values = patient_frame()
    values.loc[70:80, ["HR", "MAP"]] = [95.0, 60.0]
    results = monitor_dataframe(case_id=203, df=values)
    assert len(results["alerts"]) == 1
    alert = results["alerts"][0]
    assert "severity" in alert
    assert alert["severity"] in ("mild", "moderate", "severe", "critical")
    # Check per-vital severity in details
    for vital in alert["affected_vitals"]:
        assert "severity" in alert["vital_details"][vital]
    print("Severity classification: PASS (severity in alert output)")


def test_temporary_spike_no_alert() -> None:
    """A brief severe spike below the persistence threshold does not alert."""
    values = patient_frame()
    # Severe deviation for 3 seconds; severe persistence requires 5.
    values.loc[70:72, "HR"] = 105.0
    results = monitor_dataframe(case_id=204, df=values)
    assert results["alerts"] == []
    print("Temporary spike no alert: PASS (short spike below persistence)")


def test_recovery_from_severity() -> None:
    """A severity event properly recovers and produces lifecycle events."""
    values = patient_frame()
    # Severe single-vital event lasting 11 seconds.
    values.loc[70:80, "HR"] = 105.0
    results = monitor_dataframe(case_id=205, df=values)
    assert len(results["alerts"]) == 1
    recovered = [e for e in results["lifecycle_events"] if e["alert_state"] == "alert_recovered"]
    assert len(recovered) == 1
    assert recovered[0]["event_id"] == results["alerts"][0]["event_id"]
    print("Recovery from severity: PASS (proper lifecycle for severity event)")


def test_structured_event_output() -> None:
    """Verify every alert contains the full structured event for Risk Agent."""
    values = patient_frame()
    values.loc[70:80, "HR"] = 105.0  # severe single vital
    results = monitor_dataframe(case_id=206, df=values)
    assert len(results["alerts"]) == 1
    alert = results["alerts"][0]
    # All required fields for the Risk Prediction Agent
    required_fields = [
        "case_id", "event_id", "event", "alert_state", "severity",
        "affected_vitals", "deviation_values", "duration_seconds",
        "vital_details", "vital_summary", "source", "data_source",
        "start_timestamp", "current_timestamp",
    ]
    for field in required_fields:
        assert field in alert, f"Missing required field: {field}"
    # Per-vital detail structure
    vital_detail = alert["vital_details"]["HR"]
    detail_fields = [
        "current", "baseline", "relative_deviation", "severity",
        "direction", "trend", "signal_quality",
    ]
    for field in detail_fields:
        assert field in vital_detail, f"Missing vital detail field: {field}"
    print("Structured event output: PASS (all fields present for Risk Agent)")


# ---------------------------------------------------------------------------
# Preprocessing tests
# ---------------------------------------------------------------------------


def test_leading_invalid_map_no_one_baseline() -> None:
    """Leading invalid/zeroing MAP values must not create a baseline of 1.0 mmHg."""
    values = patient_frame(100)
    # Simulate uncalibrated arterial line: negative/zero readings, then 1.0 transitional
    values.loc[0:30, "MAP"] = -5.0
    values.at[31, "MAP"] = 1.0  # Transitional zeroing value
    values.loc[32:50, "MAP"] = np.nan  # Gap before true readings
    values.loc[51:99, "MAP"] = 85.0
    results = monitor_dataframe(case_id=301, df=values)
    # Under old preprocessing with bfill(), early MAP baseline was 1.0 mmHg.
    # Now, early MAP baseline should remain NaN throughout the uncalibrated period.
    early_baseline = results["baseline"].loc[0:40, "MAP"].dropna()
    assert len(early_baseline) == 0, f"Early MAP baseline was established prematurely: {early_baseline}"
    # Verify that a baseline of 1.0 mmHg is NEVER established anywhere
    assert (results["baseline"]["MAP"].dropna() != 1.0).all(), "Found MAP baseline equal to 1.0 mmHg"
    print("Leading invalid MAP: PASS (no baseline of 1.0 created)")


def test_long_missing_gaps_not_backfilled() -> None:
    """Long missing gaps (>5 samples) must remain NaN, not backfilled or forward-filled."""
    values = patient_frame(80)
    values.loc[15:40, "MAP"] = np.nan  # 26-sample gap
    results = monitor_dataframe(case_id=302, df=values)
    # Preprocessed dataframe should preserve NaNs inside the long gap
    cleaned_gap = results["df"].loc[21:35, "MAP"]
    assert cleaned_gap.isna().all(), "Long gap was unexpectedly filled"
    print("Long missing gaps: PASS (preserved as NaN without ffill/bfill)")


def test_short_internal_gaps_interpolated() -> None:
    """Short internal gaps (<= 5 samples) should be linearly interpolated."""
    values = patient_frame(60)
    values.loc[20:23, "HR"] = np.nan  # 4-sample gap between 70.0 and 70.0
    results = monitor_dataframe(case_id=303, df=values)
    assert not results["df"].loc[20:23, "HR"].isna().any(), "Short gap was not interpolated"
    assert (results["df"].loc[20:23, "HR"] == 70.0).all()
    print("Short internal gaps: PASS (interpolated correctly)")


def test_trailing_invalid_values_not_forward_filled() -> None:
    """Trailing disconnections/invalid values must remain NaN, not forward-filled."""
    values = patient_frame(60)
    values.loc[45:59, "MAP"] = 0.0  # Transducer disconnected at end
    results = monitor_dataframe(case_id=304, df=values)
    cleaned_trailing = results["df"].loc[45:59, "MAP"]
    assert cleaned_trailing.isna().all(), "Trailing invalid values were unexpectedly forward filled"
    print("Trailing invalid values: PASS (preserved as NaN)")


def test_valid_physiological_values_unchanged() -> None:
    """Valid physiological measurements must remain identical in original units."""
    values = patient_frame(50)
    results = monitor_dataframe(case_id=305, df=values)
    for col in ["HR", "MAP", "SpO2", "RR"]:
        assert (results["df"][col] == values[col]).all(), f"Values modified for {col}"
    print("Valid physiological values: PASS (original units preserved)")


def main() -> None:
    test_existing_stable_data()
    test_existing_persistence_data()
    test_trends()
    test_artifact_detection()
    test_lifecycle()
    test_event_evolution()
    test_active_event_snapshot()
    test_decreasing_recovery_summary()
    test_long_continuous_event()
    test_multiple_independent_events()
    test_suspicious_persistent_event()
    test_invalid_data()
    test_missing_data()
    test_sparse_persistent_data()
    # Severity tests
    test_single_severe_vital()
    test_single_critical_vital()
    test_single_mild_vital_no_alert()
    test_severity_classification()
    test_temporary_spike_no_alert()
    test_recovery_from_severity()
    test_structured_event_output()
    # Preprocessing tests
    test_leading_invalid_map_no_one_baseline()
    test_long_missing_gaps_not_backfilled()
    test_short_internal_gaps_interpolated()
    test_trailing_invalid_values_not_forward_filled()
    test_valid_physiological_values_unchanged()
    print("All enhancement and preprocessing checks passed.")


if __name__ == "__main__":
    main()
