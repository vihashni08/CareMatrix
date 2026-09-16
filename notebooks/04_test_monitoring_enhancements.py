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
    assert results["monitoring_results"].at[79, "alert_state"] == "alert_started"
    assert results["monitoring_results"].at[85, "alert_state"] == "recovering"
    assert results["monitoring_results"].at[89, "alert_state"] == "alert_recovered"
    assert results["monitoring_results"].at[119, "alert_state"] == "alert_started"
    assert started[0]["event_id"] == recovered[0]["event_id"]
    print("Persistence, deduplication, recovery, and new-event lifecycle: PASS")


def test_event_evolution() -> None:
    values = patient_frame(100)
    values.loc[70:79, "HR"] = [90, 98, 105, 115, 110, 100, 100, 100, 100, 100]
    values.loc[70:79, "MAP"] = 60.0
    results = monitor_dataframe(case_id=129, df=values)
    summary = results["alerts"][0]["vital_summary"]["HR"]
    assert summary["initial_value"] == 90.0
    assert summary["maximum_value"] == 115.0
    assert summary["latest_value"] == 100.0
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
    values.at[79, "MAP"] = 250.0
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
    print("All enhancement checks passed.")


if __name__ == "__main__":
    main()
