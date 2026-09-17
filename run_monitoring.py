"""Run the Phase 1 Monitoring Agent on one automatically selected VitalDB case."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from monitoring_agent.monitoring_agent import VITAL_COLUMNS, find_suitable_cases, monitor_dataframe, run_monitoring


PROJECT_ROOT = Path(__file__).resolve().parent


def synthetic_patient_data() -> pd.DataFrame:
    """Return a deterministic case with an 11-second multi-vital deviation."""
    sample_count = 100
    data = pd.DataFrame(
        {
            "HR": np.full(sample_count, 70.0),
            "MAP": np.full(sample_count, 85.0),
            "SpO2": np.full(sample_count, 98.0),
            "RR": np.full(sample_count, 14.0),
        },
        index=pd.RangeIndex(sample_count, name="timestamp"),
    )
    # Enough simultaneous deviations to exceed the configured thresholds for
    # longer than the severity-based persistence requirement.
    data.loc[70:80, ["HR", "MAP", "SpO2"]] = [95.0, 60.0, 90.0]
    return data[VITAL_COLUMNS]


def save_processed_data(case_id: int, df: pd.DataFrame) -> Path:
    """Save only the selected case's cleaned one-second samples."""
    output_dir = PROJECT_ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"patient_{case_id}.csv"
    df.to_csv(output_path, index_label="timestamp")
    return output_path


def print_summary(case_id: int, results: dict) -> None:
    """Print a concise report and full clinical-neutral details for every alert."""
    monitoring_results = results["monitoring_results"]
    alerts = results["alerts"]
    lifecycle_events = results.get("lifecycle_events", alerts)
    completed_events = results.get("completed_events", [])
    monitoring_summary = results.get("monitoring_summary", {})
    print("Monitoring Agent")
    print("----------------")
    print(f"Case ID: {case_id}")
    print(f"Observations: {len(results['df'])}")
    print(f"Candidate deviations: {int(monitoring_results['candidate_alert'].sum())}")
    print(f"Final alerts: {len(alerts)}")
    if monitoring_summary:
        print("\nAlert conversion breakdown:")
        print(f"  Threshold-violation samples: {monitoring_summary['threshold_violation_samples']}")
        print(f"  Failed severity gate: {monitoring_summary['failed_severity_gate_samples']}")
        print(f"  Candidate runs failed persistence: {monitoring_summary['candidate_runs_failed_persistence']}")
        print(f"  Rejected because of insufficient data: {monitoring_summary['candidate_rejected_insufficient_data']}")
        print(f"  Invalid measurements: {monitoring_summary['invalid_measurements']}")
        print(f"  Suspicious measurements: {monitoring_summary['suspicious_measurements']}")
        print(f"  Alert-started events: {monitoring_summary['alert_started_events']}")
        print(f"  Alert-recovered events: {monitoring_summary['alert_recovered_events']}")
        print(f"  Currently active events: {monitoring_summary['currently_active_events']}")
        print(f"  Independent monitoring events: {monitoring_summary['independent_monitoring_events']}")
    if alerts:
        print("\nFinal alert details:")
        for alert_number, alert in enumerate(alerts, start=1):
            timestamp = alert["timestamp"]
            print(f"\nAlert {alert_number}")
            print(f"  Case ID: {alert['case_id']}")
            print(f"  Event ID: {alert.get('event_id', 'not available')}")
            print(f"  Timestamp: {timestamp}")
            print(f"  Event start: {alert.get('start_timestamp', timestamp)}")
            print(f"  Affected vitals: {', '.join(alert['affected_vitals'])}")
            print(f"  Severity: {alert.get('severity', 'unknown')}")
            print(f"  Persistence duration: {alert['duration_seconds']} seconds")
            for vital in alert["affected_vitals"]:
                current_value = monitoring_results.at[timestamp, vital]
                baseline_value = monitoring_results.at[timestamp, f"{vital}_baseline"]
                relative_deviation = monitoring_results.at[timestamp, f"{vital}_deviation"]
                print(
                    f"  {vital}: current={current_value:.2f}, "
                    f"baseline={baseline_value:.2f}, "
                    f"relative_deviation={relative_deviation:.4f}, "
                    f"trend={alert['vital_details'][vital]['trend']}, "
                    f"signal_quality={alert['vital_details'][vital]['signal_quality']}"
                )
        recovered_count = sum(event["alert_state"] == "alert_recovered" for event in lifecycle_events)
        print(f"\nLifecycle events: {len(alerts)} alert_started, {recovered_count} alert_recovered")
    if completed_events:
        print("\nCompleted event summaries:")
        for event in completed_events:
            print(
                f"  {event['event_id']}: start={event['start_timestamp']}, "
                f"end={event['end_timestamp']}, duration={event['duration_seconds']} seconds, "
                f"vitals={', '.join(event['affected_vitals'])}"
            )
            for vital, summary in event["vital_summary"].items():
                print(
                    f"    {vital}: initial={summary['initial_value']:.2f}, "
                    f"min={summary['minimum_value']:.2f}, max={summary['maximum_value']:.2f}, "
                    f"latest={summary['latest_value']:.2f}, "
                    f"peak_deviation={summary['peak_absolute_deviation']:.4f}, "
                    f"trend={summary['current_trend']}, quality={summary['signal_quality']}"
                )
    else:
        print("\nNo persistent physiological deviations detected for this case.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CareMatrix Monitoring Agent.")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run the built-in persistence test without VitalDB or network access.",
    )
    args = parser.parse_args()

    try:
        if args.synthetic:
            case_id = 0
            results = monitor_dataframe(case_id, synthetic_patient_data())
        else:
            case_id = find_suitable_cases(limit=1)[0]
            results = run_monitoring(case_id)
        output_path = save_processed_data(case_id, results["df"])
        print_summary(case_id, results)
        print(f"\nProcessed data saved to: {output_path.relative_to(PROJECT_ROOT)}")
    except (RuntimeError, ValueError, TypeError) as exc:
        print(f"Monitoring Agent error: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
