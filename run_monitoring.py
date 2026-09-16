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
    # longer than the default 10-second persistence requirement.
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
    print("Monitoring Agent")
    print("----------------")
    print(f"Case ID: {case_id}")
    print(f"Observations: {len(results['df'])}")
    print(f"Candidate deviations: {int(monitoring_results['candidate_alert'].sum())}")
    print(f"Final alerts: {len(alerts)}")
    if alerts:
        print("\nFinal alert details:")
        for alert_number, alert in enumerate(alerts, start=1):
            timestamp = alert["timestamp"]
            print(f"\nAlert {alert_number}")
            print(f"  Case ID: {alert['case_id']}")
            print(f"  Timestamp: {timestamp}")
            print(f"  Affected vitals: {', '.join(alert['affected_vitals'])}")
            print(f"  Persistence duration: {alert['duration_seconds']} seconds")
            for vital in alert["affected_vitals"]:
                current_value = monitoring_results.at[timestamp, vital]
                baseline_value = monitoring_results.at[timestamp, f"{vital}_baseline"]
                relative_deviation = monitoring_results.at[timestamp, f"{vital}_deviation"]
                print(
                    f"  {vital}: current={current_value:.2f}, "
                    f"baseline={baseline_value:.2f}, "
                    f"relative_deviation={relative_deviation:.4f}"
                )
    else:
        print("\nNo persistent multi-vital deviations detected for this case.")


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
