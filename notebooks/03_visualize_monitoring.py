"""Plot vital signs, rolling baselines, and persistent alert timestamps."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from monitoring_agent.monitoring_agent import VITAL_COLUMNS, find_suitable_cases, monitor_dataframe, run_monitoring
from run_monitoring import synthetic_patient_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def plot_monitoring(results: dict, output_path: Path) -> None:
    """Create one simple subplot per vital and mark persistent alerts."""
    result_df = results["monitoring_results"]
    alert_times = result_df.index[result_df["persistent_alert"]]
    figure, axes = plt.subplots(len(VITAL_COLUMNS), 1, figsize=(12, 10), sharex=True)

    for axis, vital in zip(axes, VITAL_COLUMNS):
        axis.plot(result_df.index, result_df[vital], label=vital, linewidth=1)
        axis.plot(
            result_df.index,
            result_df[f"{vital}_baseline"],
            label=f"{vital} baseline",
            linewidth=1.5,
        )
        for timestamp in alert_times:
            axis.axvline(timestamp, color="tab:red", alpha=0.25, linewidth=1)
        axis.set_ylabel(vital)
        axis.legend(loc="best")
        axis.grid(alpha=0.25)

    axes[-1].set_xlabel("Time (seconds)")
    figure.suptitle("CareMatrix Phase 1: Patient Vitals and Rolling Baselines")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CareMatrix monitoring plots.")
    parser.add_argument("--synthetic", action="store_true", help="Plot the built-in offline test case.")
    parser.add_argument("--case-id", type=int, help="Specific VitalDB case ID to plot.")
    args = parser.parse_args()

    try:
        if args.synthetic:
            case_id = 0
            results = monitor_dataframe(case_id, synthetic_patient_data())
        else:
            case_id = args.case_id or find_suitable_cases(limit=1)[0]
            results = run_monitoring(case_id)
        output_path = PROJECT_ROOT / "data" / "processed" / f"monitoring_case_{case_id}.png"
        plot_monitoring(results, output_path)
        print(f"Saved plot to: {output_path.relative_to(PROJECT_ROOT)}")
    except (RuntimeError, ValueError, TypeError) as exc:
        print(f"Visualization failed: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
