from __future__ import annotations

from monitoring_agent.monitoring_agent import run_monitoring
from risk_agent.risk_prediction_agent import RiskPredictionAgent


def print_monitoring_output(monitoring_results: dict) -> None:
    """Print the existing Monitoring Agent output."""

    alerts = monitoring_results.get("alerts", [])

    print("\n" + "=" * 60)
    print("MONITORING AGENT OUTPUT")
    print("=" * 60)

    print(
        f"Case ID              : "
        f"{alerts[0]['case_id'] if alerts else 'N/A'}"
    )

    print(
        f"Persistent Alerts    : {len(alerts)}"
    )

    if not alerts:
        print("\nNo persistent physiological alerts detected.")
        return

    for number, alert in enumerate(alerts, start=1):
        print(f"\nAlert {number}")
        print(
            f"  Event ID           : "
            f"{alert.get('event_id', 'N/A')}"
        )
        print(
            f"  Timestamp          : "
            f"{alert.get('timestamp', 'N/A')}"
        )
        print(
            f"  Affected Vitals    : "
            f"{', '.join(alert.get('affected_vitals', []))}"
        )
        print(
            f"  Severity           : "
            f"{alert.get('severity', 'N/A')}"
        )
        print(
            f"  Duration           : "
            f"{alert.get('duration_seconds', 'N/A')} seconds"
        )


def print_risk_result(
    risk_results: list[dict],
) -> None:
    """Print actual Random Forest predictions."""

    print("\n" + "=" * 60)
    print("RISK PREDICTION AGENT OUTPUT")
    print("=" * 60)

    if not risk_results:
        print("\nNo alerts were passed to the Risk Prediction Agent.")
        return

    for number, result in enumerate(risk_results, start=1):
        print(f"\nAlert {number}")

        print(
            f"  Event ID            : "
            f"{result.get('event_id', 'N/A')}"
        )

        print(
            f"  Risk Probability    : "
            f"{result['risk_probability'] * 100:.2f}%"
        )

        print(
            f"  Risk Level          : "
            f"{result['risk_level']}"
        )

        print(
            f"  Model               : "
            f"{result['model']}"
        )

        print(
            f"  Threshold           : "
            f"{result['threshold']:.2f}"
        )

        print(
            f"  Alert Timestamp     : "
            f"{result.get('timestamp', result.get('window_end', 'N/A'))}"
        )

        print(
            f"  5-Min Window        : "
            f"{result['window_start']:.0f} - "
            f"{result['window_end']:.0f} sec"
        )

        print(
            f"  Window Samples      : "
            f"{result['window_samples']}"
        )

        print(
            f"  Severity            : "
            f"{result['severity']}"
        )

        print(
            f"  Affected Vitals     : "
            f"{', '.join(result['affected_vitals'])}"
        )

def run_carematrix(case_id: int) -> dict:
    """
    Run the complete CareMatrix pipeline.

    Monitoring Agent remains unchanged.
    Its output is passed directly to the Risk Prediction Agent.
    """

    print("\n" + "=" * 60)
    print("CARE MATRIX")
    print("=" * 60)

    print("\n[1] Loading Patient Data...")
    print(f"Case ID: {case_id}")

    print("\n[2] Running Monitoring Agent...")

    monitoring_results = run_monitoring(case_id)

    print_monitoring_output(monitoring_results)

    print("\n[3] Running Risk Prediction Agent...")

    risk_agent = RiskPredictionAgent()

    risk_results = risk_agent.process_monitoring_output(
        monitoring_results,
        case_id=case_id,
    )

    print_risk_result(risk_results)

    print("\n" + "=" * 60)
    print("CARE MATRIX RUN COMPLETE")
    print("=" * 60)

    return {
        "case_id": case_id,
        "monitoring_results": monitoring_results,
        "risk_results": risk_results,
    }


if __name__ == "__main__":
    CASE_ID = 4

    run_carematrix(CASE_ID)