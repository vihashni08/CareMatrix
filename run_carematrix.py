from __future__ import annotations

from monitoring_agent.monitoring_agent import run_monitoring
from risk_agent.risk_prediction_agent import RiskPredictionAgent


# ============================================================
# PRINT MONITORING OUTPUT
# ============================================================

def print_monitoring_output(monitoring_results):

    print("\n")
    print("=" * 65)
    print("MONITORING AGENT OUTPUT")
    print("=" * 65)

    alerts = monitoring_results.get(
        "alerts",
        []
    )

    # --------------------------------------------------------
    # NO DEVIATION
    # --------------------------------------------------------

    if not alerts:

        print(
            "Deviation Detected : NO"
        )

        print(
            "Status             : Normal"
        )

        print(
            "Persistent Alert   : None"
        )

        return

    # --------------------------------------------------------
    # DEVIATION DETECTED
    # --------------------------------------------------------

    print(
        "Deviation Detected : YES"
    )

    print(
        f"Persistent Alerts  : {len(alerts)}"
    )

    print()

    for i, alert in enumerate(
        alerts,
        start=1
    ):

        print(
            f"Alert {i}"
        )

        print(
            f"  Event ID        : "
            f"{alert.get('event_id')}"
        )

        print(
            f"  Timestamp       : "
            f"{alert.get('timestamp')}"
        )

        print(
            f"  Affected Vitals : "
            f"{', '.join(alert.get('affected_vitals', []))}"
        )

        print(
            f"  Duration        : "
            f"{alert.get('duration_seconds')} seconds"
        )

        print(
            "  Deviations:"
        )

        for vital, deviation in alert.get(
            "deviation_values",
            {}
        ).items():

            print(
                f"    {vital}: "
                f"{deviation * 100:.2f}%"
            )

        print()


# ============================================================
# PRINT RISK OUTPUT
# ============================================================

def print_risk_result(results):
    print("\n")
    print("=" * 65)
    print("RISK PREDICTION AGENT OUTPUT")
    print("=" * 65)

    for result in results:

        alert_number = result.get("alert_number")

        if alert_number is None:
            print("Case-Level Risk Result")
        else:
            print(f"Alert {alert_number}")

        event_id = result.get("event_id")

        if event_id is not None:
            print(f"  Event ID            : {event_id}")

        print(
            "  Deviation Detected  : "
            f"{'YES' if result.get('deviation_detected') else 'NO'}"
        )

        print(f"  Risk Level          : {result.get('risk_level')}")
        print(f"  Reason              : {result.get('reason')}")
        print(f"  Source              : {result.get('source')}")
        print()


# ============================================================
# MAIN CAREMATRIX WORKFLOW
# ============================================================

def run_carematrix(case_id):

    print("\n")
    print("=" * 65)
    print("             CAREMATRIX")
    print("     MONITORING → RISK PREDICTION")
    print("=" * 65)

    # --------------------------------------------------------
    # STEP 1: MONITORING AGENT
    # --------------------------------------------------------

    print("\n[1] Running Monitoring Agent...")

    monitoring_results = run_monitoring(
        case_id
    )

    # --------------------------------------------------------
    # DISPLAY MONITORING RESULT
    # --------------------------------------------------------

    print_monitoring_output(
        monitoring_results
    )

    # --------------------------------------------------------
    # STEP 2: RISK PREDICTION AGENT
    # --------------------------------------------------------

    print("\n[2] Running Risk Prediction Agent...")

    risk_agent = RiskPredictionAgent()

    risk_result = (
        risk_agent.process_monitoring_output(
            monitoring_results
        )
    )

    # --------------------------------------------------------
    # DISPLAY RISK RESULT
    # --------------------------------------------------------

    print_risk_result(
        risk_result
    )

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    return {
        "case_id": case_id,
        "monitoring": monitoring_results,
        "risk": risk_result,
    }


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    CASE_ID = 4

    results = run_carematrix(
        CASE_ID
    )