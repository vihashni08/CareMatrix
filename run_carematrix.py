"""CareMatrix: Multi-Agent Autonomous Physiological Monitoring and Risk Assessment.

Demonstrates independent, continuously running AI agents:
                SUPERVISOR
             independent loop
              /           \
             ↓             ↓
   MONITORING WORKER    RISK WORKER
          │                  │
          │ publish          │ consume
          └──────→ QUEUE ←──┘
                     │
                     ↓
               Risk Decision
"""

from __future__ import annotations

import argparse
import datetime
import time
from typing import Any

from communication.event_queue import EventQueue
from communication.events import MonitoringDecision, MonitoringEvent, RiskDecisionEvent
from data_analysis_agent import DataAnalysisAgent
from monitoring_agent.monitoring_agent import MonitoringAgent, monitor_dataframe, run_monitoring
from replay_stream import PatientStreamReplayer
from risk_agent.risk_agent import RiskAgent
from risk_agent.risk_prediction_agent import RiskPredictionAgent
from supervisor.supervisor import AgentSupervisor


def print_monitoring_output(monitoring_results: dict[str, Any]) -> None:
    """Print the Monitoring Agent output report."""
    alerts = monitoring_results.get("alerts", [])

    print("\n" + "=" * 60)
    print("MONITORING AGENT OUTPUT")
    print("=" * 60)

    case_id_display = (
        alerts[0].get("case_id", alerts[0].get("patient_id", monitoring_results.get("case_id", "N/A")))
        if alerts
        else monitoring_results.get("case_id", "N/A")
    )
    print(f"Case ID              : {case_id_display}")
    print(f"Persistent Alerts    : {len(alerts)}")

    if not alerts:
        print("\nNo persistent physiological alerts detected.")
        return

    for number, alert in enumerate(alerts, start=1):
        print(f"\nAlert {number}")
        print(f"  Event ID           : {alert.get('event_id', 'N/A')}")
        print(f"  Timestamp          : {alert.get('timestamp', 'N/A')}")
        print(f"  Affected Vitals    : {', '.join(alert.get('affected_vitals', []))}")
        print(f"  Severity           : {alert.get('severity', 'N/A')}")
        print(f"  Duration           : {alert.get('duration_seconds', alert.get('persistence_duration', 'N/A'))} seconds")


def print_risk_result(risk_results: list[dict[str, Any]]) -> None:
    """Print Risk Agent prediction and decision output."""
    print("\n" + "=" * 60)
    print("RISK AGENT OUTPUT")
    print("=" * 60)

    if not risk_results:
        print("\nNo alerts were escalated to the Risk Agent.")
        return

    for number, result in enumerate(risk_results, start=1):
        print(f"\nAlert {number}")
        print(f"  Event ID            : {result.get('event_id', 'N/A')}")
        prob = result.get("risk_probability", 0.0)
        print(f"  Risk Probability    : {prob * 100:.2f}%" if prob >= 0 else "  Risk Probability    : N/A")
        print(f"  Risk Level          : {result.get('risk_level', 'N/A')}")
        print(f"  Decision            : {result.get('decision', result.get('risk_level', 'N/A'))}")
        print(f"  Model Tool          : {result.get('model', 'N/A')}")
        thresh = result.get("threshold", 0.16)
        print(f"  Threshold           : {thresh:.2f}")
        print(f"  Alert Timestamp     : {result.get('timestamp', result.get('window_end', 'N/A'))}")
        print(f"  Severity            : {result.get('severity', 'N/A')}")
        vitals = result.get("affected_vitals", [])
        print(f"  Affected Vitals     : {', '.join(vitals) if vitals else 'N/A'}")
        if result.get("reason"):
            print(f"  Reason              : {result['reason']}")
        if result.get("recommended_action"):
            print(f"  Action              : {result['recommended_action']}")


def run_agent_demonstration(
    case_id: int = 0,
    max_samples: int = 100,
    sample_delay: float = 0.03,
    fail_risk: bool = False,
    fail_monitoring: bool = False,
) -> dict[str, Any]:
    """Execute end-to-end multi-agent demonstration with visible concurrent execution logs."""
    print("\n" + "=" * 70)
    print("CARE MATRIX — INDEPENDENT AGENT ARCHITECTURE DEMONSTRATION")
    print("=" * 70)

    # 1. Initialize Asynchronous FIFO Event Queue
    event_queue = EventQueue()

    # 2. Initialize Monitoring Agent
    monitoring_agent = MonitoringAgent(
        case_id=case_id,
        event_queue=event_queue,
        baseline_window=60,
        verbose=True,
    )

    # 3. Initialize Risk Agent (uses actual saved RandomForestClassifier tool)
    risk_agent = RiskAgent(
        event_queue=event_queue,
        max_retries=3,
        verbose=True,
    )
    # Data Analysis consumes completed model decisions and publishes structured
    # evidence to the Clinical Reasoning Agent input topic.
    data_analysis_agent = DataAnalysisAgent(event_queue=event_queue)
    if fail_risk:
        risk_agent.inject_failure = True
        print("[Demo Setup] Risk Agent configured with intentional ML tool failure hook.")

    if fail_monitoring:
        # Simulate unhandled hang after 30 samples to test supervisor stale detection
        monitoring_agent._hang_at_sample = 30
        print("[Demo Setup] Monitoring Agent configured with intentional freeze hook at sample 30.")

    # 4. Initialize Independent Supervisor Monitor
    supervisor = AgentSupervisor(
        event_queue=event_queue,
        heartbeat_timeout_seconds=1.0,  # Fast timeout for demo responsiveness
        check_interval_seconds=0.2,
        auto_recover=True,
        verbose=True,
    )
    supervisor.register_agent(monitoring_agent)
    supervisor.register_agent(risk_agent)

    # 5. Start Independent Background Threads
    print("\n--- Starting Independent Agent Workers ---")
    supervisor.start()
    risk_agent.start()
    data_analysis_agent.start()

    # 6. Stream patient observations through Monitoring Agent worker thread
    replayer = PatientStreamReplayer(case_id=case_id, delay_seconds=0.0)
    stream_generator = replayer.stream(max_samples=max_samples)

    monitoring_agent.start(
        stream_source=stream_generator,
        sample_delay=sample_delay,
    )

    # Wait for monitoring worker to finish (or until stopped)
    monitoring_agent.join(timeout=10.0)
    monitoring_agent.stop()

    # Allow Risk Agent worker thread to finish any pending items on event queue
    time.sleep(0.5)

    # 7. Gracefully stop workers
    risk_agent.stop()
    data_analysis_agent.stop()
    supervisor.stop()
    monitoring_agent.join(timeout=1.0)
    risk_agent.join(timeout=1.0)
    data_analysis_agent.join(timeout=1.0)
    supervisor.join(timeout=1.0)
    event_queue.shutdown()

    # 8. Compile and Print Demo Summary
    print("\n" + "=" * 70)
    print("DEMONSTRATION RUN SUMMARY")
    print("=" * 70)

    # Monitoring stats
    obs_count = monitoring_agent.state.total_observations
    esc_count = monitoring_agent.state.total_escalations
    rec_count = monitoring_agent.state.total_recoveries
    print("\n[Monitoring Agent]")
    print(f"  observations        = {obs_count}")
    print(f"  escalations         = {esc_count}")
    print(f"  recoveries          = {rec_count}")
    print(f"  lifecycle_state     = {monitoring_agent.lifecycle_state.value}")

    # Risk stats
    risk_history = event_queue.get_history("risk_decisions")
    high_risk_cnt = sum(1 for d in risk_history if getattr(d, "decision", "") == "HIGH_RISK")
    low_risk_cnt = sum(1 for d in risk_history if getattr(d, "decision", "") == "LOW_RISK")
    fail_cnt = risk_agent.state.failure_count
    analysis_history = event_queue.get_history("data_analysis_events")

    print("\n[Risk Agent]")
    print(f"  events_received     = {len(risk_agent.state.received_events)}")
    print(f"  predictions         = {len(risk_history)}")
    print(f"  high_risk           = {high_risk_cnt}")
    print(f"  low_risk            = {low_risk_cnt}")
    print(f"  failures            = {fail_cnt}")
    print(f"  model_tool          = {risk_agent.model_name}")

    print("\n[Data Analysis Agent]")
    print(f"  analyses            = {len(analysis_history)}")
    print(f"  partial_analyses    = {sum(1 for e in analysis_history if e.analysis_status == 'partial_analysis')}")

    # Supervisor stats
    health = supervisor.check_health()
    mon_status = health.get("MonitoringAgent", {}).get("status", "UNKNOWN").upper()
    risk_status = health.get("RiskAgent", {}).get("status", "UNKNOWN").upper()
    total_restarts = sum(info.get("restart_count", 0) for info in health.values())

    print("\n[Supervisor]")
    print(f"  monitoring_status   = {mon_status}")
    print(f"  risk_status         = {risk_status}")
    print(f"  restarts            = {total_restarts}")
    print(f"  failures_logged     = {len(supervisor.failure_log)}")

    print("\n" + "=" * 70)
    print("AGENT DEMONSTRATION COMPLETE")
    print("=" * 70)

    return {
        "monitoring_stats": {"observations": obs_count, "escalations": esc_count, "recoveries": rec_count},
        "risk_stats": {"predictions": len(risk_history), "high_risk": high_risk_cnt, "low_risk": low_risk_cnt},
        "data_analysis_stats": {"analyses": len(analysis_history)},
        "supervisor_stats": {"monitoring_status": mon_status, "risk_status": risk_status, "restarts": total_restarts},
    }


def run_carematrix(case_id: int) -> dict[str, Any]:
    """Run CareMatrix pipeline preserving legacy signature and backward compatibility."""
    return run_agent_demonstration(case_id=case_id, max_samples=100, sample_delay=0.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the CareMatrix Autonomous Agent System.")
    parser.add_argument("--case-id", type=int, default=0, help="Patient case ID (default: 0 for synthetic, 4 for real)")
    parser.add_argument("--samples", type=int, default=100, help="Maximum samples to stream (default: 100)")
    parser.add_argument("--delay", type=float, default=0.03, help="Stream sample delay in seconds (default: 0.03)")
    parser.add_argument("--agent-demo", action="store_true", help="Run end-to-end demonstration with visible concurrent logs")
    parser.add_argument("--fail-risk", action="store_true", help="Demonstrate Risk Agent failure, retries, and fallback")
    parser.add_argument("--fail-monitoring", action="store_true", help="Demonstrate Monitoring freeze, stale detection, and supervisor auto-restart")
    args = parser.parse_args()

    if args.agent_demo or args.fail_risk or args.fail_monitoring:
        run_agent_demonstration(
            case_id=args.case_id,
            max_samples=args.samples,
            sample_delay=args.delay,
            fail_risk=args.fail_risk,
            fail_monitoring=args.fail_monitoring,
        )
    else:
        run_agent_demonstration(
            case_id=args.case_id,
            max_samples=args.samples,
            sample_delay=0.0,
        )
