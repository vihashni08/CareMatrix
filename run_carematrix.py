"""CareMatrix: Multi-Agent Autonomous Physiological Monitoring and Risk Assessment.

Demonstrates independent, continuously running AI agents:
                SUPERVISOR
             independent loop
          /         |        \
         ↓          ↓         ↓
     MONITORING   RISK   DATA ANALYSIS   CLINICAL REASONING
         │          │          │                 │
         └──→ QUEUE ←──────────┴─────────────────┘
                │
                ↓
          Decisions & Care Pathways
"""

from __future__ import annotations

import argparse
import datetime
import time
import warnings
from typing import Any

warnings.filterwarnings("ignore")

from care_coordination_agent import CareCoordinationAgent
from clinical_reasoning_agent import ClinicalReasoningAgent
from communication.event_queue import EventQueue
from communication.events import (
    CareCoordinationEvent,
    ClinicalReasoningEvent,
    MonitoringDecision,
    MonitoringEvent,
    RiskDecisionEvent,
)
from data_analysis_agent import DataAnalysisAgent
from monitoring_agent.monitoring_agent import (
    MonitoringAgent,
    monitor_dataframe,
    run_monitoring,
)
from carematrix_runtime import PatientStreamReplayer
from risk_agent.risk_agent import RiskAgent
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


def _val(obj: Any, key: str, default: Any = None) -> Any:
    """Helper to safely extract an attribute from an object or dictionary."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def print_risk_result(risk_results: list[Any]) -> None:
    """Print Risk Agent prediction and decision output."""
    print("\n" + "=" * 60)
    print("RISK AGENT OUTPUT")
    print("=" * 60)

    if not risk_results:
        print("\nNo alerts were escalated to the Risk Agent.")
        return

    for number, result in enumerate(risk_results, start=1):
        print(f"\nRisk Assessment {number}")
        print(f"  Event ID            : {_val(result, 'event_id', 'N/A')}")
        prob = _val(result, "risk_probability", 0.0)
        print(f"  Risk Probability    : {prob * 100:.2f}%" if prob >= 0 else "  Risk Probability    : N/A")
        print(f"  Risk Level          : {_val(result, 'risk_level', 'N/A')}")
        print(f"  Decision            : {_val(result, 'decision', _val(result, 'risk_level', 'N/A'))}")
        print(f"  Model Tool          : {_val(result, 'model', _val(result, 'model_tool', 'RandomForestClassifier'))}")
        thresh = _val(result, "threshold", 0.16)
        print(f"  Threshold           : {thresh:.2f}")
        print(f"  Alert Timestamp     : {_val(result, 'timestamp', 'N/A')}")
        print(f"  Severity            : {_val(result, 'severity', 'N/A')}")
        vitals = _val(result, "affected_vitals", [])
        print(f"  Affected Vitals     : {', '.join(vitals) if vitals else 'N/A'}")
        if _val(result, "analysis_level"):
            print(f"  Analysis Level      : {_val(result, 'analysis_level')}")
        if _val(result, "requested_checks"):
            print(f"  Requested Checks    : {', '.join(_val(result, 'requested_checks'))}")
        if _val(result, "verification_required") is not None:
            print(f"  Verification Req    : {_val(result, 'verification_required')}")
        if _val(result, "reason"):
            print(f"  Reason              : {_val(result, 'reason')}")
        if _val(result, "recommended_action"):
            print(f"  Action              : {_val(result, 'recommended_action')}")


def print_data_analysis_result(analysis_results: list[Any]) -> None:
    """Print Data Analysis Agent findings and cross-agent verification output."""
    print("\n" + "=" * 60)
    print("DATA ANALYSIS AGENT OUTPUT")
    print("=" * 60)

    if not analysis_results:
        print("\nNo analytical evaluations completed.")
        return

    for number, res in enumerate(analysis_results, start=1):
        event_id = _val(res, "event_id", "N/A")
        status = _val(res, "analysis_status", "N/A")
        quality_flag = _val(res, "data_quality_flag", False)
        consistency = _val(res, "evidence_consistency", "UNKNOWN")
        verify_req = _val(res, "verification_required", False)
        metrics = _val(res, "trend_metrics", {}) or {}
        patterns = _val(res, "pattern_identified", []) or []
        changes = _val(res, "important_changes", []) or []
        conflict_flags = _val(res, "conflict_flags", []) or []

        print(f"\nData Analysis Assessment {number}")
        print(f"  Event ID            : {event_id}")
        print(f"  Analysis Status     : {status}")
        print(f"  Data Quality Flag   : {'COMPROMISED (Sensor Noise / Missingness)' if quality_flag else 'NOMINAL'}")
        print(f"  Evidence Consistency: {consistency.upper() if isinstance(consistency, str) else consistency}")
        print(f"  Verification Req    : {verify_req}")
        if metrics:
            print("  Trend Metrics       :")
            for vital, m in metrics.items():
                if isinstance(m, dict):
                    cur = m.get("current_value")
                    tr = m.get("trend", "N/A")
                    cur_str = f"{cur:.2f}" if isinstance(cur, (int, float)) else str(cur)
                    print(f"    - {vital:<6}: current={cur_str:<7} trend={tr}")
                else:
                    print(f"    - {vital:<6}: {m}")
        if patterns:
            print("  Identified Patterns :")
            for p in patterns:
                print(f"    * {p}")
        if changes:
            print("  Important Changes   :")
            for c in changes:
                print(f"    ! {c}")
        if conflict_flags:
            print(f"  Conflict Flags      : {', '.join(conflict_flags)}")


def print_clinical_reasoning_result(clinical_results: list[Any]) -> None:
    """Print Clinical Reasoning Agent synthesized clinical pathways."""
    print("\n" + "=" * 60)
    print("CLINICAL REASONING AGENT OUTPUT")
    print("=" * 60)

    if not clinical_results:
        print("\nNo clinical evaluations completed.")
        return

    for number, res in enumerate(clinical_results, start=1):
        event_id = getattr(res, "event_id", "N/A")
        priority = getattr(res, "priority", "ROUTINE")
        summary = getattr(res, "clinical_summary", "N/A")
        reliability = getattr(res, "data_reliability", "HIGH")
        actions = getattr(res, "recommended_actions", [])
        confidence = getattr(res, "confidence", 1.0)
        consistency = getattr(res, "evidence_consistency", "SUPPORTING")
        verify_req = getattr(res, "verification_required", False)
        supporting = getattr(res, "supporting_evidence", [])
        conflicting = getattr(res, "conflicting_evidence", [])
        meta = getattr(res, "metadata", {}) or {}
        reasoning_mode = meta.get("reasoning_mode", "DETERMINISTIC")
        llm_conflict = meta.get("llm_reasoning_conflict", False)
        retrieval_status = meta.get("retrieval_status")
        knowledge_sources = meta.get("knowledge_sources", [])
        retrieved_evidence = meta.get("retrieved_evidence", [])

        print(f"\nClinical Assessment {number}")
        print(f"  Event ID            : {event_id}")
        print(f"  Reasoning Mode      : {reasoning_mode}")
        if retrieval_status:
            print(f"  Retrieval Status    : {retrieval_status}")
        if knowledge_sources:
            print("  Knowledge Sources   :")
            for src in knowledge_sources:
                doc_id = src.get("document_id", "DOC")
                title = src.get("title", "")
                sec = src.get("section", "")
                print(f"    - [{doc_id}] {title} ({sec})")
        if retrieved_evidence:
            print("  Retrieved Evidence  :")
            for ev in retrieved_evidence:
                print(f"    * {ev}")
        if llm_conflict:
            print(f"  LLM Conflict        : True (Deterministic Safety Boundary Overrode LLM)")
        print(f"  Priority            : {priority}")
        print(f"  Evidence Consistency: {consistency}")
        print(f"  Confidence          : {confidence:.2f}")
        print(f"  Verification Req    : {verify_req}")
        print(f"  Data Reliability    : {reliability}")
        print(f"  Clinical Summary    : {summary}")
        if supporting:
            print("  Supporting Evidence :")
            for s in supporting:
                print(f"    + {s}")
        if conflicting:
            print("  Conflicting Evidence:")
            for c in conflicting:
                print(f"    ! {c}")
        if actions:
            print("  Recommended Actions :")
            for action in actions:
                print(f"    - {action}")


def print_care_coordination_result(coordination_results: list[Any]) -> None:
    """Print Care Coordination Agent actionable workflows, orders, and escalation pathways."""
    print("\n" + "=" * 60)
    print("CARE COORDINATION AGENT OUTPUT")
    print("=" * 60)

    if not coordination_results:
        print("\nNo care coordination workflows generated.")
        return

    for number, res in enumerate(coordination_results, start=1):
        event_id = _val(res, "event_id", "N/A")
        action_type = _val(res, "action_type", "N/A")
        priority = _val(res, "priority", "ROUTINE")
        status = _val(res, "status", "NEW")
        review_req = _val(res, "clinician_review_required", False)
        pathway = _val(res, "escalation_pathway", "Standard Inpatient Protocol")
        reason = _val(res, "reason", "N/A")
        orders = _val(res, "suggested_orders", [])
        meta = _val(res, "metadata", {}) or {}

        print(f"\nCare Coordination Workflow {number}")
        print(f"  Event ID            : {event_id}")
        print(f"  Action Type         : {action_type}")
        print(f"  Priority            : {priority}")
        print(f"  Status              : {status}")
        print(f"  Clinician Review Req: {review_req}")
        print(f"  Escalation Pathway  : {pathway}")
        print(f"  Coordination Reason : {reason}")
        if meta.get("is_duplicate_suppressed"):
            print("  Deduplication       : Duplicate ongoing condition suppressed (no alarm spam)")
        if orders:
            print("  Suggested Order Set :")
            for ord in orders:
                print(f"    [ ] {ord}")


def run_agent_demonstration(
    case_id: int = 0,
    max_samples: int = 100,
    start_sample: int = 0,
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

    # 4. Initialize Data Analysis Agent
    data_analysis_agent = DataAnalysisAgent(
        event_queue=event_queue,
        verbose=True,
    )

    # 5. Initialize Clinical Reasoning Agent
    clinical_reasoning_agent = ClinicalReasoningAgent(
        event_queue=event_queue,
        verbose=True,
    )

    # 6. Initialize Care Coordination Agent
    care_coordination_agent = CareCoordinationAgent(
        event_queue=event_queue,
        verbose=True,
    )

    if fail_risk:
        risk_agent.inject_failure = True
        print("[Demo Setup] Risk Agent configured with intentional ML tool failure hook.")

    if fail_monitoring:
        # Simulate unhandled hang after 30 samples to test supervisor stale detection
        monitoring_agent._hang_at_sample = 30
        print("[Demo Setup] Monitoring Agent configured with intentional freeze hook at sample 30.")

    # 7. Initialize Independent Supervisor Monitor (Supervises all 5 agents)
    supervisor = AgentSupervisor(
        event_queue=event_queue,
        heartbeat_timeout_seconds=1.0 if fail_monitoring else 5.0,
        check_interval_seconds=0.2,
        auto_recover=True,
        verbose=True,
    )
    supervisor.register_agent(monitoring_agent)
    supervisor.register_agent(risk_agent)
    supervisor.register_agent(data_analysis_agent)
    supervisor.register_agent(clinical_reasoning_agent)
    supervisor.register_agent(care_coordination_agent)

    # 8. Start Independent Background Threads
    print("\n--- Starting Independent Agent Workers ---")
    supervisor.start()
    risk_agent.start()
    data_analysis_agent.start()
    clinical_reasoning_agent.start()
    care_coordination_agent.start()

    # 9. Stream patient observations through Monitoring Agent worker thread
    replayer = PatientStreamReplayer(case_id=case_id, delay_seconds=0.0)
    stream_generator = replayer.stream(max_samples=max_samples, start_sample=start_sample)

    monitoring_agent.start(
        stream_source=stream_generator,
        sample_delay=sample_delay,
    )

    # Wait for monitoring worker to finish (or until stopped)
    join_timeout = max(15.0, max_samples * (sample_delay + 0.05))
    monitoring_agent.join(timeout=join_timeout)
    monitoring_agent.stop()

    # Allow downstream agent worker threads to finish any pending items on event queue
    max_wait = 90.0 if clinical_reasoning_agent.enable_llm and clinical_reasoning_agent.llm_reasoner.is_available else 6.0
    start_wait = time.time()
    while (time.time() - start_wait) < max_wait:
        analysis_count = len(event_queue.get_history("data_analysis_events"))
        clinical_count = len(event_queue.get_history("clinical_decisions"))
        care_count = len(event_queue.get_history("care_coordination_events"))
        if analysis_count > 0 and clinical_count >= analysis_count and care_count >= clinical_count:
            # Brief pause to ensure all agent states are completely recorded
            time.sleep(0.5)
            break
        time.sleep(0.3)

    # 10. Gracefully stop workers
    risk_agent.stop()
    data_analysis_agent.stop()
    clinical_reasoning_agent.stop()
    care_coordination_agent.stop()
    supervisor.stop()
    monitoring_agent.join(timeout=2.0)
    risk_agent.join(timeout=2.0)
    data_analysis_agent.join(timeout=2.0)
    clinical_reasoning_agent.join(timeout=10.0)
    care_coordination_agent.join(timeout=2.0)
    supervisor.join(timeout=1.0)
    event_queue.shutdown()

    # 11. Compile and Print Demo Summary
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

    print("\n[Risk Agent]")
    print(f"  events_received     = {len(risk_agent.state.received_events)}")
    print(f"  predictions         = {len(risk_history)}")
    print(f"  high_risk           = {high_risk_cnt}")
    print(f"  low_risk            = {low_risk_cnt}")
    print(f"  failures            = {fail_cnt}")
    print(f"  model_tool          = {risk_agent.model_name}")

    # Data Analysis stats
    analysis_history = event_queue.get_history("data_analysis_events")
    print("\n[Data Analysis Agent]")
    print(f"  analyses            = {len(analysis_history)}")
    print(f"  partial_analyses    = {sum(1 for e in analysis_history if getattr(e, 'analysis_status', '') == 'partial_analysis')}")

    # Clinical Reasoning stats
    clinical_history = event_queue.get_history("clinical_decisions")
    urgent_cnt = sum(1 for c in clinical_history if getattr(c, "priority", "") == "URGENT")
    elevated_cnt = sum(1 for c in clinical_history if getattr(c, "priority", "") == "ELEVATED")
    routine_cnt = sum(1 for c in clinical_history if getattr(c, "priority", "") == "ROUTINE")
    llm_rag_cnt = sum(1 for c in clinical_history if getattr(c, "metadata", {}).get("reasoning_mode") == "LLM_RAG")
    llm_assisted_cnt = sum(1 for c in clinical_history if getattr(c, "metadata", {}).get("reasoning_mode") == "LLM_ASSISTED")
    llm_fallback_cnt = sum(1 for c in clinical_history if getattr(c, "metadata", {}).get("reasoning_mode") == "LLM_FALLBACK")
    deterministic_cnt = sum(1 for c in clinical_history if getattr(c, "metadata", {}).get("reasoning_mode", "DETERMINISTIC") == "DETERMINISTIC")

    print("\n[Clinical Reasoning Agent]")
    print(f"  clinical_decisions  = {len(clinical_history)}")
    print(f"  urgent_priorities   = {urgent_cnt}")
    print(f"  elevated_priorities = {elevated_cnt}")
    print(f"  routine_priorities  = {routine_cnt}")
    print(f"  llm_rag             = {llm_rag_cnt}")
    print(f"  llm_assisted        = {llm_assisted_cnt}")
    print(f"  llm_fallback        = {llm_fallback_cnt}")
    print(f"  deterministic       = {deterministic_cnt}")

    # Care Coordination stats
    care_history = event_queue.get_history("care_coordination_events")
    urgent_act_cnt = sum(1 for c in care_history if getattr(c, "priority", "") == "URGENT")
    elevated_act_cnt = sum(1 for c in care_history if getattr(c, "priority", "") == "ELEVATED")
    routine_act_cnt = sum(1 for c in care_history if getattr(c, "priority", "") == "ROUTINE")
    review_req_cnt = sum(1 for c in care_history if getattr(c, "clinician_review_required", False))

    print("\n[Care Coordination Agent]")
    print(f"  workflows_generated = {len(care_history)}")
    print(f"  urgent_actions      = {urgent_act_cnt}")
    print(f"  elevated_actions    = {elevated_act_cnt}")
    print(f"  routine_actions     = {routine_act_cnt}")
    print(f"  review_required     = {review_req_cnt}")

    # Detailed Multi-Agent outputs
    print_risk_result(risk_history)
    print_data_analysis_result(analysis_history)
    print_clinical_reasoning_result(clinical_history)
    print_care_coordination_result(care_history)

    # Supervisor stats
    health = supervisor.check_health()
    mon_status = health.get("MonitoringAgent", {}).get("status", "UNKNOWN").upper()
    risk_status = health.get("RiskAgent", {}).get("status", "UNKNOWN").upper()
    analysis_status = health.get("DataAnalysisAgent", {}).get("status", "UNKNOWN").upper()
    clinical_status = health.get("ClinicalReasoningAgent", {}).get("status", "UNKNOWN").upper()
    coordination_status = health.get("CareCoordinationAgent", {}).get("status", "UNKNOWN").upper()
    total_restarts = sum(info.get("restart_count", 0) for info in health.values())

    print("\n[Supervisor]")
    print(f"  monitoring_status   = {mon_status}")
    print(f"  risk_status         = {risk_status}")
    print(f"  analysis_status     = {analysis_status}")
    print(f"  clinical_status     = {clinical_status}")
    print(f"  coordination_status = {coordination_status}")
    print(f"  total_restarts      = {total_restarts}")
    print(f"  failures_logged     = {len(supervisor.failure_log)}")

    print("\n" + "=" * 70)
    print("AGENT DEMONSTRATION COMPLETE")
    print("=" * 70)

    return {
        "monitoring_stats": {"observations": obs_count, "escalations": esc_count, "recoveries": rec_count},
        "risk_stats": {"predictions": len(risk_history), "high_risk": high_risk_cnt, "low_risk": low_risk_cnt},
        "data_analysis_stats": {"analyses": len(analysis_history)},
        "clinical_reasoning_stats": {
            "decisions": len(clinical_history),
            "urgent": urgent_cnt,
            "llm_rag": llm_rag_cnt,
            "llm_assisted": llm_assisted_cnt,
            "llm_fallback": llm_fallback_cnt,
            "deterministic": deterministic_cnt,
        },
        "care_coordination_stats": {
            "workflows": len(care_history),
            "urgent": urgent_act_cnt,
            "elevated": elevated_act_cnt,
            "routine": routine_act_cnt,
            "review_required": review_req_cnt,
        },
        "supervisor_stats": {
            "monitoring_status": mon_status,
            "risk_status": risk_status,
            "analysis_status": analysis_status,
            "clinical_status": clinical_status,
            "coordination_status": coordination_status,
            "restarts": total_restarts,
        },
    }


def run_carematrix(case_id: int) -> dict[str, Any]:
    """Run CareMatrix pipeline preserving legacy signature and backward compatibility."""
    return run_agent_demonstration(case_id=case_id, max_samples=100, sample_delay=0.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the CareMatrix Autonomous Agent System.")
    parser.add_argument("--case-id", type=int, default=0, help="Patient case ID (default: 0 for synthetic, 4 for real)")
    parser.add_argument("--samples", type=int, default=100, help="Maximum samples to stream (default: 100)")
    parser.add_argument("--start-sample", type=int, default=0, help="Starting sample offset to stream (e.g. 800 for Case 4)")
    parser.add_argument("--delay", type=float, default=0.03, help="Stream sample delay in seconds (default: 0.03)")
    parser.add_argument("--agent-demo", action="store_true", help="Run end-to-end demonstration with visible concurrent logs")
    parser.add_argument("--fail-risk", action="store_true", help="Demonstrate Risk Agent failure, retries, and fallback")
    parser.add_argument("--fail-monitoring", action="store_true", help="Demonstrate Monitoring freeze, stale detection, and supervisor auto-restart")
    args = parser.parse_args()

    if args.agent_demo or args.fail_risk or args.fail_monitoring:
        run_agent_demonstration(
            case_id=args.case_id,
            max_samples=args.samples,
            start_sample=args.start_sample,
            sample_delay=args.delay,
            fail_risk=args.fail_risk,
            fail_monitoring=args.fail_monitoring,
        )
    else:
        run_agent_demonstration(
            case_id=args.case_id,
            max_samples=args.samples,
            start_sample=args.start_sample,
            sample_delay=0.0,
        )
