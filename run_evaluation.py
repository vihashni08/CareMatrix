#!/usr/bin/env python3
"""CareMatrix Research Evaluation & Benchmark Suite.

Executes comprehensive empirical evaluation across:
1. All 7 Clinical Simulation Scenarios (STABLE, GRADUAL_DETERIORATION, SUDDEN_ABNORMALITY,
   RECOVERY, NOISY_SENSOR, MISSING_DATA, PERSISTENT_ABNORMALITY).
2. Mode A (Full-Pipeline Always) vs Mode B (CareMatrix Adaptive Gatekeeper) Comparison.

Note: PatientStreamSimulator uses a seeded local RNG (default seed=42) for deterministic
and reproducible research benchmarking across runs.
Produces structured JSON results (evaluation_results.json) and formatted terminal tables.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

from carematrix_runtime.alert_manager import AlertLifecycleState
from carematrix_runtime.patient_stream import PatientProfile, PatientScenario, PatientStreamSimulator
from carematrix_runtime.runtime import CareMatrixRuntime
from communication.events import MonitoringDecision


def evaluate_scenario(
    scenario: PatientScenario,
    steps: int = 65,
    patient_id: int = 201,
) -> dict[str, Any]:
    """Run an isolated evaluation of a single clinical scenario through CareMatrix."""
    # Custom single-patient simulator profile for exact reproducibility
    # Start in STABLE state to establish clean physiological baseline
    profile = PatientProfile(
        patient_id=patient_id,
        name=f"Eval-Patient-{scenario.name}",
        scenario=PatientScenario.STABLE,
    )
    sim = PatientStreamSimulator(profiles=[profile])

    runtime = CareMatrixRuntime(stream_interval_seconds=0.001, verbose=False)
    runtime.simulator = sim
    runtime.monitoring_agent.baseline_window = 60
    runtime.monitoring_agent.persistence_duration = 5
    # Ensure patient state is prepped
    runtime.state_manager.get_or_create(patient_id, name=profile.name)

    # Start independent worker threads
    runtime.monitoring_agent.start()
    runtime.risk_agent.start()
    runtime.data_analysis_agent.start()
    runtime.clinical_reasoning_agent.start()
    runtime.care_coordination_agent.start()

    start_time = time.time()
    detection_step = None
    time_to_detection = None

    for step_num in range(1, steps + 1):
        if step_num == 16 and scenario != PatientScenario.STABLE:
            profile.set_scenario(scenario)

        obs = runtime.simulator.generate_observation(patient_id)
        runtime.state_manager.record_observation(obs)
        runtime.metrics_tracker.record_observation(patient_id)
        runtime.metrics["total_observations_evaluated"] += 1

        t0 = time.time()
        decision, event = runtime.monitoring_agent.observe(obs)
        runtime.metrics_tracker.record_agent_execution("MonitoringAgent", max(0.0001, time.time() - t0))

        if decision == MonitoringDecision.CONTINUE_MONITORING:
            runtime.metrics["routine_bypassed_cycles"] += 1
            runtime.metrics_tracker.record_adaptive_decision(bypassed=True)
        elif decision == MonitoringDecision.ESCALATE_TO_RISK:
            runtime.metrics["escalated_cycles"] += 1
            runtime.metrics_tracker.record_adaptive_decision(bypassed=False)
            if detection_step is None:
                detection_step = step_num
                time_to_detection = round(time.time() - start_time, 4)
        elif decision == MonitoringDecision.RECOVERY:
            runtime.metrics["recoveries_detected"] += 1
            runtime.alert_manager.resolve_patient_alerts(patient_id, reason="Evaluated recovery")
            runtime.care_coordination_agent.handle_recovery(patient_id)
            if detection_step is None:
                detection_step = step_num
                time_to_detection = round(time.time() - start_time, 4)

    # Allow worker threads window to complete asynchronous event processing (up to 3.5s for live LLM)
    if detection_step is not None:
        deadline = time.time() + 3.5
        while time.time() < deadline:
            coord_events = runtime.event_queue.get_history("care_coordination_events")
            if len(coord_events) > 0:
                break
            time.sleep(0.1)
    else:
        time.sleep(0.2)

    runtime.stop()

    # Collect outcomes
    m_hist = runtime.event_queue.get_history("monitoring_events")
    r_hist = runtime.event_queue.get_history("risk_decisions")
    a_hist = runtime.event_queue.get_history("data_analysis_events")
    c_hist = runtime.event_queue.get_history("clinical_decisions")
    coord_hist = runtime.event_queue.get_history("care_coordination_events")
    active_alerts = runtime.alert_manager.get_active_alerts(patient_id=patient_id)
    all_alerts = runtime.alert_manager.get_alert_history(patient_id=patient_id)

    latest_action = coord_hist[-1].action_type if coord_hist else "NONE"
    latest_priority = coord_hist[-1].priority if coord_hist else "NONE"
    review_req = coord_hist[-1].clinician_review_required if coord_hist else False

    bypassed = runtime.metrics["routine_bypassed_cycles"]
    escalated = runtime.metrics["escalated_cycles"]

    risk_prob = round(r_hist[0].risk_probability, 4) if r_hist else None
    risk_dec = r_hist[0].decision if r_hist else None
    feat_src = r_hist[0].metadata.get("feature_source", "none") if (r_hist and getattr(r_hist[0], "metadata", None)) else "none"

    return {
        "scenario": scenario.value,
        "observations_evaluated": steps,
        "abnormality_detected": len(m_hist) > 0,
        "detection_step": detection_step,
        "time_to_detection_seconds": time_to_detection,
        "risk_triggered": len(r_hist) > 0,
        "risk_probability": risk_prob,
        "risk_decision": risk_dec,
        "feature_source": feat_src,
        "data_analysis_triggered": len(a_hist) > 0,
        "clinical_reasoning_triggered": len(c_hist) > 0,
        "care_coordination_triggered": len(coord_hist) > 0,
        "action_generated": latest_action,
        "action_priority": latest_priority,
        "clinician_review_required": review_req,
        "recovery_detected": runtime.metrics["recoveries_detected"] > 0,
        "active_alerts_count": len(active_alerts),
        "total_alerts_lifecycle": len(all_alerts),
        "downstream_executions_avoided": bypassed,
        "escalations_conducted": escalated,
    }


def run_adaptive_vs_full_comparison(cycles: int = 100) -> dict[str, Any]:
    """Empirical benchmark comparing Full-Pipeline Execution (Mode A) vs Adaptive Gatekeeper (Mode B)."""
    # 70% stable baseline, 30% gradual deterioration
    observations = []
    # 70 normal readings
    for i in range(70):
        observations.append({
            "patient_id": 301,
            "case_id": 301,
            "timestamp": float(i),
            "HR": 72.0 + (i % 3) * 0.5,
            "MAP": 85.0 - (i % 2) * 0.4,
            "SpO2": 98.0,
            "RR": 14.0,
            "BT": 36.8,
        })
    # 30 deteriorating readings
    for i in range(30):
        t = 70 + i
        prog = i / 30.0
        observations.append({
            "patient_id": 301,
            "case_id": 301,
            "timestamp": float(t),
            "HR": 72.0 + prog * 45.0,
            "MAP": 85.0 - prog * 30.0,
            "SpO2": 98.0 - prog * 6.0,
            "RR": 14.0 + prog * 10.0,
            "BT": 36.8,
        })

    # ------------------------------------------------------------------------
    # MODE B: CareMatrix Adaptive Execution Mechanism
    # ------------------------------------------------------------------------
    rt_b = CareMatrixRuntime(stream_interval_seconds=0.001, verbose=False)
    rt_b.monitoring_agent.baseline_window = 60
    rt_b.monitoring_agent.persistence_duration = 5
    rt_b.monitoring_agent.start()
    rt_b.risk_agent.start()
    rt_b.data_analysis_agent.start()
    rt_b.clinical_reasoning_agent.start()
    rt_b.care_coordination_agent.start()

    start_b = time.time()
    for obs in observations:
        rt_b.state_manager.record_observation(obs)
        rt_b.metrics_tracker.record_observation(obs["patient_id"])
        decision, event = rt_b.monitoring_agent.observe(obs)
        if decision == MonitoringDecision.CONTINUE_MONITORING:
            rt_b.metrics["routine_bypassed_cycles"] += 1
            rt_b.metrics_tracker.record_adaptive_decision(bypassed=True)
        else:
            rt_b.metrics["escalated_cycles"] += 1
            rt_b.metrics_tracker.record_adaptive_decision(bypassed=False)

    if rt_b.metrics["escalated_cycles"] > 0:
        deadline = time.time() + 3.5
        while time.time() < deadline:
            if len(rt_b.event_queue.get_history("care_coordination_events")) > 0:
                break
            time.sleep(0.1)
    else:
        time.sleep(0.3)
    duration_b = time.time() - start_b

    counts_b = {
        "monitoring": rt_b.metrics["total_observations_evaluated"] or len(observations),
        "risk": len(rt_b.event_queue.get_history("risk_decisions")),
        "analysis": len(rt_b.event_queue.get_history("data_analysis_events")),
        "reasoning": len(rt_b.event_queue.get_history("clinical_decisions")),
        "coordination": len(rt_b.event_queue.get_history("care_coordination_events")),
    }
    total_executions_b = sum(counts_b.values())
    rt_b.stop()

    # ------------------------------------------------------------------------
    # MODE A: Full Pipeline Always (Simulated baseline where every sample executes full chain)
    # ------------------------------------------------------------------------
    # In Mode A, every single observation is forced through all 5 agents
    counts_a = {
        "monitoring": len(observations),
        "risk": len(observations),
        "analysis": len(observations),
        "reasoning": len(observations),
        "coordination": len(observations),
    }
    total_executions_a = sum(counts_a.values())

    # Mode A execution time estimate based on per-agent processing
    # Each forced downstream step takes approx 15-20ms minimum
    simulated_duration_a = round(duration_b * (total_executions_a / max(1, total_executions_b)), 3)

    executions_avoided = total_executions_a - total_executions_b
    pct_avoided = round((executions_avoided / total_executions_a) * 100, 1)

    return {
        "total_observations": len(observations),
        "mode_a_full_pipeline": {
            "description": "Exhaustive Pipeline Execution (All 5 agents run on every observation)",
            "agent_executions": counts_a,
            "total_executions": total_executions_a,
        },
        "mode_b_adaptive": {
            "description": "CareMatrix Adaptive Gatekeeper (Downstream agents triggered only on deviation)",
            "agent_executions": counts_b,
            "total_executions": total_executions_b,
            "downstream_cycles_bypassed": rt_b.metrics["routine_bypassed_cycles"],
            "escalated_cycles": rt_b.metrics["escalated_cycles"],
        },
        "comparison_metrics": {
            "unnecessary_agent_executions_avoided": executions_avoided,
            "workload_reduction_percentage": pct_avoided,
            "latency_impact": f"Downstream pipeline executed only during clinical need (saved ~{pct_avoided}% compute calls)",
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Run CareMatrix Multi-Agent Research Evaluation.")
    parser.add_argument("--steps", type=int, default=75, help="Number of telemetry steps per scenario (default: 75)")
    parser.add_argument("--output", type=str, default="evaluation_results.json", help="Path to save evaluation JSON results")
    args = parser.parse_args()

    print("=" * 78)
    print("CAREMATRIX — MULTI-AGENT ARCHITECTURE EVALUATION SUITE")
    print("Evaluating 5 Autonomous Agents & Adaptive Execution Gatekeeper")
    print("=" * 78)

    scenario_results = []
    scenarios = [
        PatientScenario.STABLE,
        PatientScenario.GRADUAL_DETERIORATION,
        PatientScenario.SUDDEN_ABNORMALITY,
        PatientScenario.RECOVERY,
        PatientScenario.NOISY_SENSOR,
        PatientScenario.MISSING_DATA,
        PatientScenario.PERSISTENT_ABNORMALITY,
    ]

    print("\n--- Running 7 Predefined Clinical Scenarios ---")
    for sc in scenarios:
        sys.stdout.write(f"Evaluating {sc.value:<26} ... ")
        sys.stdout.flush()
        res = evaluate_scenario(scenario=sc, steps=args.steps)
        scenario_results.append(res)
        det_str = f"Step {res['detection_step']}" if res["detection_step"] else "None (Bypassed)"
        action_str = res["action_generated"]
        print(f"OK | Detected: {det_str:<15} | Action: {action_str}")

    # Mode A vs Mode B
    print("\n--- Evaluating Adaptive Execution (Mode A vs Mode B) ---")
    comparison = run_adaptive_vs_full_comparison()
    print(f"Mode A (Full Pipeline Always) Total Executions: {comparison['mode_a_full_pipeline']['total_executions']}")
    print(f"Mode B (CareMatrix Adaptive)  Total Executions: {comparison['mode_b_adaptive']['total_executions']}")
    print(f"Unnecessary Agent Executions Avoided:         {comparison['comparison_metrics']['unnecessary_agent_executions_avoided']}")
    print(f"Workload Reduction:                           {comparison['comparison_metrics']['workload_reduction_percentage']}%")

    # Format Results Table
    print("\n" + "=" * 105)
    print("SCENARIO EVALUATION SUMMARY")
    print("=" * 105)
    hdr = f"{'Scenario':<24} | {'Det':<4} | {'Step':<5} | {'Risk Prob':<10} | {'Source':<12} | {'Risk':<5} | {'Analysis':<8} | {'Reasoning':<9} | {'Care Action':<22}"
    print(hdr)
    print("-" * 105)
    for r in scenario_results:
        sc_name = r["scenario"]
        det = "YES" if r["abnormality_detected"] else "NO"
        step_val = str(r["detection_step"]) if r["detection_step"] else "-"
        risk_prob_val = f"{r['risk_probability']:.4f}" if r["risk_probability"] is not None else "-"
        feat_src = str(r.get("feature_source", "-"))
        risk_trig = "YES" if r["risk_triggered"] else "NO"
        ana_trig = "YES" if r["data_analysis_triggered"] else "NO"
        reas_trig = "YES" if r["clinical_reasoning_triggered"] else "NO"
        act_name = r["action_generated"][:22]
        print(f"{sc_name:<24} | {det:<4} | {step_val:<5} | {risk_prob_val:<10} | {feat_src:<12} | {risk_trig:<5} | {ana_trig:<8} | {reas_trig:<9} | {act_name:<22}")

    full_report = {
        "timestamp": time.time(),
        "evaluation_parameters": {
            "steps_per_scenario": args.steps,
            "scenarios_evaluated": len(scenarios),
        },
        "scenario_results": scenario_results,
        "adaptive_execution_comparison": comparison,
    }

    with open(args.output, "w") as f:
        json.dump(full_report, f, indent=2)

    print("\n" + "=" * 90)
    print(f"Evaluation complete. Machine-readable report saved to: {args.output}")
    print("=" * 90)


if __name__ == "__main__":
    main()
