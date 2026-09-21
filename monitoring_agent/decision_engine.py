"""Autonomous decision engine for the CareMatrix Monitoring Agent."""

from __future__ import annotations

from typing import Any

import pandas as pd

from communication.events import MonitoringDecision, MonitoringEvent
from monitoring_agent.config import (
    ALERT_COOLDOWN_SECONDS,
    DEVIATION_THRESHOLDS,
    RECOVERY_DURATION_SECONDS,
    SEVERITY_MIN_VITALS,
    SEVERITY_MULTIPLIERS,
    SEVERITY_PERSISTENCE,
)
from monitoring_agent.state import PatientMonitoringState, SEVERITY_LEVELS, VITAL_COLUMNS, _NUM_SEVERITY, _SEVERITY_NUM


class MonitoringDecisionEngine:
    """Evaluates physiological analysis and state to determine autonomous agent actions."""

    def __init__(
        self,
        cooldown_seconds: int = ALERT_COOLDOWN_SECONDS,
        recovery_duration_seconds: int = RECOVERY_DURATION_SECONDS,
    ):
        self.cooldown_seconds = cooldown_seconds
        self.recovery_duration_seconds = recovery_duration_seconds

    def evaluate(
        self,
        state: PatientMonitoringState,
        latest_clean_values: dict[str, float],
        latest_baselines: dict[str, float],
        latest_deviations: dict[str, float],
        latest_trends: dict[str, str],
        latest_signal_quality: dict[str, str],
        invalid_mask: dict[str, bool],
    ) -> tuple[MonitoringDecision, MonitoringEvent | None, dict[str, Any]]:
        """Evaluate observation against current state and make an autonomous decision.

        Returns:
            (decision, optional_event_to_publish, context_details)
        """
        timestamp = state.latest_timestamp if state.latest_timestamp is not None else 0.0
        current_sample_idx = state.total_observations - 1

        # Check if baseline is established across all required vitals
        baseline_ready = all(
            vital in latest_baselines and pd.notna(latest_baselines[vital])
            for vital in VITAL_COLUMNS
        )

        # Classify per-vital severity
        vital_severity: dict[str, str] = {}
        alert_eligible_vitals: list[str] = []

        for vital in VITAL_COLUMNS:
            dev = latest_deviations.get(vital, 0.0)
            is_invalid = invalid_mask.get(vital, False)
            base_thresh = DEVIATION_THRESHOLDS.get(vital, 0.20)

            vital_sev = "normal"
            if baseline_ready and not is_invalid and pd.notna(dev):
                for s_level in reversed(SEVERITY_LEVELS):  # critical down to mild
                    if dev >= base_thresh * SEVERITY_MULTIPLIERS[s_level]:
                        vital_sev = s_level
                        break

            vital_severity[vital] = vital_sev
            if vital_sev != "normal":
                alert_eligible_vitals.append(vital)

        # Overall severity = max across vitals
        overall_severity_num = max(
            (_SEVERITY_NUM.get(vital_severity[v], 0) for v in VITAL_COLUMNS),
            default=0,
        )
        overall_severity = _NUM_SEVERITY[overall_severity_num]

        # Candidate alert determination
        # Severe / critical allow single vital; mild / moderate require >= 2 vitals
        required_vitals = SEVERITY_MIN_VITALS.get(overall_severity, 2)
        is_candidate = (
            baseline_ready
            and overall_severity != "normal"
            and len(alert_eligible_vitals) >= required_vitals
        )

        # Update state persistence counters
        if is_candidate:
            state.consecutive_candidate_samples += 1
            if state.candidate_start_sample_idx is None:
                state.candidate_start_sample_idx = current_sample_idx
                state.candidate_start_timestamp = timestamp
        else:
            state.consecutive_candidate_samples = 0
            state.candidate_start_sample_idx = None
            state.candidate_start_timestamp = None

        # Required persistence samples based on severity
        req_persistence = SEVERITY_PERSISTENCE.get(overall_severity, 10)
        is_persistent = is_candidate and (state.consecutive_candidate_samples >= req_persistence)

        # Build detailed vital information snapshot
        vital_details: dict[str, dict[str, Any]] = {}
        for vital in VITAL_COLUMNS:
            curr = latest_clean_values.get(vital, 0.0)
            base = latest_baselines.get(vital, 0.0)
            direction = "stable"
            if pd.notna(curr) and pd.notna(base):
                if curr > base:
                    direction = "increasing"
                elif curr < base:
                    direction = "decreasing"

            vital_details[vital] = {
                "current": float(curr) if pd.notna(curr) else 0.0,
                "baseline": float(base) if pd.notna(base) else 0.0,
                "relative_deviation": float(latest_deviations.get(vital, 0.0)) if pd.notna(latest_deviations.get(vital)) else 0.0,
                "severity": vital_severity.get(vital, "normal"),
                "direction": direction,
                "trend": latest_trends.get(vital, "stable"),
                "signal_quality": latest_signal_quality.get(vital, "good"),
            }

        # Context details for state updating
        context = {
            "timestamp": timestamp,
            "baseline_ready": baseline_ready,
            "vital_severity": vital_severity,
            "overall_severity": overall_severity,
            "alert_eligible_vitals": alert_eligible_vitals,
            "is_candidate": is_candidate,
            "is_persistent": is_persistent,
            "vital_details": vital_details,
        }

        # -------------------------------------------------------------
        # Decision logic following Alert Lifecycle
        # -------------------------------------------------------------
        # Case A: An alert event is currently ACTIVE
        if state.active_event_id is not None:
            active_id = state.active_event_id
            if is_candidate:
                # Active event continues, reset recovery counter
                state.recovery_counter = 0
                state.active_event_state = "alert_active"
                state.event_tracker.update(active_id, timestamp, vital_details, "alert_active")
                # Do NOT generate duplicate escalation event for same active event
                return MonitoringDecision.NO_ACTION, None, context
            else:
                # Potential recovery in progress
                state.recovery_counter += 1
                state.active_event_state = "recovering"
                if state.recovery_counter >= self.recovery_duration_seconds:
                    # Confirmed recovery
                    state.active_event_state = "alert_recovered"
                    snapshot = state.event_tracker.recover(active_id, timestamp, vital_details)
                    state.cooldown_until_sample = current_sample_idx + self.cooldown_seconds
                    state.active_event_id = None
                    state.recovery_counter = 0
                    state.total_recoveries += 1

                    recovery_event = None
                    if snapshot is not None:
                        n_vitals = len(snapshot.get("affected_vitals", []))
                        reason = (
                            f"{'Multi-vital' if n_vitals > 1 else 'Single-vital'} "
                            f"physiological deviation recovered toward baseline"
                        )
                        recovery_event = MonitoringEvent(
                            patient_id=state.case_id,
                            event_id=snapshot["event_id"],
                            timestamp=timestamp,
                            event_type="alert_recovered",
                            severity=snapshot.get("severity", "moderate"),
                            affected_vitals=list(snapshot.get("affected_vitals", [])),
                            current_values={v: vital_details[v]["current"] for v in snapshot.get("affected_vitals", [])},
                            baseline_values={v: vital_details[v]["baseline"] for v in snapshot.get("affected_vitals", [])},
                            deviation_values={v: vital_details[v]["relative_deviation"] for v in snapshot.get("affected_vitals", [])},
                            trends={v: vital_details[v]["trend"] for v in snapshot.get("affected_vitals", [])},
                            signal_quality={v: vital_details[v]["signal_quality"] for v in snapshot.get("affected_vitals", [])},
                            persistence_duration=int(snapshot.get("duration_seconds", 0)),
                            recommended_action="close_event",
                            vital_details={v: vital_details[v] for v in snapshot.get("affected_vitals", [])},
                            vital_summary=snapshot.get("vital_summary", {}),
                            metadata={"reason": reason},
                        )

                    return MonitoringDecision.RECOVERY, recovery_event, context
                else:
                    state.event_tracker.update(active_id, timestamp, vital_details, "recovering")
                    return MonitoringDecision.NO_ACTION, None, context

        # Case B: No active event, but persistence requirement newly met
        elif is_persistent and current_sample_idx >= state.cooldown_until_sample:
            event_numeric_id = state.next_event_id
            state.next_event_id += 1
            state.active_event_id = event_numeric_id
            state.active_event_state = "alert_started"
            state.total_escalations += 1

            start_timestamp = state.candidate_start_timestamp if state.candidate_start_timestamp is not None else timestamp
            affected = list(alert_eligible_vitals)

            # Start event in tracker
            state.event_tracker.start(
                event_numeric_id,
                start_timestamp,
                affected,
                vital_details,
                severity=overall_severity,
            )
            snapshot = state.event_tracker.snapshot(event_numeric_id, timestamp, "alert_started")

            n_vitals = len(affected)
            reason = (
                f"Persistent {'multi-vital' if n_vitals > 1 else 'single-vital'} "
                f"physiological deviation detected (severity: {overall_severity})"
            )

            monitoring_event = MonitoringEvent(
                patient_id=state.case_id,
                event_id=snapshot["event_id"],
                timestamp=timestamp,
                event_type="alert_started",
                severity=overall_severity,
                affected_vitals=affected,
                current_values={v: vital_details[v]["current"] for v in affected},
                baseline_values={v: vital_details[v]["baseline"] for v in affected},
                deviation_values={v: vital_details[v]["relative_deviation"] for v in affected},
                trends={v: vital_details[v]["trend"] for v in affected},
                signal_quality={v: vital_details[v]["signal_quality"] for v in affected},
                persistence_duration=int(snapshot.get("duration_seconds", req_persistence)),
                recommended_action="assess_patient_risk",
                vital_details={v: vital_details[v] for v in affected},
                vital_summary=snapshot.get("vital_summary", {}),
                metadata={
                    "start_timestamp": start_timestamp,
                    "reason": reason,
                    "consecutive_samples": state.consecutive_candidate_samples,
                },
            )

            return MonitoringDecision.ESCALATE_TO_RISK, monitoring_event, context

        # Case C: Deviating but not yet persistent, or baseline establishing, or normal
        elif is_candidate:
            state.active_event_state = "deviating"
            return MonitoringDecision.NO_ACTION, None, context
        else:
            state.active_event_state = "normal"
            return MonitoringDecision.CONTINUE_MONITORING, None, context

