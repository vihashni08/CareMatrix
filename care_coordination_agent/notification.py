"""Clinical notification and workflow dispatching utilities."""

from __future__ import annotations

import datetime
from typing import Any

from communication.events import CareCoordinationEvent


class NotificationDispatcher:
    """Formats structured clinical alerts and handover summaries for hospital systems."""

    @staticmethod
    def format_clinician_notification(event: CareCoordinationEvent) -> dict[str, Any]:
        """Format a real-time notification payload for clinicians or nurse call systems."""
        urgency_prefix = {
            "URGENT": "🚨 [URGENT CLINICAL ALERT]",
            "ELEVATED": "⚠️ [ELEVATED SURVEILLANCE]",
            "ROUTINE": "ℹ️ [ROUTINE MONITORING]",
        }.get(event.priority, "📋 [CARE UPDATE]")

        timestamp_str = datetime.datetime.fromtimestamp(event.created_at).strftime("%Y-%m-%d %H:%M:%S")

        return {
            "title": f"{urgency_prefix} - Patient {event.patient_id}",
            "patient_id": event.patient_id,
            "action_id": event.action_id,
            "timestamp": timestamp_str,
            "action_type": event.action_type,
            "priority": event.priority,
            "pathway": event.escalation_pathway,
            "summary": event.clinical_summary,
            "reason": event.reason,
            "clinician_review_required": event.clinician_review_required,
            "orders": list(event.suggested_orders),
            "evidence_consistency": event.evidence_consistency,
            "data_reliability": event.data_reliability,
        }

    @staticmethod
    def format_handover_report(patient_id: int, active_actions: list[CareCoordinationEvent]) -> str:
        """Format a nursing/physician handover summary text."""
        lines = [
            f"=== CLINICAL CARE COORDINATION HANDOVER: PATIENT {patient_id} ===",
            f"Active Actions: {len(active_actions)}",
        ]
        for act in active_actions:
            lines.append(f"- [{act.priority}] {act.action_type}: {act.reason}")
            if act.suggested_orders:
                lines.append(f"  Suggested Orders: {', '.join(act.suggested_orders[:3])}")
        return "\n".join(lines)
