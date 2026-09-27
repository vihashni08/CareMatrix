"""Episodic patient memory and clinician feedback tracker for CareMatrix.

Maintains a bounded rolling history (default 20 episodes) of ClinicalReasoningEvent
and CareCoordinationEvent decisions per patient, backed by the SQLite event log.
Tracks clinician lifecycle actions (acknowledgments, resolutions, overrides, and
false-positive flags) with latency metrics, feeding additive historical context
into downstream agent reasoning without compromising deterministic safety rules.
"""

from __future__ import annotations

import collections
from dataclasses import asdict, dataclass, field
import json
import logging
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any

from communication.events import CareCoordinationEvent, ClinicalReasoningEvent

logger = logging.getLogger("CareMatrix.PatientMemory")


@dataclass
class EpisodeSummary:
    """Structured record of an agent decision and subsequent clinician feedback."""

    patient_id: int
    event_id: str
    timestamp: float
    event_type: str  # "clinical_reasoning" or "care_coordination"
    priority: str = "ROUTINE"
    clinical_summary: str = ""
    findings: list[str] = field(default_factory=list)
    recommended_actions: list[str] = field(default_factory=list)
    action_type: str | None = None
    action_id: str | None = None
    correlation_id: str | None = None

    # Clinician Feedback Lifecycle
    status: str = "PENDING"  # "PENDING", "ACKNOWLEDGED", "RESOLVED", "OVERRIDDEN", "FALSE_POSITIVE"
    acknowledged: bool = False
    acknowledged_at: float | None = None
    acknowledged_by: str | None = None
    time_to_acknowledge_seconds: float | None = None

    resolved: bool = False
    resolved_at: float | None = None
    resolution_reason: str | None = None
    time_to_resolve_seconds: float | None = None

    is_override: bool = False
    override_reason: str | None = None
    is_false_positive: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PatientMemory:
    """Per-patient episodic memory backed by the Phase 1 SQLite event log.

    Guarantees:
    1. Bounded memory: Stores at most `max_episodes_per_patient` (default 20) in a FIFO deque.
    2. Durable persistence: Backed by the existing SQLite `events` log without creating a separate store.
    3. Bidirectional correlation: Links clinician acknowledge/resolve actions back to originating agent event chains.
    4. Safe additive context: Provides natural language summaries for LLM prompt augmentation.
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        event_log_writer: Any | None = None,
        event_queue: Any | None = None,
        max_episodes_per_patient: int = 20,
    ):
        self.db_path = str(db_path) if db_path else None
        self.event_log_writer = event_log_writer
        self.event_queue = event_queue
        self.max_episodes_per_patient = max(1, int(max_episodes_per_patient))
        self._lock = threading.RLock()

        # Bounded per-patient rolling history: patient_id -> deque of EpisodeSummary
        self._patient_history: dict[int, collections.deque[EpisodeSummary]] = collections.defaultdict(
            lambda: collections.deque(maxlen=self.max_episodes_per_patient)
        )
        # Fast lookup by event_id: event_id -> EpisodeSummary
        self._episodes_by_event_id: dict[str, EpisodeSummary] = {}
        # Fast lookup by action_id: action_id -> EpisodeSummary
        self._episodes_by_action_id: dict[str, EpisodeSummary] = {}

        # Reconstruct state from SQLite event log if available
        if self.db_path:
            self._ensure_db_schema(self.db_path)
            if Path(self.db_path).exists():
                self._init_from_db(self.db_path)

    def _ensure_db_schema(self, db_path: str) -> None:
        """Ensure append-only events table exists in SQLite database."""
        try:
            conn = sqlite3.connect(db_path, timeout=5.0)
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        topic TEXT NOT NULL,
                        event_type TEXT,
                        patient_id INTEGER,
                        timestamp REAL NOT NULL,
                        payload_json TEXT NOT NULL
                    );
                    """
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_events_patient_id ON events(patient_id);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_events_topic ON events(topic);")
            conn.close()
        except Exception as exc:
            logger.warning("Could not ensure schema on %s: %s", db_path, exc)

    def _init_from_db(self, db_path: str) -> None:
        """Replay relevant historical events from the existing SQLite log."""
        try:
            conn = sqlite3.connect(db_path, timeout=5.0)
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT topic, event_type, patient_id, timestamp, payload_json
                FROM events
                WHERE topic IN ('clinical_decisions', 'clinical_reasoning_outputs', 'care_coordination_events', 'clinician_feedback')
                ORDER BY id ASC
                """
            )
            rows = cursor.fetchall()
            conn.close()

            for topic, ev_type, pid, ts, payload_str in rows:
                if pid is None:
                    continue
                try:
                    payload = json.loads(payload_str)
                except Exception:
                    continue

                if topic in ("clinical_decisions", "clinical_reasoning_outputs"):
                    self._record_reasoning_dict(pid, payload, ts)
                elif topic == "care_coordination_events":
                    self._record_coordination_dict(pid, payload, ts)
                elif topic == "clinician_feedback":
                    self._apply_feedback_dict(pid, payload)

        except Exception as exc:
            logger.warning("Could not replay PatientMemory from SQLite (%s): %s", db_path, exc)

    def _record_reasoning_dict(self, patient_id: int, payload: dict[str, Any], timestamp: float) -> None:
        event_id = str(payload.get("event_id", ""))
        if not event_id:
            return
        with self._lock:
            existing = self._episodes_by_event_id.get(event_id)
            if existing is None:
                ep = EpisodeSummary(
                    patient_id=patient_id,
                    event_id=event_id,
                    timestamp=float(payload.get("timestamp", timestamp)),
                    event_type="clinical_reasoning",
                    priority=str(payload.get("priority", "ROUTINE")),
                    clinical_summary=str(payload.get("clinical_summary", "")),
                    findings=list(payload.get("findings", [])),
                    recommended_actions=list(payload.get("recommended_actions", [])),
                    metadata=dict(payload.get("metadata", {})),
                )
                self._patient_history[patient_id].append(ep)
                self._episodes_by_event_id[event_id] = ep
            else:
                existing.priority = str(payload.get("priority", existing.priority))
                existing.clinical_summary = str(payload.get("clinical_summary", existing.clinical_summary))
                existing.findings = list(payload.get("findings", existing.findings))
                existing.recommended_actions = list(payload.get("recommended_actions", existing.recommended_actions))

    def _record_coordination_dict(self, patient_id: int, payload: dict[str, Any], timestamp: float) -> None:
        event_id = str(payload.get("event_id", ""))
        action_id = str(payload.get("action_id", ""))
        with self._lock:
            existing = self._episodes_by_event_id.get(event_id)
            if existing is not None:
                existing.action_id = action_id or existing.action_id
                existing.action_type = str(payload.get("action_type", existing.action_type or ""))
                existing.correlation_id = str(payload.get("correlation_id", existing.correlation_id or ""))
                if action_id:
                    self._episodes_by_action_id[action_id] = existing
            else:
                ep = EpisodeSummary(
                    patient_id=patient_id,
                    event_id=event_id or f"action_{action_id}",
                    timestamp=float(payload.get("timestamp", timestamp)),
                    event_type="care_coordination",
                    priority=str(payload.get("priority", "ROUTINE")),
                    clinical_summary=str(payload.get("clinical_summary", "")),
                    action_type=str(payload.get("action_type", "")),
                    action_id=action_id,
                    correlation_id=str(payload.get("correlation_id", "")),
                    recommended_actions=list(payload.get("suggested_orders", [])),
                    metadata=dict(payload.get("metadata", {})),
                )
                self._patient_history[patient_id].append(ep)
                if event_id:
                    self._episodes_by_event_id[event_id] = ep
                if action_id:
                    self._episodes_by_action_id[action_id] = ep

    def _apply_feedback_dict(self, patient_id: int, payload: dict[str, Any]) -> None:
        event_id = str(payload.get("event_id", ""))
        action_id = str(payload.get("care_action_id", payload.get("action_id", "")))
        action = str(payload.get("action", "")).upper()
        clinician_id = str(payload.get("clinician_id", "clinician_on_duty"))
        reason = str(payload.get("reason", ""))
        fb_time = float(payload.get("timestamp", time.time()))
        is_fp = bool(payload.get("is_false_positive", False)) or "false positive" in reason.lower()
        is_ov = bool(payload.get("is_override", False)) or "override" in reason.lower()

        with self._lock:
            ep = self._episodes_by_event_id.get(event_id)
            if not ep and action_id:
                ep = self._episodes_by_action_id.get(action_id)

            if ep is None:
                ep = EpisodeSummary(
                    patient_id=patient_id,
                    event_id=event_id or (f"action_{action_id}" if action_id else f"fb_{fb_time}"),
                    timestamp=fb_time,
                    event_type="clinician_feedback",
                    priority=str(payload.get("priority", "ROUTINE")),
                    action_id=action_id,
                )
                self._patient_history[patient_id].append(ep)
                if event_id:
                    self._episodes_by_event_id[event_id] = ep
                if action_id:
                    self._episodes_by_action_id[action_id] = ep

            if action == "ACKNOWLEDGE":
                ep.acknowledged = True
                ep.acknowledged_at = fb_time
                ep.acknowledged_by = clinician_id
                ack_lat = payload.get("time_to_acknowledge_seconds")
                if ack_lat is not None:
                    ep.time_to_acknowledge_seconds = float(ack_lat)
                else:
                    ep.time_to_acknowledge_seconds = max(0.0, fb_time - ep.timestamp)
                if ep.status == "PENDING":
                    ep.status = "ACKNOWLEDGED"
            elif action == "RESOLVE":
                ep.resolved = True
                ep.resolved_at = fb_time
                ep.resolution_reason = reason
                res_lat = payload.get("time_to_resolve_seconds")
                if res_lat is not None:
                    ep.time_to_resolve_seconds = float(res_lat)
                else:
                    ep.time_to_resolve_seconds = max(0.0, fb_time - ep.timestamp)
                ep.is_false_positive = is_fp
                ep.is_override = is_ov
                ep.status = "FALSE_POSITIVE" if is_fp else ("OVERRIDDEN" if is_ov else "RESOLVED")

    # ------------------------------------------------------------------------
    # Public Event Ingestion
    # ------------------------------------------------------------------------
    def record_clinical_reasoning(self, event: ClinicalReasoningEvent) -> None:
        """Record ClinicalReasoningEvent into patient's rolling episodic history."""
        case_id = int(getattr(event, "case_id", 0))
        event_id = str(getattr(event, "event_id", ""))
        with self._lock:
            existing = self._episodes_by_event_id.get(event_id)
            if existing is not None:
                existing.priority = event.priority
                existing.clinical_summary = event.clinical_summary
                existing.findings = list(event.findings)
                existing.recommended_actions = list(event.recommended_actions)
                existing.metadata = dict(event.metadata)
            else:
                ep = EpisodeSummary(
                    patient_id=case_id,
                    event_id=event_id,
                    timestamp=float(event.timestamp),
                    event_type="clinical_reasoning",
                    priority=event.priority,
                    clinical_summary=event.clinical_summary,
                    findings=list(event.findings),
                    recommended_actions=list(event.recommended_actions),
                    metadata=dict(event.metadata),
                )
                self._patient_history[case_id].append(ep)
                self._episodes_by_event_id[event_id] = ep

        # Standalone direct SQLite persistence fallback
        if self.event_queue is None and self.event_log_writer is None and self.db_path:
            payload = event.to_dict() if hasattr(event, "to_dict") else asdict(event)
            self._persist_event_to_db("clinical_decisions", "clinical_reasoning", case_id, float(event.timestamp), payload)

    def record_care_coordination(self, event: CareCoordinationEvent) -> None:
        """Record CareCoordinationEvent into patient's rolling episodic history."""
        patient_id = int(getattr(event, "patient_id", 0))
        event_id = str(getattr(event, "event_id", ""))
        action_id = str(getattr(event, "action_id", ""))
        with self._lock:
            existing = self._episodes_by_event_id.get(event_id)
            if existing is not None:
                existing.action_id = action_id
                existing.action_type = event.action_type
                existing.correlation_id = event.correlation_id
                if action_id:
                    self._episodes_by_action_id[action_id] = existing
            else:
                ep = EpisodeSummary(
                    patient_id=patient_id,
                    event_id=event_id,
                    timestamp=float(event.timestamp),
                    event_type="care_coordination",
                    priority=event.priority,
                    clinical_summary=event.clinical_summary,
                    action_type=event.action_type,
                    action_id=action_id,
                    correlation_id=event.correlation_id,
                    recommended_actions=list(event.suggested_orders),
                    metadata=dict(event.metadata),
                )
                self._patient_history[patient_id].append(ep)
                if event_id:
                    self._episodes_by_event_id[event_id] = ep
                if action_id:
                    self._episodes_by_action_id[action_id] = ep

        # Standalone direct SQLite persistence fallback
        if self.event_queue is None and self.event_log_writer is None and self.db_path:
            payload = event.to_dict() if hasattr(event, "to_dict") else asdict(event)
            self._persist_event_to_db("care_coordination_events", "care_coordination", patient_id, float(event.timestamp), payload)

    def _persist_event_to_db(self, topic: str, event_type: str, patient_id: int, timestamp: float, payload: dict[str, Any]) -> None:
        """Direct SQLite append fallback when running without event queue / writer daemon."""
        if self.db_path:
            try:
                conn = sqlite3.connect(self.db_path, timeout=5.0)
                with conn:
                    conn.execute(
                        """
                        INSERT INTO events (topic, event_type, patient_id, timestamp, payload_json)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (topic, event_type, patient_id, timestamp, json.dumps(payload)),
                    )
                conn.close()
            except Exception as exc:
                logger.error("Could not write event to SQLite: %s", exc)

    def record_clinician_feedback(
        self,
        patient_id: int,
        event_id: str,
        action: str,
        clinician_id: str = "clinician_on_duty",
        reason: str = "",
        care_action_id: str | None = None,
        priority: str | None = None,
        is_false_positive: bool = False,
        is_override: bool = False,
        timestamp: float | None = None,
    ) -> EpisodeSummary:
        """Record clinician acknowledgment, resolution, override, or false positive flag."""
        now = float(timestamp if timestamp is not None else time.time())
        action_upper = action.strip().upper()
        is_fp = is_false_positive or ("false positive" in reason.lower())
        is_ov = is_override or ("override" in reason.lower())

        with self._lock:
            ep = self._episodes_by_event_id.get(event_id)
            if not ep and care_action_id:
                ep = self._episodes_by_action_id.get(care_action_id)

            if ep is None:
                # If originating event wasn't in memory, create a synthetic anchor episode
                ep = EpisodeSummary(
                    patient_id=patient_id,
                    event_id=event_id,
                    timestamp=now,
                    event_type="clinician_feedback",
                    priority=priority or "ROUTINE",
                    action_id=care_action_id,
                )
                self._patient_history[patient_id].append(ep)
                if event_id:
                    self._episodes_by_event_id[event_id] = ep
                if care_action_id:
                    self._episodes_by_action_id[care_action_id] = ep

            if action_upper == "ACKNOWLEDGE":
                ep.acknowledged = True
                ep.acknowledged_at = now
                ep.acknowledged_by = clinician_id
                ep.time_to_acknowledge_seconds = max(0.0, now - ep.timestamp)
                if ep.status == "PENDING":
                    ep.status = "ACKNOWLEDGED"
            elif action_upper == "RESOLVE":
                ep.resolved = True
                ep.resolved_at = now
                ep.resolution_reason = reason
                ep.time_to_resolve_seconds = max(0.0, now - ep.timestamp)
                ep.is_false_positive = is_fp
                ep.is_override = is_ov
                ep.status = "FALSE_POSITIVE" if is_fp else ("OVERRIDDEN" if is_ov else "RESOLVED")

        # Persist feedback back into Phase 1 SQLite log
        self._persist_feedback(
            patient_id=patient_id,
            event_id=event_id,
            action=action_upper,
            clinician_id=clinician_id,
            reason=reason,
            care_action_id=care_action_id,
            priority=ep.priority,
            timestamp=now,
            is_false_positive=is_fp,
            is_override=is_ov,
            time_to_acknowledge=ep.time_to_acknowledge_seconds,
            time_to_resolve=ep.time_to_resolve_seconds,
        )

        return ep

    def _persist_feedback(
        self,
        patient_id: int,
        event_id: str,
        action: str,
        clinician_id: str,
        reason: str,
        care_action_id: str | None,
        priority: str,
        timestamp: float,
        is_false_positive: bool,
        is_override: bool,
        time_to_acknowledge: float | None,
        time_to_resolve: float | None,
    ) -> None:
        """Persist clinician feedback into the SQLite event log."""
        payload = {
            "patient_id": patient_id,
            "event_id": event_id,
            "action": action,
            "clinician_id": clinician_id,
            "reason": reason,
            "care_action_id": care_action_id,
            "priority": priority,
            "timestamp": timestamp,
            "is_false_positive": is_false_positive,
            "is_override": is_override,
            "time_to_acknowledge_seconds": time_to_acknowledge,
            "time_to_resolve_seconds": time_to_resolve,
        }

        # 1. Forward to EventQueue if available
        if self.event_queue is not None:
            self.event_queue.publish("clinician_feedback", payload)

        # 2. Forward to EventLogWriter if available
        elif self.event_log_writer is not None:
            self.event_log_writer.log_event("clinician_feedback", payload)

        # 3. Direct SQLite append fallback
        elif self.db_path:
            try:
                conn = sqlite3.connect(self.db_path, timeout=5.0)
                with conn:
                    conn.execute(
                        """
                        INSERT INTO events (topic, event_type, patient_id, timestamp, payload_json)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            "clinician_feedback",
                            f"clinician_{action.lower()}",
                            patient_id,
                            timestamp,
                            json.dumps(payload),
                        ),
                    )
                conn.close()
            except Exception as exc:
                logger.error("Could not write clinician feedback directly to SQLite: %s", exc)

    def record_clinical_reasoning_dict(self, payload: dict[str, Any], patient_id: int | None = None, timestamp: float | None = None) -> None:
        """Record reasoning payload dictionary into patient episodic history."""
        pid = int(patient_id if patient_id is not None else payload.get("patient_id", payload.get("case_id", 0)))
        ts = float(timestamp if timestamp is not None else payload.get("timestamp", time.time()))
        self._record_reasoning_dict(pid, payload, ts)

    def record_care_coordination_dict(self, payload: dict[str, Any], patient_id: int | None = None, timestamp: float | None = None) -> None:
        """Record coordination payload dictionary into patient episodic history."""
        pid = int(patient_id if patient_id is not None else payload.get("patient_id", payload.get("case_id", 0)))
        ts = float(timestamp if timestamp is not None else payload.get("timestamp", time.time()))
        self._record_coordination_dict(pid, payload, ts)

    # ------------------------------------------------------------------------
    # Memory Retrieval & Prompt Context Generation
    # ------------------------------------------------------------------------
    def get_all_patient_ids(self) -> list[int]:
        """Return list of all patient IDs tracked in memory."""
        with self._lock:
            return sorted(list(self._patient_history.keys()))

    def get_episodes(self, patient_id: int, limit: int | None = None) -> list[EpisodeSummary]:
        """Return rolling history of episodes for a patient in chronological order."""
        with self._lock:
            episodes = list(self._patient_history.get(patient_id, []))
            if limit is not None:
                return episodes[-limit:]
            return episodes

    def get_all_episodes(self) -> list[EpisodeSummary]:
        """Return all tracked episodes across all patients."""
        with self._lock:
            all_eps = []
            for eps in self._patient_history.values():
                all_eps.extend(eps)
            all_eps.sort(key=lambda e: e.timestamp)
            return all_eps

    def get_recent_summaries(self, patient_id: int, limit: int = 5) -> list[dict[str, Any]]:
        """Return serializable summary list for evidence package inclusion."""
        episodes = self.get_episodes(patient_id, limit=limit)
        return [e.to_dict() for e in episodes]

    def format_memory_for_prompt(self, patient_id: int, limit: int = 5) -> str:
        """Format episodic history into grounded clinical context for the LLM prompt.

        Safety constraint: This output is explicitly labeled as historical context only,
        instructing the reasoner never to suppress or downgrade fresh deterioration.
        """
        episodes = self.get_episodes(patient_id, limit=limit)
        if not episodes:
            return "No prior alert or clinician feedback history for this patient."

        lines: list[str] = []
        ack_times: list[float] = []
        fp_count = 0
        override_count = 0

        for ep in episodes:
            if ep.acknowledged and ep.time_to_acknowledge_seconds is not None:
                ack_times.append(ep.time_to_acknowledge_seconds)
            if ep.is_false_positive:
                fp_count += 1
            if ep.is_override:
                override_count += 1

        # High-level aggregate context
        if ack_times:
            avg_ack_min = (sum(ack_times) / len(ack_times)) / 60.0
            lines.append(
                f"- Past response timing: This patient's last {len(ack_times)} escalations were acknowledged by clinicians within {avg_ack_min:.1f} minutes on average."
            )
        if fp_count > 0:
            lines.append(
                f"- Clinical discernment: {fp_count} prior alert(s) for this patient were later identified by clinicians as false positives / sensor artifacts."
            )
        if override_count > 0:
            lines.append(
                f"- Clinician overrides: {override_count} prior escalation(s) were manually overridden by bedside clinicians."
            )

        # Recent individual episodes
        lines.append("- Recent episode log:")
        for ep in reversed(episodes[-limit:]):
            t_str = time.strftime("%H:%M:%S", time.localtime(ep.timestamp)) if ep.timestamp > 100000 else f"{ep.timestamp:.0f}s"
            status_desc = ep.status
            if ep.acknowledged and ep.time_to_acknowledge_seconds is not None:
                status_desc += f" (ack in {ep.time_to_acknowledge_seconds:.0f}s by {ep.acknowledged_by or 'clinician'})"
            if ep.resolved:
                status_desc += f" -> Resolved: '{ep.resolution_reason or 'done'}'"
            lines.append(
                f"  * [{t_str}] Event {ep.event_id} ({ep.priority}): {ep.clinical_summary or ep.action_type or 'Alert active'} -> Status: {status_desc}"
            )

        lines.append(
            "SAFETY MANDATE: Historical feedback is provided for clinical context only. "
            "A fresh physiological deterioration must NEVER be suppressed or downgraded based on past false positives."
        )

        return "\n".join(lines)

    def clear(self) -> None:
        """Clear in-memory state."""
        with self._lock:
            self._patient_history.clear()
            self._episodes_by_event_id.clear()
            self._episodes_by_action_id.clear()


__all__ = [
    "EpisodeSummary",
    "PatientMemory",
]
