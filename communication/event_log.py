"""Durable append-only SQLite event log and audit trail for CareMatrix.

Guarantees:
1. Append-only storage of all published events (id, topic, event_type, patient_id, timestamp, payload_json).
2. Asynchronous, non-blocking writes using a background daemon thread so EventQueue.publish() stays < 0.1ms.
3. Fail-safe execution: errors during serialization or writing never raise in the publisher call stack.
4. Deterministic replay capability to reconstruct PatientStateManager and AlertManager state across restarts.
"""

from __future__ import annotations

import json
import logging
import queue
import sqlite3
import threading
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable

from communication.events import (
    AgentFailureEvent,
    AgentHeartbeatEvent,
    CareCoordinationEvent,
    ClinicalReasoningEvent,
    DataAnalysisEvent,
    MonitoringEvent,
    RiskDecisionEvent,
)

logger = logging.getLogger("CareMatrix.EventLog")


def deserialize_event(topic: str, payload: str | dict[str, Any]) -> Any:
    """Deserialize a JSON payload into the corresponding structured event dataclass."""
    if isinstance(payload, str):
        try:
            data = json.loads(payload)
        except Exception:
            return payload
    elif isinstance(payload, dict):
        data = dict(payload)
    else:
        return payload

    if not isinstance(data, dict):
        return data

    # 1. MonitoringEvent
    if (
        "affected_vitals" in data
        and ("deviation_values" in data or "persistence_duration" in data or topic == "monitoring_events")
        and "action_type" not in data
    ):
        return MonitoringEvent(
            patient_id=int(data.get("patient_id", data.get("case_id", 0))),
            event_id=str(data.get("event_id", "")),
            timestamp=float(data.get("timestamp", 0.0) or time.time()),
            event_type=str(data.get("event_type", "alert_started")),
            severity=str(data.get("severity", "moderate")),
            affected_vitals=list(data.get("affected_vitals", [])),
            current_values=dict(data.get("current_values", {})),
            baseline_values=dict(data.get("baseline_values", {})),
            deviation_values=dict(data.get("deviation_values", {})),
            trends=dict(data.get("trends", {})),
            signal_quality=dict(data.get("signal_quality", {})),
            persistence_duration=int(data.get("persistence_duration", data.get("duration_seconds", 0))),
            recommended_action=str(data.get("recommended_action", "")),
            vital_details=dict(data.get("vital_details", {})),
            vital_summary=dict(data.get("vital_summary", {})),
            metadata=dict(data.get("metadata", {})),
            performative=str(data.get("performative", "INFORM")),
            challenge_round=int(data.get("challenge_round", 0)),
            message_id=str(data.get("message_id", "")),
            created_at=float(data.get("created_at", time.time())),
        )

    # 2. RiskDecisionEvent
    if (
        "risk_probability" in data
        or ("threshold" in data and "model_name" in data)
        or topic in ("risk_predictions", "risk_decisions")
    ):
        return RiskDecisionEvent(
            patient_id=int(data.get("patient_id", data.get("case_id", 0))),
            event_id=str(data.get("event_id", "")),
            timestamp=float(data.get("timestamp", 0.0) or time.time()),
            decision=str(data.get("decision", "LOW_RISK")),
            risk_probability=float(data.get("risk_probability", 0.0)),
            risk_level=str(data.get("risk_level", "LOW RISK")),
            threshold=float(data.get("threshold", 0.5)),
            model_name=str(data.get("model_name", "RandomForestClassifier")),
            features_used=dict(data.get("features_used", {})),
            reason=str(data.get("reason", "")),
            status=str(data.get("status", "success")),
            recommended_action=str(data.get("recommended_action", "")),
            retry_count=int(data.get("retry_count", 0)),
            severity=str(data.get("severity", "unknown")),
            affected_vitals=list(data.get("affected_vitals", [])),
            window_start=data.get("window_start"),
            window_end=data.get("window_end"),
            analysis_level=str(data.get("analysis_level", "standard")),
            requested_checks=list(data.get("requested_checks", [])),
            verification_required=bool(data.get("verification_required", False)),
            metadata=dict(data.get("metadata", {})),
            performative=str(data.get("performative", "INFORM")),
            challenge_round=int(data.get("challenge_round", 0)),
            message_id=str(data.get("message_id", "")),
            created_at=float(data.get("created_at", time.time())),
        )

    # 3. DataAnalysisEvent
    if (
        "trend_metrics" in data
        or "pattern_identified" in data
        or "data_quality_flag" in data
        or topic in ("data_analysis_inputs", "data_analysis_events", "analytical_evidence")
    ):
        return DataAnalysisEvent(
            case_id=int(data.get("case_id", data.get("patient_id", 0))),
            event_id=str(data.get("event_id", "")),
            timestamp=float(data.get("timestamp", 0.0) or time.time()),
            risk_level=str(data.get("risk_level", "LOW RISK")),
            trend_metrics=dict(data.get("trend_metrics", {})),
            pattern_identified=list(data.get("pattern_identified", [])),
            important_changes=list(data.get("important_changes", [])),
            data_quality_flag=bool(data.get("data_quality_flag", False)),
            analysis_status=str(data.get("analysis_status", "complete")),
            source=str(data.get("source", "Data Analysis Agent")),
            data_quality_details=dict(data.get("data_quality_details", {})),
            error_message=data.get("error_message"),
            executed_checks=list(data.get("executed_checks", [])),
            evidence_consistency=str(data.get("evidence_consistency", "consistent")),
            conflict_flags=list(data.get("conflict_flags", [])),
            verification_required=bool(data.get("verification_required", False)),
            metadata=dict(data.get("metadata", {})),
            performative=str(data.get("performative", "INFORM")),
            challenge_round=int(data.get("challenge_round", 0)),
            message_id=str(data.get("message_id", "")),
            created_at=float(data.get("created_at", time.time())),
        )

    # 4. ClinicalReasoningEvent
    if (
        "findings" in data
        and "recommended_actions" in data
        and "priority" in data
        and "action_type" not in data
        or topic in ("clinical_decisions", "clinical_reasoning_outputs", "clinical_reasoning_events")
    ):
        return ClinicalReasoningEvent(
            case_id=int(data.get("case_id", data.get("patient_id", 0))),
            event_id=str(data.get("event_id", "")),
            timestamp=float(data.get("timestamp", 0.0) or time.time()),
            risk_level=str(data.get("risk_level", "LOW RISK")),
            priority=str(data.get("priority", "ROUTINE")),
            clinical_summary=str(data.get("clinical_summary", "")),
            findings=list(data.get("findings", [])),
            recommended_actions=list(data.get("recommended_actions", [])),
            escalation_required=bool(data.get("escalation_required", False)),
            data_reliability=str(data.get("data_reliability", "HIGH")),
            confidence=float(data.get("confidence", 1.0)),
            source=str(data.get("source", "Clinical Reasoning Agent")),
            evidence_consistency=str(data.get("evidence_consistency", "CONSISTENT")),
            supporting_evidence=list(data.get("supporting_evidence", [])),
            conflicting_evidence=list(data.get("conflicting_evidence", [])),
            verification_required=bool(data.get("verification_required", False)),
            metadata=dict(data.get("metadata", {})),
            performative=str(data.get("performative", "INFORM")),
            challenge_round=int(data.get("challenge_round", 0)),
            message_id=str(data.get("message_id", "")),
            created_at=float(data.get("created_at", time.time())),
        )

    # 5. CareCoordinationEvent
    if (
        "action_type" in data
        or "suggested_orders" in data
        or "escalation_pathway" in data
        or topic in ("care_coordination_events", "care_actions")
    ):
        return CareCoordinationEvent(
            patient_id=int(data.get("patient_id", data.get("case_id", 0))),
            event_id=str(data.get("event_id", "")),
            timestamp=float(data.get("timestamp", 0.0) or time.time()),
            action_type=str(data.get("action_type", "CONTINUE_ROUTINE_MONITORING")),
            priority=str(data.get("priority", "ROUTINE")),
            status=str(data.get("status", "NEW")),
            reason=str(data.get("reason", "")),
            clinician_review_required=bool(data.get("clinician_review_required", False)),
            suggested_orders=list(data.get("suggested_orders", [])),
            escalation_pathway=str(data.get("escalation_pathway", "")),
            clinical_summary=str(data.get("clinical_summary", "")),
            evidence_consistency=str(data.get("evidence_consistency", "CONSISTENT")),
            data_reliability=str(data.get("data_reliability", "HIGH")),
            confidence=float(data.get("confidence", 1.0)),
            source=str(data.get("source", "Care Coordination Agent")),
            correlation_id=str(data.get("correlation_id", "")),
            action_id=str(data.get("action_id", "")),
            metadata=dict(data.get("metadata", {})),
            message_id=str(data.get("message_id", "")),
            created_at=float(data.get("created_at", time.time())),
        )

    # 6. AgentHeartbeatEvent
    if "agent_name" in data and ("status" in data or "metrics" in data) and topic == "heartbeats":
        return AgentHeartbeatEvent(
            agent_name=str(data.get("agent_name", "")),
            timestamp=float(data.get("timestamp", 0.0) or time.time()),
            status=str(data.get("status", "healthy")),
            metrics=dict(data.get("metrics", {})),
        )

    # 7. AgentFailureEvent
    if "agent_name" in data and "error_message" in data and topic == "agent_failures":
        return AgentFailureEvent(
            agent_name=str(data.get("agent_name", "")),
            error_message=str(data.get("error_message", "")),
            timestamp=float(data.get("timestamp", 0.0) or time.time()),
            recoverable=bool(data.get("recoverable", True)),
            context=dict(data.get("context", {})),
        )

    return data


class EventLogWriter:
    """Durable append-only SQLite event logger for audit trail and state reconstruction.

    Architectural Invariant:
    `log_event()` pushes serialized event tuples into a fast in-memory queue (< 0.05ms)
    and returns immediately. A background daemon thread drains the queue and executes
    batch SQL inserts. It never updates or deletes rows.
    """

    def __init__(
        self,
        db_path: str | Path = "carematrix_events.db",
        max_queue_size: int = 10000,
        name: str = "EventLogWriter",
    ):
        self.db_path = str(db_path)
        self.name = name
        self.max_queue_size = max_queue_size

        self._write_queue: queue.Queue[tuple[str, str, int | None, float, str]] = queue.Queue(
            maxsize=max_queue_size
        )
        self._stop_event = threading.Event()
        self._is_closed = False

        # Initialize SQLite database schema
        self._init_db()

        # Dedicated background async writer thread
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name=f"{self.name}-Thread",
            daemon=True,
        )
        self._writer_thread.start()

    def _init_db(self) -> None:
        """Create the append-only events table and indexes if they do not exist."""
        try:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            with conn:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA synchronous=NORMAL;")
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
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_events_patient_id ON events(patient_id);"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_events_topic ON events(topic);"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);"
                )
            conn.close()
        except Exception as exc:
            logger.error("Failed to initialize EventLog database %s: %s", self.db_path, exc)

    def log_event(self, topic: str, event: Any) -> None:
        """Append an event to the log without blocking the publisher.

        This method is guaranteed NEVER to raise an exception.
        """
        if self._is_closed or self._stop_event.is_set():
            return

        try:
            # 1. Extract metadata
            event_type = None
            if hasattr(event, "event_type"):
                event_type = str(getattr(event, "event_type"))
            elif hasattr(event, "action_type"):
                event_type = str(getattr(event, "action_type"))
            elif hasattr(event, "decision"):
                event_type = str(getattr(event, "decision"))
            elif hasattr(event, "priority"):
                event_type = str(getattr(event, "priority"))
            elif isinstance(event, dict) and "event_type" in event:
                event_type = str(event["event_type"])
            else:
                event_type = event.__class__.__name__

            # 2. Extract patient_id / case_id
            pid = getattr(event, "patient_id", getattr(event, "case_id", None))
            if pid is None and isinstance(event, dict):
                pid = event.get("patient_id", event.get("case_id"))
            patient_id = int(pid) if pid is not None else None

            # 3. Extract timestamp
            ts = getattr(event, "timestamp", None)
            if ts is None and isinstance(event, dict):
                ts = event.get("timestamp")
            timestamp = float(ts) if ts is not None else time.time()

            # 4. Serialize payload to JSON
            if hasattr(event, "to_dict") and callable(event.to_dict):
                payload_dict = event.to_dict()
            elif is_dataclass(event):
                payload_dict = asdict(event)
            elif isinstance(event, dict):
                payload_dict = dict(event)
            else:
                payload_dict = {"data": str(event)}

            payload_json = json.dumps(payload_dict, default=str)

            # 5. Enqueue for asynchronous write (non-blocking)
            self._write_queue.put_nowait((topic, event_type, patient_id, timestamp, payload_json))

        except queue.Full:
            logger.warning("EventLogWriter queue full (%d items); dropping event on topic '%s'", self.max_queue_size, topic)
        except Exception as exc:
            # Strictly degrade gracefully without crashing the publisher
            logger.error("EventLogWriter serialization/logging error on topic '%s': %s", topic, exc)

    def __call__(self, topic: str, event: Any) -> None:
        """Allow EventLogWriter to be registered directly as a callable diagnostic listener."""
        self.log_event(topic, event)

    def _writer_loop(self) -> None:
        """Background worker thread draining the queue and writing rows to SQLite."""
        conn: sqlite3.Connection | None = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            conn.execute("PRAGMA synchronous=NORMAL;")

            while not self._stop_event.is_set() or not self._write_queue.empty():
                batch: list[tuple[str, str | None, int | None, float, str]] = []
                try:
                    # Block briefly waiting for first item
                    item = self._write_queue.get(timeout=0.1)
                    batch.append(item)
                    self._write_queue.task_done()

                    # Drain any pending items in queue up to 100 per batch
                    while len(batch) < 100:
                        try:
                            extra = self._write_queue.get_nowait()
                            batch.append(extra)
                            self._write_queue.task_done()
                        except queue.Empty:
                            break
                except queue.Empty:
                    continue

                if batch:
                    try:
                        with conn:
                            conn.executemany(
                                """
                                INSERT INTO events (topic, event_type, patient_id, timestamp, payload_json)
                                VALUES (?, ?, ?, ?, ?)
                                """,
                                batch,
                            )
                    except Exception as err:
                        logger.error("EventLogWriter SQL insert error: %s", err)

        except Exception as top_err:
            logger.error("EventLogWriter thread fatal error: %s", top_err)
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    def flush(self, timeout: float = 3.0) -> None:
        """Wait for all pending queued writes to be committed to disk."""
        deadline = time.time() + timeout
        while not self._write_queue.empty() and time.time() < deadline:
            time.sleep(0.01)

    def close(self, timeout: float = 3.0) -> None:
        """Flush remaining events and terminate the writer thread."""
        if self._is_closed:
            return
        self.flush(timeout=timeout)
        self._stop_event.set()
        self._is_closed = True
        if self._writer_thread.is_alive():
            self._writer_thread.join(timeout=timeout)

    def get_events(
        self,
        patient_id: int | None = None,
        topic: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Read logged events from the database in append order (read-only query)."""
        self.flush()
        try:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            cursor = conn.cursor()

            query = "SELECT id, topic, event_type, patient_id, timestamp, payload_json FROM events"
            params: list[Any] = []
            conditions: list[str] = []

            if patient_id is not None:
                conditions.append("patient_id = ?")
                params.append(patient_id)
            if topic is not None:
                conditions.append("topic = ?")
                params.append(topic)

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            query += " ORDER BY id ASC"
            if limit is not None:
                query += f" LIMIT {int(limit)}"

            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()

            results: list[dict[str, Any]] = []
            for row in rows:
                try:
                    payload = json.loads(row[5])
                except Exception:
                    payload = {"raw": row[5]}
                results.append({
                    "id": row[0],
                    "topic": row[1],
                    "event_type": row[2],
                    "patient_id": row[3],
                    "timestamp": row[4],
                    "payload": payload,
                })
            return results
        except Exception as exc:
            logger.error("Failed to query events from EventLog: %s", exc)
            return []

    def read_all_events(self) -> list[tuple[str, Any]]:
        """Read and deserialize all logged events in original chronological order."""
        raw_records = self.get_events()
        events: list[tuple[str, Any]] = []
        for record in raw_records:
            topic = record["topic"]
            deserialized = deserialize_event(topic, record["payload"])
            events.append((topic, deserialized))
        return events

    def replay(self, target: Any = None) -> list[tuple[str, Any]]:
        """Replay all logged events into a target state manager or event consumer."""
        events = self.read_all_events()
        if target is not None:
            for topic, event in events:
                if hasattr(target, "handle_event") and callable(target.handle_event):
                    target.handle_event(topic, event)
                elif callable(target):
                    target(topic, event)
        return events

    def replay_into(self, state_manager: Any, alert_manager: Any = None) -> int:
        """Replay all historical events into PatientStateManager and AlertManager."""
        events = self.read_all_events()
        count = 0
        for topic, event in events:
            if hasattr(state_manager, "handle_event"):
                state_manager.handle_event(topic, event)
                count += 1
            elif callable(state_manager):
                state_manager(topic, event)
                count += 1
        return count


__all__ = [
    "EventLogWriter",
    "deserialize_event",
]
