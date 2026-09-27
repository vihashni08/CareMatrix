"""Unit tests for CareMatrix Alert Lifecycle and deduplication."""

from __future__ import annotations

import time
import unittest

from carematrix_runtime.alert_manager import AlertLifecycleState, AlertManager
from communication.events import CareActionType, CareCoordinationEvent, MonitoringEvent


class TestAlertLifecycle(unittest.TestCase):
    def setUp(self):
        self.mgr = AlertManager()

    def _make_monitoring_event(
        self,
        patient_id: int = 101,
        event_id: str = "alert_001",
        event_type: str = "alert_started",
        severity: str = "severe",
        affected_vitals: list[str] | None = None,
    ) -> MonitoringEvent:
        return MonitoringEvent(
            patient_id=patient_id,
            event_id=event_id,
            timestamp=time.time(),
            event_type=event_type,
            severity=severity,
            affected_vitals=affected_vitals or ["HR", "MAP"],
            current_values={"HR": 125.0, "MAP": 58.0},
            baseline_values={"HR": 72.0, "MAP": 85.0},
            deviation_values={"HR": 53.0, "MAP": -27.0},
            trends={"HR": "increasing", "MAP": "decreasing"},
            signal_quality={"HR": "reliable", "MAP": "reliable"},
            persistence_duration=5,
            recommended_action="Initiate emergency clinical review",
        )

    # 1. New alert transitions from NEW
    def test_new_alert_creation(self):
        ev = self._make_monitoring_event(patient_id=101, event_id="ev_101")
        alert = self.mgr.handle_monitoring_event(ev)

        self.assertIsNotNone(alert)
        self.assertEqual(alert.patient_id, 101)
        self.assertEqual(alert.state, AlertLifecycleState.NEW)
        self.assertEqual(alert.severity, "severe")
        self.assertEqual(alert.priority, "URGENT")
        self.assertEqual(len(self.mgr.get_active_alerts()), 1)

    # 2. Duplicate active alert updates duration without creating duplicate alarm
    def test_duplicate_active_alert_deduplication(self):
        ev1 = self._make_monitoring_event(event_id="ev_001")
        alert1 = self.mgr.handle_monitoring_event(ev1)
        initial_id = alert1.alert_id

        # Second event for same patient and same vitals while first is still active
        ev2 = self._make_monitoring_event(event_id="ev_002")
        alert2 = self.mgr.handle_monitoring_event(ev2)

        self.assertEqual(alert2.alert_id, initial_id)
        self.assertEqual(alert2.state, AlertLifecycleState.ACTIVE)
        # Total active alerts must still be 1 (no spam)
        self.assertEqual(len(self.mgr.get_active_alerts()), 1)

    # 3. Clinician acknowledgment transitions alert to ACKNOWLEDGED
    def test_clinician_acknowledgment(self):
        ev = self._make_monitoring_event()
        alert = self.mgr.handle_monitoring_event(ev)

        ack_alert = self.mgr.acknowledge_alert(alert.alert_id, clinician_id="Dr. House")
        self.assertIsNotNone(ack_alert)
        self.assertEqual(ack_alert.state, AlertLifecycleState.ACKNOWLEDGED)
        self.assertEqual(ack_alert.acknowledged_by, "Dr. House")
        self.assertIsNotNone(ack_alert.acknowledged_at)

    # 4. Recovery event resolves active alerts
    def test_recovery_event_resolves_alerts(self):
        ev_start = self._make_monitoring_event(patient_id=102, event_id="ev_102")
        self.mgr.handle_monitoring_event(ev_start)
        self.assertEqual(len(self.mgr.get_active_alerts(patient_id=102)), 1)

        # Emit recovery event
        ev_rec = self._make_monitoring_event(patient_id=102, event_type="alert_recovered")
        self.mgr.handle_monitoring_event(ev_rec)

        self.assertEqual(len(self.mgr.get_active_alerts(patient_id=102)), 0)
        history = self.mgr.get_alert_history(patient_id=102)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["state"], AlertLifecycleState.RESOLVED.value)


if __name__ == "__main__":
    unittest.main()
