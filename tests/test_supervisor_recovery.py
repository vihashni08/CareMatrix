"""Unit and integration tests for Supervisor Heartbeat Monitoring and Auto-Recovery across all 5 agents."""

from __future__ import annotations

import time
import unittest

from care_coordination_agent import CareCoordinationAgent
from clinical_reasoning_agent import ClinicalReasoningAgent
from communication.event_queue import EventQueue
from communication.events import AgentFailureEvent
from data_analysis_agent import DataAnalysisAgent
from monitoring_agent.monitoring_agent import MonitoringAgent
from risk_agent.risk_agent import RiskAgent
from supervisor.supervisor import AgentSupervisor


class TestSupervisorRecovery(unittest.TestCase):
    def setUp(self):
        self.queue = EventQueue()
        self.supervisor = AgentSupervisor(
            event_queue=self.queue,
            heartbeat_timeout_seconds=0.3,
            check_interval_seconds=0.1,
            auto_recover=True,
            verbose=False,
        )
        self.monitoring_agent = MonitoringAgent(case_id=999, event_queue=self.queue, verbose=False)
        self.risk_agent = RiskAgent(event_queue=self.queue, verbose=False)
        self.data_analysis_agent = DataAnalysisAgent(event_queue=self.queue, verbose=False)
        self.clinical_reasoning_agent = ClinicalReasoningAgent(event_queue=self.queue, enable_llm=False, verbose=False)
        self.care_coordination_agent = CareCoordinationAgent(event_queue=self.queue, verbose=False)

    def test_all_five_agents_registered(self):
        """Verify all 5 agents register correctly with supervisor."""
        self.supervisor.register_agent(self.monitoring_agent)
        self.supervisor.register_agent(self.risk_agent)
        self.supervisor.register_agent(self.data_analysis_agent)
        self.supervisor.register_agent(self.clinical_reasoning_agent)
        self.supervisor.register_agent(self.care_coordination_agent)

        health = self.supervisor.get_system_health_snapshot()
        self.assertEqual(health["total_agents"], 5)
        self.assertIn("MonitoringAgent", health["agents"])
        self.assertIn("RiskAgent", health["agents"])
        self.assertIn("DataAnalysisAgent", health["agents"])
        self.assertIn("ClinicalReasoningAgent", health["agents"])
        self.assertIn("CareCoordinationAgent", health["agents"])

    def test_heartbeat_timeout_detection(self):
        """Verify supervisor detects missed heartbeat and triggers recovery."""
        self.supervisor.register_agent(self.monitoring_agent)

        # Make heartbeat stale
        self.supervisor.agent_records["MonitoringAgent"]["last_heartbeat"] = time.time() - 2.0

        # Run supervisor check step
        self.supervisor.step()

        record = self.supervisor.agent_records["MonitoringAgent"]
        self.assertGreater(record["restart_count"], 0)

    def test_agent_failure_event_handling(self):
        """Verify supervisor processes AgentFailureEvent from event queue without crashing."""
        self.supervisor.register_agent(self.risk_agent)

        failure_event = AgentFailureEvent(
            agent_name="RiskAgent",
            error_message="Test unhandled exception",
            recoverable=True,
            context={"step": 1},
        )
        self.queue.publish("agent_failures", failure_event)

        # Allow queue consumption
        self.supervisor.step()

        record = self.supervisor.agent_records["RiskAgent"]
        self.assertGreaterEqual(record["failure_count"], 1)
        self.assertIn("Test unhandled exception", str(record["last_error"]))


if __name__ == "__main__":
    unittest.main()
