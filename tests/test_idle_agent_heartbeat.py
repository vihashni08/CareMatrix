"""Test idle agent heartbeats for DataAnalysisAgent and ClinicalReasoningAgent.

Verifies that when both agents are started with an active worker thread but no events
are arriving, periodic idle heartbeats prevent the Supervisor from falsely detecting
them as stale or triggering unneeded restarts.
"""

from __future__ import annotations

import time
import unittest

from clinical_reasoning_agent import ClinicalReasoningAgent
from communication.event_queue import EventQueue
from data_analysis_agent import DataAnalysisAgent
from supervisor.supervisor import AgentSupervisor


class TestIdleAgentHeartbeat(unittest.TestCase):
    """Test idle periodic heartbeats in DataAnalysisAgent and ClinicalReasoningAgent."""

    def setUp(self):
        self.queue = EventQueue()
        self.timeout = 1.5
        self.supervisor = AgentSupervisor(
            event_queue=self.queue,
            heartbeat_timeout_seconds=self.timeout,
            check_interval_seconds=0.2,
            auto_recover=True,
            max_restart_attempts=3,
            verbose=False,
        )
        self.data_analysis_agent = DataAnalysisAgent(event_queue=self.queue, verbose=False)
        self.clinical_reasoning_agent = ClinicalReasoningAgent(
            event_queue=self.queue,
            enable_llm=False,
            verbose=False,
        )

    def tearDown(self):
        self.data_analysis_agent.stop()
        self.clinical_reasoning_agent.stop()
        self.data_analysis_agent.join(timeout=2.0)
        self.clinical_reasoning_agent.join(timeout=2.0)
        self.queue.shutdown()

    def test_idle_agents_emit_heartbeats_and_remain_healthy(self):
        """Start both agents with no events, wait 2x supervisor timeout, assert healthy and restart_count == 0."""
        self.supervisor.register_agent(self.data_analysis_agent)
        self.supervisor.register_agent(self.clinical_reasoning_agent)

        # Start both worker threads with no incoming events on the queue
        self.data_analysis_agent.start()
        self.clinical_reasoning_agent.start()

        # Wait 2x the supervisor timeout
        wait_seconds = 2 * self.timeout
        time.sleep(wait_seconds)

        # Evaluate health through supervisor
        health_report = self.supervisor.check_health()

        # Assert DataAnalysisAgent is healthy and has 0 restarts
        self.assertIn("DataAnalysisAgent", health_report)
        da_record = health_report["DataAnalysisAgent"]
        self.assertEqual(da_record["status"], "healthy")
        self.assertEqual(da_record["restart_count"], 0)
        self.assertLessEqual(da_record["last_heartbeat_ago_seconds"], self.timeout)

        # Assert ClinicalReasoningAgent is healthy and has 0 restarts
        self.assertIn("ClinicalReasoningAgent", health_report)
        cr_record = health_report["ClinicalReasoningAgent"]
        self.assertEqual(cr_record["status"], "healthy")
        self.assertEqual(cr_record["restart_count"], 0)
        self.assertLessEqual(cr_record["last_heartbeat_ago_seconds"], self.timeout)


if __name__ == "__main__":
    unittest.main()
