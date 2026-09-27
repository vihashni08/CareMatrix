"""Schemas and state enums for the Monitoring Agent."""

from __future__ import annotations

from enum import Enum

from communication.events import (
    AgentHeartbeatEvent,
    MonitoringDecision,
    MonitoringEvent,
)


class AgentLifecycleState(str, Enum):
    """Lifecycle states of the autonomous Monitoring Agent."""

    STARTING = "STARTING"
    RUNNING = "RUNNING"
    DEGRADED = "DEGRADED"
    RECOVERING = "RECOVERING"
    STOPPED = "STOPPED"


__all__ = [
    "AgentLifecycleState",
    "AgentHeartbeatEvent",
    "MonitoringDecision",
    "MonitoringEvent",
]

