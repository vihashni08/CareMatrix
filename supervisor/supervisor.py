"""CareMatrix Lightweight Agent Supervisor running as an independent health monitor loop."""

from __future__ import annotations

import datetime
import logging
import threading
import time
from typing import Any, Callable

from communication.event_queue import EventQueue
from communication.events import AgentFailureEvent, AgentHeartbeatEvent

logger = logging.getLogger("CareMatrix.Supervisor")


def _format_log(name: str, action: str, details: str) -> str:
    now_str = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    return f"[{now_str}] {name:<16} | {action:<10} | {details}"


class AgentSupervisor:
    """Non-LLM independent supervisor running in its own thread managing agent heartbeats, staleness, and fault recovery."""

    def __init__(
        self,
        event_queue: EventQueue | None = None,
        heartbeat_timeout_seconds: float = 3.0,
        check_interval_seconds: float = 0.5,
        max_restart_attempts: int = 3,
        auto_recover: bool = True,
        name: str = "Supervisor",
        verbose: bool = False,
    ):
        self.name = name
        self.event_queue = event_queue
        self.heartbeat_timeout = float(heartbeat_timeout_seconds)
        self.check_interval = float(check_interval_seconds)
        self.max_restart_attempts = int(max_restart_attempts)
        self.auto_recover = auto_recover
        self.verbose = verbose

        self.agents: dict[str, Any] = {}
        self.restart_factories: dict[str, Callable[[dict[str, Any]], Any]] = {}
        self.agent_snapshots: dict[str, dict[str, Any]] = {}
        self.agent_records: dict[str, dict[str, Any]] = {}
        self.failure_log: list[dict[str, Any]] = []

        # Independent monitor thread management
        self._monitor_thread: threading.Thread | None = None
        self._is_running: bool = False
        self._stop_event = threading.Event()

    def register_agent(
        self,
        agent: Any,
        name: str | None = None,
        restart_factory: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        """Register an agent for supervisor health tracking."""
        agent_name = name or getattr(agent, "name", agent.__class__.__name__)
        self.agents[agent_name] = agent
        if restart_factory is not None:
            self.restart_factories[agent_name] = restart_factory

        now = time.time()
        initial_status = "healthy"
        if hasattr(agent, "lifecycle_state"):
            initial_status = str(getattr(agent, "lifecycle_state").value).lower()

        self.agent_records[agent_name] = {
            "name": agent_name,
            "status": initial_status,
            "last_heartbeat": now,
            "restart_count": 0,
            "failure_count": 0,
            "last_error": None,
        }

        if hasattr(agent, "get_state_snapshot"):
            try:
                self.agent_snapshots[agent_name] = agent.get_state_snapshot()
            except Exception:
                pass

        if self.verbose:
            print(_format_log(self.name, "REGISTER", f"Agent '{agent_name}' registered (Timeout: {self.heartbeat_timeout}s)"))

    def record_heartbeat(
        self,
        agent_name: str,
        metrics: dict[str, Any] | None = None,
        status: str | None = None,
    ) -> None:
        """Record a heartbeat timestamp for an agent, respecting reported status."""
        now = time.time()
        if agent_name in self.agent_records:
            record = self.agent_records[agent_name]
            record["last_heartbeat"] = now

            # Respect the heartbeat's actual lifecycle/status!
            # Do not automatically overwrite degraded/recovering/failed to healthy.
            reported_status = status
            if reported_status is None and metrics:
                reported_status = metrics.get("lifecycle_state") or metrics.get("status")
            if reported_status is None:
                reported_status = "healthy"

            record["status"] = str(reported_status).lower()

            if metrics:
                record["metrics"] = metrics

            agent = self.agents.get(agent_name)
            if agent is not None and hasattr(agent, "get_state_snapshot"):
                try:
                    self.agent_snapshots[agent_name] = agent.get_state_snapshot()
                except Exception:
                    pass

    def record_failure(self, agent_name: str, error_message: str, context: dict[str, Any] | None = None) -> None:
        """Record an explicit agent error/failure."""
        now = time.time()
        entry = {
            "timestamp": now,
            "agent_name": agent_name,
            "error": error_message,
            "context": context or {},
        }
        self.failure_log.append(entry)
        if agent_name in self.agent_records:
            self.agent_records[agent_name]["status"] = "degraded"
            self.agent_records[agent_name]["failure_count"] += 1
            self.agent_records[agent_name]["last_error"] = error_message

        if self.verbose:
            print(_format_log(self.name, "FAILURE", f"Agent '{agent_name}' reported error: {error_message}"))

    def check_health(self, current_time: float | None = None) -> dict[str, dict[str, Any]]:
        """Evaluate agent health using heartbeats + thread liveness, detecting stale/dead workers and triggering recovery."""
        # Drain heartbeat and failure events from the queue if present
        if self.event_queue is not None:
            for hb in self.event_queue.get_all("heartbeats"):
                if isinstance(hb, AgentHeartbeatEvent):
                    status = hb.status or (hb.metrics.get("lifecycle_state") if hb.metrics else "healthy")
                    self.record_heartbeat(hb.agent_name, hb.metrics, status=status)
            for fail in self.event_queue.get_all("agent_failures"):
                if isinstance(fail, AgentFailureEvent):
                    self.record_failure(fail.agent_name, fail.error_message, fail.context)

        now = current_time if current_time is not None else time.time()
        report: dict[str, dict[str, Any]] = {}

        for name, record in list(self.agent_records.items()):
            agent = self.agents.get(name)
            time_since_hb = now - record["last_heartbeat"]
            is_stale = time_since_hb > self.heartbeat_timeout

            # Inspect thread liveness
            worker_thread = getattr(agent, "_worker_thread", None) if agent else None
            is_worker_dead = False
            if worker_thread is not None and not worker_thread.is_alive():
                stop_event = getattr(agent, "_stop_event", None)
                is_explicitly_stopped = stop_event.is_set() if stop_event is not None else False
                agent_healthy = getattr(agent, "is_healthy", True)
                agent_lifecycle = getattr(agent, "lifecycle_state", None)
                lifecycle_name = agent_lifecycle.name if hasattr(agent_lifecycle, "name") else str(agent_lifecycle)

                if not is_explicitly_stopped:
                    if not agent_healthy or lifecycle_name in ("DEGRADED", "FAILED") or record["status"] in ("stale", "degraded"):
                        is_worker_dead = True

            needs_recovery = False
            if is_worker_dead:
                record["status"] = "crashed"
                needs_recovery = True
                if self.verbose:
                    print(_format_log(self.name, "CRASH_WARN", f"Agent '{name}' worker thread has died!"))
            elif is_stale:
                record["status"] = "stale"
                needs_recovery = True
                if self.verbose:
                    print(_format_log(self.name, "STALE_WARN", f"Agent '{name}' unresponsive ({time_since_hb:.2f}s > {self.heartbeat_timeout}s)"))

            if needs_recovery and self.auto_recover and record["restart_count"] < self.max_restart_attempts:
                if self.verbose:
                    print(_format_log(self.name, "RECOVER", f"Triggering recovery for '{name}'..."))
                self.restart_agent(name)

            report[name] = {
                "status": record["status"],
                "last_heartbeat_ago_seconds": round(time_since_hb, 2),
                "restart_count": record["restart_count"],
                "failure_count": record["failure_count"],
                "last_error": record["last_error"],
            }

        return report

    def restart_agent(self, agent_name: str) -> bool:
        """Restart a failed/stale agent with real worker recreation and state restoration."""
        if agent_name not in self.agents:
            return False

        agent = self.agents[agent_name]
        record = self.agent_records[agent_name]

        # Preserve latest valid state snapshot before restart
        if hasattr(agent, "get_state_snapshot"):
            try:
                self.agent_snapshots[agent_name] = agent.get_state_snapshot()
            except Exception:
                pass
        snapshot = self.agent_snapshots.get(agent_name)

        worker_thread = getattr(agent, "_worker_thread", None)
        was_worker_launched = worker_thread is not None
        is_thread_alive = worker_thread is not None and worker_thread.is_alive()

        try:
            # Case 1: Custom restart factory registered
            if agent_name in self.restart_factories:
                # Safely clean up old agent
                if hasattr(agent, "stop"):
                    agent.stop()
                factory = self.restart_factories[agent_name]
                new_agent = factory(snapshot or {})
                self.agents[agent_name] = new_agent
                agent = new_agent
                if hasattr(agent, "start"):
                    agent.start()

            # Case 2: Dead/stopped worker thread -> Recreate and start a NEW worker thread
            elif was_worker_launched and not is_thread_alive and hasattr(agent, "restart_worker"):
                restarted_live = agent.restart_worker(snapshot)
                if not restarted_live:
                    raise RuntimeError(f"New worker thread for '{agent_name}' failed to start.")

            # Case 3: Deliberately hung worker that is still alive -> release and recover it
            elif is_thread_alive and hasattr(agent, "is_healthy"):
                if hasattr(agent, "restore_state_snapshot") and snapshot:
                    agent.restore_state_snapshot(snapshot)
                agent.is_healthy = True
                if hasattr(agent, "lifecycle_state"):
                    from monitoring_agent.monitoring_agent import AgentLifecycleState
                    agent.lifecycle_state = AgentLifecycleState.RUNNING

            # Case 4: Non-threaded agent or manual step mode that became stale/degraded
            else:
                if hasattr(agent, "reset"):
                    agent.reset()
                if hasattr(agent, "restore_state_snapshot") and snapshot:
                    agent.restore_state_snapshot(snapshot)
                if hasattr(agent, "is_healthy"):
                    agent.is_healthy = True
                if hasattr(agent, "lifecycle_state"):
                    from monitoring_agent.monitoring_agent import AgentLifecycleState
                    agent.lifecycle_state = AgentLifecycleState.RUNNING

            # Verify that heartbeats resume and status is healthy
            now = time.time()
            record["status"] = "healthy"
            record["last_heartbeat"] = now
            record["restart_count"] += 1
            record["last_error"] = None

            if self.verbose:
                print(_format_log(self.name, "RESTORED", f"Agent '{agent_name}' restarted successfully (Attempt {record['restart_count']}). New worker live & state restored."))
            return True

        except Exception as exc:
            record["status"] = "failed"
            record["last_error"] = f"Restart failed: {exc}"
            if self.verbose:
                print(_format_log(self.name, "CRITICAL", f"Failed to restart agent '{agent_name}': {exc}"))
            return False

    # ------------------------------------------------------------------------
    # Independent Monitor Loop
    # ------------------------------------------------------------------------
    def start(self) -> None:
        """Start the supervisor as an independent background monitor thread."""
        if self._is_running:
            return

        self._stop_event.clear()
        self._is_running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name=f"{self.name}-Monitor",
            daemon=True,
        )
        self._monitor_thread.start()
        if self.verbose:
            print(_format_log(self.name, "STARTED", f"Independent supervisor loop online (Interval: {self.check_interval}s)"))

    def _monitor_loop(self) -> None:
        """Continuous background loop inspecting agent health at regular intervals."""
        while not self._stop_event.is_set():
            self.check_health()
            time.sleep(self.check_interval)
        self._is_running = False

    def stop(self) -> None:
        """Stop the background monitor thread."""
        self._stop_event.set()
        self._is_running = False

    def join(self, timeout: float | None = 2.0) -> None:
        """Wait for the monitor thread to finish."""
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=timeout)

    def is_all_healthy(self) -> bool:
        """Return True if all registered agents are currently healthy."""
        health = self.check_health()
        return all(info["status"] == "healthy" for info in health.values())

    def get_summary(self) -> dict[str, Any]:
        """Return full supervisor metrics and health status."""
        return {
            "health_report": self.check_health(),
            "total_failures_recorded": len(self.failure_log),
            "recent_failures": self.failure_log[-5:],
        }
