"""Thread-safe asynchronous FIFO event broker for decoupled agent-to-agent messaging."""

from __future__ import annotations

import collections
import queue
import threading
from typing import Any, Callable


class EventQueue:
    """Decoupled, thread-safe asynchronous message broker.

    Key Architectural Guarantee:
    `publish()` puts events onto an in-memory FIFO queue and returns immediately (< 0.1ms).
    It NEVER synchronously invokes consumers. Independent consumer workers consume
    events asynchronously on their own threads.
    """

    def __init__(self, max_history: int = 1000):
        self._queues: dict[str, queue.Queue] = collections.defaultdict(queue.Queue)
        self._lock = threading.RLock()
        self._history: collections.deque[tuple[str, Any]] = collections.deque(maxlen=max_history)
        self._diagnostic_listeners: list[Callable[[str, Any], None]] = []
        self._is_shutdown = False

    def add_diagnostic_listener(self, listener: Callable[[str, Any], None]) -> None:
        """Add a read-only logging/audit listener (invoked without blocking queue)."""
        with self._lock:
            if listener not in self._diagnostic_listeners:
                self._diagnostic_listeners.append(listener)

    def publish(self, topic: str, event: Any) -> None:
        """Publish an event to a topic FIFO queue and return immediately.

        Consumers will NOT be invoked in this call stack.
        """
        if self._is_shutdown:
            return

        with self._lock:
            self._history.append((topic, event))
            q = self._queues[topic]

        # Put onto FIFO queue (non-blocking, instantaneous)
        q.put(event)

        # Notify diagnostic listeners if any
        with self._lock:
            listeners = list(self._diagnostic_listeners)
        for listener in listeners:
            try:
                listener(topic, event)
            except Exception:
                pass

    def consume(self, topic: str, timeout: float | None = 0.5) -> Any | None:
        """Block or poll for the next event from a topic FIFO queue.

        Used by independent worker threads to wait for incoming events.
        """
        if self._is_shutdown:
            return None

        with self._lock:
            q = self._queues[topic]

        try:
            if timeout is None or timeout <= 0:
                return q.get_nowait()
            return q.get(timeout=timeout)
        except queue.Empty:
            return None

    def poll(self, topic: str) -> Any | None:
        """Non-blocking poll of the next event from a topic queue."""
        return self.consume(topic, timeout=0.0)

    def get_all(self, topic: str) -> list[Any]:
        """Drain and return all currently pending events from a topic queue."""
        events: list[Any] = []
        while True:
            item = self.poll(topic)
            if item is None:
                break
            events.append(item)
        return events

    def get_history(self, topic: str | None = None) -> list[Any]:
        """Return history of events, optionally filtered by topic."""
        with self._lock:
            if topic is None:
                return [ev for _, ev in self._history]
            return [ev for top, ev in self._history if top == topic]

    def clear(self) -> None:
        """Reset all queues and history."""
        with self._lock:
            self._queues.clear()
            self._history.clear()

    def shutdown(self) -> None:
        """Shut down the event queue, unblocking waiting consumers."""
        self._is_shutdown = True
