"""Priority queue for discrete simulation events with millisecond resolution.

Events are ordered first by timestamp ascending, then by integer priority
ascending for events sharing the same timestamp.
"""

import heapq
from dataclasses import dataclass, field
from typing import Callable


@dataclass(order=True)
class ScheduledEvent:
    """A discrete event scheduled at a specific simulation time.

    Ordering is by (time, priority) ascending — lower values processed first.
    The callback is excluded from comparison to avoid issues with non-comparable
    callables.
    """

    time: float
    priority: int
    callback: Callable = field(compare=False)


class EventQueue:
    """Min-heap priority queue for discrete simulation events.

    Events are processed in order of (timestamp ascending, priority ascending).
    Supports scheduling events at any time, including in the past relative to
    the current drain time — such events are processed immediately during the
    next drain_until() call.
    """

    def __init__(self) -> None:
        self._heap: list[ScheduledEvent] = []

    def schedule(self, time: float, callback: Callable, priority: int = 0) -> None:
        """Schedule an event at the given simulation time.

        Args:
            time: Simulation time (seconds) at which the event should fire.
            callback: Callable to invoke when the event is processed.
            priority: Integer priority for tie-breaking same-time events.
                      Lower values are processed first. Defaults to 0.
        """
        event = ScheduledEvent(time=time, priority=priority, callback=callback)
        heapq.heappush(self._heap, event)

    def drain_until(self, t: float) -> int:
        """Process all events with timestamp <= t in priority-queue order.

        Events scheduled in the past (timestamp < current simulation time)
        are processed immediately — they are not discarded.

        Args:
            t: The current simulation time. All events with time <= t
               will be popped and their callbacks invoked.

        Returns:
            The number of events processed.
        """
        count = 0
        while self._heap and self._heap[0].time <= t:
            event = heapq.heappop(self._heap)
            event.callback()
            count += 1
        return count

    def is_empty(self) -> bool:
        """Return True if no events are queued."""
        return len(self._heap) == 0
