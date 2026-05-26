"""Event bus for decoupled publish-subscribe communication between simulation components."""

import logging
import uuid
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


# --- Event type constants ---

POSITION_UPDATE = "position_update"
LINK_FORMED = "link_formed"
LINK_BROKEN = "link_broken"
STEP_COMPLETE = "step_complete"
SIMULATION_END = "simulation_end"


# --- Event payload dataclasses ---


@dataclass
class PositionUpdateEvent:
    """Payload for position_update events."""

    timestamp: float
    positions: np.ndarray  # (N, 2) reference, not copy
    velocities: np.ndarray  # (N, 2) reference


@dataclass
class LinkFormedEvent:
    """Payload for link_formed events."""

    timestamp: float
    node_a: int
    node_b: int
    distance: float


@dataclass
class LinkBrokenEvent:
    """Payload for link_broken events."""

    timestamp: float
    node_a: int
    node_b: int


@dataclass
class StepCompleteEvent:
    """Payload for step_complete events."""

    timestamp: float
    step_number: int
    active_links: int
    wall_clock_ms: float


@dataclass
class SimulationEndEvent:
    """Payload for simulation_end events."""

    total_time: float
    total_steps: int
    wall_clock_seconds: float


# --- Subscriber record ---


@dataclass
class _Subscription:
    """Internal record for a single subscription."""

    subscription_id: str
    event_type: str
    handler: Callable
    enabled: bool = True


# --- EventBus ---


class EventBus:
    """Publish-subscribe event bus for simulation components.

    Components publish events by type and zero or more collectors subscribe
    to receive events of specified types. Events are delivered to subscribers
    in the order they were published. Subscriber exceptions are caught and
    logged without interrupting delivery to other subscribers.
    """

    def __init__(self) -> None:
        # Map subscription_id -> _Subscription
        self._subscriptions: dict[str, _Subscription] = {}
        # Map event_type -> list of subscription_ids (preserves insertion order)
        self._subscribers_by_type: dict[str, list[str]] = {}

    def subscribe(self, event_type: str, handler: Callable) -> str:
        """Register a handler for a given event type.

        Args:
            event_type: The event type string to subscribe to.
            handler: A callable that accepts a single payload argument.

        Returns:
            A unique subscription ID that can be used to unsubscribe,
            enable, or disable this subscription.
        """
        subscription_id = str(uuid.uuid4())
        sub = _Subscription(
            subscription_id=subscription_id,
            event_type=event_type,
            handler=handler,
            enabled=True,
        )
        self._subscriptions[subscription_id] = sub

        if event_type not in self._subscribers_by_type:
            self._subscribers_by_type[event_type] = []
        self._subscribers_by_type[event_type].append(subscription_id)

        return subscription_id

    def unsubscribe(self, subscription_id: str) -> None:
        """Remove a subscription entirely.

        Args:
            subscription_id: The ID returned by subscribe().

        Raises:
            KeyError: If the subscription_id is not found.
        """
        if subscription_id not in self._subscriptions:
            raise KeyError(f"Subscription '{subscription_id}' not found")

        sub = self._subscriptions.pop(subscription_id)
        event_type = sub.event_type

        if event_type in self._subscribers_by_type:
            self._subscribers_by_type[event_type].remove(subscription_id)
            if not self._subscribers_by_type[event_type]:
                del self._subscribers_by_type[event_type]

    def publish(self, event_type: str, payload: Any) -> None:
        """Publish an event to all active subscribers of the given type.

        Events are delivered in subscription order. If a subscriber raises
        an exception, it is caught and logged, and delivery continues to
        remaining subscribers.

        Args:
            event_type: The event type string.
            payload: The event payload to deliver to each subscriber.
        """
        sub_ids = self._subscribers_by_type.get(event_type, [])

        for sub_id in sub_ids:
            sub = self._subscriptions.get(sub_id)
            if sub is None:
                continue
            if not sub.enabled:
                continue
            try:
                sub.handler(payload)
            except Exception:
                logger.exception(
                    "Subscriber %s raised an exception handling event '%s'",
                    sub_id,
                    event_type,
                )

    def enable(self, subscription_id: str) -> None:
        """Re-enable a disabled subscription to resume event delivery.

        Args:
            subscription_id: The ID returned by subscribe().

        Raises:
            KeyError: If the subscription_id is not found.
        """
        if subscription_id not in self._subscriptions:
            raise KeyError(f"Subscription '{subscription_id}' not found")
        self._subscriptions[subscription_id].enabled = True

    def disable(self, subscription_id: str) -> None:
        """Disable a subscription to stop event delivery without removing it.

        Args:
            subscription_id: The ID returned by subscribe().

        Raises:
            KeyError: If the subscription_id is not found.
        """
        if subscription_id not in self._subscriptions:
            raise KeyError(f"Subscription '{subscription_id}' not found")
        self._subscriptions[subscription_id].enabled = False
