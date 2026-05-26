"""Unit tests for EventBus."""

import numpy as np
import pytest

from manet_sim.core.event_bus import (
    LINK_BROKEN,
    LINK_FORMED,
    POSITION_UPDATE,
    SIMULATION_END,
    STEP_COMPLETE,
    EventBus,
    LinkBrokenEvent,
    LinkFormedEvent,
    PositionUpdateEvent,
    SimulationEndEvent,
    StepCompleteEvent,
)


class TestEventBusSubscribe:
    """Tests for subscribe and unsubscribe."""

    def test_subscribe_returns_unique_id(self):
        """Each subscription gets a unique ID."""
        bus = EventBus()
        id1 = bus.subscribe("test_event", lambda p: None)
        id2 = bus.subscribe("test_event", lambda p: None)
        assert id1 != id2

    def test_subscribe_different_types(self):
        """Can subscribe to different event types."""
        bus = EventBus()
        id1 = bus.subscribe("type_a", lambda p: None)
        id2 = bus.subscribe("type_b", lambda p: None)
        assert id1 != id2

    def test_unsubscribe_removes_subscription(self):
        """Unsubscribed handler no longer receives events."""
        bus = EventBus()
        received = []
        sub_id = bus.subscribe("test", lambda p: received.append(p))
        bus.unsubscribe(sub_id)
        bus.publish("test", "payload")
        assert received == []

    def test_unsubscribe_unknown_id_raises(self):
        """Unsubscribing with unknown ID raises KeyError."""
        bus = EventBus()
        with pytest.raises(KeyError):
            bus.unsubscribe("nonexistent-id")


class TestEventBusPublish:
    """Tests for event publishing and delivery."""

    def test_publish_delivers_to_subscriber(self):
        """Published event is delivered to subscriber."""
        bus = EventBus()
        received = []
        bus.subscribe("test", lambda p: received.append(p))
        bus.publish("test", "hello")
        assert received == ["hello"]

    def test_publish_delivers_to_multiple_subscribers(self):
        """Published event is delivered to all subscribers of that type."""
        bus = EventBus()
        received_a = []
        received_b = []
        bus.subscribe("test", lambda p: received_a.append(p))
        bus.subscribe("test", lambda p: received_b.append(p))
        bus.publish("test", "data")
        assert received_a == ["data"]
        assert received_b == ["data"]

    def test_publish_only_delivers_to_matching_type(self):
        """Events are only delivered to subscribers of the matching type."""
        bus = EventBus()
        received_a = []
        received_b = []
        bus.subscribe("type_a", lambda p: received_a.append(p))
        bus.subscribe("type_b", lambda p: received_b.append(p))
        bus.publish("type_a", "payload_a")
        assert received_a == ["payload_a"]
        assert received_b == []

    def test_publish_preserves_order(self):
        """Subscribers receive events in subscription order."""
        bus = EventBus()
        order = []
        bus.subscribe("test", lambda p: order.append(1))
        bus.subscribe("test", lambda p: order.append(2))
        bus.subscribe("test", lambda p: order.append(3))
        bus.publish("test", None)
        assert order == [1, 2, 3]

    def test_publish_multiple_events_in_order(self):
        """Multiple publishes deliver events in publication order."""
        bus = EventBus()
        received = []
        bus.subscribe("test", lambda p: received.append(p))
        bus.publish("test", "first")
        bus.publish("test", "second")
        bus.publish("test", "third")
        assert received == ["first", "second", "third"]

    def test_publish_no_subscribers_does_not_error(self):
        """Publishing to a type with no subscribers does not raise."""
        bus = EventBus()
        bus.publish("no_subscribers", "data")  # Should not raise


class TestEventBusFaultIsolation:
    """Tests for subscriber exception handling."""

    def test_exception_does_not_interrupt_other_subscribers(self):
        """A failing subscriber does not prevent delivery to others."""
        bus = EventBus()
        received = []

        def failing_handler(p):
            raise RuntimeError("I failed!")

        bus.subscribe("test", lambda p: received.append("before"))
        bus.subscribe("test", failing_handler)
        bus.subscribe("test", lambda p: received.append("after"))

        bus.publish("test", "data")
        assert received == ["before", "after"]

    def test_exception_is_logged(self, caplog):
        """Subscriber exceptions are logged."""
        bus = EventBus()

        def failing_handler(p):
            raise ValueError("test error")

        bus.subscribe("test", failing_handler)

        import logging

        with caplog.at_level(logging.ERROR):
            bus.publish("test", "data")

        assert "raised an exception" in caplog.text

    def test_multiple_failing_subscribers(self):
        """Multiple failing subscribers don't block each other."""
        bus = EventBus()
        received = []

        def fail_1(p):
            raise RuntimeError("fail 1")

        def fail_2(p):
            raise RuntimeError("fail 2")

        bus.subscribe("test", fail_1)
        bus.subscribe("test", lambda p: received.append("ok_1"))
        bus.subscribe("test", fail_2)
        bus.subscribe("test", lambda p: received.append("ok_2"))

        bus.publish("test", "data")
        assert received == ["ok_1", "ok_2"]


class TestEventBusEnableDisable:
    """Tests for enable/disable mechanism."""

    def test_disable_stops_delivery(self):
        """Disabled subscriber does not receive events."""
        bus = EventBus()
        received = []
        sub_id = bus.subscribe("test", lambda p: received.append(p))
        bus.disable(sub_id)
        bus.publish("test", "data")
        assert received == []

    def test_enable_resumes_delivery(self):
        """Re-enabled subscriber receives subsequent events."""
        bus = EventBus()
        received = []
        sub_id = bus.subscribe("test", lambda p: received.append(p))
        bus.disable(sub_id)
        bus.publish("test", "missed")
        bus.enable(sub_id)
        bus.publish("test", "received")
        assert received == ["received"]

    def test_disable_does_not_remove_subscription(self):
        """Disabled subscription can be re-enabled."""
        bus = EventBus()
        received = []
        sub_id = bus.subscribe("test", lambda p: received.append(p))
        bus.disable(sub_id)
        bus.enable(sub_id)
        bus.publish("test", "data")
        assert received == ["data"]

    def test_disable_unknown_id_raises(self):
        """Disabling unknown subscription raises KeyError."""
        bus = EventBus()
        with pytest.raises(KeyError):
            bus.disable("nonexistent")

    def test_enable_unknown_id_raises(self):
        """Enabling unknown subscription raises KeyError."""
        bus = EventBus()
        with pytest.raises(KeyError):
            bus.enable("nonexistent")

    def test_disable_only_affects_target_subscriber(self):
        """Disabling one subscriber does not affect others."""
        bus = EventBus()
        received_a = []
        received_b = []
        sub_a = bus.subscribe("test", lambda p: received_a.append(p))
        sub_b = bus.subscribe("test", lambda p: received_b.append(p))
        bus.disable(sub_a)
        bus.publish("test", "data")
        assert received_a == []
        assert received_b == ["data"]

    def test_no_replay_on_reenable(self):
        """Re-enabling does not replay events missed while disabled."""
        bus = EventBus()
        received = []
        sub_id = bus.subscribe("test", lambda p: received.append(p))
        bus.publish("test", "event_1")
        bus.disable(sub_id)
        bus.publish("test", "event_2")
        bus.publish("test", "event_3")
        bus.enable(sub_id)
        bus.publish("test", "event_4")
        assert received == ["event_1", "event_4"]


class TestEventPayloadDataclasses:
    """Tests for event payload dataclass definitions."""

    def test_position_update_event(self):
        """PositionUpdateEvent stores timestamp, positions, velocities."""
        positions = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        velocities = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        event = PositionUpdateEvent(
            timestamp=10.0, positions=positions, velocities=velocities
        )
        assert event.timestamp == 10.0
        np.testing.assert_array_equal(event.positions, positions)
        np.testing.assert_array_equal(event.velocities, velocities)

    def test_link_formed_event(self):
        """LinkFormedEvent stores timestamp, node_a, node_b, distance."""
        event = LinkFormedEvent(timestamp=5.0, node_a=0, node_b=7, distance=0.85)
        assert event.timestamp == 5.0
        assert event.node_a == 0
        assert event.node_b == 7
        assert event.distance == 0.85

    def test_link_broken_event(self):
        """LinkBrokenEvent stores timestamp, node_a, node_b."""
        event = LinkBrokenEvent(timestamp=12.0, node_a=3, node_b=9)
        assert event.timestamp == 12.0
        assert event.node_a == 3
        assert event.node_b == 9

    def test_step_complete_event(self):
        """StepCompleteEvent stores timestamp, step_number, active_links, wall_clock_ms."""
        event = StepCompleteEvent(
            timestamp=100.0, step_number=100, active_links=450, wall_clock_ms=23.5
        )
        assert event.timestamp == 100.0
        assert event.step_number == 100
        assert event.active_links == 450
        assert event.wall_clock_ms == 23.5

    def test_simulation_end_event(self):
        """SimulationEndEvent stores total_time, total_steps, wall_clock_seconds."""
        event = SimulationEndEvent(
            total_time=3600.0, total_steps=3600, wall_clock_seconds=120.5
        )
        assert event.total_time == 3600.0
        assert event.total_steps == 3600
        assert event.wall_clock_seconds == 120.5


class TestEventBusEventTypes:
    """Tests for standard event type constants."""

    def test_event_type_constants_defined(self):
        """All required event type constants are defined."""
        assert POSITION_UPDATE == "position_update"
        assert LINK_FORMED == "link_formed"
        assert LINK_BROKEN == "link_broken"
        assert STEP_COMPLETE == "step_complete"
        assert SIMULATION_END == "simulation_end"

    def test_publish_with_standard_event_types(self):
        """Standard event types work with publish/subscribe."""
        bus = EventBus()
        received = {}

        for event_type in [
            POSITION_UPDATE,
            LINK_FORMED,
            LINK_BROKEN,
            STEP_COMPLETE,
            SIMULATION_END,
        ]:
            received[event_type] = []
            bus.subscribe(event_type, lambda p, et=event_type: received[et].append(p))

        bus.publish(POSITION_UPDATE, "pos_data")
        bus.publish(LINK_FORMED, "link_data")

        assert received[POSITION_UPDATE] == ["pos_data"]
        assert received[LINK_FORMED] == ["link_data"]
        assert received[LINK_BROKEN] == []
        assert received[STEP_COMPLETE] == []
        assert received[SIMULATION_END] == []


class TestEventBusIntegration:
    """Integration-style tests combining multiple features."""

    def test_full_workflow(self):
        """Full workflow: subscribe, publish, disable, publish, enable, publish."""
        bus = EventBus()
        received = []
        sub_id = bus.subscribe(STEP_COMPLETE, lambda p: received.append(p))

        event1 = StepCompleteEvent(
            timestamp=1.0, step_number=1, active_links=10, wall_clock_ms=5.0
        )
        bus.publish(STEP_COMPLETE, event1)
        assert len(received) == 1
        assert received[0].step_number == 1

        bus.disable(sub_id)
        event2 = StepCompleteEvent(
            timestamp=2.0, step_number=2, active_links=12, wall_clock_ms=4.5
        )
        bus.publish(STEP_COMPLETE, event2)
        assert len(received) == 1  # Still 1, disabled

        bus.enable(sub_id)
        event3 = StepCompleteEvent(
            timestamp=3.0, step_number=3, active_links=15, wall_clock_ms=6.0
        )
        bus.publish(STEP_COMPLETE, event3)
        assert len(received) == 2
        assert received[1].step_number == 3

    def test_unsubscribe_after_disable(self):
        """Can unsubscribe a disabled subscription."""
        bus = EventBus()
        received = []
        sub_id = bus.subscribe("test", lambda p: received.append(p))
        bus.disable(sub_id)
        bus.unsubscribe(sub_id)
        bus.publish("test", "data")
        assert received == []

    def test_multiple_event_types_same_handler(self):
        """Same handler can subscribe to multiple event types."""
        bus = EventBus()
        received = []
        handler = lambda p: received.append(p)
        bus.subscribe(LINK_FORMED, handler)
        bus.subscribe(LINK_BROKEN, handler)

        bus.publish(LINK_FORMED, "formed")
        bus.publish(LINK_BROKEN, "broken")
        assert received == ["formed", "broken"]
