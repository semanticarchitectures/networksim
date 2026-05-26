"""Unit tests for EventQueue."""

import pytest
from manet_sim.core.event_queue import EventQueue, ScheduledEvent


class TestScheduledEvent:
    """Tests for ScheduledEvent dataclass ordering."""

    def test_ordering_by_time(self):
        """Events are ordered by time first."""
        e1 = ScheduledEvent(time=1.0, priority=0, callback=lambda: None)
        e2 = ScheduledEvent(time=2.0, priority=0, callback=lambda: None)
        assert e1 < e2

    def test_ordering_by_priority_when_same_time(self):
        """Events with same time are ordered by priority."""
        e1 = ScheduledEvent(time=1.0, priority=0, callback=lambda: None)
        e2 = ScheduledEvent(time=1.0, priority=1, callback=lambda: None)
        assert e1 < e2

    def test_equal_time_and_priority(self):
        """Events with same time and priority are considered equal in ordering."""
        e1 = ScheduledEvent(time=1.0, priority=0, callback=lambda: None)
        e2 = ScheduledEvent(time=1.0, priority=0, callback=lambda: None)
        assert not (e1 < e2)
        assert not (e2 < e1)

    def test_callback_excluded_from_comparison(self):
        """Different callbacks don't affect ordering."""
        e1 = ScheduledEvent(time=1.0, priority=0, callback=lambda: "a")
        e2 = ScheduledEvent(time=1.0, priority=0, callback=lambda: "b")
        assert e1 == e2  # Equal because callback is excluded


class TestEventQueueSchedule:
    """Tests for EventQueue.schedule()."""

    def test_schedule_single_event(self):
        """Scheduling an event makes the queue non-empty."""
        eq = EventQueue()
        assert eq.is_empty()
        eq.schedule(1.0, lambda: None)
        assert not eq.is_empty()

    def test_schedule_with_default_priority(self):
        """Events can be scheduled with default priority of 0."""
        eq = EventQueue()
        eq.schedule(1.0, lambda: None)
        assert not eq.is_empty()

    def test_schedule_with_explicit_priority(self):
        """Events can be scheduled with explicit priority."""
        eq = EventQueue()
        eq.schedule(1.0, lambda: None, priority=5)
        assert not eq.is_empty()

    def test_schedule_multiple_events(self):
        """Multiple events can be scheduled."""
        eq = EventQueue()
        eq.schedule(1.0, lambda: None)
        eq.schedule(2.0, lambda: None)
        eq.schedule(0.5, lambda: None)
        assert not eq.is_empty()


class TestEventQueueDrainUntil:
    """Tests for EventQueue.drain_until()."""

    def test_drain_empty_queue(self):
        """Draining an empty queue returns 0 and does nothing."""
        eq = EventQueue()
        count = eq.drain_until(10.0)
        assert count == 0
        assert eq.is_empty()

    def test_drain_processes_events_at_time(self):
        """Events at exactly time t are processed."""
        results = []
        eq = EventQueue()
        eq.schedule(1.0, lambda: results.append("a"))
        count = eq.drain_until(1.0)
        assert count == 1
        assert results == ["a"]
        assert eq.is_empty()

    def test_drain_processes_events_before_time(self):
        """Events before time t are processed."""
        results = []
        eq = EventQueue()
        eq.schedule(0.5, lambda: results.append("a"))
        eq.schedule(0.8, lambda: results.append("b"))
        count = eq.drain_until(1.0)
        assert count == 2
        assert results == ["a", "b"]

    def test_drain_does_not_process_future_events(self):
        """Events after time t are not processed."""
        results = []
        eq = EventQueue()
        eq.schedule(1.0, lambda: results.append("a"))
        eq.schedule(2.0, lambda: results.append("b"))
        count = eq.drain_until(1.5)
        assert count == 1
        assert results == ["a"]
        assert not eq.is_empty()

    def test_drain_time_ordering(self):
        """Events are processed in time order."""
        results = []
        eq = EventQueue()
        eq.schedule(3.0, lambda: results.append("c"))
        eq.schedule(1.0, lambda: results.append("a"))
        eq.schedule(2.0, lambda: results.append("b"))
        count = eq.drain_until(3.0)
        assert count == 3
        assert results == ["a", "b", "c"]

    def test_drain_priority_ordering_same_time(self):
        """Events at same time are processed in priority order (ascending)."""
        results = []
        eq = EventQueue()
        eq.schedule(1.0, lambda: results.append("high"), priority=10)
        eq.schedule(1.0, lambda: results.append("low"), priority=1)
        eq.schedule(1.0, lambda: results.append("mid"), priority=5)
        count = eq.drain_until(1.0)
        assert count == 3
        assert results == ["low", "mid", "high"]

    def test_drain_time_then_priority_ordering(self):
        """Events ordered by time first, then priority within same time."""
        results = []
        eq = EventQueue()
        eq.schedule(2.0, lambda: results.append("t2_p1"), priority=1)
        eq.schedule(1.0, lambda: results.append("t1_p2"), priority=2)
        eq.schedule(1.0, lambda: results.append("t1_p1"), priority=1)
        eq.schedule(2.0, lambda: results.append("t2_p0"), priority=0)
        count = eq.drain_until(2.0)
        assert count == 4
        assert results == ["t1_p1", "t1_p2", "t2_p0", "t2_p1"]

    def test_drain_past_events_processed_immediately(self):
        """Events scheduled in the past are processed during current drain."""
        results = []
        eq = EventQueue()
        # Schedule event at time 0.5, but drain at time 5.0
        eq.schedule(0.5, lambda: results.append("past"))
        eq.schedule(3.0, lambda: results.append("current"))
        count = eq.drain_until(5.0)
        assert count == 2
        assert results == ["past", "current"]

    def test_drain_returns_count(self):
        """drain_until returns the number of events processed."""
        eq = EventQueue()
        eq.schedule(1.0, lambda: None)
        eq.schedule(2.0, lambda: None)
        eq.schedule(3.0, lambda: None)
        eq.schedule(4.0, lambda: None)
        count = eq.drain_until(2.5)
        assert count == 2

    def test_drain_multiple_calls(self):
        """Multiple drain calls process events incrementally."""
        results = []
        eq = EventQueue()
        eq.schedule(1.0, lambda: results.append("a"))
        eq.schedule(2.0, lambda: results.append("b"))
        eq.schedule(3.0, lambda: results.append("c"))

        count1 = eq.drain_until(1.5)
        assert count1 == 1
        assert results == ["a"]

        count2 = eq.drain_until(2.5)
        assert count2 == 1
        assert results == ["a", "b"]

        count3 = eq.drain_until(3.0)
        assert count3 == 1
        assert results == ["a", "b", "c"]

    def test_drain_no_events_at_time(self):
        """When no events exist at or before time t, returns 0."""
        eq = EventQueue()
        eq.schedule(5.0, lambda: None)
        count = eq.drain_until(3.0)
        assert count == 0
        assert not eq.is_empty()

    def test_drain_millisecond_resolution(self):
        """Events with millisecond time differences are ordered correctly."""
        results = []
        eq = EventQueue()
        eq.schedule(1.002, lambda: results.append("c"))
        eq.schedule(1.001, lambda: results.append("b"))
        eq.schedule(1.000, lambda: results.append("a"))
        count = eq.drain_until(1.002)
        assert count == 3
        assert results == ["a", "b", "c"]

    def test_drain_negative_priority(self):
        """Negative priority values are valid and sort before zero."""
        results = []
        eq = EventQueue()
        eq.schedule(1.0, lambda: results.append("zero"), priority=0)
        eq.schedule(1.0, lambda: results.append("neg"), priority=-1)
        count = eq.drain_until(1.0)
        assert count == 2
        assert results == ["neg", "zero"]


class TestEventQueueIsEmpty:
    """Tests for EventQueue.is_empty()."""

    def test_empty_on_creation(self):
        """New queue is empty."""
        eq = EventQueue()
        assert eq.is_empty()

    def test_not_empty_after_schedule(self):
        """Queue is not empty after scheduling."""
        eq = EventQueue()
        eq.schedule(1.0, lambda: None)
        assert not eq.is_empty()

    def test_empty_after_drain_all(self):
        """Queue is empty after draining all events."""
        eq = EventQueue()
        eq.schedule(1.0, lambda: None)
        eq.schedule(2.0, lambda: None)
        eq.drain_until(2.0)
        assert eq.is_empty()

    def test_not_empty_after_partial_drain(self):
        """Queue is not empty after partial drain."""
        eq = EventQueue()
        eq.schedule(1.0, lambda: None)
        eq.schedule(3.0, lambda: None)
        eq.drain_until(2.0)
        assert not eq.is_empty()


class TestEventQueueEdgeCases:
    """Edge case tests for EventQueue."""

    def test_schedule_at_time_zero(self):
        """Events can be scheduled at time 0."""
        results = []
        eq = EventQueue()
        eq.schedule(0.0, lambda: results.append("zero"))
        count = eq.drain_until(0.0)
        assert count == 1
        assert results == ["zero"]

    def test_many_events_same_time(self):
        """Many events at the same time are all processed."""
        results = []
        eq = EventQueue()
        for i in range(100):
            eq.schedule(1.0, lambda i=i: results.append(i), priority=i)
        count = eq.drain_until(1.0)
        assert count == 100
        assert results == list(range(100))

    def test_events_scheduled_during_drain_not_processed(self):
        """Events added by callbacks during drain are not processed in same drain
        if their time > t."""
        eq = EventQueue()
        results = []

        def callback_that_schedules():
            results.append("first")
            eq.schedule(5.0, lambda: results.append("scheduled_during"))

        eq.schedule(1.0, callback_that_schedules)
        count = eq.drain_until(2.0)
        assert count == 1
        assert results == ["first"]
        assert not eq.is_empty()

        # Now drain the newly scheduled event
        count2 = eq.drain_until(5.0)
        assert count2 == 1
        assert results == ["first", "scheduled_during"]

    def test_callback_scheduling_event_at_current_drain_time(self):
        """Events scheduled by callbacks at time <= t during drain are processed."""
        eq = EventQueue()
        results = []

        def callback_that_schedules():
            results.append("first")
            eq.schedule(1.0, lambda: results.append("also_at_1"), priority=5)

        eq.schedule(1.0, callback_that_schedules, priority=0)
        count = eq.drain_until(1.0)
        # The newly scheduled event at time 1.0 should also be processed
        assert count == 2
        assert results == ["first", "also_at_1"]
