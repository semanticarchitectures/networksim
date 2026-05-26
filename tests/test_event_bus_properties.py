"""Property-based tests for EventBus.

Feature: manet-simulation-engine
Property 24: Event bus ordered delivery
Property 25: Event bus fault isolation
Property 26: Event bus enable/disable
"""

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from manet_sim.core.event_bus import EventBus


# --- Strategies ---


@st.composite
def event_type_strategy(draw):
    """Generate a valid event type string."""
    return draw(st.sampled_from([
        "position_update", "link_formed", "link_broken",
        "step_complete", "simulation_end", "custom_event",
    ]))


@st.composite
def event_payloads(draw):
    """Generate a list of distinct payloads to publish."""
    n = draw(st.integers(min_value=1, max_value=50))
    return [f"event_{i}" for i in range(n)]


@st.composite
def subscriber_count(draw):
    """Generate a reasonable number of subscribers."""
    return draw(st.integers(min_value=1, max_value=10))


# --- Property 24: Event bus ordered delivery ---


@given(payloads=event_payloads(), event_type=event_type_strategy())
@settings(max_examples=100, deadline=None)
def test_event_bus_ordered_delivery(payloads, event_type):
    """
    Property 24: Event bus ordered delivery.

    For any sequence of events published to the event bus, each subscriber
    SHALL receive events in the exact order they were published.

    **Validates: Requirements 15.1**
    """
    bus = EventBus()
    received = []
    bus.subscribe(event_type, lambda p: received.append(p))

    # Publish all payloads in order
    for payload in payloads:
        bus.publish(event_type, payload)

    # Subscriber must have received events in exact publication order
    assert received == payloads, (
        f"Events not received in publication order. "
        f"Published: {payloads}, Received: {received}"
    )


@given(
    payloads=event_payloads(),
    event_type=event_type_strategy(),
    num_subscribers=subscriber_count(),
)
@settings(max_examples=100, deadline=None)
def test_event_bus_ordered_delivery_multiple_subscribers(payloads, event_type, num_subscribers):
    """
    Property 24 (extended): Multiple subscribers each receive events in publication order.

    For any number of subscribers and any sequence of published events,
    every subscriber SHALL receive the events in the exact order they were published.

    **Validates: Requirements 15.1**
    """
    bus = EventBus()
    received_lists = [[] for _ in range(num_subscribers)]

    for i in range(num_subscribers):
        idx = i  # capture for closure
        bus.subscribe(event_type, lambda p, idx=idx: received_lists[idx].append(p))

    # Publish all payloads
    for payload in payloads:
        bus.publish(event_type, payload)

    # Each subscriber must have received events in exact publication order
    for i, received in enumerate(received_lists):
        assert received == payloads, (
            f"Subscriber {i} did not receive events in publication order. "
            f"Published: {payloads}, Received: {received}"
        )


# --- Property 25: Event bus fault isolation ---


@given(
    payloads=event_payloads(),
    event_type=event_type_strategy(),
    failing_index=st.integers(min_value=0, max_value=9),
    num_subscribers=st.integers(min_value=2, max_value=10),
)
@settings(max_examples=100, deadline=None)
def test_event_bus_fault_isolation(payloads, event_type, failing_index, num_subscribers):
    """
    Property 25: Event bus fault isolation.

    For any event published to the bus where one subscriber raises an exception,
    all other active subscribers registered for that event type SHALL still
    receive the event.

    **Validates: Requirements 15.4**
    """
    # Ensure failing_index is within subscriber range
    failing_index = failing_index % num_subscribers

    bus = EventBus()
    received_lists = [[] for _ in range(num_subscribers)]

    for i in range(num_subscribers):
        if i == failing_index:
            def failing_handler(p):
                raise RuntimeError("Subscriber failure!")
            bus.subscribe(event_type, failing_handler)
        else:
            idx = i
            bus.subscribe(event_type, lambda p, idx=idx: received_lists[idx].append(p))

    # Publish all payloads
    for payload in payloads:
        bus.publish(event_type, payload)

    # All non-failing subscribers must have received all events in order
    for i, received in enumerate(received_lists):
        if i == failing_index:
            # Failing subscriber's list should be empty (it raises instead of appending)
            assert received == [], (
                f"Failing subscriber {i} should not have recorded events."
            )
        else:
            assert received == payloads, (
                f"Subscriber {i} did not receive all events despite another subscriber failing. "
                f"Published: {payloads}, Received: {received}"
            )


@given(
    payloads=event_payloads(),
    event_type=event_type_strategy(),
    num_failing=st.integers(min_value=1, max_value=5),
    num_healthy=st.integers(min_value=1, max_value=5),
)
@settings(max_examples=100, deadline=None)
def test_event_bus_fault_isolation_multiple_failures(payloads, event_type, num_failing, num_healthy):
    """
    Property 25 (extended): Multiple failing subscribers don't block healthy ones.

    For any number of failing subscribers, all healthy subscribers SHALL still
    receive all events in publication order.

    **Validates: Requirements 15.4**
    """
    bus = EventBus()
    healthy_received = [[] for _ in range(num_healthy)]

    # Register failing subscribers
    for _ in range(num_failing):
        def failing_handler(p):
            raise RuntimeError("Failure!")
        bus.subscribe(event_type, failing_handler)

    # Register healthy subscribers
    for i in range(num_healthy):
        idx = i
        bus.subscribe(event_type, lambda p, idx=idx: healthy_received[idx].append(p))

    # Publish all payloads
    for payload in payloads:
        bus.publish(event_type, payload)

    # All healthy subscribers must have received all events
    for i, received in enumerate(healthy_received):
        assert received == payloads, (
            f"Healthy subscriber {i} did not receive all events. "
            f"Published: {payloads}, Received: {received}"
        )


# --- Property 26: Event bus enable/disable ---


@given(
    payloads_before=event_payloads(),
    payloads_during=event_payloads(),
    payloads_after=event_payloads(),
    event_type=event_type_strategy(),
)
@settings(max_examples=100, deadline=None)
def test_event_bus_enable_disable(payloads_before, payloads_during, payloads_after, event_type):
    """
    Property 26: Event bus enable/disable.

    For any subscriber that is disabled, the event bus SHALL not deliver events
    to it. Re-enabling the subscriber SHALL resume delivery for subsequent events
    without replaying missed events.

    **Validates: Requirements 15.5**
    """
    bus = EventBus()
    received = []
    sub_id = bus.subscribe(event_type, lambda p: received.append(p))

    # Phase 1: Publish while enabled — subscriber receives all
    for payload in payloads_before:
        bus.publish(event_type, payload)

    assert received == payloads_before, (
        f"Subscriber did not receive events while enabled. "
        f"Expected: {payloads_before}, Got: {received}"
    )

    # Phase 2: Disable and publish — subscriber receives nothing new
    bus.disable(sub_id)
    for payload in payloads_during:
        bus.publish(event_type, payload)

    assert received == payloads_before, (
        f"Disabled subscriber received events it should not have. "
        f"Expected: {payloads_before}, Got: {received}"
    )

    # Phase 3: Re-enable and publish — subscriber receives only new events
    bus.enable(sub_id)
    for payload in payloads_after:
        bus.publish(event_type, payload)

    expected = payloads_before + payloads_after
    assert received == expected, (
        f"Re-enabled subscriber did not receive correct events. "
        f"Expected: {expected}, Got: {received}. "
        f"Events during disable should NOT be replayed."
    )


@given(
    payloads=event_payloads(),
    event_type=event_type_strategy(),
    num_subscribers=st.integers(min_value=2, max_value=6),
    disable_index=st.integers(min_value=0, max_value=5),
)
@settings(max_examples=100, deadline=None)
def test_event_bus_disable_only_affects_target(payloads, event_type, num_subscribers, disable_index):
    """
    Property 26 (extended): Disabling one subscriber does not affect others.

    When one subscriber is disabled, all other subscribers SHALL continue
    to receive events normally.

    **Validates: Requirements 15.5**
    """
    disable_index = disable_index % num_subscribers

    bus = EventBus()
    received_lists = [[] for _ in range(num_subscribers)]
    sub_ids = []

    for i in range(num_subscribers):
        idx = i
        sub_id = bus.subscribe(event_type, lambda p, idx=idx: received_lists[idx].append(p))
        sub_ids.append(sub_id)

    # Disable one subscriber
    bus.disable(sub_ids[disable_index])

    # Publish events
    for payload in payloads:
        bus.publish(event_type, payload)

    # Disabled subscriber should have received nothing
    assert received_lists[disable_index] == [], (
        f"Disabled subscriber {disable_index} received events."
    )

    # All other subscribers should have received all events in order
    for i, received in enumerate(received_lists):
        if i != disable_index:
            assert received == payloads, (
                f"Subscriber {i} did not receive all events while another was disabled. "
                f"Published: {payloads}, Received: {received}"
            )
