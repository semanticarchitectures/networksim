"""Property-based tests for EventQueue ordering.

Feature: manet-simulation-engine, Property 5: Event queue ordering
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from manet_sim.core.event_queue import EventQueue


# Strategy that generates a list of (time, priority) tuples representing events
@st.composite
def event_schedule_and_drain_time(draw):
    """Generate a list of events with arbitrary timestamps and priorities, plus a drain time T.

    Events have:
    - time: float in [0.0, 1000.0] (simulation seconds)
    - priority: int in [0, 100]

    Drain time T is drawn from the same range so it may be above, below, or
    within the range of event timestamps.
    """
    events = draw(st.lists(
        st.tuples(
            st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
            st.integers(min_value=0, max_value=100),
        ),
        min_size=0,
        max_size=50,
    ))
    drain_time = draw(st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False))
    return events, drain_time


@given(params=event_schedule_and_drain_time())
@settings(max_examples=100, deadline=None)
def test_event_queue_ordering(params):
    """
    Property 5: Event queue ordering.

    For any set of scheduled events with arbitrary timestamps and priorities,
    draining the queue up to time T SHALL process all events with timestamp <= T
    in order of (timestamp ascending, priority ascending), and SHALL not process
    any event with timestamp > T.

    **Validates: Requirements 5.2, 5.3**
    """
    events, drain_time = params

    # Track the order in which callbacks are invoked
    processed_order: list[tuple[float, int]] = []

    # Schedule all events
    eq = EventQueue()
    for time, priority in events:
        # Capture time and priority in the callback closure
        t, p = time, priority
        eq.schedule(t, lambda t=t, p=p: processed_order.append((t, p)), priority=p)

    # Drain up to drain_time
    count = eq.drain_until(drain_time)

    # Determine which events should have been processed (timestamp <= drain_time)
    expected_processed = sorted(
        [(t, p) for t, p in events if t <= drain_time],
        key=lambda x: (x[0], x[1]),
    )

    # 1. The count returned must match the number of events with timestamp <= T
    assert count == len(expected_processed), (
        f"drain_until returned count={count}, expected {len(expected_processed)}. "
        f"drain_time={drain_time}"
    )

    # 2. The processed events must match exactly the set of events with timestamp <= T
    assert len(processed_order) == len(expected_processed), (
        f"Processed {len(processed_order)} events, expected {len(expected_processed)}"
    )

    # 3. The processing order must be (timestamp ascending, priority ascending)
    assert processed_order == expected_processed, (
        f"Events not processed in correct (time, priority) order.\n"
        f"Got:      {processed_order}\n"
        f"Expected: {expected_processed}"
    )

    # 4. Events with timestamp > T must NOT have been processed (remain in queue)
    remaining_count = sum(1 for t, _ in events if t > drain_time)
    # The queue should still contain exactly the events with timestamp > T
    assert not eq.is_empty() if remaining_count > 0 else eq.is_empty(), (
        f"Queue empty status incorrect. "
        f"Expected {remaining_count} remaining events, "
        f"is_empty={eq.is_empty()}"
    )
