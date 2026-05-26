"""Property-based tests for SimulationClock.

Feature: manet-simulation-engine, Property 3: Clock advancement reaches exact end time
"""

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from manet_sim.core.clock import SimulationClock


# Strategy that generates valid (duration, step_size) pairs ensuring
# step_size <= duration and the number of steps is bounded for test performance.
# We constrain so that duration / step_size <= 100_000 to keep tests fast.
@st.composite
def valid_duration_and_step_size(draw):
    """Generate a (duration, step_size) pair where step_size <= duration.

    Covers the full valid ranges:
    - duration: 1.0 to 86400.0
    - step_size: 0.001 to 3600.0

    Constrains max iterations to 100,000 to keep tests within deadline.
    """
    duration = draw(st.floats(min_value=1.0, max_value=86400.0, allow_nan=False, allow_infinity=False))
    # step_size must be <= duration and within valid range
    max_step = min(duration, 3600.0)
    # Ensure at most 100,000 steps for performance
    min_step = max(0.001, duration / 100_000.0)
    assume(min_step <= max_step)
    step_size = draw(st.floats(min_value=min_step, max_value=max_step, allow_nan=False, allow_infinity=False))
    return duration, step_size


@given(params=valid_duration_and_step_size())
@settings(max_examples=100, deadline=None)
def test_clock_advancement_reaches_exact_end_time(params):
    """
    Property 3: Clock advancement reaches exact end time.

    For any valid duration (1.0–86400.0) and step_size (0.001–3600.0) where
    step_size <= duration, advancing the SimulationClock to completion SHALL
    result in a final current_time exactly equal to the configured end time,
    regardless of whether duration is evenly divisible by step_size.

    **Validates: Requirements 3.1, 3.3, 3.5**
    """
    duration, step_size = params

    start = 0.0
    end = duration
    clock = SimulationClock(start=start, end=end, step_size=step_size)

    # Advance clock to completion
    while not clock.is_finished():
        clock.advance()

    # The final current_time must be exactly the configured end time
    assert clock.current_time == end, (
        f"Clock did not reach exact end time. "
        f"Expected {end}, got {clock.current_time}. "
        f"duration={duration}, step_size={step_size}"
    )
