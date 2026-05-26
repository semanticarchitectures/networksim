"""Property-based tests for LinkManager hysteresis.

Feature: manet-simulation-engine
- Property 17: Hysteresis link formation
- Property 18: Hysteresis link teardown
- Property 19: Hysteresis stability in buffer zone
"""

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from manet_sim.topology.link_manager import LinkManager, LinkState


# --- Strategies ---


@st.composite
def link_manager_config(draw):
    """Generate a valid LinkManager configuration.

    Produces:
        - radio_range: positive float (0.1 to 100.0)
        - hysteresis_pct: non-negative float (0.01 to 0.49)
    """
    radio_range = draw(
        st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False)
    )
    hysteresis_pct = draw(
        st.floats(min_value=0.01, max_value=0.49, allow_nan=False, allow_infinity=False)
    )
    return radio_range, hysteresis_pct


@st.composite
def distance_below_inner(draw, radio_range, hysteresis_pct):
    """Generate a distance strictly below the inner threshold."""
    inner = radio_range - radio_range * hysteresis_pct
    # Distance in [0, inner) — strictly less than inner
    assume(inner > 0)
    dist = draw(
        st.floats(min_value=0.0, max_value=inner, allow_nan=False, allow_infinity=False,
                  exclude_max=True)
    )
    return dist


@st.composite
def distance_above_outer(draw, radio_range, hysteresis_pct):
    """Generate a distance strictly above the outer threshold."""
    outer = radio_range + radio_range * hysteresis_pct
    # Distance in (outer, outer * 3] — strictly greater than outer
    dist = draw(
        st.floats(min_value=outer, max_value=outer * 3.0 + 1.0,
                  allow_nan=False, allow_infinity=False, exclude_min=True)
    )
    return dist


@st.composite
def distance_in_buffer_zone(draw, radio_range, hysteresis_pct):
    """Generate a distance within the hysteresis buffer zone [inner, outer]."""
    inner = radio_range - radio_range * hysteresis_pct
    outer = radio_range + radio_range * hysteresis_pct
    assume(inner < outer)
    dist = draw(
        st.floats(min_value=inner, max_value=outer, allow_nan=False, allow_infinity=False)
    )
    return dist


# --- Property Tests ---


@given(data=st.data(), config=link_manager_config())
@settings(max_examples=100, deadline=None)
def test_hysteresis_link_formation(data, config):
    """
    Property 17: Hysteresis link formation.

    For any pair of nodes in ABSENT link state whose distance falls below the
    inner threshold (radio_range - hysteresis_margin), the link state SHALL
    transition to ACTIVE.

    **Validates: Requirements 12.1**
    """
    radio_range, hysteresis_pct = config
    lm = LinkManager(radio_range=radio_range, hysteresis_pct=hysteresis_pct)

    inner = lm.inner_threshold
    assume(inner > 0)

    # Generate a distance strictly below the inner threshold
    dist = data.draw(
        st.floats(min_value=0.0, max_value=inner, allow_nan=False,
                  allow_infinity=False, exclude_max=True)
    )

    result = lm.evaluate(LinkState.ABSENT, dist)
    assert result == LinkState.ACTIVE, (
        f"ABSENT link with distance {dist:.6f} < inner_threshold {inner:.6f} "
        f"should transition to ACTIVE, but got {result}. "
        f"radio_range={radio_range:.6f}, hysteresis_pct={hysteresis_pct:.6f}"
    )


@given(data=st.data(), config=link_manager_config())
@settings(max_examples=100, deadline=None)
def test_hysteresis_link_teardown(data, config):
    """
    Property 18: Hysteresis link teardown.

    For any pair of nodes in ACTIVE link state whose distance exceeds the
    outer threshold (radio_range + hysteresis_margin), the link state SHALL
    transition to ABSENT.

    **Validates: Requirements 12.2**
    """
    radio_range, hysteresis_pct = config
    lm = LinkManager(radio_range=radio_range, hysteresis_pct=hysteresis_pct)

    outer = lm.outer_threshold

    # Generate a distance strictly above the outer threshold
    dist = data.draw(
        st.floats(min_value=outer, max_value=outer * 3.0 + 1.0,
                  allow_nan=False, allow_infinity=False, exclude_min=True)
    )

    result = lm.evaluate(LinkState.ACTIVE, dist)
    assert result == LinkState.ABSENT, (
        f"ACTIVE link with distance {dist:.6f} > outer_threshold {outer:.6f} "
        f"should transition to ABSENT, but got {result}. "
        f"radio_range={radio_range:.6f}, hysteresis_pct={hysteresis_pct:.6f}"
    )


@given(data=st.data(), config=link_manager_config())
@settings(max_examples=100, deadline=None)
def test_hysteresis_stability_in_buffer_zone(data, config):
    """
    Property 19: Hysteresis stability in buffer zone.

    For any pair of nodes whose distance is between the inner threshold and
    outer threshold (the hysteresis zone), the link state SHALL remain
    unchanged — ACTIVE pairs stay ACTIVE, ABSENT pairs stay ABSENT.

    **Validates: Requirements 12.3, 12.6**
    """
    radio_range, hysteresis_pct = config
    lm = LinkManager(radio_range=radio_range, hysteresis_pct=hysteresis_pct)

    inner = lm.inner_threshold
    outer = lm.outer_threshold
    assume(inner < outer)

    # Generate a distance within the buffer zone [inner, outer]
    dist = data.draw(
        st.floats(min_value=inner, max_value=outer,
                  allow_nan=False, allow_infinity=False)
    )

    # Choose a starting state
    initial_state = data.draw(st.sampled_from([LinkState.ABSENT, LinkState.ACTIVE]))

    result = lm.evaluate(initial_state, dist)
    assert result == initial_state, (
        f"Link in state {initial_state.name} with distance {dist:.6f} in buffer zone "
        f"[{inner:.6f}, {outer:.6f}] should remain unchanged, but got {result.name}. "
        f"radio_range={radio_range:.6f}, hysteresis_pct={hysteresis_pct:.6f}"
    )
