"""Property-based tests for steering file serialization round-trip and printer determinism.

Feature: manet-simulation-engine
Property 22: Steering file serialization round-trip
Property 23: Pretty-printer determinism
"""

import json
import tempfile
from pathlib import Path

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from manet_sim.output.steering_parser import (
    LinkEvent,
    SimulationState,
    SteeringParser,
    SteeringPrinter,
)


# --- Strategies ---


@st.composite
def link_events(draw, timestamp: float = 0.0):
    """Generate a valid LinkEvent."""
    node_a = draw(st.integers(min_value=0, max_value=999))
    node_b = draw(
        st.integers(min_value=0, max_value=999).filter(lambda x: x != node_a)
    )
    distance = draw(
        st.floats(min_value=0.01, max_value=50.0, allow_nan=False, allow_infinity=False)
    )
    return LinkEvent(
        node_a=min(node_a, node_b),
        node_b=max(node_a, node_b),
        distance=distance,
        timestamp=timestamp,
    )


@st.composite
def simulation_states(draw):
    """Generate a valid SimulationState with realistic values.

    Constrains:
    - node_count between 1 and 50 (keep tests fast)
    - positions in [0, 100] miles
    - velocities in [-1, 1] miles/second
    - group_ids in [0, 19]
    - timestamp non-negative
    - link events reference valid node IDs
    """
    node_count = draw(st.integers(min_value=1, max_value=50))
    timestamp = draw(
        st.floats(min_value=0.0, max_value=3600.0, allow_nan=False, allow_infinity=False)
    )

    # Generate positions in [0, 100]
    positions = draw(
        st.lists(
            st.tuples(
                st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
                st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
            ),
            min_size=node_count,
            max_size=node_count,
        )
    )

    # Generate velocities in [-1, 1]
    velocities = draw(
        st.lists(
            st.tuples(
                st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            ),
            min_size=node_count,
            max_size=node_count,
        )
    )

    # Generate group_ids in [0, 19]
    group_ids = draw(
        st.lists(
            st.integers(min_value=0, max_value=19),
            min_size=node_count,
            max_size=node_count,
        )
    )

    # Generate link events (formed)
    num_formed = draw(st.integers(min_value=0, max_value=min(5, node_count // 2)))
    links_formed = []
    used_pairs = set()
    for _ in range(num_formed):
        le = draw(link_events(timestamp=timestamp))
        # Ensure node IDs are within range and pair is unique
        a = le.node_a % node_count
        b = le.node_b % node_count
        if a == b:
            b = (b + 1) % node_count
        pair = (min(a, b), max(a, b))
        if pair not in used_pairs:
            used_pairs.add(pair)
            links_formed.append(
                LinkEvent(node_a=pair[0], node_b=pair[1],
                          distance=le.distance, timestamp=timestamp)
            )

    # Generate link events (broken)
    num_broken = draw(st.integers(min_value=0, max_value=min(5, node_count // 2)))
    links_broken = []
    for _ in range(num_broken):
        le = draw(link_events(timestamp=timestamp))
        a = le.node_a % node_count
        b = le.node_b % node_count
        if a == b:
            b = (b + 1) % node_count
        pair = (min(a, b), max(a, b))
        if pair not in used_pairs:
            used_pairs.add(pair)
            links_broken.append(
                LinkEvent(node_a=pair[0], node_b=pair[1],
                          distance=le.distance, timestamp=timestamp)
            )

    return SimulationState(
        timestamp=timestamp,
        positions=np.array(positions, dtype=np.float64),
        velocities=np.array(velocities, dtype=np.float64),
        group_ids=np.array(group_ids, dtype=np.int32),
        links_formed=links_formed,
        links_broken=links_broken,
        node_count=node_count,
    )


# --- Property Tests ---


@given(state=simulation_states())
@settings(max_examples=100, deadline=None)
def test_steering_file_serialization_round_trip(state, tmp_path_factory):
    """
    Property 22: Steering file serialization round-trip.

    For any valid SimulationState, printing to JSON then parsing back
    produces a state that is field-by-field equal within 1e-6 tolerance.

    **Validates: Requirements 14.4, 14.5**
    """
    printer = SteeringPrinter()
    parser = SteeringParser()

    # Print state to JSON string
    json_str = printer.format(state)

    # Wrap in a valid steering file structure for parsing
    step_data = json.loads(json_str)
    steering_data = {
        "metadata": {
            "schema_version": "1.0.0",
            "node_count": state.node_count,
        },
        "steps": [step_data],
        "snapshots": [],
    }

    # Write to a temporary file
    tmp_dir = tmp_path_factory.mktemp("steering")
    filepath = str(tmp_dir / "roundtrip.json")
    with open(filepath, "w") as f:
        json.dump(steering_data, f)

    # Parse back using the rounded timestamp (as stored in the file)
    # The printer rounds timestamp to 6 decimal places, so we must
    # use the rounded value for lookup.
    rounded_timestamp = round(float(state.timestamp), 6)
    parsed_state = parser.parse(filepath, rounded_timestamp)

    # Verify field-by-field equality within tolerance
    assert abs(parsed_state.timestamp - state.timestamp) < 1e-6, (
        f"Timestamp mismatch: {parsed_state.timestamp} vs {state.timestamp}"
    )
    assert parsed_state.node_count == state.node_count, (
        f"Node count mismatch: {parsed_state.node_count} vs {state.node_count}"
    )

    # Positions within 1e-6 tolerance
    assert np.allclose(parsed_state.positions, state.positions, atol=1e-6), (
        f"Positions mismatch:\n  parsed={parsed_state.positions}\n  original={state.positions}"
    )

    # Velocities within 1e-6 tolerance
    assert np.allclose(parsed_state.velocities, state.velocities, atol=1e-6), (
        f"Velocities mismatch:\n  parsed={parsed_state.velocities}\n  original={state.velocities}"
    )

    # Group IDs exact match
    assert np.array_equal(parsed_state.group_ids, state.group_ids), (
        f"Group IDs mismatch:\n  parsed={parsed_state.group_ids}\n  original={state.group_ids}"
    )

    # Links formed count and content
    assert len(parsed_state.links_formed) == len(state.links_formed), (
        f"Links formed count mismatch: {len(parsed_state.links_formed)} vs {len(state.links_formed)}"
    )
    for i, (parsed_link, orig_link) in enumerate(
        zip(parsed_state.links_formed, state.links_formed)
    ):
        assert parsed_link.node_a == orig_link.node_a, (
            f"Link formed[{i}] node_a mismatch: {parsed_link.node_a} vs {orig_link.node_a}"
        )
        assert parsed_link.node_b == orig_link.node_b, (
            f"Link formed[{i}] node_b mismatch: {parsed_link.node_b} vs {orig_link.node_b}"
        )
        assert abs(parsed_link.distance - orig_link.distance) < 1e-6, (
            f"Link formed[{i}] distance mismatch: {parsed_link.distance} vs {orig_link.distance}"
        )
        assert abs(parsed_link.timestamp - orig_link.timestamp) < 1e-6, (
            f"Link formed[{i}] timestamp mismatch: {parsed_link.timestamp} vs {orig_link.timestamp}"
        )

    # Links broken count and content
    assert len(parsed_state.links_broken) == len(state.links_broken), (
        f"Links broken count mismatch: {len(parsed_state.links_broken)} vs {len(state.links_broken)}"
    )
    for i, (parsed_link, orig_link) in enumerate(
        zip(parsed_state.links_broken, state.links_broken)
    ):
        assert parsed_link.node_a == orig_link.node_a, (
            f"Link broken[{i}] node_a mismatch: {parsed_link.node_a} vs {orig_link.node_a}"
        )
        assert parsed_link.node_b == orig_link.node_b, (
            f"Link broken[{i}] node_b mismatch: {parsed_link.node_b} vs {orig_link.node_b}"
        )


@given(state=simulation_states())
@settings(max_examples=100, deadline=None)
def test_pretty_printer_determinism(state):
    """
    Property 23: Pretty-printer determinism.

    For any valid SimulationState, calling format() twice produces
    identical output strings.

    **Validates: Requirements 14.4, 14.5**
    """
    printer = SteeringPrinter()

    result1 = printer.format(state)
    result2 = printer.format(state)

    assert result1 == result2, (
        f"Non-deterministic output detected.\n"
        f"First call length: {len(result1)}\n"
        f"Second call length: {len(result2)}\n"
        f"First 200 chars diff:\n  result1={result1[:200]}\n  result2={result2[:200]}"
    )

    # Additionally verify the output is valid JSON
    parsed = json.loads(result1)
    assert isinstance(parsed, dict)

    # Verify sorted keys at top level
    keys = list(parsed.keys())
    assert keys == sorted(keys), f"Top-level keys not sorted: {keys}"
