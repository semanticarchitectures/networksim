"""Property-based tests for node serialization.

Feature: manet-simulation-engine, Property 6: Node serialization completeness
"""

from hypothesis import given, settings
from hypothesis import strategies as st
import math

from manet_sim.output.steering_writer import _serialize_node


# Required keys that must be present in the serialized dict
REQUIRED_KEYS = {"node_id", "group_id", "x", "y", "vx", "vy", "transmission_range", "active"}


def _count_decimal_places(value: float) -> int:
    """Count the number of decimal places in a float's string representation.

    Returns the number of digits after the decimal point, or 0 if there is
    no decimal point.
    """
    s = str(value)
    if "." not in s:
        return 0
    # Handle scientific notation
    if "e" in s or "E" in s:
        # Convert to decimal representation
        s = f"{value:.20f}".rstrip("0")
        if "." not in s:
            return 0
    decimal_part = s.split(".")[1]
    return len(decimal_part)


@st.composite
def valid_node_state(draw):
    """Generate a valid node state for serialization.

    Covers:
    - node_id: non-negative integers
    - group_id: non-negative integers (typically 0-19 for 20 groups)
    - x, y: positions in [0, 100] miles
    - vx, vy: velocities (realistic range for MANET simulation)
    - transmission_range: positive float
    - active: boolean
    """
    node_id = draw(st.integers(min_value=0, max_value=9999))
    group_id = draw(st.integers(min_value=0, max_value=99))
    x = draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False))
    y = draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False))
    vx = draw(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    vy = draw(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    transmission_range = draw(st.floats(min_value=0.1, max_value=50.0, allow_nan=False, allow_infinity=False))
    active = draw(st.booleans())
    return node_id, group_id, x, y, vx, vy, transmission_range, active


@given(state=valid_node_state())
@settings(max_examples=100, deadline=None)
def test_node_serialization_completeness(state):
    """
    Property 6: Node serialization completeness.

    For any valid node state (position in [0,100]², velocity, group_id,
    transmission_range, active status), serializing to a dictionary SHALL
    produce a dict containing keys node_id, group_id, x, y, vx, vy,
    transmission_range, and active, with numeric values rounded to no more
    than 6 decimal places.

    **Validates: Requirements 6.2**
    """
    node_id, group_id, x, y, vx, vy, transmission_range, active = state

    result = _serialize_node(
        node_id=node_id,
        group_id=group_id,
        x=x,
        y=y,
        vx=vx,
        vy=vy,
        transmission_range=transmission_range,
        active=active,
    )

    # 1. Result must be a dict
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"

    # 2. All required keys must be present
    assert set(result.keys()) == REQUIRED_KEYS, (
        f"Missing or extra keys. Expected {REQUIRED_KEYS}, got {set(result.keys())}"
    )

    # 3. Verify correct types
    assert isinstance(result["node_id"], int), f"node_id should be int, got {type(result['node_id'])}"
    assert isinstance(result["group_id"], int), f"group_id should be int, got {type(result['group_id'])}"
    assert isinstance(result["x"], float), f"x should be float, got {type(result['x'])}"
    assert isinstance(result["y"], float), f"y should be float, got {type(result['y'])}"
    assert isinstance(result["vx"], float), f"vx should be float, got {type(result['vx'])}"
    assert isinstance(result["vy"], float), f"vy should be float, got {type(result['vy'])}"
    assert isinstance(result["transmission_range"], float), (
        f"transmission_range should be float, got {type(result['transmission_range'])}"
    )
    assert isinstance(result["active"], bool), f"active should be bool, got {type(result['active'])}"

    # 4. Numeric values must have no more than 6 decimal places
    numeric_keys = ["x", "y", "vx", "vy", "transmission_range"]
    for key in numeric_keys:
        value = result[key]
        # Verify rounding: round(value, 6) should equal itself
        assert value == round(value, 6), (
            f"Key '{key}' has more than 6 decimal places: {value}"
        )

    # 5. Verify values are preserved correctly
    assert result["node_id"] == node_id
    assert result["group_id"] == group_id
    assert result["active"] == active
    assert result["x"] == round(float(x), 6)
    assert result["y"] == round(float(y), 6)
    assert result["vx"] == round(float(vx), 6)
    assert result["vy"] == round(float(vy), 6)
    assert result["transmission_range"] == round(float(transmission_range), 6)
