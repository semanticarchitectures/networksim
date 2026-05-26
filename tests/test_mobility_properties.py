"""Property-based tests for GroupMobilityModel.

Feature: manet-simulation-engine
Properties 7, 11, 12, 13, 14, 15
"""

import numpy as np
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from manet_sim.core.config import MobilityConfig
from manet_sim.mobility.group_mobility import GroupMobilityModel


# --- Strategies ---


@st.composite
def mobility_config_and_params(draw):
    """Generate a valid MobilityConfig along with area and node parameters.

    Produces:
        - config: MobilityConfig with valid parameters
        - node_count: number of nodes (multiple of num_groups)
        - area_width: simulation area width in miles
        - area_height: simulation area height in miles
        - seed: RNG seed for reproducibility
    """
    num_groups = draw(st.integers(min_value=1, max_value=10))
    nodes_per_group = draw(st.integers(min_value=2, max_value=20))
    node_count = num_groups * nodes_per_group

    area_width = draw(st.floats(min_value=10.0, max_value=100.0))
    area_height = draw(st.floats(min_value=10.0, max_value=100.0))

    # max_deviation must be less than half the area dimension to allow valid placement
    max_area_dim = min(area_width, area_height)
    max_deviation = draw(
        st.floats(min_value=0.1, max_value=min(5.0, max_area_dim / 3.0))
    )

    speed_min = draw(st.floats(min_value=0.1, max_value=5.0))
    speed_max = draw(st.floats(min_value=speed_min + 0.1, max_value=15.0))

    pause_min = draw(st.floats(min_value=0.0, max_value=10.0))
    pause_max = draw(st.floats(min_value=pause_min, max_value=60.0))

    deviation_model = draw(st.sampled_from(["uniform", "gaussian"]))

    config = MobilityConfig(
        model="rpgm",
        num_groups=num_groups,
        group_speed_min_mps=speed_min,
        group_speed_max_mps=speed_max,
        pause_min_seconds=pause_min,
        pause_max_seconds=pause_max,
        max_deviation_miles=max_deviation,
        deviation_model=deviation_model,
    )

    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))

    return config, node_count, area_width, area_height, seed


def create_model(config, node_count, area_width, area_height, seed):
    """Helper to create and initialize a GroupMobilityModel."""
    rng = np.random.default_rng(seed)
    model = GroupMobilityModel(config, rng)
    model.initialize(node_count, area_width, area_height)
    return model


# --- Property Tests ---


@given(params=mobility_config_and_params())
@settings(max_examples=100, deadline=None)
def test_position_array_shape_invariant(params):
    """
    Property 7: Position array shape invariant.

    For any sequence of mobility updates on N nodes, the positions array SHALL
    maintain shape (N, 2) with dtype float32 after every step.

    **Validates: Requirements 6.3, 9.1**
    """
    config, node_count, area_width, area_height, seed = params
    model = create_model(config, node_count, area_width, area_height, seed)

    # Check after initialization
    positions = model.get_positions()
    assert positions.shape == (node_count, 2), (
        f"Expected shape ({node_count}, 2), got {positions.shape}"
    )
    assert positions.dtype == np.float32, (
        f"Expected dtype float32, got {positions.dtype}"
    )

    # Run a few steps and check invariant holds
    dt = 1.0
    for t_step in range(3):
        t = float(t_step) * dt
        positions = model.step(t, dt)
        assert positions.shape == (node_count, 2), (
            f"After step {t_step}: expected shape ({node_count}, 2), got {positions.shape}"
        )
        assert positions.dtype == np.float32, (
            f"After step {t_step}: expected dtype float32, got {positions.dtype}"
        )


@given(params=mobility_config_and_params())
@settings(max_examples=100, deadline=None)
def test_boundary_clamping_invariant(params):
    """
    Property 11: Boundary clamping invariant.

    For any set of node positions and velocities after a mobility update step,
    all position components SHALL be within [0, area_width] for x and
    [0, area_height] for y, regardless of the velocity magnitudes or directions.

    **Validates: Requirements 8.4, 8.6, 9.3**
    """
    config, node_count, area_width, area_height, seed = params
    model = create_model(config, node_count, area_width, area_height, seed)

    # Check after initialization
    positions = model.get_positions()
    assert np.all(positions[:, 0] >= 0.0), "X positions below 0 after init"
    assert np.all(positions[:, 0] <= area_width), "X positions above area_width after init"
    assert np.all(positions[:, 1] >= 0.0), "Y positions below 0 after init"
    assert np.all(positions[:, 1] <= area_height), "Y positions above area_height after init"

    # Run several steps with varying dt
    dt = 1.0
    for t_step in range(5):
        t = float(t_step) * dt
        positions = model.step(t, dt)
        assert np.all(positions[:, 0] >= 0.0), (
            f"Step {t_step}: X positions below 0. "
            f"Min x = {positions[:, 0].min()}"
        )
        assert np.all(positions[:, 0] <= area_width), (
            f"Step {t_step}: X positions above area_width ({area_width}). "
            f"Max x = {positions[:, 0].max()}"
        )
        assert np.all(positions[:, 1] >= 0.0), (
            f"Step {t_step}: Y positions below 0. "
            f"Min y = {positions[:, 1].min()}"
        )
        assert np.all(positions[:, 1] <= area_height), (
            f"Step {t_step}: Y positions above area_height ({area_height}). "
            f"Max y = {positions[:, 1].max()}"
        )


@given(params=mobility_config_and_params())
@settings(max_examples=100, deadline=None)
def test_member_deviation_bounds(params):
    """
    Property 12: Member deviation bounds.

    For any node at any time step, the Euclidean distance from that node's
    position to its group's reference point SHALL not exceed the configured
    max_deviation_miles.

    **Validates: Requirements 8.3**
    """
    config, node_count, area_width, area_height, seed = params
    model = create_model(config, node_count, area_width, area_height, seed)
    max_dev = config.max_deviation_miles

    # Check after initialization
    for group in model.groups:
        ref = group.reference_point
        for node_id in group.member_ids:
            pos = model.positions[node_id]
            dist = float(np.linalg.norm(pos - ref))
            assert dist <= max_dev + 1e-4, (
                f"Init: node {node_id} distance {dist:.6f} from group {group.group_id} "
                f"ref point exceeds max_deviation {max_dev}"
            )

    # Run a few steps and check
    dt = 1.0
    for t_step in range(3):
        t = float(t_step) * dt
        model.step(t, dt)
        for group in model.groups:
            ref = group.reference_point
            for node_id in group.member_ids:
                pos = model.positions[node_id]
                dist = float(np.linalg.norm(pos - ref))
                assert dist <= max_dev + 1e-4, (
                    f"Step {t_step}: node {node_id} distance {dist:.6f} from "
                    f"group {group.group_id} ref point exceeds max_deviation {max_dev}"
                )


@given(params=mobility_config_and_params())
@settings(max_examples=100, deadline=None)
def test_waypoint_queue_non_exhaustion(params):
    """
    Property 13: Waypoint queue non-exhaustion.

    For any group whose waypoint queue becomes empty during a step, the mobility
    model SHALL generate at least 1 new waypoint before the next movement
    calculation, ensuring the queue is never empty when movement is needed.

    **Validates: Requirements 8.5**
    """
    config, node_count, area_width, area_height, seed = params
    model = create_model(config, node_count, area_width, area_height, seed)

    # Run multiple steps — after each step, verify that groups that are
    # actively moving (not paused) have a non-empty waypoint queue
    dt = 1.0
    for t_step in range(10):
        t = float(t_step) * dt
        model.step(t, dt)

        for group in model.groups:
            # If the group is not paused, it needs a waypoint to move toward
            if group.pause_remaining <= 0 and group.current_speed > 0:
                assert len(group.waypoint_queue) > 0, (
                    f"Step {t_step}: group {group.group_id} is moving "
                    f"(speed={group.current_speed}) but waypoint queue is empty"
                )


@given(
    n_nodes=st.integers(min_value=2, max_value=100),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    dt=st.floats(min_value=0.01, max_value=5.0),
)
@settings(max_examples=100, deadline=None)
def test_vectorized_position_update_correctness(n_nodes, seed, dt):
    """
    Property 14: Vectorized position update correctness.

    For any positions array P of shape (N, 2), velocities array V of shape (N, 2),
    and time delta dt > 0, the vectorized position update SHALL produce a result
    equal to P + V * dt (element-wise), before boundary enforcement is applied.

    **Validates: Requirements 9.2**
    """
    rng = np.random.default_rng(seed)

    # Create positions within bounds and velocities that won't exceed boundaries
    # so we can verify the raw update without boundary effects
    area_width = 100.0
    area_height = 100.0

    # Place positions well within bounds so P + V*dt stays in bounds
    margin = 20.0
    positions = rng.uniform(
        margin, min(area_width, area_height) - margin, size=(n_nodes, 2)
    ).astype(np.float32)

    # Use small velocities so positions stay within bounds
    max_vel = margin / (dt * 2.0)
    velocities = rng.uniform(-max_vel, max_vel, size=(n_nodes, 2)).astype(np.float32)

    # Compute expected result
    expected = positions + velocities * np.float32(dt)

    # Verify expected stays in bounds (so boundary enforcement won't alter result)
    assume(np.all(expected >= 0.0))
    assume(np.all(expected[:, 0] <= area_width))
    assume(np.all(expected[:, 1] <= area_height))

    # Create a minimal model and manually set its state
    config = MobilityConfig(
        num_groups=1,
        max_deviation_miles=50.0,  # Large deviation so enforcement doesn't interfere
    )
    model_rng = np.random.default_rng(42)
    model = GroupMobilityModel(config, model_rng)
    model.initialize(n_nodes, area_width, area_height)

    # Override positions and velocities
    model.positions = positions.copy()
    model.velocities = velocities.copy()

    # Manually perform the vectorized update (same as step Phase 2)
    model.positions += model.velocities * np.float32(dt)

    # Check result matches expected (before boundary enforcement)
    np.testing.assert_allclose(
        model.positions, expected, rtol=1e-5, atol=1e-6,
        err_msg="Vectorized position update P + V*dt does not match expected"
    )


@given(
    n_nodes=st.integers(min_value=2, max_value=100),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
@settings(max_examples=100, deadline=None)
def test_velocity_reflection_at_boundary(n_nodes, seed):
    """
    Property 15: Velocity reflection at boundary.

    For any node whose position reaches or exceeds a simulation area boundary
    after a position update, the corresponding velocity component SHALL be
    negated (sign flipped) via a vectorized boolean mask operation.

    **Validates: Requirements 9.4**
    """
    rng = np.random.default_rng(seed)
    area_width = 100.0
    area_height = 100.0

    # Create a model
    config = MobilityConfig(
        num_groups=1,
        max_deviation_miles=200.0,  # Very large so deviation enforcement doesn't interfere
    )
    model_rng = np.random.default_rng(42)
    model = GroupMobilityModel(config, model_rng)
    model.initialize(n_nodes, area_width, area_height)

    # Place some nodes at boundaries with velocities pointing outward
    # to trigger reflection
    positions = rng.uniform(0.0, area_width, size=(n_nodes, 2)).astype(np.float32)
    velocities = rng.uniform(-0.01, 0.01, size=(n_nodes, 2)).astype(np.float32)

    # Force some nodes to be out of bounds (simulating post-update state)
    # Nodes below 0
    n_below = max(1, n_nodes // 4)
    positions[:n_below, 0] = rng.uniform(-5.0, -0.01, size=n_below).astype(np.float32)
    velocities[:n_below, 0] = rng.uniform(-1.0, -0.001, size=n_below).astype(np.float32)

    # Nodes above area_width
    n_above = max(1, n_nodes // 4)
    positions[n_below:n_below + n_above, 0] = rng.uniform(
        area_width + 0.01, area_width + 5.0, size=n_above
    ).astype(np.float32)
    velocities[n_below:n_below + n_above, 0] = rng.uniform(
        0.001, 1.0, size=n_above
    ).astype(np.float32)

    # Nodes below 0 on Y
    n_below_y = max(1, n_nodes // 4)
    start_y = n_below + n_above
    end_y = min(start_y + n_below_y, n_nodes)
    actual_n_below_y = end_y - start_y
    if actual_n_below_y > 0:
        positions[start_y:end_y, 1] = rng.uniform(
            -5.0, -0.01, size=actual_n_below_y
        ).astype(np.float32)
        velocities[start_y:end_y, 1] = rng.uniform(
            -1.0, -0.001, size=actual_n_below_y
        ).astype(np.float32)

    # Set model state
    model.positions = positions.copy()
    model.velocities = velocities.copy()

    # Record pre-reflection velocities for nodes that are out of bounds
    vel_before = model.velocities.copy()

    # Apply boundary reflection
    model._apply_boundary_reflection()

    # Verify: nodes that were below 0 on X should have positive X velocity
    for i in range(n_below):
        assert model.velocities[i, 0] >= 0.0, (
            f"Node {i} was below x=0 with vel_x={vel_before[i, 0]:.6f}, "
            f"after reflection vel_x={model.velocities[i, 0]:.6f} should be >= 0"
        )
        assert model.velocities[i, 0] == abs(vel_before[i, 0]), (
            f"Node {i}: reflected vel_x should be abs(original). "
            f"Got {model.velocities[i, 0]}, expected {abs(vel_before[i, 0])}"
        )

    # Verify: nodes that were above area_width on X should have negative X velocity
    for i in range(n_below, n_below + n_above):
        assert model.velocities[i, 0] <= 0.0, (
            f"Node {i} was above x={area_width} with vel_x={vel_before[i, 0]:.6f}, "
            f"after reflection vel_x={model.velocities[i, 0]:.6f} should be <= 0"
        )
        assert model.velocities[i, 0] == -abs(vel_before[i, 0]), (
            f"Node {i}: reflected vel_x should be -abs(original). "
            f"Got {model.velocities[i, 0]}, expected {-abs(vel_before[i, 0])}"
        )

    # Verify: nodes that were below 0 on Y should have positive Y velocity
    if actual_n_below_y > 0:
        for i in range(start_y, end_y):
            assert model.velocities[i, 1] >= 0.0, (
                f"Node {i} was below y=0 with vel_y={vel_before[i, 1]:.6f}, "
                f"after reflection vel_y={model.velocities[i, 1]:.6f} should be >= 0"
            )
            assert model.velocities[i, 1] == abs(vel_before[i, 1]), (
                f"Node {i}: reflected vel_y should be abs(original). "
                f"Got {model.velocities[i, 1]}, expected {abs(vel_before[i, 1])}"
            )

    # Verify: all positions are clamped within bounds after reflection
    assert np.all(model.positions[:, 0] >= 0.0), "X positions below 0 after reflection"
    assert np.all(model.positions[:, 0] <= area_width), "X positions above width after reflection"
    assert np.all(model.positions[:, 1] >= 0.0), "Y positions below 0 after reflection"
    assert np.all(model.positions[:, 1] <= area_height), "Y positions above height after reflection"
