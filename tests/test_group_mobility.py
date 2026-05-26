"""Unit tests for GroupMobilityModel (RPGM implementation)."""

import numpy as np
import pytest

from manet_sim.core.config import MobilityConfig
from manet_sim.mobility.group_mobility import GroupMobilityModel, METERS_PER_MILE


@pytest.fixture
def default_config():
    """Return default MobilityConfig."""
    return MobilityConfig()


@pytest.fixture
def rng():
    """Return a seeded RNG for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def model(default_config, rng):
    """Return an initialized GroupMobilityModel with 1000 nodes."""
    m = GroupMobilityModel(default_config, rng)
    m.initialize(1000, 100.0, 100.0)
    return m


class TestInitialization:
    """Tests for GroupMobilityModel.initialize()."""

    def test_positions_shape_and_dtype(self, model):
        """Positions array has shape (1000, 2) and dtype float32."""
        assert model.positions.shape == (1000, 2)
        assert model.positions.dtype == np.float32

    def test_velocities_shape_and_dtype(self, model):
        """Velocities array has shape (1000, 2) and dtype float32."""
        assert model.velocities.shape == (1000, 2)
        assert model.velocities.dtype == np.float32

    def test_group_ids_shape_and_dtype(self, model):
        """Group IDs array has shape (1000,) and dtype int32."""
        assert model.group_ids.shape == (1000,)
        assert model.group_ids.dtype == np.int32

    def test_creates_20_groups(self, model):
        """Initializes exactly 20 groups."""
        assert len(model.groups) == 20

    def test_50_nodes_per_group(self, model):
        """Each group has exactly 50 members."""
        for group in model.groups:
            assert group.size == 50

    def test_all_nodes_assigned_to_groups(self, model):
        """All 1000 nodes are assigned to a group."""
        all_members = set()
        for group in model.groups:
            all_members.update(group.member_ids)
        assert len(all_members) == 1000
        assert all_members == set(range(1000))

    def test_group_ids_match_membership(self, model):
        """group_ids array matches actual group membership."""
        for group in model.groups:
            for node_id in group.member_ids:
                assert model.group_ids[node_id] == group.group_id

    def test_positions_within_bounds(self, model):
        """All initial positions are within [0, 100] x [0, 100]."""
        assert np.all(model.positions >= 0.0)
        assert np.all(model.positions[:, 0] <= 100.0)
        assert np.all(model.positions[:, 1] <= 100.0)

    def test_initial_deviation_within_bounds(self, model):
        """All nodes are within max_deviation_miles of their group reference point."""
        max_dev = model._config.max_deviation_miles
        for group in model.groups:
            ref = group.reference_point
            for node_id in group.member_ids:
                pos = model.positions[node_id]
                dist = np.linalg.norm(pos - ref)
                assert dist <= max_dev + 1e-5, (
                    f"Node {node_id} is {dist:.4f} miles from group {group.group_id} "
                    f"reference point (max: {max_dev})"
                )

    def test_waypoint_queues_not_empty(self, model):
        """Each group has at least one waypoint after initialization."""
        for group in model.groups:
            assert len(group.waypoint_queue) >= 1

    def test_reference_points_within_bounds(self, model):
        """All group reference points are within the simulation area."""
        for group in model.groups:
            ref = group.reference_point
            assert 0.0 <= ref[0] <= 100.0
            assert 0.0 <= ref[1] <= 100.0

    def test_each_group_has_leader(self, model):
        """Each group has a designated leader."""
        for group in model.groups:
            assert group.leader_id is not None
            assert group.leader_id in group.member_ids


class TestStep:
    """Tests for GroupMobilityModel.step()."""

    def test_step_returns_positions(self, model):
        """step() returns positions array of shape (N, 2)."""
        positions = model.step(0.0, 1.0)
        assert positions.shape == (1000, 2)
        assert positions.dtype == np.float32

    def test_positions_within_bounds_after_step(self, model):
        """All positions remain within bounds after a step."""
        model.step(0.0, 1.0)
        assert np.all(model.positions >= 0.0)
        assert np.all(model.positions[:, 0] <= 100.0)
        assert np.all(model.positions[:, 1] <= 100.0)

    def test_positions_change_after_step(self, model):
        """Positions change after a step (nodes are moving)."""
        initial_positions = model.positions.copy()
        model.step(0.0, 1.0)
        # At least some positions should change (not all groups are paused initially)
        assert not np.allclose(model.positions, initial_positions)

    def test_deviation_bounds_maintained_after_steps(self, model):
        """Deviation bounds are maintained after multiple steps."""
        max_dev = model._config.max_deviation_miles
        for step in range(50):
            model.step(float(step), 1.0)

        for group in model.groups:
            ref = group.reference_point
            for node_id in group.member_ids:
                pos = model.positions[node_id]
                dist = np.linalg.norm(pos - ref)
                assert dist <= max_dev + 1e-5, (
                    f"Node {node_id} at step 50 is {dist:.4f} miles from "
                    f"group {group.group_id} ref (max: {max_dev})"
                )

    def test_boundary_clamping_after_many_steps(self, model):
        """Positions remain within bounds after many steps."""
        for step in range(200):
            model.step(float(step), 1.0)
        assert np.all(model.positions >= 0.0)
        assert np.all(model.positions[:, 0] <= 100.0)
        assert np.all(model.positions[:, 1] <= 100.0)

    def test_get_positions_returns_same_array(self, model):
        """get_positions() returns the same array as step()."""
        positions = model.step(0.0, 1.0)
        assert positions is model.get_positions()

    def test_get_velocities(self, model):
        """get_velocities() returns velocities array."""
        model.step(0.0, 1.0)
        velocities = model.get_velocities()
        assert velocities.shape == (1000, 2)
        assert velocities.dtype == np.float32

    def test_get_group_ids(self, model):
        """get_group_ids() returns group IDs array."""
        group_ids = model.get_group_ids()
        assert group_ids.shape == (1000,)
        assert group_ids.dtype == np.int32


class TestVelocityReflection:
    """Tests for boundary velocity reflection."""

    def test_velocity_negated_at_boundary(self):
        """Velocity component is negated when position hits boundary."""
        config = MobilityConfig()
        rng = np.random.default_rng(123)
        model = GroupMobilityModel(config, rng)
        model.initialize(1000, 100.0, 100.0)

        # Manually place a node near the right boundary with positive x-velocity
        node_id = 0
        model.positions[node_id] = np.array([99.999, 50.0], dtype=np.float32)
        model.velocities[node_id] = np.array([0.01, 0.0], dtype=np.float32)

        # Step should push it past boundary, then reflect
        model.step(0.0, 1.0)

        # After reflection, x-velocity should be negative (or zero if clamped)
        assert model.velocities[node_id, 0] <= 0.0
        # Position should be clamped to boundary
        assert model.positions[node_id, 0] <= 100.0

    def test_velocity_negated_at_lower_boundary(self):
        """Velocity component is negated when position hits lower boundary."""
        config = MobilityConfig()
        rng = np.random.default_rng(456)
        model = GroupMobilityModel(config, rng)
        model.initialize(1000, 100.0, 100.0)

        # Place a node near the lower boundary with negative y-velocity
        node_id = 0
        model.positions[node_id] = np.array([50.0, 0.001], dtype=np.float32)
        model.velocities[node_id] = np.array([0.0, -0.01], dtype=np.float32)

        model.step(0.0, 1.0)

        # After reflection, y-velocity should be positive
        assert model.velocities[node_id, 1] >= 0.0
        # Position should be clamped to boundary
        assert model.positions[node_id, 1] >= 0.0


class TestWaypointGeneration:
    """Tests for waypoint queue management."""

    def test_waypoint_generated_when_queue_exhausts(self):
        """New waypoint is generated when queue becomes empty."""
        config = MobilityConfig(pause_min_seconds=0.0, pause_max_seconds=0.0)
        rng = np.random.default_rng(789)
        model = GroupMobilityModel(config, rng)
        model.initialize(1000, 100.0, 100.0)

        # Run many steps to exhaust initial waypoints
        for step in range(500):
            model.step(float(step), 1.0)
            # After each step, no group should have an empty queue
            # (unless it's paused, but pause is 0 here)
            for group in model.groups:
                # Either paused or has waypoints
                assert (
                    group.pause_remaining > 0 or len(group.waypoint_queue) > 0
                ), f"Group {group.group_id} has empty queue and is not paused"

    def test_waypoints_within_area_bounds(self, model):
        """Generated waypoints are within the simulation area."""
        for group in model.groups:
            for wp in group.waypoint_queue:
                assert 0.0 <= wp[0] <= 100.0
                assert 0.0 <= wp[1] <= 100.0


class TestSpeedConversion:
    """Tests for speed unit conversion (m/s to miles/s)."""

    def test_speed_conversion(self):
        """Speeds are correctly converted from m/s to miles/s."""
        config = MobilityConfig(
            group_speed_min_mps=0.5, group_speed_max_mps=13.4
        )
        rng = np.random.default_rng(42)
        model = GroupMobilityModel(config, rng)

        expected_min = 0.5 / METERS_PER_MILE
        expected_max = 13.4 / METERS_PER_MILE

        assert abs(model._speed_min - expected_min) < 1e-10
        assert abs(model._speed_max - expected_max) < 1e-10

    def test_group_speeds_within_converted_range(self, model):
        """Group speeds are within the converted range."""
        speed_min = model._speed_min
        speed_max = model._speed_max

        for group in model.groups:
            if group.current_speed > 0:
                assert speed_min <= group.current_speed <= speed_max, (
                    f"Group {group.group_id} speed {group.current_speed} "
                    f"outside [{speed_min}, {speed_max}]"
                )


class TestReproducibility:
    """Tests for deterministic behavior with same seed."""

    def test_same_seed_same_positions(self):
        """Same seed produces identical positions."""
        config = MobilityConfig()

        rng1 = np.random.default_rng(42)
        model1 = GroupMobilityModel(config, rng1)
        model1.initialize(1000, 100.0, 100.0)
        for i in range(10):
            model1.step(float(i), 1.0)

        rng2 = np.random.default_rng(42)
        model2 = GroupMobilityModel(config, rng2)
        model2.initialize(1000, 100.0, 100.0)
        for i in range(10):
            model2.step(float(i), 1.0)

        np.testing.assert_array_equal(model1.positions, model2.positions)
        np.testing.assert_array_equal(model1.velocities, model2.velocities)

    def test_different_seed_different_positions(self):
        """Different seeds produce different positions."""
        config = MobilityConfig()

        rng1 = np.random.default_rng(42)
        model1 = GroupMobilityModel(config, rng1)
        model1.initialize(1000, 100.0, 100.0)

        rng2 = np.random.default_rng(99)
        model2 = GroupMobilityModel(config, rng2)
        model2.initialize(1000, 100.0, 100.0)

        assert not np.allclose(model1.positions, model2.positions)


class TestGaussianDeviation:
    """Tests for Gaussian deviation model."""

    def test_gaussian_deviation_within_bounds(self):
        """Gaussian deviation model respects max_deviation_miles."""
        config = MobilityConfig(deviation_model="gaussian")
        rng = np.random.default_rng(42)
        model = GroupMobilityModel(config, rng)
        model.initialize(1000, 100.0, 100.0)

        max_dev = config.max_deviation_miles
        for group in model.groups:
            ref = group.reference_point
            for node_id in group.member_ids:
                pos = model.positions[node_id]
                dist = np.linalg.norm(pos - ref)
                assert dist <= max_dev + 1e-5

    def test_gaussian_deviation_after_steps(self):
        """Gaussian deviation stays bounded after multiple steps."""
        config = MobilityConfig(deviation_model="gaussian")
        rng = np.random.default_rng(42)
        model = GroupMobilityModel(config, rng)
        model.initialize(1000, 100.0, 100.0)

        max_dev = config.max_deviation_miles
        for step in range(50):
            model.step(float(step), 1.0)

        for group in model.groups:
            ref = group.reference_point
            for node_id in group.member_ids:
                pos = model.positions[node_id]
                dist = np.linalg.norm(pos - ref)
                assert dist <= max_dev + 1e-5
