"""Group mobility model: Group data structure and RPGM implementation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from manet_sim.core.config import MobilityConfig

# Conversion factor: 1 mile = 1609.344 meters
METERS_PER_MILE = 1609.344


@dataclass
class Group:
    """
    Represents a mobility group in the MANET simulation.

    A group maintains a reference point that moves via random waypoint,
    and member nodes deviate from this reference point within bounded offsets.

    Attributes:
        group_id: Unique integer identifier for this group.
        member_ids: Ordered list of node IDs, ordered by join time
                    (first element = longest-tenured member).
        leader_id: The current leader node ID, or None if group is empty.
        reference_point: 2D position (x, y) in miles, shape (2,), float32.
        reference_velocity: 2D velocity vector in miles/second, shape (2,), float32.
        waypoint_queue: FIFO deque of waypoints (np.ndarray shape (2,)),
                        capacity 1-100.
        pause_remaining: Seconds remaining in current pause state.
        current_speed: Current group speed in miles/second.
    """

    group_id: int
    member_ids: list[int] = field(default_factory=list)
    leader_id: Optional[int] = None
    reference_point: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float32)
    )
    reference_velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float32)
    )
    waypoint_queue: deque = field(default_factory=lambda: deque(maxlen=100))
    pause_remaining: float = 0.0
    current_speed: float = 0.0

    def add_member(self, node_id: int) -> bool:
        """
        Add a node to this group.

        Args:
            node_id: The ID of the node to add.

        Returns:
            True if the node was added successfully, False if it was
            already a member (duplicate rejected).
        """
        if node_id in self.member_ids:
            return False
        self.member_ids.append(node_id)
        # If this is the first member, make them the leader
        if self.leader_id is None:
            self.leader_id = node_id
        return True

    def remove_member(self, node_id: int) -> bool:
        """
        Remove a node from this group.

        If the removed node is the leader and other members remain,
        the longest-tenured remaining member (first in the ordered list)
        is promoted to leader.

        If the removed node is the last member, the group becomes empty
        with no leader.

        Args:
            node_id: The ID of the node to remove.

        Returns:
            True if the node was removed, False if it was not a member.
        """
        if node_id not in self.member_ids:
            return False

        self.member_ids.remove(node_id)

        if self.leader_id == node_id:
            # Promote longest-tenured remaining member (first in list)
            if self.member_ids:
                self.leader_id = self.member_ids[0]
            else:
                self.leader_id = None

        return True

    @property
    def size(self) -> int:
        """Return the number of members in this group."""
        return len(self.member_ids)

    @property
    def is_empty(self) -> bool:
        """Return True if the group has no members."""
        return len(self.member_ids) == 0


class GroupMobilityModel:
    """
    Reference Point Group Mobility (RPGM) model with vectorized position updates.

    Organizes nodes into groups. Each group's reference point moves via random
    waypoint navigation. Member nodes are positioned at the reference point plus
    a bounded random deviation. Position updates are vectorized using NumPy.

    Speeds in config are m/s; positions are in miles.
    Conversion: speed_miles_per_sec = speed_mps / 1609.344
    """

    def __init__(self, config: MobilityConfig, rng: np.random.Generator) -> None:
        """
        Initialize the group mobility model.

        Args:
            config: Mobility configuration parameters.
            rng: NumPy random generator for reproducibility.
        """
        self._config = config
        self._rng = rng

        # Convert speeds from m/s to miles/s
        self._speed_min = config.group_speed_min_mps / METERS_PER_MILE
        self._speed_max = config.group_speed_max_mps / METERS_PER_MILE

        # State arrays (initialized in initialize())
        self.positions: np.ndarray = np.empty((0, 2), dtype=np.float32)
        self.velocities: np.ndarray = np.empty((0, 2), dtype=np.float32)
        self.group_ids: np.ndarray = np.empty(0, dtype=np.int32)

        # Group state
        self.groups: list[Group] = []

        # Area bounds
        self._area_width: float = 0.0
        self._area_height: float = 0.0
        self._node_count: int = 0

        # Per-node deviation offsets from reference point (miles)
        self._deviations: np.ndarray = np.empty((0, 2), dtype=np.float32)

    def initialize(
        self, node_count: int, area_width: float, area_height: float
    ) -> None:
        """
        Initialize node positions, groups, and reference points.

        Creates num_groups groups with node_count / num_groups nodes each.
        Places reference points randomly within the area and assigns
        member nodes with random deviations.

        Args:
            node_count: Total number of nodes (e.g. 1000).
            area_width: Simulation area width in miles.
            area_height: Simulation area height in miles.
        """
        self._node_count = node_count
        self._area_width = area_width
        self._area_height = area_height

        num_groups = self._config.num_groups
        nodes_per_group = node_count // num_groups

        # Initialize arrays
        self.positions = np.zeros((node_count, 2), dtype=np.float32)
        self.velocities = np.zeros((node_count, 2), dtype=np.float32)
        self.group_ids = np.zeros(node_count, dtype=np.int32)
        self._deviations = np.zeros((node_count, 2), dtype=np.float32)

        self.groups = []
        node_idx = 0

        for gid in range(num_groups):
            # Place reference point randomly within the area
            ref_x = self._rng.uniform(0.0, area_width)
            ref_y = self._rng.uniform(0.0, area_height)
            ref_point = np.array([ref_x, ref_y], dtype=np.float32)

            # Create group
            group = Group(group_id=gid)
            group.reference_point = ref_point

            # Generate initial waypoint for the group
            self._generate_waypoint(group)

            # Select initial speed and compute velocity toward first waypoint
            speed = self._rng.uniform(self._speed_min, self._speed_max)
            group.current_speed = speed

            # Compute direction to first waypoint
            waypoint = group.waypoint_queue[0]
            direction = waypoint - group.reference_point
            dist = np.linalg.norm(direction)
            if dist > 0:
                direction = direction / dist
            else:
                direction = np.zeros(2, dtype=np.float32)

            group.reference_velocity = (direction * speed).astype(np.float32)

            # Assign nodes to this group
            group_start = node_idx
            group_end = node_idx + nodes_per_group

            for i in range(group_start, group_end):
                group.add_member(i)
                self.group_ids[i] = gid

                # Sample deviation for this node
                deviation = self._sample_deviation()
                self._deviations[i] = deviation

                # Position = reference point + deviation, clamped to bounds
                pos = ref_point + deviation
                pos[0] = np.clip(pos[0], 0.0, area_width)
                pos[1] = np.clip(pos[1], 0.0, area_height)
                self.positions[i] = pos

                # Initial velocity matches group reference velocity
                self.velocities[i] = group.reference_velocity

            self.groups.append(group)
            node_idx = group_end

    def step(self, t: float, dt: float) -> np.ndarray:
        """
        Advance the mobility model by one time step.

        Updates group reference points via random waypoint, then updates
        all node positions using vectorized operations with boundary clamping
        and velocity reflection.

        Args:
            t: Current simulation time in seconds.
            dt: Time step size in seconds.

        Returns:
            Positions array of shape (N, 2), float32.
        """
        # Phase 1: Update each group's reference point
        for group in self.groups:
            self._update_reference_point(group, dt)

        # Phase 2: Vectorized position update for all nodes
        # positions += velocities * dt
        self.positions += self.velocities * np.float32(dt)

        # Phase 3: Boundary enforcement with velocity reflection
        self._apply_boundary_reflection()

        # Phase 4: Enforce member deviation bounds
        self._enforce_deviation_bounds()

        return self.positions

    def get_positions(self) -> np.ndarray:
        """Return current positions array, shape (N, 2), float32."""
        return self.positions

    def get_velocities(self) -> np.ndarray:
        """Return current velocities array, shape (N, 2), float32."""
        return self.velocities

    def get_group_ids(self) -> np.ndarray:
        """Return group IDs array, shape (N,), int32."""
        return self.group_ids

    def _update_reference_point(self, group: Group, dt: float) -> None:
        """
        Update a group's reference point using random waypoint model.

        If paused, decrement pause timer. Otherwise, move toward current
        waypoint. On arrival, pause and generate new waypoint if queue empty.
        """
        # Handle pause state
        if group.pause_remaining > 0:
            group.pause_remaining -= dt
            if group.pause_remaining > 0:
                # Still paused: set velocity to zero for group members
                group.reference_velocity = np.zeros(2, dtype=np.float32)
                self._set_group_velocities(group, group.reference_velocity)
                return
            else:
                # Pause ended: select new speed and direction
                group.pause_remaining = 0.0
                self._start_movement_to_waypoint(group)
                return

        # Ensure waypoint queue is not empty
        if len(group.waypoint_queue) == 0:
            self._generate_waypoint(group)

        # Move toward current waypoint
        waypoint = group.waypoint_queue[0]
        direction = waypoint - group.reference_point
        dist_to_waypoint = float(np.linalg.norm(direction))

        step_distance = group.current_speed * dt

        if step_distance >= dist_to_waypoint and dist_to_waypoint > 0:
            # Arrive at waypoint
            group.reference_point = waypoint.copy()
            group.waypoint_queue.popleft()

            # Generate new waypoint if queue is exhausted
            if len(group.waypoint_queue) == 0:
                self._generate_waypoint(group)

            # Begin pause
            pause_duration = self._rng.uniform(
                self._config.pause_min_seconds, self._config.pause_max_seconds
            )
            group.pause_remaining = pause_duration
            group.reference_velocity = np.zeros(2, dtype=np.float32)
            group.current_speed = 0.0
            self._set_group_velocities(group, group.reference_velocity)
        else:
            # Move toward waypoint
            if dist_to_waypoint > 0:
                unit_dir = direction / dist_to_waypoint
            else:
                unit_dir = np.zeros(2, dtype=np.float32)

            group.reference_velocity = (unit_dir * group.current_speed).astype(
                np.float32
            )
            group.reference_point = (
                group.reference_point + group.reference_velocity * dt
            ).astype(np.float32)

            # Clamp reference point to bounds
            group.reference_point[0] = np.clip(
                group.reference_point[0], 0.0, self._area_width
            )
            group.reference_point[1] = np.clip(
                group.reference_point[1], 0.0, self._area_height
            )

            # Update velocities for group members
            self._set_group_velocities(group, group.reference_velocity)

    def _start_movement_to_waypoint(self, group: Group) -> None:
        """Start moving toward the next waypoint after a pause ends."""
        # Ensure waypoint queue is not empty
        if len(group.waypoint_queue) == 0:
            self._generate_waypoint(group)

        # Select new speed
        speed = self._rng.uniform(self._speed_min, self._speed_max)
        group.current_speed = speed

        # Compute direction to waypoint
        waypoint = group.waypoint_queue[0]
        direction = waypoint - group.reference_point
        dist = float(np.linalg.norm(direction))
        if dist > 0:
            unit_dir = direction / dist
        else:
            unit_dir = np.zeros(2, dtype=np.float32)

        group.reference_velocity = (unit_dir * speed).astype(np.float32)
        self._set_group_velocities(group, group.reference_velocity)

    def _set_group_velocities(
        self, group: Group, velocity: np.ndarray
    ) -> None:
        """Set velocities for all members of a group."""
        for node_id in group.member_ids:
            self.velocities[node_id] = velocity

    def _apply_boundary_reflection(self) -> None:
        """
        Apply boundary enforcement with velocity reflection.

        Uses vectorized boolean mask operations to negate velocity components
        when positions exceed boundaries, then clamps positions.
        """
        # X-axis reflection
        mask_x_lo = self.positions[:, 0] < 0.0
        mask_x_hi = self.positions[:, 0] > self._area_width
        self.velocities[mask_x_lo, 0] = np.abs(self.velocities[mask_x_lo, 0])
        self.velocities[mask_x_hi, 0] = -np.abs(self.velocities[mask_x_hi, 0])

        # Y-axis reflection
        mask_y_lo = self.positions[:, 1] < 0.0
        mask_y_hi = self.positions[:, 1] > self._area_height
        self.velocities[mask_y_lo, 1] = np.abs(self.velocities[mask_y_lo, 1])
        self.velocities[mask_y_hi, 1] = -np.abs(self.velocities[mask_y_hi, 1])

        # Clamp positions
        self.positions[:, 0] = np.clip(
            self.positions[:, 0], 0.0, self._area_width
        )
        self.positions[:, 1] = np.clip(
            self.positions[:, 1], 0.0, self._area_height
        )

    def _enforce_deviation_bounds(self) -> None:
        """
        Ensure all nodes remain within max_deviation_miles of their
        group's reference point. Nodes that exceed the bound are pulled
        back to the boundary of the allowed deviation radius.
        """
        max_dev = self._config.max_deviation_miles

        for group in self.groups:
            ref = group.reference_point
            for node_id in group.member_ids:
                pos = self.positions[node_id]
                offset = pos - ref
                dist = float(np.linalg.norm(offset))
                if dist > max_dev:
                    # Pull node back to max_deviation distance from ref
                    unit_offset = offset / dist
                    new_pos = ref + unit_offset * max_dev
                    # Clamp to area bounds
                    new_pos[0] = np.clip(new_pos[0], 0.0, self._area_width)
                    new_pos[1] = np.clip(new_pos[1], 0.0, self._area_height)
                    self.positions[node_id] = new_pos

    def _generate_waypoint(self, group: Group) -> None:
        """
        Generate a new random waypoint within the simulation area
        and add it to the group's waypoint queue.
        """
        wp_x = self._rng.uniform(0.0, self._area_width)
        wp_y = self._rng.uniform(0.0, self._area_height)
        waypoint = np.array([wp_x, wp_y], dtype=np.float32)
        group.waypoint_queue.append(waypoint)

    def _sample_deviation(self) -> np.ndarray:
        """
        Sample a 2D deviation vector for a node relative to its group
        reference point. Bounded by max_deviation_miles.

        Returns:
            Deviation vector of shape (2,), float32.
        """
        max_dev = self._config.max_deviation_miles

        if self._config.deviation_model == "gaussian":
            # Gaussian with sigma = max_dev / 3 (so ~99.7% within bounds)
            sigma = max_dev / 3.0
            dx = self._rng.normal(0.0, sigma)
            dy = self._rng.normal(0.0, sigma)
            magnitude = np.sqrt(dx * dx + dy * dy)
            if magnitude > max_dev:
                scale = max_dev / magnitude
                dx *= scale
                dy *= scale
            return np.array([dx, dy], dtype=np.float32)
        else:
            # Uniform disk: random angle, random radius within max_dev
            angle = self._rng.uniform(0.0, 2.0 * np.pi)
            radius = self._rng.uniform(0.0, max_dev)
            dx = radius * np.cos(angle)
            dy = radius * np.sin(angle)
            return np.array([dx, dy], dtype=np.float32)
