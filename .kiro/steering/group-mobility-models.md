---
generated_by: llm
model: claude-sonnet-4-6
---
# Group Mobility Models

## Overview

Group mobility models capture the coordinated movement patterns observed in real-world mobile ad hoc network (MANET) scenarios, where nodes move collectively rather than independently. In tactical military operations, emergency response teams, vehicular convoys, and disaster recovery efforts, devices cluster into logical groups that share common destinations and movement intent. Implementing accurate group mobility in Python requires careful attention to hierarchical data structures, vector mathematics, and scalable algorithms capable of handling 1000 nodes across a 10,000 square mile simulation area.

This guide covers three primary group mobility models — Reference Point Group Mobility (RPGM), Nomadic Community Mobility, and Convoy Mobility — along with the data structures, parameterization strategies, and output formats required to generate valid Kiro steering files.

---

## Foundational Data Structures

Before implementing any specific mobility model, establish the core data structures that will be shared across all group mobility implementations. Efficiency at scale (1000 nodes, 10,000 sq miles) demands careful design choices.

### Coordinate System and Spatial Indexing

The 10,000 square mile area (roughly 160 miles × 62.5 miles, or configurable as a square ~100 miles × 100 miles) must be mapped to a consistent internal coordinate system. Using meters internally and converting for output is recommended.

```python
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from scipy.spatial import cKDTree
import heapq

# Constants
MILES_TO_METERS = 1609.344
AREA_SIDE_MILES = 100.0          # 100 x 100 mile square = 10,000 sq miles
AREA_SIDE_METERS = AREA_SIDE_MILES * MILES_TO_METERS  # ~160,934 meters
SIMULATION_DURATION = 3600.0     # 1 hour in seconds
DEFAULT_TIMESTEP = 1.0           # seconds

@dataclass
class Position:
    """2D position in meters within the simulation area."""
    x: float
    y: float

    def distance_to(self, other: 'Position') -> float:
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y])

    def clamp(self, min_val: float, max_val: float) -> 'Position':
        return Position(
            x=np.clip(self.x, min_val, max_val),
            y=np.clip(self.y, min_val, max_val)
        )

@dataclass
class Velocity:
    """2D velocity vector in meters per second."""
    vx: float
    vy: float

    @property
    def speed(self) -> float:
        return np.sqrt(self.vx**2 + self.vy**2)

    def to_array(self) -> np.ndarray:
        return np.array([self.vx, self.vy])

    @classmethod
    def from_direction_speed(cls, direction_rad: float, speed: float) -> 'Velocity':
        return cls(
            vx=speed * np.cos(direction_rad),
            vy=speed * np.sin(direction_rad)
        )

@dataclass
class Node:
    """Represents a single MANET node."""
    node_id: int
    position: Position
    velocity: Velocity
    group_id: int
    is_leader: bool = False
    waypoint_queue: List[Position] = field(default_factory=list)
    pause_time_remaining: float = 0.0
    
    # Trajectory history for steering file output
    trajectory: List[Tuple[float, float, float]] = field(default_factory=list)  # (time, x, y)

    def record_position(self, time: float):
        self.trajectory.append((time, self.position.x, self.position.y))

@dataclass
class Group:
    """Represents a mobility group with a leader and member nodes."""
    group_id: int
    leader_id: int
    member_ids: List[int]
    reference_point: Position       # Current group reference point (RPGM)
    reference_waypoints: List[Position] = field(default_factory=list)
    group_speed: float = 0.0        # meters per second
    group_direction: float = 0.0    # radians
    pause_time_remaining: float = 0.0
```

### Spatial Index for Efficient Neighbor Lookup

With 1000 nodes spread over 160,000 × 160,000 meters, O(n²) neighbor searches are prohibitively expensive. Use a k-d tree that is rebuilt periodically.

```python
class SpatialIndex:
    """
    Wraps scipy's cKDTree for efficient spatial queries.
    Rebuild interval trades accuracy for performance.
    """
    def __init__(self, rebuild_interval: int = 10):
        self._tree: Optional[cKDTree] = None
        self._node_ids: List[int] = []
        self._positions: np.ndarray = np.empty((0, 2))
        self.rebuild_interval = rebuild_interval
        self._step_count = 0

    def update(self, nodes: Dict[int, Node], force: bool = False):
        self._step_count += 1
        if force or self._step_count % self.rebuild_interval == 0:
            self._rebuild(nodes)

    def _rebuild(self, nodes: Dict[int, Node]):
        self._node_ids = list(nodes.keys())
        self._positions = np.array(
            [[n.position.x, n.position.y] for n in nodes.values()]
        )
        if len(self._positions) > 0:
            self._tree = cKDTree(self._positions)

    def query_radius(self, position: Position, radius: float) -> List[int]:
        if self._tree is None:
            return []
        indices = self._tree.query_ball_point(
            [position.x, position.y], r=radius
        )
        return [self._node_ids[i] for i in indices]

    def query_k_nearest(self, position: Position, k: int) -> List[int]:
        if self._tree is None or len(self._node_ids) == 0:
            return []
        k = min(k, len(self._node_ids))
        distances, indices = self._tree.query([position.x, position.y], k=k)
        if k == 1:
            return [self._node_ids[indices]]
        return [self._node_ids[i] for i in indices]
```

---

## Reference Point Group Mobility (RPGM)

RPGM is the most widely studied group mobility model in MANET research. Each group has a logical center called the **reference point** that moves according to a random waypoint model. Individual member nodes deviate from this reference point by a random offset vector, creating realistic cluster behavior with controllable cohesion.

### Mathematical Foundation

For a group *g* with reference point **RP**_g at time *t*:

- The reference point moves via random waypoint: select destination **D**, move at speed *v* until arrival, pause for *p* seconds, repeat.
- Each member node *i* has position: **pos**_i(*t*) = **RP**_g(*t*) + **δ**_i(*t*)
- Where **δ**_i is drawn from a bounded random distribution, controlling how tightly members cluster.

The deviation **δ**_i can be modeled as:
- **Uniform disk**: δ sampled uniformly within radius *R_max* of the reference point.
- **Gaussian**: δ ~ N(0, σ²) for both x and y, clamped to area bounds.
- **Weighted**: δ magnitude scales with distance from group center, pulling outliers back.

### RPGM Implementation

```python
import random
import math
from typing import Optional

class RPGMModel:
    """
    Reference Point Group Mobility Model.
    
    Groups move via random waypoint at the reference level.
    Members deviate from the reference point with bounded random offsets.
    Suitable for tactical unit simulations, emergency teams, etc.
    """

    def __init__(
        self,
        area_size: float = AREA_SIDE_METERS,
        num_groups: int = 20,
        nodes_per_group_range: Tuple[int, int] = (40, 60),  # avg 50 -> 1000 nodes
        min_speed_mps: float = 0.5,       # ~1 mph walking
        max_speed_mps: float = 13.4,      # ~30 mph vehicle
        min_pause: float = 0.0,           # seconds
        max_pause: float = 300.0,         # 5 minutes
        max_deviation_meters: float = 2000.0,  # ~1.25 miles cluster radius
        deviation_model: str = 'uniform', # 'uniform', 'gaussian'
        gaussian_sigma: float = 800.0,    # meters, if gaussian
        timestep: float = DEFAULT_TIMESTEP,
        seed: Optional[int] = None,
    ):
        self.area_size = area_size
        self.num_groups = num_groups
        self.nodes_per_group_range = nodes_per_group_range
        self.min_speed = min_speed_mps
        self.max_speed = max_speed_mps
        self.min_pause = min_pause
        self.max_pause = max_pause
        self.max_deviation = max_deviation_meters
        self.deviation_model = deviation_model
        self.gaussian_sigma = gaussian_sigma
        self.timestep = timestep
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        self.nodes: Dict[int, Node] = {}
        self.groups: Dict[int, Group] = {}
        self.spatial_index = SpatialIndex(rebuild_interval=10)
        self.current_time = 0.0

    def initialize(self):
        """Create groups and assign nodes to them."""
        node_id = 0
        for gid in range(self.num_groups):
            # Determine group size
            group_size = random.randint(*self.nodes_per_group_range)
            
            # Place reference point randomly in the area
            ref_pos = Position(
                x=random.uniform(0, self.area_size),
                y=random.uniform(0, self.area_size)
            )
            
            # Select initial waypoints for reference point
            waypoints = self._generate_waypoint_sequence(num_waypoints=5)
            
            # Initial group speed and direction
            grp_speed = random.uniform(self.min_speed, self.max_speed)
            grp_dir = random.uniform(0, 2 * math.pi)
            
            group = Group(
                group_id=gid,
                leader_id=node_id,  # First node is leader
                member_ids=[],
                reference_point=ref_pos,
                reference_waypoints=waypoints,
                group_speed=grp_speed,
                group_direction=grp_dir,
            )
            
            # Create member nodes
            for i in range(group_size):
                deviation = self._sample_deviation()
                node_pos = Position(
                    x=np.clip(ref_pos.x + deviation[0], 0, self.area_size),
                    y=np.clip(ref_pos.y + deviation[1], 0, self.area_size)
                )
                node_vel = Velocity.from_direction_speed(grp_dir, grp_speed)
                is_leader = (i == 0)
                
                node = Node(
                    node_id=node_id,
                    position=node_pos,
                    velocity=node_vel,
                    group_id=gid,
                    is_leader=is_leader,
                )
                node.record_position(0.0)
                self.nodes[node_id] = node
                group.member_ids.append(node_id)
                node_id += 1
            
            self.groups[gid] = group
        
        self.spatial_index.update(self.nodes, force=True)
        print(f"Initialized {len(self.nodes)} nodes in {self.num_groups} groups.")

    def _generate_waypoint_sequence(self, num_waypoints: int = 5) -> List[Position]:
        """Generate a sequence of random waypoints within the simulation area."""
        return [
            Position(
                x=random.uniform(0, self.area_size),
                y=random.uniform(0, self.area_size)
            )
            for _ in range(num_waypoints)
        ]

    def _sample_deviation(self) -> np.ndarray:
        """Sample a 2D deviation vector for a node relative to reference point."""
        if self.deviation_model == 'gaussian':
            dx = np.random.normal(0, self.gaussian_sigma)
            dy = np.random.normal(0, self.gaussian_sigma)
            # Clamp to max deviation
            magnitude = np.sqrt(dx**2 + dy**2)
            if magnitude > self.max_deviation:
                scale = self.max_deviation / magnitude
                dx, dy = dx * scale, dy * scale
            return np.array([dx, dy])
        else:  # uniform disk
            angle = random.uniform(0, 2 * math.pi)
            radius = random.uniform(0, self.max_deviation)
            return np.array([radius * math.cos(angle), radius * math.sin(angle)])

    def _update_reference_point(self, group: Group):
        """
        Move the group's reference point toward its next waypoint.
        Uses random waypoint logic: move, arrive, pause, select next waypoint.
        """
        if group.pause_time_remaining > 0:
            group.pause_time_remaining -= self.timestep
            # Set group velocity to zero during pause
            group.group_speed = 0.0
            return

        if not group.reference_waypoints:
            # Generate new waypoints when exhausted
            group.reference_waypoints = self._generate_waypoint_sequence()

        target = group.reference_waypoints[0]
        ref = group.reference_point
        dist = ref.distance_to(target)

        if group.group_speed == 0.0:
            # Just finished pausing, select new speed
            group.group_speed = random.uniform(self.min_speed, self.max_speed)

        step_dist = group.group_speed * self.timestep

        if step_dist >= dist:
            # Arrive at waypoint
            group.reference_point = target
            group.reference_waypoints.pop(0)
            # Begin pause
            group.pause_time_remaining = random.uniform(self.min_pause, self.max_pause)
            group.group_speed = 0.0
        else:
            # Move toward waypoint
            direction = math.atan2(target.y - ref.y, target.x - ref.x)
            group.group_direction = direction
            group.reference_point = Position(
                x=ref.x + step_dist * math.cos(direction),
                y=ref.y + step_dist * math.sin(direction)
            )

    def _update_node_position(self, node: Node, group: Group):
        """
        Update a member node's position based on reference point movement
        plus individual deviation drift.
        """
        ref = group.reference_point
        
        # Slowly drift deviation toward reference point when group is paused
        # or maintain relative offset when moving
        deviation = self._sample_deviation() if group.group_speed == 0 else \
                    self._drift_deviation(node, group)
        
        new_pos = Position(
            x=np.clip(ref.x + deviation[0], 0, self.area_size),
            y=np.clip(ref.y + deviation[1], 0, self.area_size)
        )
        
        # Compute implied velocity for steering file output
        dx = new_pos.x - node.position.x
        dy = new_pos.y - node.position.y
        node.velocity = Velocity(vx=

## See Also

- [Related: Project Architecture and Simulation Overview](./manet-simulation-architecture.md)