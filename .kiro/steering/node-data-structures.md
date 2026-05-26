---
generated_by: llm
model: claude-sonnet-4-6
---
# Node and Group Data Structures

## Overview

Efficient data structure design is the foundation of a performant MANET simulation at scale. With 1,000 nodes distributed across 10,000 square miles and a 1-hour simulation window, naive approaches—such as flat lists with O(n²) neighbor searches—become computational bottlenecks within the first few simulation steps. This section covers Python class design for `Node` and `Group` objects, spatial indexing strategies, state management, and memory optimization techniques tailored to the Kiro-based steering file output pipeline.

---

## Design Goals and Constraints

Before examining individual data structures, it is worth stating the constraints that drive design decisions:

| Constraint | Implication |
|---|---|
| 1,000 nodes | Neighbor lookup must scale sub-linearly; O(n²) is ~500,000 comparisons per tick |
| 10,000 sq miles (~6,400 km²) | Coordinates span a large continuous space; floating-point precision matters |
| 1-hour simulation (3,600 seconds at 1s ticks) | State snapshots accumulate; memory-efficient history storage is required |
| Group mobility model | Nodes belong to groups with shared waypoints; hierarchical data access patterns |
| Kiro steering file output | Serialization must be fast and deterministic; output format must be stable |

---

## Node Data Structure

### Core Node Class

The `Node` class is the atomic unit of the simulation. It tracks spatial state, network identity, group membership, and communication parameters.

```python
from __future__ import annotations
import uuid
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import time

@dataclass
class NodeState:
    """
    Immutable snapshot of a node's physical state at a single time step.
    Designed for efficient history storage and Kiro steering file serialization.
    """
    timestamp: float          # Simulation time in seconds
    position: np.ndarray      # [x, y] in meters (float32 for memory efficiency)
    velocity: np.ndarray      # [vx, vy] in m/s
    heading: float            # Radians, 0 = East, counterclockwise positive
    speed: float              # Scalar speed in m/s

    def __post_init__(self):
        # Enforce float32 to halve memory vs float64 for large histories
        self.position = np.array(self.position, dtype=np.float32)
        self.velocity = np.array(self.velocity, dtype=np.float32)


class Node:
    """
    Represents a single mobile node in the MANET simulation.

    Coordinate system: metric (meters), origin at simulation area SW corner.
    10,000 sq miles ≈ 25,899,881,103 m² → bounding box ~160,937m x 160,937m
    """

    # Class-level slot definition reduces per-instance memory overhead
    __slots__ = (
        '_id', '_group_id', 'position', 'velocity', 'heading', 'speed',
        'tx_range', 'tx_power_dbm', 'is_group_leader', '_state_history',
        '_history_limit', 'metadata'
    )

    def __init__(
        self,
        node_id: Optional[str] = None,
        position: Tuple[float, float] = (0.0, 0.0),
        velocity: Tuple[float, float] = (0.0, 0.0),
        tx_range: float = 250.0,        # meters, typical 802.11 outdoor range
        tx_power_dbm: float = 20.0,
        group_id: Optional[str] = None,
        history_limit: int = 3600,      # 1 tick/second × 3600 seconds
    ):
        self._id: str = node_id or str(uuid.uuid4())
        self._group_id: Optional[str] = group_id
        self.position: np.ndarray = np.array(position, dtype=np.float32)
        self.velocity: np.ndarray = np.array(velocity, dtype=np.float32)
        self.heading: float = float(np.arctan2(velocity[1], velocity[0]))
        self.speed: float = float(np.linalg.norm(velocity))
        self.tx_range: float = tx_range
        self.tx_power_dbm: float = tx_power_dbm
        self.is_group_leader: bool = False
        self._state_history: List[NodeState] = []
        self._history_limit: int = history_limit
        self.metadata: dict = {}

    @property
    def node_id(self) -> str:
        return self._id

    @property
    def group_id(self) -> Optional[str]:
        return self._group_id

    @group_id.setter
    def group_id(self, value: Optional[str]) -> None:
        self._group_id = value

    @property
    def x(self) -> float:
        return float(self.position[0])

    @property
    def y(self) -> float:
        return float(self.position[1])

    def update_state(self, new_position: np.ndarray, new_velocity: np.ndarray,
                     timestamp: float) -> None:
        """
        Update node's physical state and record a history snapshot.
        Enforces the history limit to prevent unbounded memory growth.
        """
        self.position = np.array(new_position, dtype=np.float32)
        self.velocity = np.array(new_velocity, dtype=np.float32)
        self.speed = float(np.linalg.norm(self.velocity))
        self.heading = float(np.arctan2(self.velocity[1], self.velocity[0]))

        snapshot = NodeState(
            timestamp=timestamp,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            heading=self.heading,
            speed=self.speed,
        )
        self._state_history.append(snapshot)

        # Rolling window: discard oldest states beyond the limit
        if len(self._state_history) > self._history_limit:
            self._state_history.pop(0)

    def get_state_at(self, timestamp: float) -> Optional[NodeState]:
        """Binary search through history for a specific timestamp snapshot."""
        lo, hi = 0, len(self._state_history) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            t = self._state_history[mid].timestamp
            if t == timestamp:
                return self._state_history[mid]
            elif t < timestamp:
                lo = mid + 1
            else:
                hi = mid - 1
        return None

    def distance_to(self, other: 'Node') -> float:
        """Euclidean distance in meters to another node."""
        return float(np.linalg.norm(self.position - other.position))

    def is_in_range(self, other: 'Node') -> bool:
        """Check if another node is within this node's transmission range."""
        return self.distance_to(other) <= self.tx_range

    def to_steering_dict(self, timestamp: float) -> dict:
        """
        Serialize node state to a dictionary suitable for Kiro steering file output.
        Matches the expected KIRO node record format.
        """
        return {
            "node_id": self._id,
            "group_id": self._group_id,
            "timestamp": timestamp,
            "x": round(self.x, 2),
            "y": round(self.y, 2),
            "vx": round(float(self.velocity[0]), 4),
            "vy": round(float(self.velocity[1]), 4),
            "speed": round(self.speed, 4),
            "heading_deg": round(float(np.degrees(self.heading)), 2),
            "tx_range": self.tx_range,
            "is_leader": self.is_group_leader,
        }

    def __repr__(self) -> str:
        return (f"Node(id={self._id[:8]}, group={self._group_id}, "
                f"pos=({self.x:.1f}, {self.y:.1f}), speed={self.speed:.2f}m/s)")
```

### Memory Estimation for Node History

At 1,000 nodes with 3,600 time steps, the history budget is significant:

```python
# Memory analysis for NodeState storage
# Each NodeState: 2×float32 (position) + 2×float32 (velocity) + float32 (heading) 
# + float32 (speed) + float64 (timestamp) = 32 bytes core + Python object overhead ~200 bytes

nodes = 1000
timesteps = 3600
bytes_per_state_conservative = 256  # includes Python object overhead

total_mb = (nodes * timesteps * bytes_per_state_conservative) / (1024 ** 2)
print(f"Estimated history memory: {total_mb:.1f} MB")
# Output: Estimated history memory: 878.9 MB
```

For memory-critical deployments, replace the list-based history with a `numpy` structured array:

```python
import numpy as np

# Define a compact structured dtype for state records
STATE_DTYPE = np.dtype([
    ('timestamp', np.float32),
    ('x',         np.float32),
    ('y',         np.float32),
    ('vx',        np.float32),
    ('vy',        np.float32),
    ('heading',   np.float32),
    ('speed',     np.float32),
])
# 7 × 4 bytes = 28 bytes per record vs ~256 bytes with Python objects
# 1000 nodes × 3600 steps × 28 bytes = ~100 MB total

class CompactNodeHistory:
    """
    Fixed-size circular buffer for node state history using a structured numpy array.
    Reduces memory by ~9× compared to Python object lists.
    """
    def __init__(self, capacity: int = 3600):
        self.capacity = capacity
        self._buffer = np.zeros(capacity, dtype=STATE_DTYPE)
        self._head = 0
        self._size = 0

    def record(self, timestamp, x, y, vx, vy, heading, speed) -> None:
        idx = self._head % self.capacity
        self._buffer[idx] = (timestamp, x, y, vx, vy, heading, speed)
        self._head += 1
        self._size = min(self._size + 1, self.capacity)

    def get_all(self) -> np.ndarray:
        """Return records in chronological order."""
        if self._size < self.capacity:
            return self._buffer[:self._size]
        tail = self._head % self.capacity
        return np.concatenate([self._buffer[tail:], self._buffer[:tail]])
```

---

## Group Data Structure

### Core Group Class

Groups are the primary organizational unit in the group mobility model. Each group manages a set of member nodes, maintains a reference point (virtual center), and coordinates waypoint navigation.

```python
from typing import Dict, Set, Optional, List, Tuple
import numpy as np
from dataclasses import dataclass, field
import uuid


@dataclass
class Waypoint:
    """A target location for group movement with optional dwell time."""
    position: np.ndarray         # [x, y] in meters
    dwell_time: float = 0.0      # Seconds to pause at this waypoint
    arrival_time: Optional[float] = None  # Filled when group arrives

    def __post_init__(self):
        self.position = np.array(self.position, dtype=np.float32)


class Group:
    """
    Represents a mobility group in the MANET simulation.

    A group has:
      - A reference point (RP) representing the group's logical center
      - A leader node whose position approximates the RP
      - Member nodes that follow the RP with individual random deviations
      - A waypoint queue driving coordinated movement
    """

    __slots__ = (
        '_id', '_leader_id', '_member_ids', 'reference_point',
        'reference_velocity', '_waypoint_queue', '_current_waypoint_index',
        'max_member_deviation', 'group_speed_mean', 'group_speed_std',
        'pause_remaining', '_formation_radius', 'metadata'
    )

    def __init__(
        self,
        group_id: Optional[str] = None,
        initial_position: Tuple[float, float] = (0.0, 0.0),
        max_member_deviation: float = 500.0,   # meters from RP
        group_speed_mean: float = 1.4,          # m/s (~walking pace)
        group_speed_std: float = 0.3,
        formation_radius: float = 200.0,        # meters, initial spread
    ):
        self._id: str = group_id or str(uuid.uuid4())
        self._leader_id: Optional[str] = None
        self._member_ids: Set[str] = set()
        self.reference_point: np.ndarray = np.array(initial_position, dtype=np.float32)
        self.reference_velocity: np.ndarray = np.zeros(2, dtype=np.float32)
        self._waypoint_queue: List[Waypoint] = []
        self._current_waypoint_index: int = 0
        self.max_member_deviation: float = max_member_deviation
        self.group_speed_mean: float = group_speed_mean
        self.group_speed_std: float = group_speed_std
        self.pause_remaining: float = 0.0
        self._formation_radius: float = formation_radius
        self.metadata: dict = {}

    @property
    def group_id(self) -> str:
        return self._id

    @property
    def leader_id(self) -> Optional[str]:
        return self._leader_id

    @property
    def member_ids(self) -> Set[str]:
        return frozenset(self._member_ids)

    @property
    def size(self) -> int:
        return len(self._member_ids)

    @property
    def formation_radius(self) -> float:
        return self._formation_radius

    def add_member(self, node_id: str, is_leader: bool = False) -> None:
        """Add a node to this group, optionally designating it as leader."""
        self._member_ids.add(node_id)
        if is_leader:
            self._leader_id = node_id

    def remove_member(self, node_id: str) -> None:
        """Remove a node from the group; reassign leader if needed."""
        self._member_ids.discard(node_id)
        if self._leader_id == node_id:
            # Promote another member to leader
            remaining = self._member_ids - {node_id}
            self._leader_id = next(iter(remaining)) if remaining else None

    def add_waypoint(self, position: Tuple[float, float],
                     dwell_time: float = 0.0) -> None:
        self._waypoint_queue.append(Waypoint(
            position=np.array(position, dtype=np.float32),
            dwell_time=dwell_time,
        ))

    def get_current_waypoint(self) -> Optional[Waypoint]:
        if self._current_waypoint_index < len(self._waypoint_queue):
            return self._waypoint_queue[self._current_waypoint_index]
        return None

    def advance_waypoint(self, timestamp: float) -> bool:
        """
        Move to the next waypoint. Records arrival time on the current one.
        Returns True if a new waypoint is available, False if route is exhausted.
        """
        current = self.get_current_waypoint()
        if current is not None:
            current.arrival_time = timestamp
        self._current_waypoint_index += 1
        return self._current_waypoint_index < len(self._waypoint_queue)

    def distance_to_waypoint(self) -> float:
        """Distance from current reference point to active waypoint."""
        wp = self.get_current_waypoint()
        if wp is None:
            return 0.0
        return float(np.linalg.norm(

## See Also

- [Related: Group Mobility Models](./group-mobility-models.md)