---
generated_by: llm
model: claude-sonnet-4-6
---
# Network Topology Management

## Overview

Network topology management is one of the most computationally demanding aspects of simulating a 1000-node MANET over a 10,000 square mile area. As nodes move — individually or in groups — links continuously form and dissolve based on radio transmission range. The topology manager must detect these changes efficiently, update graph representations incrementally, and expose metrics that downstream components (routing, steering file generation) can consume without triggering full graph recomputation on every timestep.

This section covers the complete topology management subsystem: from spatial indexing and link detection to NetworkX graph maintenance, incremental update strategies, and analytical metrics including connectivity, clustering coefficient, and network partition detection.

---

## Radio Range-Based Link Formation and Teardown

### The Physical Link Model

In a MANET simulation, two nodes share a bidirectional link if and only if the Euclidean distance between them falls within a defined radio transmission range `R`. For a 10,000 square mile simulation area (roughly 100 miles × 100 miles), a typical tactical radio range might be 5–25 miles depending on the scenario.

```python
import math
from typing import Optional

# Constants for the simulation
RADIO_RANGE_MILES = 10.0          # Radio transmission range
AREA_SIDE_MILES = 100.0           # Square simulation area side length
SIMULATION_AREA = (0.0, 0.0, AREA_SIDE_MILES, AREA_SIDE_MILES)

def euclidean_distance(pos1: tuple[float, float], pos2: tuple[float, float]) -> float:
    """
    Compute Euclidean distance between two 2D positions.
    
    Args:
        pos1: (x, y) coordinates of first node
        pos2: (x, y) coordinates of second node
    
    Returns:
        Distance in miles
    """
    dx = pos1[0] - pos2[0]
    dy = pos1[1] - pos2[1]
    return math.sqrt(dx * dx + dy * dy)

def nodes_in_range(
    pos1: tuple[float, float],
    pos2: tuple[float, float],
    radio_range: float = RADIO_RANGE_MILES
) -> bool:
    """
    Determine whether two nodes are within radio range of each other.
    
    Uses squared distance comparison to avoid the sqrt when possible,
    falling back to the full computation only when needed.
    """
    dx = pos1[0] - pos2[0]
    dy = pos1[1] - pos2[1]
    dist_sq = dx * dx + dy * dy
    return dist_sq <= (radio_range * radio_range)
```

### Link State Transitions

Each potential link between nodes `i` and `j` can be in one of three states:

| State | Description | Trigger |
|-------|-------------|---------|
| `ABSENT` | Nodes out of range, no link | Distance > R |
| `ACTIVE` | Nodes in range, link exists | Distance ≤ R |
| `HYSTERESIS` | Recently transitioned, stabilization window | Configurable buffer zone |

A hysteresis buffer prevents rapid link flapping when nodes hover near the boundary of radio range:

```python
from enum import Enum, auto
from dataclasses import dataclass, field

class LinkState(Enum):
    ABSENT = auto()
    ACTIVE = auto()
    HYSTERESIS = auto()

@dataclass
class LinkRecord:
    """
    Tracks the history and state of a potential link between two nodes.
    """
    node_a: int
    node_b: int
    state: LinkState = LinkState.ABSENT
    formed_at: float = 0.0          # Simulation time when link was formed
    last_distance: float = float('inf')
    flap_count: int = 0             # Number of state transitions

HYSTERESIS_MARGIN = 0.5            # Miles: buffer zone around radio range boundary

def evaluate_link_state(
    current_state: LinkState,
    distance: float,
    radio_range: float = RADIO_RANGE_MILES
) -> LinkState:
    """
    Determine new link state given current state and measured distance.
    Implements hysteresis to prevent rapid link flapping near boundary.
    
    Args:
        current_state: The existing link state
        distance: Current distance between nodes in miles
        radio_range: Radio transmission range in miles
    
    Returns:
        New LinkState
    """
    inner_threshold = radio_range - HYSTERESIS_MARGIN
    outer_threshold = radio_range + HYSTERESIS_MARGIN

    if current_state == LinkState.ABSENT:
        if distance <= inner_threshold:
            return LinkState.ACTIVE
        return LinkState.ABSENT

    elif current_state == LinkState.ACTIVE:
        if distance > outer_threshold:
            return LinkState.ABSENT
        elif distance > inner_threshold:
            return LinkState.HYSTERESIS  # Near the edge, watch carefully
        return LinkState.ACTIVE

    elif current_state == LinkState.HYSTERESIS:
        if distance <= inner_threshold:
            return LinkState.ACTIVE
        elif distance > outer_threshold:
            return LinkState.ABSENT
        return LinkState.HYSTERESIS

    return LinkState.ABSENT
```

---

## Spatial Indexing for Efficient Neighbor Discovery

### Why Spatial Indexing Is Non-Negotiable at Scale

With 1,000 nodes, a naive O(n²) distance check every timestep requires ~500,000 computations per update cycle. At a 10-second timestep over a 1-hour simulation (360 timesteps), that is 180 million distance evaluations — a significant bottleneck. A spatial index reduces neighbor queries to O(k log n) where k is the expected number of neighbors per node.

### R-Tree Based Spatial Index

```python
from rtree import index as rtree_index
import numpy as np
from typing import Iterator

class SpatialIndex:
    """
    R-Tree spatial index for efficient radio range neighbor queries.
    Wraps the rtree library with a MANET-specific interface.
    
    For 1000 nodes over 10000 sq miles, this provides roughly 10-50x
    speedup over brute-force distance calculations.
    """

    def __init__(self, radio_range: float = RADIO_RANGE_MILES):
        self.radio_range = radio_range
        self._idx = rtree_index.Index()
        self._positions: dict[int, tuple[float, float]] = {}

    def insert_node(self, node_id: int, x: float, y: float) -> None:
        """Insert or update a node's position in the spatial index."""
        # R-tree stores bounding boxes; a point is a degenerate box
        self._idx.insert(node_id, (x, y, x, y))
        self._positions[node_id] = (x, y)

    def update_node(self, node_id: int, x: float, y: float) -> None:
        """Update position: remove old entry, insert new one."""
        if node_id in self._positions:
            old_x, old_y = self._positions[node_id]
            self._idx.delete(node_id, (old_x, old_y, old_x, old_y))
        self.insert_node(node_id, x, y)

    def remove_node(self, node_id: int) -> None:
        """Remove a node from the spatial index."""
        if node_id in self._positions:
            x, y = self._positions[node_id]
            self._idx.delete(node_id, (x, y, x, y))
            del self._positions[node_id]

    def query_neighbors(self, node_id: int) -> list[int]:
        """
        Return list of node IDs within radio range of the given node.
        Uses bounding box pre-filter followed by exact distance check.
        
        Args:
            node_id: The querying node's ID
        
        Returns:
            List of neighbor node IDs (excludes the querying node itself)
        """
        if node_id not in self._positions:
            return []

        x, y = self._positions[node_id]
        r = self.radio_range

        # Bounding box query: retrieve candidates
        candidates = list(self._idx.intersection(
            (x - r, y - r, x + r, y + r)
        ))

        # Exact distance filter
        neighbors = []
        for candidate_id in candidates:
            if candidate_id == node_id:
                continue
            cx, cy = self._positions[candidate_id]
            if nodes_in_range((x, y), (cx, cy), self.radio_range):
                neighbors.append(candidate_id)

        return neighbors

    def query_range_box(
        self,
        x_min: float, y_min: float,
        x_max: float, y_max: float
    ) -> list[int]:
        """Return all node IDs within a rectangular region."""
        return list(self._idx.intersection((x_min, y_min, x_max, y_max)))

    def bulk_load(self, positions: dict[int, tuple[float, float]]) -> None:
        """
        Efficiently load all node positions at initialization.
        
        Args:
            positions: dict mapping node_id -> (x, y)
        """
        for node_id, (x, y) in positions.items():
            self._idx.insert(node_id, (x, y, x, y))
            self._positions[node_id] = (x, y)
```

### Grid-Based Alternative for Uniform Distributions

For scenarios with relatively uniform node distribution, a hash-grid offers O(1) amortized neighbor lookup with lower constant overhead than an R-tree:

```python
import math
from collections import defaultdict

class GridSpatialIndex:
    """
    Uniform grid spatial index optimized for evenly distributed nodes.
    Cell size is set to radio_range so each query touches at most 9 cells.
    
    Memory: O(n) where n is number of nodes
    Query: O(k) where k is average nodes per cell
    """

    def __init__(
        self,
        radio_range: float = RADIO_RANGE_MILES,
        area_width: float = AREA_SIDE_MILES,
        area_height: float = AREA_SIDE_MILES
    ):
        self.radio_range = radio_range
        self.cell_size = radio_range  # One cell = one radio range width
        self.cols = math.ceil(area_width / self.cell_size) + 1
        self.rows = math.ceil(area_height / self.cell_size) + 1

        # Grid: (col, row) -> set of node_ids
        self.grid: dict[tuple[int, int], set[int]] = defaultdict(set)
        self.positions: dict[int, tuple[float, float]] = {}
        self.node_cells: dict[int, tuple[int, int]] = {}

    def _get_cell(self, x: float, y: float) -> tuple[int, int]:
        col = int(x / self.cell_size)
        row = int(y / self.cell_size)
        return (col, row)

    def insert_node(self, node_id: int, x: float, y: float) -> None:
        cell = self._get_cell(x, y)
        self.grid[cell].add(node_id)
        self.positions[node_id] = (x, y)
        self.node_cells[node_id] = cell

    def update_node(self, node_id: int, x: float, y: float) -> None:
        if node_id in self.node_cells:
            old_cell = self.node_cells[node_id]
            self.grid[old_cell].discard(node_id)
        self.insert_node(node_id, x, y)

    def query_neighbors(self, node_id: int) -> list[int]:
        if node_id not in self.positions:
            return []

        x, y = self.positions[node_id]
        col, row = self.node_cells[node_id]
        neighbors = []

        # Check 3x3 grid of neighboring cells
        for dc in (-1, 0, 1):
            for dr in (-1, 0, 1):
                cell = (col + dc, row + dr)
                for candidate_id in self.grid.get(cell, set()):
                    if candidate_id == node_id:
                        continue
                    cx, cy = self.positions[candidate_id]
                    if nodes_in_range((x, y), (cx, cy), self.radio_range):
                        neighbors.append(candidate_id)

        return neighbors
```

---

## NetworkX Graph Representation

### Graph Structure and Initialization

The topology manager uses NetworkX as the primary graph representation. A `Graph` (undirected) models symmetric radio links, with node and edge attributes carrying simulation metadata.

```python
import networkx as nx
from typing import Any

class MANETTopologyGraph:
    """
    NetworkX-backed graph representing the current MANET topology.
    
    Node attributes:
        - pos: (x, y) position in miles
        - group_id: group membership from group mobility model
        - velocity: (vx, vy) current velocity vector
        - active: bool, whether node is currently operational
    
    Edge attributes:
        - distance: current link distance in miles
        - formed_at: simulation time when link was established
        - link_quality: normalized 0.0–1.0 quality score
        - state: LinkState enum value
    """

    def __init__(self, radio_range: float = RADIO_RANGE_MILES):
        self.G = nx.Graph()
        self.radio_range = radio_range
        self._link_records: dict[tuple[int, int], LinkRecord] = {}

    def initialize_nodes(
        self,
        node_data: dict[int, dict[str, Any]]
    ) -> None:
        """
        Populate the graph with initial node set.
        
        Args:
            node_data: dict mapping node_id -> attribute dict
                       Expected keys: 'pos', 'group_id', 'velocity'
        """
        for node_id, attrs in node_data.items():
            self.G.add_node(node_id, **attrs, active=True)

    def add_link(
        self,
        node_a: int,
        node_b: int,
        distance: float,
        current_time: float
    ) -> None:
        """Add a new radio link between two nodes."""
        link_quality = self._compute_link_quality(distance)
        self.G.add_edge(
            node_a, node_b,
            distance=distance,
            formed_at=current_time,
            link_quality=link_quality,
            state=LinkState.ACTIVE
        )

        # Canonical key: smaller node_id first
        key = (min(node_a, node_b), max(node_a, node_b))
        self._link_records[key] = LinkRecord(
            node_a=node_a,
            node_b=node_b,
            state=LinkState.ACTIVE,
            formed_at=current_time,
            last_distance=distance
        )

    def remove_link(self, node_a: int, node_b: int) -> None:
        """Remove a radio link, recording flap count."""
        if self.G.has_edge(node_a, node_b):
            self.G.remove_edge(node_a, node_b)
        key = (min(node_a, node_b), max(node_a, node_b))
        if key in self._link_records:
            self._link_records[key].flap_count += 1
            self._link_records[key].state = LinkState.ABSENT

    def update_node_position(
        self,
        node_id: int,
        x: float,
        y: float,
        velocity: tuple[float, float]
    ) -> None:
        """Update node position and velocity attributes in the graph."""
        if self.G.has_node(node_id):
            self.G.nodes[node_id]['pos'] = (x, y)
            self.G.nodes[node_id]['velocity'] = velocity

    def _compute_link_quality(self, distance: float) -> float:
        """
        Compute a normalized link quality score based on

## See Also

- [Related: Project Architecture and Simulation Overview](./manet-simulation-architecture.md)
- [Related: Node and Group Data Structures](./node-data-structures.md)
- [Related: Group Mobility Models](./group-mobility-models.md)