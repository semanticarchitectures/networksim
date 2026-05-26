---
generated_by: llm
model: claude-sonnet-4-6
---
# Simulation Performance Optimization

## Overview

Simulating 1000 nodes across 10,000 square miles for 3,600 time steps presents significant computational challenges. Without deliberate optimization, a naïve Python implementation will spend the majority of its runtime on neighbor searches, position updates, and topology rebuilds — operations that execute once per time step for every node. This section provides concrete, profile-driven strategies to keep your Kiro-based MANET simulation running at interactive or near-real-time speeds while producing accurate steering files.

---

## Understanding the Performance Problem

Before optimizing, it helps to quantify the baseline complexity:

| Operation | Naïve Complexity | Frequency |
|---|---|---|
| Neighbor search (all pairs) | O(n²) = ~500,000 checks | Every time step |
| Position update (loop) | O(n) with Python overhead | Every time step |
| Topology rebuild | O(e) edges recomputed | Every time step |
| Group centroid recalculation | O(n) | Every time step |
| Steering file serialization | O(n) | Every time step or interval |

At 3,600 time steps, a naïve O(n²) neighbor search produces approximately **1.8 billion pairwise distance calculations**. This is the primary target for optimization.

---

## Profiling First: Finding Real Bottlenecks

Never optimize blindly. Instrument the simulation loop before rewriting any code.

### Using cProfile

```python
import cProfile
import pstats
import io
from pstats import SortKey

def profile_simulation(sim, steps=100):
    """
    Run a short profiling pass before a full simulation.
    Limits to 100 steps to get representative data quickly.
    """
    profiler = cProfile.Profile()
    profiler.enable()

    for _ in range(steps):
        sim.step()

    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats(SortKey.CUMULATIVE)
    stats.print_stats(20)  # top 20 functions

    print(stream.getvalue())
    return stats
```

### Using line_profiler for Hot Functions

Install with `pip install line_profiler` and decorate suspect functions:

```python
# In your simulation runner:
# kernprof -l -v run_simulation.py

@profile  # line_profiler decorator, active only when kernprof is used
def find_neighbors(self, node_id: int) -> list[int]:
    node_pos = self.positions[node_id]
    neighbors = []
    for other_id, other_pos in self.positions.items():
        if other_id != node_id:
            dist = np.linalg.norm(node_pos - other_pos)
            if dist <= self.transmission_range:
                neighbors.append(other_id)
    return neighbors
```

### Lightweight In-Loop Timing

```python
import time
from collections import defaultdict

class PerformanceMonitor:
    def __init__(self):
        self.timings: dict[str, list[float]] = defaultdict(list)

    def measure(self, label: str):
        """Context manager for timing simulation phases."""
        return self._Timer(self.timings, label)

    def report(self, last_n: int = 100):
        print(f"\n{'Phase':<30} {'Avg (ms)':>10} {'Max (ms)':>10} {'% Total':>8}")
        print("-" * 60)
        total = sum(
            sum(v[-last_n:]) for v in self.timings.values()
        )
        for label, times in sorted(self.timings.items()):
            recent = times[-last_n:]
            avg_ms = (sum(recent) / len(recent)) * 1000
            max_ms = max(recent) * 1000
            pct = (sum(recent) / total * 100) if total > 0 else 0
            print(f"{label:<30} {avg_ms:>10.2f} {max_ms:>10.2f} {pct:>7.1f}%")

    class _Timer:
        def __init__(self, store, label):
            self.store = store
            self.label = label
        def __enter__(self):
            self.start = time.perf_counter()
            return self
        def __exit__(self, *args):
            self.store[self.label].append(time.perf_counter() - self.start)

# Usage inside the simulation loop:
monitor = PerformanceMonitor()

for step in range(3600):
    with monitor.measure("mobility_update"):
        sim.update_positions()
    with monitor.measure("neighbor_search"):
        sim.rebuild_topology()
    with monitor.measure("steering_output"):
        sim.write_steering_step()

monitor.report()
```

---

## Spatial Partitioning: The Highest-Impact Optimization

Spatial partitioning reduces neighbor search from O(n²) to approximately O(n log n) or O(n · k) where k is the average number of nodes per cell.

### Uniform Grid (Recommended for Dense, Uniform Distributions)

A uniform grid works best when nodes are spread relatively evenly across the simulation area — a reasonable assumption for MANET scenarios.

```python
import numpy as np
from collections import defaultdict
from typing import Iterator

class UniformGrid:
    """
    Spatial index using a uniform grid for O(1) cell lookup
    and O(k) neighbor queries where k = nodes per cell.
    
    Covers 10,000 sq miles. Cell size = transmission range
    ensures only adjacent cells need to be checked.
    """

    def __init__(
        self,
        world_width: float,
        world_height: float,
        cell_size: float,
    ):
        self.world_width = world_width
        self.world_height = world_height
        self.cell_size = cell_size

        # Number of cells in each dimension
        self.cols = int(np.ceil(world_width / cell_size))
        self.rows = int(np.ceil(world_height / cell_size))

        # Grid storage: dict of cell -> list of node IDs
        # Using defaultdict avoids key-existence checks
        self.cells: dict[tuple[int, int], list[int]] = defaultdict(list)

        # Reverse map: node_id -> cell key (for fast removal)
        self.node_to_cell: dict[int, tuple[int, int]] = {}

    def _cell_key(self, x: float, y: float) -> tuple[int, int]:
        col = int(np.clip(x / self.cell_size, 0, self.cols - 1))
        row = int(np.clip(y / self.cell_size, 0, self.rows - 1))
        return (col, row)

    def insert(self, node_id: int, x: float, y: float) -> None:
        key = self._cell_key(x, y)
        self.cells[key].append(node_id)
        self.node_to_cell[node_id] = key

    def remove(self, node_id: int) -> None:
        key = self.node_to_cell.pop(node_id, None)
        if key and key in self.cells:
            self.cells[key].remove(node_id)
            if not self.cells[key]:
                del self.cells[key]

    def update(self, node_id: int, new_x: float, new_y: float) -> None:
        """Move a node to its new cell if the cell changed."""
        new_key = self._cell_key(new_x, new_y)
        old_key = self.node_to_cell.get(node_id)
        if old_key == new_key:
            return  # No cell change — skip the update
        if old_key and old_key in self.cells:
            self.cells[old_key].remove(node_id)
            if not self.cells[old_key]:
                del self.cells[old_key]
        self.cells[new_key].append(node_id)
        self.node_to_cell[node_id] = new_key

    def query_neighbors(
        self,
        x: float,
        y: float,
        radius: float,
    ) -> Iterator[int]:
        """
        Yield node IDs in cells within bounding box of radius.
        Caller must filter by exact distance if needed.
        """
        min_col = int(max(0, (x - radius) / self.cell_size))
        max_col = int(min(self.cols - 1, (x + radius) / self.cell_size))
        min_row = int(max(0, (y - radius) / self.cell_size))
        max_row = int(min(self.rows - 1, (y + radius) / self.cell_size))

        for col in range(min_col, max_col + 1):
            for row in range(min_row, max_row + 1):
                yield from self.cells.get((col, row), [])

    def bulk_rebuild(
        self,
        positions: np.ndarray,  # shape (n, 2)
        node_ids: np.ndarray,   # shape (n,)
    ) -> None:
        """
        Rebuild the entire grid from a position array.
        Faster than calling update() for each node when
        many nodes change cells in a single step.
        """
        self.cells.clear()
        self.node_to_cell.clear()
        for node_id, (x, y) in zip(node_ids, positions):
            self.insert(int(node_id), float(x), float(y))
```

### Grid Configuration for the Target Scale

```python
# Simulation area: 10,000 sq miles
# Approximate: ~100 miles x 100 miles
WORLD_WIDTH_MILES  = 100.0
WORLD_HEIGHT_MILES = 100.0

# Transmission range: typical MANET assumption ~1 mile
TRANSMISSION_RANGE_MILES = 1.0

# Cell size = transmission range for optimal neighbor search.
# Each query touches at most 9 cells (3x3 neighborhood).
grid = UniformGrid(
    world_width=WORLD_WIDTH_MILES,
    world_height=WORLD_HEIGHT_MILES,
    cell_size=TRANSMISSION_RANGE_MILES,
)

# Expected nodes per cell at uniform density:
# 1000 nodes / (100x100 cells) = 0.1 nodes/cell average
# Dense clusters may have 10-30 nodes per cell — still fast
```

### Quadtree (Recommended for Clustered Group Mobility)

Group mobility creates spatial clusters, which can unbalance a uniform grid. A quadtree adapts to node density:

```python
from dataclasses import dataclass, field

@dataclass
class BoundingBox:
    x: float
    y: float
    width: float
    height: float

    def contains(self, px: float, py: float) -> bool:
        return (self.x <= px < self.x + self.width and
                self.y <= py < self.y + self.height)

    def intersects_circle(self, cx: float, cy: float, r: float) -> bool:
        # Nearest point on box to circle center
        nearest_x = np.clip(cx, self.x, self.x + self.width)
        nearest_y = np.clip(cy, self.y, self.y + self.height)
        dist_sq = (cx - nearest_x)**2 + (cy - nearest_y)**2
        return dist_sq <= r * r


class Quadtree:
    MAX_CAPACITY = 8   # Nodes before splitting
    MAX_DEPTH    = 10  # Prevent infinite subdivision

    def __init__(self, bounds: BoundingBox, depth: int = 0):
        self.bounds = bounds
        self.depth  = depth
        self.nodes: list[tuple[int, float, float]] = []  # (id, x, y)
        self.children: list['Quadtree'] = []

    def _subdivide(self) -> None:
        hw = self.bounds.width  / 2
        hh = self.bounds.height / 2
        x, y = self.bounds.x, self.bounds.y
        self.children = [
            Quadtree(BoundingBox(x,      y,      hw, hh), self.depth + 1),
            Quadtree(BoundingBox(x + hw, y,      hw, hh), self.depth + 1),
            Quadtree(BoundingBox(x,      y + hh, hw, hh), self.depth + 1),
            Quadtree(BoundingBox(x + hw, y + hh, hw, hh), self.depth + 1),
        ]
        # Redistribute existing nodes
        for item in self.nodes:
            for child in self.children:
                if child.bounds.contains(item[1], item[2]):
                    child.insert(*item)
                    break
        self.nodes = []

    def insert(self, node_id: int, x: float, y: float) -> bool:
        if not self.bounds.contains(x, y):
            return False
        if self.children:
            for child in self.children:
                if child.insert(node_id, x, y):
                    return True
            return False
        self.nodes.append((node_id, x, y))
        if (len(self.nodes) > self.MAX_CAPACITY and
                self.depth < self.MAX_DEPTH):
            self._subdivide()
        return True

    def query_radius(
        self,
        cx: float,
        cy: float,
        radius: float,
        result: list[int] | None = None,
    ) -> list[int]:
        if result is None:
            result = []
        if not self.bounds.intersects_circle(cx, cy, radius):
            return result
        r_sq = radius * radius
        for node_id, x, y in self.nodes:
            if (x - cx)**2 + (y - cy)**2 <= r_sq:
                result.append(node_id)
        for child in self.children:
            child.query_radius(cx, cy, radius, result)
        return result
```

---

## NumPy Vectorization for Batch Operations

Python loops over 1,000 nodes incur interpreter overhead on every iteration. NumPy operates on entire arrays in optimized C/Fortran code.

### Vectorized Position Updates

```python
class VectorizedMobilityEngine:
    """
    Maintains all node positions and velocities as NumPy arrays
    for O(n) batch updates with minimal Python overhead.
    """

    def __init__(self, n_nodes: int, world_bounds: tuple[float, float]):
        self.n = n_nodes
        self.world_w, self.world_h = world_bounds

        # Shape (n, 2) — x and y packed together for cache efficiency
        self.positions  = np.zeros((n_nodes, 2), dtype=np.float32)
        self.velocities = np.zeros((n_nodes, 2), dtype=np.float32)

        # Group membership: shape (n,) — integer group IDs
        self.group_ids  = np.zeros(n_nodes, dtype=np.int32)

        # Group centroids indexed by group ID
        self.group_centroids: dict[int, np.ndarray] = {}

    def update_positions(self, dt: float) -> None:
        """Update all positions in one vectorized operation."""
        self.positions += self.velocities * dt

        # Reflect boundary conditions (bounce off walls)
        # X-axis
        mask_x_lo = self.positions[:, 0] < 0
        mask_x_hi = self.positions[:, 0] > self.world_w
        self.velocities[mask_x_lo, 0] *= -1
        self.velocities[mask_x_hi, 0] *= -1
        self.positions[:, 0] = np.clip(self.positions[:, 0], 0, self.world_w)

        # Y-axis
        mask_y_lo = self.positions[:, 1] < 0
        mask_y_hi = self.positions

## See Also

- [Related: Project Architecture and Simulation Overview](./manet-simulation-architecture.md)
- [Related: Node and Group Data Structures](./node-data-structures.md)
- [Related: Network Topology Management](./network-topology-management.md)
- [Related: Group Mobility Models](./group-mobility-models.md)