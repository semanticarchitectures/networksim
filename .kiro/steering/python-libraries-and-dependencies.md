---
generated_by: llm
model: claude-sonnet-4-6
---
# Python Libraries and Dependencies

## Overview

Building a MANET simulation from scratch in Kiro requires a carefully selected set of Python libraries that work together to handle graph topology, spatial mathematics, mobility calculations, and output formatting. At the scale of 1,000 nodes across 10,000 square miles with a one-hour simulation duration, library choices directly affect both correctness and performance. This section provides a curated reference of recommended libraries, version guidance, and integration patterns tailored to generating Kiro-compatible steering files.

---

## Core Dependency Stack

The following table summarizes the primary libraries, their roles in the simulation, and minimum recommended versions.

| Library | Role | Recommended Version |
|---|---|---|
| `networkx` | Graph topology, link management, connectivity analysis | `>= 3.2` |
| `numpy` | Array math, distance calculations, coordinate transforms | `>= 1.26` |
| `scipy` | Spatial indexing (KD-tree), statistical distributions | `>= 1.11` |
| `shapely` | Geographic boundary modeling, polygon intersections | `>= 2.0` |
| `matplotlib` | Static visualization, trajectory plotting | `>= 3.8` |
| `pyvis` | Interactive network graph visualization | `>= 0.3.2` |
| `pandas` | Structured output, time-series event logging | `>= 2.1` |
| `h5py` | High-performance binary storage for simulation state | `>= 3.10` |
| `tqdm` | Progress reporting during long simulations | `>= 4.66` |
| `pyyaml` | Steering file serialization and configuration loading | `>= 6.0` |

### Minimum `requirements.txt` for Kiro Project

```text
networkx>=3.2.0
numpy>=1.26.0
scipy>=1.11.0
shapely>=2.0.0
matplotlib>=3.8.0
pyvis>=0.3.2
pandas>=2.1.0
h5py>=3.10.0
tqdm>=4.66.0
pyyaml>=6.0.1
```

Place this file in the root of your Kiro project workspace. Kiro can reference it during environment setup to ensure consistent dependency resolution across simulation runs.

---

## NetworkX — Graph Topology Management

NetworkX provides the graph primitives used to represent MANET link topology at each simulation timestep. For a 1,000-node network, `networkx.Graph` or `networkx.DiGraph` hold node attributes (position, velocity, group ID, transmission radius) alongside dynamically updated edge sets representing active wireless links.

### Version Notes

NetworkX 3.x introduced significant performance improvements to graph copying and subgraph views. Avoid 2.x for this project because edge iteration patterns in the older API incur unnecessary overhead when rebuilding topology every simulation tick.

### Integration Pattern

```python
import networkx as nx
import numpy as np

def build_topology(node_positions: np.ndarray, tx_radius: float) -> nx.Graph:
    """
    Construct a MANET topology graph from current node positions.

    Args:
        node_positions: Array of shape (N, 2) containing (x, y) coordinates in miles.
        tx_radius: Transmission radius in miles.

    Returns:
        NetworkX Graph with node attributes and proximity-based edges.
    """
    G = nx.Graph()
    n_nodes = len(node_positions)

    # Add nodes with positional attributes
    for i in range(n_nodes):
        G.add_node(i, x=node_positions[i, 0], y=node_positions[i, 1])

    # Edge construction is handled externally via SciPy KD-tree (see below)
    # to avoid O(N^2) distance comparisons
    return G


def add_edges_from_pairs(G: nx.Graph, neighbor_pairs: np.ndarray) -> None:
    """Bulk-insert edges from pre-computed neighbor index pairs."""
    G.add_edges_from(neighbor_pairs.tolist())
```

### Useful NetworkX Operations for MANET Analysis

```python
# Degree distribution — useful for connectivity health checks
degrees = dict(G.degree())

# Connected components — identify network partitions
components = list(nx.connected_components(G))
largest_component_size = max(len(c) for c in components)

# Shortest path for routing analysis
try:
    path = nx.shortest_path(G, source=0, target=999)
except nx.NetworkXNoPath:
    path = []

# Clustering coefficient — measures local mesh density
avg_clustering = nx.average_clustering(G)

# Export adjacency for steering file output
adj_dict = nx.to_dict_of_lists(G)
```

### Storing Topology Snapshots

For steering file generation, NetworkX graphs can be serialized at each checkpoint interval:

```python
import json

def snapshot_topology(G: nx.Graph, timestep: float) -> dict:
    """Serialize topology state for steering file output."""
    return {
        "timestep": timestep,
        "node_count": G.number_of_nodes(),
        "edge_count": G.number_of_edges(),
        "adjacency": nx.to_dict_of_lists(G),
        "node_attributes": {
            str(n): data for n, data in G.nodes(data=True)
        }
    }
```

---

## NumPy — Array Mathematics and Coordinate Handling

NumPy underpins almost every numerical operation in the simulation: position updates, velocity vector composition, distance normalization, and group centroid calculations. Vectorized operations on arrays of shape `(N, 2)` replace Python loops and are critical for staying within acceptable runtime at 1,000 nodes.

### Version Notes

NumPy 1.26 is the last stable 1.x release and is compatible with all other dependencies listed here. NumPy 2.x introduced API changes that affect some SciPy internals; unless you have tested compatibility across the full stack, pin to `>=1.26,<2.0` for stability in the Kiro environment.

```text
numpy>=1.26.0,<2.0.0
```

### Key Usage Patterns

```python
import numpy as np

# Initialize positions for 1000 nodes across a 100x100 mile grid (10,000 sq miles)
AREA_SIDE = 100.0  # miles
N_NODES = 1000

positions = np.random.uniform(0, AREA_SIDE, size=(N_NODES, 2))
velocities = np.zeros((N_NODES, 2))

# Vectorized Euclidean distance from one reference point to all nodes
def distances_from_point(positions: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Return array of distances from point to each node."""
    delta = positions - point  # broadcasting (N, 2) - (2,)
    return np.sqrt(np.einsum('ij,ij->i', delta, delta))

# Clip positions to simulation boundary without loops
def apply_boundary_reflection(
    positions: np.ndarray,
    velocities: np.ndarray,
    bounds: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reflect nodes off simulation area boundaries.

    Args:
        positions: (N, 2) current positions.
        velocities: (N, 2) current velocity vectors.
        bounds: (min_val, max_val) applied to both axes.

    Returns:
        Updated positions and velocities after boundary reflection.
    """
    lo, hi = bounds
    for axis in range(2):
        under = positions[:, axis] < lo
        over = positions[:, axis] > hi
        velocities[under, axis] = np.abs(velocities[under, axis])
        velocities[over, axis] = -np.abs(velocities[over, axis])
        positions[:, axis] = np.clip(positions[:, axis], lo, hi)
    return positions, velocities

# Group centroid calculation
def compute_group_centroids(
    positions: np.ndarray,
    group_ids: np.ndarray,
    n_groups: int
) -> np.ndarray:
    """
    Compute centroid for each group using vectorized accumulation.

    Returns:
        Array of shape (n_groups, 2) with centroid coordinates.
    """
    centroids = np.zeros((n_groups, 2))
    counts = np.zeros(n_groups, dtype=int)
    np.add.at(centroids, group_ids, positions)
    np.add.at(counts, group_ids, 1)
    counts = np.maximum(counts, 1)  # prevent division by zero
    return centroids / counts[:, np.newaxis]
```

### Timestep Integration

```python
def update_positions(
    positions: np.ndarray,
    velocities: np.ndarray,
    dt: float
) -> np.ndarray:
    """Euler integration for position update."""
    return positions + velocities * dt
```

---

## SciPy — Spatial Indexing and Statistical Distributions

SciPy's `spatial` submodule provides the `cKDTree` (C-backed KD-tree) data structure, which reduces neighbor-finding from O(N²) to O(N log N). At 1,000 nodes, this difference is pronounced when topology is rebuilt every simulation tick. SciPy's `stats` module handles probability distributions used in mobility models (Gaussian waypoint variance, exponential pause times).

### Version Notes

SciPy 1.11+ includes stability improvements to `cKDTree.query_pairs`. Versions before 1.10 had a regression in large-radius queries that returned duplicate pairs; ensure you are on 1.11 or later.

### KD-Tree Neighbor Discovery

```python
from scipy.spatial import cKDTree

def find_neighbors(
    positions: np.ndarray,
    tx_radius: float
) -> np.ndarray:
    """
    Find all node pairs within transmission radius using KD-tree.

    Args:
        positions: (N, 2) node coordinates.
        tx_radius: Maximum link distance in miles.

    Returns:
        Array of shape (M, 2) containing index pairs of linked nodes.
    """
    tree = cKDTree(positions)
    pairs = tree.query_pairs(r=tx_radius, output_type='ndarray')
    return pairs  # shape (M, 2), each row is [node_i, node_j]


def find_k_nearest(
    positions: np.ndarray,
    query_indices: np.ndarray,
    k: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return distances and indices of k nearest neighbors for each query node.
    Useful for group leader election and cluster formation.
    """
    tree = cKDTree(positions)
    distances, indices = tree.query(positions[query_indices], k=k + 1)
    # Exclude self (index 0 is always the node itself)
    return distances[:, 1:], indices[:, 1:]
```

### Statistical Distributions for Mobility

```python
from scipy.stats import truncnorm, expon

def sample_group_speed(
    mean_speed: float,
    std_speed: float,
    n_nodes: int,
    min_speed: float = 0.5,
    max_speed: float = 60.0
) -> np.ndarray:
    """
    Sample node speeds from a truncated normal distribution.
    Speeds in miles per hour.
    """
    a = (min_speed - mean_speed) / std_speed
    b = (max_speed - mean_speed) / std_speed
    return truncnorm.rvs(a, b, loc=mean_speed, scale=std_speed, size=n_nodes)


def sample_pause_times(mean_pause: float, n_nodes: int) -> np.ndarray:
    """Sample pause durations from exponential distribution (seconds)."""
    return expon.rvs(scale=mean_pause, size=n_nodes)
```

---

## Shapely — Geographic Boundary Handling

At 10,000 square miles, the simulation area may need to model irregular terrain boundaries, exclusion zones, or subdivided geographic sectors. Shapely 2.0 introduced a vectorized geometry engine that processes thousands of point-in-polygon queries significantly faster than the 1.x series.

### Version Notes

**Shapely 2.0 is a breaking change from 1.x.** The 2.0 API removed the `shapely.geometry.mapping` function and restructured the module layout. Always target `>=2.0` for new projects. The 2.0 release also ships with a built-in GEOS upgrade that handles large-coordinate geometries (relevant for mile-scale coordinates) without floating-point precision loss.

### Boundary Definition and Containment

```python
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.vectorized import contains  # Shapely 2.0 vectorized API
import numpy as np

# Define simulation area as a polygon (can be irregular)
SIMULATION_BOUNDARY = Polygon([
    (0, 0), (100, 0), (100, 100), (0, 100)  # 100x100 miles = 10,000 sq miles
])

# Define exclusion zones (e.g., water bodies, restricted areas)
EXCLUSION_ZONE = Polygon([
    (40, 40), (60, 40), (60, 60), (40, 60)
])

def enforce_boundaries(
    positions: np.ndarray,
    boundary: Polygon,
    exclusion_zones: list[Polygon] | None = None
) -> np.ndarray:
    """
    Clamp node positions to valid simulation area.
    Nodes outside boundary are snapped to nearest boundary point.
    Nodes inside exclusion zones are repositioned to nearest valid location.

    Args:
        positions: (N, 2) node positions.
        boundary: Valid simulation area polygon.
        exclusion_zones: Optional list of forbidden polygons.

    Returns:
        Corrected position array.
    """
    corrected = positions.copy()

    # Vectorized containment check using Shapely 2.0
    inside_boundary = contains(boundary, positions[:, 0], positions[:, 1])

    # Snap out-of-bounds nodes to nearest point on boundary
    for i in np.where(~inside_boundary)[0]:
        pt = Point(positions[i])
        nearest = boundary.exterior.interpolate(
            boundary.exterior.project(pt)
        )
        corrected[i] = [nearest.x, nearest.y]

    # Handle exclusion zones
    if exclusion_zones:
        for zone in exclusion_zones:
            in_zone = contains(zone, corrected[:, 0], corrected[:, 1])
            for i in np.where(in_zone)[0]:
                pt = Point(corrected[i])
                nearest = zone.exterior.interpolate(
                    zone.exterior.project(pt)
                )
                corrected[i] = [nearest.x, nearest.y]

    return corrected


def generate_valid_initial_positions(
    n_nodes: int,
    boundary: Polygon,
    exclusion_zones: list[Polygon] | None = None,
    rng: np.random.Generator | None = None
) -> np.ndarray:
    """
    Sample uniformly random positions within valid simulation area.
    Rejection sampling with exclusion zone filtering.
    """
    rng = rng or np.random.default_rng()
    minx, miny, maxx, maxy = boundary.bounds
    positions = []

    while len(positions) < n_nodes:
        batch = rng.uniform(
            low=[minx, miny],
            high=[maxx, maxy],
            size=(n_nodes * 2, 2)
        )
        valid = contains(boundary, batch[:, 0], batch[:, 1])
        if exclusion_zones:
            for zone in exclusion_zones:
                valid &= ~contains(zone, batch[:, 0], batch[:, 1])
        positions.extend(batch[valid].tolist())

    return np.array(positions[:n_nodes])
```

---

## Matplotlib — Static Visualization and Trajectory Plotting

Matplotlib is the primary tool for producing static simulation snapshots: node position maps, group cluster diagrams, link topology overlays, and trajectory heatmaps. These outputs support both visual debugging during development and documentation artifacts exported alongside steering files.

### Integration Pattern

```python
import matplotlib.pyplot as plt
import matplotlib.collections as mc

## See Also

- [Related: Project Architecture and Simulation Overview](./manet-simulation-architecture.md)
- [Related: Network Topology Management](./network-topology-management.md)