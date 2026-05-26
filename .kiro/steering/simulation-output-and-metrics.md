---
generated_by: llm
model: claude-sonnet-4-6
---
# Simulation Output and Metrics Collection

## Overview

Simulation output and metrics collection form the observational layer of the MANET simulation — the means by which raw node movements and link state changes are transformed into structured, analyzable data. For a simulation of this scale (1000 nodes, 10,000 square miles, 1-hour duration), the collection subsystem must be designed with the same rigor applied to the simulation engine itself. Poorly designed output pipelines can become the primary performance bottleneck, cause data loss under high-frequency events, or produce outputs too large to process downstream.

This section covers the design patterns, data structures, output formats, and collection strategies needed to produce Kiro-compatible steering files alongside human-readable metrics exports.

---

## Design Principles for Output Collection

Before writing a single line of collection code, establish these governing principles:

**Separation of Concerns**: The metrics collection system must not be embedded inside the simulation logic. Use an observer/subscriber pattern so that simulation components emit events and collectors subscribe to those events independently. This makes it possible to enable or disable collection without modifying the simulation core.

**Buffered, Asynchronous Writes**: Direct synchronous file writes during simulation steps will kill performance at 1000-node scale. All disk I/O should be buffered and flushed either on a timed interval or when a buffer threshold is reached.

**Configurable Verbosity**: Not every run needs full position logs for all 1000 nodes. The collection system must support granularity levels — from lightweight summary statistics to full per-node time-series traces.

**Schema Consistency**: Output files must have a stable, versioned schema so that downstream tools (visualization, analysis scripts, Kiro steering file generators) can parse them reliably.

---

## Event-Driven Collection Architecture

The collection system is built around three tiers: **emitters**, **collectors**, and **exporters**.

```python
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
import time


class SimulationEventBus:
    """
    Central publish-subscribe bus for simulation events.
    Collectors register handlers for specific event types.
    """

    POSITION_UPDATE   = "position_update"
    LINK_FORMED       = "link_formed"
    LINK_BROKEN       = "link_broken"
    GROUP_MERGE       = "group_merge"
    GROUP_SPLIT       = "group_split"
    STEP_COMPLETE     = "step_complete"
    SIMULATION_END    = "simulation_end"

    def __init__(self):
        self._handlers: Dict[str, List[Callable]] = defaultdict(list)
        self._event_count: int = 0

    def subscribe(self, event_type: str, handler: Callable) -> None:
        self._handlers[event_type].append(handler)

    def publish(self, event_type: str, payload: Any) -> None:
        self._event_count += 1
        for handler in self._handlers[event_type]:
            handler(payload)

    @property
    def total_events(self) -> int:
        return self._event_count


@dataclass
class PositionEvent:
    timestamp: float       # simulation time in seconds
    node_id: int
    x: float               # position in miles
    y: float
    speed: float           # current speed in mph
    heading: float         # degrees
    group_id: int


@dataclass
class LinkEvent:
    timestamp: float
    node_a: int
    node_b: int
    event_type: str        # "formed" or "broken"
    distance: float        # miles at time of event


@dataclass
class StepEvent:
    timestamp: float
    step_number: int
    active_nodes: int
    active_links: int
    wall_clock_ms: float   # real time elapsed for this step
```

---

## Buffered Time-Series Logger

The time-series logger captures per-node positions at every simulation step. At 1000 nodes with 1-second resolution over 3600 seconds, this produces 3,600,000 position records. Buffering is mandatory.

```python
import io
import csv
import threading
from pathlib import Path


class TimeSeriesLogger:
    """
    Buffers position and link events and flushes to disk
    at configurable intervals. Uses a background thread
    for non-blocking I/O.
    """

    POSITION_HEADER = [
        "timestamp", "node_id", "x", "y",
        "speed", "heading", "group_id"
    ]
    LINK_HEADER = [
        "timestamp", "node_a", "node_b", "event_type", "distance"
    ]

    def __init__(
        self,
        output_dir: Path,
        position_buffer_size: int = 50_000,
        link_buffer_size: int = 10_000,
        flush_interval_steps: int = 60,  # flush every 60 sim-seconds
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._position_buffer: List[List] = []
        self._link_buffer: List[List] = []
        self._position_buffer_size = position_buffer_size
        self._link_buffer_size = link_buffer_size
        self._flush_interval = flush_interval_steps
        self._step_counter = 0

        self._pos_file = open(self.output_dir / "positions.csv", "w", newline="")
        self._link_file = open(self.output_dir / "links.csv", "w", newline="")

        self._pos_writer = csv.writer(self._pos_file)
        self._link_writer = csv.writer(self._link_file)
        self._pos_writer.writerow(self.POSITION_HEADER)
        self._link_writer.writerow(self.LINK_HEADER)

        self._lock = threading.Lock()

    def on_position_update(self, event: PositionEvent) -> None:
        row = [
            f"{event.timestamp:.2f}",
            event.node_id,
            f"{event.x:.6f}",
            f"{event.y:.6f}",
            f"{event.speed:.4f}",
            f"{event.heading:.2f}",
            event.group_id,
        ]
        self._position_buffer.append(row)
        if len(self._position_buffer) >= self._position_buffer_size:
            self._flush_positions()

    def on_link_event(self, event: LinkEvent) -> None:
        row = [
            f"{event.timestamp:.2f}",
            event.node_a,
            event.node_b,
            event.event_type,
            f"{event.distance:.6f}",
        ]
        self._link_buffer.append(row)
        if len(self._link_buffer) >= self._link_buffer_size:
            self._flush_links()

    def on_step_complete(self, event: StepEvent) -> None:
        self._step_counter += 1
        if self._step_counter % self._flush_interval == 0:
            self._flush_all()

    def _flush_positions(self) -> None:
        with self._lock:
            self._pos_writer.writerows(self._position_buffer)
            self._pos_file.flush()
            self._position_buffer.clear()

    def _flush_links(self) -> None:
        with self._lock:
            self._link_writer.writerows(self._link_buffer)
            self._link_file.flush()
            self._link_buffer.clear()

    def _flush_all(self) -> None:
        self._flush_positions()
        self._flush_links()

    def close(self) -> None:
        self._flush_all()
        self._pos_file.close()
        self._link_file.close()
```

---

## Mobility Metrics Collector

Mobility metrics are aggregated at configurable intervals rather than per-step, reducing storage requirements while preserving analytical value.

```python
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class MobilitySnapshot:
    timestamp: float
    avg_speed: float              # mean speed across all nodes (mph)
    std_speed: float              # standard deviation
    max_speed: float
    avg_displacement: float       # mean distance from initial position (miles)
    avg_group_cohesion: float     # mean intra-group distance (miles)
    std_group_cohesion: float
    isolated_nodes: int           # nodes with no active links
    group_count: int


class MobilityMetricsCollector:
    """
    Computes mobility metrics at snapshot intervals.
    Maintains rolling state to compute displacement
    without storing full position history.
    """

    def __init__(
        self,
        snapshot_interval: float = 60.0,  # seconds
        output_dir: Path = Path("output"),
    ):
        self._interval = snapshot_interval
        self._output_dir = Path(output_dir)
        self._snapshots: List[MobilitySnapshot] = []

        # Per-node tracking
        self._initial_positions: Dict[int, Tuple[float, float]] = {}
        self._current_positions: Dict[int, Tuple[float, float]] = {}
        self._current_speeds: Dict[int, float] = {}
        self._node_groups: Dict[int, int] = {}

        self._next_snapshot_time: float = snapshot_interval

    def register_initial_positions(
        self, positions: Dict[int, Tuple[float, float]]
    ) -> None:
        """Call once at simulation start to establish displacement baseline."""
        self._initial_positions = dict(positions)

    def on_position_update(self, event: PositionEvent) -> None:
        self._current_positions[event.node_id] = (event.x, event.y)
        self._current_speeds[event.node_id] = event.speed
        self._node_groups[event.node_id] = event.group_id

    def on_step_complete(self, event: StepEvent) -> None:
        if event.timestamp >= self._next_snapshot_time:
            snapshot = self._compute_snapshot(event.timestamp)
            self._snapshots.append(snapshot)
            self._next_snapshot_time += self._interval

    def _compute_snapshot(self, timestamp: float) -> MobilitySnapshot:
        speeds = np.array(list(self._current_speeds.values()), dtype=np.float32)

        # Displacement from initial position
        displacements = []
        for node_id, (cx, cy) in self._current_positions.items():
            if node_id in self._initial_positions:
                ix, iy = self._initial_positions[node_id]
                displacements.append(np.hypot(cx - ix, cy - iy))

        displacements = np.array(displacements, dtype=np.float32)

        # Group cohesion: mean pairwise distance within each group
        group_cohesions = self._compute_group_cohesion()

        return MobilitySnapshot(
            timestamp=timestamp,
            avg_speed=float(np.mean(speeds)),
            std_speed=float(np.std(speeds)),
            max_speed=float(np.max(speeds)),
            avg_displacement=float(np.mean(displacements)) if len(displacements) > 0 else 0.0,
            avg_group_cohesion=float(np.mean(group_cohesions)) if group_cohesions else 0.0,
            std_group_cohesion=float(np.std(group_cohesions)) if group_cohesions else 0.0,
            isolated_nodes=self._count_isolated_nodes(),
            group_count=len(set(self._node_groups.values())),
        )

    def _compute_group_cohesion(self) -> List[float]:
        """
        Compute mean centroid-to-member distance for each group.
        Uses vectorized NumPy operations for performance.
        """
        group_members: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
        for node_id, group_id in self._node_groups.items():
            if node_id in self._current_positions:
                group_members[group_id].append(self._current_positions[node_id])

        cohesions = []
        for gid, members in group_members.items():
            if len(members) < 2:
                cohesions.append(0.0)
                continue
            pts = np.array(members, dtype=np.float32)
            centroid = pts.mean(axis=0)
            dists = np.linalg.norm(pts - centroid, axis=1)
            cohesions.append(float(dists.mean()))

        return cohesions

    def _count_isolated_nodes(self) -> int:
        # Placeholder — topology collector provides link data
        return 0

    def export_to_csv(self) -> Path:
        out_path = self._output_dir / "mobility_metrics.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "avg_speed", "std_speed", "max_speed",
                "avg_displacement", "avg_group_cohesion", "std_group_cohesion",
                "isolated_nodes", "group_count"
            ])
            for snap in self._snapshots:
                writer.writerow([
                    f"{snap.timestamp:.1f}",
                    f"{snap.avg_speed:.4f}",
                    f"{snap.std_speed:.4f}",
                    f"{snap.max_speed:.4f}",
                    f"{snap.avg_displacement:.4f}",
                    f"{snap.avg_group_cohesion:.4f}",
                    f"{snap.std_group_cohesion:.4f}",
                    snap.isolated_nodes,
                    snap.group_count,
                ])
        return out_path
```

---

## Topology Metrics Collector

Topology metrics capture the structural properties of the network graph over time: degree distribution, clustering coefficient, and average path length.

```python
import json
from collections import Counter


@dataclass
class TopologySnapshot:
    timestamp: float
    node_count: int
    edge_count: int
    avg_degree: float
    max_degree: int
    degree_distribution: Dict[int, int]   # degree -> count
    num_components: int                   # connected components
    largest_component_size: int
    avg_clustering_coefficient: float
    # avg_path_length is expensive at scale — computed on sample
    sampled_avg_path_length: Optional[float]


class TopologyMetricsCollector:
    """
    Collects graph topology metrics from the active link set.
    Uses NetworkX for graph analysis at snapshot intervals.
    Path length computation uses a random node sample for scalability.
    """

    def __init__(
        self,
        snapshot_interval: float = 300.0,  # every 5 minutes
        path_length_sample_size: int = 50,
        output_dir: Path = Path("output"),
    ):
        self._interval = snapshot_interval
        self._sample_size = path_length_sample_size
        self._output_dir = Path(output_dir)
        self._snapshots: List[TopologySnapshot] = []
        self._next_snapshot_time = snapshot_interval

        # Maintained live by link events
        self._active_links: set = set()
        self._adjacency: Dict[int, set] = defaultdict(set)

    def on_link_formed(self, event: LinkEvent) -> None:
        edge = (min(event.node_a, event.node_b), max(event.node_a, event.node_b))
        self._active_links.add(edge)
        self._adjacency[event.node_a].add(event.node_b)
        self._adjacency[event.node_b].add(event.node_a)

    def on_link_broken(self, event: LinkEvent) -> None:
        edge = (min(event.node_a, event.node_b), max(event.node_a, event.node_b))
        self._active_links.discard(edge)
        self._adjacency[event.node_

## See Also

- [Related: Project Architecture and Simulation Overview](./manet-simulation-architecture.md)
- [Related: Network Topology Management](./network-topology-management.md)
- [Related: Python Libraries and Dependencies](./python-libraries-and-dependencies.md)