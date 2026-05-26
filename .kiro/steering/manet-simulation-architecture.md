---
generated_by: llm
model: claude-sonnet-4-6
---
# Project Architecture and Simulation Overview

## Project Architecture and Simulation Overview

### Introduction and Scope

This document describes the high-level architecture for a Python-based Mobile Ad Hoc Network (MANET) simulator built from scratch using Kiro, targeting large-scale simulations of **1,000 nodes** moving across a **10,000 square mile** area over a **1-hour simulation window**. The simulator's primary output is a set of **steering files** — structured data artifacts consumed by downstream tools to drive node behavior, routing decisions, and network topology analysis.

At this scale, architectural decisions made early have compounding consequences. A poorly structured simulation loop, inefficient spatial indexing, or unoptimized neighbor-discovery algorithm will not show problems at 10 nodes but will make a 1,000-node simulation unusable. Every design choice documented here is made with that scale constraint as a first-class requirement.

---

### Design Philosophy

The simulator follows three guiding principles:

1. **Separation of concerns** — Mobility, topology, and network layers are independently modeled and communicate through well-defined interfaces. You can swap a mobility model without touching the link-state computation code.
2. **Output-first design** — Because this simulator's purpose is generating steering files rather than real-time visualization, the architecture prioritizes throughput and correctness of output serialization over interactive feedback.
3. **Reproducibility** — All randomness is seeded and recorded. A steering file run must be reproducible from a stored seed and configuration file alone.

---

### Top-Level Module Structure

The project is organized as a Python package. Kiro generates and manages this structure, but understanding it is essential for prompt engineering and code extension.

```
manet_sim/
│
├── config/
│   ├── sim_config.yaml          # Master simulation parameters
│   ├── mobility_config.yaml     # Group mobility model parameters
│   └── network_config.yaml      # Radio range, link thresholds, protocols
│
├── core/
│   ├── simulation.py            # SimulationEngine: main loop orchestration
│   ├── event_queue.py           # Priority queue for event-driven scheduling
│   ├── clock.py                 # Simulation clock and time step management
│   └── seed_manager.py          # RNG seeding and reproducibility
│
├── mobility/
│   ├── base_mobility.py         # Abstract MobilityModel interface
│   ├── group_mobility.py        # Group mobility implementation (e.g., Reference Point Group Mobility)
│   ├── waypoint.py              # Individual node waypoint tracking
│   └── spatial_index.py        # R-tree / grid-based spatial indexing for neighbor queries
│
├── topology/
│   ├── node.py                  # Node data class: position, velocity, ID, group membership
│   ├── link.py                  # Link data class: endpoints, signal strength, latency
│   ├── graph.py                 # NetworkGraph: adjacency management, neighbor queries
│   └── topology_updater.py      # Reconciles mobility positions into topology changes
│
├── network/
│   ├── base_protocol.py         # Abstract routing protocol interface
│   ├── link_state.py            # Optional: link state snapshot for steering file output
│   └── metrics.py               # Per-step metrics: node degree, partition count, etc.
│
├── output/
│   ├── steering_writer.py       # Serializes simulation state to steering file format
│   ├── snapshot.py              # Captures point-in-time topology snapshots
│   └── formats/
│       ├── json_format.py       # JSON steering file serializer
│       └── binary_format.py     # Binary/protobuf serializer for large outputs
│
├── utils/
│   ├── geo.py                   # Geographic coordinate math, distance calculations
│   ├── logging_config.py        # Structured logging setup
│   └── profiler.py              # Performance profiling hooks
│
├── tests/
│   ├── unit/
│   └── integration/
│
├── main.py                      # Entry point: load config, build engine, run simulation
└── requirements.txt
```

---

### Simulation Engine Design

The core of the simulator is the `SimulationEngine` class in `core/simulation.py`. It owns the simulation clock, coordinates layer updates, and drives output generation.

#### Time-Stepped vs. Event-Driven: A Hybrid Approach

For a MANET of this scale and duration, a **pure event-driven approach** is theoretically optimal in time complexity but introduces significant implementation complexity for continuous mobility models — every position update for every node becomes an event, creating millions of scheduled items per simulated second.

A **pure time-stepped approach** is simple but wastes CPU cycles at steps where nothing meaningful changes.

This architecture uses a **hybrid approach**:

- **Coarse time steps** (configurable, default: **1 second**) drive mobility updates and topology reconciliation.
- **An event queue** handles discrete network events (link formations, link breaks, protocol triggers) that occur between steps at sub-second precision.

```
Simulation Time: 0s ──────────────────────────────── 3600s
                  |←── 1s step ──→|←── 1s step ──→|...

Within each step:
  [Mobility Update] → [Topology Reconciliation] → [Event Processing] → [Metrics & Output]
```

This hybrid model keeps the implementation tractable while preserving temporal fidelity for network events that matter to the steering file consumer.

#### The Main Simulation Loop

```python
# core/simulation.py

import heapq
from dataclasses import dataclass, field
from typing import List, Callable
from core.clock import SimulationClock
from mobility.group_mobility import GroupMobilityModel
from topology.topology_updater import TopologyUpdater
from output.steering_writer import SteeringWriter
from network.metrics import MetricsCollector

class SimulationEngine:
    """
    Orchestrates the MANET simulation across all layers.
    Uses a hybrid time-stepped + event-driven design.
    """

    def __init__(self, config: dict):
        self.config = config
        self.clock = SimulationClock(
            start_time=0.0,
            end_time=config['simulation']['duration_seconds'],  # 3600
            step_size=config['simulation']['step_size']         # 1.0
        )
        self.mobility_model = GroupMobilityModel(config['mobility'])
        self.topology_updater = TopologyUpdater(config['network'])
        self.metrics = MetricsCollector()
        self.steering_writer = SteeringWriter(config['output'])
        self.event_queue: List = []  # min-heap: (time, priority, event)

    def run(self):
        """Main simulation loop."""
        self._initialize()

        while not self.clock.is_finished():
            t = self.clock.current_time

            # 1. Update mobility for all nodes
            positions = self.mobility_model.step(t, self.clock.step_size)

            # 2. Reconcile topology from new positions
            topology_delta = self.topology_updater.update(positions, t)

            # 3. Process any discrete events up to current time
            self._process_events_until(t)

            # 4. Collect metrics
            self.metrics.record(t, topology_delta)

            # 5. Write steering file output for this step
            self.steering_writer.write_step(t, positions, topology_delta)

            # 6. Advance clock
            self.clock.advance()

        self._finalize()

    def _initialize(self):
        """Seed RNG, place initial nodes, open output streams."""
        self.mobility_model.initialize()
        self.topology_updater.initialize(self.mobility_model.get_positions())
        self.steering_writer.open()

    def _finalize(self):
        """Flush buffers, close output, write summary."""
        self.steering_writer.close()
        self.metrics.write_summary()

    def schedule_event(self, time: float, callback: Callable, priority: int = 0):
        """Schedule a discrete network event at a specific simulation time."""
        heapq.heappush(self.event_queue, (time, priority, callback))

    def _process_events_until(self, t: float):
        """Drain the event queue up to time t."""
        while self.event_queue and self.event_queue[0][0] <= t:
            event_time, _, callback = heapq.heappop(self.event_queue)
            callback(event_time)
```

---

### Layer Architecture and Data Flow

The simulation has three primary computational layers. Understanding their data flow is critical for extending the codebase or debugging steering file anomalies.

```
┌─────────────────────────────────────────────────────────────────┐
│                        SIMULATION ENGINE                         │
│                     (core/simulation.py)                         │
└────────────────────────────┬────────────────────────────────────┘
                             │ orchestrates
        ┌────────────────────┼─────────────────────┐
        ▼                    ▼                      ▼
┌───────────────┐   ┌────────────────┐   ┌─────────────────┐
│  MOBILITY     │   │   TOPOLOGY     │   │    NETWORK      │
│  LAYER        │──▶│   LAYER        │──▶│    LAYER        │
│               │   │                │   │                 │
│ Group Mobility│   │ Graph/Links    │   │ Link State,     │
│ Waypoints     │   │ Neighbor Sets  │   │ Metrics,        │
│ Spatial Index │   │ Delta Events   │   │ Protocol State  │
└───────────────┘   └────────────────┘   └────────┬────────┘
        │                   │                      │
        └───────────────────┴──────────────────────┘
                             │
                             ▼
                   ┌──────────────────┐
                   │   OUTPUT LAYER   │
                   │  Steering Files  │
                   │  Snapshots       │
                   │  Metrics Logs    │
                   └──────────────────┘
```

#### Inter-Layer Data Contracts

Each layer communicates via typed data structures rather than raw dictionaries. This makes Kiro-generated code easier to validate and extend.

```python
# topology/node.py
from dataclasses import dataclass, field
from typing import Tuple, Optional

@dataclass
class Node:
    node_id: int
    position: Tuple[float, float]    # (x, y) in miles from origin
    velocity: Tuple[float, float]    # (vx, vy) in miles/second
    group_id: int                    # Group membership for group mobility
    neighbors: set = field(default_factory=set)  # Current neighbor node IDs
    is_active: bool = True

# topology/link.py
@dataclass
class Link:
    source_id: int
    target_id: int
    distance: float                  # Miles
    signal_strength: float           # dBm (computed from distance)
    formed_at: float                 # Simulation time link was established
    broken_at: Optional[float] = None

# topology/topology_updater.py
@dataclass
class TopologyDelta:
    timestamp: float
    new_links: List[Link]
    broken_links: List[Link]
    updated_positions: Dict[int, Tuple[float, float]]
```

---

### Scale Considerations for 1,000 Nodes over 10,000 Square Miles

Naively checking all node pairs for link formation is O(n²) per time step — at 1,000 nodes and 3,600 steps, that is **3.6 billion pair checks**. This is not acceptable.

The architecture addresses this at three levels:

#### 1. Spatial Indexing in the Mobility Layer

The `spatial_index.py` module maintains a **grid-based spatial hash** (for uniform distributions) or an **R-tree** (for clustered group mobility). At each step, neighbor queries are reduced from O(n²) to approximately O(n · k) where k is the average number of nodes within radio range.

```python
# mobility/spatial_index.py

class GridSpatialIndex:
    """
    Divides the 10,000 sq mile area into grid cells.
    Cell size = radio range, so neighbor candidates are only
    checked in the 9 surrounding cells.
    """

    def __init__(self, area_width: float, area_height: float, cell_size: float):
        self.cell_size = cell_size  # Should equal radio transmission range
        self.cols = int(area_width / cell_size) + 1
        self.rows = int(area_height / cell_size) + 1
        self.grid: Dict[Tuple[int, int], List[int]] = {}  # cell -> [node_ids]

    def update(self, positions: Dict[int, Tuple[float, float]]):
        """Rebuild grid from current positions. O(n)."""
        self.grid.clear()
        for node_id, (x, y) in positions.items():
            cell = (int(x / self.cell_size), int(y / self.cell_size))
            self.grid.setdefault(cell, []).append(node_id)

    def get_candidates(self, node_id: int, position: Tuple[float, float]) -> List[int]:
        """Return node IDs in adjacent cells. O(k) where k << n."""
        cx, cy = int(position[0] / self.cell_size), int(position[1] / self.cell_size)
        candidates = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                candidates.extend(self.grid.get((cx + dx, cy + dy), []))
        return [nid for nid in candidates if nid != node_id]
```

#### 2. Incremental Topology Updates

Rather than rebuilding the entire topology graph each step, `TopologyUpdater` computes only the **delta** — which links formed and which broke — using the previous step's neighbor sets compared to the current ones. This reduces the work done by the output layer to only what changed.

#### 3. Buffered, Batched Output

Writing 1,000 node positions and topology deltas every second for 3,600 seconds produces substantial output. The `SteeringWriter` uses **write buffering** and **batch serialization** to avoid I/O becoming the bottleneck. The binary output format option further reduces file size and write time for large-scale runs.

---

### Configuration-Driven Initialization

All simulation parameters are externalized to YAML configuration files, which Kiro uses to generate or validate initialization code. This is the master configuration structure:

```yaml
# config/sim_config.yaml

simulation:
  duration_seconds: 3600
  step_size: 1.0              # seconds per time step
  seed: 42
  area:
    width_miles: 100.0        # 100 x 100 = 10,000 sq miles
    height_miles: 100.0

nodes:
  count: 1000
  initial_placement: random   # or: clustered, grid

mobility:
  model: rpgm                 # Reference Point Group Mobility
  num_groups: 20              # 50 nodes per group on average
  group_speed_min: 0.01       # miles/second (~36 mph)
  group_speed_max: 0.03       # miles/second (~108 mph)
  node_deviation: 0.005       # Individual node deviation from group center

network:
  radio_range_miles: 1.0      # Transmission range
  link_quality_threshold: -90 # dBm minimum for viable link

output:
  format: json                # or: binary
  output_dir: ./steering_output
  snapshot_interval: 60       # Write full topology snapshot every 60s
  buffer_size: 100            # Steps to buffer before flushing to disk
```

---

### Entry Point and Startup Sequence

The `main.py` entry point follows a predictable initialization sequence that Kiro scaffolds and that should not be reordered, as each stage depends on the previous:

```python
# main.py

import yaml
import logging
from core.simulation import SimulationEngine
from utils.logging_config import configure_logging

def main():
    # 1. Load configuration
    with open('config/sim_config.yaml') as f:
        config = yaml.