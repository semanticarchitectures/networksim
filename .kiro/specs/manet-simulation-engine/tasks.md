# Implementation Plan: MANET Simulation Engine

## Overview

This plan implements a Python-based MANET simulation engine for 1,000 mobile nodes across 10,000 square miles over a 1-hour duration. The implementation follows a bottom-up approach: core infrastructure first, then mobility, topology, output, metrics, and finally the main simulation loop that wires everything together. All components use NumPy vectorized operations for performance and Hypothesis for property-based testing.

## Tasks

- [x] 1. Set up project structure, dependencies, and core interfaces
  - [x] 1.1 Create Python package structure with __init__.py files
    - Create directories: `manet_sim/core/`, `manet_sim/mobility/`, `manet_sim/topology/`, `manet_sim/network/`, `manet_sim/output/`, `manet_sim/utils/`, `config/`, `tests/`
    - Add `__init__.py` to each package directory
    - Create `requirements.txt` with pinned dependencies: networkx>=3.2.0, numpy>=1.26.0, scipy>=1.11.0, shapely>=2.0.0, pandas>=2.1.0, pyyaml>=6.0.1, tqdm>=4.66.0, hypothesis>=6.0.0
    - Create `config/sim_config.yaml` with default configuration (1000 nodes, 100x100 miles, 3600s, seed 42)
    - _Requirements: 1.1, 1.2_

  - [x] 1.2 Implement ConfigurationManager (`manet_sim/core/config.py`)
    - Define dataclasses: `SimulationConfig`, `MobilityConfig`, `NetworkConfig`, `OutputConfig`
    - Implement `load(path: str) -> SimulationConfig` to parse YAML and validate ranges
    - Implement `validate(config: dict) -> list[str]` returning validation errors
    - Raise `ConfigValidationError` for missing fields, out-of-range values, invalid seed
    - Raise `ConfigParseError` for invalid YAML with line number
    - Support default configuration that produces a runnable simulation
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

  - [x]* 1.3 Write property tests for ConfigurationManager
    - **Property 1: Configuration round-trip** — valid config dict → YAML → load → fields match original
    - **Property 2: Configuration validation error reporting** — invalid fields produce exact error set
    - **Validates: Requirements 2.1, 2.2, 2.3**

  - [x] 1.4 Implement SimulationClock (`manet_sim/core/clock.py`)
    - Implement `SimulationClock` class with `__init__(start, end, step_size)`
    - Implement `current_time` property, `is_finished()`, and `advance()` methods
    - Handle partial final step (remaining time < step_size)
    - Validate step_size > 0 and step_size <= duration at construction
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

  - [x]* 1.5 Write property test for SimulationClock
    - **Property 3: Clock advancement reaches exact end time** — for any valid duration/step_size, final time == end time
    - **Validates: Requirements 3.1, 3.3, 3.5**

  - [x] 1.6 Implement SeedManager (`manet_sim/core/seed_manager.py`)
    - Implement `SeedManager` class with `__init__(seed: Optional[int])`
    - Auto-generate seed if None provided; expose via `seed` property
    - Implement `get_rng(subsystem: str) -> np.random.Generator` for independent RNG streams
    - Validate seed range (0 to 2^32-1)
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

  - [x]* 1.7 Write property test for SeedManager reproducibility
    - **Property 4: Simulation reproducibility** — same seed produces identical RNG sequences across subsystems
    - **Validates: Requirements 4.1, 4.2, 4.3**

- [x] 2. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 3. Implement event system and data structures
  - [x] 3.1 Implement EventQueue (`manet_sim/core/event_queue.py`)
    - Define `ScheduledEvent` dataclass with time, priority, callback (ordered by time then priority)
    - Implement min-heap priority queue with `schedule()`, `drain_until(t)`, `is_empty()`
    - `drain_until(t)` processes all events with timestamp <= t in correct order, returns count
    - Handle events scheduled in the past (process immediately)
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

  - [x]* 3.2 Write property test for EventQueue ordering
    - **Property 5: Event queue ordering** — drain up to T processes exactly events <= T in (time, priority) order
    - **Validates: Requirements 5.2, 5.3**

  - [x] 3.3 Implement EventBus (`manet_sim/core/event_bus.py`)
    - Implement pub/sub with `subscribe(event_type, handler) -> str`, `unsubscribe(id)`, `publish(event_type, payload)`
    - Implement `enable(id)` and `disable(id)` for toggling subscribers
    - Catch and log subscriber exceptions without interrupting delivery to other subscribers
    - Define event payload dataclasses: `PositionUpdateEvent`, `LinkFormedEvent`, `LinkBrokenEvent`, `StepCompleteEvent`, `SimulationEndEvent`
    - _Requirements: 15.1, 15.2, 15.3, 15.4, 15.5, 15.6_

  - [x]* 3.4 Write property tests for EventBus
    - **Property 24: Event bus ordered delivery** — subscribers receive events in publication order
    - **Property 25: Event bus fault isolation** — one failing subscriber doesn't block others
    - **Property 26: Event bus enable/disable** — disabled subscribers don't receive events; re-enabled ones resume without replay
    - **Validates: Requirements 15.1, 15.4, 15.5**

  - [x] 3.5 Implement Group data structure (`manet_sim/mobility/group_mobility.py` — Group class)
    - Define `Group` dataclass with reference_point, reference_velocity, member_ids (ordered list), leader_id, waypoint_queue (deque, capacity 1-100), pause_remaining, current_speed
    - Implement `add_member(node_id)` — reject duplicates
    - Implement `remove_member(node_id)` — promote longest-tenured on leader removal
    - Handle empty group case
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_

  - [x]* 3.6 Write property tests for Group data structure
    - **Property 8: Group membership add-then-remove identity** — add then remove returns to original state
    - **Property 9: Duplicate member rejection** — adding existing member is rejected, list unchanged
    - **Property 10: Leader succession on removal** — new leader is longest-tenured remaining member
    - **Validates: Requirements 7.2, 7.3, 7.4, 7.5**

- [x] 4. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Implement mobility layer
  - [x] 5.1 Implement SpatialIndex (`manet_sim/mobility/spatial_index.py`)
    - Implement `GridSpatialIndex` with cell_size = radio_range
    - Implement `rebuild(positions: np.ndarray)` — O(n) full rebuild from positions array
    - Implement `query_radius(x, y, radius) -> list[int]` — check 3x3 neighboring cells
    - Implement `query_pairs(radius) -> set[tuple[int, int]]` — all pairs within radius
    - Return empty collection when no nodes in range
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_

  - [x]* 5.2 Write property test for SpatialIndex
    - **Property 16: Spatial index zero false negatives** — radius query returns superset of true neighbors
    - **Validates: Requirements 10.1**

  - [x] 5.3 Implement GroupMobilityModel (`manet_sim/mobility/group_mobility.py`)
    - Implement `__init__(config: MobilityConfig, rng: np.random.Generator)`
    - Implement `initialize(node_count, area_width, area_height)` — create 20 groups of 50, place reference points, assign members
    - Implement `step(t, dt) -> np.ndarray` — update reference points via random waypoint, update node positions with vectorized operations, apply boundary clamping
    - Store positions as (N, 2) float32 array, velocities as (N, 2) float32 array, group_ids as (N,) int32 array
    - Implement waypoint generation when queue exhausts
    - Member deviation bounded by max_deviation_miles (1.25 miles)
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 9.1, 9.2, 9.3, 9.4_

  - [x]* 5.4 Write property tests for mobility model
    - **Property 7: Position array shape invariant** — positions maintain shape (N, 2) with correct dtype after every step
    - **Property 11: Boundary clamping invariant** — all positions within [0, width] × [0, height] after every step
    - **Property 12: Member deviation bounds** — distance from node to group reference point <= max_deviation
    - **Property 13: Waypoint queue non-exhaustion** — queue never empty when movement needed
    - **Property 14: Vectorized position update correctness** — result equals P + V * dt before boundary enforcement
    - **Property 15: Velocity reflection at boundary** — velocity component negated when position hits boundary
    - **Validates: Requirements 6.3, 8.3, 8.4, 8.5, 8.6, 9.1, 9.2, 9.3, 9.4**

- [x] 6. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. Implement topology layer
  - [x] 7.1 Implement LinkManager (`manet_sim/topology/link_manager.py`)
    - Implement `LinkManager` with radio_range and hysteresis_pct
    - Implement `evaluate(current_state: LinkState, distance: float) -> LinkState`
    - Define `inner_threshold` = radio_range - margin, `outer_threshold` = radio_range + margin
    - ABSENT → ACTIVE when distance < inner_threshold
    - ACTIVE → ABSENT when distance > outer_threshold
    - No state change in hysteresis zone
    - _Requirements: 12.1, 12.2, 12.3, 12.6_

  - [x]* 7.2 Write property tests for LinkManager hysteresis
    - **Property 17: Hysteresis link formation** — ABSENT + distance < inner → ACTIVE
    - **Property 18: Hysteresis link teardown** — ACTIVE + distance > outer → ABSENT
    - **Property 19: Hysteresis stability in buffer zone** — state unchanged in buffer zone
    - **Validates: Requirements 12.1, 12.2, 12.3, 12.6**

  - [x] 7.3 Implement TopologyUpdater (`manet_sim/topology/topology_updater.py`)
    - Implement `TopologyUpdater` with NetworkX undirected graph
    - Implement `initialize(positions) -> TopologyDelta` — evaluate all pairs against inner threshold
    - Implement `update(positions, t) -> TopologyDelta` — use spatial index query_pairs, apply hysteresis via LinkManager, compute incremental delta
    - Store node attributes (position, group_id, velocity) on graph nodes
    - Store edge attributes (distance, formation_time, link_quality) on graph edges
    - Implement `get_graph() -> nx.Graph`
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 12.4, 12.5_

  - [x]* 7.4 Write property tests for TopologyUpdater
    - **Property 20: Topology delta correctness** — applying delta to old state produces new state
    - **Property 21: Initial topology correctness** — initial edges exactly match pairs below inner threshold
    - **Validates: Requirements 12.4, 12.5**

- [x] 8. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 9. Implement output layer
  - [x] 9.1 Implement SteeringWriter (`manet_sim/output/steering_writer.py`)
    - Implement buffered JSON output with configurable flush interval (default 100 steps, max 1000)
    - Implement `open()` — create output file, write metadata header
    - Implement `write_step(t, positions, velocities, group_ids, delta)` — buffer step data
    - Implement `close()` — flush remaining buffer, write valid JSON closing structure
    - Write full topology snapshots at configurable interval (default 60s)
    - Include metadata: seed, config, start/end time, total_steps, node_count, schema_version
    - Handle flush failures: retry once, log error with lost record count on second failure
    - Node serialization: node_id, group_id, x, y, vx, vy, transmission_range, active — numeric values ≤ 6 decimal places
    - _Requirements: 6.1, 6.2, 13.1, 13.2, 13.3, 13.4, 13.5, 13.6_

  - [x]* 9.2 Write property test for node serialization
    - **Property 6: Node serialization completeness** — serialized dict contains all required keys with correct precision
    - **Validates: Requirements 6.2**

  - [x] 9.3 Implement SteeringParser (`manet_sim/output/steering_parser.py`)
    - Implement `parse(filepath, timestep) -> SimulationState` — reconstruct state for a given timestep
    - Implement `validate_schema(filepath) -> list[str]` — check JSON structure
    - Return error for missing timestep, malformed JSON, or schema violations
    - Implement `SteeringPrinter` with `format(state) -> str` — 2-space indent, sorted keys
    - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5_

  - [x]* 9.4 Write property tests for steering file round-trip
    - **Property 22: Steering file serialization round-trip** — print then parse produces field-equal state within 1e-6 tolerance
    - **Property 23: Pretty-printer determinism** — calling printer twice produces identical output
    - **Validates: Requirements 14.4, 14.5**

- [x] 10. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 11. Implement metrics collectors
  - [x] 11.1 Implement MobilityMetricsCollector (`manet_sim/network/metrics.py`)
    - Subscribe to position_update, step_complete, simulation_end via EventBus
    - Compute snapshots at configurable interval (default 60s): avg_speed, std_speed, max_speed, avg_displacement, avg_group_cohesion
    - Report 0.0 cohesion for groups with < 2 members
    - Export to CSV with header row, 4 decimal places
    - _Requirements: 16.1, 16.2, 16.3_

  - [x] 11.2 Implement TopologyMetricsCollector (`manet_sim/network/metrics.py`)
    - Subscribe to link_formed, link_broken, step_complete, simulation_end via EventBus
    - Compute snapshots at configurable interval (default 300s): node_count, edge_count, avg_degree, max_degree, num_connected_components, largest_component_size, avg_clustering_coefficient
    - Handle zero-node graph (all metrics = 0)
    - Export to JSON with timestamp
    - _Requirements: 17.1, 17.2, 17.3, 17.4_

  - [x]* 11.3 Write property test for topology metrics
    - **Property 27: Topology metrics correctness** — computed metrics match NetworkX library functions on same graph
    - **Validates: Requirements 17.1**

- [x] 12. Implement main simulation engine and entry point
  - [x] 12.1 Implement SimulationEngine (`manet_sim/core/simulation.py`)
    - Wire all components: ConfigurationManager, SimulationClock, SeedManager, EventQueue, GroupMobilityModel, SpatialIndex, TopologyUpdater, SteeringWriter, EventBus, MetricsCollectors
    - Implement main loop: mobility update → spatial index rebuild → topology update → event drain → metrics publish → output write → clock advance
    - Record seed in output metadata
    - _Requirements: 5.1, 4.2, 4.3_

  - [x] 12.2 Implement main.py entry point
    - Accept config file path as CLI argument (default: `config/sim_config.yaml`)
    - Load config, build SimulationEngine, run simulation
    - Display progress bar via tqdm (completed steps / total steps)
    - Print completion summary: total simulated time, wall-clock duration, records written, output paths
    - Exit code 1 for config errors, exit code 2 for runtime errors
    - _Requirements: 1.3, 1.4, 18.1, 18.2, 18.3, 18.4, 18.5, 18.6_

  - [x] 12.3 Implement Node memory budget validation
    - Verify core state per node ≤ 128 bytes (4×float32 position/velocity + int32 group_id + bool active = 21 bytes)
    - Verify 1000 nodes aggregate core state ≤ 200 KB
    - _Requirements: 6.4_

- [x] 13. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document using Hypothesis
- Unit tests validate specific examples and edge cases
- The implementation language is Python (as specified in the design document)
- All vectorized operations use NumPy float32 arrays for memory efficiency
- The spatial index uses grid-based hashing with cell_size = radio_range for O(n·k) neighbor queries

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2", "1.4", "1.6"] },
    { "id": 2, "tasks": ["1.3", "1.5", "1.7", "3.1", "3.3", "3.5"] },
    { "id": 3, "tasks": ["3.2", "3.4", "3.6", "5.1"] },
    { "id": 4, "tasks": ["5.2", "5.3", "7.1"] },
    { "id": 5, "tasks": ["5.4", "7.2", "7.3"] },
    { "id": 6, "tasks": ["7.4", "9.1"] },
    { "id": 7, "tasks": ["9.2", "9.3"] },
    { "id": 8, "tasks": ["9.4", "11.1", "11.2"] },
    { "id": 9, "tasks": ["11.3", "12.1"] },
    { "id": 10, "tasks": ["12.2", "12.3"] }
  ]
}
```
