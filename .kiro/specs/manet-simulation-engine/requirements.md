# Requirements Document

## Introduction

This document specifies the requirements for the core MANET (Mobile Ad Hoc Network) Simulation Engine — the foundational feature of the NetworkSim project. The engine simulates 1,000 mobile nodes moving across a 10,000 square mile area (100×100 miles) over a 1-hour duration (3,600 time steps at 1-second intervals). Nodes are organized into approximately 20 groups using the Reference Point Group Mobility (RPGM) model. The simulator produces structured steering files (JSON format) capturing node positions, topology changes, and network metrics at each time step.

The engine follows a hybrid time-stepped + event-driven architecture with separation of concerns between mobility, topology, network, and output layers. All simulation runs are reproducible via seeded random number generation and YAML-based configuration.

## Glossary

- **Simulation_Engine**: The top-level orchestrator class that owns the simulation clock, coordinates layer updates, and drives output generation
- **Mobility_Layer**: The subsystem responsible for computing node positions using the RPGM group mobility model
- **Topology_Layer**: The subsystem responsible for managing the network graph, detecting link formation and teardown based on radio range
- **Output_Layer**: The subsystem responsible for serializing simulation state to steering files with buffered writes
- **Metrics_Collector**: The subsystem responsible for collecting and aggregating simulation metrics via an event bus
- **Spatial_Index**: A grid-based or KD-tree data structure used for efficient neighbor discovery, reducing complexity from O(n²) to O(n log n)
- **RPGM**: Reference Point Group Mobility — a group mobility model where each group has a reference point that moves via random waypoint, and member nodes deviate from it within bounded offsets
- **Steering_File**: A structured JSON output file consumed by downstream tools for analysis, visualization, and routing decisions
- **Topology_Delta**: A data structure representing the set of links formed and broken during a single time step
- **Hysteresis**: A buffer zone around the radio range boundary that prevents rapid link state flapping
- **Event_Bus**: A publish-subscribe mechanism for decoupled communication between simulation components and metrics collectors
- **Node**: A mobile network entity with position, velocity, group membership, and transmission parameters
- **Group**: An organizational unit containing member nodes that share a reference point and coordinated movement
- **Configuration_Manager**: The component responsible for loading and validating YAML configuration files

## Requirements

### Requirement 1: Project Structure and Dependencies

**User Story:** As a developer, I want a well-organized Python package structure with all dependencies declared, so that I can build, test, and extend the simulator reliably.

#### Acceptance Criteria

1. THE Simulation_Engine SHALL be organized as a Python package with an `__init__.py` file at the root and separate subpackages (each containing their own `__init__.py`) for core, mobility, topology, network, output, and utils modules
2. THE Simulation_Engine SHALL declare all dependencies in a requirements.txt file including networkx>=3.2.0, numpy>=1.26.0, scipy>=1.11.0, shapely>=2.0.0, pandas>=2.1.0, pyyaml>=6.0.1, and tqdm>=4.66.0
3. THE Simulation_Engine SHALL provide a main.py entry point that loads a YAML configuration file, builds the SimulationEngine instance, and runs the simulation loop to completion
4. IF the configuration file path provided to main.py does not exist or contains invalid YAML, THEN THE Simulation_Engine SHALL exit with a non-zero exit code and print an error message indicating the nature of the configuration failure

### Requirement 2: Configuration Management

**User Story:** As a simulation operator, I want to configure all simulation parameters via YAML files, so that I can run different scenarios without modifying code.

#### Acceptance Criteria

1. THE Configuration_Manager SHALL load simulation parameters from a YAML configuration file specifying duration (1–86,400 seconds), step size (0.1–60.0 seconds), seed (integer), area dimensions (width and height each 1.0–1,000.0 miles), node count (1–10,000), mobility model parameters, network parameters, and output settings
2. IF a configuration file is missing one or more required fields, THEN THE Configuration_Manager SHALL raise an error that lists each missing field by name
3. IF a configuration file contains values outside acceptable ranges, THEN THE Configuration_Manager SHALL raise an error that identifies each invalid field, its provided value, and the acceptable range for that field
4. THE Configuration_Manager SHALL support a default configuration that produces a runnable 1,000-node simulation over 10,000 square miles for 3,600 seconds with a step size of 1.0 second and a deterministic seed
5. IF the specified configuration file path does not exist or is not readable, THEN THE Configuration_Manager SHALL raise an error indicating the file path that could not be accessed
6. IF the configuration file contains syntactically invalid YAML, THEN THE Configuration_Manager SHALL raise an error indicating the parse failure location (line number)

### Requirement 3: Simulation Clock and Time Management

**User Story:** As a simulation developer, I want a deterministic simulation clock with configurable step size, so that the simulation advances predictably and reproducibly.

#### Acceptance Criteria

1. THE Simulation_Engine SHALL maintain a simulation clock that starts at 0.0 seconds and advances in discrete steps of configurable size, where step size is a positive value between 0.001 and 3600.0 seconds inclusive with a default of 1.0 second
2. THE Simulation_Engine SHALL support a configurable simulation duration between 1.0 and 86,400.0 seconds inclusive with a default of 3,600 seconds
3. WHEN the simulation clock reaches or exceeds the configured end time, THE Simulation_Engine SHALL terminate the simulation loop and not execute any further time steps
4. IF the configured step size is less than or equal to zero or exceeds the configured duration, THEN THE Simulation_Engine SHALL reject the configuration at initialization and provide an error message indicating the invalid step size value and acceptable range
5. WHEN the remaining simulation time is less than the configured step size, THE Simulation_Engine SHALL advance the clock by the remaining time to reach exactly the configured end time rather than overshooting or omitting the final partial step

### Requirement 4: Reproducibility via Seeded RNG

**User Story:** As a researcher, I want simulation runs to be exactly reproducible from a stored seed and configuration, so that I can validate results and compare scenarios.

#### Acceptance Criteria

1. THE Simulation_Engine SHALL accept a random seed as a non-negative integer (0 to 2^32 - 1) via configuration and use it to initialize all random number generators used in mobility calculations, topology updates, and initial node placement
2. WHEN the same seed and configuration are provided on the same platform and runtime environment, THE Simulation_Engine SHALL produce bit-for-bit identical steering file output across multiple runs
3. THE Simulation_Engine SHALL record the seed used in the steering file output metadata
4. IF no seed is provided in the configuration, THEN THE Simulation_Engine SHALL generate a seed, use it for the run, and record the generated seed in the steering file output metadata
5. IF the provided seed value is outside the valid range (0 to 2^32 - 1) or is not a non-negative integer, THEN THE Simulation_Engine SHALL reject the configuration with an error message indicating the invalid seed value

### Requirement 5: Hybrid Simulation Loop

**User Story:** As a simulation architect, I want a hybrid time-stepped and event-driven simulation loop, so that mobility updates happen at regular intervals while discrete network events are processed with sub-second precision.

#### Acceptance Criteria

1. THE Simulation_Engine SHALL execute a main loop that processes each configurable time step (default 1 second) in the fixed sequence: mobility update, topology reconciliation, event processing, metrics collection, output writing
2. THE Simulation_Engine SHALL maintain a priority queue for discrete events, where each event is scheduled at a simulation time with millisecond resolution and ordered first by timestamp ascending, then by integer priority value ascending for events sharing the same timestamp
3. WHEN processing a time step, THE Simulation_Engine SHALL drain all events from the queue with timestamps less than or equal to the current step's end time, processing them in priority-queue order, before advancing the simulation clock to the next step
4. IF an event is scheduled with a timestamp earlier than the current simulation time, THEN THE Simulation_Engine SHALL process that event immediately during the current time step's event processing phase without discarding it
5. WHEN no events exist in the queue with timestamps less than or equal to the current time, THE Simulation_Engine SHALL skip the event processing phase and proceed directly to metrics collection

### Requirement 6: Node Data Structure

**User Story:** As a simulation developer, I want a memory-efficient node data structure that tracks position, velocity, group membership, and transmission parameters, so that 1,000 nodes can be simulated without excessive memory consumption.

#### Acceptance Criteria

1. THE Node SHALL store position as a 2D coordinate (x, y) in miles with values ranging from 0.0 to 100.0 per axis, velocity as a 2D vector (vx, vy) in miles per second, an integer group membership ID, transmission range in miles, and a boolean active status flag, using float32 precision for position and velocity components
2. THE Node SHALL support serialization to a dictionary containing the following keys: node_id, group_id, x, y, vx, vy, transmission_range, and active status, with numeric values rounded to no more than 6 decimal places
3. THE Simulation_Engine SHALL maintain node positions as NumPy float32 arrays of shape (N, 2) where N is up to 1,000, for vectorized batch operations
4. THE Node data structure SHALL consume no more than 128 bytes of core state per node instance (excluding history), so that 1,000 nodes require no more than 200 KB of aggregate core state memory

### Requirement 7: Group Data Structure

**User Story:** As a simulation developer, I want a group data structure that manages member nodes, a reference point, and a waypoint queue, so that coordinated group movement can be modeled.

#### Acceptance Criteria

1. THE Group SHALL maintain a reference point position (2D coordinates), a reference velocity (2D vector), a list of member node IDs (1 to 1000 members), a designated leader node ID selected from the member list, and a waypoint queue ordered for first-in-first-out retrieval with a capacity of 1 to 100 waypoints
2. WHEN a node ID is added to a Group, THE Group SHALL append that node ID to its member list and confirm the addition
3. WHEN a node ID that is already in the Group's member list is added, THE Group SHALL reject the addition and indicate that the node is already a member
4. WHEN a member node that is not the leader is removed from a Group, THE Group SHALL remove that node ID from its member list
5. WHEN the leader node is removed from a Group that has at least one other member, THE Group SHALL remove the leader, promote the longest-tenured remaining member to leader, and update the designated leader node ID
6. IF the last remaining member is removed from a Group, THEN THE Group SHALL become empty with no leader and no members

### Requirement 8: RPGM Group Mobility Model

**User Story:** As a simulation operator, I want the Reference Point Group Mobility model implemented, so that nodes move in coordinated groups with realistic cluster behavior.

#### Acceptance Criteria

1. THE Mobility_Layer SHALL organize 1,000 nodes into 20 groups of 50 nodes each
2. THE Mobility_Layer SHALL move each group reference point via random waypoint navigation: select a destination within the simulation area, move at a speed selected uniformly at random between 0.5 and 13.4 meters per second, pause for a duration selected uniformly at random between 0 and 300 seconds upon arrival, then select a new destination
3. THE Mobility_Layer SHALL position each member node at the group reference point plus a random deviation sampled from a configurable distribution (uniform disk or Gaussian), where the deviation magnitude shall not exceed 1.25 miles from the reference point
4. THE Mobility_Layer SHALL clamp all node positions to remain within the simulation area boundaries (0 to 100 miles on each axis)
5. WHEN a group exhausts its waypoint queue, THE Mobility_Layer SHALL generate at least 1 new waypoint to continue movement without interruption
6. IF a member node's computed position (reference point plus deviation) falls outside the simulation area boundaries, THEN THE Mobility_Layer SHALL clamp the node position to the nearest point within the boundary before recording the position

### Requirement 9: Vectorized Position Updates

**User Story:** As a performance engineer, I want position updates computed using NumPy vectorized operations, so that batch updates for 1,000 nodes execute with minimal Python interpreter overhead.

#### Acceptance Criteria

1. THE Mobility_Layer SHALL store all node positions in a NumPy array of shape (N, 2) with float32 or float64 dtype, and store all node velocities in a NumPy array of shape (N, 2) with the same dtype
2. THE Mobility_Layer SHALL compute position updates for all N nodes in a single vectorized array operation (positions += velocities * dt) without iterating over individual nodes in a Python-level loop, completing a batch update of 1,000 nodes in under 1 millisecond of wall-clock time
3. THE Mobility_Layer SHALL apply boundary enforcement using vectorized clipping operations that constrain all node positions to within the configured simulation area bounds (0 to world_width on the x-axis, 0 to world_height on the y-axis)
4. WHEN a node position reaches or exceeds a simulation area boundary after a position update, THEN THE Mobility_Layer SHALL negate the corresponding velocity component for that node using a vectorized boolean mask operation, so that the node reflects off the boundary

### Requirement 10: Spatial Indexing for Neighbor Discovery

**User Story:** As a performance engineer, I want spatial indexing for neighbor queries, so that link detection scales sub-linearly rather than requiring O(n²) pairwise distance checks.

#### Acceptance Criteria

1. THE Spatial_Index SHALL support radius queries that return all node IDs whose positions are within a specified distance of a query point, excluding the querying node itself, with zero false negatives (every node within the distance is returned)
2. WHEN a simulation time step completes, THE Spatial_Index SHALL be rebuilt or updated to reflect all current node positions before any neighbor queries are issued for that step
3. THE Spatial_Index SHALL reduce neighbor discovery complexity from O(n²) to O(n log n) or O(n·k) where k is the average number of neighbors per cell
4. THE Spatial_Index SHALL use either a grid-based spatial hash with cell size equal to radio range, or a scipy cKDTree rebuilt each step
5. IF no nodes exist within the specified query distance, THEN THE Spatial_Index SHALL return an empty collection

### Requirement 11: Network Topology Graph

**User Story:** As a simulation developer, I want a NetworkX-based topology graph that tracks active radio links, so that connectivity analysis and steering file output have an accurate representation of the network state.

#### Acceptance Criteria

1. THE Topology_Layer SHALL maintain a NetworkX undirected graph where nodes represent mobile entities and edges represent active radio links, supporting up to 1000 nodes and up to 10000 concurrent edges
2. THE Topology_Layer SHALL store node attributes including position as an (x, y) coordinate pair in miles, integer group ID, and velocity as a (vx, vy) vector in miles per second on each graph node
3. THE Topology_Layer SHALL store edge attributes including distance in miles between the two endpoint nodes, formation time as the simulation timestamp in seconds when the edge was created, and link quality as a normalized float value in the range 0.0 to 1.0 where 1.0 represents minimum distance and 0.0 represents maximum radio range distance
4. WHEN two nodes transition from distance greater than the configured radio range to distance less than or equal to the configured radio range, THE Topology_Layer SHALL add an edge between those nodes with distance set to the current inter-node distance, formation time set to the current simulation timestamp, and link quality computed from the distance
5. WHEN two connected nodes transition from distance less than or equal to the configured radio range to distance greater than the configured radio range, THE Topology_Layer SHALL remove the edge between those nodes
6. WHEN node positions are updated at each simulation time step, THE Topology_Layer SHALL update the position and velocity attributes of the corresponding graph nodes within the same time step

### Requirement 12: Link Formation and Teardown with Hysteresis

**User Story:** As a network modeler, I want link state transitions governed by radio range with hysteresis, so that links form and break realistically without rapid flapping near the range boundary.

#### Acceptance Criteria

1. WHEN the distance between two nodes falls below the inner threshold (radio range minus hysteresis margin, where hysteresis margin defaults to 10% of radio range), THE Topology_Layer SHALL form a link between the nodes and transition the link state from ABSENT to ACTIVE
2. WHEN the distance between two nodes exceeds the outer threshold (radio range plus hysteresis margin), THE Topology_Layer SHALL break the link between the nodes and transition the link state from ACTIVE or HYSTERESIS to ABSENT
3. WHILE the distance between two linked nodes is between the inner and outer thresholds, THE Topology_Layer SHALL maintain the existing link state unchanged, keeping ACTIVE links active and not forming new links for ABSENT pairs
4. THE Topology_Layer SHALL compute topology changes incrementally as a delta containing the list of newly formed links and the list of broken links for each time step, rather than rebuilding the entire graph each step
5. WHEN the simulation initializes, THE Topology_Layer SHALL evaluate all node pairs against the inner threshold and form links for any pair whose distance is below the inner threshold, establishing the initial topology before the first simulation step
6. IF a previously unlinked node pair enters the zone between the inner and outer thresholds without first crossing the inner threshold, THEN THE Topology_Layer SHALL keep the link state as ABSENT until the distance falls below the inner threshold

### Requirement 13: Steering File Output

**User Story:** As a downstream tool consumer, I want structured JSON steering files produced by the simulation, so that I can analyze node trajectories, topology evolution, and network metrics.

#### Acceptance Criteria

1. THE Output_Layer SHALL write simulation state to JSON-formatted steering files containing node positions, velocities, group memberships, and topology deltas (links formed and links broken, each identified by the pair of node IDs and the simulation timestamp of the event) at each time step
2. THE Output_Layer SHALL include a metadata header in each output file containing at minimum: simulation seed, all user-specified configuration parameters, simulation start time (ISO 8601), simulation end time, total time steps, node count, and schema version (in semantic versioning format)
3. THE Output_Layer SHALL write full topology snapshots at a configurable interval (default: every 60 seconds, minimum: 1 second) where a full snapshot includes the complete node adjacency list and each node's current position, velocity, and group membership
4. THE Output_Layer SHALL buffer output writes and flush to disk at configurable intervals (default: every 100 steps, maximum: 1000 steps) so that no more than the configured buffer interval of data is held in memory at any time
5. IF a flush to disk fails, THEN THE Output_Layer SHALL retain the buffered data, retry the write once, and if the retry fails, log an error message indicating the failure reason and the number of lost records, then continue simulation execution
6. WHEN the simulation completes or is terminated, THE Output_Layer SHALL flush all remaining buffered data to disk and write a valid JSON closing structure so that the output file is parseable

### Requirement 14: Steering File Parsing and Round-Trip

**User Story:** As a developer, I want to parse steering files back into simulation state objects, so that I can validate output correctness and resume simulations from checkpoints.

#### Acceptance Criteria

1. THE Output_Layer SHALL provide a parser that reads a JSON steering file and reconstructs node position, velocity, and topology state for a specified recorded time step, where position values are preserved to at least 6 decimal places and velocity values to at least 4 decimal places
2. IF the parser is invoked with a time step that does not exist in the steering file, THEN THE Output_Layer SHALL return an error indication specifying that the requested time step was not found, without modifying any existing state
3. IF the parser receives malformed JSON or a file that does not conform to the steering file schema, THEN THE Output_Layer SHALL return an error indication describing the parsing failure, without producing a partial state object
4. THE Output_Layer SHALL provide a pretty-printer that formats simulation state objects into valid JSON steering files using 2-space indentation and sorted keys for deterministic output
5. THE Output_Layer SHALL satisfy the round-trip property: for any valid simulation state, parsing a steering file then printing it then parsing the result SHALL produce a state object that is field-by-field equal to the original, with numeric values matching within a tolerance of 1e-6

### Requirement 15: Event Bus for Metrics Collection

**User Story:** As a simulation analyst, I want metrics collected via a decoupled event bus, so that collection can be enabled or disabled without modifying simulation logic.

#### Acceptance Criteria

1. THE Event_Bus SHALL support publish-subscribe semantics where simulation components publish events by type and zero or more collectors subscribe to receive events of specified types, with events delivered to subscribers in the order they were published
2. THE Event_Bus SHALL support the following event types: position_update, link_formed, link_broken, step_complete, and simulation_end
3. THE Metrics_Collector SHALL subscribe to events solely through the Event_Bus subscribe interface, requiring no modifications to simulation core code to add, remove, or replace collectors
4. WHEN a subscriber raises an exception during event handling, THE Event_Bus SHALL continue delivering that event to remaining subscribers and SHALL not interrupt the simulation execution
5. THE Event_Bus SHALL provide an enable and disable mechanism per subscriber that stops event delivery to disabled subscribers without removing their subscriptions, allowing collection to be toggled at runtime without modifying simulation components
6. WHEN an event is published, THE Event_Bus SHALL deliver the event payload containing at minimum the event type identifier and a simulation timestamp to each active subscriber registered for that event type

### Requirement 16: Mobility Metrics

**User Story:** As a simulation analyst, I want mobility metrics computed at configurable intervals, so that I can assess group cohesion, speed distributions, and displacement patterns.

#### Acceptance Criteria

1. THE Metrics_Collector SHALL compute mobility metrics at a configurable snapshot interval between 1 and 3600 seconds (default 60 seconds), where each snapshot includes: average speed (mph), speed standard deviation (mph), maximum speed (mph), average Euclidean displacement from each node's initial position (miles), and average group cohesion defined as the mean distance from each group member to its group centroid (miles)
2. IF a group contains fewer than 2 members at snapshot time, THEN THE Metrics_Collector SHALL report a group cohesion value of 0.0 for that group
3. WHEN the simulation completes, THE Metrics_Collector SHALL export all collected mobility snapshots to a CSV file with a header row containing columns: timestamp, avg_speed, std_speed, max_speed, avg_displacement, avg_group_cohesion, with numeric values rounded to 4 decimal places

### Requirement 17: Topology Metrics

**User Story:** As a network analyst, I want topology metrics computed at configurable intervals, so that I can assess network connectivity, degree distribution, and partition behavior.

#### Acceptance Criteria

1. THE Metrics_Collector SHALL compute topology metrics including node count, edge count, average degree, maximum degree, number of connected components, largest component size, and average clustering coefficient at configurable snapshot intervals between 1 and 3600 seconds (default: 300 seconds)
2. THE Metrics_Collector SHALL include a simulation timestamp with each topology metrics snapshot to enable time-series correlation
3. WHEN a topology metrics snapshot is computed, THE Metrics_Collector SHALL export the snapshot to a JSON file containing all computed metric fields and the associated timestamp
4. IF the topology graph contains zero nodes at snapshot time, THEN THE Metrics_Collector SHALL record node count as 0, edge count as 0, average degree as 0.0, maximum degree as 0, number of connected components as 0, largest component size as 0, and average clustering coefficient as 0.0

### Requirement 18: Entry Point and Execution

**User Story:** As a simulation operator, I want a command-line entry point that loads configuration and runs the simulation with progress reporting, so that I can execute simulations and monitor their progress.

#### Acceptance Criteria

1. THE Simulation_Engine SHALL provide a main.py entry point that accepts a configuration file path as a command-line argument
2. WHEN no configuration file path is provided, THE Simulation_Engine SHALL use the default path "config/sim_config.yaml" relative to the working directory
3. THE Simulation_Engine SHALL display a progress bar during execution indicating the number of completed time steps out of the total time steps configured for the simulation
4. WHEN the simulation completes, THE Simulation_Engine SHALL print a summary including total simulated time, wall-clock duration, number of steering file records written, and output file paths
5. IF the specified configuration file does not exist or cannot be parsed, THEN THE Simulation_Engine SHALL exit with a non-zero exit code and print an error message indicating the file path and the nature of the failure
6. IF the simulation encounters an unrecoverable error during execution, THEN THE Simulation_Engine SHALL exit with a non-zero exit code and print an error message indicating the simulation time at which the failure occurred
