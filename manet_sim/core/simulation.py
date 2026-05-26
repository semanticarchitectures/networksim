"""Simulation engine: main loop orchestration for the MANET simulator.

Wires all components together and executes the hybrid time-stepped + event-driven
simulation loop. The main loop sequence per step:
1. mobility_model.step(t, dt) → positions
2. spatial_index.rebuild(positions)
3. topology_updater.update(positions, t) → delta
4. event_queue.drain_until(t)
5. event_bus.publish(step_complete, ...)
6. steering_writer.write_step(t, positions, velocities, group_ids, delta)
7. clock.advance()
"""

from __future__ import annotations

import time

from manet_sim.core.clock import SimulationClock
from manet_sim.core.config import SimulationConfig
from manet_sim.core.event_bus import (
    LINK_BROKEN,
    LINK_FORMED,
    POSITION_UPDATE,
    SIMULATION_END,
    STEP_COMPLETE,
    EventBus,
    LinkBrokenEvent,
    LinkFormedEvent,
    PositionUpdateEvent,
    SimulationEndEvent,
    StepCompleteEvent,
)
from manet_sim.core.event_queue import EventQueue
from manet_sim.core.seed_manager import SeedManager
from manet_sim.mobility.group_mobility import GroupMobilityModel
from manet_sim.mobility.spatial_index import GridSpatialIndex
from manet_sim.network.metrics import MobilityMetricsCollector, TopologyMetricsCollector
from manet_sim.output.steering_writer import SteeringWriter
from manet_sim.topology.topology_updater import TopologyUpdater


class SimulationEngine:
    """Orchestrates the MANET simulation across all layers.

    Uses a hybrid time-stepped + event-driven design. Coarse time steps
    (configurable, default 1 second) drive mobility updates and topology
    reconciliation. An event queue handles discrete network events between
    steps at sub-second precision.

    Components wired:
    - SimulationClock: deterministic time advancement
    - SeedManager: reproducible RNG streams
    - EventQueue: discrete event scheduling
    - EventBus: pub/sub for metrics collection
    - GroupMobilityModel: RPGM node movement
    - GridSpatialIndex: efficient neighbor discovery
    - TopologyUpdater: incremental link management with hysteresis
    - SteeringWriter: buffered JSON output
    - MobilityMetricsCollector: mobility statistics
    - TopologyMetricsCollector: topology statistics
    """

    def __init__(self, config: SimulationConfig) -> None:
        """Initialize the simulation engine from configuration.

        Args:
            config: Validated SimulationConfig instance.
        """
        self._config = config

        # Core infrastructure
        self._seed_manager = SeedManager(seed=config.seed)
        self._clock = SimulationClock(
            start=0.0,
            end=config.duration_seconds,
            step_size=config.step_size,
        )
        self._event_queue = EventQueue()
        self._event_bus = EventBus()

        # Mobility layer
        self._mobility_model = GroupMobilityModel(
            config=config.mobility,
            rng=self._seed_manager.get_rng("mobility"),
        )

        # Spatial index (cell_size = radio_range for optimal neighbor queries)
        self._spatial_index = GridSpatialIndex(
            width=config.area_width,
            height=config.area_height,
            cell_size=config.network.radio_range_miles,
        )

        # Topology layer
        self._topology_updater = TopologyUpdater(config=config.network)

        # Output layer
        self._steering_writer = SteeringWriter(
            config=config.output,
            metadata={
                "seed": self._seed_manager.seed,
                "config": self._config_to_dict(config),
                "total_steps": int(config.duration_seconds / config.step_size),
                "node_count": config.node_count,
                "transmission_range": config.network.radio_range_miles,
            },
        )

        # Metrics collectors (subscribe to event bus)
        self._mobility_metrics = MobilityMetricsCollector(
            snapshot_interval=60.0,
            output_dir=config.output.output_dir,
            event_bus=self._event_bus,
        )
        self._topology_metrics = TopologyMetricsCollector(
            snapshot_interval=300.0,
            output_dir=config.output.output_dir,
            event_bus=self._event_bus,
        )

        # Step counter
        self._step_number: int = 0
        self._wall_clock_start: float = 0.0

    def run(self) -> None:
        """Execute the full simulation loop.

        Initializes all components, runs the main loop until the clock
        is finished, then finalizes output and publishes simulation_end.
        """
        self._initialize()

        self._wall_clock_start = time.perf_counter()

        while not self._clock.is_finished():
            step_start = time.perf_counter()
            t = self._clock.current_time
            dt = self._clock.step_size

            # 1. Mobility update: advance all node positions
            positions = self._mobility_model.step(t, dt)
            velocities = self._mobility_model.get_velocities()
            group_ids = self._mobility_model.get_group_ids()

            # 2. Spatial index rebuild
            self._spatial_index.rebuild(positions)

            # 3. Topology update: compute link changes with hysteresis
            delta = self._topology_updater.update(positions, t)

            # Update node attributes on the topology graph
            self._topology_updater.update_node_attributes(
                positions, group_ids, velocities
            )

            # 4. Event queue drain: process discrete events up to current time
            self._event_queue.drain_until(t)

            # 5. Publish events via event bus
            # Publish position update for metrics collectors
            self._event_bus.publish(
                POSITION_UPDATE,
                PositionUpdateEvent(
                    timestamp=t,
                    positions=positions,
                    velocities=velocities,
                ),
            )

            # Publish link events for topology metrics
            for node_a, node_b, distance in delta.new_links:
                self._event_bus.publish(
                    LINK_FORMED,
                    LinkFormedEvent(
                        timestamp=t,
                        node_a=node_a,
                        node_b=node_b,
                        distance=distance,
                    ),
                )
            for item in delta.broken_links:
                node_a, node_b = item[0], item[1]
                self._event_bus.publish(
                    LINK_BROKEN,
                    LinkBrokenEvent(
                        timestamp=t,
                        node_a=node_a,
                        node_b=node_b,
                    ),
                )

            # Publish step_complete
            step_wall_ms = (time.perf_counter() - step_start) * 1000.0
            active_links = self._topology_updater.get_graph().number_of_edges()
            self._event_bus.publish(
                STEP_COMPLETE,
                StepCompleteEvent(
                    timestamp=t,
                    step_number=self._step_number,
                    active_links=active_links,
                    wall_clock_ms=step_wall_ms,
                ),
            )

            # 6. Output: write step data to steering file
            self._steering_writer.write_step(
                t, positions, velocities, group_ids, delta
            )

            # 7. Advance clock
            self._clock.advance()
            self._step_number += 1

        self._finalize()

    def _initialize(self) -> None:
        """Initialize all components before the simulation loop.

        - Initialize mobility model (place nodes, create groups)
        - Build initial spatial index
        - Initialize topology (form initial links)
        - Set up metrics collectors
        - Open steering writer
        """
        # Initialize mobility model
        self._mobility_model.initialize(
            node_count=self._config.node_count,
            area_width=self._config.area_width,
            area_height=self._config.area_height,
        )

        # Get initial positions
        positions = self._mobility_model.get_positions()
        group_ids = self._mobility_model.get_group_ids()

        # Build initial spatial index
        self._spatial_index.rebuild(positions)

        # Initialize topology (forms initial links based on inner threshold)
        initial_delta = self._topology_updater.initialize(
            positions,
            area_width=self._config.area_width,
            area_height=self._config.area_height,
        )

        # Update node attributes on topology graph
        self._topology_updater.update_node_attributes(
            positions,
            group_ids,
            self._mobility_model.get_velocities(),
        )

        # Set up metrics collectors with initial state
        self._mobility_metrics.set_group_ids(group_ids)
        self._topology_metrics.set_node_count(self._config.node_count)

        # Publish initial link events so topology metrics collector is aware
        for node_a, node_b, distance in initial_delta.new_links:
            self._event_bus.publish(
                LINK_FORMED,
                LinkFormedEvent(
                    timestamp=0.0,
                    node_a=node_a,
                    node_b=node_b,
                    distance=distance,
                ),
            )

        # Open steering writer
        self._steering_writer.open()

    def _finalize(self) -> None:
        """Finalize the simulation after the loop completes.

        - Publish simulation_end event
        - Close steering writer (flushes remaining buffer)
        """
        wall_clock_seconds = time.perf_counter() - self._wall_clock_start

        # Publish simulation_end event
        self._event_bus.publish(
            SIMULATION_END,
            SimulationEndEvent(
                total_time=self._config.duration_seconds,
                total_steps=self._step_number,
                wall_clock_seconds=wall_clock_seconds,
            ),
        )

        # Close steering writer (flushes remaining buffer, writes valid JSON)
        self._steering_writer.close()

    @property
    def seed(self) -> int:
        """Return the seed used for this simulation run."""
        return self._seed_manager.seed

    @property
    def clock(self) -> SimulationClock:
        """Return the simulation clock."""
        return self._clock

    @property
    def event_bus(self) -> EventBus:
        """Return the event bus."""
        return self._event_bus

    @property
    def event_queue(self) -> EventQueue:
        """Return the event queue."""
        return self._event_queue

    @property
    def steps_completed(self) -> int:
        """Return the number of steps completed."""
        return self._step_number

    @staticmethod
    def _config_to_dict(config: SimulationConfig) -> dict:
        """Convert SimulationConfig to a serializable dictionary."""
        return {
            "simulation": {
                "duration_seconds": config.duration_seconds,
                "step_size": config.step_size,
                "seed": config.seed,
                "area": {
                    "width_miles": config.area_width,
                    "height_miles": config.area_height,
                },
            },
            "nodes": {
                "count": config.node_count,
            },
            "mobility": {
                "model": config.mobility.model,
                "num_groups": config.mobility.num_groups,
                "group_speed_min_mps": config.mobility.group_speed_min_mps,
                "group_speed_max_mps": config.mobility.group_speed_max_mps,
                "pause_min_seconds": config.mobility.pause_min_seconds,
                "pause_max_seconds": config.mobility.pause_max_seconds,
                "max_deviation_miles": config.mobility.max_deviation_miles,
                "deviation_model": config.mobility.deviation_model,
            },
            "network": {
                "radio_range_miles": config.network.radio_range_miles,
                "hysteresis_margin_pct": config.network.hysteresis_margin_pct,
            },
            "output": {
                "format": config.output.format,
                "output_dir": config.output.output_dir,
                "snapshot_interval": config.output.snapshot_interval,
                "buffer_size": config.output.buffer_size,
            },
        }
