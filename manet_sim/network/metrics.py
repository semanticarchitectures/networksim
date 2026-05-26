"""Metrics collectors for mobility and topology analysis.

Collectors subscribe to EventBus events and compute aggregate metrics
at configurable snapshot intervals. Results are exported to CSV (mobility)
and JSON (topology) files.
"""

import csv
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np

from manet_sim.core.event_bus import (
    EventBus,
    LinkBrokenEvent,
    LinkFormedEvent,
    PositionUpdateEvent,
    SimulationEndEvent,
    StepCompleteEvent,
    LINK_BROKEN,
    LINK_FORMED,
    POSITION_UPDATE,
    SIMULATION_END,
    STEP_COMPLETE,
)


# --- Mobility Metrics ---


@dataclass
class MobilitySnapshot:
    """A single mobility metrics snapshot at a point in time."""

    timestamp: float
    avg_speed: float
    std_speed: float
    max_speed: float
    avg_displacement: float
    avg_group_cohesion: float


class MobilityMetricsCollector:
    """Collects mobility metrics at configurable snapshot intervals.

    Subscribes to position_update, step_complete, and simulation_end events
    via the EventBus. Computes snapshots including average speed, speed
    standard deviation, max speed, average displacement from initial position,
    and average group cohesion.

    Group cohesion is defined as the mean distance from each group member
    to its group centroid. Groups with fewer than 2 members report 0.0 cohesion.
    """

    def __init__(
        self,
        snapshot_interval: float = 60.0,
        output_dir: str = "output",
        event_bus: Optional[EventBus] = None,
    ) -> None:
        """Initialize the mobility metrics collector.

        Args:
            snapshot_interval: Seconds between metric snapshots (default 60).
            output_dir: Directory for CSV output file.
            event_bus: EventBus instance to subscribe to. If provided,
                       automatically subscribes to relevant events.
        """
        self._interval = snapshot_interval
        self._output_dir = Path(output_dir)
        self._snapshots: list[MobilitySnapshot] = []

        # Current state tracking
        self._positions: Optional[np.ndarray] = None  # (N, 2)
        self._velocities: Optional[np.ndarray] = None  # (N, 2)
        self._initial_positions: Optional[np.ndarray] = None  # (N, 2)
        self._group_ids: Optional[np.ndarray] = None  # (N,)

        # Snapshot timing
        self._next_snapshot_time: float = snapshot_interval

        # Subscribe to events if bus provided
        if event_bus is not None:
            self.subscribe(event_bus)

    def subscribe(self, event_bus: EventBus) -> None:
        """Subscribe to relevant events on the given EventBus."""
        event_bus.subscribe(POSITION_UPDATE, self._on_position_update)
        event_bus.subscribe(STEP_COMPLETE, self._on_step_complete)
        event_bus.subscribe(SIMULATION_END, self._on_simulation_end)

    def set_group_ids(self, group_ids: np.ndarray) -> None:
        """Set the group membership array for cohesion calculations.

        Args:
            group_ids: Array of shape (N,) with integer group IDs.
        """
        self._group_ids = group_ids.copy()

    def _on_position_update(self, event: PositionUpdateEvent) -> None:
        """Handle position_update event."""
        self._positions = event.positions
        self._velocities = event.velocities

        # Record initial positions on first update
        if self._initial_positions is None:
            self._initial_positions = event.positions.copy()

    def _on_step_complete(self, event: StepCompleteEvent) -> None:
        """Handle step_complete event — check if snapshot is due."""
        if event.timestamp >= self._next_snapshot_time:
            snapshot = self._compute_snapshot(event.timestamp)
            self._snapshots.append(snapshot)
            self._next_snapshot_time += self._interval

    def _on_simulation_end(self, event: SimulationEndEvent) -> None:
        """Handle simulation_end event — export results."""
        self.export_to_csv()

    def _compute_snapshot(self, timestamp: float) -> MobilitySnapshot:
        """Compute a mobility metrics snapshot at the given timestamp."""
        # Handle case where no position data is available
        if self._positions is None or len(self._positions) == 0:
            return MobilitySnapshot(
                timestamp=timestamp,
                avg_speed=0.0,
                std_speed=0.0,
                max_speed=0.0,
                avg_displacement=0.0,
                avg_group_cohesion=0.0,
            )

        # Compute speeds from velocities
        if self._velocities is not None:
            speeds = np.linalg.norm(self._velocities, axis=1)
        else:
            speeds = np.zeros(len(self._positions))

        avg_speed = float(np.mean(speeds)) if len(speeds) > 0 else 0.0
        std_speed = float(np.std(speeds)) if len(speeds) > 0 else 0.0
        max_speed = float(np.max(speeds)) if len(speeds) > 0 else 0.0

        # Compute displacement from initial positions
        avg_displacement = 0.0
        if self._initial_positions is not None:
            displacements = np.linalg.norm(
                self._positions - self._initial_positions, axis=1
            )
            avg_displacement = float(np.mean(displacements))

        # Compute group cohesion
        avg_group_cohesion = self._compute_avg_group_cohesion()

        return MobilitySnapshot(
            timestamp=timestamp,
            avg_speed=avg_speed,
            std_speed=std_speed,
            max_speed=max_speed,
            avg_displacement=avg_displacement,
            avg_group_cohesion=avg_group_cohesion,
        )

    def _compute_avg_group_cohesion(self) -> float:
        """Compute average group cohesion across all groups.

        Cohesion for a group is the mean distance from each member to the
        group centroid. Groups with fewer than 2 members have cohesion 0.0.

        Returns:
            Average cohesion across all groups.
        """
        if self._positions is None or self._group_ids is None:
            return 0.0

        unique_groups = np.unique(self._group_ids)
        cohesions: list[float] = []

        for gid in unique_groups:
            mask = self._group_ids == gid
            group_positions = self._positions[mask]

            if len(group_positions) < 2:
                cohesions.append(0.0)
                continue

            centroid = group_positions.mean(axis=0)
            distances = np.linalg.norm(group_positions - centroid, axis=1)
            cohesions.append(float(distances.mean()))

        if not cohesions:
            return 0.0

        return float(np.mean(cohesions))

    def export_to_csv(self) -> Path:
        """Export all collected snapshots to a CSV file.

        Returns:
            Path to the written CSV file.
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._output_dir / "mobility_metrics.csv"

        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "avg_speed",
                "std_speed",
                "max_speed",
                "avg_displacement",
                "avg_group_cohesion",
            ])
            for snap in self._snapshots:
                writer.writerow([
                    f"{snap.timestamp:.4f}",
                    f"{snap.avg_speed:.4f}",
                    f"{snap.std_speed:.4f}",
                    f"{snap.max_speed:.4f}",
                    f"{snap.avg_displacement:.4f}",
                    f"{snap.avg_group_cohesion:.4f}",
                ])

        return out_path

    @property
    def snapshots(self) -> list[MobilitySnapshot]:
        """Return collected snapshots (for testing)."""
        return self._snapshots


# --- Topology Metrics ---


@dataclass
class TopologySnapshot:
    """A single topology metrics snapshot at a point in time."""

    timestamp: float
    node_count: int
    edge_count: int
    avg_degree: float
    max_degree: int
    num_connected_components: int
    largest_component_size: int
    avg_clustering_coefficient: float


class TopologyMetricsCollector:
    """Collects topology metrics at configurable snapshot intervals.

    Subscribes to link_formed, link_broken, step_complete, and simulation_end
    events via the EventBus. Maintains an internal NetworkX graph that is
    updated incrementally as links form and break.

    Computes snapshots including node count, edge count, average degree,
    maximum degree, number of connected components, largest component size,
    and average clustering coefficient.
    """

    def __init__(
        self,
        snapshot_interval: float = 300.0,
        output_dir: str = "output",
        event_bus: Optional[EventBus] = None,
    ) -> None:
        """Initialize the topology metrics collector.

        Args:
            snapshot_interval: Seconds between metric snapshots (default 300).
            output_dir: Directory for JSON output file.
            event_bus: EventBus instance to subscribe to. If provided,
                       automatically subscribes to relevant events.
        """
        self._interval = snapshot_interval
        self._output_dir = Path(output_dir)
        self._snapshots: list[TopologySnapshot] = []

        # Internal graph maintained via link events
        self._graph: nx.Graph = nx.Graph()

        # Snapshot timing
        self._next_snapshot_time: float = snapshot_interval

        # Subscribe to events if bus provided
        if event_bus is not None:
            self.subscribe(event_bus)

    def subscribe(self, event_bus: EventBus) -> None:
        """Subscribe to relevant events on the given EventBus."""
        event_bus.subscribe(LINK_FORMED, self._on_link_formed)
        event_bus.subscribe(LINK_BROKEN, self._on_link_broken)
        event_bus.subscribe(STEP_COMPLETE, self._on_step_complete)
        event_bus.subscribe(SIMULATION_END, self._on_simulation_end)

    def set_node_count(self, node_count: int) -> None:
        """Set the number of nodes in the topology graph.

        Adds nodes 0..node_count-1 to the internal graph so that
        isolated nodes are counted in metrics.

        Args:
            node_count: Total number of nodes in the simulation.
        """
        for i in range(node_count):
            if not self._graph.has_node(i):
                self._graph.add_node(i)

    def _on_link_formed(self, event: LinkFormedEvent) -> None:
        """Handle link_formed event — add edge to internal graph."""
        # Ensure nodes exist
        if not self._graph.has_node(event.node_a):
            self._graph.add_node(event.node_a)
        if not self._graph.has_node(event.node_b):
            self._graph.add_node(event.node_b)
        self._graph.add_edge(event.node_a, event.node_b, distance=event.distance)

    def _on_link_broken(self, event: LinkBrokenEvent) -> None:
        """Handle link_broken event — remove edge from internal graph."""
        if self._graph.has_edge(event.node_a, event.node_b):
            self._graph.remove_edge(event.node_a, event.node_b)

    def _on_step_complete(self, event: StepCompleteEvent) -> None:
        """Handle step_complete event — check if snapshot is due."""
        if event.timestamp >= self._next_snapshot_time:
            snapshot = self._compute_snapshot(event.timestamp)
            self._snapshots.append(snapshot)
            self._next_snapshot_time += self._interval

    def _on_simulation_end(self, event: SimulationEndEvent) -> None:
        """Handle simulation_end event — export results."""
        self.export_to_json()

    def _compute_snapshot(self, timestamp: float) -> TopologySnapshot:
        """Compute a topology metrics snapshot at the given timestamp."""
        node_count = self._graph.number_of_nodes()
        edge_count = self._graph.number_of_edges()

        # Handle zero-node graph
        if node_count == 0:
            return TopologySnapshot(
                timestamp=timestamp,
                node_count=0,
                edge_count=0,
                avg_degree=0.0,
                max_degree=0,
                num_connected_components=0,
                largest_component_size=0,
                avg_clustering_coefficient=0.0,
            )

        # Degree metrics
        degrees = [d for _, d in self._graph.degree()]
        avg_degree = float(np.mean(degrees)) if degrees else 0.0
        max_degree = max(degrees) if degrees else 0

        # Connected components
        components = list(nx.connected_components(self._graph))
        num_connected_components = len(components)
        largest_component_size = max(len(c) for c in components) if components else 0

        # Clustering coefficient
        avg_clustering_coefficient = float(nx.average_clustering(self._graph))

        return TopologySnapshot(
            timestamp=timestamp,
            node_count=node_count,
            edge_count=edge_count,
            avg_degree=avg_degree,
            max_degree=max_degree,
            num_connected_components=num_connected_components,
            largest_component_size=largest_component_size,
            avg_clustering_coefficient=avg_clustering_coefficient,
        )

    def export_to_json(self) -> Path:
        """Export all collected snapshots to a JSON file.

        Returns:
            Path to the written JSON file.
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._output_dir / "topology_metrics.json"

        data = []
        for snap in self._snapshots:
            data.append({
                "timestamp": snap.timestamp,
                "node_count": snap.node_count,
                "edge_count": snap.edge_count,
                "avg_degree": snap.avg_degree,
                "max_degree": snap.max_degree,
                "num_connected_components": snap.num_connected_components,
                "largest_component_size": snap.largest_component_size,
                "avg_clustering_coefficient": snap.avg_clustering_coefficient,
            })

        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)

        return out_path

    @property
    def snapshots(self) -> list[TopologySnapshot]:
        """Return collected snapshots (for testing)."""
        return self._snapshots

    @property
    def graph(self) -> nx.Graph:
        """Return the internal graph (for testing)."""
        return self._graph
