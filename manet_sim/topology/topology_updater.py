"""Topology updater for MANET simulation.

Reconciles node positions into topology changes using a NetworkX undirected
graph and hysteresis-based link management. Computes incremental topology
deltas (new links formed, links broken) rather than rebuilding the full
graph each step.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from manet_sim.core.config import NetworkConfig
from manet_sim.mobility.spatial_index import GridSpatialIndex
from manet_sim.topology.link_manager import LinkManager, LinkState


@dataclass
class TopologyDelta:
    """Represents topology changes for a single time step.

    Attributes:
        timestamp: Simulation time when the delta was computed.
        new_links: List of (node_a, node_b, distance) for newly formed links.
        broken_links: List of (node_a, node_b) for links that were broken.
    """

    timestamp: float
    new_links: list[tuple[int, int, float]] = field(default_factory=list)
    broken_links: list[tuple[int, int]] = field(default_factory=list)


class TopologyUpdater:
    """Manages the network topology graph with incremental updates.

    Uses a GridSpatialIndex for efficient pair discovery and a LinkManager
    for hysteresis-based link state transitions. Maintains a NetworkX
    undirected graph with node and edge attributes.

    Node attributes stored on graph:
        - position: (x, y) tuple in miles
        - group_id: integer group membership
        - velocity: (vx, vy) tuple in miles/second

    Edge attributes stored on graph:
        - distance: current distance between endpoints in miles
        - formation_time: simulation timestamp when edge was created
        - link_quality: normalized float in [0, 1], where 1.0 = minimum distance
    """

    def __init__(self, config: NetworkConfig) -> None:
        """Initialize the TopologyUpdater.

        Args:
            config: Network configuration with radio_range_miles and
                    hysteresis_margin_pct.
        """
        self._config = config
        self._link_manager = LinkManager(
            radio_range=config.radio_range_miles,
            hysteresis_pct=config.hysteresis_margin_pct,
        )
        self._graph = nx.Graph()
        # Track link states for all known pairs: (min_id, max_id) -> LinkState
        self._link_states: dict[tuple[int, int], LinkState] = {}
        # Spatial index for efficient pair queries
        self._spatial_index: GridSpatialIndex | None = None
        # Store area dimensions (set during initialize)
        self._area_width: float = 100.0
        self._area_height: float = 100.0

    def initialize(
        self,
        positions: np.ndarray,
        area_width: float = 100.0,
        area_height: float = 100.0,
    ) -> TopologyDelta:
        """Initialize the topology from initial node positions.

        Evaluates all node pairs against the inner threshold and forms
        links for any pair whose distance is below the inner threshold.

        Args:
            positions: NumPy array of shape (N, 2) with node positions in miles.
            area_width: Simulation area width in miles.
            area_height: Simulation area height in miles.

        Returns:
            TopologyDelta with all initially formed links.
        """
        self._area_width = area_width
        self._area_height = area_height
        n_nodes = len(positions)

        # Create spatial index
        self._spatial_index = GridSpatialIndex(
            width=area_width,
            height=area_height,
            cell_size=self._config.radio_range_miles,
        )

        # Initialize graph nodes
        self._graph.clear()
        for i in range(n_nodes):
            x = float(positions[i, 0])
            y = float(positions[i, 1])
            self._graph.add_node(
                i,
                position=(x, y),
                group_id=0,
                velocity=(0.0, 0.0),
            )

        # Build spatial index and find all pairs within inner threshold
        self._spatial_index.rebuild(positions)
        inner = self._link_manager.inner_threshold

        # Use spatial index to find candidate pairs within the outer threshold
        # (to capture all potentially relevant pairs), then evaluate against
        # inner threshold for initial formation
        candidate_pairs = self._spatial_index.query_pairs(
            self._link_manager.outer_threshold
        )

        delta = TopologyDelta(timestamp=0.0)

        for node_a, node_b in candidate_pairs:
            dist = self._compute_distance(positions, node_a, node_b)
            pair_key = (min(node_a, node_b), max(node_a, node_b))

            # For initialization, only form links below inner threshold
            if dist < inner:
                self._link_states[pair_key] = LinkState.ACTIVE
                link_quality = self._compute_link_quality(dist)
                self._graph.add_edge(
                    node_a,
                    node_b,
                    distance=dist,
                    formation_time=0.0,
                    link_quality=link_quality,
                )
                delta.new_links.append((node_a, node_b, dist))
            else:
                self._link_states[pair_key] = LinkState.ABSENT

        return delta

    def update(self, positions: np.ndarray, t: float) -> TopologyDelta:
        """Update topology based on new node positions.

        Uses the spatial index to find candidate pairs, applies hysteresis
        via LinkManager, and computes the incremental delta.

        Args:
            positions: NumPy array of shape (N, 2) with current positions.
            t: Current simulation timestamp.

        Returns:
            TopologyDelta with newly formed and broken links.
        """
        if self._spatial_index is None:
            raise RuntimeError(
                "TopologyUpdater.initialize() must be called before update()"
            )

        # Rebuild spatial index with new positions
        self._spatial_index.rebuild(positions)

        # Update node attributes on the graph
        n_nodes = len(positions)
        for i in range(n_nodes):
            if self._graph.has_node(i):
                x = float(positions[i, 0])
                y = float(positions[i, 1])
                self._graph.nodes[i]["position"] = (x, y)

        delta = TopologyDelta(timestamp=t)

        # Find all candidate pairs within the outer threshold
        candidate_pairs = self._spatial_index.query_pairs(
            self._link_manager.outer_threshold
        )

        # Track which pairs we've evaluated this step
        evaluated_pairs: set[tuple[int, int]] = set()

        # Evaluate candidate pairs
        for node_a, node_b in candidate_pairs:
            pair_key = (min(node_a, node_b), max(node_a, node_b))
            evaluated_pairs.add(pair_key)

            dist = self._compute_distance(positions, node_a, node_b)
            current_state = self._link_states.get(pair_key, LinkState.ABSENT)
            new_state = self._link_manager.evaluate(current_state, dist)

            if new_state != current_state:
                self._link_states[pair_key] = new_state

                if new_state == LinkState.ACTIVE:
                    # Link formed
                    link_quality = self._compute_link_quality(dist)
                    self._graph.add_edge(
                        pair_key[0],
                        pair_key[1],
                        distance=dist,
                        formation_time=t,
                        link_quality=link_quality,
                    )
                    delta.new_links.append((pair_key[0], pair_key[1], dist))

                elif new_state == LinkState.ABSENT:
                    # Link broken
                    if self._graph.has_edge(pair_key[0], pair_key[1]):
                        self._graph.remove_edge(pair_key[0], pair_key[1])
                    delta.broken_links.append((pair_key[0], pair_key[1]))

            elif current_state == LinkState.ACTIVE:
                # Update edge distance and link quality for active links
                link_quality = self._compute_link_quality(dist)
                self._graph.edges[pair_key[0], pair_key[1]]["distance"] = dist
                self._graph.edges[pair_key[0], pair_key[1]][
                    "link_quality"
                ] = link_quality

        # Check for active links whose pairs are no longer within outer threshold
        # (nodes moved far apart — beyond spatial index query range)
        pairs_to_remove: list[tuple[int, int]] = []
        for pair_key, state in self._link_states.items():
            if state == LinkState.ACTIVE and pair_key not in evaluated_pairs:
                # Pair not found by spatial index — compute actual distance
                node_a, node_b = pair_key
                if node_a < n_nodes and node_b < n_nodes:
                    dist = self._compute_distance(positions, node_a, node_b)
                    new_state = self._link_manager.evaluate(state, dist)
                    if new_state == LinkState.ABSENT:
                        pairs_to_remove.append(pair_key)
                        if self._graph.has_edge(node_a, node_b):
                            self._graph.remove_edge(node_a, node_b)
                        delta.broken_links.append((node_a, node_b))

        for pair_key in pairs_to_remove:
            self._link_states[pair_key] = LinkState.ABSENT

        return delta

    def get_graph(self) -> nx.Graph:
        """Return the current topology graph.

        Returns:
            NetworkX undirected graph with node and edge attributes.
        """
        return self._graph

    def update_node_attributes(
        self,
        positions: np.ndarray,
        group_ids: np.ndarray | None = None,
        velocities: np.ndarray | None = None,
    ) -> None:
        """Update node attributes on the graph.

        Args:
            positions: NumPy array of shape (N, 2) with positions.
            group_ids: Optional NumPy array of shape (N,) with group IDs.
            velocities: Optional NumPy array of shape (N, 2) with velocities.
        """
        n_nodes = len(positions)
        for i in range(n_nodes):
            if self._graph.has_node(i):
                self._graph.nodes[i]["position"] = (
                    float(positions[i, 0]),
                    float(positions[i, 1]),
                )
                if group_ids is not None:
                    self._graph.nodes[i]["group_id"] = int(group_ids[i])
                if velocities is not None:
                    self._graph.nodes[i]["velocity"] = (
                        float(velocities[i, 0]),
                        float(velocities[i, 1]),
                    )

    def _compute_distance(
        self, positions: np.ndarray, node_a: int, node_b: int
    ) -> float:
        """Compute Euclidean distance between two nodes.

        Args:
            positions: Positions array of shape (N, 2).
            node_a: Index of first node.
            node_b: Index of second node.

        Returns:
            Euclidean distance in miles.
        """
        dx = float(positions[node_a, 0] - positions[node_b, 0])
        dy = float(positions[node_a, 1] - positions[node_b, 1])
        return (dx * dx + dy * dy) ** 0.5

    def _compute_link_quality(self, distance: float) -> float:
        """Compute normalized link quality from distance.

        link_quality = 1.0 - (distance / radio_range), clamped to [0, 1].

        Args:
            distance: Distance between nodes in miles.

        Returns:
            Link quality value in [0.0, 1.0].
        """
        radio_range = self._config.radio_range_miles
        if radio_range <= 0:
            return 0.0
        quality = 1.0 - (distance / radio_range)
        return max(0.0, min(1.0, quality))
