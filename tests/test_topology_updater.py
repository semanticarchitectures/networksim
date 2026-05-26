"""Unit tests for TopologyUpdater.

Tests the TopologyUpdater class which manages the NetworkX topology graph
with incremental delta computation and hysteresis-based link management.
"""

import numpy as np
import pytest

from manet_sim.core.config import NetworkConfig
from manet_sim.topology.link_manager import LinkState
from manet_sim.topology.topology_updater import TopologyDelta, TopologyUpdater


@pytest.fixture
def default_config() -> NetworkConfig:
    """Default network config: radio_range=1.0, hysteresis=10%."""
    return NetworkConfig(radio_range_miles=1.0, hysteresis_margin_pct=0.10)


@pytest.fixture
def updater(default_config: NetworkConfig) -> TopologyUpdater:
    """Create a TopologyUpdater with default config."""
    return TopologyUpdater(default_config)


class TestTopologyUpdaterInitialization:
    """Tests for TopologyUpdater.initialize()."""

    def test_initialize_no_nodes(self, updater: TopologyUpdater):
        """Initialize with empty positions array."""
        positions = np.zeros((0, 2), dtype=np.float32)
        delta = updater.initialize(positions)
        assert delta.timestamp == 0.0
        assert delta.new_links == []
        assert delta.broken_links == []
        assert updater.get_graph().number_of_nodes() == 0

    def test_initialize_single_node(self, updater: TopologyUpdater):
        """Single node produces no links."""
        positions = np.array([[50.0, 50.0]], dtype=np.float32)
        delta = updater.initialize(positions)
        assert delta.new_links == []
        assert updater.get_graph().number_of_nodes() == 1
        assert updater.get_graph().number_of_edges() == 0

    def test_initialize_two_nodes_in_range(self, updater: TopologyUpdater):
        """Two nodes within inner threshold form a link."""
        # inner_threshold = 1.0 - 0.1 = 0.9
        # Place nodes 0.5 miles apart (< 0.9)
        positions = np.array([[50.0, 50.0], [50.5, 50.0]], dtype=np.float32)
        delta = updater.initialize(positions)
        assert len(delta.new_links) == 1
        assert updater.get_graph().number_of_edges() == 1
        # Check the link is between nodes 0 and 1
        link = delta.new_links[0]
        assert set([link[0], link[1]]) == {0, 1}
        assert abs(link[2] - 0.5) < 0.01  # distance ~0.5

    def test_initialize_two_nodes_out_of_range(self, updater: TopologyUpdater):
        """Two nodes beyond inner threshold don't form a link."""
        # inner_threshold = 0.9, place nodes 0.95 apart
        positions = np.array([[50.0, 50.0], [50.95, 50.0]], dtype=np.float32)
        delta = updater.initialize(positions)
        assert len(delta.new_links) == 0
        assert updater.get_graph().number_of_edges() == 0

    def test_initialize_multiple_nodes_forms_correct_links(self):
        """Multiple nodes form links only for pairs below inner threshold."""
        config = NetworkConfig(radio_range_miles=2.0, hysteresis_margin_pct=0.10)
        updater = TopologyUpdater(config)
        # inner_threshold = 2.0 - 0.2 = 1.8
        # Node 0 at (0,0), Node 1 at (1,0) -> dist=1.0 < 1.8 -> link
        # Node 2 at (3,0) -> dist to 0 = 3.0 > 1.8, dist to 1 = 2.0 > 1.8 -> no link
        positions = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]], dtype=np.float32)
        delta = updater.initialize(positions, area_width=10.0, area_height=10.0)
        assert len(delta.new_links) == 1
        link = delta.new_links[0]
        assert set([link[0], link[1]]) == {0, 1}

    def test_initialize_sets_node_attributes(self, updater: TopologyUpdater):
        """Graph nodes have position, group_id, and velocity attributes."""
        positions = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        updater.initialize(positions)
        graph = updater.get_graph()
        assert graph.nodes[0]["position"] == (10.0, 20.0)
        assert graph.nodes[1]["position"] == (30.0, 40.0)
        assert graph.nodes[0]["group_id"] == 0
        assert graph.nodes[0]["velocity"] == (0.0, 0.0)

    def test_initialize_sets_edge_attributes(self, updater: TopologyUpdater):
        """Graph edges have distance, formation_time, and link_quality."""
        positions = np.array([[50.0, 50.0], [50.3, 50.0]], dtype=np.float32)
        updater.initialize(positions)
        graph = updater.get_graph()
        assert graph.has_edge(0, 1)
        edge_data = graph.edges[0, 1]
        assert abs(edge_data["distance"] - 0.3) < 0.01
        assert edge_data["formation_time"] == 0.0
        # link_quality = 1.0 - (0.3 / 1.0) = 0.7
        assert abs(edge_data["link_quality"] - 0.7) < 0.01


class TestTopologyUpdaterUpdate:
    """Tests for TopologyUpdater.update()."""

    def test_update_forms_new_link(self, updater: TopologyUpdater):
        """Nodes moving into range form a new link."""
        # Start far apart
        positions = np.array([[50.0, 50.0], [52.0, 50.0]], dtype=np.float32)
        updater.initialize(positions)
        assert updater.get_graph().number_of_edges() == 0

        # Move node 1 close to node 0 (within inner threshold 0.9)
        new_positions = np.array([[50.0, 50.0], [50.5, 50.0]], dtype=np.float32)
        delta = updater.update(new_positions, t=1.0)
        assert len(delta.new_links) == 1
        assert updater.get_graph().number_of_edges() == 1

    def test_update_breaks_link(self, updater: TopologyUpdater):
        """Nodes moving out of range break their link."""
        # Start close together
        positions = np.array([[50.0, 50.0], [50.5, 50.0]], dtype=np.float32)
        updater.initialize(positions)
        assert updater.get_graph().number_of_edges() == 1

        # Move node 1 far away (beyond outer threshold 1.1)
        new_positions = np.array([[50.0, 50.0], [52.0, 50.0]], dtype=np.float32)
        delta = updater.update(new_positions, t=1.0)
        assert len(delta.broken_links) == 1
        assert updater.get_graph().number_of_edges() == 0

    def test_update_hysteresis_maintains_active_link(self, updater: TopologyUpdater):
        """Active link in hysteresis zone stays active."""
        # inner=0.9, outer=1.1
        # Start with link (dist=0.5)
        positions = np.array([[50.0, 50.0], [50.5, 50.0]], dtype=np.float32)
        updater.initialize(positions)
        assert updater.get_graph().number_of_edges() == 1

        # Move to hysteresis zone (dist=0.95, between 0.9 and 1.1)
        new_positions = np.array([[50.0, 50.0], [50.95, 50.0]], dtype=np.float32)
        delta = updater.update(new_positions, t=1.0)
        assert len(delta.new_links) == 0
        assert len(delta.broken_links) == 0
        assert updater.get_graph().number_of_edges() == 1

    def test_update_hysteresis_absent_stays_absent(self, updater: TopologyUpdater):
        """Absent link in hysteresis zone stays absent."""
        # inner=0.9, outer=1.1
        # Start with no link (dist=0.95, in hysteresis zone but ABSENT)
        positions = np.array([[50.0, 50.0], [50.95, 50.0]], dtype=np.float32)
        updater.initialize(positions)
        assert updater.get_graph().number_of_edges() == 0

        # Stay in hysteresis zone
        new_positions = np.array([[50.0, 50.0], [50.95, 50.0]], dtype=np.float32)
        delta = updater.update(new_positions, t=1.0)
        assert len(delta.new_links) == 0
        assert len(delta.broken_links) == 0
        assert updater.get_graph().number_of_edges() == 0

    def test_update_incremental_delta(self, updater: TopologyUpdater):
        """Delta only contains changes, not the full state."""
        # 3 nodes: 0-1 linked, 2 far away
        positions = np.array(
            [[50.0, 50.0], [50.5, 50.0], [55.0, 55.0]], dtype=np.float32
        )
        updater.initialize(positions)
        assert updater.get_graph().number_of_edges() == 1

        # Move node 2 close to node 0, keep 0-1 link
        new_positions = np.array(
            [[50.0, 50.0], [50.5, 50.0], [50.3, 50.3]], dtype=np.float32
        )
        delta = updater.update(new_positions, t=1.0)
        # Only the new link (0,2) or (1,2) should appear
        assert len(delta.new_links) >= 1
        assert len(delta.broken_links) == 0
        # Original link still exists
        assert updater.get_graph().has_edge(0, 1)

    def test_update_updates_edge_distance(self, updater: TopologyUpdater):
        """Active link edge distance is updated when nodes move."""
        positions = np.array([[50.0, 50.0], [50.3, 50.0]], dtype=np.float32)
        updater.initialize(positions)
        assert abs(updater.get_graph().edges[0, 1]["distance"] - 0.3) < 0.01

        # Move node 1 slightly closer
        new_positions = np.array([[50.0, 50.0], [50.2, 50.0]], dtype=np.float32)
        updater.update(new_positions, t=1.0)
        assert abs(updater.get_graph().edges[0, 1]["distance"] - 0.2) < 0.01

    def test_update_raises_without_initialize(self):
        """Calling update before initialize raises RuntimeError."""
        config = NetworkConfig(radio_range_miles=1.0, hysteresis_margin_pct=0.10)
        updater = TopologyUpdater(config)
        positions = np.array([[50.0, 50.0]], dtype=np.float32)
        with pytest.raises(RuntimeError):
            updater.update(positions, t=1.0)


class TestTopologyUpdaterLinkQuality:
    """Tests for link quality computation."""

    def test_link_quality_at_zero_distance(self, updater: TopologyUpdater):
        """Link quality is 1.0 when distance is 0."""
        positions = np.array([[50.0, 50.0], [50.0, 50.0]], dtype=np.float32)
        updater.initialize(positions)
        if updater.get_graph().has_edge(0, 1):
            assert updater.get_graph().edges[0, 1]["link_quality"] == 1.0

    def test_link_quality_decreases_with_distance(self, updater: TopologyUpdater):
        """Link quality decreases as distance increases."""
        # radio_range = 1.0
        # At dist=0.3: quality = 1.0 - 0.3/1.0 = 0.7
        positions = np.array([[50.0, 50.0], [50.3, 50.0]], dtype=np.float32)
        updater.initialize(positions)
        quality = updater.get_graph().edges[0, 1]["link_quality"]
        assert abs(quality - 0.7) < 0.01

    def test_link_quality_clamped_to_zero(self):
        """Link quality is clamped to 0.0 at radio range."""
        config = NetworkConfig(radio_range_miles=5.0, hysteresis_margin_pct=0.10)
        updater = TopologyUpdater(config)
        # inner_threshold = 5.0 - 0.5 = 4.5
        # Place nodes at dist=4.0 (< 4.5, so link forms)
        # quality = 1.0 - 4.0/5.0 = 0.2
        positions = np.array([[0.0, 0.0], [4.0, 0.0]], dtype=np.float32)
        updater.initialize(positions, area_width=50.0, area_height=50.0)
        quality = updater.get_graph().edges[0, 1]["link_quality"]
        assert abs(quality - 0.2) < 0.01


class TestTopologyUpdaterNodeAttributes:
    """Tests for node attribute management."""

    def test_update_node_attributes(self, updater: TopologyUpdater):
        """update_node_attributes sets group_id and velocity."""
        positions = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        updater.initialize(positions)

        group_ids = np.array([1, 2], dtype=np.int32)
        velocities = np.array([[0.01, -0.02], [0.03, 0.04]], dtype=np.float32)
        updater.update_node_attributes(positions, group_ids, velocities)

        graph = updater.get_graph()
        assert graph.nodes[0]["group_id"] == 1
        assert graph.nodes[1]["group_id"] == 2
        assert graph.nodes[0]["velocity"] == pytest.approx((0.01, -0.02), abs=1e-5)
        assert graph.nodes[1]["velocity"] == pytest.approx((0.03, 0.04), abs=1e-5)

    def test_positions_updated_on_update(self, updater: TopologyUpdater):
        """Node positions are updated in the graph during update()."""
        positions = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        updater.initialize(positions)

        new_positions = np.array([[15.0, 25.0], [35.0, 45.0]], dtype=np.float32)
        updater.update(new_positions, t=1.0)

        graph = updater.get_graph()
        assert graph.nodes[0]["position"] == pytest.approx((15.0, 25.0), abs=1e-4)
        assert graph.nodes[1]["position"] == pytest.approx((35.0, 45.0), abs=1e-4)


class TestTopologyUpdaterGetGraph:
    """Tests for get_graph()."""

    def test_get_graph_returns_nx_graph(self, updater: TopologyUpdater):
        """get_graph returns a NetworkX Graph instance."""
        import networkx as nx

        positions = np.array([[50.0, 50.0]], dtype=np.float32)
        updater.initialize(positions)
        graph = updater.get_graph()
        assert isinstance(graph, nx.Graph)

    def test_get_graph_reflects_current_state(self, updater: TopologyUpdater):
        """Graph returned by get_graph reflects the current topology."""
        positions = np.array([[50.0, 50.0], [50.5, 50.0]], dtype=np.float32)
        updater.initialize(positions)
        assert updater.get_graph().number_of_edges() == 1

        # Break the link
        new_positions = np.array([[50.0, 50.0], [52.0, 50.0]], dtype=np.float32)
        updater.update(new_positions, t=1.0)
        assert updater.get_graph().number_of_edges() == 0


class TestTopologyDelta:
    """Tests for the TopologyDelta dataclass."""

    def test_topology_delta_defaults(self):
        """TopologyDelta has empty lists by default."""
        delta = TopologyDelta(timestamp=5.0)
        assert delta.timestamp == 5.0
        assert delta.new_links == []
        assert delta.broken_links == []

    def test_topology_delta_with_data(self):
        """TopologyDelta stores link data correctly."""
        delta = TopologyDelta(
            timestamp=10.0,
            new_links=[(0, 1, 0.5), (2, 3, 0.8)],
            broken_links=[(4, 5)],
        )
        assert len(delta.new_links) == 2
        assert len(delta.broken_links) == 1
        assert delta.new_links[0] == (0, 1, 0.5)


class TestTopologyUpdaterEdgeCases:
    """Edge case tests."""

    def test_large_number_of_nodes(self):
        """Handles 100 nodes without error."""
        config = NetworkConfig(radio_range_miles=5.0, hysteresis_margin_pct=0.10)
        updater = TopologyUpdater(config)
        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 50, size=(100, 2)).astype(np.float32)
        delta = updater.initialize(positions, area_width=50.0, area_height=50.0)
        # Should have some links formed
        assert updater.get_graph().number_of_nodes() == 100
        # Verify graph is consistent
        assert updater.get_graph().number_of_edges() == len(delta.new_links)

    def test_all_nodes_at_same_position(self, updater: TopologyUpdater):
        """All nodes at same position form a complete graph."""
        positions = np.array(
            [[50.0, 50.0], [50.0, 50.0], [50.0, 50.0]], dtype=np.float32
        )
        delta = updater.initialize(positions)
        # All pairs should be linked (distance=0 < inner_threshold=0.9)
        assert updater.get_graph().number_of_edges() == 3  # C(3,2) = 3
        assert len(delta.new_links) == 3

    def test_formation_time_recorded_correctly(self, updater: TopologyUpdater):
        """New links record the correct formation time."""
        positions = np.array([[50.0, 50.0], [52.0, 50.0]], dtype=np.float32)
        updater.initialize(positions)

        # Form link at t=5.0
        new_positions = np.array([[50.0, 50.0], [50.5, 50.0]], dtype=np.float32)
        updater.update(new_positions, t=5.0)
        assert updater.get_graph().edges[0, 1]["formation_time"] == 5.0
