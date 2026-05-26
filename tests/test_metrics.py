"""Unit tests for MobilityMetricsCollector and TopologyMetricsCollector."""

import csv
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

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
from manet_sim.network.metrics import (
    MobilityMetricsCollector,
    MobilitySnapshot,
    TopologyMetricsCollector,
    TopologySnapshot,
)


# --- MobilityMetricsCollector Tests ---


class TestMobilityMetricsCollector:
    """Tests for MobilityMetricsCollector."""

    def test_subscribes_to_event_bus(self):
        """Collector subscribes to position_update, step_complete, simulation_end."""
        bus = EventBus()
        collector = MobilityMetricsCollector(
            snapshot_interval=60.0, output_dir="/tmp/test", event_bus=bus
        )
        # Verify subscriptions exist by publishing events without error
        bus.publish(POSITION_UPDATE, PositionUpdateEvent(
            timestamp=0.0,
            positions=np.zeros((5, 2), dtype=np.float32),
            velocities=np.zeros((5, 2), dtype=np.float32),
        ))
        bus.publish(STEP_COMPLETE, StepCompleteEvent(
            timestamp=1.0, step_number=1, active_links=0, wall_clock_ms=1.0
        ))

    def test_snapshot_at_interval(self):
        """Snapshot is computed when step timestamp reaches interval."""
        bus = EventBus()
        collector = MobilityMetricsCollector(
            snapshot_interval=60.0, output_dir="/tmp/test", event_bus=bus
        )

        positions = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        velocities = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)

        # Publish position update
        bus.publish(POSITION_UPDATE, PositionUpdateEvent(
            timestamp=0.0, positions=positions, velocities=velocities
        ))

        # Step at t=59 should not trigger snapshot
        bus.publish(STEP_COMPLETE, StepCompleteEvent(
            timestamp=59.0, step_number=59, active_links=0, wall_clock_ms=1.0
        ))
        assert len(collector.snapshots) == 0

        # Step at t=60 should trigger snapshot
        bus.publish(STEP_COMPLETE, StepCompleteEvent(
            timestamp=60.0, step_number=60, active_links=0, wall_clock_ms=1.0
        ))
        assert len(collector.snapshots) == 1
        assert collector.snapshots[0].timestamp == 60.0

    def test_speed_metrics(self):
        """Avg, std, and max speed are computed correctly from velocities."""
        bus = EventBus()
        collector = MobilityMetricsCollector(
            snapshot_interval=10.0, output_dir="/tmp/test", event_bus=bus
        )

        # Node 0: speed = sqrt(3^2 + 4^2) = 5.0
        # Node 1: speed = sqrt(0^2 + 0^2) = 0.0
        velocities = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
        positions = np.array([[0.0, 0.0], [50.0, 50.0]], dtype=np.float32)

        bus.publish(POSITION_UPDATE, PositionUpdateEvent(
            timestamp=0.0, positions=positions, velocities=velocities
        ))
        bus.publish(STEP_COMPLETE, StepCompleteEvent(
            timestamp=10.0, step_number=10, active_links=0, wall_clock_ms=1.0
        ))

        snap = collector.snapshots[0]
        assert snap.avg_speed == pytest.approx(2.5, abs=1e-4)
        assert snap.std_speed == pytest.approx(2.5, abs=1e-4)
        assert snap.max_speed == pytest.approx(5.0, abs=1e-4)

    def test_displacement_from_initial(self):
        """Average displacement is computed from initial positions."""
        bus = EventBus()
        collector = MobilityMetricsCollector(
            snapshot_interval=10.0, output_dir="/tmp/test", event_bus=bus
        )

        # Initial positions
        initial_pos = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32)
        bus.publish(POSITION_UPDATE, PositionUpdateEvent(
            timestamp=0.0,
            positions=initial_pos,
            velocities=np.zeros((2, 2), dtype=np.float32),
        ))

        # Move nodes
        new_pos = np.array([[3.0, 4.0], [10.0, 10.0]], dtype=np.float32)
        bus.publish(POSITION_UPDATE, PositionUpdateEvent(
            timestamp=5.0,
            positions=new_pos,
            velocities=np.zeros((2, 2), dtype=np.float32),
        ))

        bus.publish(STEP_COMPLETE, StepCompleteEvent(
            timestamp=10.0, step_number=10, active_links=0, wall_clock_ms=1.0
        ))

        snap = collector.snapshots[0]
        # Node 0 displacement: sqrt(3^2 + 4^2) = 5.0
        # Node 1 displacement: 0.0
        # Average: 2.5
        assert snap.avg_displacement == pytest.approx(2.5, abs=1e-4)

    def test_group_cohesion_with_groups(self):
        """Group cohesion is mean distance from members to centroid."""
        bus = EventBus()
        collector = MobilityMetricsCollector(
            snapshot_interval=10.0, output_dir="/tmp/test", event_bus=bus
        )

        # Two groups: group 0 has nodes 0,1; group 1 has nodes 2,3
        group_ids = np.array([0, 0, 1, 1], dtype=np.int32)
        collector.set_group_ids(group_ids)

        # Group 0: nodes at (0,0) and (2,0) -> centroid (1,0), distances 1.0 each
        # Group 1: nodes at (10,10) and (10,12) -> centroid (10,11), distances 1.0 each
        positions = np.array(
            [[0.0, 0.0], [2.0, 0.0], [10.0, 10.0], [10.0, 12.0]],
            dtype=np.float32,
        )
        velocities = np.zeros((4, 2), dtype=np.float32)

        bus.publish(POSITION_UPDATE, PositionUpdateEvent(
            timestamp=0.0, positions=positions, velocities=velocities
        ))
        bus.publish(STEP_COMPLETE, StepCompleteEvent(
            timestamp=10.0, step_number=10, active_links=0, wall_clock_ms=1.0
        ))

        snap = collector.snapshots[0]
        # Both groups have cohesion 1.0, average = 1.0
        assert snap.avg_group_cohesion == pytest.approx(1.0, abs=1e-4)

    def test_group_cohesion_single_member_is_zero(self):
        """Groups with fewer than 2 members report 0.0 cohesion."""
        bus = EventBus()
        collector = MobilityMetricsCollector(
            snapshot_interval=10.0, output_dir="/tmp/test", event_bus=bus
        )

        # One group with 1 member, one group with 2 members
        group_ids = np.array([0, 1, 1], dtype=np.int32)
        collector.set_group_ids(group_ids)

        # Group 0: single node at (5,5) -> cohesion 0.0
        # Group 1: nodes at (0,0) and (4,0) -> centroid (2,0), distances 2.0 each
        positions = np.array(
            [[5.0, 5.0], [0.0, 0.0], [4.0, 0.0]], dtype=np.float32
        )
        velocities = np.zeros((3, 2), dtype=np.float32)

        bus.publish(POSITION_UPDATE, PositionUpdateEvent(
            timestamp=0.0, positions=positions, velocities=velocities
        ))
        bus.publish(STEP_COMPLETE, StepCompleteEvent(
            timestamp=10.0, step_number=10, active_links=0, wall_clock_ms=1.0
        ))

        snap = collector.snapshots[0]
        # Group 0 cohesion: 0.0, Group 1 cohesion: 2.0
        # Average: (0.0 + 2.0) / 2 = 1.0
        assert snap.avg_group_cohesion == pytest.approx(1.0, abs=1e-4)

    def test_export_to_csv(self):
        """Export produces CSV with header row and 4 decimal places."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bus = EventBus()
            collector = MobilityMetricsCollector(
                snapshot_interval=10.0, output_dir=tmpdir, event_bus=bus
            )

            positions = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            velocities = np.array([[0.5, 0.5], [1.0, 1.0]], dtype=np.float32)

            bus.publish(POSITION_UPDATE, PositionUpdateEvent(
                timestamp=0.0, positions=positions, velocities=velocities
            ))
            bus.publish(STEP_COMPLETE, StepCompleteEvent(
                timestamp=10.0, step_number=10, active_links=0, wall_clock_ms=1.0
            ))

            out_path = collector.export_to_csv()
            assert out_path.exists()

            with open(out_path) as f:
                reader = csv.reader(f)
                header = next(reader)
                assert header == [
                    "timestamp", "avg_speed", "std_speed",
                    "max_speed", "avg_displacement", "avg_group_cohesion",
                ]
                row = next(reader)
                # Check 4 decimal places format
                for val in row:
                    parts = val.split(".")
                    assert len(parts) == 2
                    assert len(parts[1]) == 4

    def test_multiple_snapshots(self):
        """Multiple snapshots are collected at each interval."""
        bus = EventBus()
        collector = MobilityMetricsCollector(
            snapshot_interval=10.0, output_dir="/tmp/test", event_bus=bus
        )

        positions = np.array([[0.0, 0.0]], dtype=np.float32)
        velocities = np.array([[1.0, 0.0]], dtype=np.float32)

        bus.publish(POSITION_UPDATE, PositionUpdateEvent(
            timestamp=0.0, positions=positions, velocities=velocities
        ))

        # Trigger snapshots at t=10, t=20, t=30
        for t in [10.0, 20.0, 30.0]:
            bus.publish(STEP_COMPLETE, StepCompleteEvent(
                timestamp=t, step_number=int(t), active_links=0, wall_clock_ms=1.0
            ))

        assert len(collector.snapshots) == 3
        assert collector.snapshots[0].timestamp == 10.0
        assert collector.snapshots[1].timestamp == 20.0
        assert collector.snapshots[2].timestamp == 30.0

    def test_no_data_produces_zero_metrics(self):
        """When no position data is available, all metrics are 0."""
        bus = EventBus()
        collector = MobilityMetricsCollector(
            snapshot_interval=10.0, output_dir="/tmp/test", event_bus=bus
        )

        # Trigger snapshot without any position updates
        bus.publish(STEP_COMPLETE, StepCompleteEvent(
            timestamp=10.0, step_number=10, active_links=0, wall_clock_ms=1.0
        ))

        snap = collector.snapshots[0]
        assert snap.avg_speed == 0.0
        assert snap.std_speed == 0.0
        assert snap.max_speed == 0.0
        assert snap.avg_displacement == 0.0
        assert snap.avg_group_cohesion == 0.0


# --- TopologyMetricsCollector Tests ---


class TestTopologyMetricsCollector:
    """Tests for TopologyMetricsCollector."""

    def test_subscribes_to_event_bus(self):
        """Collector subscribes to link_formed, link_broken, step_complete, simulation_end."""
        bus = EventBus()
        collector = TopologyMetricsCollector(
            snapshot_interval=300.0, output_dir="/tmp/test", event_bus=bus
        )
        # Verify subscriptions exist by publishing events without error
        bus.publish(LINK_FORMED, LinkFormedEvent(
            timestamp=0.0, node_a=0, node_b=1, distance=0.5
        ))
        bus.publish(LINK_BROKEN, LinkBrokenEvent(
            timestamp=1.0, node_a=0, node_b=1
        ))

    def test_link_formed_adds_edge(self):
        """link_formed event adds an edge to the internal graph."""
        bus = EventBus()
        collector = TopologyMetricsCollector(
            snapshot_interval=300.0, output_dir="/tmp/test", event_bus=bus
        )

        bus.publish(LINK_FORMED, LinkFormedEvent(
            timestamp=0.0, node_a=0, node_b=1, distance=0.5
        ))

        assert collector.graph.has_edge(0, 1)
        assert collector.graph.number_of_edges() == 1

    def test_link_broken_removes_edge(self):
        """link_broken event removes an edge from the internal graph."""
        bus = EventBus()
        collector = TopologyMetricsCollector(
            snapshot_interval=300.0, output_dir="/tmp/test", event_bus=bus
        )

        bus.publish(LINK_FORMED, LinkFormedEvent(
            timestamp=0.0, node_a=0, node_b=1, distance=0.5
        ))
        bus.publish(LINK_BROKEN, LinkBrokenEvent(
            timestamp=1.0, node_a=0, node_b=1
        ))

        assert not collector.graph.has_edge(0, 1)
        assert collector.graph.number_of_edges() == 0

    def test_snapshot_at_interval(self):
        """Snapshot is computed when step timestamp reaches interval."""
        bus = EventBus()
        collector = TopologyMetricsCollector(
            snapshot_interval=300.0, output_dir="/tmp/test", event_bus=bus
        )
        collector.set_node_count(5)

        # Add some edges
        bus.publish(LINK_FORMED, LinkFormedEvent(
            timestamp=0.0, node_a=0, node_b=1, distance=0.5
        ))
        bus.publish(LINK_FORMED, LinkFormedEvent(
            timestamp=0.0, node_a=1, node_b=2, distance=0.7
        ))

        # Step at t=299 should not trigger snapshot
        bus.publish(STEP_COMPLETE, StepCompleteEvent(
            timestamp=299.0, step_number=299, active_links=2, wall_clock_ms=1.0
        ))
        assert len(collector.snapshots) == 0

        # Step at t=300 should trigger snapshot
        bus.publish(STEP_COMPLETE, StepCompleteEvent(
            timestamp=300.0, step_number=300, active_links=2, wall_clock_ms=1.0
        ))
        assert len(collector.snapshots) == 1

    def test_topology_metrics_correctness(self):
        """Computed metrics match expected values for a known graph."""
        bus = EventBus()
        collector = TopologyMetricsCollector(
            snapshot_interval=10.0, output_dir="/tmp/test", event_bus=bus
        )
        collector.set_node_count(4)

        # Create a triangle (0-1-2) plus isolated node 3
        bus.publish(LINK_FORMED, LinkFormedEvent(
            timestamp=0.0, node_a=0, node_b=1, distance=0.5
        ))
        bus.publish(LINK_FORMED, LinkFormedEvent(
            timestamp=0.0, node_a=1, node_b=2, distance=0.6
        ))
        bus.publish(LINK_FORMED, LinkFormedEvent(
            timestamp=0.0, node_a=0, node_b=2, distance=0.7
        ))

        bus.publish(STEP_COMPLETE, StepCompleteEvent(
            timestamp=10.0, step_number=10, active_links=3, wall_clock_ms=1.0
        ))

        snap = collector.snapshots[0]
        assert snap.node_count == 4
        assert snap.edge_count == 3
        assert snap.avg_degree == pytest.approx(1.5, abs=1e-4)  # (2+2+2+0)/4
        assert snap.max_degree == 2
        assert snap.num_connected_components == 2  # triangle + isolated node
        assert snap.largest_component_size == 3
        # Triangle has clustering 1.0 for each node in it, isolated node has 0
        # Average clustering = (1.0 + 1.0 + 1.0 + 0.0) / 4 = 0.75
        assert snap.avg_clustering_coefficient == pytest.approx(0.75, abs=1e-4)

    def test_zero_node_graph(self):
        """Zero-node graph produces all-zero metrics."""
        bus = EventBus()
        collector = TopologyMetricsCollector(
            snapshot_interval=10.0, output_dir="/tmp/test", event_bus=bus
        )
        # Don't set any nodes

        bus.publish(STEP_COMPLETE, StepCompleteEvent(
            timestamp=10.0, step_number=10, active_links=0, wall_clock_ms=1.0
        ))

        snap = collector.snapshots[0]
        assert snap.node_count == 0
        assert snap.edge_count == 0
        assert snap.avg_degree == 0.0
        assert snap.max_degree == 0
        assert snap.num_connected_components == 0
        assert snap.largest_component_size == 0
        assert snap.avg_clustering_coefficient == 0.0

    def test_export_to_json(self):
        """Export produces valid JSON with timestamp and all metric fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bus = EventBus()
            collector = TopologyMetricsCollector(
                snapshot_interval=10.0, output_dir=tmpdir, event_bus=bus
            )
            collector.set_node_count(3)

            bus.publish(LINK_FORMED, LinkFormedEvent(
                timestamp=0.0, node_a=0, node_b=1, distance=0.5
            ))
            bus.publish(STEP_COMPLETE, StepCompleteEvent(
                timestamp=10.0, step_number=10, active_links=1, wall_clock_ms=1.0
            ))

            out_path = collector.export_to_json()
            assert out_path.exists()

            with open(out_path) as f:
                data = json.load(f)

            assert isinstance(data, list)
            assert len(data) == 1
            entry = data[0]
            assert "timestamp" in entry
            assert "node_count" in entry
            assert "edge_count" in entry
            assert "avg_degree" in entry
            assert "max_degree" in entry
            assert "num_connected_components" in entry
            assert "largest_component_size" in entry
            assert "avg_clustering_coefficient" in entry
            assert entry["timestamp"] == 10.0

    def test_multiple_snapshots(self):
        """Multiple snapshots are collected at each interval."""
        bus = EventBus()
        collector = TopologyMetricsCollector(
            snapshot_interval=100.0, output_dir="/tmp/test", event_bus=bus
        )
        collector.set_node_count(2)

        bus.publish(LINK_FORMED, LinkFormedEvent(
            timestamp=0.0, node_a=0, node_b=1, distance=0.5
        ))

        for t in [100.0, 200.0, 300.0]:
            bus.publish(STEP_COMPLETE, StepCompleteEvent(
                timestamp=t, step_number=int(t), active_links=1, wall_clock_ms=1.0
            ))

        assert len(collector.snapshots) == 3
        assert collector.snapshots[0].timestamp == 100.0
        assert collector.snapshots[1].timestamp == 200.0
        assert collector.snapshots[2].timestamp == 300.0

    def test_link_broken_nonexistent_edge_no_error(self):
        """Breaking a non-existent link does not raise an error."""
        bus = EventBus()
        collector = TopologyMetricsCollector(
            snapshot_interval=300.0, output_dir="/tmp/test", event_bus=bus
        )

        # This should not raise
        bus.publish(LINK_BROKEN, LinkBrokenEvent(
            timestamp=1.0, node_a=99, node_b=100
        ))

    def test_simulation_end_triggers_export(self):
        """simulation_end event triggers JSON export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bus = EventBus()
            collector = TopologyMetricsCollector(
                snapshot_interval=10.0, output_dir=tmpdir, event_bus=bus
            )
            collector.set_node_count(2)

            bus.publish(LINK_FORMED, LinkFormedEvent(
                timestamp=0.0, node_a=0, node_b=1, distance=0.5
            ))
            bus.publish(STEP_COMPLETE, StepCompleteEvent(
                timestamp=10.0, step_number=10, active_links=1, wall_clock_ms=1.0
            ))

            # Trigger simulation end
            bus.publish(SIMULATION_END, SimulationEndEvent(
                total_time=100.0, total_steps=100, wall_clock_seconds=5.0
            ))

            out_path = Path(tmpdir) / "topology_metrics.json"
            assert out_path.exists()

    def test_mobility_simulation_end_triggers_export(self):
        """simulation_end event triggers CSV export for mobility collector."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bus = EventBus()
            collector = MobilityMetricsCollector(
                snapshot_interval=10.0, output_dir=tmpdir, event_bus=bus
            )

            positions = np.array([[1.0, 2.0]], dtype=np.float32)
            velocities = np.array([[0.5, 0.5]], dtype=np.float32)

            bus.publish(POSITION_UPDATE, PositionUpdateEvent(
                timestamp=0.0, positions=positions, velocities=velocities
            ))
            bus.publish(STEP_COMPLETE, StepCompleteEvent(
                timestamp=10.0, step_number=10, active_links=0, wall_clock_ms=1.0
            ))

            # Trigger simulation end
            bus.publish(SIMULATION_END, SimulationEndEvent(
                total_time=100.0, total_steps=100, wall_clock_seconds=5.0
            ))

            out_path = Path(tmpdir) / "mobility_metrics.csv"
            assert out_path.exists()
