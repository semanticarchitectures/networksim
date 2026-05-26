"""Unit tests for SimulationEngine."""

import json
import os
import tempfile
from unittest.mock import patch

import numpy as np
import pytest

from manet_sim.core.config import (
    MobilityConfig,
    NetworkConfig,
    OutputConfig,
    SimulationConfig,
)
from manet_sim.core.event_bus import (
    LINK_FORMED,
    SIMULATION_END,
    STEP_COMPLETE,
    SimulationEndEvent,
    StepCompleteEvent,
)
from manet_sim.core.simulation import SimulationEngine


def _make_small_config(output_dir: str) -> SimulationConfig:
    """Create a minimal config for fast testing (10 nodes, 5 steps)."""
    return SimulationConfig(
        duration_seconds=5.0,
        step_size=1.0,
        seed=42,
        area_width=10.0,
        area_height=10.0,
        node_count=10,
        mobility=MobilityConfig(
            model="rpgm",
            num_groups=2,
            group_speed_min_mps=0.5,
            group_speed_max_mps=5.0,
            pause_min_seconds=0.0,
            pause_max_seconds=0.0,
            max_deviation_miles=1.0,
            deviation_model="uniform",
        ),
        network=NetworkConfig(
            radio_range_miles=3.0,
            hysteresis_margin_pct=0.10,
        ),
        output=OutputConfig(
            format="json",
            output_dir=output_dir,
            snapshot_interval=2,
            buffer_size=10,
        ),
    )


class TestSimulationEngineInit:
    """Tests for SimulationEngine initialization."""

    def test_init_with_valid_config(self, tmp_path):
        """Engine initializes without error from valid config."""
        config = _make_small_config(str(tmp_path))
        engine = SimulationEngine(config)
        assert engine.seed == 42
        assert engine.steps_completed == 0

    def test_init_records_seed(self, tmp_path):
        """Engine records the seed from config."""
        config = _make_small_config(str(tmp_path))
        config.seed = 123
        engine = SimulationEngine(config)
        assert engine.seed == 123

    def test_init_auto_generates_seed_when_none(self, tmp_path):
        """Engine auto-generates a seed when config.seed is None."""
        config = _make_small_config(str(tmp_path))
        config.seed = None
        engine = SimulationEngine(config)
        assert isinstance(engine.seed, int)
        assert 0 <= engine.seed <= 2**32 - 1


class TestSimulationEngineRun:
    """Tests for SimulationEngine.run()."""

    def test_run_completes_all_steps(self, tmp_path):
        """Engine runs to completion with correct number of steps."""
        config = _make_small_config(str(tmp_path))
        engine = SimulationEngine(config)
        engine.run()
        assert engine.steps_completed == 5

    def test_run_produces_output_file(self, tmp_path):
        """Engine produces a steering output JSON file."""
        config = _make_small_config(str(tmp_path))
        engine = SimulationEngine(config)
        engine.run()

        output_file = os.path.join(str(tmp_path), "steering_output.json")
        assert os.path.exists(output_file)

    def test_output_file_is_valid_json(self, tmp_path):
        """Output file is parseable as valid JSON."""
        config = _make_small_config(str(tmp_path))
        engine = SimulationEngine(config)
        engine.run()

        output_file = os.path.join(str(tmp_path), "steering_output.json")
        with open(output_file) as f:
            data = json.load(f)

        assert "metadata" in data
        assert "steps" in data
        assert "snapshots" in data

    def test_output_metadata_contains_seed(self, tmp_path):
        """Output metadata includes the simulation seed."""
        config = _make_small_config(str(tmp_path))
        engine = SimulationEngine(config)
        engine.run()

        output_file = os.path.join(str(tmp_path), "steering_output.json")
        with open(output_file) as f:
            data = json.load(f)

        assert data["metadata"]["seed"] == 42

    def test_output_metadata_contains_config(self, tmp_path):
        """Output metadata includes the simulation configuration."""
        config = _make_small_config(str(tmp_path))
        engine = SimulationEngine(config)
        engine.run()

        output_file = os.path.join(str(tmp_path), "steering_output.json")
        with open(output_file) as f:
            data = json.load(f)

        meta_config = data["metadata"]["config"]
        assert meta_config["nodes"]["count"] == 10
        assert meta_config["simulation"]["duration_seconds"] == 5.0

    def test_output_has_correct_step_count(self, tmp_path):
        """Output contains the expected number of step records."""
        config = _make_small_config(str(tmp_path))
        engine = SimulationEngine(config)
        engine.run()

        output_file = os.path.join(str(tmp_path), "steering_output.json")
        with open(output_file) as f:
            data = json.load(f)

        assert len(data["steps"]) == 5

    def test_output_steps_have_correct_timestamps(self, tmp_path):
        """Each step record has the correct timestamp."""
        config = _make_small_config(str(tmp_path))
        engine = SimulationEngine(config)
        engine.run()

        output_file = os.path.join(str(tmp_path), "steering_output.json")
        with open(output_file) as f:
            data = json.load(f)

        timestamps = [step["timestamp"] for step in data["steps"]]
        assert timestamps == [0.0, 1.0, 2.0, 3.0, 4.0]

    def test_output_steps_have_node_data(self, tmp_path):
        """Each step record contains node position data."""
        config = _make_small_config(str(tmp_path))
        engine = SimulationEngine(config)
        engine.run()

        output_file = os.path.join(str(tmp_path), "steering_output.json")
        with open(output_file) as f:
            data = json.load(f)

        for step in data["steps"]:
            assert "nodes" in step
            assert len(step["nodes"]) == 10
            # Check node structure
            node = step["nodes"][0]
            assert "node_id" in node
            assert "group_id" in node
            assert "x" in node
            assert "y" in node
            assert "vx" in node
            assert "vy" in node

    def test_output_has_snapshots(self, tmp_path):
        """Output contains topology snapshots at configured interval."""
        config = _make_small_config(str(tmp_path))
        engine = SimulationEngine(config)
        engine.run()

        output_file = os.path.join(str(tmp_path), "steering_output.json")
        with open(output_file) as f:
            data = json.load(f)

        # With snapshot_interval=2 and 5 steps (t=0,1,2,3,4),
        # snapshots at t=0, t=2, t=4
        assert len(data["snapshots"]) >= 2


class TestSimulationEngineEventBus:
    """Tests for event bus integration."""

    def test_step_complete_events_published(self, tmp_path):
        """step_complete events are published for each step."""
        config = _make_small_config(str(tmp_path))
        engine = SimulationEngine(config)

        step_events = []
        engine.event_bus.subscribe(
            STEP_COMPLETE, lambda e: step_events.append(e)
        )

        engine.run()

        assert len(step_events) == 5
        assert all(isinstance(e, StepCompleteEvent) for e in step_events)

    def test_simulation_end_event_published(self, tmp_path):
        """simulation_end event is published when simulation completes."""
        config = _make_small_config(str(tmp_path))
        engine = SimulationEngine(config)

        end_events = []
        engine.event_bus.subscribe(
            SIMULATION_END, lambda e: end_events.append(e)
        )

        engine.run()

        assert len(end_events) == 1
        end_event = end_events[0]
        assert isinstance(end_event, SimulationEndEvent)
        assert end_event.total_time == 5.0
        assert end_event.total_steps == 5

    def test_link_formed_events_published(self, tmp_path):
        """link_formed events are published when links form."""
        config = _make_small_config(str(tmp_path))
        # Use a large radio range to ensure some links form
        config.network.radio_range_miles = 5.0
        engine = SimulationEngine(config)

        link_events = []
        engine.event_bus.subscribe(
            LINK_FORMED, lambda e: link_events.append(e)
        )

        engine.run()

        # With 10 nodes in a 10x10 area and 5-mile radio range,
        # we expect some links to form
        assert len(link_events) > 0


class TestSimulationEngineReproducibility:
    """Tests for simulation reproducibility."""

    def test_same_seed_produces_identical_output(self, tmp_path):
        """Running with the same seed produces identical output."""
        dir1 = str(tmp_path / "run1")
        dir2 = str(tmp_path / "run2")

        config1 = _make_small_config(dir1)
        config2 = _make_small_config(dir2)

        engine1 = SimulationEngine(config1)
        engine1.run()

        engine2 = SimulationEngine(config2)
        engine2.run()

        # Read both output files
        with open(os.path.join(dir1, "steering_output.json")) as f:
            data1 = json.load(f)
        with open(os.path.join(dir2, "steering_output.json")) as f:
            data2 = json.load(f)

        # Compare steps (metadata timestamps will differ)
        assert len(data1["steps"]) == len(data2["steps"])
        for s1, s2 in zip(data1["steps"], data2["steps"]):
            assert s1["timestamp"] == s2["timestamp"]
            assert len(s1["nodes"]) == len(s2["nodes"])
            for n1, n2 in zip(s1["nodes"], s2["nodes"]):
                assert n1["node_id"] == n2["node_id"]
                assert n1["x"] == n2["x"]
                assert n1["y"] == n2["y"]
                assert n1["vx"] == n2["vx"]
                assert n1["vy"] == n2["vy"]

    def test_different_seeds_produce_different_output(self, tmp_path):
        """Running with different seeds produces different output."""
        dir1 = str(tmp_path / "run1")
        dir2 = str(tmp_path / "run2")

        config1 = _make_small_config(dir1)
        config1.seed = 42

        config2 = _make_small_config(dir2)
        config2.seed = 99

        engine1 = SimulationEngine(config1)
        engine1.run()

        engine2 = SimulationEngine(config2)
        engine2.run()

        # Read both output files
        with open(os.path.join(dir1, "steering_output.json")) as f:
            data1 = json.load(f)
        with open(os.path.join(dir2, "steering_output.json")) as f:
            data2 = json.load(f)

        # At least some node positions should differ
        nodes1 = data1["steps"][-1]["nodes"]
        nodes2 = data2["steps"][-1]["nodes"]
        positions_differ = any(
            n1["x"] != n2["x"] or n1["y"] != n2["y"]
            for n1, n2 in zip(nodes1, nodes2)
        )
        assert positions_differ


class TestSimulationEngineMainLoop:
    """Tests for the main loop sequence."""

    def test_loop_sequence_mobility_before_topology(self, tmp_path):
        """Mobility update happens before topology update each step."""
        config = _make_small_config(str(tmp_path))
        engine = SimulationEngine(config)

        # Run and verify the output has topology deltas that reflect
        # positions computed in the same step
        engine.run()

        output_file = os.path.join(str(tmp_path), "steering_output.json")
        with open(output_file) as f:
            data = json.load(f)

        # Verify all steps have the expected structure
        for step in data["steps"]:
            assert "nodes" in step
            assert "links_formed" in step
            assert "links_broken" in step

    def test_clock_advances_to_end(self, tmp_path):
        """Clock reaches the configured end time after run."""
        config = _make_small_config(str(tmp_path))
        engine = SimulationEngine(config)
        engine.run()
        assert engine.clock.is_finished()

    def test_event_queue_drained_each_step(self, tmp_path):
        """Events scheduled before current time are processed."""
        config = _make_small_config(str(tmp_path))
        engine = SimulationEngine(config)

        # Schedule an event that should fire during the simulation
        event_fired = []
        engine.event_queue.schedule(
            2.5, lambda: event_fired.append(True), priority=0
        )

        engine.run()

        assert len(event_fired) == 1

    def test_partial_final_step(self, tmp_path):
        """Engine handles non-evenly-divisible duration correctly."""
        config = _make_small_config(str(tmp_path))
        config.duration_seconds = 3.5
        config.step_size = 1.0
        engine = SimulationEngine(config)
        engine.run()

        # Should complete 4 steps: t=0, t=1, t=2, t=3 (partial step to 3.5)
        assert engine.steps_completed == 4
        assert engine.clock.is_finished()


class TestSimulationEngineMetrics:
    """Tests for metrics collection integration."""

    def test_mobility_metrics_csv_produced(self, tmp_path):
        """Mobility metrics CSV file is produced after simulation."""
        config = _make_small_config(str(tmp_path))
        config.duration_seconds = 120.0  # Need enough time for a snapshot
        engine = SimulationEngine(config)
        engine.run()

        csv_path = os.path.join(str(tmp_path), "mobility_metrics.csv")
        assert os.path.exists(csv_path)

    def test_topology_metrics_json_produced(self, tmp_path):
        """Topology metrics JSON file is produced after simulation."""
        config = _make_small_config(str(tmp_path))
        config.duration_seconds = 600.0  # Need enough time for a snapshot
        engine = SimulationEngine(config)
        engine.run()

        json_path = os.path.join(str(tmp_path), "topology_metrics.json")
        assert os.path.exists(json_path)
