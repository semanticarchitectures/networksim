"""Unit tests for SteeringWriter."""

import json
import os
import tempfile

import numpy as np
import pytest

from manet_sim.core.config import OutputConfig
from manet_sim.output.steering_writer import SteeringWriter, _serialize_node
from manet_sim.topology.topology_updater import TopologyDelta


@pytest.fixture
def tmp_output_dir():
    """Create a temporary directory for test output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def default_config(tmp_output_dir):
    """Create a default OutputConfig pointing to temp directory."""
    return OutputConfig(
        format="json",
        output_dir=tmp_output_dir,
        snapshot_interval=60,
        buffer_size=100,
    )


@pytest.fixture
def default_metadata():
    """Create default metadata for tests."""
    return {
        "seed": 42,
        "config": {"duration_seconds": 3600, "step_size": 1.0},
        "node_count": 5,
        "total_steps": 3600,
        "transmission_range": 1.0,
    }


@pytest.fixture
def sample_positions():
    """Create sample positions array for 5 nodes."""
    return np.array(
        [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0], [70.0, 80.0], [90.0, 10.0]],
        dtype=np.float32,
    )


@pytest.fixture
def sample_velocities():
    """Create sample velocities array for 5 nodes."""
    return np.array(
        [[0.001, -0.002], [0.003, 0.004], [-0.001, 0.002], [0.0, 0.0], [0.005, -0.003]],
        dtype=np.float32,
    )


@pytest.fixture
def sample_group_ids():
    """Create sample group IDs for 5 nodes."""
    return np.array([0, 0, 1, 1, 2], dtype=np.int32)


@pytest.fixture
def empty_delta():
    """Create an empty TopologyDelta."""
    return TopologyDelta(timestamp=0.0)


class TestSerializeNode:
    """Tests for the _serialize_node helper function."""

    def test_contains_all_required_keys(self):
        """Serialized node dict contains all required keys."""
        result = _serialize_node(
            node_id=0, group_id=1, x=50.123456, y=78.654321,
            vx=0.001234, vy=-0.000567, transmission_range=1.0, active=True,
        )
        required_keys = {
            "node_id", "group_id", "x", "y", "vx", "vy",
            "transmission_range", "active",
        }
        assert set(result.keys()) == required_keys

    def test_numeric_values_max_6_decimal_places(self):
        """Numeric values are rounded to no more than 6 decimal places."""
        result = _serialize_node(
            node_id=0, group_id=1, x=50.12345678, y=78.65432198,
            vx=0.00123456789, vy=-0.00056789012,
            transmission_range=1.123456789, active=True,
        )
        # Check that string representation has at most 6 decimal places
        for key in ["x", "y", "vx", "vy", "transmission_range"]:
            value_str = f"{result[key]:.10f}".rstrip("0")
            decimal_part = value_str.split(".")[1] if "." in value_str else ""
            assert len(decimal_part) <= 6, f"{key} has more than 6 decimal places"

    def test_correct_types(self):
        """Serialized values have correct types."""
        result = _serialize_node(
            node_id=5, group_id=2, x=10.0, y=20.0,
            vx=0.001, vy=-0.002, transmission_range=1.0, active=False,
        )
        assert isinstance(result["node_id"], int)
        assert isinstance(result["group_id"], int)
        assert isinstance(result["x"], float)
        assert isinstance(result["y"], float)
        assert isinstance(result["vx"], float)
        assert isinstance(result["vy"], float)
        assert isinstance(result["transmission_range"], float)
        assert isinstance(result["active"], bool)

    def test_preserves_node_id(self):
        """Node ID is preserved correctly."""
        result = _serialize_node(
            node_id=42, group_id=3, x=0.0, y=0.0,
            vx=0.0, vy=0.0, transmission_range=1.0, active=True,
        )
        assert result["node_id"] == 42

    def test_preserves_active_status(self):
        """Active status is preserved correctly."""
        result_active = _serialize_node(
            node_id=0, group_id=0, x=0.0, y=0.0,
            vx=0.0, vy=0.0, transmission_range=1.0, active=True,
        )
        result_inactive = _serialize_node(
            node_id=0, group_id=0, x=0.0, y=0.0,
            vx=0.0, vy=0.0, transmission_range=1.0, active=False,
        )
        assert result_active["active"] is True
        assert result_inactive["active"] is False


class TestSteeringWriterOpen:
    """Tests for SteeringWriter.open()."""

    def test_creates_output_directory(self, tmp_output_dir, default_metadata):
        """open() creates the output directory if it doesn't exist."""
        nested_dir = os.path.join(tmp_output_dir, "nested", "output")
        config = OutputConfig(
            format="json", output_dir=nested_dir,
            snapshot_interval=60, buffer_size=100,
        )
        writer = SteeringWriter(config, default_metadata)
        writer.open()
        assert os.path.isdir(nested_dir)
        writer.close()

    def test_creates_output_file(self, default_config, default_metadata):
        """open() creates the output file."""
        writer = SteeringWriter(default_config, default_metadata)
        writer.open()
        writer.close()
        assert os.path.isfile(writer.file_path)

    def test_file_path_set_after_open(self, default_config, default_metadata):
        """file_path property is set after open()."""
        writer = SteeringWriter(default_config, default_metadata)
        writer.open()
        assert writer.file_path.endswith("steering_output.json")
        writer.close()


class TestSteeringWriterWriteStep:
    """Tests for SteeringWriter.write_step()."""

    def test_write_step_without_open_raises(
        self, default_config, default_metadata, sample_positions,
        sample_velocities, sample_group_ids, empty_delta,
    ):
        """write_step() raises RuntimeError if open() not called."""
        writer = SteeringWriter(default_config, default_metadata)
        with pytest.raises(RuntimeError):
            writer.write_step(
                0.0, sample_positions, sample_velocities,
                sample_group_ids, empty_delta,
            )

    def test_single_step_buffered(
        self, default_config, default_metadata, sample_positions,
        sample_velocities, sample_group_ids, empty_delta,
    ):
        """A single step is buffered without immediate flush."""
        writer = SteeringWriter(default_config, default_metadata)
        writer.open()
        writer.write_step(
            0.0, sample_positions, sample_velocities,
            sample_group_ids, empty_delta,
        )
        # Buffer should have 1 item, not yet flushed
        assert len(writer._buffer) == 1
        writer.close()

    def test_step_data_contains_correct_timestamp(
        self, default_config, default_metadata, sample_positions,
        sample_velocities, sample_group_ids, empty_delta,
    ):
        """Written step data contains the correct timestamp."""
        writer = SteeringWriter(default_config, default_metadata)
        writer.open()
        writer.write_step(
            42.5, sample_positions, sample_velocities,
            sample_group_ids, empty_delta,
        )
        assert writer._buffer[0]["timestamp"] == 42.5
        writer.close()

    def test_step_data_contains_all_nodes(
        self, default_config, default_metadata, sample_positions,
        sample_velocities, sample_group_ids, empty_delta,
    ):
        """Written step data contains entries for all nodes."""
        writer = SteeringWriter(default_config, default_metadata)
        writer.open()
        writer.write_step(
            0.0, sample_positions, sample_velocities,
            sample_group_ids, empty_delta,
        )
        assert len(writer._buffer[0]["nodes"]) == 5
        writer.close()

    def test_step_data_contains_topology_delta(
        self, default_config, default_metadata, sample_positions,
        sample_velocities, sample_group_ids,
    ):
        """Written step data includes topology delta information."""
        delta = TopologyDelta(
            timestamp=1.0,
            new_links=[(0, 1, 0.85), (2, 3, 0.92)],
            broken_links=[(4, 5)],
        )
        writer = SteeringWriter(default_config, default_metadata)
        writer.open()
        writer.write_step(
            1.0, sample_positions, sample_velocities,
            sample_group_ids, delta,
        )
        step = writer._buffer[0]
        assert len(step["links_formed"]) == 2
        assert len(step["links_broken"]) == 1
        assert step["links_formed"][0]["node_a"] == 0
        assert step["links_formed"][0]["node_b"] == 1
        assert step["links_formed"][0]["distance"] == 0.85
        writer.close()


class TestSteeringWriterBuffering:
    """Tests for buffer flush behavior."""

    def test_buffer_flushes_at_configured_size(
        self, tmp_output_dir, default_metadata, sample_positions,
        sample_velocities, sample_group_ids, empty_delta,
    ):
        """Buffer flushes when it reaches configured buffer_size."""
        config = OutputConfig(
            format="json", output_dir=tmp_output_dir,
            snapshot_interval=60, buffer_size=5,
        )
        writer = SteeringWriter(config, default_metadata)
        writer.open()

        for i in range(5):
            delta = TopologyDelta(timestamp=float(i))
            writer.write_step(
                float(i), sample_positions, sample_velocities,
                sample_group_ids, delta,
            )

        # Buffer should be empty after flush
        assert len(writer._buffer) == 0
        assert writer.steps_written == 5
        writer.close()

    def test_buffer_size_clamped_to_max_1000(
        self, tmp_output_dir, default_metadata,
    ):
        """Buffer size is clamped to maximum of 1000."""
        config = OutputConfig(
            format="json", output_dir=tmp_output_dir,
            snapshot_interval=60, buffer_size=2000,
        )
        writer = SteeringWriter(config, default_metadata)
        assert writer._buffer_size == 1000

    def test_partial_buffer_flushed_on_close(
        self, tmp_output_dir, default_metadata, sample_positions,
        sample_velocities, sample_group_ids, empty_delta,
    ):
        """Remaining buffer is flushed when close() is called."""
        config = OutputConfig(
            format="json", output_dir=tmp_output_dir,
            snapshot_interval=3600,  # No snapshots during test
            buffer_size=100,
        )
        writer = SteeringWriter(config, default_metadata)
        writer.open()

        # Write 3 steps (less than buffer_size)
        for i in range(3):
            delta = TopologyDelta(timestamp=float(i))
            writer.write_step(
                float(i), sample_positions, sample_velocities,
                sample_group_ids, delta,
            )

        writer.close()

        # Verify output file contains all 3 steps
        with open(writer.file_path, "r") as f:
            output = json.load(f)
        assert len(output["steps"]) == 3


class TestSteeringWriterClose:
    """Tests for SteeringWriter.close()."""

    def test_produces_valid_json(
        self, default_config, default_metadata, sample_positions,
        sample_velocities, sample_group_ids, empty_delta,
    ):
        """close() produces a valid, parseable JSON file."""
        writer = SteeringWriter(default_config, default_metadata)
        writer.open()
        writer.write_step(
            0.0, sample_positions, sample_velocities,
            sample_group_ids, empty_delta,
        )
        writer.close()

        with open(writer.file_path, "r") as f:
            output = json.load(f)  # Should not raise

        assert "metadata" in output
        assert "steps" in output
        assert "snapshots" in output

    def test_metadata_contains_required_fields(
        self, default_config, default_metadata, sample_positions,
        sample_velocities, sample_group_ids, empty_delta,
    ):
        """Output metadata contains all required fields."""
        writer = SteeringWriter(default_config, default_metadata)
        writer.open()
        writer.write_step(
            0.0, sample_positions, sample_velocities,
            sample_group_ids, empty_delta,
        )
        writer.close()

        with open(writer.file_path, "r") as f:
            output = json.load(f)

        metadata = output["metadata"]
        assert metadata["schema_version"] == "1.0.0"
        assert metadata["seed"] == 42
        assert metadata["config"] == {"duration_seconds": 3600, "step_size": 1.0}
        assert metadata["simulation_start"] is not None
        assert metadata["simulation_end"] is not None
        assert metadata["total_steps"] == 3600
        assert metadata["node_count"] == 5

    def test_close_without_open_does_not_raise(
        self, default_config, default_metadata,
    ):
        """close() without prior open() does not raise."""
        writer = SteeringWriter(default_config, default_metadata)
        writer.close()  # Should not raise

    def test_simulation_end_time_set_on_close(
        self, default_config, default_metadata, sample_positions,
        sample_velocities, sample_group_ids, empty_delta,
    ):
        """simulation_end is set to a valid ISO 8601 timestamp on close."""
        writer = SteeringWriter(default_config, default_metadata)
        writer.open()
        writer.write_step(
            0.0, sample_positions, sample_velocities,
            sample_group_ids, empty_delta,
        )
        writer.close()

        with open(writer.file_path, "r") as f:
            output = json.load(f)

        end_time = output["metadata"]["simulation_end"]
        assert end_time is not None
        # Should be parseable as ISO 8601
        from datetime import datetime
        datetime.fromisoformat(end_time)


class TestSteeringWriterSnapshots:
    """Tests for topology snapshot writing."""

    def test_snapshot_written_at_interval(
        self, tmp_output_dir, default_metadata, sample_positions,
        sample_velocities, sample_group_ids,
    ):
        """Full topology snapshot is written at configured interval."""
        config = OutputConfig(
            format="json", output_dir=tmp_output_dir,
            snapshot_interval=60, buffer_size=100,
        )
        writer = SteeringWriter(config, default_metadata)
        writer.open()

        # Write steps at t=0 (triggers snapshot), t=1, ..., t=60 (triggers snapshot)
        for t in range(61):
            delta = TopologyDelta(timestamp=float(t))
            writer.write_step(
                float(t), sample_positions, sample_velocities,
                sample_group_ids, delta,
            )

        writer.close()

        with open(writer.file_path, "r") as f:
            output = json.load(f)

        # Should have snapshots at t=0 and t=60
        assert len(output["snapshots"]) == 2
        assert output["snapshots"][0]["timestamp"] == 0.0
        assert output["snapshots"][1]["timestamp"] == 60.0

    def test_snapshot_contains_all_nodes(
        self, tmp_output_dir, default_metadata, sample_positions,
        sample_velocities, sample_group_ids, empty_delta,
    ):
        """Snapshot contains entries for all nodes."""
        config = OutputConfig(
            format="json", output_dir=tmp_output_dir,
            snapshot_interval=60, buffer_size=100,
        )
        writer = SteeringWriter(config, default_metadata)
        writer.open()

        # First step triggers snapshot at t=0
        writer.write_step(
            0.0, sample_positions, sample_velocities,
            sample_group_ids, empty_delta,
        )
        writer.close()

        with open(writer.file_path, "r") as f:
            output = json.load(f)

        snapshot = output["snapshots"][0]
        assert len(snapshot["nodes"]) == 5

    def test_snapshot_node_contains_required_fields(
        self, tmp_output_dir, default_metadata, sample_positions,
        sample_velocities, sample_group_ids, empty_delta,
    ):
        """Snapshot node entries contain required fields."""
        config = OutputConfig(
            format="json", output_dir=tmp_output_dir,
            snapshot_interval=60, buffer_size=100,
        )
        writer = SteeringWriter(config, default_metadata)
        writer.open()
        writer.write_step(
            0.0, sample_positions, sample_velocities,
            sample_group_ids, empty_delta,
        )
        writer.close()

        with open(writer.file_path, "r") as f:
            output = json.load(f)

        node = output["snapshots"][0]["nodes"][0]
        assert "node_id" in node
        assert "x" in node
        assert "y" in node
        assert "vx" in node
        assert "vy" in node
        assert "group_id" in node


class TestSteeringWriterFlushFailure:
    """Tests for flush failure handling."""

    def test_flush_retry_on_failure(
        self, tmp_output_dir, default_metadata, sample_positions,
        sample_velocities, sample_group_ids, empty_delta,
    ):
        """Flush retries once on failure and succeeds on retry."""
        config = OutputConfig(
            format="json", output_dir=tmp_output_dir,
            snapshot_interval=3600, buffer_size=2,
        )
        writer = SteeringWriter(config, default_metadata)
        writer.open()

        # Patch _commit_buffer to fail on first call, succeed on retry
        call_count = [0]
        original_commit = writer._commit_buffer

        def failing_commit(buffer):
            call_count[0] += 1
            if call_count[0] == 1:
                raise OSError("Simulated write failure")
            original_commit(buffer)

        writer._commit_buffer = failing_commit

        delta = TopologyDelta(timestamp=0.0)
        writer.write_step(
            0.0, sample_positions, sample_velocities,
            sample_group_ids, delta,
        )
        delta = TopologyDelta(timestamp=1.0)
        writer.write_step(
            1.0, sample_positions, sample_velocities,
            sample_group_ids, delta,
        )

        # Should have succeeded on retry
        assert writer.steps_written == 2
        writer.close()

    def test_flush_logs_error_on_double_failure(
        self, tmp_output_dir, default_metadata, sample_positions,
        sample_velocities, sample_group_ids, caplog,
    ):
        """Flush logs error with lost record count on second failure."""
        config = OutputConfig(
            format="json", output_dir=tmp_output_dir,
            snapshot_interval=3600, buffer_size=2,
        )
        writer = SteeringWriter(config, default_metadata)
        writer.open()

        # Patch _commit_buffer to always fail
        def always_fail(buffer):
            raise OSError("Persistent failure")

        writer._commit_buffer = always_fail

        import logging
        with caplog.at_level(logging.ERROR):
            writer.write_step(
                0.0, sample_positions, sample_velocities,
                sample_group_ids, TopologyDelta(timestamp=0.0),
            )
            writer.write_step(
                1.0, sample_positions, sample_velocities,
                sample_group_ids, TopologyDelta(timestamp=1.0),
            )

        assert "Lost 2 step records" in caplog.text
        # Restore for close
        writer._commit_buffer = lambda buf: writer._all_steps.extend(buf)
        writer.close()


class TestSteeringWriterIntegration:
    """Integration tests for the full write workflow."""

    def test_full_workflow_produces_valid_output(
        self, tmp_output_dir, sample_positions, sample_velocities,
        sample_group_ids,
    ):
        """Full workflow: open, write multiple steps, close produces valid JSON."""
        config = OutputConfig(
            format="json", output_dir=tmp_output_dir,
            snapshot_interval=5, buffer_size=3,
        )
        metadata = {
            "seed": 123,
            "config": {"test": True},
            "node_count": 5,
            "total_steps": 10,
            "transmission_range": 1.0,
        }
        writer = SteeringWriter(config, metadata)
        writer.open()

        for t in range(10):
            delta = TopologyDelta(
                timestamp=float(t),
                new_links=[(0, 1, 0.5)] if t == 0 else [],
                broken_links=[(0, 1)] if t == 9 else [],
            )
            writer.write_step(
                float(t), sample_positions, sample_velocities,
                sample_group_ids, delta,
            )

        writer.close()

        with open(writer.file_path, "r") as f:
            output = json.load(f)

        # Verify structure
        assert output["metadata"]["seed"] == 123
        assert output["metadata"]["node_count"] == 5
        assert output["metadata"]["total_steps"] == 10
        assert len(output["steps"]) == 10

        # Verify first step has link formed
        assert len(output["steps"][0]["links_formed"]) == 1
        assert output["steps"][0]["links_formed"][0]["node_a"] == 0
        assert output["steps"][0]["links_formed"][0]["node_b"] == 1

        # Verify last step has link broken
        assert len(output["steps"][9]["links_broken"]) == 1

        # Verify snapshots at interval of 5
        # Snapshots at t=0, t=5
        assert len(output["snapshots"]) == 2

    def test_node_positions_preserved_in_output(
        self, tmp_output_dir, default_metadata,
    ):
        """Node positions are correctly preserved in the output."""
        config = OutputConfig(
            format="json", output_dir=tmp_output_dir,
            snapshot_interval=3600, buffer_size=100,
        )
        positions = np.array([[12.345678, 98.765432]], dtype=np.float32)
        velocities = np.array([[0.001234, -0.005678]], dtype=np.float32)
        group_ids = np.array([3], dtype=np.int32)

        metadata = {**default_metadata, "node_count": 1}
        writer = SteeringWriter(config, metadata)
        writer.open()
        writer.write_step(
            0.0, positions, velocities, group_ids,
            TopologyDelta(timestamp=0.0),
        )
        writer.close()

        with open(writer.file_path, "r") as f:
            output = json.load(f)

        node = output["steps"][0]["nodes"][0]
        assert node["node_id"] == 0
        assert node["group_id"] == 3
        # Float32 precision means values may differ slightly
        assert abs(node["x"] - 12.345678) < 1e-4
        assert abs(node["y"] - 98.765432) < 1e-4
        assert node["active"] is True

    def test_steps_written_property_tracks_count(
        self, tmp_output_dir, default_metadata, sample_positions,
        sample_velocities, sample_group_ids,
    ):
        """steps_written property accurately tracks flushed step count."""
        config = OutputConfig(
            format="json", output_dir=tmp_output_dir,
            snapshot_interval=3600, buffer_size=5,
        )
        writer = SteeringWriter(config, default_metadata)
        writer.open()

        for i in range(7):
            writer.write_step(
                float(i), sample_positions, sample_velocities,
                sample_group_ids, TopologyDelta(timestamp=float(i)),
            )

        # 5 flushed, 2 still in buffer
        assert writer.steps_written == 5
        writer.close()
        # After close, all should be flushed
        assert writer.steps_written == 7
