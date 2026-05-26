"""Unit tests for SteeringParser and SteeringPrinter."""

import json
import os
import tempfile

import numpy as np
import pytest

from manet_sim.output.steering_parser import (
    LinkEvent,
    MalformedFileError,
    SchemaViolationError,
    SimulationState,
    SteeringParser,
    SteeringPrinter,
    TimestepNotFoundError,
)


@pytest.fixture
def parser():
    """Create a SteeringParser instance."""
    return SteeringParser()


@pytest.fixture
def printer():
    """Create a SteeringPrinter instance."""
    return SteeringPrinter()


@pytest.fixture
def valid_steering_data():
    """Create valid steering file data."""
    return {
        "metadata": {
            "schema_version": "1.0.0",
            "seed": 42,
            "config": {"duration_seconds": 3600, "step_size": 1.0},
            "simulation_start": "2024-01-01T00:00:00+00:00",
            "simulation_end": "2024-01-01T01:00:00+00:00",
            "total_steps": 3,
            "node_count": 3,
        },
        "steps": [
            {
                "timestamp": 0.0,
                "nodes": [
                    {"node_id": 0, "group_id": 0, "x": 10.123456,
                     "y": 20.654321, "vx": 0.001, "vy": -0.002},
                    {"node_id": 1, "group_id": 0, "x": 11.0,
                     "y": 21.0, "vx": 0.002, "vy": 0.001},
                    {"node_id": 2, "group_id": 1, "x": 50.0,
                     "y": 60.0, "vx": -0.001, "vy": 0.003},
                ],
                "links_formed": [
                    {"node_a": 0, "node_b": 1,
                     "distance": 0.85, "timestamp": 0.0}
                ],
                "links_broken": [],
            },
            {
                "timestamp": 1.0,
                "nodes": [
                    {"node_id": 0, "group_id": 0, "x": 10.124456,
                     "y": 20.652321, "vx": 0.001, "vy": -0.002},
                    {"node_id": 1, "group_id": 0, "x": 11.002,
                     "y": 21.001, "vx": 0.002, "vy": 0.001},
                    {"node_id": 2, "group_id": 1, "x": 49.999,
                     "y": 60.003, "vx": -0.001, "vy": 0.003},
                ],
                "links_formed": [],
                "links_broken": [],
            },
            {
                "timestamp": 2.0,
                "nodes": [
                    {"node_id": 0, "group_id": 0, "x": 10.125456,
                     "y": 20.650321, "vx": 0.001, "vy": -0.002},
                    {"node_id": 1, "group_id": 0, "x": 11.004,
                     "y": 21.002, "vx": 0.002, "vy": 0.001},
                    {"node_id": 2, "group_id": 1, "x": 49.998,
                     "y": 60.006, "vx": -0.001, "vy": 0.003},
                ],
                "links_formed": [],
                "links_broken": [{"node_a": 0, "node_b": 1}],
            },
        ],
        "snapshots": [
            {
                "timestamp": 0.0,
                "nodes": [
                    {"node_id": 0, "x": 10.123456, "y": 20.654321,
                     "vx": 0.001, "vy": -0.002, "group_id": 0},
                    {"node_id": 1, "x": 11.0, "y": 21.0,
                     "vx": 0.002, "vy": 0.001, "group_id": 0},
                    {"node_id": 2, "x": 50.0, "y": 60.0,
                     "vx": -0.001, "vy": 0.003, "group_id": 1},
                ],
            }
        ],
    }


@pytest.fixture
def steering_file(valid_steering_data, tmp_path):
    """Write valid steering data to a temp file and return the path."""
    filepath = str(tmp_path / "steering_output.json")
    with open(filepath, "w") as f:
        json.dump(valid_steering_data, f)
    return filepath


class TestSteeringParserParse:
    """Tests for SteeringParser.parse()."""

    def test_parse_valid_timestep_returns_state(self, parser, steering_file):
        """Parsing a valid timestep returns a SimulationState."""
        state = parser.parse(steering_file, 0.0)
        assert isinstance(state, SimulationState)
        assert state.timestamp == 0.0
        assert state.node_count == 3

    def test_parse_reconstructs_positions(self, parser, steering_file):
        """Parsed state has correct positions array."""
        state = parser.parse(steering_file, 0.0)
        assert state.positions.shape == (3, 2)
        assert abs(state.positions[0, 0] - 10.123456) < 1e-6
        assert abs(state.positions[0, 1] - 20.654321) < 1e-6

    def test_parse_reconstructs_velocities(self, parser, steering_file):
        """Parsed state has correct velocities array."""
        state = parser.parse(steering_file, 0.0)
        assert state.velocities.shape == (3, 2)
        assert abs(state.velocities[0, 0] - 0.001) < 1e-6
        assert abs(state.velocities[0, 1] - (-0.002)) < 1e-6

    def test_parse_reconstructs_group_ids(self, parser, steering_file):
        """Parsed state has correct group_ids array."""
        state = parser.parse(steering_file, 0.0)
        assert state.group_ids.shape == (3,)
        assert state.group_ids[0] == 0
        assert state.group_ids[1] == 0
        assert state.group_ids[2] == 1

    def test_parse_reconstructs_links_formed(self, parser, steering_file):
        """Parsed state has correct links_formed list."""
        state = parser.parse(steering_file, 0.0)
        assert len(state.links_formed) == 1
        link = state.links_formed[0]
        assert link.node_a == 0
        assert link.node_b == 1
        assert abs(link.distance - 0.85) < 1e-6

    def test_parse_reconstructs_links_broken(self, parser, steering_file):
        """Parsed state has correct links_broken list."""
        state = parser.parse(steering_file, 2.0)
        assert len(state.links_broken) == 1
        link = state.links_broken[0]
        assert link.node_a == 0
        assert link.node_b == 1

    def test_parse_different_timesteps(self, parser, steering_file):
        """Parsing different timesteps returns different states."""
        state0 = parser.parse(steering_file, 0.0)
        state1 = parser.parse(steering_file, 1.0)
        # Positions should differ between timesteps
        assert not np.allclose(state0.positions, state1.positions)

    def test_parse_missing_timestep_raises(self, parser, steering_file):
        """Parsing a non-existent timestep raises TimestepNotFoundError."""
        with pytest.raises(TimestepNotFoundError) as exc_info:
            parser.parse(steering_file, 99.0)
        assert "99.0" in str(exc_info.value)

    def test_parse_malformed_json_raises(self, parser, tmp_path):
        """Parsing a file with invalid JSON raises MalformedFileError."""
        filepath = str(tmp_path / "bad.json")
        with open(filepath, "w") as f:
            f.write("{not valid json")
        with pytest.raises(MalformedFileError):
            parser.parse(filepath, 0.0)

    def test_parse_nonexistent_file_raises(self, parser):
        """Parsing a non-existent file raises MalformedFileError."""
        with pytest.raises(MalformedFileError):
            parser.parse("/nonexistent/path.json", 0.0)

    def test_parse_schema_violation_raises(self, parser, tmp_path):
        """Parsing a file missing required fields raises SchemaViolationError."""
        filepath = str(tmp_path / "incomplete.json")
        with open(filepath, "w") as f:
            json.dump({"metadata": {"schema_version": "1.0.0"}}, f)
        with pytest.raises(SchemaViolationError):
            parser.parse(filepath, 0.0)

    def test_parse_preserves_position_precision(self, parser, tmp_path):
        """Position values are preserved to at least 6 decimal places."""
        data = {
            "metadata": {"schema_version": "1.0.0", "node_count": 1},
            "steps": [{
                "timestamp": 0.0,
                "nodes": [{"node_id": 0, "group_id": 0,
                           "x": 12.345678, "y": 98.765432,
                           "vx": 0.001234, "vy": -0.005678}],
                "links_formed": [],
                "links_broken": [],
            }],
            "snapshots": [],
        }
        filepath = str(tmp_path / "precision.json")
        with open(filepath, "w") as f:
            json.dump(data, f)
        state = parser.parse(filepath, 0.0)
        assert abs(state.positions[0, 0] - 12.345678) < 1e-6
        assert abs(state.positions[0, 1] - 98.765432) < 1e-6
        assert abs(state.velocities[0, 0] - 0.001234) < 1e-6
        assert abs(state.velocities[0, 1] - (-0.005678)) < 1e-6


class TestSteeringParserValidateSchema:
    """Tests for SteeringParser.validate_schema()."""

    def test_valid_file_returns_empty_list(self, parser, steering_file):
        """A valid steering file returns no errors."""
        errors = parser.validate_schema(steering_file)
        assert errors == []

    def test_malformed_json_returns_error(self, parser, tmp_path):
        """Malformed JSON returns an error message."""
        filepath = str(tmp_path / "bad.json")
        with open(filepath, "w") as f:
            f.write("not json at all")
        errors = parser.validate_schema(filepath)
        assert len(errors) == 1
        assert "Malformed JSON" in errors[0]

    def test_missing_metadata_returns_error(self, parser, tmp_path):
        """Missing metadata field returns an error."""
        data = {"steps": [], "snapshots": []}
        filepath = str(tmp_path / "no_meta.json")
        with open(filepath, "w") as f:
            json.dump(data, f)
        errors = parser.validate_schema(filepath)
        assert any("metadata" in e for e in errors)

    def test_missing_steps_returns_error(self, parser, tmp_path):
        """Missing steps field returns an error."""
        data = {
            "metadata": {"schema_version": "1.0.0", "node_count": 0},
            "snapshots": [],
        }
        filepath = str(tmp_path / "no_steps.json")
        with open(filepath, "w") as f:
            json.dump(data, f)
        errors = parser.validate_schema(filepath)
        assert any("steps" in e for e in errors)

    def test_missing_snapshots_returns_error(self, parser, tmp_path):
        """Missing snapshots field returns an error."""
        data = {
            "metadata": {"schema_version": "1.0.0", "node_count": 0},
            "steps": [],
        }
        filepath = str(tmp_path / "no_snap.json")
        with open(filepath, "w") as f:
            json.dump(data, f)
        errors = parser.validate_schema(filepath)
        assert any("snapshots" in e for e in errors)

    def test_step_missing_timestamp_returns_error(self, parser, tmp_path):
        """A step missing 'timestamp' returns an error."""
        data = {
            "metadata": {"schema_version": "1.0.0", "node_count": 1},
            "steps": [{"nodes": [], "links_formed": [], "links_broken": []}],
            "snapshots": [],
        }
        filepath = str(tmp_path / "no_ts.json")
        with open(filepath, "w") as f:
            json.dump(data, f)
        errors = parser.validate_schema(filepath)
        assert any("timestamp" in e for e in errors)

    def test_step_missing_nodes_returns_error(self, parser, tmp_path):
        """A step missing 'nodes' returns an error."""
        data = {
            "metadata": {"schema_version": "1.0.0", "node_count": 1},
            "steps": [{"timestamp": 0.0, "links_formed": [],
                       "links_broken": []}],
            "snapshots": [],
        }
        filepath = str(tmp_path / "no_nodes.json")
        with open(filepath, "w") as f:
            json.dump(data, f)
        errors = parser.validate_schema(filepath)
        assert any("nodes" in e for e in errors)

    def test_metadata_missing_schema_version_returns_error(
        self, parser, tmp_path
    ):
        """Metadata missing schema_version returns an error."""
        data = {
            "metadata": {"node_count": 1},
            "steps": [],
            "snapshots": [],
        }
        filepath = str(tmp_path / "no_version.json")
        with open(filepath, "w") as f:
            json.dump(data, f)
        errors = parser.validate_schema(filepath)
        assert any("schema_version" in e for e in errors)

    def test_multiple_errors_reported(self, parser, tmp_path):
        """Multiple schema violations are all reported."""
        data = {"metadata": "not_a_dict", "steps": "not_a_list"}
        filepath = str(tmp_path / "multi_err.json")
        with open(filepath, "w") as f:
            json.dump(data, f)
        errors = parser.validate_schema(filepath)
        assert len(errors) >= 2


class TestSteeringPrinterFormat:
    """Tests for SteeringPrinter.format()."""

    def test_format_returns_valid_json(self, printer):
        """format() returns a valid JSON string."""
        state = SimulationState(
            timestamp=0.0,
            positions=np.array([[10.0, 20.0], [30.0, 40.0]]),
            velocities=np.array([[0.001, -0.002], [0.003, 0.004]]),
            group_ids=np.array([0, 1], dtype=np.int32),
            links_formed=[],
            links_broken=[],
            node_count=2,
        )
        result = printer.format(state)
        parsed = json.loads(result)  # Should not raise
        assert isinstance(parsed, dict)

    def test_format_uses_2_space_indent(self, printer):
        """format() uses 2-space indentation."""
        state = SimulationState(
            timestamp=0.0,
            positions=np.array([[10.0, 20.0]]),
            velocities=np.array([[0.001, -0.002]]),
            group_ids=np.array([0], dtype=np.int32),
            links_formed=[],
            links_broken=[],
            node_count=1,
        )
        result = printer.format(state)
        lines = result.split("\n")
        # Second line should be indented with 2 spaces
        indented_lines = [l for l in lines if l.startswith("  ")]
        assert len(indented_lines) > 0
        # No 4-space indentation at top level
        assert any(l.startswith("  ") and not l.startswith("    ") for l in lines)

    def test_format_uses_sorted_keys(self, printer):
        """format() uses sorted keys for deterministic output."""
        state = SimulationState(
            timestamp=5.0,
            positions=np.array([[10.0, 20.0]]),
            velocities=np.array([[0.001, -0.002]]),
            group_ids=np.array([0], dtype=np.int32),
            links_formed=[],
            links_broken=[],
            node_count=1,
        )
        result = printer.format(state)
        parsed = json.loads(result)
        # Top-level keys should be sorted
        keys = list(parsed.keys())
        assert keys == sorted(keys)

    def test_format_deterministic(self, printer):
        """Calling format() twice produces identical output."""
        state = SimulationState(
            timestamp=1.0,
            positions=np.array([[10.5, 20.3], [30.7, 40.1]]),
            velocities=np.array([[0.001, -0.002], [0.003, 0.004]]),
            group_ids=np.array([0, 1], dtype=np.int32),
            links_formed=[LinkEvent(node_a=0, node_b=1,
                                    distance=0.85, timestamp=1.0)],
            links_broken=[],
            node_count=2,
        )
        result1 = printer.format(state)
        result2 = printer.format(state)
        assert result1 == result2

    def test_format_includes_all_nodes(self, printer):
        """format() includes all nodes in the output."""
        state = SimulationState(
            timestamp=0.0,
            positions=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            velocities=np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
            group_ids=np.array([0, 0, 1], dtype=np.int32),
            links_formed=[],
            links_broken=[],
            node_count=3,
        )
        result = printer.format(state)
        parsed = json.loads(result)
        assert len(parsed["nodes"]) == 3

    def test_format_includes_link_events(self, printer):
        """format() includes link formed and broken events."""
        state = SimulationState(
            timestamp=5.0,
            positions=np.array([[10.0, 20.0], [11.0, 21.0]]),
            velocities=np.array([[0.001, -0.002], [0.002, 0.001]]),
            group_ids=np.array([0, 0], dtype=np.int32),
            links_formed=[LinkEvent(node_a=0, node_b=1,
                                    distance=1.41, timestamp=5.0)],
            links_broken=[LinkEvent(node_a=2, node_b=3)],
            node_count=2,
        )
        result = printer.format(state)
        parsed = json.loads(result)
        assert len(parsed["links_formed"]) == 1
        assert parsed["links_formed"][0]["node_a"] == 0
        assert parsed["links_formed"][0]["node_b"] == 1
        assert len(parsed["links_broken"]) == 1

    def test_format_numeric_precision(self, printer):
        """format() rounds numeric values to at most 6 decimal places."""
        state = SimulationState(
            timestamp=0.0,
            positions=np.array([[12.34567890123, 98.76543210987]]),
            velocities=np.array([[0.00123456789, -0.00567890123]]),
            group_ids=np.array([0], dtype=np.int32),
            links_formed=[],
            links_broken=[],
            node_count=1,
        )
        result = printer.format(state)
        parsed = json.loads(result)
        node = parsed["nodes"][0]
        # Values should be rounded to 6 decimal places
        x_str = f"{node['x']:.10f}".rstrip("0")
        decimal_part = x_str.split(".")[1] if "." in x_str else ""
        assert len(decimal_part) <= 6


class TestRoundTrip:
    """Tests for parse-then-print round-trip property."""

    def test_round_trip_preserves_state(self, parser, printer, tmp_path):
        """Printing a state then parsing it back preserves the state."""
        # Create a state
        original_state = SimulationState(
            timestamp=5.0,
            positions=np.array([[10.123456, 20.654321],
                                [30.111111, 40.222222]]),
            velocities=np.array([[0.001234, -0.005678],
                                 [0.003456, 0.007890]]),
            group_ids=np.array([0, 1], dtype=np.int32),
            links_formed=[LinkEvent(node_a=0, node_b=1,
                                    distance=0.85, timestamp=5.0)],
            links_broken=[],
            node_count=2,
        )

        # Print to JSON
        json_str = printer.format(original_state)

        # Wrap in a valid steering file structure for parsing
        step_data = json.loads(json_str)
        steering_data = {
            "metadata": {"schema_version": "1.0.0", "node_count": 2},
            "steps": [step_data],
            "snapshots": [],
        }

        filepath = str(tmp_path / "roundtrip.json")
        with open(filepath, "w") as f:
            json.dump(steering_data, f)

        # Parse back
        parsed_state = parser.parse(filepath, 5.0)

        # Verify field-by-field equality within tolerance
        assert abs(parsed_state.timestamp - original_state.timestamp) < 1e-6
        assert parsed_state.node_count == original_state.node_count
        assert np.allclose(
            parsed_state.positions, original_state.positions, atol=1e-6
        )
        assert np.allclose(
            parsed_state.velocities, original_state.velocities, atol=1e-6
        )
        assert np.array_equal(parsed_state.group_ids, original_state.group_ids)
        assert len(parsed_state.links_formed) == len(original_state.links_formed)
        assert (
            parsed_state.links_formed[0].node_a
            == original_state.links_formed[0].node_a
        )
        assert (
            parsed_state.links_formed[0].node_b
            == original_state.links_formed[0].node_b
        )
        assert abs(
            parsed_state.links_formed[0].distance
            - original_state.links_formed[0].distance
        ) < 1e-6
