"""Unit tests for main.py entry point."""

import os
import sys
import tempfile
from unittest.mock import patch

import pytest
import yaml

from main import main, parse_args


class TestParseArgs:
    """Tests for CLI argument parsing."""

    def test_default_config_path(self):
        """Default config path is config/sim_config.yaml."""
        args = parse_args([])
        assert args.config == "config/sim_config.yaml"

    def test_custom_config_path(self):
        """Custom config path is accepted as positional argument."""
        args = parse_args(["my_config.yaml"])
        assert args.config == "my_config.yaml"

    def test_custom_config_path_with_directory(self):
        """Config path with directory is accepted."""
        args = parse_args(["/path/to/config.yaml"])
        assert args.config == "/path/to/config.yaml"


class TestMainConfigErrors:
    """Tests for configuration error handling (exit code 1)."""

    def test_missing_config_file_exits_1(self, capsys):
        """Missing config file produces exit code 1."""
        result = main(["nonexistent_file.yaml"])
        assert result == 1
        captured = capsys.readouterr()
        assert "not found" in captured.err.lower() or "not found" in captured.err

    def test_invalid_yaml_exits_1(self, tmp_path, capsys):
        """Invalid YAML syntax produces exit code 1."""
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("invalid: yaml: content: [unclosed")
        result = main([str(bad_yaml)])
        assert result == 1
        captured = capsys.readouterr()
        assert "error" in captured.err.lower()

    def test_invalid_config_values_exits_1(self, tmp_path, capsys):
        """Config with invalid values produces exit code 1."""
        config = {
            "simulation": {
                "duration_seconds": -1,  # Invalid: must be >= 1
                "step_size": 1.0,
                "seed": 42,
                "area": {"width_miles": 100.0, "height_miles": 100.0},
            },
            "nodes": {"count": 1000},
            "mobility": {
                "model": "rpgm",
                "num_groups": 20,
                "group_speed_min_mps": 0.5,
                "group_speed_max_mps": 13.4,
                "pause_min_seconds": 0,
                "pause_max_seconds": 300,
                "max_deviation_miles": 1.25,
                "deviation_model": "uniform",
            },
            "network": {
                "radio_range_miles": 1.0,
                "hysteresis_margin_pct": 0.10,
            },
            "output": {
                "format": "json",
                "output_dir": str(tmp_path / "output"),
                "snapshot_interval": 60,
                "buffer_size": 100,
            },
        }
        config_file = tmp_path / "invalid_config.yaml"
        config_file.write_text(yaml.dump(config))

        result = main([str(config_file)])
        assert result == 1
        captured = capsys.readouterr()
        assert "error" in captured.err.lower()


class TestMainSuccessfulRun:
    """Tests for successful simulation execution."""

    def _make_config_file(self, tmp_path):
        """Create a minimal valid config file for fast testing."""
        output_dir = str(tmp_path / "output")
        config = {
            "simulation": {
                "duration_seconds": 3,
                "step_size": 1.0,
                "seed": 42,
                "area": {"width_miles": 10.0, "height_miles": 10.0},
            },
            "nodes": {"count": 10},
            "mobility": {
                "model": "rpgm",
                "num_groups": 2,
                "group_speed_min_mps": 0.5,
                "group_speed_max_mps": 5.0,
                "pause_min_seconds": 0,
                "pause_max_seconds": 0,
                "max_deviation_miles": 1.0,
                "deviation_model": "uniform",
            },
            "network": {
                "radio_range_miles": 3.0,
                "hysteresis_margin_pct": 0.10,
            },
            "output": {
                "format": "json",
                "output_dir": output_dir,
                "snapshot_interval": 2,
                "buffer_size": 10,
            },
        }
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml.dump(config))
        return str(config_file), output_dir

    def test_successful_run_exits_0(self, tmp_path):
        """Successful simulation run returns exit code 0."""
        config_file, _ = self._make_config_file(tmp_path)
        result = main([config_file])
        assert result == 0

    def test_successful_run_produces_output(self, tmp_path):
        """Successful run produces steering output file."""
        config_file, output_dir = self._make_config_file(tmp_path)
        main([config_file])
        assert os.path.exists(os.path.join(output_dir, "steering_output.json"))

    def test_successful_run_prints_summary(self, tmp_path, capsys):
        """Successful run prints completion summary."""
        config_file, _ = self._make_config_file(tmp_path)
        main([config_file])
        captured = capsys.readouterr()
        assert "SIMULATION COMPLETE" in captured.out
        assert "Total simulated time" in captured.out
        assert "Wall-clock duration" in captured.out
        assert "Steps completed" in captured.out
        assert "Records written" in captured.out

    def test_summary_shows_output_paths(self, tmp_path, capsys):
        """Summary includes output file paths."""
        config_file, output_dir = self._make_config_file(tmp_path)
        main([config_file])
        captured = capsys.readouterr()
        assert "steering_output.json" in captured.out


class TestMainRuntimeErrors:
    """Tests for runtime error handling (exit code 2)."""

    def test_runtime_error_exits_2(self, tmp_path, capsys):
        """Runtime error during simulation produces exit code 2."""
        config_file, output_dir = self._make_config_file_for_runtime_error(tmp_path)

        # Patch the engine's run method to raise an exception
        with patch(
            "main.SimulationEngine.run",
            side_effect=RuntimeError("Simulated failure"),
        ):
            result = main([config_file])

        assert result == 2
        captured = capsys.readouterr()
        assert "error" in captured.err.lower()

    def _make_config_file_for_runtime_error(self, tmp_path):
        """Create a valid config file for runtime error testing."""
        output_dir = str(tmp_path / "output")
        config = {
            "simulation": {
                "duration_seconds": 3,
                "step_size": 1.0,
                "seed": 42,
                "area": {"width_miles": 10.0, "height_miles": 10.0},
            },
            "nodes": {"count": 10},
            "mobility": {
                "model": "rpgm",
                "num_groups": 2,
                "group_speed_min_mps": 0.5,
                "group_speed_max_mps": 5.0,
                "pause_min_seconds": 0,
                "pause_max_seconds": 0,
                "max_deviation_miles": 1.0,
                "deviation_model": "uniform",
            },
            "network": {
                "radio_range_miles": 3.0,
                "hysteresis_margin_pct": 0.10,
            },
            "output": {
                "format": "json",
                "output_dir": output_dir,
                "snapshot_interval": 2,
                "buffer_size": 10,
            },
        }
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml.dump(config))
        return str(config_file), output_dir
