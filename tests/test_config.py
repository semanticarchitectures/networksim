"""Unit tests for the ConfigurationManager (manet_sim/core/config.py)."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from manet_sim.core.config import (
    ConfigParseError,
    ConfigValidationError,
    MobilityConfig,
    NetworkConfig,
    OutputConfig,
    SimulationConfig,
    default_config,
    load,
    validate,
)


# --- Fixtures ---


@pytest.fixture
def valid_config_dict():
    """A complete, valid configuration dictionary."""
    return {
        "simulation": {
            "duration_seconds": 3600,
            "step_size": 1.0,
            "seed": 42,
            "area": {
                "width_miles": 100.0,
                "height_miles": 100.0,
            },
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
            "output_dir": "./steering_output",
            "snapshot_interval": 60,
            "buffer_size": 100,
        },
    }


@pytest.fixture
def config_file(valid_config_dict, tmp_path):
    """Write a valid config dict to a temp YAML file and return the path."""
    path = tmp_path / "test_config.yaml"
    with open(path, "w") as f:
        yaml.dump(valid_config_dict, f)
    return str(path)


# --- Tests for validate() ---


class TestValidate:
    def test_valid_config_returns_no_errors(self, valid_config_dict):
        errors = validate(valid_config_dict)
        assert errors == []

    def test_missing_top_level_section(self, valid_config_dict):
        del valid_config_dict["simulation"]
        errors = validate(valid_config_dict)
        assert any("simulation" in e for e in errors)

    def test_missing_multiple_sections(self):
        errors = validate({})
        assert len(errors) >= 1
        # Should report all missing sections
        combined = " ".join(errors)
        for section in ["simulation", "nodes", "mobility", "network", "output"]:
            assert section in combined

    def test_missing_required_field_duration(self, valid_config_dict):
        del valid_config_dict["simulation"]["duration_seconds"]
        errors = validate(valid_config_dict)
        assert any("duration_seconds" in e for e in errors)

    def test_missing_required_field_step_size(self, valid_config_dict):
        del valid_config_dict["simulation"]["step_size"]
        errors = validate(valid_config_dict)
        assert any("step_size" in e for e in errors)

    def test_missing_area_section(self, valid_config_dict):
        del valid_config_dict["simulation"]["area"]
        errors = validate(valid_config_dict)
        assert any("area" in e for e in errors)

    def test_missing_node_count(self, valid_config_dict):
        del valid_config_dict["nodes"]["count"]
        errors = validate(valid_config_dict)
        assert any("count" in e for e in errors)

    def test_duration_out_of_range_low(self, valid_config_dict):
        valid_config_dict["simulation"]["duration_seconds"] = 0.5
        errors = validate(valid_config_dict)
        assert any("duration_seconds" in e and "out of range" in e for e in errors)

    def test_duration_out_of_range_high(self, valid_config_dict):
        valid_config_dict["simulation"]["duration_seconds"] = 100000
        errors = validate(valid_config_dict)
        assert any("duration_seconds" in e and "out of range" in e for e in errors)

    def test_step_size_out_of_range(self, valid_config_dict):
        valid_config_dict["simulation"]["step_size"] = 0.0001
        errors = validate(valid_config_dict)
        assert any("step_size" in e and "out of range" in e for e in errors)

    def test_area_width_out_of_range(self, valid_config_dict):
        valid_config_dict["simulation"]["area"]["width_miles"] = 0.5
        errors = validate(valid_config_dict)
        assert any("width_miles" in e and "out of range" in e for e in errors)

    def test_node_count_out_of_range(self, valid_config_dict):
        valid_config_dict["nodes"]["count"] = 0
        errors = validate(valid_config_dict)
        assert any("count" in e and "out of range" in e for e in errors)

    def test_invalid_seed_negative(self, valid_config_dict):
        valid_config_dict["simulation"]["seed"] = -1
        errors = validate(valid_config_dict)
        assert any("seed" in e for e in errors)

    def test_invalid_seed_too_large(self, valid_config_dict):
        valid_config_dict["simulation"]["seed"] = 2**32
        errors = validate(valid_config_dict)
        assert any("seed" in e for e in errors)

    def test_seed_none_is_valid(self, valid_config_dict):
        del valid_config_dict["simulation"]["seed"]
        errors = validate(valid_config_dict)
        assert errors == []

    def test_seed_zero_is_valid(self, valid_config_dict):
        valid_config_dict["simulation"]["seed"] = 0
        errors = validate(valid_config_dict)
        assert errors == []

    def test_seed_max_is_valid(self, valid_config_dict):
        valid_config_dict["simulation"]["seed"] = 2**32 - 1
        errors = validate(valid_config_dict)
        assert errors == []

    def test_invalid_mobility_model(self, valid_config_dict):
        valid_config_dict["mobility"]["model"] = "invalid_model"
        errors = validate(valid_config_dict)
        assert any("model" in e for e in errors)

    def test_invalid_deviation_model(self, valid_config_dict):
        valid_config_dict["mobility"]["deviation_model"] = "invalid"
        errors = validate(valid_config_dict)
        assert any("deviation_model" in e for e in errors)

    def test_gaussian_deviation_model_valid(self, valid_config_dict):
        valid_config_dict["mobility"]["deviation_model"] = "gaussian"
        errors = validate(valid_config_dict)
        assert errors == []

    def test_speed_min_greater_than_max(self, valid_config_dict):
        valid_config_dict["mobility"]["group_speed_min_mps"] = 20.0
        valid_config_dict["mobility"]["group_speed_max_mps"] = 5.0
        errors = validate(valid_config_dict)
        assert any("group_speed_min_mps" in e for e in errors)

    def test_pause_min_greater_than_max(self, valid_config_dict):
        valid_config_dict["mobility"]["pause_min_seconds"] = 500
        valid_config_dict["mobility"]["pause_max_seconds"] = 100
        errors = validate(valid_config_dict)
        assert any("pause_min_seconds" in e for e in errors)

    def test_invalid_output_format(self, valid_config_dict):
        valid_config_dict["output"]["format"] = "xml"
        errors = validate(valid_config_dict)
        assert any("format" in e for e in errors)

    def test_buffer_size_out_of_range(self, valid_config_dict):
        valid_config_dict["output"]["buffer_size"] = 1001
        errors = validate(valid_config_dict)
        assert any("buffer_size" in e and "out of range" in e for e in errors)

    def test_snapshot_interval_out_of_range(self, valid_config_dict):
        valid_config_dict["output"]["snapshot_interval"] = 0
        errors = validate(valid_config_dict)
        assert any("snapshot_interval" in e and "out of range" in e for e in errors)

    def test_step_size_exceeds_duration(self, valid_config_dict):
        valid_config_dict["simulation"]["step_size"] = 100.0
        valid_config_dict["simulation"]["duration_seconds"] = 50.0
        errors = validate(valid_config_dict)
        assert any("step_size" in e and "exceed" in e for e in errors)

    def test_multiple_errors_reported(self, valid_config_dict):
        valid_config_dict["simulation"]["duration_seconds"] = 0.5
        valid_config_dict["nodes"]["count"] = 0
        valid_config_dict["output"]["buffer_size"] = 2000
        errors = validate(valid_config_dict)
        assert len(errors) >= 3


# --- Tests for load() ---


class TestLoad:
    def test_load_valid_config(self, config_file):
        config = load(config_file)
        assert isinstance(config, SimulationConfig)
        assert config.duration_seconds == 3600.0
        assert config.step_size == 1.0
        assert config.seed == 42
        assert config.area_width == 100.0
        assert config.area_height == 100.0
        assert config.node_count == 1000

    def test_load_mobility_config(self, config_file):
        config = load(config_file)
        assert config.mobility.model == "rpgm"
        assert config.mobility.num_groups == 20
        assert config.mobility.group_speed_min_mps == 0.5
        assert config.mobility.group_speed_max_mps == 13.4
        assert config.mobility.pause_min_seconds == 0.0
        assert config.mobility.pause_max_seconds == 300.0
        assert config.mobility.max_deviation_miles == 1.25
        assert config.mobility.deviation_model == "uniform"

    def test_load_network_config(self, config_file):
        config = load(config_file)
        assert config.network.radio_range_miles == 1.0
        assert config.network.hysteresis_margin_pct == 0.10

    def test_load_output_config(self, config_file):
        config = load(config_file)
        assert config.output.format == "json"
        assert config.output.output_dir == "./steering_output"
        assert config.output.snapshot_interval == 60
        assert config.output.buffer_size == 100

    def test_load_file_not_found(self):
        with pytest.raises(FileNotFoundError) as exc_info:
            load("/nonexistent/path/config.yaml")
        assert "not found" in str(exc_info.value).lower()

    def test_load_invalid_yaml(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("invalid: yaml: [unclosed")
        with pytest.raises(ConfigParseError) as exc_info:
            load(str(path))
        assert exc_info.value.line is not None

    def test_load_empty_file(self, tmp_path):
        path = tmp_path / "empty.yaml"
        path.write_text("")
        with pytest.raises(ConfigValidationError) as exc_info:
            load(str(path))
        assert "empty" in str(exc_info.value).lower()

    def test_load_missing_fields_raises_validation_error(self, tmp_path):
        path = tmp_path / "partial.yaml"
        path.write_text(yaml.dump({"simulation": {"duration_seconds": 100}}))
        with pytest.raises(ConfigValidationError) as exc_info:
            load(str(path))
        assert len(exc_info.value.errors) > 0

    def test_load_out_of_range_raises_validation_error(
        self, valid_config_dict, tmp_path
    ):
        valid_config_dict["simulation"]["duration_seconds"] = 0.1
        path = tmp_path / "bad_range.yaml"
        with open(path, "w") as f:
            yaml.dump(valid_config_dict, f)
        with pytest.raises(ConfigValidationError) as exc_info:
            load(str(path))
        assert any("duration_seconds" in e for e in exc_info.value.errors)

    def test_load_seed_none_when_omitted(self, valid_config_dict, tmp_path):
        del valid_config_dict["simulation"]["seed"]
        path = tmp_path / "no_seed.yaml"
        with open(path, "w") as f:
            yaml.dump(valid_config_dict, f)
        config = load(str(path))
        assert config.seed is None

    def test_load_actual_config_file(self):
        """Test loading the actual project config file."""
        project_config = os.path.join(
            os.path.dirname(__file__), "..", "config", "sim_config.yaml"
        )
        if os.path.exists(project_config):
            config = load(project_config)
            assert config.node_count == 1000
            assert config.duration_seconds == 3600.0


# --- Tests for default_config() ---


class TestDefaultConfig:
    def test_default_config_returns_simulation_config(self):
        config = default_config()
        assert isinstance(config, SimulationConfig)

    def test_default_config_values(self):
        config = default_config()
        assert config.duration_seconds == 3600.0
        assert config.step_size == 1.0
        assert config.seed == 42
        assert config.area_width == 100.0
        assert config.area_height == 100.0
        assert config.node_count == 1000

    def test_default_config_mobility(self):
        config = default_config()
        assert config.mobility.model == "rpgm"
        assert config.mobility.num_groups == 20
        assert config.mobility.deviation_model == "uniform"

    def test_default_config_network(self):
        config = default_config()
        assert config.network.radio_range_miles == 1.0
        assert config.network.hysteresis_margin_pct == 0.10

    def test_default_config_output(self):
        config = default_config()
        assert config.output.format == "json"
        assert config.output.snapshot_interval == 60
        assert config.output.buffer_size == 100


# --- Tests for ConfigParseError ---


class TestConfigParseError:
    def test_includes_line_number(self):
        err = ConfigParseError("unexpected token", line=5)
        assert err.line == 5
        assert "line 5" in str(err)

    def test_no_line_number(self):
        err = ConfigParseError("generic error")
        assert err.line is None


# --- Tests for ConfigValidationError ---


class TestConfigValidationError:
    def test_stores_errors_list(self):
        errors = ["error 1", "error 2"]
        err = ConfigValidationError(errors)
        assert err.errors == errors
        assert "error 1" in str(err)
        assert "error 2" in str(err)
