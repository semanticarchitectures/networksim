"""Property-based tests for ConfigurationManager (manet_sim/core/config.py).

Uses Hypothesis to verify correctness properties of the configuration
loading and validation system.

Feature: manet-simulation-engine
"""

import tempfile
from pathlib import Path

import yaml
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from manet_sim.core.config import (
    ConfigValidationError,
    SimulationConfig,
    MobilityConfig,
    NetworkConfig,
    OutputConfig,
    load,
    validate,
)


# --- Hypothesis Strategies ---


@st.composite
def valid_config_dict(draw):
    """Generate a valid configuration dictionary with all fields within acceptable ranges.

    Constrains values to valid ranges as defined in the ConfigurationManager:
    - duration_seconds: [1.0, 86400.0]
    - step_size: [0.001, 3600.0] and <= duration_seconds
    - seed: [0, 2^32-1] or omitted
    - area width/height: [1.0, 1000.0]
    - node_count: [1, 10000]
    - snapshot_interval: [1, 3600]
    - buffer_size: [1, 1000]
    """
    duration = draw(st.floats(min_value=1.0, max_value=86400.0, allow_nan=False, allow_infinity=False))
    # step_size must be <= duration and within [0.001, 3600.0]
    step_max = min(duration, 3600.0)
    step_size = draw(st.floats(min_value=0.001, max_value=step_max, allow_nan=False, allow_infinity=False))

    # Seed: either present (valid int) or absent
    include_seed = draw(st.booleans())
    seed = draw(st.integers(min_value=0, max_value=2**32 - 1)) if include_seed else None

    area_width = draw(st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False))
    area_height = draw(st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False))
    node_count = draw(st.integers(min_value=1, max_value=10000))

    # Mobility config
    speed_min = draw(st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False))
    speed_max = draw(st.floats(min_value=speed_min, max_value=100.0, allow_nan=False, allow_infinity=False))
    pause_min = draw(st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False))
    pause_max = draw(st.floats(min_value=pause_min, max_value=5000.0, allow_nan=False, allow_infinity=False))
    num_groups = draw(st.integers(min_value=1, max_value=100))
    max_deviation = draw(st.floats(min_value=0.01, max_value=50.0, allow_nan=False, allow_infinity=False))
    deviation_model = draw(st.sampled_from(["uniform", "gaussian"]))

    # Network config
    radio_range = draw(st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False))
    hysteresis_pct = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))

    # Output config
    snapshot_interval = draw(st.integers(min_value=1, max_value=3600))
    buffer_size = draw(st.integers(min_value=1, max_value=1000))

    config = {
        "simulation": {
            "duration_seconds": duration,
            "step_size": step_size,
            "area": {
                "width_miles": area_width,
                "height_miles": area_height,
            },
        },
        "nodes": {"count": node_count},
        "mobility": {
            "model": "rpgm",
            "num_groups": num_groups,
            "group_speed_min_mps": speed_min,
            "group_speed_max_mps": speed_max,
            "pause_min_seconds": pause_min,
            "pause_max_seconds": pause_max,
            "max_deviation_miles": max_deviation,
            "deviation_model": deviation_model,
        },
        "network": {
            "radio_range_miles": radio_range,
            "hysteresis_margin_pct": hysteresis_pct,
        },
        "output": {
            "format": "json",
            "output_dir": "./steering_output",
            "snapshot_interval": snapshot_interval,
            "buffer_size": buffer_size,
        },
    }

    if seed is not None:
        config["simulation"]["seed"] = seed

    return config


@st.composite
def invalid_range_config_dict(draw):
    """Generate a configuration dictionary with one or more fields set to out-of-range values.

    Returns a tuple of (config_dict, set_of_invalid_field_names) where the set
    contains the dotted field names that are out of range.
    """
    # Start with a base valid config
    duration = 3600.0
    step_size = 1.0
    area_width = 100.0
    area_height = 100.0
    node_count = 1000
    snapshot_interval = 60
    buffer_size = 100

    invalid_fields = set()

    # Decide which fields to make invalid (at least one)
    make_duration_invalid = draw(st.booleans())
    make_step_size_invalid = draw(st.booleans())
    make_width_invalid = draw(st.booleans())
    make_height_invalid = draw(st.booleans())
    make_node_count_invalid = draw(st.booleans())
    make_snapshot_invalid = draw(st.booleans())
    make_buffer_invalid = draw(st.booleans())

    # Ensure at least one field is invalid
    any_invalid = (
        make_duration_invalid or make_step_size_invalid or make_width_invalid
        or make_height_invalid or make_node_count_invalid
        or make_snapshot_invalid or make_buffer_invalid
    )
    if not any_invalid:
        make_duration_invalid = True

    if make_duration_invalid:
        # Out of range: < 1.0 or > 86400.0
        duration = draw(st.one_of(
            st.floats(min_value=-1000.0, max_value=0.99, allow_nan=False, allow_infinity=False),
            st.floats(min_value=86400.1, max_value=200000.0, allow_nan=False, allow_infinity=False),
        ))
        invalid_fields.add("simulation.duration_seconds")

    if make_step_size_invalid:
        # Out of range: < 0.001 or > 3600.0
        step_size = draw(st.one_of(
            st.floats(min_value=-100.0, max_value=0.0009, allow_nan=False, allow_infinity=False),
            st.floats(min_value=3600.1, max_value=10000.0, allow_nan=False, allow_infinity=False),
        ))
        invalid_fields.add("simulation.step_size")

    if make_width_invalid:
        area_width = draw(st.one_of(
            st.floats(min_value=-100.0, max_value=0.99, allow_nan=False, allow_infinity=False),
            st.floats(min_value=1000.1, max_value=5000.0, allow_nan=False, allow_infinity=False),
        ))
        invalid_fields.add("simulation.area.width_miles")

    if make_height_invalid:
        area_height = draw(st.one_of(
            st.floats(min_value=-100.0, max_value=0.99, allow_nan=False, allow_infinity=False),
            st.floats(min_value=1000.1, max_value=5000.0, allow_nan=False, allow_infinity=False),
        ))
        invalid_fields.add("simulation.area.height_miles")

    if make_node_count_invalid:
        node_count = draw(st.one_of(
            st.integers(min_value=-100, max_value=0),
            st.integers(min_value=10001, max_value=100000),
        ))
        invalid_fields.add("nodes.count")

    if make_snapshot_invalid:
        snapshot_interval = draw(st.one_of(
            st.integers(min_value=-100, max_value=0),
            st.integers(min_value=3601, max_value=10000),
        ))
        invalid_fields.add("output.snapshot_interval")

    if make_buffer_invalid:
        buffer_size = draw(st.one_of(
            st.integers(min_value=-100, max_value=0),
            st.integers(min_value=1001, max_value=5000),
        ))
        invalid_fields.add("output.buffer_size")

    # Ensure step_size doesn't exceed duration (unless step_size is already invalid)
    # This avoids triggering the "step_size exceeds duration" error which is a
    # cross-field validation, not a range error on a single field
    if not make_step_size_invalid and not make_duration_invalid:
        step_size = min(step_size, duration)

    config = {
        "simulation": {
            "duration_seconds": duration,
            "step_size": step_size,
            "seed": 42,
            "area": {
                "width_miles": area_width,
                "height_miles": area_height,
            },
        },
        "nodes": {"count": node_count},
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
            "snapshot_interval": snapshot_interval,
            "buffer_size": buffer_size,
        },
    }

    return config, invalid_fields


# --- Property 1: Configuration round-trip ---
# Feature: manet-simulation-engine, Property 1: Configuration round-trip
# **Validates: Requirements 2.1**


class TestConfigRoundTrip:
    """Property 1: Configuration round-trip.

    For any valid configuration dictionary with all required fields and values
    within acceptable ranges, serializing it to YAML and loading it via
    ConfigurationManager SHALL produce a SimulationConfig whose fields match
    the original dictionary values.
    """

    @given(config=valid_config_dict())
    @settings(max_examples=100)
    def test_valid_config_round_trips_through_yaml(self, config, tmp_path_factory):
        """Valid config dict → YAML → load → fields match original.

        **Validates: Requirements 2.1**
        """
        # Write config to a temporary YAML file
        tmp_dir = tmp_path_factory.mktemp("config")
        config_path = tmp_dir / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        # Load it back via ConfigurationManager
        result = load(str(config_path))

        # Verify all fields match the original dictionary
        assert isinstance(result, SimulationConfig)

        # Simulation-level fields
        assert result.duration_seconds == float(config["simulation"]["duration_seconds"])
        assert result.step_size == float(config["simulation"]["step_size"])
        assert result.area_width == float(config["simulation"]["area"]["width_miles"])
        assert result.area_height == float(config["simulation"]["area"]["height_miles"])
        assert result.node_count == int(config["nodes"]["count"])

        # Seed handling
        expected_seed = config["simulation"].get("seed")
        if expected_seed is not None:
            assert result.seed == int(expected_seed)
        else:
            assert result.seed is None

        # Mobility config
        mob = config["mobility"]
        assert result.mobility.model == mob["model"]
        assert result.mobility.num_groups == int(mob["num_groups"])
        assert result.mobility.group_speed_min_mps == float(mob["group_speed_min_mps"])
        assert result.mobility.group_speed_max_mps == float(mob["group_speed_max_mps"])
        assert result.mobility.pause_min_seconds == float(mob["pause_min_seconds"])
        assert result.mobility.pause_max_seconds == float(mob["pause_max_seconds"])
        assert result.mobility.max_deviation_miles == float(mob["max_deviation_miles"])
        assert result.mobility.deviation_model == mob["deviation_model"]

        # Network config
        net = config["network"]
        assert result.network.radio_range_miles == float(net["radio_range_miles"])
        assert result.network.hysteresis_margin_pct == float(net["hysteresis_margin_pct"])

        # Output config
        out = config["output"]
        assert result.output.format == out["format"]
        assert result.output.output_dir == out["output_dir"]
        assert result.output.snapshot_interval == int(out["snapshot_interval"])
        assert result.output.buffer_size == int(out["buffer_size"])


# --- Property 2: Configuration validation error reporting ---
# Feature: manet-simulation-engine, Property 2: Configuration validation error reporting
# **Validates: Requirements 2.2, 2.3**


class TestConfigValidationErrorReporting:
    """Property 2: Configuration validation error reporting.

    For any configuration dictionary with one or more fields set to values
    outside their acceptable ranges, the ConfigurationManager SHALL raise an
    error that identifies each invalid field name, its provided value, and the
    acceptable range — and the set of reported fields SHALL exactly equal the
    set of fields with out-of-range values.
    """

    @given(data=invalid_range_config_dict())
    @settings(max_examples=100)
    def test_invalid_fields_produce_exact_error_set(self, data):
        """Invalid fields produce exact error set.

        **Validates: Requirements 2.2, 2.3**
        """
        config, expected_invalid_fields = data

        # The step_size > duration cross-validation may add an extra error.
        # We need to account for this: if step_size is valid on its own but
        # exceeds duration, the validator reports it as a separate error.
        # We filter to only check range-based errors.
        errors = validate(config)

        # There should be at least one error
        assert len(errors) > 0, (
            f"Expected errors for fields {expected_invalid_fields}, got none"
        )

        # Each expected invalid field should appear in at least one error message
        for field_name in expected_invalid_fields:
            # Extract the leaf field name for matching (e.g., "duration_seconds" from
            # "simulation.duration_seconds")
            leaf_name = field_name.split(".")[-1]
            field_mentioned = any(leaf_name in error for error in errors)
            assert field_mentioned, (
                f"Expected field '{field_name}' (leaf: '{leaf_name}') to be mentioned "
                f"in errors, but got: {errors}"
            )

        # Each error about "out of range" should correspond to a field we made invalid
        # (filtering out cross-field validations like "step_size exceeds duration")
        range_errors = [e for e in errors if "out of range" in e]
        for error in range_errors:
            # At least one of our expected invalid fields should be mentioned
            found = any(
                field_name.split(".")[-1] in error
                for field_name in expected_invalid_fields
            )
            assert found, (
                f"Unexpected range error not matching any invalid field: {error}"
            )
