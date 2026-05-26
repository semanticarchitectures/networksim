"""Configuration management for the MANET simulation engine.

Loads and validates YAML configuration files, providing typed access
to all simulation parameters via dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import yaml


class ConfigParseError(Exception):
    """Raised when YAML parsing fails. Includes line number information."""

    def __init__(self, message: str, line: Optional[int] = None):
        self.line = line
        if line is not None:
            message = f"YAML parse error at line {line}: {message}"
        super().__init__(message)


class ConfigValidationError(Exception):
    """Raised when configuration validation fails.

    Attributes:
        errors: List of validation error messages.
    """

    def __init__(self, errors: list[str]):
        self.errors = errors
        message = "Configuration validation failed:\n" + "\n".join(
            f"  - {e}" for e in errors
        )
        super().__init__(message)


@dataclass
class MobilityConfig:
    """Configuration for the mobility model."""

    model: str = "rpgm"
    num_groups: int = 20
    group_speed_min_mps: float = 0.5
    group_speed_max_mps: float = 13.4
    pause_min_seconds: float = 0.0
    pause_max_seconds: float = 300.0
    max_deviation_miles: float = 1.25
    deviation_model: str = "uniform"


@dataclass
class NetworkConfig:
    """Configuration for network topology parameters."""

    radio_range_miles: float = 1.0
    hysteresis_margin_pct: float = 0.10


@dataclass
class OutputConfig:
    """Configuration for simulation output."""

    format: str = "json"
    output_dir: str = "./steering_output"
    snapshot_interval: int = 60
    buffer_size: int = 100


@dataclass
class SimulationConfig:
    """Top-level simulation configuration."""

    duration_seconds: float = 3600.0
    step_size: float = 1.0
    seed: Optional[int] = 42
    area_width: float = 100.0
    area_height: float = 100.0
    node_count: int = 1000
    mobility: MobilityConfig = field(default_factory=MobilityConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


# Validation range definitions
_RANGES = {
    "simulation.duration_seconds": (1.0, 86400.0),
    "simulation.step_size": (0.001, 3600.0),
    "simulation.area.width_miles": (1.0, 1000.0),
    "simulation.area.height_miles": (1.0, 1000.0),
    "nodes.count": (1, 10000),
    "output.snapshot_interval": (1, 3600),
    "output.buffer_size": (1, 1000),
}

_SEED_MAX = 2**32 - 1

_VALID_MOBILITY_MODELS = {"rpgm"}
_VALID_DEVIATION_MODELS = {"uniform", "gaussian"}
_VALID_OUTPUT_FORMATS = {"json"}


def _get_nested(d: dict, dotted_key: str, default=None):
    """Retrieve a value from a nested dict using a dotted key path."""
    keys = dotted_key.split(".")
    current = d
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def validate(config: dict) -> list[str]:
    """Validate a configuration dictionary and return a list of errors.

    Args:
        config: Raw configuration dictionary (as parsed from YAML).

    Returns:
        List of validation error strings. Empty list means valid.
    """
    errors: list[str] = []

    # Check required top-level sections
    required_sections = ["simulation", "nodes", "mobility", "network", "output"]
    for section in required_sections:
        if section not in config or not isinstance(config.get(section), dict):
            errors.append(f"Missing required section: '{section}'")

    # If top-level sections are missing, we can't validate further
    if errors:
        return errors

    # Check required fields within simulation section
    sim = config.get("simulation", {})
    if "duration_seconds" not in sim:
        errors.append("Missing required field: 'simulation.duration_seconds'")
    if "step_size" not in sim:
        errors.append("Missing required field: 'simulation.step_size'")

    # Area subsection
    if "area" not in sim or not isinstance(sim.get("area"), dict):
        errors.append("Missing required field: 'simulation.area'")
    else:
        area = sim["area"]
        if "width_miles" not in area:
            errors.append("Missing required field: 'simulation.area.width_miles'")
        if "height_miles" not in area:
            errors.append("Missing required field: 'simulation.area.height_miles'")

    # Nodes section
    nodes = config.get("nodes", {})
    if "count" not in nodes:
        errors.append("Missing required field: 'nodes.count'")

    # Mobility section required fields
    mobility = config.get("mobility", {})
    mobility_required = [
        "model",
        "num_groups",
        "group_speed_min_mps",
        "group_speed_max_mps",
        "pause_min_seconds",
        "pause_max_seconds",
        "max_deviation_miles",
        "deviation_model",
    ]
    for f in mobility_required:
        if f not in mobility:
            errors.append(f"Missing required field: 'mobility.{f}'")

    # Network section required fields
    network = config.get("network", {})
    network_required = ["radio_range_miles", "hysteresis_margin_pct"]
    for f in network_required:
        if f not in network:
            errors.append(f"Missing required field: 'network.{f}'")

    # Output section required fields
    output = config.get("output", {})
    output_required = ["format", "output_dir", "snapshot_interval", "buffer_size"]
    for f in output_required:
        if f not in output:
            errors.append(f"Missing required field: 'output.{f}'")

    # If there are missing fields, return early before range checks
    if errors:
        return errors

    # Range validations
    range_checks = [
        ("simulation.duration_seconds", sim.get("duration_seconds"), 1.0, 86400.0),
        ("simulation.step_size", sim.get("step_size"), 0.001, 3600.0),
        (
            "simulation.area.width_miles",
            sim.get("area", {}).get("width_miles"),
            1.0,
            1000.0,
        ),
        (
            "simulation.area.height_miles",
            sim.get("area", {}).get("height_miles"),
            1.0,
            1000.0,
        ),
        ("nodes.count", nodes.get("count"), 1, 10000),
        ("output.snapshot_interval", output.get("snapshot_interval"), 1, 3600),
        ("output.buffer_size", output.get("buffer_size"), 1, 1000),
    ]

    for field_name, value, min_val, max_val in range_checks:
        if value is not None:
            try:
                numeric_val = float(value) if isinstance(min_val, float) else int(value)
                if numeric_val < min_val or numeric_val > max_val:
                    errors.append(
                        f"Field '{field_name}' value {value} is out of range "
                        f"[{min_val}, {max_val}]"
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"Field '{field_name}' value {value!r} is not a valid number"
                )

    # Seed validation (optional field)
    seed = sim.get("seed")
    if seed is not None:
        try:
            seed_val = int(seed)
            if seed_val < 0 or seed_val > _SEED_MAX:
                errors.append(
                    f"Field 'simulation.seed' value {seed} is out of range "
                    f"[0, {_SEED_MAX}]"
                )
        except (TypeError, ValueError):
            errors.append(
                f"Field 'simulation.seed' value {seed!r} is not a valid integer"
            )

    # Mobility model validation
    model = mobility.get("model")
    if model not in _VALID_MOBILITY_MODELS:
        errors.append(
            f"Field 'mobility.model' value '{model}' is not valid. "
            f"Must be one of: {sorted(_VALID_MOBILITY_MODELS)}"
        )

    deviation_model = mobility.get("deviation_model")
    if deviation_model not in _VALID_DEVIATION_MODELS:
        errors.append(
            f"Field 'mobility.deviation_model' value '{deviation_model}' is not valid. "
            f"Must be one of: {sorted(_VALID_DEVIATION_MODELS)}"
        )

    # Speed range consistency
    speed_min = mobility.get("group_speed_min_mps")
    speed_max = mobility.get("group_speed_max_mps")
    if speed_min is not None and speed_max is not None:
        try:
            if float(speed_min) > float(speed_max):
                errors.append(
                    f"Field 'mobility.group_speed_min_mps' ({speed_min}) must be "
                    f"<= 'mobility.group_speed_max_mps' ({speed_max})"
                )
        except (TypeError, ValueError):
            pass

    # Pause range consistency
    pause_min = mobility.get("pause_min_seconds")
    pause_max = mobility.get("pause_max_seconds")
    if pause_min is not None and pause_max is not None:
        try:
            if float(pause_min) > float(pause_max):
                errors.append(
                    f"Field 'mobility.pause_min_seconds' ({pause_min}) must be "
                    f"<= 'mobility.pause_max_seconds' ({pause_max})"
                )
        except (TypeError, ValueError):
            pass

    # Output format validation
    fmt = output.get("format")
    if fmt not in _VALID_OUTPUT_FORMATS:
        errors.append(
            f"Field 'output.format' value '{fmt}' is not valid. "
            f"Must be one of: {sorted(_VALID_OUTPUT_FORMATS)}"
        )

    # Step size must not exceed duration
    duration = sim.get("duration_seconds")
    step_size = sim.get("step_size")
    if duration is not None and step_size is not None:
        try:
            if float(step_size) > float(duration):
                errors.append(
                    f"Field 'simulation.step_size' ({step_size}) must not exceed "
                    f"'simulation.duration_seconds' ({duration})"
                )
        except (TypeError, ValueError):
            pass

    return errors


def load(path: str) -> SimulationConfig:
    """Load and validate a YAML configuration file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        A validated SimulationConfig instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ConfigParseError: If the YAML is syntactically invalid.
        ConfigValidationError: If the config has missing fields or out-of-range values.
    """
    import os

    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")

    try:
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        line = None
        if hasattr(e, "problem_mark") and e.problem_mark is not None:
            line = e.problem_mark.line + 1  # YAML uses 0-based lines
        msg = str(e.problem) if hasattr(e, "problem") else str(e)
        raise ConfigParseError(msg, line=line) from e

    if raw is None:
        raise ConfigValidationError(["Configuration file is empty"])

    if not isinstance(raw, dict):
        raise ConfigValidationError(
            ["Configuration file must contain a YAML mapping (dictionary)"]
        )

    # Validate the raw config
    errors = validate(raw)
    if errors:
        raise ConfigValidationError(errors)

    # Build typed config from validated dict
    sim = raw["simulation"]
    area = sim["area"]
    nodes = raw["nodes"]
    mobility = raw["mobility"]
    network = raw["network"]
    output = raw["output"]

    mobility_config = MobilityConfig(
        model=mobility["model"],
        num_groups=int(mobility["num_groups"]),
        group_speed_min_mps=float(mobility["group_speed_min_mps"]),
        group_speed_max_mps=float(mobility["group_speed_max_mps"]),
        pause_min_seconds=float(mobility["pause_min_seconds"]),
        pause_max_seconds=float(mobility["pause_max_seconds"]),
        max_deviation_miles=float(mobility["max_deviation_miles"]),
        deviation_model=mobility["deviation_model"],
    )

    network_config = NetworkConfig(
        radio_range_miles=float(network["radio_range_miles"]),
        hysteresis_margin_pct=float(network["hysteresis_margin_pct"]),
    )

    output_config = OutputConfig(
        format=output["format"],
        output_dir=output["output_dir"],
        snapshot_interval=int(output["snapshot_interval"]),
        buffer_size=int(output["buffer_size"]),
    )

    seed = sim.get("seed")
    if seed is not None:
        seed = int(seed)

    return SimulationConfig(
        duration_seconds=float(sim["duration_seconds"]),
        step_size=float(sim["step_size"]),
        seed=seed,
        area_width=float(area["width_miles"]),
        area_height=float(area["height_miles"]),
        node_count=int(nodes["count"]),
        mobility=mobility_config,
        network=network_config,
        output=output_config,
    )


def default_config() -> SimulationConfig:
    """Return a default configuration that produces a runnable simulation.

    Returns:
        SimulationConfig with default values: 1000 nodes, 100x100 miles,
        3600 seconds duration, step size 1.0, seed 42.
    """
    return SimulationConfig()
