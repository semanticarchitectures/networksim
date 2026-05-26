"""MANET Simulation Engine entry point.

Loads configuration, builds the SimulationEngine, runs the simulation
with a tqdm progress bar, and prints a completion summary.

Exit codes:
    0 — Simulation completed successfully
    1 — Configuration error (file not found, parse error, validation error)
    2 — Runtime error during simulation execution
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from tqdm import tqdm

from manet_sim.core.config import (
    ConfigParseError,
    ConfigValidationError,
    load,
)
from manet_sim.core.event_bus import STEP_COMPLETE
from manet_sim.core.simulation import SimulationEngine


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to sys.argv[1:] if None).

    Returns:
        Parsed namespace with config_path attribute.
    """
    parser = argparse.ArgumentParser(
        description="MANET Simulation Engine — simulate mobile ad hoc networks"
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="config/sim_config.yaml",
        help="Path to YAML configuration file (default: config/sim_config.yaml)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the MANET simulation.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:] if None).

    Returns:
        Exit code: 0 for success, 1 for config errors, 2 for runtime errors.
    """
    args = parse_args(argv)
    config_path = args.config

    # --- Load configuration ---
    try:
        config = load(config_path)
    except FileNotFoundError:
        print(
            f"Error: Configuration file not found: {config_path}",
            file=sys.stderr,
        )
        return 1
    except ConfigParseError as e:
        print(
            f"Error: Failed to parse configuration file '{config_path}': {e}",
            file=sys.stderr,
        )
        return 1
    except ConfigValidationError as e:
        print(
            f"Error: Configuration validation failed for '{config_path}':\n{e}",
            file=sys.stderr,
        )
        return 1

    # --- Build simulation engine ---
    try:
        engine = SimulationEngine(config)
    except Exception as e:
        print(
            f"Error: Failed to initialize simulation engine: {e}",
            file=sys.stderr,
        )
        return 1

    # --- Compute total steps for progress bar ---
    total_steps = int(config.duration_seconds / config.step_size)
    # Account for partial final step
    if config.duration_seconds % config.step_size != 0:
        total_steps += 1

    # --- Set up progress bar with event bus subscription ---
    progress_bar = tqdm(
        total=total_steps,
        desc="Simulating",
        unit="step",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} steps [{elapsed}<{remaining}]",
    )

    def on_step_complete(event) -> None:
        """Update progress bar on each step completion."""
        progress_bar.update(1)

    engine.event_bus.subscribe(STEP_COMPLETE, on_step_complete)

    # --- Run simulation ---
    wall_clock_start = time.perf_counter()
    try:
        engine.run()
    except KeyboardInterrupt:
        progress_bar.close()
        print("\nSimulation interrupted by user.", file=sys.stderr)
        return 2
    except Exception as e:
        progress_bar.close()
        sim_time = engine.clock.current_time
        print(
            f"Error: Runtime error at simulation time {sim_time:.2f}s: {e}",
            file=sys.stderr,
        )
        return 2
    finally:
        if not progress_bar.disable:
            progress_bar.close()

    wall_clock_duration = time.perf_counter() - wall_clock_start

    # --- Print completion summary ---
    output_dir = config.output.output_dir
    steering_file = os.path.join(output_dir, "steering_output.json")
    metrics_csv = os.path.join(output_dir, "mobility_metrics.csv")
    topology_json = os.path.join(output_dir, "topology_metrics.json")

    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)
    print(f"  Total simulated time:  {config.duration_seconds:.1f} seconds")
    print(f"  Wall-clock duration:   {wall_clock_duration:.2f} seconds")
    print(f"  Steps completed:       {engine.steps_completed}")
    print(f"  Records written:       {engine.steps_completed} step records")
    print(f"  Seed used:             {engine.seed}")
    print("-" * 60)
    print("  Output files:")
    print(f"    Steering file:       {steering_file}")
    if os.path.exists(metrics_csv):
        print(f"    Mobility metrics:    {metrics_csv}")
    if os.path.exists(topology_json):
        print(f"    Topology metrics:    {topology_json}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
