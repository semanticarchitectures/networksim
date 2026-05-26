# NetworkSim — MANET Simulation Engine

A Python-based Mobile Ad Hoc Network (MANET) simulator that models 1,000 mobile nodes moving across a 10,000 square mile area over a 1-hour duration. Nodes are organized into 20 groups using the Reference Point Group Mobility (RPGM) model. The simulator produces structured JSON steering files capturing node positions, topology changes, and network metrics at each time step.

## Features

- **1,000 nodes** across 100×100 miles with group-coordinated movement
- **Hybrid time-stepped + event-driven** architecture for efficiency
- **Hysteresis-based link management** preventing link flapping near radio range boundaries
- **Vectorized NumPy operations** for batch position updates
- **Grid-based spatial indexing** reducing neighbor queries from O(n²) to O(n·k)
- **Buffered JSON output** with configurable flush intervals and topology snapshots
- **Full reproducibility** via seeded RNG with independent subsystem streams
- **431 tests** including Hypothesis property-based tests validating 27 correctness properties

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run with default configuration (1000 nodes, 3600s, seed 42)
python main.py

# Run with a custom config
python main.py path/to/config.yaml
```

## Configuration

All parameters are specified in YAML. The default config is at `config/sim_config.yaml`:

```yaml
simulation:
  duration_seconds: 3600
  step_size: 1.0
  seed: 42
  area:
    width_miles: 100.0
    height_miles: 100.0

nodes:
  count: 1000

mobility:
  model: rpgm
  num_groups: 20
  group_speed_min_mps: 0.5
  group_speed_max_mps: 13.4
  pause_min_seconds: 0
  pause_max_seconds: 300
  max_deviation_miles: 1.25
  deviation_model: uniform

network:
  radio_range_miles: 1.0
  hysteresis_margin_pct: 0.10

output:
  format: json
  output_dir: ./steering_output
  snapshot_interval: 60
  buffer_size: 100
```

## Project Structure

```
NetworkSim/
├── main.py                          # CLI entry point
├── config/sim_config.yaml           # Default configuration
├── requirements.txt                 # Python dependencies
├── manet_sim/
│   ├── core/
│   │   ├── config.py                # Configuration loading and validation
│   │   ├── clock.py                 # Simulation clock with partial final step
│   │   ├── seed_manager.py          # Reproducible RNG streams per subsystem
│   │   ├── event_queue.py           # Min-heap priority queue for discrete events
│   │   ├── event_bus.py             # Pub/sub with fault isolation
│   │   └── simulation.py           # Main simulation engine orchestrator
│   ├── mobility/
│   │   ├── group_mobility.py        # RPGM model + Group data structure
│   │   └── spatial_index.py         # Grid-based spatial hash for neighbor queries
│   ├── topology/
│   │   ├── link_manager.py          # Hysteresis-based link state transitions
│   │   └── topology_updater.py      # Incremental NetworkX graph management
│   ├── network/
│   │   └── metrics.py               # Mobility and topology metrics collectors
│   └── output/
│       ├── steering_writer.py       # Buffered JSON steering file output
│       └── steering_parser.py       # Parser and pretty-printer for steering files
└── tests/                           # 431 tests (unit + property-based)
```

## Architecture

The simulation loop executes per time step:

1. **Mobility update** — advance all node positions via RPGM
2. **Spatial index rebuild** — O(n) grid reconstruction
3. **Topology update** — compute link deltas with hysteresis
4. **Event drain** — process discrete events up to current time
5. **Metrics publish** — deliver events to collectors via event bus
6. **Output write** — buffer step data to steering file
7. **Clock advance** — move to next time step

## Output

The simulator produces:

- **Steering file** (`steering_output/steering_output.json`) — per-step node positions, velocities, group memberships, and topology deltas
- **Mobility metrics** (`steering_output/mobility_metrics.csv`) — speed, displacement, and group cohesion snapshots
- **Topology metrics** (`steering_output/topology_metrics.json`) — degree, components, and clustering snapshots

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run only property-based tests
python -m pytest tests/ -v -k "properties"

# Run with coverage
python -m pytest tests/ --cov=manet_sim --cov-report=html
```

The test suite includes 27 Hypothesis property-based tests validating invariants such as:
- Clock always reaches exact end time
- Positions remain within simulation bounds after every step
- Spatial index has zero false negatives
- Hysteresis prevents link flapping in the buffer zone
- Steering file round-trip preserves state within 1e-6 tolerance

## Dependencies

- Python 3.10+
- NumPy ≥ 1.26
- NetworkX ≥ 3.2
- SciPy ≥ 1.11
- PyYAML ≥ 6.0.1
- tqdm ≥ 4.66
- Hypothesis ≥ 6.0 (testing)

## License

MIT
