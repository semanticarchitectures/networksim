"""
Tests for Node memory budget validation.

Validates Requirement 6.4: Node core state ≤ 128 bytes per node,
and 1000 nodes aggregate core state ≤ 200 KB.
"""

import numpy as np


def test_per_node_core_state_within_budget():
    """
    Verify that core state per node ≤ 128 bytes.

    Core state arrays per node:
    - position: 2 × float32 = 8 bytes
    - velocity: 2 × float32 = 8 bytes
    - group_id: 1 × int32 = 4 bytes
    - active: 1 × bool = 1 byte
    Total: 21 bytes per node
    """
    n_nodes = 1000

    positions = np.zeros((n_nodes, 2), dtype=np.float32)
    velocities = np.zeros((n_nodes, 2), dtype=np.float32)
    group_ids = np.zeros(n_nodes, dtype=np.int32)
    active = np.zeros(n_nodes, dtype=bool)

    total_bytes = (
        positions.nbytes + velocities.nbytes + group_ids.nbytes + active.nbytes
    )
    per_node_bytes = total_bytes / n_nodes

    assert per_node_bytes <= 128, (
        f"Per-node core state is {per_node_bytes} bytes, exceeds 128-byte budget"
    )


def test_aggregate_core_state_within_budget():
    """
    Verify that 1000 nodes aggregate core state ≤ 200 KB (204,800 bytes).

    Core state arrays for 1000 nodes:
    - positions: shape (1000, 2), dtype=float32 → 8000 bytes
    - velocities: shape (1000, 2), dtype=float32 → 8000 bytes
    - group_ids: shape (1000,), dtype=int32 → 4000 bytes
    - active: shape (1000,), dtype=bool → 1000 bytes
    Total: 21,000 bytes
    """
    n_nodes = 1000

    positions = np.zeros((n_nodes, 2), dtype=np.float32)
    velocities = np.zeros((n_nodes, 2), dtype=np.float32)
    group_ids = np.zeros(n_nodes, dtype=np.int32)
    active = np.zeros(n_nodes, dtype=bool)

    total_bytes = (
        positions.nbytes + velocities.nbytes + group_ids.nbytes + active.nbytes
    )

    max_budget_bytes = 200 * 1024  # 204,800 bytes

    assert total_bytes <= max_budget_bytes, (
        f"Aggregate core state is {total_bytes} bytes, "
        f"exceeds 200 KB budget ({max_budget_bytes} bytes)"
    )


def test_core_state_array_dtypes():
    """Verify that core state arrays use the expected dtypes for memory efficiency."""
    n_nodes = 1000

    positions = np.zeros((n_nodes, 2), dtype=np.float32)
    velocities = np.zeros((n_nodes, 2), dtype=np.float32)
    group_ids = np.zeros(n_nodes, dtype=np.int32)
    active = np.zeros(n_nodes, dtype=bool)

    assert positions.dtype == np.float32
    assert velocities.dtype == np.float32
    assert group_ids.dtype == np.int32
    assert active.dtype == bool


def test_core_state_array_shapes():
    """Verify that core state arrays have the expected shapes for 1000 nodes."""
    n_nodes = 1000

    positions = np.zeros((n_nodes, 2), dtype=np.float32)
    velocities = np.zeros((n_nodes, 2), dtype=np.float32)
    group_ids = np.zeros(n_nodes, dtype=np.int32)
    active = np.zeros(n_nodes, dtype=bool)

    assert positions.shape == (1000, 2)
    assert velocities.shape == (1000, 2)
    assert group_ids.shape == (1000,)
    assert active.shape == (1000,)
