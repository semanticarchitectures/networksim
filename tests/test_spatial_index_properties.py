"""Property-based tests for GridSpatialIndex.

Feature: manet-simulation-engine, Property 16: Spatial index zero false negatives
"""

import numpy as np
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from manet_sim.mobility.spatial_index import GridSpatialIndex


# --- Strategies ---


@st.composite
def spatial_index_scenario(draw):
    """Generate a spatial index scenario with positions, a query point, and a radius.

    Produces:
        - width, height: simulation area dimensions (10.0 to 200.0 miles)
        - cell_size: grid cell size (0.5 to 20.0 miles)
        - positions: (N, 2) float32 array with N in [1, 200]
        - query_x, query_y: query point within the area
        - radius: search radius (0.1 to 30.0 miles)
    """
    # Area dimensions
    width = draw(st.floats(min_value=10.0, max_value=200.0))
    height = draw(st.floats(min_value=10.0, max_value=200.0))

    # Cell size — between 0.5 and min(width, height) / 2
    max_cell = min(width, height) / 2.0
    cell_size = draw(st.floats(min_value=0.5, max_value=min(max_cell, 20.0)))

    # Number of nodes
    n_nodes = draw(st.integers(min_value=1, max_value=200))

    # Generate positions within the area bounds
    positions_list = draw(
        st.lists(
            st.tuples(
                st.floats(min_value=0.0, max_value=width, allow_nan=False, allow_infinity=False),
                st.floats(min_value=0.0, max_value=height, allow_nan=False, allow_infinity=False),
            ),
            min_size=n_nodes,
            max_size=n_nodes,
        )
    )
    positions = np.array(positions_list, dtype=np.float32)

    # Query point — can be anywhere in the area
    query_x = draw(st.floats(min_value=0.0, max_value=width, allow_nan=False, allow_infinity=False))
    query_y = draw(st.floats(min_value=0.0, max_value=height, allow_nan=False, allow_infinity=False))

    # Radius
    radius = draw(st.floats(min_value=0.1, max_value=30.0, allow_nan=False, allow_infinity=False))

    return width, height, cell_size, positions, query_x, query_y, radius


# --- Property Tests ---


@given(scenario=spatial_index_scenario())
@settings(max_examples=100, deadline=None)
def test_spatial_index_zero_false_negatives(scenario):
    """
    Property 16: Spatial index zero false negatives.

    For any set of N node positions and any query point with radius R, the
    spatial index radius query SHALL return a superset of (or exactly) the set
    of node IDs whose Euclidean distance from the query point is <= R. There
    SHALL be zero false negatives.

    **Validates: Requirements 10.1**
    """
    width, height, cell_size, positions, query_x, query_y, radius = scenario

    # Build the spatial index
    idx = GridSpatialIndex(width=width, height=height, cell_size=cell_size)
    idx.rebuild(positions)

    # Query the spatial index
    result_set = set(idx.query_radius(query_x, query_y, radius))

    # Brute-force: compute the true set of neighbors within radius
    n_nodes = len(positions)
    for i in range(n_nodes):
        dx = float(positions[i, 0]) - query_x
        dy = float(positions[i, 1]) - query_y
        dist_sq = dx * dx + dy * dy
        # Use the same comparison as the spatial index (distance squared <= radius squared)
        if dist_sq <= radius * radius:
            assert i in result_set, (
                f"False negative: node {i} at position ({positions[i, 0]:.6f}, "
                f"{positions[i, 1]:.6f}) has distance {np.sqrt(dist_sq):.6f} from "
                f"query point ({query_x:.6f}, {query_y:.6f}) which is <= radius "
                f"{radius:.6f}, but was not returned by query_radius. "
                f"cell_size={cell_size:.2f}, width={width:.2f}, height={height:.2f}"
            )
