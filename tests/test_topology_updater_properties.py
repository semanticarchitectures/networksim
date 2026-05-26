"""Property-based tests for TopologyUpdater.

Feature: manet-simulation-engine
- Property 20: Topology delta correctness
- Property 21: Initial topology correctness
"""

from hypothesis import given, settings, assume
from hypothesis import strategies as st

import numpy as np

from manet_sim.core.config import NetworkConfig
from manet_sim.topology.topology_updater import TopologyUpdater, TopologyDelta
from manet_sim.topology.link_manager import LinkState


# --- Strategies ---


@st.composite
def network_config(draw):
    """Generate a valid NetworkConfig.

    Produces:
        - radio_range_miles: positive float (0.5 to 5.0)
        - hysteresis_margin_pct: float (0.05 to 0.30)
    """
    radio_range = draw(
        st.floats(min_value=0.5, max_value=5.0, allow_nan=False, allow_infinity=False)
    )
    hysteresis_pct = draw(
        st.floats(min_value=0.05, max_value=0.30, allow_nan=False, allow_infinity=False)
    )
    return NetworkConfig(
        radio_range_miles=radio_range,
        hysteresis_margin_pct=hysteresis_pct,
    )


@st.composite
def positions_array(draw, n_nodes, area_width=50.0, area_height=50.0):
    """Generate a positions array of shape (n_nodes, 2) within area bounds."""
    positions = draw(
        st.lists(
            st.tuples(
                st.floats(min_value=0.0, max_value=area_width, allow_nan=False, allow_infinity=False),
                st.floats(min_value=0.0, max_value=area_height, allow_nan=False, allow_infinity=False),
            ),
            min_size=n_nodes,
            max_size=n_nodes,
        )
    )
    return np.array(positions, dtype=np.float64)


@st.composite
def small_node_count(draw):
    """Generate a small node count suitable for property testing."""
    return draw(st.integers(min_value=2, max_value=15))


# --- Helper Functions ---


def compute_brute_force_pairs_below_threshold(positions: np.ndarray, threshold: float) -> set:
    """Compute all pairs whose distance is strictly below the threshold."""
    n = len(positions)
    pairs = set()
    for i in range(n):
        for j in range(i + 1, n):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < threshold:
                pairs.add((i, j))
    return pairs


def get_graph_edges(graph) -> set:
    """Get all edges from a NetworkX graph as a set of (min, max) tuples."""
    edges = set()
    for u, v in graph.edges():
        edges.add((min(u, v), max(u, v)))
    return edges


# --- Property Tests ---


@given(data=st.data(), config=network_config(), n_nodes=small_node_count())
@settings(max_examples=100, deadline=None)
def test_topology_delta_correctness(data, config, n_nodes):
    """
    Property 20: Topology delta correctness.

    For any two consecutive topology states (before and after a position update),
    the computed TopologyDelta SHALL contain exactly the set of newly formed links
    and exactly the set of broken links, such that applying the delta to the old
    state produces the new state.

    **Validates: Requirements 12.4, 12.5**
    """
    area_width = 50.0
    area_height = 50.0

    # Generate initial positions
    initial_positions = data.draw(
        positions_array(n_nodes, area_width, area_height),
        label="initial_positions",
    )

    # Create and initialize the updater
    updater = TopologyUpdater(config)
    init_delta = updater.initialize(initial_positions, area_width, area_height)

    # Record the state after initialization
    old_edges = get_graph_edges(updater.get_graph())

    # Generate new positions (move nodes)
    new_positions = data.draw(
        positions_array(n_nodes, area_width, area_height),
        label="new_positions",
    )

    # Perform the update
    delta = updater.update(new_positions, t=1.0)

    # Record the state after update
    new_edges = get_graph_edges(updater.get_graph())

    # Compute expected changes by comparing old and new edge sets
    # New links in delta should be exactly the edges in new_edges but not in old_edges
    delta_new_links = set()
    for node_a, node_b, dist in delta.new_links:
        delta_new_links.add((min(node_a, node_b), max(node_a, node_b)))

    # Broken links in delta should be exactly the edges in old_edges but not in new_edges
    delta_broken_links = set()
    for node_a, node_b in delta.broken_links:
        delta_broken_links.add((min(node_a, node_b), max(node_a, node_b)))

    # Verify: applying delta to old state produces new state
    # old_edges - broken_links + new_links == new_edges
    reconstructed = (old_edges - delta_broken_links) | delta_new_links

    assert reconstructed == new_edges, (
        f"Applying delta to old state does not produce new state.\n"
        f"Old edges: {sorted(old_edges)}\n"
        f"Delta new_links: {sorted(delta_new_links)}\n"
        f"Delta broken_links: {sorted(delta_broken_links)}\n"
        f"Reconstructed: {sorted(reconstructed)}\n"
        f"Actual new edges: {sorted(new_edges)}\n"
        f"Missing from reconstructed: {sorted(new_edges - reconstructed)}\n"
        f"Extra in reconstructed: {sorted(reconstructed - new_edges)}"
    )

    # Also verify that new_links and broken_links are disjoint
    assert delta_new_links.isdisjoint(delta_broken_links), (
        f"Delta new_links and broken_links should be disjoint.\n"
        f"Overlap: {sorted(delta_new_links & delta_broken_links)}"
    )

    # Verify that broken_links were actually in old_edges
    assert delta_broken_links.issubset(old_edges), (
        f"Broken links should be a subset of old edges.\n"
        f"Not in old edges: {sorted(delta_broken_links - old_edges)}"
    )

    # Verify that new_links were not in old_edges
    assert delta_new_links.isdisjoint(old_edges), (
        f"New links should not have been in old edges.\n"
        f"Already in old edges: {sorted(delta_new_links & old_edges)}"
    )


@given(data=st.data(), config=network_config(), n_nodes=small_node_count())
@settings(max_examples=100, deadline=None)
def test_initial_topology_correctness(data, config, n_nodes):
    """
    Property 21: Initial topology correctness.

    For any set of initial node positions, the initial topology SHALL contain
    edges for all and only those node pairs whose distance is below the inner
    threshold.

    **Validates: Requirements 12.4, 12.5**
    """
    area_width = 50.0
    area_height = 50.0

    # Generate positions
    positions = data.draw(
        positions_array(n_nodes, area_width, area_height),
        label="positions",
    )

    # Create and initialize the updater
    updater = TopologyUpdater(config)
    init_delta = updater.initialize(positions, area_width, area_height)

    # Get the edges in the graph after initialization
    graph_edges = get_graph_edges(updater.get_graph())

    # Compute expected edges: all pairs with distance < inner_threshold
    inner_threshold = config.radio_range_miles - (
        config.radio_range_miles * config.hysteresis_margin_pct
    )
    expected_edges = compute_brute_force_pairs_below_threshold(positions, inner_threshold)

    # The graph should contain exactly the expected edges
    assert graph_edges == expected_edges, (
        f"Initial topology does not match expected edges.\n"
        f"Inner threshold: {inner_threshold:.6f}\n"
        f"Expected edges: {sorted(expected_edges)}\n"
        f"Actual edges: {sorted(graph_edges)}\n"
        f"Missing (should be in graph): {sorted(expected_edges - graph_edges)}\n"
        f"Extra (should not be in graph): {sorted(graph_edges - expected_edges)}"
    )

    # Also verify the delta reports the same new links
    delta_new_links = set()
    for node_a, node_b, dist in init_delta.new_links:
        delta_new_links.add((min(node_a, node_b), max(node_a, node_b)))

    assert delta_new_links == expected_edges, (
        f"Initial delta new_links do not match expected edges.\n"
        f"Expected: {sorted(expected_edges)}\n"
        f"Delta new_links: {sorted(delta_new_links)}\n"
        f"Missing from delta: {sorted(expected_edges - delta_new_links)}\n"
        f"Extra in delta: {sorted(delta_new_links - expected_edges)}"
    )

    # Verify no broken links in initial delta
    assert len(init_delta.broken_links) == 0, (
        f"Initial delta should have no broken links, but has: {init_delta.broken_links}"
    )
