"""Property-based tests for TopologyMetricsCollector.

Feature: manet-simulation-engine
Property 27: Topology metrics correctness
"""

from hypothesis import given, settings, assume
from hypothesis import strategies as st
import networkx as nx
import numpy as np

from manet_sim.network.metrics import TopologyMetricsCollector


# --- Strategies ---


@st.composite
def random_graph(draw):
    """Generate a random NetworkX graph representing a MANET topology.

    Generates graphs with 0 to 50 nodes and random edges between them,
    simulating various MANET topology states (sparse, dense, disconnected).
    """
    n_nodes = draw(st.integers(min_value=0, max_value=50))
    if n_nodes == 0:
        return nx.Graph()

    # Choose edge probability to get a mix of sparse and dense graphs
    edge_prob = draw(st.floats(min_value=0.0, max_value=1.0))

    # Generate an Erdos-Renyi random graph
    G = nx.erdos_renyi_graph(n_nodes, edge_prob, seed=draw(st.integers(min_value=0, max_value=2**31)))
    return G


@st.composite
def graph_with_isolated_nodes(draw):
    """Generate a graph that may have isolated nodes (no edges).

    This tests the edge case where some nodes have degree 0.
    """
    n_nodes = draw(st.integers(min_value=1, max_value=30))
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))

    # Add a subset of possible edges
    if n_nodes >= 2:
        n_edges = draw(st.integers(min_value=0, max_value=min(n_nodes * (n_nodes - 1) // 2, 100)))
        possible_edges = list(nx.non_edges(G))
        if possible_edges and n_edges > 0:
            n_edges = min(n_edges, len(possible_edges))
            # Draw indices for edges to add
            indices = draw(st.lists(
                st.integers(min_value=0, max_value=len(possible_edges) - 1),
                min_size=n_edges,
                max_size=n_edges,
                unique=True,
            ))
            edges_to_add = [possible_edges[i] for i in indices]
            G.add_edges_from(edges_to_add)

    return G


# --- Property 27: Topology metrics correctness ---


@given(G=random_graph())
@settings(max_examples=100, deadline=None)
def test_topology_metrics_correctness(G):
    """
    Property 27: Topology metrics correctness.

    For any NetworkX graph representing a MANET topology, the computed
    topology metrics (node_count, edge_count, average_degree, max_degree,
    num_connected_components, largest_component_size, avg_clustering_coefficient)
    SHALL match the values computed by the corresponding NetworkX library
    functions on the same graph.

    **Validates: Requirements 17.1**
    """
    # Create a TopologyMetricsCollector and inject the graph
    collector = TopologyMetricsCollector(snapshot_interval=300.0, output_dir="/tmp/test_metrics")
    collector._graph = G.copy()

    # Compute snapshot via the collector
    snapshot = collector._compute_snapshot(timestamp=100.0)

    # Compute expected values directly from NetworkX
    expected_node_count = G.number_of_nodes()
    expected_edge_count = G.number_of_edges()

    if expected_node_count == 0:
        # Zero-node graph: all metrics should be zero
        assert snapshot.node_count == 0
        assert snapshot.edge_count == 0
        assert snapshot.avg_degree == 0.0
        assert snapshot.max_degree == 0
        assert snapshot.num_connected_components == 0
        assert snapshot.largest_component_size == 0
        assert snapshot.avg_clustering_coefficient == 0.0
        return

    # Degree metrics from NetworkX
    degrees = [d for _, d in G.degree()]
    expected_avg_degree = float(np.mean(degrees))
    expected_max_degree = max(degrees)

    # Connected components from NetworkX
    components = list(nx.connected_components(G))
    expected_num_components = len(components)
    expected_largest_component = max(len(c) for c in components)

    # Clustering coefficient from NetworkX
    expected_avg_clustering = float(nx.average_clustering(G))

    # Assert all metrics match
    assert snapshot.node_count == expected_node_count, (
        f"node_count mismatch: got {snapshot.node_count}, expected {expected_node_count}"
    )
    assert snapshot.edge_count == expected_edge_count, (
        f"edge_count mismatch: got {snapshot.edge_count}, expected {expected_edge_count}"
    )
    assert abs(snapshot.avg_degree - expected_avg_degree) < 1e-10, (
        f"avg_degree mismatch: got {snapshot.avg_degree}, expected {expected_avg_degree}"
    )
    assert snapshot.max_degree == expected_max_degree, (
        f"max_degree mismatch: got {snapshot.max_degree}, expected {expected_max_degree}"
    )
    assert snapshot.num_connected_components == expected_num_components, (
        f"num_connected_components mismatch: got {snapshot.num_connected_components}, "
        f"expected {expected_num_components}"
    )
    assert snapshot.largest_component_size == expected_largest_component, (
        f"largest_component_size mismatch: got {snapshot.largest_component_size}, "
        f"expected {expected_largest_component}"
    )
    assert abs(snapshot.avg_clustering_coefficient - expected_avg_clustering) < 1e-10, (
        f"avg_clustering_coefficient mismatch: got {snapshot.avg_clustering_coefficient}, "
        f"expected {expected_avg_clustering}"
    )


@given(G=graph_with_isolated_nodes())
@settings(max_examples=100, deadline=None)
def test_topology_metrics_with_isolated_nodes(G):
    """
    Property 27 (extended): Metrics correctness with isolated nodes.

    Graphs with isolated nodes (degree 0) should still produce correct
    metrics, particularly for average degree and connected components.

    **Validates: Requirements 17.1**
    """
    collector = TopologyMetricsCollector(snapshot_interval=300.0, output_dir="/tmp/test_metrics")
    collector._graph = G.copy()

    snapshot = collector._compute_snapshot(timestamp=200.0)

    # Verify against NetworkX
    expected_node_count = G.number_of_nodes()
    expected_edge_count = G.number_of_edges()

    degrees = [d for _, d in G.degree()]
    expected_avg_degree = float(np.mean(degrees))
    expected_max_degree = max(degrees)

    components = list(nx.connected_components(G))
    expected_num_components = len(components)
    expected_largest_component = max(len(c) for c in components)

    expected_avg_clustering = float(nx.average_clustering(G))

    assert snapshot.node_count == expected_node_count
    assert snapshot.edge_count == expected_edge_count
    assert abs(snapshot.avg_degree - expected_avg_degree) < 1e-10
    assert snapshot.max_degree == expected_max_degree
    assert snapshot.num_connected_components == expected_num_components
    assert snapshot.largest_component_size == expected_largest_component
    assert abs(snapshot.avg_clustering_coefficient - expected_avg_clustering) < 1e-10
