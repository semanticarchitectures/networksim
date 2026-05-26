"""Unit tests for GridSpatialIndex."""

import numpy as np
import pytest

from manet_sim.mobility.spatial_index import GridSpatialIndex


class TestGridSpatialIndexInit:
    """Tests for GridSpatialIndex initialization."""

    def test_basic_initialization(self):
        idx = GridSpatialIndex(width=100.0, height=100.0, cell_size=1.0)
        assert idx.width == 100.0
        assert idx.height == 100.0
        assert idx.cell_size == 1.0
        assert idx.cols == 100
        assert idx.rows == 100

    def test_non_integer_cell_count(self):
        """When area isn't evenly divisible by cell_size, cols/rows round up."""
        idx = GridSpatialIndex(width=10.0, height=7.0, cell_size=3.0)
        assert idx.cols == 4  # ceil(10/3) = 4
        assert idx.rows == 3  # ceil(7/3) = 3

    def test_invalid_cell_size_raises(self):
        with pytest.raises(ValueError, match="cell_size must be positive"):
            GridSpatialIndex(width=100.0, height=100.0, cell_size=0.0)
        with pytest.raises(ValueError, match="cell_size must be positive"):
            GridSpatialIndex(width=100.0, height=100.0, cell_size=-1.0)

    def test_invalid_dimensions_raises(self):
        with pytest.raises(ValueError, match="width and height must be positive"):
            GridSpatialIndex(width=0.0, height=100.0, cell_size=1.0)
        with pytest.raises(ValueError, match="width and height must be positive"):
            GridSpatialIndex(width=100.0, height=-5.0, cell_size=1.0)


class TestRebuild:
    """Tests for GridSpatialIndex.rebuild()."""

    def test_rebuild_with_empty_array(self):
        idx = GridSpatialIndex(width=100.0, height=100.0, cell_size=1.0)
        positions = np.empty((0, 2), dtype=np.float32)
        idx.rebuild(positions)
        assert idx._n_nodes == 0

    def test_rebuild_with_single_node(self):
        idx = GridSpatialIndex(width=100.0, height=100.0, cell_size=1.0)
        positions = np.array([[50.0, 50.0]], dtype=np.float32)
        idx.rebuild(positions)
        assert idx._n_nodes == 1

    def test_rebuild_clears_previous_data(self):
        idx = GridSpatialIndex(width=100.0, height=100.0, cell_size=1.0)
        positions1 = np.array([[10.0, 10.0], [20.0, 20.0]], dtype=np.float32)
        idx.rebuild(positions1)
        assert idx._n_nodes == 2

        positions2 = np.array([[50.0, 50.0]], dtype=np.float32)
        idx.rebuild(positions2)
        assert idx._n_nodes == 1

    def test_rebuild_with_many_nodes(self):
        idx = GridSpatialIndex(width=100.0, height=100.0, cell_size=1.0)
        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 100, size=(1000, 2)).astype(np.float32)
        idx.rebuild(positions)
        assert idx._n_nodes == 1000


class TestQueryRadius:
    """Tests for GridSpatialIndex.query_radius()."""

    def test_empty_index_returns_empty(self):
        idx = GridSpatialIndex(width=100.0, height=100.0, cell_size=1.0)
        result = idx.query_radius(50.0, 50.0, 5.0)
        assert result == []

    def test_no_nodes_in_range_returns_empty(self):
        idx = GridSpatialIndex(width=100.0, height=100.0, cell_size=1.0)
        positions = np.array([[10.0, 10.0]], dtype=np.float32)
        idx.rebuild(positions)
        # Query far from the node
        result = idx.query_radius(90.0, 90.0, 1.0)
        assert result == []

    def test_single_node_within_radius(self):
        idx = GridSpatialIndex(width=100.0, height=100.0, cell_size=1.0)
        positions = np.array([[50.0, 50.0]], dtype=np.float32)
        idx.rebuild(positions)
        result = idx.query_radius(50.5, 50.5, 1.0)
        assert result == [0]

    def test_node_exactly_at_radius_boundary(self):
        """Node exactly at radius distance should be included (<=)."""
        idx = GridSpatialIndex(width=100.0, height=100.0, cell_size=1.0)
        positions = np.array([[50.0, 50.0]], dtype=np.float32)
        idx.rebuild(positions)
        # Distance from (51.0, 50.0) to (50.0, 50.0) is exactly 1.0
        result = idx.query_radius(51.0, 50.0, 1.0)
        assert 0 in result

    def test_node_just_outside_radius(self):
        """Node just beyond radius should not be included."""
        idx = GridSpatialIndex(width=100.0, height=100.0, cell_size=1.0)
        positions = np.array([[50.0, 50.0]], dtype=np.float32)
        idx.rebuild(positions)
        # Distance from (51.1, 50.0) to (50.0, 50.0) is 1.1 > 1.0
        result = idx.query_radius(51.1, 50.0, 1.0)
        assert result == []

    def test_multiple_nodes_some_in_range(self):
        idx = GridSpatialIndex(width=100.0, height=100.0, cell_size=1.0)
        positions = np.array([
            [50.0, 50.0],  # 0: in range
            [50.5, 50.5],  # 1: in range
            [55.0, 55.0],  # 2: out of range
            [49.5, 50.0],  # 3: in range
        ], dtype=np.float32)
        idx.rebuild(positions)
        result = idx.query_radius(50.0, 50.0, 1.0)
        assert set(result) == {0, 1, 3}

    def test_query_at_corner(self):
        """Nodes near the corner of the simulation area."""
        idx = GridSpatialIndex(width=100.0, height=100.0, cell_size=1.0)
        positions = np.array([
            [0.5, 0.5],   # near origin
            [1.0, 0.0],   # near origin
            [99.0, 99.0], # far corner
        ], dtype=np.float32)
        idx.rebuild(positions)
        result = idx.query_radius(0.0, 0.0, 1.5)
        assert 0 in result
        assert 1 in result
        assert 2 not in result

    def test_query_includes_self_position(self):
        """query_radius does not exclude any node by index — caller filters if needed."""
        idx = GridSpatialIndex(width=100.0, height=100.0, cell_size=1.0)
        positions = np.array([[50.0, 50.0]], dtype=np.float32)
        idx.rebuild(positions)
        # Querying at the exact node position should return that node
        result = idx.query_radius(50.0, 50.0, 0.1)
        assert 0 in result

    def test_zero_false_negatives(self):
        """Every node within radius must be returned (Requirement 10.1)."""
        idx = GridSpatialIndex(width=100.0, height=100.0, cell_size=1.0)
        rng = np.random.default_rng(123)
        positions = rng.uniform(0, 100, size=(100, 2)).astype(np.float32)
        idx.rebuild(positions)

        query_x, query_y = 50.0, 50.0
        radius = 5.0

        result_set = set(idx.query_radius(query_x, query_y, radius))

        # Brute-force check: every node within radius must be in result
        for i in range(len(positions)):
            dx = positions[i, 0] - query_x
            dy = positions[i, 1] - query_y
            dist = np.sqrt(dx * dx + dy * dy)
            if dist <= radius:
                assert i in result_set, (
                    f"Node {i} at distance {dist:.4f} should be in result"
                )


class TestQueryPairs:
    """Tests for GridSpatialIndex.query_pairs()."""

    def test_empty_index_returns_empty_set(self):
        idx = GridSpatialIndex(width=100.0, height=100.0, cell_size=1.0)
        result = idx.query_pairs(1.0)
        assert result == set()

    def test_single_node_returns_empty_set(self):
        idx = GridSpatialIndex(width=100.0, height=100.0, cell_size=1.0)
        positions = np.array([[50.0, 50.0]], dtype=np.float32)
        idx.rebuild(positions)
        result = idx.query_pairs(1.0)
        assert result == set()

    def test_two_nodes_within_radius(self):
        idx = GridSpatialIndex(width=100.0, height=100.0, cell_size=1.0)
        positions = np.array([
            [50.0, 50.0],
            [50.5, 50.0],  # distance = 0.5 < 1.0
        ], dtype=np.float32)
        idx.rebuild(positions)
        result = idx.query_pairs(1.0)
        assert result == {(0, 1)}

    def test_two_nodes_outside_radius(self):
        idx = GridSpatialIndex(width=100.0, height=100.0, cell_size=1.0)
        positions = np.array([
            [10.0, 10.0],
            [20.0, 20.0],  # distance = ~14.14 > 1.0
        ], dtype=np.float32)
        idx.rebuild(positions)
        result = idx.query_pairs(1.0)
        assert result == set()

    def test_pair_ordering_is_canonical(self):
        """Pairs are always (min_id, max_id)."""
        idx = GridSpatialIndex(width=100.0, height=100.0, cell_size=1.0)
        positions = np.array([
            [50.0, 50.0],
            [50.3, 50.0],
        ], dtype=np.float32)
        idx.rebuild(positions)
        result = idx.query_pairs(1.0)
        for pair in result:
            assert pair[0] < pair[1]

    def test_multiple_pairs(self):
        idx = GridSpatialIndex(width=100.0, height=100.0, cell_size=1.0)
        positions = np.array([
            [50.0, 50.0],   # 0
            [50.5, 50.0],   # 1: within 1.0 of 0
            [50.0, 50.5],   # 2: within 1.0 of 0 and 1
            [60.0, 60.0],   # 3: far from all others
        ], dtype=np.float32)
        idx.rebuild(positions)
        result = idx.query_pairs(1.0)
        # 0-1: dist=0.5, 0-2: dist=0.5, 1-2: dist=sqrt(0.25+0.25)≈0.707
        assert (0, 1) in result
        assert (0, 2) in result
        assert (1, 2) in result
        assert (0, 3) not in result
        assert (1, 3) not in result
        assert (2, 3) not in result

    def test_nodes_at_exact_radius(self):
        """Nodes exactly at radius distance should be included."""
        idx = GridSpatialIndex(width=100.0, height=100.0, cell_size=1.0)
        positions = np.array([
            [50.0, 50.0],
            [51.0, 50.0],  # distance = exactly 1.0
        ], dtype=np.float32)
        idx.rebuild(positions)
        result = idx.query_pairs(1.0)
        assert (0, 1) in result

    def test_no_duplicate_pairs(self):
        """Each pair should appear exactly once."""
        idx = GridSpatialIndex(width=100.0, height=100.0, cell_size=5.0)
        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 100, size=(50, 2)).astype(np.float32)
        idx.rebuild(positions)
        result = idx.query_pairs(5.0)
        # Since it's a set, duplicates are impossible, but verify canonical form
        for pair in result:
            assert pair[0] < pair[1]

    def test_zero_false_negatives_pairs(self):
        """Every pair within radius must be found (Requirement 10.1)."""
        idx = GridSpatialIndex(width=100.0, height=100.0, cell_size=1.0)
        rng = np.random.default_rng(456)
        positions = rng.uniform(0, 100, size=(100, 2)).astype(np.float32)
        idx.rebuild(positions)
        radius = 1.0

        result = idx.query_pairs(radius)

        # Brute-force: find all true pairs
        n = len(positions)
        for i in range(n):
            for j in range(i + 1, n):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist <= radius:
                    assert (i, j) in result, (
                        f"Pair ({i}, {j}) at distance {dist:.4f} should be in result"
                    )

    def test_nodes_across_cell_boundaries(self):
        """Nodes in adjacent cells within radius should be found."""
        idx = GridSpatialIndex(width=100.0, height=100.0, cell_size=1.0)
        # Place nodes just on either side of a cell boundary
        positions = np.array([
            [0.99, 50.0],  # cell (0, 50)
            [1.01, 50.0],  # cell (1, 50)
        ], dtype=np.float32)
        idx.rebuild(positions)
        result = idx.query_pairs(0.1)
        # Distance = 0.02, well within 0.1
        assert (0, 1) in result


class TestQueryRadiusLargerThanCellSize:
    """Tests for query_radius when radius > cell_size."""

    def test_radius_larger_than_cell_size(self):
        """query_radius should still work when radius > cell_size."""
        idx = GridSpatialIndex(width=100.0, height=100.0, cell_size=1.0)
        positions = np.array([
            [50.0, 50.0],
            [52.0, 50.0],  # distance = 2.0
            [53.0, 50.0],  # distance = 3.0
            [54.0, 50.0],  # distance = 4.0
        ], dtype=np.float32)
        idx.rebuild(positions)
        result = idx.query_radius(50.0, 50.0, 3.0)
        assert 0 in result  # distance = 0
        assert 1 in result  # distance = 2.0
        assert 2 in result  # distance = 3.0
        assert 3 not in result  # distance = 4.0 > 3.0


class TestEdgeCases:
    """Edge case tests for GridSpatialIndex."""

    def test_nodes_at_origin(self):
        idx = GridSpatialIndex(width=100.0, height=100.0, cell_size=1.0)
        positions = np.array([[0.0, 0.0], [0.1, 0.1]], dtype=np.float32)
        idx.rebuild(positions)
        result = idx.query_radius(0.0, 0.0, 0.5)
        assert set(result) == {0, 1}

    def test_nodes_at_max_boundary(self):
        idx = GridSpatialIndex(width=100.0, height=100.0, cell_size=1.0)
        positions = np.array([[100.0, 100.0], [99.9, 99.9]], dtype=np.float32)
        idx.rebuild(positions)
        result = idx.query_radius(100.0, 100.0, 0.5)
        assert set(result) == {0, 1}

    def test_all_nodes_in_same_cell(self):
        """All nodes clustered in one cell."""
        idx = GridSpatialIndex(width=100.0, height=100.0, cell_size=1.0)
        positions = np.array([
            [50.1, 50.1],
            [50.2, 50.2],
            [50.3, 50.3],
            [50.4, 50.4],
        ], dtype=np.float32)
        idx.rebuild(positions)
        result = idx.query_pairs(1.0)
        # All pairs should be found since max distance is ~0.42
        expected = {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)}
        assert result == expected

    def test_rebuild_after_positions_change(self):
        """Rebuild correctly reflects new positions."""
        idx = GridSpatialIndex(width=100.0, height=100.0, cell_size=1.0)
        positions = np.array([[50.0, 50.0], [50.5, 50.0]], dtype=np.float32)
        idx.rebuild(positions)
        assert (0, 1) in idx.query_pairs(1.0)

        # Move node 1 far away
        positions[1] = [90.0, 90.0]
        idx.rebuild(positions)
        assert (0, 1) not in idx.query_pairs(1.0)
