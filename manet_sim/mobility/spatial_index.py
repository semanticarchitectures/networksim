"""Grid-based spatial index for efficient neighbor discovery.

Provides O(n·k) neighbor queries where k is the average number of nodes
per cell, compared to O(n²) brute-force pairwise distance checks.

The grid cell size is set equal to the radio range so that any pair of
nodes within radio range must reside in the same cell or adjacent cells
(3×3 neighborhood check).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


class GridSpatialIndex:
    """
    Uniform grid spatial index for efficient radius queries.

    Cell size equals the radio range, ensuring that neighbor candidates
    are found by checking only the 3×3 neighborhood of cells around a
    query point. Rebuilt each simulation step from the full positions array.

    Attributes:
        width: Simulation area width in miles.
        height: Simulation area height in miles.
        cell_size: Grid cell size in miles (equal to radio range).
        cols: Number of grid columns.
        rows: Number of grid rows.
    """

    def __init__(self, width: float, height: float, cell_size: float) -> None:
        """
        Initialize the spatial index.

        Args:
            width: Simulation area width in miles.
            height: Simulation area height in miles.
            cell_size: Grid cell size in miles (should equal radio range).
        """
        if cell_size <= 0:
            raise ValueError(f"cell_size must be positive, got {cell_size}")
        if width <= 0 or height <= 0:
            raise ValueError(
                f"width and height must be positive, got width={width}, height={height}"
            )

        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.cols = int(np.ceil(width / cell_size))
        self.rows = int(np.ceil(height / cell_size))

        # Grid storage: (col, row) -> list of node indices
        self._cells: dict[tuple[int, int], list[int]] = defaultdict(list)
        # Positions array reference (set during rebuild)
        self._positions: np.ndarray | None = None
        # Number of nodes currently indexed
        self._n_nodes: int = 0

    def _cell_key(self, x: float, y: float) -> tuple[int, int]:
        """Compute the grid cell key for a given position."""
        col = int(x / self.cell_size)
        row = int(y / self.cell_size)
        # Clamp to valid range
        col = max(0, min(col, self.cols - 1))
        row = max(0, min(row, self.rows - 1))
        return (col, row)

    def rebuild(self, positions: np.ndarray) -> None:
        """
        Rebuild the spatial index from a positions array. O(n) operation.

        Clears all existing data and re-inserts all nodes based on their
        current positions.

        Args:
            positions: NumPy array of shape (N, 2) containing (x, y)
                       coordinates for each node.
        """
        self._cells.clear()
        self._positions = positions
        self._n_nodes = len(positions)

        for i in range(self._n_nodes):
            x = float(positions[i, 0])
            y = float(positions[i, 1])
            key = self._cell_key(x, y)
            self._cells[key].append(i)

    def query_radius(self, x: float, y: float, radius: float) -> list[int]:
        """
        Find all node indices within a given radius of a query point.

        Checks cells in the neighborhood that could contain nodes within
        the specified radius. Performs exact Euclidean distance filtering
        to ensure zero false negatives.

        Args:
            x: Query point x-coordinate in miles.
            y: Query point y-coordinate in miles.
            radius: Search radius in miles.

        Returns:
            List of node indices whose positions are within the radius
            of the query point. Returns empty list if no nodes are in range.
        """
        if self._positions is None or self._n_nodes == 0:
            return []

        radius_sq = radius * radius
        result: list[int] = []

        # Determine the range of cells to check
        min_col = max(0, int((x - radius) / self.cell_size))
        max_col = min(self.cols - 1, int((x + radius) / self.cell_size))
        min_row = max(0, int((y - radius) / self.cell_size))
        max_row = min(self.rows - 1, int((y + radius) / self.cell_size))

        for col in range(min_col, max_col + 1):
            for row in range(min_row, max_row + 1):
                cell_nodes = self._cells.get((col, row))
                if cell_nodes is None:
                    continue
                for node_idx in cell_nodes:
                    nx_ = float(self._positions[node_idx, 0])
                    ny_ = float(self._positions[node_idx, 1])
                    dx = nx_ - x
                    dy = ny_ - y
                    if dx * dx + dy * dy <= radius_sq:
                        result.append(node_idx)

        return result

    def query_pairs(self, radius: float) -> set[tuple[int, int]]:
        """
        Find all pairs of nodes within a given radius of each other.

        Iterates over all occupied cells and checks nodes against nodes
        in the same cell and neighboring cells to find all pairs within
        the specified radius. Each pair is returned as (min_id, max_id)
        to avoid duplicates.

        Args:
            radius: Maximum distance in miles for a pair to be included.

        Returns:
            Set of tuples (node_i, node_j) where node_i < node_j and
            the Euclidean distance between them is <= radius.
            Returns empty set if no pairs are within range.
        """
        if self._positions is None or self._n_nodes == 0:
            return set()

        radius_sq = radius * radius
        pairs: set[tuple[int, int]] = set()

        # For each occupied cell, check pairs within the cell and with
        # neighboring cells (only forward neighbors to avoid duplicates)
        occupied_cells = list(self._cells.keys())

        for cell_key in occupied_cells:
            col, row = cell_key
            cell_nodes = self._cells[cell_key]

            # Check pairs within the same cell
            for i in range(len(cell_nodes)):
                node_i = cell_nodes[i]
                xi = float(self._positions[node_i, 0])
                yi = float(self._positions[node_i, 1])

                # Same-cell pairs (only j > i to avoid duplicates within cell)
                for j in range(i + 1, len(cell_nodes)):
                    node_j = cell_nodes[j]
                    xj = float(self._positions[node_j, 0])
                    yj = float(self._positions[node_j, 1])
                    dx = xi - xj
                    dy = yi - yj
                    if dx * dx + dy * dy <= radius_sq:
                        pair = (min(node_i, node_j), max(node_i, node_j))
                        pairs.add(pair)

                # Check against neighboring cells (only "forward" neighbors
                # to avoid checking each pair twice)
                for dcol, drow in [(1, 0), (1, 1), (0, 1), (-1, 1)]:
                    neighbor_key = (col + dcol, row + drow)
                    neighbor_nodes = self._cells.get(neighbor_key)
                    if neighbor_nodes is None:
                        continue
                    for node_j in neighbor_nodes:
                        xj = float(self._positions[node_j, 0])
                        yj = float(self._positions[node_j, 1])
                        dx = xi - xj
                        dy = yi - yj
                        if dx * dx + dy * dy <= radius_sq:
                            pair = (min(node_i, node_j), max(node_i, node_j))
                            pairs.add(pair)

        return pairs
