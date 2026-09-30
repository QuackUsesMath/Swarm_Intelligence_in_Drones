"""
Pheromone grid module for decentralized stigmergy-based coordination.
Drones deposit pheromones on traversed cells, which evaporate over time,
providing repulsion against redundant exploration.
"""

from typing import Dict, List, Tuple
import numpy as np


class PheromoneMap:
    """Decentralized pheromone layer maintained individually by each drone."""

    def __init__(self, width: int, height: int, evap_rate: float, deposit_val: float = 1.0):
        self.width = width
        self.height = height
        self.evap_rate = max(0.001, min(0.99, evap_rate))
        self.deposit_val = deposit_val
        self.grid = np.zeros((height, width), dtype=np.float32)

    def deposit(self, x: int, y: int, amount: float = None) -> None:
        """Deposit pheromone at cell (x, y)."""
        if 0 <= x < self.width and 0 <= y < self.height:
            val = self.deposit_val if amount is None else amount
            self.grid[y, x] = min(10.0, self.grid[y, x] + val)

    def step_evaporation(self) -> None:
        """Evaporate pheromone across the local belief grid."""
        self.grid *= (1.0 - self.evap_rate)
        # Numerical cleanup of near-zero values
        self.grid[self.grid < 1e-4] = 0.0

    def get_intensity(self, x: int, y: int) -> float:
        """Get pheromone intensity at (x, y)."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return float(self.grid[y, x])
        return 0.0

    def get_average_neighborhood_intensity(self, x: int, y: int, radius: int = 1) -> float:
        """Compute average pheromone intensity around (x, y) within radius."""
        x_min = max(0, x - radius)
        x_max = min(self.width, x + radius + 1)
        y_min = max(0, y - radius)
        y_max = min(self.height, y + radius + 1)
        sub = self.grid[y_min:y_max, x_min:x_max]
        if sub.size == 0:
            return 0.0
        return float(np.mean(sub))

    def extract_sparse_deltas(self, threshold: float = 0.05) -> Dict[Tuple[int, int], float]:
        """Extract non-zero pheromone entries for low-bandwidth gossip transmission."""
        nz_y, nz_x = np.nonzero(self.grid > threshold)
        deltas: Dict[Tuple[int, int], float] = {}
        for x, y in zip(nz_x, nz_y):
            deltas[(int(x), int(y))] = float(self.grid[y, x])
        return deltas

    def merge_remote_deltas(self, remote_deltas: Dict[Tuple[int, int], float]) -> None:
        """
        Merge pheromone deltas from a communicating neighbor via gossip.
        Takes max of local and remote intensity to retain freshest stigmergic trails.
        """
        for (x, y), val in remote_deltas.items():
            if 0 <= x < self.width and 0 <= y < self.height:
                if val > self.grid[y, x]:
                    self.grid[y, x] = min(10.0, val)

    def merge_full_grid(self, remote_grid: np.ndarray) -> None:
        """Merge full remote pheromone grid (element-wise maximum)."""
        np.maximum(self.grid, remote_grid, out=self.grid)
