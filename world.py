"""
World environment module representing the 2D disaster area ground truth.
Maintains obstacles, survivors, dynamic collapses, and sensing queries.
"""

from typing import Dict, List, Set, Tuple
import numpy as np

from config import SwarmConfig

# Ground truth cell constants
CELL_FREE = 0
CELL_OBSTACLE = 1
CELL_SURVIVOR = 2


class DisasterWorld:
    """Ground truth environment for search and rescue drone swarm simulation."""

    def __init__(self, config: SwarmConfig):
        self.config = config
        self.width = config.grid_width
        self.height = config.grid_height
        self.rng = np.random.RandomState(config.seed)

        # Ground truth grid: 0 = free, 1 = obstacle, 2 = survivor
        self.grid = np.zeros((self.height, self.width), dtype=np.int8)

        # Dynamic collapse schedule: tick -> list of (x, y)
        self.dynamic_schedule: Dict[int, List[Tuple[int, int]]] = {
            tick: list(coords) for tick, coords in config.dynamic_obstacles.items()
        }

        # Sets for tracking ground truth entities
        self.survivors: Set[Tuple[int, int]] = set()
        self.obstacles: Set[Tuple[int, int]] = set()
        self.discovered_survivors: Set[Tuple[int, int]] = set()

        self._initialize_world()

    def _initialize_world(self) -> None:
        """Procedurally generate static obstacles (building clusters) and hidden survivors."""
        # 1. Base station clearance: clear 5x5 area around base station
        bx, by = self.config.base_station
        clear_zone = {
            (bx + dx, by + dy)
            for dx in range(-2, 3)
            for dy in range(-2, 3)
            if 0 <= bx + dx < self.width and 0 <= by + dy < self.height
        }

        # 2. Generate building clusters to simulate urban disaster zone
        num_blocks = int((self.width * self.height * self.config.obstacle_density) / 8)
        for _ in range(num_blocks):
            cx = self.rng.randint(0, self.width)
            cy = self.rng.randint(0, self.height)
            bw = self.rng.randint(2, 5)
            bh = self.rng.randint(2, 5)
            for ox in range(cx, min(self.width, cx + bw)):
                for oy in range(cy, min(self.height, cy + bh)):
                    if (ox, oy) not in clear_zone:
                        self.grid[oy, ox] = CELL_OBSTACLE
                        self.obstacles.add((ox, oy))

        # Additional random scattered obstacles
        scatter_count = int(self.width * self.height * self.config.obstacle_density * 0.2)
        for _ in range(scatter_count):
            ox = self.rng.randint(0, self.width)
            oy = self.rng.randint(0, self.height)
            if (ox, oy) not in clear_zone:
                self.grid[oy, ox] = CELL_OBSTACLE
                self.obstacles.add((ox, oy))

        # 3. Place K hidden survivors in valid free cells
        free_candidates = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if self.grid[y, x] == CELL_FREE and (x, y) not in clear_zone
        ]

        if len(free_candidates) < self.config.num_survivors:
            raise ValueError("Too many obstacles: insufficient free cells for survivors.")

        chosen_indices = self.rng.choice(
            len(free_candidates), size=self.config.num_survivors, replace=False
        )
        for idx in chosen_indices:
            sx, sy = free_candidates[idx]
            self.grid[sy, sx] = CELL_SURVIVOR
            self.survivors.add((sx, sy))

    def trigger_dynamic_perturbations(self, current_tick: int) -> List[Tuple[int, int]]:
        """Trigger dynamic building collapses scheduled for the current tick."""
        newly_collapsed: List[Tuple[int, int]] = []
        if current_tick in self.dynamic_schedule:
            for x, y in self.dynamic_schedule[current_tick]:
                if 0 <= x < self.width and 0 <= y < self.height:
                    # Do not place obstacle directly on base station
                    if (x, y) == self.config.base_station:
                        continue
                    # If survivor was here, survivor is buried / moved to adjacent free cell if possible
                    if (x, y) in self.survivors:
                        self.survivors.remove((x, y))
                    self.grid[y, x] = CELL_OBSTACLE
                    self.obstacles.add((x, y))
                    newly_collapsed.append((x, y))
        return newly_collapsed

    def sense_environment(
        self, drone_pos: Tuple[int, int], radius: int
    ) -> List[Tuple[int, int, int]]:
        """
        Sense local area within sensing radius r.
        Returns list of (x, y, true_cell_state).
        """
        px, py = drone_pos
        observations: List[Tuple[int, int, int]] = []
        r2 = radius * radius

        for dy in range(-radius, radius + 1):
            ny = py + dy
            if not (0 <= ny < self.height):
                continue
            for dx in range(-radius, radius + 1):
                nx = px + dx
                if not (0 <= nx < self.width):
                    continue
                # Euclidean distance check within radius
                if dx * dx + dy * dy <= r2:
                    cell_val = int(self.grid[ny, nx])
                    observations.append((nx, ny, cell_val))
                    if cell_val == CELL_SURVIVOR:
                        self.discovered_survivors.add((nx, ny))

        return observations

    def is_valid_coordinate(self, x: int, y: int) -> bool:
        """Check if (x, y) is within world boundaries."""
        return 0 <= x < self.width and 0 <= y < self.height

    def is_obstacle(self, x: int, y: int) -> bool:
        """Return True if (x, y) is an obstacle."""
        if not self.is_valid_coordinate(x, y):
            return True
        return self.grid[y, x] == CELL_OBSTACLE

    def get_total_free_cells(self) -> int:
        """Return total count of non-obstacle cells in the ground truth world."""
        return int(np.sum(self.grid != CELL_OBSTACLE))

    def get_remaining_survivors_count(self) -> int:
        """Count of survivors not yet sensed by any drone."""
        return len(self.survivors - self.discovered_survivors)
