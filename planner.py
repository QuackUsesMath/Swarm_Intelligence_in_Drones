"""
Path planning module implementing A* search with obstacle inflation and greedy fallback.
Guarantees per-tick computational budget enforcement so no tick is ever missed.
"""

import heapq
import time
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

# Map belief constants
BELIEF_UNKNOWN = -1
BELIEF_FREE = 0
BELIEF_OBSTACLE = 1
BELIEF_SURVIVOR = 2


class AStarPlanner:
    """Decentralized path planner with obstacle inflation and budget-bounded execution."""

    def __init__(
        self,
        safety_margin: float = 1.0,
        max_expansions: int = 500,
        time_budget_ms: float = 20.0
    ):
        self.safety_margin = safety_margin
        self.max_expansions = max_expansions
        self.time_budget_sec = time_budget_ms / 1000.0

    @staticmethod
    def _heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Octile / Manhattan-Euclidean hybrid heuristic for 4-connected grid."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def plan_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        belief_grid: np.ndarray,
        known_obstacles: Set[Tuple[int, int]],
        pheromone_grid: Optional[np.ndarray] = None,
        w_phero: float = 0.5,
        remaining_time_budget_sec: Optional[float] = None
    ) -> List[Tuple[int, int]]:
        """
        Compute path from start to goal using budget-bounded A*.
        Falls back to greedy best-neighbor step if compute budget or expansion limit is exceeded.
        """
        if start == goal:
            return [start]

        height, width = belief_grid.shape
        start_time = time.perf_counter()
        time_limit = remaining_time_budget_sec if remaining_time_budget_sec is not None else self.time_budget_sec

        # Build dynamic obstacle inflation lookup or check on-the-fly
        # 4-connected neighbors
        neighbors_delta = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        # Precompute safe proximity penalty for cells near known obstacles
        def get_traversal_cost(nx: int, ny: int) -> float:
            base_cost = 1.0
            # Obstacle inflation cost: penalty if adjacent to known obstacle
            if self.safety_margin > 0.0:
                for dx, dy in neighbors_delta:
                    ox, oy = nx + dx, ny + dy
                    if (ox, oy) in known_obstacles:
                        base_cost += 2.0 * self.safety_margin
                        break
            # Minor pheromone repulsion to avoid saturated trails during transit
            if pheromone_grid is not None:
                base_cost += w_phero * float(pheromone_grid[ny, nx]) * 0.2
            return base_cost

        # Priority queue stores (f_score, g_score, (x, y))
        open_set: List[Tuple[float, float, Tuple[int, int]]] = []
        heapq.heappush(open_set, (self._heuristic(start, goal), 0.0, start))

        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {start: 0.0}
        closed_set: Set[Tuple[int, int]] = set()

        expansions = 0
        budget_exceeded = False
        best_partial_node = start
        min_h = self._heuristic(start, goal)

        while open_set:
            expansions += 1
            # Check expansion budget
            if expansions > self.max_expansions:
                budget_exceeded = True
                break

            # Check time budget periodically
            if expansions % 15 == 0:
                if (time.perf_counter() - start_time) > time_limit:
                    budget_exceeded = True
                    break

            _, current_g, current = heapq.heappop(open_set)

            if current == goal:
                # Reconstruct full path
                return self._reconstruct_path(came_from, current)

            if current in closed_set:
                continue
            closed_set.add(current)

            # Track closest node reached in case of budget cutoff
            h_curr = self._heuristic(current, goal)
            if h_curr < min_h:
                min_h = h_curr
                best_partial_node = current

            cx, cy = current
            for dx, dy in neighbors_delta:
                nx, ny = cx + dx, cy + dy

                # Bounds check
                if not (0 <= nx < width and 0 <= ny < height):
                    continue

                # Collision check against local belief (cannot traverse known obstacles)
                if (nx, ny) in known_obstacles or belief_grid[ny, nx] == BELIEF_OBSTACLE:
                    continue

                move_cost = get_traversal_cost(nx, ny)
                tentative_g = current_g + move_cost

                if tentative_g < g_score.get((nx, ny), float('inf')):
                    came_from[(nx, ny)] = current
                    g_score[(nx, ny)] = tentative_g
                    f_score = tentative_g + self._heuristic((nx, ny), goal)
                    heapq.heappush(open_set, (f_score, tentative_g, (nx, ny)))

        # Fallback handling:
        # If budget exceeded and we found a partial path towards goal, reconstruct to best node
        if budget_exceeded and best_partial_node != start:
            partial_path = self._reconstruct_path(came_from, best_partial_node)
            if len(partial_path) > 1:
                return partial_path

        # If A* fails or times out immediately, execute greedy single-step fallback
        greedy_step = self._greedy_fallback_step(
            start, goal, belief_grid, known_obstacles, pheromone_grid, w_phero
        )
        return [start, greedy_step] if greedy_step != start else [start]

    def _reconstruct_path(
        self, came_from: Dict[Tuple[int, int], Tuple[int, int]], current: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """Reconstruct path from start to target."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def _greedy_fallback_step(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        belief_grid: np.ndarray,
        known_obstacles: Set[Tuple[int, int]],
        pheromone_grid: Optional[np.ndarray],
        w_phero: float
    ) -> Tuple[int, int]:
        """
        Greedy fallback step to guarantee zero missed ticks when time budget is exhausted.
        Chooses the neighboring cell that minimizes distance to target + hazard penalty.
        """
        height, width = belief_grid.shape
        sx, sy = start
        candidates = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        best_step = start
        best_cost = float('inf')

        for dx, dy in candidates:
            nx, ny = sx + dx, sy + dy
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            if (nx, ny) in known_obstacles or belief_grid[ny, nx] == BELIEF_OBSTACLE:
                continue

            h = self._heuristic((nx, ny), goal)
            phero = float(pheromone_grid[ny, nx]) if pheromone_grid is not None else 0.0
            cost = h + w_phero * phero

            if cost < best_cost:
                best_cost = cost
                best_step = (nx, ny)

        return best_step
