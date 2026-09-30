"""
Decentralized auction and bidding module for frontier exploration.
Calculates multidimensional bid costs, resolves conflicts deterministically,
and manages consensus claims with heartbeat timeout releases.
"""

from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from planner import BELIEF_FREE, BELIEF_UNKNOWN, BELIEF_OBSTACLE


class AuctionManager:
    """Manages local frontier bidding, claims ledger, and decentralized conflict resolution."""

    def __init__(
        self,
        drone_id: int,
        w_dist: float,
        w_energy: float,
        w_cong: float,
        w_phero: float,
        congestion_radius: float = 6.0,
        max_candidates: int = 16
    ):
        self.drone_id = drone_id
        self.w_dist = w_dist
        self.w_energy = w_energy
        self.w_cong = w_cong
        self.w_phero = w_phero
        self.congestion_radius = congestion_radius
        self.max_candidates = max_candidates

        # Active claims ledger: target (x, y) -> (winner_id, bid_cost, claim_tick)
        self.claims: Dict[Tuple[int, int], Tuple[int, float, int]] = {}

        # Current target committed by this drone
        self.current_target: Optional[Tuple[int, int]] = None
        self.current_bid: float = float('inf')

    def find_frontiers(
        self, belief_grid: np.ndarray, known_obstacles: Set[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """
        Identify frontier cells (known free cells adjacent to at least one unknown cell)
        using fast vectorized numpy operations.
        """
        height, width = belief_grid.shape
        is_unknown = (belief_grid == BELIEF_UNKNOWN)

        # Vectorized neighbor check (4-connected)
        has_unknown_neighbor = np.zeros((height, width), dtype=bool)
        has_unknown_neighbor[:-1, :] |= is_unknown[1:, :]   # neighbor below
        has_unknown_neighbor[1:, :] |= is_unknown[:-1, :]   # neighbor above
        has_unknown_neighbor[:, :-1] |= is_unknown[:, 1:]   # neighbor right
        has_unknown_neighbor[:, 1:] |= is_unknown[:, :-1]   # neighbor left

        # Frontier cells are free cells with at least one unknown neighbor
        frontier_mask = (belief_grid == BELIEF_FREE) & has_unknown_neighbor

        # Extract coordinates
        fy, fx = np.nonzero(frontier_mask)
        frontiers: List[Tuple[int, int]] = []
        for x, y in zip(fx, fy):
            pt = (int(x), int(y))
            if pt not in known_obstacles:
                frontiers.append(pt)

        return frontiers

    def compute_bid(
        self,
        current_pos: Tuple[int, int],
        frontier: Tuple[int, int],
        battery: float,
        neighbor_targets: List[Tuple[int, int]],
        local_phero_grid: Optional[np.ndarray]
    ) -> float:
        """
        Calculate weighted bid cost:
        mix of path distance, energy depletion, neighbor congestion, and pheromone repulsion.
        Lowest bid wins.
        """
        px, py = current_pos
        fx, fy = frontier

        # 1. Distance metric (Manhattan proxy for fast calculation)
        dist = float(abs(px - fx) + abs(py - fy))

        # 2. Energy cost: relative to remaining battery
        energy_cost = (dist / max(10.0, battery)) * 20.0

        # 3. Congestion / overlap penalty: proximity to neighbor claims
        congestion_count = 0
        r2 = self.congestion_radius * self.congestion_radius
        for tx, ty in neighbor_targets:
            if (fx - tx) ** 2 + (fy - ty) ** 2 <= r2:
                congestion_count += 1
        congestion_cost = congestion_count * 15.0

        # 4. Pheromone repulsion: penalize heavily trodden areas
        phero_cost = 0.0
        if local_phero_grid is not None and 0 <= fx < local_phero_grid.shape[1] and 0 <= fy < local_phero_grid.shape[0]:
            phero_cost = float(local_phero_grid[fy, fx]) * 10.0

        total_cost = (
            self.w_dist * dist +
            self.w_energy * energy_cost +
            self.w_cong * congestion_cost +
            self.w_phero * phero_cost
        )
        return total_cost

    def select_best_target(
        self,
        current_pos: Tuple[int, int],
        belief_grid: np.ndarray,
        known_obstacles: Set[Tuple[int, int]],
        battery: float,
        neighbor_targets: List[Tuple[int, int]],
        local_phero_grid: Optional[np.ndarray],
        current_tick: int
    ) -> Optional[Tuple[int, int]]:
        """
        Evaluate candidate frontiers and claim the best target with lowest bid.
        Resolves conflicts using (bid_cost, drone_id).
        """
        all_frontiers = self.find_frontiers(belief_grid, known_obstacles)
        if not all_frontiers:
            self.current_target = None
            self.current_bid = float('inf')
            return None

        # Filter out frontiers already firmly won by another drone
        valid_frontiers = []
        for f in all_frontiers:
            if f in self.claims:
                winner_id, winner_bid, _ = self.claims[f]
                if winner_id != self.drone_id:
                    # Only consider if we might beat the current claim
                    px, py = current_pos
                    approx_dist = abs(px - f[0]) + abs(py - f[1])
                    if approx_dist >= winner_bid:
                        continue
            valid_frontiers.append(f)

        if not valid_frontiers:
            valid_frontiers = all_frontiers

        # Subsample candidates closest to current pos to fit compute budget
        valid_frontiers.sort(key=lambda c: abs(current_pos[0] - c[0]) + abs(current_pos[1] - c[1]))
        candidate_subset = valid_frontiers[: self.max_candidates]

        best_target: Optional[Tuple[int, int]] = None
        best_bid = float('inf')

        for candidate in candidate_subset:
            bid = self.compute_bid(
                current_pos, candidate, battery, neighbor_targets, local_phero_grid
            )

            # Check if someone else claims candidate
            if candidate in self.claims:
                winner_id, winner_bid, _ = self.claims[candidate]
                if winner_id != self.drone_id:
                    # Compare bids; tie-break with drone ID
                    if bid > winner_bid or (bid == winner_bid and self.drone_id > winner_id):
                        continue  # Lose to existing claim

            if bid < best_bid:
                best_bid = bid
                best_target = candidate

        if best_target is not None:
            self.current_target = best_target
            self.current_bid = best_bid
            self.claims[best_target] = (self.drone_id, best_bid, current_tick)
        else:
            self.current_target = None
            self.current_bid = float('inf')

        return self.current_target

    def update_with_neighbor_claims(
        self,
        neighbor_id: int,
        neighbor_claims: Dict[Tuple[int, int], Tuple[int, float, int]],
        current_tick: int
    ) -> bool:
        """
        Merge claims from gossip communication.
        Resolves conflicts using lowest bid, tie-broken by drone ID.
        Returns True if this drone's current target was lost in an auction conflict.
        """
        replan_needed = False

        for target, (cand_id, cand_bid, cand_tick) in neighbor_claims.items():
            if target not in self.claims:
                self.claims[target] = (cand_id, cand_bid, cand_tick)
                if target == self.current_target and cand_id != self.drone_id:
                    if cand_bid < self.current_bid or (cand_bid == self.current_bid and cand_id < self.drone_id):
                        self.current_target = None
                        self.current_bid = float('inf')
                        replan_needed = True
            else:
                curr_id, curr_bid, curr_tick = self.claims[target]
                # Lower bid wins, tie broken by lower drone ID
                cand_is_better = (cand_bid < curr_bid) or (cand_bid == curr_bid and cand_id < curr_id)
                if cand_is_better:
                    self.claims[target] = (cand_id, cand_bid, cand_tick)
                    if target == self.current_target and cand_id != self.drone_id:
                        self.current_target = None
                        self.current_bid = float('inf')
                        replan_needed = True

        return replan_needed

    def release_failed_drone_claims(self, failed_drone_ids: Set[int]) -> None:
        """
        Release all targets claimed by failed drones back to the open pool.
        """
        targets_to_remove = [
            t for t, (d_id, _, _) in self.claims.items() if d_id in failed_drone_ids
        ]
        for t in targets_to_remove:
            del self.claims[t]

    def release_completed_target(self, target: Tuple[int, int]) -> None:
        """Release target once it has been reached and explored."""
        if target in self.claims and self.claims[target][0] == self.drone_id:
            del self.claims[target]
        if self.current_target == target:
            self.current_target = None
            self.current_bid = float('inf')
