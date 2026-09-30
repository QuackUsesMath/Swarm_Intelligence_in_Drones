"""
Unit tests for area division, frontier assignment, and spatial dispersion.
Verifies that:
1. Frontier detection correctly locates free cells bordering unknown territory.
2. Market auction resolves claims and prevents duplicate target claims.
3. Swarm disperses across distinct exploration sectors rather than clustering.
4. Stigmergic pheromone repulsion drives drones away from heavily trodden zones.
"""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import SwarmConfig
from auction import AuctionManager
from drone import DroneAgent
from planner import BELIEF_FREE, BELIEF_OBSTACLE, BELIEF_UNKNOWN
from simulator import SwarmSimulator


class TestAreaDivision(unittest.TestCase):
    """Tests for spatial area division and decentralized target allocation."""

    def setUp(self):
        self.config = SwarmConfig(grid_width=30, grid_height=30, seed=42)

    def test_frontier_detection_accuracy(self):
        """Verify that only free cells adjacent to unknown cells are identified as frontiers."""
        belief_grid = np.full((30, 30), BELIEF_UNKNOWN, dtype=np.int8)
        # Clear a 5x5 center square
        belief_grid[10:15, 10:15] = BELIEF_FREE

        auction = AuctionManager(
            drone_id=0,
            w_dist=1.0,
            w_energy=0.5,
            w_cong=1.0,
            w_phero=1.0
        )
        frontiers = auction.find_frontiers(belief_grid, known_obstacles=set())

        # Interior cell (12, 12) has all free neighbors -> NOT a frontier
        self.assertNotIn((12, 12), frontiers)

        # Boundary cell (10, 10) borders unknown -> MUST be a frontier
        self.assertIn((10, 10), frontiers)
        self.assertIn((14, 14), frontiers)

    def test_auction_prevents_duplicate_targeting(self):
        """Verify that when multiple drones evaluate frontiers, conflict resolution prevents duplicates."""
        sim = SwarmSimulator(self.config, record_history=False)

        # Run for 15 ticks so drones detect frontiers and bid
        for _ in range(15):
            sim.step()

        # Collect targets of all operational drones
        active_targets = [
            d.target for d in sim.drones.values()
            if d.is_operational() and d.target is not None
        ]

        # All claimed targets must be distinct
        self.assertEqual(
            len(active_targets),
            len(set(active_targets)),
            f"Duplicate target claims detected among active drones: {active_targets}"
        )

    def test_pheromone_repels_exploration(self):
        """Verify that higher local pheromone increases bid cost and drives drones elsewhere."""
        auction = AuctionManager(
            drone_id=0,
            w_dist=1.0,
            w_energy=0.0,
            w_cong=0.0,
            w_phero=2.0
        )
        pos = (5, 5)
        frontier = (10, 10)

        # Zero pheromone grid
        clean_phero = np.zeros((30, 30), dtype=np.float32)
        cost_clean = auction.compute_bid(pos, frontier, battery=500.0, neighbor_targets=[], local_phero_grid=clean_phero)

        # Saturated pheromone grid
        hot_phero = np.zeros((30, 30), dtype=np.float32)
        hot_phero[10, 10] = 5.0
        cost_hot = auction.compute_bid(pos, frontier, battery=500.0, neighbor_targets=[], local_phero_grid=hot_phero)

        self.assertGreater(cost_hot, cost_clean, "High pheromone trail must produce higher bid cost.")

    def test_swarm_spatial_dispersion(self):
        """Verify that over time, the swarm spreads across quadrants instead of remaining clustered."""
        sim = SwarmSimulator(self.config, record_history=False)
        for _ in range(40):
            sim.step()

        positions = [d.pos for d in sim.drones.values() if d.is_operational()]
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]

        x_spread = max(xs) - min(xs)
        y_spread = max(ys) - min(ys)

        self.assertGreater(x_spread, 8, "Drones failed to disperse horizontally.")
        self.assertGreater(y_spread, 8, "Drones failed to disperse vertically.")


if __name__ == "__main__":
    unittest.main()
