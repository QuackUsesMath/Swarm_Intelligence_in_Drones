"""
Unit tests for multi-agent collision avoidance and priority-based yield rules.
Verifies that:
1. Vertex collisions (two drones entering same cell) are resolved by priority.
2. Edge swap collisions (head-on position swaps) are prevented.
3. Stationary and failed drone cells are protected from being entered.
4. Cascading yield chains resolve without deadlocks.
5. Long-duration multi-agent runs maintain zero physical collisions.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import SwarmConfig
from simulator import SwarmSimulator


class TestCollisionAvoidance(unittest.TestCase):
    """Rigorous tests for priority-based multi-agent collision avoidance."""

    def setUp(self):
        self.config = SwarmConfig(grid_width=40, grid_height=40, num_drones=4, seed=42)
        self.sim = SwarmSimulator(self.config, record_history=False)

    def test_vertex_collision_resolution(self):
        """When two drones propose moving into the exact same cell, lowest ID moves and other yields."""
        self.sim.drones[0].pos = (10, 10)
        self.sim.drones[1].pos = (11, 11)

        # Both propose moving to (10, 11)
        proposed = {
            0: (10, 11),
            1: (10, 11),
            2: (0, 0),
            3: (0, 1)
        }
        resolved, _ = self.sim._arbitrate_collisions(proposed)

        # Drone 0 has higher priority (id 0 < id 1) -> moves to (10, 11)
        self.assertEqual(resolved[0], (10, 11))
        # Drone 1 yields -> remains at original pos (11, 11)
        self.assertEqual(resolved[1], (11, 11))
        # No duplicate positions
        self.assertEqual(len(set(resolved.values())), len(resolved))

    def test_edge_swap_collision_resolution(self):
        """When two adjacent drones propose swapping positions, swap is prevented."""
        self.sim.drones[0].pos = (10, 10)
        self.sim.drones[1].pos = (10, 11)

        # Drone 0 proposes (10, 11), Drone 1 proposes (10, 10)
        proposed = {
            0: (10, 11),
            1: (10, 10),
            2: (0, 0),
            3: (0, 1)
        }
        resolved, _ = self.sim._arbitrate_collisions(proposed)

        # Cannot swap: Drone 1 (lower priority) must yield and stay at (10, 11)
        # And because Drone 1 stays at (10, 11), Drone 0 cannot move to (10, 11) either
        self.assertNotEqual((resolved[0], resolved[1]), ((10, 11), (10, 10)))
        self.assertEqual(len(set(resolved.values())), len(resolved))

    def test_stationary_drone_cell_protection(self):
        """A drone cannot enter a cell occupied by a stationary or failed drone."""
        self.sim.drones[2].pos = (15, 15)
        self.sim.drones[2].fail_drone()  # Stationary

        self.sim.drones[0].pos = (15, 14)
        proposed = {
            0: (15, 15),  # Attempts to enter failed drone's cell
            1: (1, 1),
            2: (15, 15),  # Stays
            3: (2, 2)
        }
        resolved, _ = self.sim._arbitrate_collisions(proposed)

        # Drone 0 must be denied entry and remain at (15, 14)
        self.assertEqual(resolved[0], (15, 14))
        self.assertEqual(resolved[2], (15, 15))

    def test_cascading_yield_chain(self):
        """A blocked drone forces any drone following directly behind it to also yield."""
        self.sim.drones[0].pos = (5, 5)   # Blocked by obstacle / stationary
        self.sim.drones[1].pos = (5, 4)   # Wants (5, 5)
        self.sim.drones[2].pos = (5, 3)   # Wants (5, 4)
        self.sim.drones[3].pos = (1, 1)

        # Drone 0 proposes staying at (5, 5)
        proposed = {
            0: (5, 5),
            1: (5, 5),
            2: (5, 4),
            3: (1, 2)
        }
        resolved, _ = self.sim._arbitrate_collisions(proposed)

        # Cascade: 0 stays at (5, 5) -> 1 must stay at (5, 4) -> 2 must stay at (5, 3)
        self.assertEqual(resolved[0], (5, 5))
        self.assertEqual(resolved[1], (5, 4))
        self.assertEqual(resolved[2], (5, 3))
        self.assertEqual(len(set(resolved.values())), len(resolved))

    def test_zero_collisions_over_full_mission(self):
        """Assert zero collisions across 100 simulation ticks with active drone failures."""
        cfg = SwarmConfig(
            grid_width=50,
            grid_height=50,
            num_drones=8,
            max_ticks=100,
            failure_schedule={25: [2], 50: [5]},
            seed=77
        )
        sim = SwarmSimulator(cfg, record_history=False)
        for _ in range(100):
            sim.step()

        metrics = sim.metrics_tracker.calculate_metrics(100)
        self.assertEqual(metrics.collisions, 0, "No collisions may occur during the entire mission.")


if __name__ == "__main__":
    unittest.main()
