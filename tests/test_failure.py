"""
Unit tests for drone failure perturbations, heartbeat timeout detection,
consensus target release, and swarm resilience.
Verifies that:
1. Scheduled failures permanently disable the target UAV.
2. Silent neighbors are detected via heartbeat timeouts.
3. Abandoned target claims are released back to the open pool.
4. Surviving drones take over abandoned search areas.
5. Battery depletion leads to safe stationary status.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import SwarmConfig
from simulator import SwarmSimulator


class TestDroneFailure(unittest.TestCase):
    """Tests for drone failure, heartbeat detection, and target re-assignment."""

    def test_scheduled_failure_execution(self):
        """Verify that scheduled failure halts the drone permanently."""
        cfg = SwarmConfig(
            grid_width=30,
            grid_height=30,
            num_drones=3,
            max_ticks=30,
            failure_schedule={15: [1]},
            comm_dropouts=[],
            seed=42
        )
        sim = SwarmSimulator(cfg, record_history=False)

        for _ in range(14):
            sim.step()
        self.assertEqual(sim.drones[1].status, "ACTIVE")
        self.assertTrue(sim.drones[1].is_operational())

        # Step 15: drone 1 fails
        sim.step()
        self.assertEqual(sim.drones[1].status, "FAILED")
        self.assertFalse(sim.drones[1].is_operational())

        pos_at_fail = sim.drones[1].pos
        # Step through remaining ticks: pos must not change
        for _ in range(15):
            sim.step()
        self.assertEqual(sim.drones[1].pos, pos_at_fail, "Failed drone must remain stationary.")

    def test_heartbeat_timeout_clears_claims(self):
        """Verify that when a drone fails, other drones detect timeout and release its claims."""
        cfg = SwarmConfig(
            grid_width=40,
            grid_height=40,
            num_drones=4,
            heartbeat_timeout=5,
            comm_range=100.0,
            seed=42
        )
        sim = SwarmSimulator(cfg, record_history=False)

        # Force drone 2 to claim target (20, 20)
        target = (20, 20)
        sim.drones[2].target = target
        sim.drones[2].auction.claims[target] = (2, 5.0, 1)

        # Propagate claim to drone 0
        sim.drones[0].auction.claims[target] = (2, 5.0, 1)
        sim.drones[0].last_heartbeat[2] = 1

        # Simulate drone 2 failure
        sim.drones[2].fail_drone()

        # At tick 10 (> 1 + heartbeat_timeout), drone 0 checks heartbeats
        sim.drones[0].check_heartbeats(current_tick=10)

        # Assert drone 0 detected failure and purged claim
        self.assertIn(2, sim.drones[0].failed_neighbors)
        self.assertNotIn(target, sim.drones[0].auction.claims)

    def test_coverage_recovery_after_multiple_failures(self):
        """Verify that when 2 out of 6 drones fail, remaining 4 drones continue expanding coverage."""
        cfg = SwarmConfig(
            grid_width=50,
            grid_height=50,
            num_drones=6,
            max_ticks=100,
            failure_schedule={30: [1, 4]},
            comm_dropouts=[],
            seed=42
        )
        sim = SwarmSimulator(cfg, record_history=False)

        for _ in range(30):
            sim.step()
        cov_30 = sim.metrics_tracker.calculate_metrics(30).coverage_fraction

        for _ in range(70):
            sim.step()
        cov_final = sim.metrics_tracker.calculate_metrics(100).coverage_fraction

        # Remaining swarm must expand coverage significantly after failure
        growth = cov_final - cov_30
        self.assertGreater(growth, 0.15, f"Swarm coverage did not recover; growth was only {growth*100:.1f}%.")

    def test_battery_depletion_handling(self):
        """Verify that when battery reaches zero, drone transitions to DEPLETED and halts safely."""
        cfg = SwarmConfig(grid_width=30, grid_height=30, num_drones=2, initial_battery=10.0, seed=42)
        sim = SwarmSimulator(cfg, record_history=False)

        # Drain drone 0 battery
        sim.drones[0].battery = 0.5
        sim.drones[0].execute_move((1, 0))  # Drains move cost (1.0) -> battery becomes 0.0

        self.assertEqual(sim.drones[0].status, "DEPLETED")
        self.assertFalse(sim.drones[0].is_operational())
        self.assertEqual(sim.drones[0].planned_path, [])
        self.assertIsNone(sim.drones[0].target)


if __name__ == "__main__":
    unittest.main()
