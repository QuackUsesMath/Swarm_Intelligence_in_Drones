"""
Automated unit and integration test suite.
Asserts:
1. Strict zero-collision guarantee across multi-agent runs.
2. Per-tick computation stays strictly within real-time budget.
3. Swarm exploration coverage actively recovers and expands following mid-mission drone failures.
4. Heartbeat failure detection and consensus target release.
"""

import sys
import os
import unittest
import numpy as np

# Ensure parent directory is on sys.path for direct module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import SwarmConfig
from simulator import SwarmSimulator


class TestSwarmSimulation(unittest.TestCase):
    """Test suite verifying safety, timeliness, and decentralized resilience."""

    def test_no_collisions_across_perturbations(self):
        """Verify that no two drones ever occupy the same cell or swap cells."""
        for test_seed in [42, 99, 137]:
            config = SwarmConfig(
                seed=test_seed,
                max_ticks=100,
                num_drones=8,
                failure_schedule={25: [1], 50: [3]},
                comm_dropouts=[(30, 45)],
                dynamic_obstacles={40: [(10, 10), (10, 11), (11, 10)]}
            )
            sim = SwarmSimulator(config)

            for t in range(1, config.max_ticks + 1):
                prev_positions = {d_id: d.pos for d_id, d in sim.drones.items()}
                sim.step()
                curr_positions = {d_id: d.pos for d_id, d in sim.drones.items()}

                # 1. Assert no two drones occupy the same cell
                unique_positions = set(curr_positions.values())
                self.assertEqual(
                    len(unique_positions),
                    len(curr_positions),
                    f"Vertex collision detected at tick {t} with seed {test_seed}: positions {curr_positions}"
                )

                # 2. Assert no edge swaps occurred
                for id_a, pos_a in curr_positions.items():
                    prev_a = prev_positions[id_a]
                    for id_b, pos_b in curr_positions.items():
                        if id_a < id_b:
                            prev_b = prev_positions[id_b]
                            swap_occurred = (pos_a == prev_b and pos_b == prev_a and pos_a != prev_a)
                            self.assertFalse(
                                swap_occurred,
                                f"Edge swap collision between UAV {id_a} and {id_b} at tick {t}"
                            )

            metrics = sim.metrics_tracker.calculate_metrics(config.max_ticks)
            self.assertEqual(metrics.collisions, 0, "Collisions counter must remain zero.")

    def test_tick_time_under_budget(self):
        """Verify that per-drone decision cycle strictly stays under tick compute budget."""
        config = SwarmConfig(
            seed=42,
            max_ticks=80,
            tick_compute_budget_ms=25.0
        )
        sim = SwarmSimulator(config)

        for _ in range(config.max_ticks):
            sim.step()

        metrics = sim.metrics_tracker.calculate_metrics(config.max_ticks)
        # Max compute time must respect the 25ms budget
        self.assertLessEqual(
            metrics.max_drone_tick_compute_ms,
            config.tick_compute_budget_ms,
            f"Drone planning exceeded budget: {metrics.max_drone_tick_compute_ms:.2f} ms > {config.tick_compute_budget_ms} ms"
        )
        # Average compute time should be well below budget (< 5ms)
        self.assertLess(
            metrics.avg_drone_tick_compute_ms,
            5.0,
            f"Average drone compute time too high: {metrics.avg_drone_tick_compute_ms:.2f} ms"
        )

    def test_coverage_recovers_after_drone_failure(self):
        """
        Verify swarm resilience: after 2 drones permanently fail at tick 30,
        the remaining drones recover the lost targets and continue to actively expand coverage.
        """
        config = SwarmConfig(
            seed=42,
            max_ticks=120,
            num_drones=8,
            failure_schedule={30: [2, 5]},
            comm_dropouts=[],
            dynamic_obstacles={}
        )
        sim = SwarmSimulator(config)

        coverage_at_failure = 0.0
        for t in range(1, config.max_ticks + 1):
            sim.step()
            if t == 30:
                metrics_at_fail = sim.metrics_tracker.calculate_metrics(t)
                coverage_at_failure = metrics_at_fail.coverage_fraction

        final_metrics = sim.metrics_tracker.calculate_metrics(config.max_ticks)

        # Coverage must have significantly grown past the failure point
        coverage_growth = final_metrics.coverage_fraction - coverage_at_failure
        self.assertGreater(
            coverage_growth,
            0.15,
            f"Coverage failed to recover sufficiently after drone failure. Growth was only {coverage_growth*100:.1f}%"
        )
        # Assert failed drones remain permanently failed
        self.assertEqual(sim.drones[2].status, "FAILED")
        self.assertEqual(sim.drones[5].status, "FAILED")

    def test_heartbeat_timeout_and_claim_release(self):
        """Verify that silent neighbors are detected as failed and their claimed frontiers released."""
        config = SwarmConfig(seed=42, heartbeat_timeout=4)
        sim = SwarmSimulator(config)

        # Force drone 3 to claim a specific target
        sim.drones[3].auction.claims[(20, 20)] = (3, 5.0, 1)
        sim.drones[0].auction.claims[(20, 20)] = (3, 5.0, 1)
        sim.drones[0].last_heartbeat[3] = 1

        # Check at tick 10 (> heartbeat_timeout)
        sim.drones[0].check_heartbeats(current_tick=10)

        # Drone 0 should mark drone 3 as failed and release its claims
        self.assertIn(3, sim.drones[0].failed_neighbors)
        self.assertNotIn((20, 20), sim.drones[0].auction.claims)

    def test_schedule_string_parsers(self):
        """Verify parsing of dynamic schedule strings from CLI."""
        from config import parse_failures, parse_dropouts, parse_collapses, parse_coord

        self.assertEqual(parse_coord("12,34"), (12, 34))
        self.assertEqual(parse_failures("30:2;60:4,5"), {30: [2], 60: [4, 5]})
        self.assertEqual(parse_failures("none"), {})
        self.assertEqual(parse_dropouts("50-75;100-120"), [(50, 75), (100, 120)])
        self.assertEqual(parse_dropouts("none"), [])

        collapses = parse_collapses("50:10,10;60:20,20,2")
        self.assertEqual(collapses[50], [(10, 10)])
        self.assertEqual(len(collapses[60]), 4)  # 2x2 cluster
        self.assertIn((20, 20), collapses[60])
        self.assertIn((21, 21), collapses[60])

    def test_json_roundtrip_serialization(self):
        """Verify JSON export and import fidelity."""
        cfg_original = SwarmConfig(
            grid_width=80,
            grid_height=80,
            num_drones=12,
            failure_schedule={20: [1], 40: [3, 7]},
            comm_dropouts=[(30, 50)],
            dynamic_obstacles={45: [(10, 10), (11, 11)]}
        )
        json_str = cfg_original.to_json()
        cfg_restored = SwarmConfig.from_json(json_str)

        self.assertEqual(cfg_restored.grid_width, 80)
        self.assertEqual(cfg_restored.grid_height, 80)
        self.assertEqual(cfg_restored.num_drones, 12)
        self.assertEqual(cfg_restored.failure_schedule, {20: [1], 40: [3, 7]})
        self.assertEqual(cfg_restored.comm_dropouts, [(30, 50)])
        self.assertEqual(cfg_restored.dynamic_obstacles, {45: [(10, 10), (11, 11)]})


if __name__ == "__main__":
    unittest.main()
