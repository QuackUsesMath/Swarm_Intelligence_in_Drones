"""
Unit tests for metrics tracking, fitness calculation, and resilience scoring.
Verifies that:
1. Coverage fraction accurately reflects free-cell exploration.
2. Survivors found fraction correctly counts discovered survivors.
3. Path overlap ratio properly computes trajectory redundancy.
4. Resilience score evaluates performance under perturbations against baseline.
5. Multi-objective fitness function strictly conforms to mathematical weights.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import SwarmConfig
from metrics import MetricsTracker, SwarmMetrics


class TestMetrics(unittest.TestCase):
    """Tests for metrics tracking and fitness scoring formulas."""

    def setUp(self):
        self.config = SwarmConfig(
            max_ticks=100,
            fit_a_survivors=100.0,
            fit_b_coverage=50.0,
            fit_c_time=0.1,
            fit_d_collisions=50.0,
            fit_e_energy=0.01,
            fit_f_overlap=20.0,
            fit_g_resilience=30.0
        )
        self.tracker = MetricsTracker(
            config=self.config,
            total_free_cells=1000,
            total_survivors=10
        )

    def test_survivors_and_coverage_fractions(self):
        """Verify fractional calculations of coverage and survivor recovery."""
        # Record 5 survivors found and 400 cells explored
        survivors = {(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)}
        cells = {(x, 0) for x in range(400)}
        positions = {(0, 0), (1, 1)}

        self.tracker.record_tick(
            current_tick=1,
            discovered_survivors_now=survivors,
            explored_cells_now=cells,
            drone_positions=positions,
            tick_collisions=0,
            tick_energy=2.0,
            drone_compute_times_ms=[1.2, 1.4]
        )

        metrics = self.tracker.calculate_metrics(final_tick=1)
        self.assertEqual(metrics.survivors_found, 5)
        self.assertAlmostEqual(metrics.survivors_found_fraction, 0.5)
        self.assertEqual(metrics.explored_free_cells, 400)
        self.assertAlmostEqual(metrics.coverage_fraction, 0.4)

    def test_overlap_ratio_calculation(self):
        """Verify that repeated cell visits yield non-zero overlap ratio."""
        # 10 moves over only 2 distinct cells -> 80% overlap
        positions = {(0, 0), (0, 1)}
        for t in range(1, 6):
            self.tracker.record_tick(
                current_tick=t,
                discovered_survivors_now=set(),
                explored_cells_now=set(),
                drone_positions=positions,
                tick_collisions=0,
                tick_energy=1.0,
                drone_compute_times_ms=[0.5]
            )

        metrics = self.tracker.calculate_metrics(final_tick=5)
        # total_moves = 10, unique_cells = 2 -> overlap = 1 - 2/10 = 0.8
        self.assertAlmostEqual(metrics.overlap_ratio, 0.8)

    def test_resilience_score_calculation(self):
        """Verify resilience score formula against undisturbed baseline."""
        # 600 cells explored (60%), 8 survivors found (80%)
        self.tracker.explored_cells = {(x, 0) for x in range(600)}
        self.tracker.discovered_survivors = {(x, 1) for x in range(8)}

        # Baseline: 80% coverage, 100% survivors
        baseline_cov = 0.80
        baseline_surv = 1.00

        metrics = self.tracker.calculate_metrics(
            final_tick=100,
            baseline_coverage=baseline_cov,
            baseline_survivors=baseline_surv
        )

        # Expected resilience = 0.5 * (0.6 / 0.8) + 0.5 * (0.8 / 1.0) = 0.5 * 0.75 + 0.5 * 0.8 = 0.375 + 0.4 = 0.775
        self.assertAlmostEqual(metrics.resilience_score, 0.775)

    def test_multi_objective_fitness_value(self):
        """Verify fitness value matches exact formula: a*S + b*C - c*T - d*N - e*E - f*R + g*res."""
        self.tracker.explored_cells = {(x, 0) for x in range(500)}  # 50% cov
        self.tracker.discovered_survivors = {(x, 1) for x in range(10)}  # 100% surv
        self.tracker.all_survivors_found_tick = 50
        self.tracker.total_swarm_moves = 100
        self.tracker.all_visited_cells = {(x, 0) for x in range(50)}  # overlap = 1 - 50/100 = 0.5
        self.tracker.total_energy_expended = 100.0
        self.tracker.collision_events = 0

        metrics = self.tracker.calculate_metrics(final_tick=100)

        # Expected:
        # a * 1.0 = 100.0
        # b * 0.5 = 25.0
        # - c * 50 = -5.0
        # - d * 0 = 0.0
        # - e * 100.0 = -1.0
        # - f * 0.5 = -10.0
        # + g * 1.0 = 30.0
        # Total = 100 + 25 - 5 - 1 - 10 + 30 = 139.0
        expected_fitness = 100.0 + 25.0 - 5.0 - 1.0 - 10.0 + 30.0
        self.assertAlmostEqual(metrics.fitness, expected_fitness)


if __name__ == "__main__":
    unittest.main()
