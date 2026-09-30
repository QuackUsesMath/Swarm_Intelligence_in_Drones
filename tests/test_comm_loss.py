"""
Unit tests for communication loss, dropout windows, and decentralized gossip recovery.
Verifies that:
1. CommsChannel drops all transmissions during active blackout windows.
2. Range limits prevent message passing between distant drones.
3. Drones continue decentralized exploration independently during comms loss.
4. Gossip synchronizes discovered maps and claims immediately upon reconnection.
"""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from comms import CommsChannel, GossipMessage
from config import SwarmConfig
from simulator import SwarmSimulator


class TestCommLoss(unittest.TestCase):
    """Tests for communication blackout, range constraints, and reconnection."""

    def test_dropout_window_enforcement(self):
        """Verify that is_in_dropout accurately detects blackout intervals."""
        dropouts = [(20, 35), (60, 70)]
        channel = CommsChannel(comm_range=10.0, dropouts=dropouts)

        self.assertFalse(channel.is_in_dropout(10))
        self.assertTrue(channel.is_in_dropout(20))
        self.assertTrue(channel.is_in_dropout(28))
        self.assertTrue(channel.is_in_dropout(35))
        self.assertFalse(channel.is_in_dropout(36))
        self.assertTrue(channel.is_in_dropout(65))
        self.assertFalse(channel.is_in_dropout(75))

    def test_zero_communicating_pairs_during_dropout(self):
        """Even if drones are adjacent, zero pairs communicate during blackout."""
        channel = CommsChannel(comm_range=10.0, dropouts=[(10, 20)])
        positions = {0: (5, 5), 1: (5, 6)}  # Adjacent
        active_ids = {0, 1}

        # During blackout
        pairs = channel.get_communicating_pairs(positions, active_ids, current_tick=15)
        self.assertEqual(len(pairs), 0, "No messages may be exchanged during communication dropout.")

        # Outside blackout
        pairs = channel.get_communicating_pairs(positions, active_ids, current_tick=25)
        self.assertEqual(len(pairs), 1, "Adjacent drones must communicate outside blackout window.")

    def test_comm_range_filtering(self):
        """Verify that drones beyond comm_range cannot communicate."""
        channel = CommsChannel(comm_range=10.0, dropouts=[])
        # Distance = 15 > comm_range (10.0)
        positions = {0: (0, 0), 1: (15, 0)}
        pairs = channel.get_communicating_pairs(positions, {0, 1}, current_tick=1)
        self.assertEqual(len(pairs), 0)

        # Distance = 8 <= comm_range
        positions = {0: (0, 0), 1: (8, 0)}
        pairs = channel.get_communicating_pairs(positions, {0, 1}, current_tick=1)
        self.assertEqual(pairs, [(0, 1)])

    def test_exploration_continues_during_blackout(self):
        """Verify that drones continue to move and explore during communication blackout."""
        cfg = SwarmConfig(
            grid_width=40,
            grid_height=40,
            num_drones=4,
            max_ticks=50,
            comm_dropouts=[(10, 40)],  # Long blackout
            seed=42
        )
        sim = SwarmSimulator(cfg, record_history=False)

        # Step through ticks before blackout
        for _ in range(10):
            sim.step()
        coverage_at_blackout_start = sim.metrics_tracker.calculate_metrics(10).coverage_fraction

        # Step through blackout window
        for _ in range(25):
            sim.step()
        coverage_mid_blackout = sim.metrics_tracker.calculate_metrics(35).coverage_fraction

        # Drones must have expanded coverage autonomously without comms
        self.assertGreater(
            coverage_mid_blackout,
            coverage_at_blackout_start,
            "Swarm froze during communication blackout; must continue autonomous exploration."
        )

    def test_gossip_reconnection_sync(self):
        """Verify that when reconnection occurs, drones sync newly discovered map cells."""
        cfg = SwarmConfig(
            grid_width=30,
            grid_height=30,
            num_drones=2,
            comm_range=100.0,  # Always in range
            comm_dropouts=[(1, 10)],  # Blackout at start
            seed=42
        )
        sim = SwarmSimulator(cfg, record_history=False)

        # Step through blackout
        for _ in range(10):
            sim.step()

        # Step at tick 11 (reconnected)
        sim.step()

        # Drone 0 and 1 should now share their discovered obstacles
        d0_obs = sim.drones[0].known_obstacles
        d1_obs = sim.drones[1].known_obstacles
        self.assertEqual(d0_obs, d1_obs, "Reconnected drones failed to synchronize known obstacles via gossip.")


if __name__ == "__main__":
    unittest.main()
