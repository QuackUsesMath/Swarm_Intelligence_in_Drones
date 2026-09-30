"""
Swarm simulation engine.
Coordinates decentralized drone cycles, physical environment ground truth,
perturbations (failures, comm dropouts, dynamic collapses), and priority yield collision avoidance.
"""

from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from comms import CommsChannel
from config import SwarmConfig
from drone import DroneAgent
from metrics import MetricsTracker, SwarmMetrics
from world import DisasterWorld


class SwarmSimulator:
    """Master simulator for decentralized drone search and rescue."""

    def __init__(self, config: SwarmConfig, record_history: bool = False):
        self.config = config
        self.world = DisasterWorld(config)
        self.comms = CommsChannel(
            comm_range=config.comm_range,
            dropouts=config.comm_dropouts
        )
        self.current_tick: int = 0
        self.record_history = record_history

        # Initialize N drones stationed on base apron
        self.drones: Dict[int, DroneAgent] = {}
        self._initialize_drones()

        # Metrics tracker
        self.metrics_tracker = MetricsTracker(
            config=config,
            total_free_cells=self.world.get_total_free_cells(),
            total_survivors=len(self.world.survivors)
        )

        # History log for visualization: tick -> snapshot
        self.history: List[Dict] = []

    def _initialize_drones(self) -> None:
        """Position drones at distinct initial cells around base station apron."""
        bx, by = self.config.base_station
        for i in range(self.config.num_drones):
            agent = DroneAgent(drone_id=i, config=self.config)
            # Arrange in 3-wide apron: (bx + i%3, by + i//3)
            init_x = bx + (i % 3)
            init_y = by + (i // 3)
            agent.pos = (init_x, init_y)
            agent.belief_grid[init_y, init_x] = 0  # free
            self.drones[i] = agent

        # Initial sensing for each drone
        for agent in self.drones.values():
            obs = self.world.sense_environment(agent.pos, self.config.sensing_radius)
            agent.sense(obs)

    def step(self) -> None:
        """Execute one simulation tick across all agents and world systems."""
        self.current_tick += 1
        t = self.current_tick

        # -------------------------------------------------------------
        # 1. Perturbations: Drone Failures
        # -------------------------------------------------------------
        if t in self.config.failure_schedule:
            for fail_id in self.config.failure_schedule[t]:
                if fail_id in self.drones and self.drones[fail_id].status == "ACTIVE":
                    self.drones[fail_id].fail_drone()

        # -------------------------------------------------------------
        # 2. Perturbations: Dynamic Obstacles (Building Collapses)
        # -------------------------------------------------------------
        newly_collapsed = self.world.trigger_dynamic_perturbations(t)
        # Note: Drones do NOT instantly know global collapses; they must sense them!

        # -------------------------------------------------------------
        # 3. Decentralized Gossip Communication (subject to dropouts & range)
        # -------------------------------------------------------------
        active_drone_ids = {
            d_id for d_id, d in self.drones.items() if d.is_operational()
        }
        drone_positions = {
            d_id: d.pos for d_id, d in self.drones.items()
        }

        # Generate messages from operational drones
        outgoing_msgs = {}
        for d_id in active_drone_ids:
            msg = self.drones[d_id].generate_gossip_message(t)
            if msg is not None:
                outgoing_msgs[d_id] = msg

        # Broadcast via gossip channel
        comm_pairs = self.comms.get_communicating_pairs(
            drone_positions, active_drone_ids, t
        )
        for id_a, id_b in comm_pairs:
            if id_a in outgoing_msgs:
                self.drones[id_b].receive_gossip_message(outgoing_msgs[id_a], t)
            if id_b in outgoing_msgs:
                self.drones[id_a].receive_gossip_message(outgoing_msgs[id_b], t)

        # -------------------------------------------------------------
        # 4. Heartbeat Inspection (Detect failed neighbors & release claims)
        # -------------------------------------------------------------
        for d_id in active_drone_ids:
            self.drones[d_id].check_heartbeats(t)

        # -------------------------------------------------------------
        # 5. Planning & Move Proposal
        # -------------------------------------------------------------
        proposed_moves: Dict[int, Tuple[int, int]] = {}
        compute_times: List[float] = []

        for d_id, drone in self.drones.items():
            if drone.is_operational():
                prop = drone.plan_next_action(t)
                proposed_moves[d_id] = prop
                compute_times.append(drone.compute_time_last_tick)
            else:
                # Failed/depleted drones stay stationary
                proposed_moves[d_id] = drone.pos

        # -------------------------------------------------------------
        # 6. Priority-Based Yield Collision Avoidance Arbitration
        # -------------------------------------------------------------
        actual_moves, tick_collisions = self._arbitrate_collisions(proposed_moves)

        # -------------------------------------------------------------
        # 7. Execute Moves, Sense Environment & Deposit Pheromone
        # -------------------------------------------------------------
        tick_energy = 0.0
        drone_poses_set = set()

        for d_id, drone in self.drones.items():
            prev_battery = drone.battery
            actual_pos = actual_moves[d_id]
            drone.execute_move(actual_pos)
            tick_energy += max(0.0, prev_battery - drone.battery)
            drone_poses_set.add(drone.pos)

            if drone.is_operational():
                # Sense ground truth
                obs = self.world.sense_environment(drone.pos, self.config.sensing_radius)
                drone.sense(obs)
                # Deposit and evaporate local pheromone
                drone.update_pheromone_step()

        # -------------------------------------------------------------
        # 8. Record Telemetry
        # -------------------------------------------------------------
        # Exploration is the union of free/survivor cells known across operational drones
        explored_union = set()
        for drone in self.drones.values():
            known_y, known_x = np.nonzero(drone.belief_grid >= 0)
            for x, y in zip(known_x, known_y):
                if drone.belief_grid[y, x] != 1:  # Not obstacle
                    explored_union.add((int(x), int(y)))

        self.metrics_tracker.record_tick(
            current_tick=t,
            discovered_survivors_now=self.world.discovered_survivors,
            explored_cells_now=explored_union,
            drone_positions=drone_poses_set,
            tick_collisions=tick_collisions,
            tick_energy=tick_energy,
            drone_compute_times_ms=compute_times
        )

        # Record visualization snapshot only if requested
        if self.record_history:
            self._record_snapshot(t)

    def _arbitrate_collisions(
        self, proposed_moves: Dict[int, Tuple[int, int]]
    ) -> Tuple[Dict[int, Tuple[int, int]], int]:
        """
        Priority-based yield rule for decentralized multi-agent collision avoidance:
        - Prevents multiple drones occupying the same destination cell (vertex collision).
        - Prevents head-on position swaps between pairs of drones (edge collision).
        - Prevents moving into a cell occupied by a stationary or yielding drone.
        - Lowest drone_id holds highest priority; yielding drone stays in place.
        Returns (actual_moves, collision_events_prevented).
        """
        resolved_moves = dict(proposed_moves)
        current_pos = {d_id: self.drones[d_id].pos for d_id in self.drones}
        collision_events = 0
        sorted_ids = sorted(self.drones.keys())  # Lower ID = higher priority

        # Multi-pass conflict resolution until fully stable
        while True:
            changed = False

            # 1. Edge swap collisions (A -> B's pos and B -> A's pos)
            for i in range(len(sorted_ids)):
                id_a = sorted_ids[i]
                for j in range(i + 1, len(sorted_ids)):
                    id_b = sorted_ids[j]
                    dest_a = resolved_moves[id_a]
                    dest_b = resolved_moves[id_b]
                    pos_a = current_pos[id_a]
                    pos_b = current_pos[id_b]

                    if dest_a == pos_b and dest_b == pos_a and pos_a != pos_b:
                        # id_b yields (since id_a < id_b)
                        if resolved_moves[id_b] != pos_b:
                            resolved_moves[id_b] = pos_b
                            collision_events += 1
                            changed = True

            # 2. Vertex collisions (multiple drones proposing same destination)
            dest_to_ids: Dict[Tuple[int, int], List[int]] = {}
            for d_id in sorted_ids:
                d = resolved_moves[d_id]
                dest_to_ids.setdefault(d, []).append(d_id)

            for dest, cand_ids in dest_to_ids.items():
                if len(cand_ids) > 1:
                    winner = min(cand_ids)
                    for yielder in cand_ids:
                        if yielder != winner and resolved_moves[yielder] != current_pos[yielder]:
                            resolved_moves[yielder] = current_pos[yielder]
                            collision_events += 1
                            changed = True

            # 3. Collision with stationary or yielding drone
            stationary_cells = {
                current_pos[d_id]
                for d_id in sorted_ids
                if resolved_moves[d_id] == current_pos[d_id]
            }

            for d_id in sorted_ids:
                if resolved_moves[d_id] != current_pos[d_id]:
                    if resolved_moves[d_id] in stationary_cells:
                        resolved_moves[d_id] = current_pos[d_id]
                        collision_events += 1
                        changed = True

            if not changed:
                break

        # Count actual physical collisions (should be 0 with proper arbitration)
        actual_collisions = 0
        final_dest_counts: Dict[Tuple[int, int], List[int]] = {}
        for d_id, dest in resolved_moves.items():
            final_dest_counts.setdefault(dest, []).append(d_id)
        for dest, cands in final_dest_counts.items():
            if len(cands) > 1:
                actual_collisions += (len(cands) - 1)

        return resolved_moves, actual_collisions

    def _record_snapshot(self, tick: int) -> None:
        """Capture lightweight simulation state for post-run animation and visualization."""
        drone_states = []
        for d_id, d in self.drones.items():
            drone_states.append({
                "id": d_id,
                "pos": d.pos,
                "status": d.status,
                "battery": d.battery,
                "target": d.target
            })

        # Union pheromone for visualization
        merged_phero = np.zeros((self.config.grid_height, self.config.grid_width), dtype=np.float32)
        for d in self.drones.values():
            if d.is_operational():
                merged_phero = np.maximum(merged_phero, d.pheromone.grid)

        self.history.append({
            "tick": tick,
            "drones": drone_states,
            "discovered_survivors": set(self.world.discovered_survivors),
            "total_survivors": len(self.world.survivors),
            "obstacles": set(self.world.obstacles),
            "pheromone": merged_phero,
            "in_dropout": self.comms.is_in_dropout(tick)
        })

    def run(self, baseline_metrics: Optional[SwarmMetrics] = None) -> SwarmMetrics:
        """Run full simulation until max_ticks or all survivors found."""
        for _ in range(self.config.max_ticks):
            self.step()

        base_cov = baseline_metrics.coverage_fraction if baseline_metrics else None
        base_surv = baseline_metrics.survivors_found_fraction if baseline_metrics else None
        return self.metrics_tracker.calculate_metrics(
            final_tick=self.current_tick,
            baseline_coverage=base_cov,
            baseline_survivors=base_surv
        )
