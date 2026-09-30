"""
Autonomous Drone agent module.
Maintains individual decentralized local belief, stigmergic pheromone map,
A* path planning, auction bidding, and peer-to-peer gossip synchronisation.
"""

import time
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from auction import AuctionManager
from comms import GossipMessage
from config import SwarmConfig
from pheromone import PheromoneMap
from planner import (
    AStarPlanner,
    BELIEF_FREE,
    BELIEF_OBSTACLE,
    BELIEF_SURVIVOR,
    BELIEF_UNKNOWN,
)


class DroneAgent:
    """Fully decentralized search-and-rescue UAV agent."""

    def __init__(self, drone_id: int, config: SwarmConfig):
        self.drone_id = drone_id
        self.config = config
        self.pos: Tuple[int, int] = config.base_station
        self.battery: float = config.initial_battery
        self.status: str = "ACTIVE"  # ACTIVE, FAILED, DEPLETED

        # Local belief grid: -1 = unknown, 0 = free, 1 = obstacle, 2 = survivor
        self.belief_grid = np.full(
            (config.grid_height, config.grid_width), BELIEF_UNKNOWN, dtype=np.int8
        )
        # Base station is known free
        bx, by = config.base_station
        self.belief_grid[by, bx] = BELIEF_FREE

        self.known_obstacles: Set[Tuple[int, int]] = set()
        self.detected_survivors: Set[Tuple[int, int]] = set()

        # Decentralized subsystems
        self.pheromone = PheromoneMap(
            config.grid_width,
            config.grid_height,
            config.evap_rate,
            config.pheromone_deposit
        )
        self.planner = AStarPlanner(
            safety_margin=config.safety_margin,
            max_expansions=config.max_a_star_expansions,
            time_budget_ms=config.tick_compute_budget_ms
        )
        self.auction = AuctionManager(
            drone_id=drone_id,
            w_dist=config.w_dist,
            w_energy=config.w_energy,
            w_cong=config.w_cong,
            w_phero=config.w_phero,
            congestion_radius=config.congestion_radius,
            max_candidates=config.auction_max_frontiers
        )

        # Path & Target
        self.target: Optional[Tuple[int, int]] = None
        self.planned_path: List[Tuple[int, int]] = []

        # Gossip tracking
        self.last_heartbeat: Dict[int, int] = {}
        self.known_neighbor_positions: Dict[int, Tuple[int, int]] = {}
        self.known_neighbor_targets: Dict[int, Tuple[int, int]] = {}
        self.failed_neighbors: Set[int] = set()
        self.untransmitted_deltas: Dict[Tuple[int, int], int] = {}

        # Performance & telemetry
        self.compute_time_last_tick: float = 0.0
        self.total_distance_moved: int = 0
        self.unique_cells_visited: Set[Tuple[int, int]] = {self.pos}
        self.total_moves: int = 0
        self.replan_requested: bool = False

    def is_operational(self) -> bool:
        """Returns True if the drone can compute and navigate."""
        return self.status == "ACTIVE" and self.battery > 0.0

    def fail_drone(self) -> None:
        """Permanently disable this drone upon failure perturbation."""
        self.status = "FAILED"
        self.planned_path.clear()
        self.target = None

    def sense(self, observations: List[Tuple[int, int, int]]) -> None:
        """
        Process local sensing observations into belief map.
        Triggers replanning if new obstacle blocks planned trajectory.
        """
        if not self.is_operational():
            return

        for x, y, state in observations:
            prev_state = self.belief_grid[y, x]
            if prev_state != state:
                self.belief_grid[y, x] = state
                self.untransmitted_deltas[(x, y)] = state

                if state == BELIEF_OBSTACLE:
                    self.known_obstacles.add((x, y))
                    # If this newly spotted obstacle is on planned path, replan
                    if (x, y) in self.planned_path:
                        self.replan_requested = True
                elif state == BELIEF_SURVIVOR:
                    self.detected_survivors.add((x, y))

    def update_pheromone_step(self) -> None:
        """Deposit pheromone on current location and evaporate local grid."""
        if not self.is_operational():
            return
        self.pheromone.step_evaporation()
        self.pheromone.deposit(self.pos[0], self.pos[1])

    def generate_gossip_message(self, current_tick: int) -> Optional[GossipMessage]:
        """Produce outgoing gossip message with heartbeats, deltas, and auction claims."""
        if not self.is_operational():
            return None

        phero_deltas = self.pheromone.extract_sparse_deltas()
        msg = GossipMessage(
            sender_id=self.drone_id,
            tick=current_tick,
            pos=self.pos,
            battery=self.battery,
            status=self.status,
            target=self.target,
            map_deltas=dict(self.untransmitted_deltas),
            phero_deltas=phero_deltas,
            claims=dict(self.auction.claims)
        )
        self.untransmitted_deltas.clear()
        return msg

    def receive_gossip_message(self, msg: GossipMessage, current_tick: int) -> None:
        """
        Merge gossip message from peer drone: updates heartbeats, belief grid,
        pheromone stigmergy, and auction claims.
        """
        if not self.is_operational():
            return

        # 1. Update heartbeat and neighbor state
        self.last_heartbeat[msg.sender_id] = current_tick
        self.known_neighbor_positions[msg.sender_id] = msg.pos
        if msg.target is not None:
            self.known_neighbor_targets[msg.sender_id] = msg.target
        elif msg.sender_id in self.known_neighbor_targets:
            del self.known_neighbor_targets[msg.sender_id]

        if msg.sender_id in self.failed_neighbors:
            self.failed_neighbors.remove(msg.sender_id)

        # 2. Merge map deltas
        for (x, y), state in msg.map_deltas.items():
            prev = self.belief_grid[y, x]
            if prev == BELIEF_UNKNOWN or (prev != state and state == BELIEF_OBSTACLE):
                self.belief_grid[y, x] = state
                if state == BELIEF_OBSTACLE:
                    self.known_obstacles.add((x, y))
                    if (x, y) in self.planned_path:
                        self.replan_requested = True
                elif state == BELIEF_SURVIVOR:
                    self.detected_survivors.add((x, y))

        # 3. Merge pheromone stigmergy
        self.pheromone.merge_remote_deltas(msg.phero_deltas)

        # 4. Merge auction claims and check conflict resolution
        lost_target = self.auction.update_with_neighbor_claims(
            msg.sender_id, msg.claims, current_tick
        )
        if lost_target:
            self.target = None
            self.planned_path.clear()
            self.replan_requested = True

    def check_heartbeats(self, current_tick: int) -> None:
        """
        Inspect heartbeat timeouts. If a neighbor is silent past threshold,
        declare it failed and release its claims back to the exploration pool.
        """
        if not self.is_operational():
            return

        newly_failed = set()
        for nid, last_tick in list(self.last_heartbeat.items()):
            if current_tick - last_tick > self.config.heartbeat_timeout:
                if nid not in self.failed_neighbors:
                    newly_failed.add(nid)
                    self.failed_neighbors.add(nid)
                    last_pos = self.known_neighbor_positions.pop(nid, None)
                    if last_pos is not None:
                        self.known_obstacles.add(last_pos)
                    if nid in self.known_neighbor_targets:
                        del self.known_neighbor_targets[nid]

        if newly_failed:
            self.auction.release_failed_drone_claims(newly_failed)
            # Re-evaluate targets to seize newly freed frontiers
            self.replan_requested = True

    def plan_next_action(self, current_tick: int) -> Tuple[int, int]:
        """
        Decision cycle: select target frontier via auction, plan path via A*,
        and propose next coordinate move within the tick compute budget.
        """
        t_start = time.perf_counter()

        if not self.is_operational():
            self.compute_time_last_tick = 0.0
            return self.pos

        # Check if target reached or explored
        if self.target is not None:
            tx, ty = self.target
            # If target reached or no longer adjacent to unknown
            if self.pos == self.target or self.belief_grid[ty, tx] == BELIEF_OBSTACLE:
                self.auction.release_completed_target(self.target)
                self.target = None
                self.planned_path.clear()
                self.replan_requested = True

        # Need target or replan requested
        if self.target is None or self.replan_requested or not self.planned_path:
            neighbor_targets = list(self.known_neighbor_targets.values())
            best_target = self.auction.select_best_target(
                current_pos=self.pos,
                belief_grid=self.belief_grid,
                known_obstacles=self.known_obstacles,
                battery=self.battery,
                neighbor_targets=neighbor_targets,
                local_phero_grid=self.pheromone.grid,
                current_tick=current_tick
            )

            self.target = best_target
            self.replan_requested = False

            if self.target is not None:
                elapsed_so_far = time.perf_counter() - t_start
                rem_sec = max(0.001, (self.config.tick_compute_budget_ms / 1000.0) - elapsed_so_far - 0.001)
                raw_path = self.planner.plan_path(
                    start=self.pos,
                    goal=self.target,
                    belief_grid=self.belief_grid,
                    known_obstacles=self.known_obstacles,
                    pheromone_grid=self.pheromone.grid,
                    w_phero=self.config.w_phero,
                    remaining_time_budget_sec=rem_sec
                )
                # Omit current start pos from remaining steps
                if len(raw_path) > 1 and raw_path[0] == self.pos:
                    self.planned_path = raw_path[1:]
                else:
                    self.planned_path = raw_path
            else:
                self.planned_path = []

        # Propose next cell
        if self.planned_path:
            proposed_move = self.planned_path[0]
        else:
            # No frontier available, remain stationary or perform greedy fallback exploration
            proposed_move = self.pos

        self.compute_time_last_tick = (time.perf_counter() - t_start) * 1000.0
        return proposed_move

    def execute_move(self, actual_pos: Tuple[int, int]) -> None:
        """
        Finalize step after collision arbitration has verified or adjusted proposed cell.
        """
        if not self.is_operational():
            return

        moved = (actual_pos != self.pos)
        self.pos = actual_pos
        self.unique_cells_visited.add(self.pos)
        self.total_moves += 1

        if moved:
            self.battery = max(0.0, self.battery - self.config.battery_drain_move)
            self.total_distance_moved += 1
            if self.planned_path and self.planned_path[0] == actual_pos:
                self.planned_path.pop(0)
        else:
            self.battery = max(0.0, self.battery - self.config.battery_drain_idle)

        if self.battery <= 0.0:
            self.status = "DEPLETED"
            self.planned_path.clear()
            self.target = None
