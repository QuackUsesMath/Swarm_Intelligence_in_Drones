"""
Configuration module for the decentralized drone swarm search-and-rescue simulation.
Everything is configurable; nothing is hardcoded.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class SwarmConfig:
    # -------------------------------------------------------------
    # 1. Environment Parameters
    # -------------------------------------------------------------
    grid_width: int = 60
    grid_height: int = 60
    obstacle_density: float = 0.15          # Probability/density of static obstacles
    num_survivors: int = 10                 # Total survivors hidden in free cells
    num_drones: int = 8                     # Number of autonomous drones
    base_station: Tuple[int, int] = (0, 0)  # Starting deployment coordinate (x, y)
    sensing_radius: int = 2                 # Radius within which survivors/obstacles are detected
    comm_range: float = 10.0                # Decentralized gossip communication radius (Euclidean)
    initial_battery: float = 500.0          # Starting energy budget per drone (ticks of life)
    battery_drain_move: float = 1.0         # Battery drain when executing a move
    battery_drain_idle: float = 0.2         # Battery drain when idle or yielding
    tick_compute_budget_ms: float = 25.0    # Hard computational budget per tick per drone (ms)
    max_ticks: int = 200                    # Maximum simulation episode length
    seed: int = 42                          # Master random seed for deterministic reproduction

    # -------------------------------------------------------------
    # 2. Perturbation Schedules (Failures, Comms, Obstacles)
    # -------------------------------------------------------------
    # Drone failure schedule: tick -> list of drone IDs that permanently fail
    failure_schedule: Dict[int, List[int]] = field(default_factory=lambda: {
        40: [2],
        80: [5]
    })

    # Comm dropout time intervals: [(start_tick, end_tick), ...]
    comm_dropouts: List[Tuple[int, int]] = field(default_factory=lambda: [
        (50, 75)
    ])

    # Dynamic obstacles schedule: tick -> list of (x, y) coordinates collapsing mid-mission
    dynamic_obstacles: Dict[int, List[Tuple[int, int]]] = field(default_factory=lambda: {
        60: [(15, 15), (15, 16), (16, 15), (16, 16), (30, 30), (30, 31), (31, 30), (31, 31)]
    })

    # -------------------------------------------------------------
    # 3. Decentralized Swarm Algorithm Parameters (Tunable via PSO)
    # -------------------------------------------------------------
    w_dist: float = 1.0                     # Weight of path distance in frontier auction bid
    w_energy: float = 0.6                   # Weight of battery cost / depletion in bid
    w_cong: float = 1.5                     # Weight of neighbor target congestion / overlap penalty
    w_phero: float = 1.8                    # Repulsion weight from pheromone trail
    evap_rate: float = 0.04                 # Pheromone evaporation rate per tick [0.01, 0.20]
    safety_margin: float = 1.0              # Obstacle inflation distance for A* planning

    # -------------------------------------------------------------
    # 4. Heartbeat, Gossip & Replanning Settings
    # -------------------------------------------------------------
    heartbeat_timeout: int = 6              # Ticks without message before neighbor treated as failed
    auction_max_frontiers: int = 16         # Max frontier candidates evaluated per drone per tick
    max_a_star_expansions: int = 600        # A* node expansion limit before falling back to greedy step
    pheromone_deposit: float = 1.0          # Pheromone units deposited per cell visit
    congestion_radius: float = 6.0          # Radius within which neighbor claims cause bid congestion

    # -------------------------------------------------------------
    # 5. Fitness Function Evaluation Coefficients
    # fitness = a*survivors + b*coverage - c*time - d*collisions - e*energy - f*overlap + g*resilience
    # -------------------------------------------------------------
    fit_a_survivors: float = 100.0          # Reward for fraction of survivors discovered [0, 1]
    fit_b_coverage: float = 50.0            # Reward for fraction of total free space explored [0, 1]
    fit_c_time: float = 0.05                # Penalty coefficient per tick taken to locate all survivors
    fit_d_collisions: float = 80.0          # Severe penalty per collision occurrence
    fit_e_energy: float = 0.005             # Penalty coefficient for total cumulative energy expenditure
    fit_f_overlap: float = 15.0             # Penalty coefficient for redundant path overlaps
    fit_g_resilience: float = 35.0          # Bonus for maintaining coverage under failure vs baseline
