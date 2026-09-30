"""
Configuration module for the decentralized drone swarm search-and-rescue simulation.
Everything is configurable via dataclass, CLI flags, JSON configuration, or interactive wizard.
Nothing is hardcoded.
"""

from dataclasses import asdict, dataclass, field
import json
import os
from typing import Any, Dict, List, Optional, Tuple


def parse_coord(coord_str: str) -> Tuple[int, int]:
    """Parse coordinate string 'x,y' into Tuple[int, int]."""
    parts = coord_str.strip().split(",")
    if len(parts) != 2:
        raise ValueError(f"Invalid coordinate format '{coord_str}'. Expected 'x,y'.")
    return int(parts[0].strip()), int(parts[1].strip())


def parse_failures(failure_str: str) -> Dict[int, List[int]]:
    """
    Parse failure schedule string into Dict[int, List[int]].
    Format: 'tick:drone_id,drone_id;tick:drone_id' or 'none'/''.
    Example: '30:2;60:4,5' -> {30: [2], 60: [4, 5]}
    """
    cleaned = failure_str.strip()
    if not cleaned or cleaned.lower() == "none":
        return {}

    schedule: Dict[int, List[int]] = {}
    entries = cleaned.split(";")
    for entry in entries:
        if not entry.strip():
            continue
        if ":" not in entry:
            raise ValueError(f"Invalid failure entry '{entry}'. Expected 'tick:drone1,drone2'.")
        tick_part, drones_part = entry.split(":", 1)
        tick = int(tick_part.strip())
        drone_ids = [int(d.strip()) for d in drones_part.split(",") if d.strip()]
        schedule[tick] = drone_ids
    return schedule


def parse_dropouts(dropout_str: str) -> List[Tuple[int, int]]:
    """
    Parse comm dropouts string into List[Tuple[int, int]].
    Format: 'start-end;start-end' or 'none'/''.
    Example: '50-75;100-120' -> [(50, 75), (100, 120)]
    """
    cleaned = dropout_str.strip()
    if not cleaned or cleaned.lower() == "none":
        return []

    dropouts: List[Tuple[int, int]] = []
    entries = cleaned.split(";")
    for entry in entries:
        if not entry.strip():
            continue
        if "-" not in entry:
            raise ValueError(f"Invalid dropout entry '{entry}'. Expected 'start-end'.")
        start_part, end_part = entry.split("-", 1)
        dropouts.append((int(start_part.strip()), int(end_part.strip())))
    return dropouts


def parse_collapses(collapse_str: str) -> Dict[int, List[Tuple[int, int]]]:
    """
    Parse dynamic obstacles schedule string into Dict[int, List[Tuple[int, int]]].
    Format: 'tick:x,y;tick:x,y,size' or 'none'/''.
    Example: '60:15,15,2;80:30,30' -> collapses at tick 60 a 2x2 block, and tick 80 at (30,30).
    """
    cleaned = collapse_str.strip()
    if not cleaned or cleaned.lower() == "none":
        return {}

    schedule: Dict[int, List[Tuple[int, int]]] = {}
    entries = cleaned.split(";")
    for entry in entries:
        if not entry.strip():
            continue
        if ":" not in entry:
            raise ValueError(f"Invalid collapse entry '{entry}'. Expected 'tick:x,y' or 'tick:x,y,size'.")
        tick_part, coords_part = entry.split(":", 1)
        tick = int(tick_part.strip())
        schedule.setdefault(tick, [])

        coord_entries = coords_part.split("|")
        for c in coord_entries:
            pieces = [int(p.strip()) for p in c.split(",") if p.strip()]
            if len(pieces) == 2:
                schedule[tick].append((pieces[0], pieces[1]))
            elif len(pieces) == 3:
                cx, cy, size = pieces[0], pieces[1], pieces[2]
                for ox in range(cx, cx + size):
                    for oy in range(cy, cy + size):
                        schedule[tick].append((ox, oy))
            else:
                raise ValueError(f"Invalid collapse coordinates '{c}'. Expected 'x,y' or 'x,y,size'.")
    return schedule


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

    def to_dict(self) -> Dict[str, Any]:
        """Convert SwarmConfig to a JSON-serializable dictionary."""
        d = asdict(self)
        # Ensure dictionary keys that were ints are stringified for JSON compliance
        d["failure_schedule"] = {str(k): list(v) for k, v in self.failure_schedule.items()}
        d["dynamic_obstacles"] = {
            str(k): [list(pt) for pt in v] for k, v in self.dynamic_obstacles.items()
        }
        d["comm_dropouts"] = [list(interval) for interval in self.comm_dropouts]
        d["base_station"] = list(self.base_station)
        return d

    def to_json(self, filepath: Optional[str] = None, indent: int = 2) -> str:
        """Export config to JSON file or return as formatted JSON string."""
        data = self.to_dict()
        json_str = json.dumps(data, indent=indent)
        if filepath is not None:
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(json_str)
        return json_str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SwarmConfig":
        """Instantiate SwarmConfig from dictionary with type safety."""
        clean_data = dict(data)

        # Convert base station back to tuple
        if "base_station" in clean_data and isinstance(clean_data["base_station"], list):
            clean_data["base_station"] = tuple(clean_data["base_station"])

        # Convert failure schedule keys to int
        if "failure_schedule" in clean_data:
            clean_data["failure_schedule"] = {
                int(k): list(v) for k, v in clean_data["failure_schedule"].items()
            }

        # Convert comm dropouts to list of tuples
        if "comm_dropouts" in clean_data:
            clean_data["comm_dropouts"] = [
                tuple(interval) for interval in clean_data["comm_dropouts"]
            ]

        # Convert dynamic obstacles keys to int and coords to tuples
        if "dynamic_obstacles" in clean_data:
            clean_data["dynamic_obstacles"] = {
                int(k): [tuple(pt) for pt in v] for k, v in clean_data["dynamic_obstacles"].items()
            }

        return cls(**clean_data)

    @classmethod
    def from_json(cls, filepath_or_str: str) -> "SwarmConfig":
        """Load SwarmConfig from a JSON file path or a raw JSON string."""
        if os.path.exists(filepath_or_str):
            with open(filepath_or_str, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = json.loads(filepath_or_str)
        return cls.from_dict(data)
