"""
Metrics tracking and fitness function calculation for search and rescue swarm.
Evaluates coverage, survivor recovery, collisions, energy, overlap, and resilience.
"""

from dataclasses import dataclass
from typing import Optional, Set, Tuple
from config import SwarmConfig


@dataclass
class SwarmMetrics:
    """Consolidated metrics for a single simulation episode."""
    total_ticks: int
    survivors_found: int
    total_survivors: int
    survivors_found_fraction: float
    total_free_cells: int
    explored_free_cells: int
    coverage_fraction: float
    time_to_find_all: int
    collisions: int
    total_energy_expended: float
    overlap_ratio: float
    resilience_score: float
    fitness: float
    max_drone_tick_compute_ms: float
    avg_drone_tick_compute_ms: float


class MetricsTracker:
    """Tracks global telemetry and computes mathematical fitness."""

    def __init__(self, config: SwarmConfig, total_free_cells: int, total_survivors: int):
        self.config = config
        self.total_free_cells = total_free_cells
        self.total_survivors = total_survivors

        self.discovered_survivors: Set[Tuple[int, int]] = set()
        self.explored_cells: Set[Tuple[int, int]] = set()
        self.all_visited_cells: Set[Tuple[int, int]] = set()
        self.total_swarm_moves: int = 0
        self.total_energy_expended: float = 0.0
        self.collision_events: int = 0
        self.time_to_find_all: int = config.max_ticks
        self.all_survivors_found_tick: Optional[int] = None

        self.max_compute_ms: float = 0.0
        self.total_compute_ms: float = 0.0
        self.compute_ticks_count: int = 0

    def record_tick(
        self,
        current_tick: int,
        discovered_survivors_now: Set[Tuple[int, int]],
        explored_cells_now: Set[Tuple[int, int]],
        drone_positions: Set[Tuple[int, int]],
        tick_collisions: int,
        tick_energy: float,
        drone_compute_times_ms: list
    ) -> None:
        """Accumulate per-tick simulation measurements."""
        self.discovered_survivors.update(discovered_survivors_now)
        self.explored_cells.update(explored_cells_now)
        self.all_visited_cells.update(drone_positions)
        self.total_swarm_moves += len(drone_positions)
        self.total_energy_expended += tick_energy
        self.collision_events += tick_collisions

        if len(self.discovered_survivors) >= self.total_survivors and self.all_survivors_found_tick is None:
            self.all_survivors_found_tick = current_tick
            self.time_to_find_all = current_tick

        for t_ms in drone_compute_times_ms:
            if t_ms > self.max_compute_ms:
                self.max_compute_ms = t_ms
            self.total_compute_ms += t_ms
            self.compute_ticks_count += 1

    def calculate_metrics(
        self,
        final_tick: int,
        baseline_coverage: Optional[float] = None,
        baseline_survivors: Optional[float] = None
    ) -> SwarmMetrics:
        """
        Compute consolidated metrics and multi-objective fitness score.
        fitness = a*survivors + b*coverage - c*time - d*collisions - e*energy - f*overlap + g*resilience
        """
        # Fractions
        surv_frac = (
            len(self.discovered_survivors) / max(1, self.total_survivors)
        )
        cov_frac = (
            len(self.explored_cells) / max(1, self.total_free_cells)
        )
        time_score = (
            self.all_survivors_found_tick
            if self.all_survivors_found_tick is not None
            else self.config.max_ticks
        )

        # Overlap ratio: ratio of redundant revisits
        total_unique = len(self.all_visited_cells)
        if self.total_swarm_moves > 0:
            overlap = max(0.0, 1.0 - (total_unique / self.total_swarm_moves))
        else:
            overlap = 0.0

        # Resilience score: ratio of performance relative to undisturbed baseline
        if baseline_coverage is not None and baseline_survivors is not None:
            base_cov = max(0.05, baseline_coverage)
            base_surv = max(0.05, baseline_survivors)
            resilience = 0.5 * (cov_frac / base_cov) + 0.5 * (surv_frac / base_surv)
        else:
            resilience = 1.0

        # Primary fitness formulation
        cfg = self.config
        fitness = (
            cfg.fit_a_survivors * surv_frac
            + cfg.fit_b_coverage * cov_frac
            - cfg.fit_c_time * time_score
            - cfg.fit_d_collisions * self.collision_events
            - cfg.fit_e_energy * self.total_energy_expended
            - cfg.fit_f_overlap * overlap
            + cfg.fit_g_resilience * resilience
        )

        avg_compute = (
            self.total_compute_ms / max(1, self.compute_ticks_count)
        )

        return SwarmMetrics(
            total_ticks=final_tick,
            survivors_found=len(self.discovered_survivors),
            total_survivors=self.total_survivors,
            survivors_found_fraction=surv_frac,
            total_free_cells=self.total_free_cells,
            explored_free_cells=len(self.explored_cells),
            coverage_fraction=cov_frac,
            time_to_find_all=time_score,
            collisions=self.collision_events,
            total_energy_expended=self.total_energy_expended,
            overlap_ratio=overlap,
            resilience_score=resilience,
            fitness=fitness,
            max_drone_tick_compute_ms=self.max_compute_ms,
            avg_drone_tick_compute_ms=avg_compute
        )
