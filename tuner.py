"""
Particle Swarm Optimization (PSO) tuner module for decentralized swarm weights.
Tunes distance, energy, congestion, pheromone, evaporation, and safety margin.
Includes diversity checks, premature convergence recovery, and convergence plotting.
"""

from copy import deepcopy
import time
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np

from config import SwarmConfig
from metrics import SwarmMetrics
from simulator import SwarmSimulator


class Particle:
    """Represents a candidate parameter configuration in the PSO swarm."""

    def __init__(self, bounds_min: np.ndarray, bounds_max: np.ndarray, rng: np.random.RandomState):
        self.bounds_min = bounds_min
        self.bounds_max = bounds_max
        self.position = rng.uniform(bounds_min, bounds_max)
        self.velocity = rng.uniform(-(bounds_max - bounds_min) * 0.1, (bounds_max - bounds_min) * 0.1)
        self.best_position = np.copy(self.position)
        self.best_fitness = -float('inf')
        self.current_fitness = -float('inf')

    def apply_bounds(self) -> None:
        """Clamp position and velocity to valid bounds."""
        self.position = np.clip(self.position, self.bounds_min, self.bounds_max)
        max_v = (self.bounds_max - self.bounds_min) * 0.25
        self.velocity = np.clip(self.velocity, -max_v, max_v)

    def to_param_dict(self) -> Dict[str, float]:
        """Convert vector position to parameter dictionary."""
        return {
            "w_dist": float(self.position[0]),
            "w_energy": float(self.position[1]),
            "w_cong": float(self.position[2]),
            "w_phero": float(self.position[3]),
            "evap_rate": float(self.position[4]),
            "safety_margin": float(self.position[5])
        }


class PSOTuner:
    """Outer-loop Particle Swarm Optimizer for tuning multi-agent swarm parameters."""

    # Parameters: [w_dist, w_energy, w_cong, w_phero, evap_rate, safety_margin]
    BOUNDS_MIN = np.array([0.2, 0.0, 0.2, 0.2, 0.01, 0.0], dtype=np.float64)
    BOUNDS_MAX = np.array([4.0, 2.5, 4.0, 4.0, 0.15, 2.0], dtype=np.float64)

    def __init__(
        self,
        base_config: SwarmConfig,
        num_particles: int = 8,
        num_generations: int = 6,
        eval_seeds: List[int] = None,
        diversity_threshold: float = 0.12,
        seed: int = 123
    ):
        self.base_config = base_config
        self.num_particles = num_particles
        self.num_generations = num_generations
        self.eval_seeds = eval_seeds if eval_seeds is not None else [101, 202]
        self.diversity_threshold = diversity_threshold
        self.rng = np.random.RandomState(seed)

        self.particles: List[Particle] = [
            Particle(self.BOUNDS_MIN, self.BOUNDS_MAX, self.rng)
            for _ in range(num_particles)
        ]
        self.global_best_position = np.copy(self.particles[0].position)
        self.global_best_fitness = -float('inf')

        self.best_fitness_history: List[float] = []
        self.avg_fitness_history: List[float] = []
        self.diversity_history: List[float] = []

    def _evaluate_particle(self, particle: Particle) -> float:
        """
        Evaluate particle fitness across multiple seeded scenarios (with failures)
        relative to the undisturbed baseline.
        """
        params = particle.to_param_dict()
        fitness_scores = []

        for s in self.eval_seeds:
            # 1. Undisturbed Baseline run to compute baseline coverage/survivors
            cfg_base = deepcopy(self.base_config)
            cfg_base.seed = s
            cfg_base.failure_schedule = {}
            cfg_base.comm_dropouts = []
            cfg_base.dynamic_obstacles = {}
            # Quick evaluation length for tuning efficiency
            cfg_base.max_ticks = min(120, self.base_config.max_ticks)
            for k, v in params.items():
                setattr(cfg_base, k, v)

            sim_base = SwarmSimulator(cfg_base)
            metrics_base = sim_base.run()

            # 2. Perturbed run with failures, dropouts, and collapses
            cfg_eval = deepcopy(self.base_config)
            cfg_eval.seed = s
            cfg_eval.max_ticks = min(120, self.base_config.max_ticks)
            for k, v in params.items():
                setattr(cfg_eval, k, v)

            sim_eval = SwarmSimulator(cfg_eval)
            metrics_eval = sim_eval.run(baseline_metrics=metrics_base)
            fitness_scores.append(metrics_eval.fitness)

        return float(np.mean(fitness_scores))

    def _calculate_diversity(self) -> float:
        """Compute average positional standard deviation across all particles."""
        positions = np.array([p.position for p in self.particles])
        # Normalized diversity metric
        span = self.BOUNDS_MAX - self.BOUNDS_MIN
        norm_std = np.std(positions, axis=0) / span
        return float(np.mean(norm_std))

    def _restore_diversity(self) -> None:
        """Mutate/reinitialize subpopulation to break out of premature convergence."""
        print("  [!] Population collapse detected. Reinitializing subpopulation for diversity.")
        for p in self.particles:
            if not np.array_equal(p.position, self.global_best_position):
                # 60% chance to reinitialize random position, 40% heavy mutation
                if self.rng.rand() < 0.6:
                    p.position = self.rng.uniform(self.BOUNDS_MIN, self.BOUNDS_MAX)
                else:
                    mutation = self.rng.normal(0, (self.BOUNDS_MAX - self.BOUNDS_MIN) * 0.3)
                    p.position += mutation
                p.velocity = self.rng.uniform(
                    -(self.BOUNDS_MAX - self.BOUNDS_MIN) * 0.1, (self.BOUNDS_MAX - self.BOUNDS_MIN) * 0.1
                )
                p.apply_bounds()

    def optimize(self) -> Tuple[Dict[str, float], float]:
        """
        Execute PSO outer optimization loop across generations.
        Returns (best_parameters_dict, best_fitness).
        """
        w_inertia = 0.72
        c1 = 1.49  # cognitive weight
        c2 = 1.49  # social weight

        print(f"\n==================================================")
        print(f"Starting PSO Optimization ({self.num_particles} particles, {self.num_generations} gens)")
        print(f"==================================================")

        for gen in range(self.num_generations):
            gen_start = time.perf_counter()
            current_fits = []

            for p_idx, particle in enumerate(self.particles):
                fit = self._evaluate_particle(particle)
                particle.current_fitness = fit
                current_fits.append(fit)

                # Update personal best
                if fit > particle.best_fitness:
                    particle.best_fitness = fit
                    particle.best_position = np.copy(particle.position)

                # Update global best
                if fit > self.global_best_fitness:
                    self.global_best_fitness = fit
                    self.global_best_position = np.copy(particle.position)

            best_gen_fit = max(current_fits)
            avg_gen_fit = float(np.mean(current_fits))
            diversity = self._calculate_diversity()

            self.best_fitness_history.append(self.global_best_fitness)
            self.avg_fitness_history.append(avg_gen_fit)
            self.diversity_history.append(diversity)

            elapsed = time.perf_counter() - gen_start
            print(
                f"Gen {gen+1:02d}/{self.num_generations:02d} | "
                f"Best Fit: {self.global_best_fitness:7.2f} | "
                f"Gen Avg: {avg_gen_fit:7.2f} | "
                f"Diversity: {diversity:5.3f} | Time: {elapsed:5.1f}s",
                flush=True
            )

            # Check for premature convergence
            if diversity < self.diversity_threshold and gen < self.num_generations - 1:
                self._restore_diversity()

            # Update particle velocities and positions
            r1 = self.rng.rand(self.num_particles, len(self.BOUNDS_MIN))
            r2 = self.rng.rand(self.num_particles, len(self.BOUNDS_MIN))

            for i, p in enumerate(self.particles):
                cognitive = c1 * r1[i] * (p.best_position - p.position)
                social = c2 * r2[i] * (self.global_best_position - p.position)
                p.velocity = w_inertia * p.velocity + cognitive + social
                p.position = p.position + p.velocity
                p.apply_bounds()

        best_params = {
            "w_dist": float(self.global_best_position[0]),
            "w_energy": float(self.global_best_position[1]),
            "w_cong": float(self.global_best_position[2]),
            "w_phero": float(self.global_best_position[3]),
            "evap_rate": float(self.global_best_position[4]),
            "safety_margin": float(self.global_best_position[5])
        }

        print(f"==================================================")
        print(f"Optimization Complete! Best Fitness: {self.global_best_fitness:.2f}")
        print("Tuned Parameters:")
        for k, v in best_params.items():
            print(f"  - {k:15s}: {v:.4f}")
        print(f"==================================================\n")

        return best_params, self.global_best_fitness

    def plot_convergence(self, output_path: str = "pso_convergence.png") -> None:
        """Generate and save publication-quality convergence plot."""
        generations = list(range(1, len(self.best_fitness_history) + 1))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

        ax1.plot(generations, self.best_fitness_history, marker='o', color='#2b5c8f', label='Global Best Fitness', lw=2)
        ax1.plot(generations, self.avg_fitness_history, marker='s', color='#d95f02', linestyle='--', label='Population Avg Fitness', lw=1.5)
        ax1.set_ylabel("Fitness Score")
        ax1.set_title("PSO Swarm Parameter Tuning Convergence")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2.plot(generations, self.diversity_history, marker='^', color='#7570b3', label='Population Diversity', lw=1.8)
        ax2.axhline(y=self.diversity_threshold, color='red', linestyle=':', label='Diversity Threshold')
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Diversity Index")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
        print(f"Saved PSO convergence plot to {output_path}")
