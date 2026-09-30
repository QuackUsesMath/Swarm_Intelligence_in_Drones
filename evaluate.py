"""
Evaluation script running the three disaster scenarios:
1. Baseline (undisturbed)
2. Drone Failures mid-mission
3. Failures + Communication Dropouts + Dynamic Building Collapses

Computes metrics and outputs a comprehensive comparison table.
"""

from copy import deepcopy
from typing import Dict, Tuple
from config import SwarmConfig
from metrics import SwarmMetrics
from simulator import SwarmSimulator


def run_evaluation_suite(base_seed: int = 42) -> Dict[str, SwarmMetrics]:
    """Execute the three benchmark scenarios and return evaluated metrics."""
    results: Dict[str, SwarmMetrics] = {}

    print("\n" + "=" * 80)
    print("RUNNING MULTI-AGENT SWARM EVALUATION BENCHMARK")
    print("=" * 80)

    # -------------------------------------------------------------
    # Scenario 1: Baseline (Clean Exploration, Zero Perturbations)
    # -------------------------------------------------------------
    print("\n>>> Scenario 1: Baseline (Zero Failures, Full Comms, Static Obstacles)...")
    cfg_base = SwarmConfig(seed=base_seed)
    cfg_base.failure_schedule = {}
    cfg_base.comm_dropouts = []
    cfg_base.dynamic_obstacles = {}

    sim_base = SwarmSimulator(cfg_base)
    metrics_base = sim_base.run()
    results["Baseline"] = metrics_base
    print(f"    Done: Coverage={metrics_base.coverage_fraction*100:.1f}%, "
          f"Survivors={metrics_base.survivors_found}/{metrics_base.total_survivors}, "
          f"Fitness={metrics_base.fitness:.2f}")

    # -------------------------------------------------------------
    # Scenario 2: Mid-Mission Drone Failures
    # -------------------------------------------------------------
    print("\n>>> Scenario 2: Drone Failures (UAV 2 fails at tick 35, UAV 5 at tick 75)...")
    cfg_fail = SwarmConfig(seed=base_seed)
    cfg_fail.failure_schedule = {35: [2], 75: [5]}
    cfg_fail.comm_dropouts = []
    cfg_fail.dynamic_obstacles = {}

    sim_fail = SwarmSimulator(cfg_fail)
    metrics_fail = sim_fail.run(baseline_metrics=metrics_base)
    results["Drone Failures"] = metrics_fail
    print(f"    Done: Coverage={metrics_fail.coverage_fraction*100:.1f}%, "
          f"Survivors={metrics_fail.survivors_found}/{metrics_fail.total_survivors}, "
          f"Resilience={metrics_fail.resilience_score:.3f}, Fitness={metrics_fail.fitness:.2f}")

    # -------------------------------------------------------------
    # Scenario 3: Full Stress Test (Failures + Comm Dropout + Dynamic Collapse)
    # -------------------------------------------------------------
    print("\n>>> Scenario 3: Stress Test (Failures + Comms Blackout [45-70] + Dynamic Collapse)...")
    cfg_stress = SwarmConfig(seed=base_seed)
    cfg_stress.failure_schedule = {35: [2], 75: [5]}
    cfg_stress.comm_dropouts = [(45, 70)]
    cfg_stress.dynamic_obstacles = {
        55: [(15, 15), (15, 16), (16, 15), (16, 16), (30, 30), (30, 31), (31, 30), (31, 31)]
    }

    sim_stress = SwarmSimulator(cfg_stress)
    metrics_stress = sim_stress.run(baseline_metrics=metrics_base)
    results["Failures + Comms + Collapses"] = metrics_stress
    print(f"    Done: Coverage={metrics_stress.coverage_fraction*100:.1f}%, "
          f"Survivors={metrics_stress.survivors_found}/{metrics_stress.total_survivors}, "
          f"Resilience={metrics_stress.resilience_score:.3f}, Fitness={metrics_stress.fitness:.2f}")

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS COMPARISON TABLE")
    print("=" * 80)
    print_metrics_table(results)

    return results


def print_metrics_table(results: Dict[str, SwarmMetrics]) -> None:
    """Print ASCII comparison table of evaluation results across scenarios."""
    scenarios = list(results.keys())
    col_width = 32
    header = f"{'Metric':<32}" + "".join([f"{s:>{col_width}}" for s in scenarios])
    divider = "-" * len(header)

    rows = [
        ("Total Mission Ticks", lambda m: f"{m.total_ticks}"),
        ("Survivors Found", lambda m: f"{m.survivors_found}/{m.total_survivors} ({m.survivors_found_fraction*100:.1f}%)"),
        ("Map Coverage Fraction", lambda m: f"{m.coverage_fraction*100:.2f}%"),
        ("Explored Free Cells", lambda m: f"{m.explored_free_cells}/{m.total_free_cells}"),
        ("Time to Find All (ticks)", lambda m: f"{m.time_to_find_all}"),
        ("Collisions Detected", lambda m: f"{m.collisions}"),
        ("Total Swarm Energy", lambda m: f"{m.total_energy_expended:.1f}"),
        ("Path Overlap Ratio", lambda m: f"{m.overlap_ratio:.3f}"),
        ("Resilience Score", lambda m: f"{m.resilience_score:.3f}"),
        ("Multi-Objective Fitness", lambda m: f"{m.fitness:.2f}"),
        ("Peak Tick Compute (ms)", lambda m: f"{m.max_drone_tick_compute_ms:.2f}"),
        ("Avg Tick Compute (ms)", lambda m: f"{m.avg_drone_tick_compute_ms:.3f}"),
    ]

    print(divider)
    print(header)
    print(divider)
    for label, fn in rows:
        row_str = f"{label:<32}" + "".join([f"{fn(results[s]):>{col_width}}" for s in scenarios])
        print(row_str)
    print(divider)
    print()


if __name__ == "__main__":
    run_evaluation_suite(base_seed=42)
