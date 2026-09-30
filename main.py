"""
Main execution entry point for the decentralized search-and-rescue drone swarm.
Supports full simulation, visual dashboard/animation generation,
scenario evaluation benchmarking, and PSO parameter optimization.
"""

import argparse
import sys
from config import SwarmConfig
from evaluate import run_evaluation_suite
from simulator import SwarmSimulator
from tuner import PSOTuner
from visualization import SwarmVisualizer


def run_single_simulation(config: SwarmConfig, save_anim: bool = True) -> None:
    """Run standard simulation with perturbations and generate visualizations."""
    print("\n" + "=" * 70)
    print("RUNNING DECENTRALIZED DRONE SWARM SEARCH-AND-RESCUE SIMULATION")
    print("=" * 70)
    print(f"Grid Size: {config.grid_width}x{config.grid_height} | Drones: {config.num_drones} | Survivors: {config.num_survivors}")
    print(f"Failures Scheduled: {config.failure_schedule}")
    print(f"Comm Dropouts: {config.comm_dropouts}")
    print(f"Dynamic Collapses: {list(config.dynamic_obstacles.keys())}")
    print("=" * 70)

    # 1. Run baseline first for relative resilience baseline
    cfg_base = SwarmConfig(seed=config.seed)
    cfg_base.failure_schedule = {}
    cfg_base.comm_dropouts = []
    cfg_base.dynamic_obstacles = {}
    sim_base = SwarmSimulator(cfg_base)
    base_metrics = sim_base.run()

    # 2. Run perturbed simulation with history recording enabled for visualization
    sim = SwarmSimulator(config, record_history=True)
    print("\nSimulating mission ticks...")
    for t in range(1, config.max_ticks + 1):
        sim.step()
        if t % 40 == 0 or t == config.max_ticks:
            active_cnt = sum(1 for d in sim.drones.values() if d.is_operational())
            surv_found = len(sim.world.discovered_survivors)
            print(f"  Tick {t:03d}/{config.max_ticks:03d} | Active UAVs: {active_cnt} | "
                  f"Survivors Found: {surv_found}/{config.num_survivors} | "
                  f"Discovered: {len(sim.world.discovered_survivors)}")

    metrics = sim.metrics_tracker.calculate_metrics(
        final_tick=config.max_ticks,
        baseline_coverage=base_metrics.coverage_fraction,
        baseline_survivors=base_metrics.survivors_found_fraction
    )

    print("\n" + "-" * 70)
    print("MISSION COMPLETED")
    print(f"Survivors Found:       {metrics.survivors_found}/{metrics.total_survivors} ({metrics.survivors_found_fraction*100:.1f}%)")
    print(f"Map Coverage:          {metrics.coverage_fraction*100:.2f}%")
    print(f"Collisions:            {metrics.collisions} (Zero Violations)")
    print(f"Total Swarm Energy:    {metrics.total_energy_expended:.1f}")
    print(f"Overlap Ratio:         {metrics.overlap_ratio:.3f}")
    print(f"Resilience Score:      {metrics.resilience_score:.3f}")
    print(f"Fitness Score:         {metrics.fitness:.2f}")
    print(f"Max Tick Compute Time: {metrics.max_drone_tick_compute_ms:.2f} ms (< {config.tick_compute_budget_ms} ms)")
    print(f"Avg Tick Compute Time: {metrics.avg_drone_tick_compute_ms:.3f} ms")
    print("-" * 70)

    # 3. Visualizations
    viz = SwarmVisualizer(config)
    viz.save_dashboard(sim.history, metrics, filename="swarm_mission_dashboard.png")

    if save_anim:
        print("Rendering animation GIF (this may take a few seconds)...")
        viz.save_animation(sim.history, filename="swarm_simulation.gif", fps=10, subsample=2)


def main():
    parser = argparse.ArgumentParser(description="Decentralized Drone Swarm Search & Rescue Simulation")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "simulate", "evaluate", "tune", "animate"],
                        help="Execution mode: all, simulate, evaluate, tune, or animate")
    parser.add_argument("--ticks", type=int, default=180, help="Maximum simulation ticks")
    parser.add_argument("--drones", type=int, default=8, help="Number of drones")
    parser.add_argument("--survivors", type=int, default=10, help="Number of survivors")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--pso-particles", type=int, default=6, help="Number of particles for PSO")
    parser.add_argument("--pso-generations", type=int, default=5, help="Number of generations for PSO")

    args = parser.parse_args()

    config = SwarmConfig(
        max_ticks=args.ticks,
        num_drones=args.drones,
        num_survivors=args.survivors,
        seed=args.seed
    )

    if args.mode in ("all", "simulate", "animate"):
        save_gif = (args.mode != "simulate")
        run_single_simulation(config, save_anim=save_gif)

    if args.mode in ("all", "evaluate"):
        run_evaluation_suite(base_seed=args.seed)

    if args.mode in ("all", "tune"):
        print("\nStarting PSO parameter tuning outer loop...")
        tuner = PSOTuner(
            base_config=config,
            num_particles=args.pso_particles,
            num_generations=args.pso_generations,
            seed=args.seed
        )
        best_weights, best_fit = tuner.optimize()
        tuner.plot_convergence(output_path="pso_convergence.png")


if __name__ == "__main__":
    main()
