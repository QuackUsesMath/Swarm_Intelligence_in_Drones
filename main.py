"""
Main execution entry point for the decentralized search-and-rescue drone swarm.
Supports full simulation, visual dashboard/animation generation,
scenario evaluation benchmarking, PSO parameter optimization,
JSON configuration importing/exporting, and an interactive terminal wizard.
"""

import argparse
import sys
from typing import Optional
from config import (
    SwarmConfig,
    parse_collapses,
    parse_coord,
    parse_dropouts,
    parse_failures,
)
from evaluate import run_evaluation_suite
from simulator import SwarmSimulator
from tuner import PSOTuner
from visualization import SwarmVisualizer


def run_interactive_wizard(base_config: SwarmConfig) -> SwarmConfig:
    """Guided terminal prompt for dynamically configuring swarm parameters."""
    print("\n" + "=" * 60)
    print("OPTIFORGE: INTERACTIVE SWARM CONFIGURATION WIZARD")
    print("Press [Enter] to accept the default value shown in brackets.")
    print("=" * 60)

    cfg = SwarmConfig()

    def ask(prompt: str, default: any, cast_fn=str):
        user_val = input(f"{prompt} [{default}]: ").strip()
        if not user_val:
            return default
        try:
            return cast_fn(user_val)
        except Exception as e:
            print(f"  Invalid input ({e}), keeping default: {default}")
            return default

    # 1. Environment
    cfg.grid_width = ask("Grid Width", cfg.grid_width, int)
    cfg.grid_height = ask("Grid Height", cfg.grid_height, int)
    cfg.num_drones = ask("Number of Drones", cfg.num_drones, int)
    cfg.num_survivors = ask("Number of Survivors", cfg.num_survivors, int)
    cfg.obstacle_density = ask("Obstacle Density (0.0 - 0.5)", cfg.obstacle_density, float)
    cfg.max_ticks = ask("Max Simulation Ticks", cfg.max_ticks, int)
    cfg.seed = ask("Random Seed", cfg.seed, int)

    # 2. Physics & Comms
    cfg.comm_range = ask("Communication Range (cells)", cfg.comm_range, float)
    cfg.sensing_radius = ask("Sensing Radius (cells)", cfg.sensing_radius, int)
    cfg.initial_battery = ask("Initial Battery (units)", cfg.initial_battery, float)

    # 3. Perturbations
    fail_default = "; ".join(f"{tick}:{','.join(map(str, ids))}" for tick, ids in cfg.failure_schedule.items())
    fail_str = ask("Failure Schedule (e.g. '30:2;60:4,5' or 'none')", fail_default)
    cfg.failure_schedule = parse_failures(fail_str)

    drop_default = "; ".join(f"{s}-{e}" for s, e in cfg.comm_dropouts)
    drop_str = ask("Comm Dropouts (e.g. '50-75;100-120' or 'none')", drop_default)
    cfg.comm_dropouts = parse_dropouts(drop_str)

    # 4. Swarm Weights
    cfg.w_dist = ask("Distance Weight (w_dist)", cfg.w_dist, float)
    cfg.w_energy = ask("Energy Weight (w_energy)", cfg.w_energy, float)
    cfg.w_cong = ask("Congestion Weight (w_cong)", cfg.w_cong, float)
    cfg.w_phero = ask("Pheromone Repulsion Weight (w_phero)", cfg.w_phero, float)
    cfg.evap_rate = ask("Evaporation Rate (0.01 - 0.20)", cfg.evap_rate, float)
    cfg.safety_margin = ask("Safety Margin (cells)", cfg.safety_margin, float)

    print("\nConfiguration synthesized successfully!")
    return cfg


def run_single_simulation(config: SwarmConfig, save_anim: bool = True) -> None:
    """Run standard simulation with perturbations and generate visualizations."""
    print("\n" + "=" * 70)
    print("RUNNING DECENTRALIZED DRONE SWARM SEARCH-AND-RESCUE SIMULATION")
    print("=" * 70)
    print(f"Grid Size: {config.grid_width}x{config.grid_height} | Drones: {config.num_drones} | Survivors: {config.num_survivors}")
    print(f"Failures Scheduled: {config.failure_schedule}")
    print(f"Comm Dropouts: {config.comm_dropouts}")
    print(f"Dynamic Collapses: {list(config.dynamic_obstacles.keys())}")
    print(f"Weights: w_dist={config.w_dist}, w_energy={config.w_energy}, w_cong={config.w_cong}, w_phero={config.w_phero}, evap={config.evap_rate}")
    print("=" * 70)

    # 1. Run baseline first for relative resilience baseline
    cfg_base = SwarmConfig(seed=config.seed)
    cfg_base.failure_schedule = {}
    cfg_base.comm_dropouts = []
    cfg_base.dynamic_obstacles = {}
    cfg_base.grid_width = config.grid_width
    cfg_base.grid_height = config.grid_height
    cfg_base.num_drones = config.num_drones
    cfg_base.num_survivors = config.num_survivors
    cfg_base.obstacle_density = config.obstacle_density
    cfg_base.max_ticks = config.max_ticks
    sim_base = SwarmSimulator(cfg_base, record_history=False)
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
    parser = argparse.ArgumentParser(
        description="Decentralized Drone Swarm Search & Rescue Simulation with Dynamic Parameter Configuration"
    )

    # Mode selection
    parser.add_argument(
        "--mode", type=str, default="all", choices=["all", "simulate", "evaluate", "tune", "animate"],
        help="Execution mode: all, simulate, evaluate, tune, or animate"
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true",
        help="Launch interactive terminal wizard to configure parameters step-by-step"
    )

    # Configuration file options
    parser.add_argument(
        "--config", "-c", type=str, default=None,
        help="Path to JSON configuration file to load parameters from"
    )
    parser.add_argument(
        "--export-config", type=str, default=None,
        help="Export effective configuration to specified JSON file and exit"
    )

    # Environment Group
    grp_env = parser.add_argument_group("Environment Settings")
    grp_env.add_argument("--width", type=int, default=None, help="Grid width (default: 60)")
    grp_env.add_argument("--height", type=int, default=None, help="Grid height (default: 60)")
    grp_env.add_argument("--density", type=float, default=None, help="Obstacle density [0.0, 0.5] (default: 0.15)")
    grp_env.add_argument("--drones", type=int, default=None, help="Number of UAVs (default: 8)")
    grp_env.add_argument("--survivors", type=int, default=None, help="Number of hidden survivors (default: 10)")
    grp_env.add_argument("--base-station", type=str, default=None, help="Base station coords 'x,y' (default: 0,0)")
    grp_env.add_argument("--ticks", type=int, default=None, help="Max mission ticks (default: 200)")
    grp_env.add_argument("--seed", type=int, default=None, help="Master random seed (default: 42)")

    # Physics & Comms Group
    grp_phys = parser.add_argument_group("Physics & Comms")
    grp_phys.add_argument("--sensing-radius", type=int, default=None, help="Sensor radius in cells (default: 2)")
    grp_phys.add_argument("--comm-range", type=float, default=None, help="Gossip communication radius (default: 10.0)")
    grp_phys.add_argument("--battery", type=float, default=None, help="Initial drone battery (default: 500.0)")
    grp_phys.add_argument("--drain-move", type=float, default=None, help="Battery drain per move (default: 1.0)")
    grp_phys.add_argument("--drain-idle", type=float, default=None, help="Battery drain per idle tick (default: 0.2)")
    grp_phys.add_argument("--budget-ms", type=float, default=None, help="Tick compute budget ms (default: 25.0)")

    # Perturbations Group
    grp_pert = parser.add_argument_group("Perturbations")
    grp_pert.add_argument("--failures", type=str, default=None, help="Failure schedule 'tick:id,id;tick:id' or 'none'")
    grp_pert.add_argument("--dropouts", type=str, default=None, help="Comm dropouts 'start-end;start-end' or 'none'")
    grp_pert.add_argument("--collapses", type=str, default=None, help="Collapses 'tick:x,y;tick:x,y,size' or 'none'")

    # Swarm Weights Group
    grp_alg = parser.add_argument_group("Swarm Algorithm Weights")
    grp_alg.add_argument("--w-dist", type=float, default=None, help="Weight of distance in auction bid")
    grp_alg.add_argument("--w-energy", type=float, default=None, help="Weight of battery depletion in bid")
    grp_alg.add_argument("--w-cong", type=float, default=None, help="Weight of neighbor congestion in bid")
    grp_alg.add_argument("--w-phero", type=float, default=None, help="Repulsion weight from pheromone")
    grp_alg.add_argument("--evap-rate", type=float, default=None, help="Pheromone evaporation rate [0.01, 0.20]")
    grp_alg.add_argument("--safety-margin", type=float, default=None, help="Obstacle safety margin in cells")

    # Fitness Coefficients Group
    grp_fit = parser.add_argument_group("Fitness Function Coefficients")
    grp_fit.add_argument("--fit-survivors", type=float, default=None, help="Weight for survivors discovered")
    grp_fit.add_argument("--fit-coverage", type=float, default=None, help="Weight for map coverage fraction")
    grp_fit.add_argument("--fit-time", type=float, default=None, help="Penalty per tick to find all survivors")
    grp_fit.add_argument("--fit-collisions", type=float, default=None, help="Penalty per collision")
    grp_fit.add_argument("--fit-energy", type=float, default=None, help="Penalty for total energy consumed")
    grp_fit.add_argument("--fit-overlap", type=float, default=None, help="Penalty for path overlap ratio")
    grp_fit.add_argument("--fit-resilience", type=float, default=None, help="Bonus for resilience ratio")

    # PSO Tuning Options
    grp_pso = parser.add_argument_group("PSO Outer Loop")
    grp_pso.add_argument("--pso-particles", type=int, default=6, help="Number of particles for PSO")
    grp_pso.add_argument("--pso-generations", type=int, default=5, help="Number of generations for PSO")

    args = parser.parse_args()

    # 1. Load base configuration (from JSON or default)
    if args.config:
        config = SwarmConfig.from_json(args.config)
        print(f"Loaded base configuration from '{args.config}'.")
    else:
        config = SwarmConfig()

    # 2. Interactive Wizard Mode (if requested)
    if args.interactive:
        config = run_interactive_wizard(config)

    # 3. Apply CLI Overrides
    if args.width is not None: config.grid_width = args.width
    if args.height is not None: config.grid_height = args.height
    if args.density is not None: config.obstacle_density = args.density
    if args.drones is not None: config.num_drones = args.drones
    if args.survivors is not None: config.num_survivors = args.survivors
    if args.base_station is not None: config.base_station = parse_coord(args.base_station)
    if args.ticks is not None: config.max_ticks = args.ticks
    if args.seed is not None: config.seed = args.seed

    if args.sensing_radius is not None: config.sensing_radius = args.sensing_radius
    if args.comm_range is not None: config.comm_range = args.comm_range
    if args.battery is not None: config.initial_battery = args.battery
    if args.drain_move is not None: config.battery_drain_move = args.drain_move
    if args.drain_idle is not None: config.battery_drain_idle = args.drain_idle
    if args.budget_ms is not None: config.tick_compute_budget_ms = args.budget_ms

    if args.failures is not None: config.failure_schedule = parse_failures(args.failures)
    if args.dropouts is not None: config.comm_dropouts = parse_dropouts(args.dropouts)
    if args.collapses is not None: config.dynamic_obstacles = parse_collapses(args.collapses)

    if args.w_dist is not None: config.w_dist = args.w_dist
    if args.w_energy is not None: config.w_energy = args.w_energy
    if args.w_cong is not None: config.w_cong = args.w_cong
    if args.w_phero is not None: config.w_phero = args.w_phero
    if args.evap_rate is not None: config.evap_rate = args.evap_rate
    if args.safety_margin is not None: config.safety_margin = args.safety_margin

    if args.fit_survivors is not None: config.fit_a_survivors = args.fit_survivors
    if args.fit_coverage is not None: config.fit_b_coverage = args.fit_coverage
    if args.fit_time is not None: config.fit_c_time = args.fit_time
    if args.fit_collisions is not None: config.fit_d_collisions = args.fit_collisions
    if args.fit_energy is not None: config.fit_e_energy = args.fit_energy
    if args.fit_overlap is not None: config.fit_f_overlap = args.fit_overlap
    if args.fit_resilience is not None: config.fit_g_resilience = args.fit_resilience

    # 4. Handle Export Option
    if args.export_config:
        config.to_json(args.export_config)
        print(f"Exported configuration to '{args.export_config}'. Exiting.")
        sys.exit(0)

    # 5. Dispatch Modes
    if args.mode in ("all", "simulate", "animate"):
        save_gif = (args.mode != "simulate")
        run_single_simulation(config, save_anim=save_gif)

    if args.mode in ("all", "evaluate"):
        run_evaluation_suite(base_seed=config.seed)

    if args.mode in ("all", "tune"):
        print("\nStarting PSO parameter tuning outer loop...")
        tuner = PSOTuner(
            base_config=config,
            num_particles=args.pso_particles,
            num_generations=args.pso_generations,
            seed=config.seed
        )
        best_weights, best_fit = tuner.optimize()
        tuner.plot_convergence(output_path="pso_convergence.png")


if __name__ == "__main__":
    main()
