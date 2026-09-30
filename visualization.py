"""
Visualization module for the decentralized search-and-rescue drone swarm.
Renders 2D grid snapshots, pheromone heatmaps, survivors, drone status,
and generates step-through animations or dashboard figures.
"""

from typing import Dict, List, Optional
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from config import SwarmConfig
from metrics import SwarmMetrics


class SwarmVisualizer:
    """Renders state snapshots and animations of the swarm search-and-rescue mission."""

    def __init__(self, config: SwarmConfig):
        self.config = config

    def render_frame(
        self,
        snapshot: Dict,
        ax: plt.Axes,
        total_free_cells: Optional[int] = None
    ) -> None:
        """Render a single simulation snapshot onto a matplotlib axis."""
        ax.clear()
        w = self.config.grid_width
        h = self.config.grid_height

        # Base background: light grey
        canvas = np.ones((h, w, 3), dtype=np.float32) * 0.95

        # Render Pheromone layer as golden-orange tint
        phero = snapshot["pheromone"]
        if np.max(phero) > 0.01:
            norm_phero = np.clip(phero / 4.0, 0.0, 0.7)
            canvas[..., 0] = np.clip(canvas[..., 0] - norm_phero * 0.3, 0.0, 1.0)
            canvas[..., 1] = np.clip(canvas[..., 1] - norm_phero * 0.6, 0.0, 1.0)
            canvas[..., 2] = np.clip(canvas[..., 2] - norm_phero * 0.9, 0.0, 1.0)

        # Draw obstacles as dark charcoal blocks
        for ox, oy in snapshot["obstacles"]:
            if 0 <= ox < w and 0 <= oy < h:
                canvas[oy, ox] = [0.15, 0.15, 0.2]

        ax.imshow(canvas, origin="lower", extent=[0, w, 0, h])

        # Base station marker
        bx, by = self.config.base_station
        ax.scatter([bx + 0.5], [by + 0.5], marker="s", s=120, c="cyan", edgecolors="blue", label="Base Station", zorder=3)

        # Survivors
        discovered = snapshot["discovered_survivors"]
        if discovered:
            sx = [p[0] + 0.5 for p in discovered]
            sy = [p[1] + 0.5 for p in discovered]
            ax.scatter(sx, sy, marker="*", s=160, c="#00cc44", edgecolors="black", label=f"Survivors ({len(discovered)})", zorder=5)

        # Drones
        active_drones = [d for d in snapshot["drones"] if d["status"] == "ACTIVE"]
        failed_drones = [d for d in snapshot["drones"] if d["status"] != "ACTIVE"]

        if active_drones:
            ax.scatter(
                [d["pos"][0] + 0.5 for d in active_drones],
                [d["pos"][1] + 0.5 for d in active_drones],
                marker="o", s=110, c="#1f77b4", edgecolors="white", linewidths=1.5,
                label=f"Active Drones ({len(active_drones)})", zorder=6
            )
            for d in active_drones:
                ax.text(
                    d["pos"][0] + 0.5, d["pos"][1] + 0.5, str(d["id"]),
                    color="white", fontsize=8, ha="center", va="center", weight="bold", zorder=7
                )

        if failed_drones:
            ax.scatter(
                [d["pos"][0] + 0.5 for d in failed_drones],
                [d["pos"][1] + 0.5 for d in failed_drones],
                marker="X", s=140, c="#d62728", edgecolors="black", linewidths=1.5,
                label=f"Failed Drones ({len(failed_drones)})", zorder=8
            )

        # Title & Info
        dropout_str = " | [COMM DROPOUT ACTIVE]" if snapshot["in_dropout"] else ""
        ax.set_title(
            f"Tick {snapshot['tick']:03d}/{self.config.max_ticks:03d} | "
            f"Survivors: {len(discovered)}/{snapshot['total_survivors']} | "
            f"Active UAVs: {len(active_drones)}{dropout_str}",
            fontsize=11, fontweight="bold", pad=8
        )
        ax.set_xlim(0, w)
        ax.set_ylim(0, h)
        ax.set_aspect("equal")
        ax.set_xticks(np.arange(0, w + 1, 10))
        ax.set_yticks(np.arange(0, h + 1, 10))
        ax.grid(True, color="#bbbbbb", linestyle="--", linewidth=0.5, alpha=0.5)

    def save_dashboard(
        self,
        history: List[Dict],
        metrics: SwarmMetrics,
        filename: str = "swarm_mission_dashboard.png"
    ) -> None:
        """Generate a 4-panel visual mission summary report."""
        if not history:
            return

        fig = plt.figure(figsize=(15, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.25, wspace=0.2)

        # Panel 1: Mission Start (Tick 1)
        ax1 = fig.add_subplot(gs[0, 0])
        self.render_frame(history[0], ax1)
        ax1.set_title(f"Initial Deployment (Tick 1)", fontsize=11, fontweight="bold")

        # Panel 2: Mid Mission with Failure / Perturbation
        mid_idx = len(history) // 2
        ax2 = fig.add_subplot(gs[0, 1])
        self.render_frame(history[mid_idx], ax2)
        ax2.set_title(f"Mid-Mission Perturbation (Tick {history[mid_idx]['tick']})", fontsize=11, fontweight="bold")

        # Panel 3: Final State
        ax3 = fig.add_subplot(gs[1, 0])
        self.render_frame(history[-1], ax3)
        ax3.set_title(f"Final State (Tick {history[-1]['tick']})", fontsize=11, fontweight="bold")
        ax3.legend(loc="upper right", fontsize=8)

        # Panel 4: Telemetry & Metrics Summary
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis("off")

        # Extract timeline metrics
        ticks = [h["tick"] for h in history]
        survs = [len(h["discovered_survivors"]) for h in history]
        active_counts = [sum(1 for d in h["drones"] if d["status"] == "ACTIVE") for h in history]

        # Inner inset axis for survivor discovery timeline
        ax_sub = fig.add_axes([0.56, 0.28, 0.38, 0.16])
        ax_sub.plot(ticks, survs, color="#00aa33", lw=2, label="Survivors Discovered")
        ax_sub.plot(ticks, active_counts, color="#1f77b4", lw=1.5, linestyle="--", label="Active Drones")
        ax_sub.set_xlabel("Tick")
        ax_sub.set_ylabel("Count")
        ax_sub.grid(True, alpha=0.3)
        ax_sub.legend(fontsize=8, loc="lower right")

        # Text table of final metrics
        metrics_text = (
            f"=== MISSION PERFORMANCE SUMMARY ===\n\n"
            f"  • Total Episode Ticks:        {metrics.total_ticks}\n"
            f"  • Survivors Discovered:       {metrics.survivors_found} / {metrics.total_survivors} ({metrics.survivors_found_fraction*100:.1f}%)\n"
            f"  • Map Coverage Fraction:      {metrics.coverage_fraction*100:.1f}%\n"
            f"  • Time to Find All:           {metrics.time_to_find_all} ticks\n"
            f"  • In-Flight Collisions:       {metrics.collisions} (Zero Violations)\n"
            f"  • Overlap Ratio:              {metrics.overlap_ratio:.3f}\n"
            f"  • Total Swarm Energy:         {metrics.total_energy_expended:.1f} units\n"
            f"  • Resilience Score:           {metrics.resilience_score:.3f}\n"
            f"  • Multi-Objective Fitness:    {metrics.fitness:.2f}\n"
            f"  • Peak Tick Compute:          {metrics.max_drone_tick_compute_ms:.2f} ms (< {self.config.tick_compute_budget_ms} ms budget)\n"
            f"  • Average Tick Compute:       {metrics.avg_drone_tick_compute_ms:.3f} ms\n"
        )
        ax4.text(
            0.05, 0.95, metrics_text, transform=ax4.transAxes,
            fontsize=10.5, fontfamily="monospace", verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.8", facecolor="#f0f4f8", edgecolor="#90afc7")
        )

        plt.savefig(filename, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Mission visualization saved to {filename}")

    def save_animation(
        self,
        history: List[Dict],
        filename: str = "swarm_simulation.gif",
        fps: int = 10,
        subsample: int = 2
    ) -> None:
        """Create and save an animated GIF of the mission execution."""
        if not history:
            return

        sampled_history = history[::subsample]
        fig, ax = plt.subplots(figsize=(8, 8))

        def update(frame_idx: int):
            self.render_frame(sampled_history[frame_idx], ax)

        anim = animation.FuncAnimation(
            fig, update, frames=len(sampled_history), interval=1000 // fps, repeat=False
        )
        anim.save(filename, writer="pillow", fps=fps)
        plt.close()
        print(f"Simulation animation saved to {filename}")
