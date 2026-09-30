A fun hackathon project that attempts to create a search-&-rescue algorithm using swarm intelligent meta-heuristics (PSO). 

The dataset/world_seed is pre-fixed; Unlike the one in demonstration (which is what we aim to achieve).

**Quick Summary of the Algorithm:**

# OptiForge: Decentralized Drone Swarm for Search-and-Rescue

An autonomous, fully decentralized multi-agent simulation of a drone swarm searching a disaster zone. The system handles drone crashes, communication blackouts, and secondary structural collapses in real time with zero centralized coordinator.

---

## Table of Contents
1. [Quickstart in 60 Seconds](#1-quickstart-in-60-seconds)
2. [How It Works (In Plain English)](#2-how-it-works-in-plain-english)
3. [Interactive Configuration Wizard & Dynamic Terminal Flags](#3-interactive-configuration-wizard--dynamic-terminal-flags)
4. [Mathematical Formulation & Coordination Operators](#4-mathematical-formulation--coordination-operators)
5. [System Architecture & File Layout](#5-system-architecture--file-layout)
6. [Benchmark Evaluation & Resilience Results](#6-benchmark-evaluation--resilience-results)
7. [Running the Automated Test Suite](#7-running-the-automated-test-suite)
8. [Changelog & Security](#8-changelog--security)

---

## 1. Quickstart in 60 Seconds

### Step 1: Set Up Python Virtual Environment
```bash
# Clone or navigate to the repository
cd c:/Users/aniru/Desktop/optiforge

# Create virtual environment
python -m venv venv

# Activate on Windows (PowerShell)
.\venv\Scripts\Activate.ps1
# (Or on Linux / macOS: source venv/bin/activate)

# Install dependencies (only NumPy, Matplotlib, and Pillow)
pip install -r requirements.txt
```

### Step 2: Run Out-of-the-Box
```bash
# Run simulation, generate visual dashboard, animated GIF, benchmarks, and PSO tuning:
python main.py
```

### Step 3: Run with Interactive Terminal Wizard
```bash
# Don't want to edit config.py? Use the terminal wizard!
python main.py --interactive
```

---

## 2. How It Works (In Plain English)

Imagine 8 search-and-rescue drones launched into an earthquake zone with collapsed buildings:

1. **No Boss / No Central Map (Decentralized)**:
   There is no central server giving orders. Each drone only knows what its own onboard camera has seen. When two drones come within radio range ($r \le 10$ cells), they chat (**gossip**) and share map discoveries.

2. **Finding the Edge of the Unknown (Frontier Exploration)**:
   Drones look for "frontiers"—free tiles on their local map that border unexplored fog-of-war.

3. **Who Goes Where? (Auction System)**:
   Instead of drones arguing or colliding over the same target, they hold a local auction. Each drone calculates a bid cost:
   * *How far is it?* (closer is cheaper)
   * *How low is my battery?* (low battery drones pick closer targets)
   * *Are other drones already heading there?* (avoid crowded areas)
   * *Have we flown here recently?* (avoid trodden paths)
   Lowest bid wins. Ties are broken by drone ID.

4. **Digital Scent Trails (Pheromones)**:
   Drones drop a digital pheromone trail as they fly. The trail slowly evaporates over time. Drones are naturally repelled by strong trails, which stops the swarm from bunching up.

5. **Crashing Mid-Mission? (Heartbeat Recovery)**:
   Drones ping each other with a heartbeat. If Drone #2 goes silent for more than 6 ticks, the others know it crashed, mark its last position as rubble, and release all the areas it was exploring so other drones take over.

6. **Zero Crashes Guarantee (Yield Priority)**:
   If two drones are about to enter the same cell or swap cells, the lower ID drone has right-of-way. The other drone yields and waits.

---

## 3. Interactive Configuration Wizard & Dynamic Terminal Flags

You do **not** need to open `config.py` to change parameters. You can configure everything directly from your terminal.

### 3.1 Interactive Terminal Wizard (`--interactive` or `-i`)
Launch a friendly step-by-step terminal prompt:
```bash
python main.py -i
```
Press `<Enter>` on any question to accept the sensible default value.

### 3.2 Granular Terminal Flags
You can override any parameter from the command line:

```bash
# Example: 80x80 grid, 12 drones, 15 survivors, custom weights, custom failures
python main.py --mode simulate \
  --width 80 --height 80 \
  --drones 12 --survivors 15 \
  --w-phero 3.0 --w-cong 2.5 \
  --failures "30:1;60:4,5" \
  --dropouts "45-70"
```

#### Complete Command-Line Flag Reference

| Category | Flag | Default | Description | Example |
| :--- | :--- | :--- | :--- | :--- |
| **Mode** | `--mode` | `all` | `all`, `simulate`, `evaluate`, `tune`, or `animate` | `--mode evaluate` |
| **Wizard** | `-i`, `--interactive` | `False` | Launch guided configuration wizard | `python main.py -i` |
| **Config File** | `-c`, `--config` | `None` | Load all parameters from a JSON file | `--config my_run.json` |
| **Export** | `--export-config` | `None` | Export current settings to JSON and exit | `--export-config preset.json` |
| **World Size** | `--width`, `--height` | `60`, `60` | Grid dimensions in cells | `--width 75 --height 75` |
| **Drones** | `--drones` | `8` | Number of autonomous UAVs | `--drones 10` |
| **Survivors** | `--survivors` | `10` | Number of hidden victims to find | `--survivors 15` |
| **Obstacles** | `--density` | `0.15` | Rubble density ($0.0$ to $0.5$) | `--density 0.20` |
| **Ticks & Seed** | `--ticks`, `--seed` | `200`, `42` | Max episode steps and random seed | `--ticks 150 --seed 99` |
| **Comms Range** | `--comm-range` | `10.0` | Radio range in cells | `--comm-range 15.0` |
| **Failures** | `--failures` | `40:2;80:5` | Scheduled drone failures (`tick:id,id`) | `--failures "25:1;50:3,4"` |
| **Blackouts** | `--dropouts` | `50-75` | Radio communication dropouts (`start-end`) | `--dropouts "40-60;100-120"` |
| **Collapses** | `--collapses` | `60:...` | Dynamic building collapses mid-mission | `--collapses "50:15,15,2"` |
| **Weights** | `--w-dist`, `--w-energy` | `1.0`, `0.6` | Auction weights: distance and battery | `--w-dist 1.5` |
| **Weights** | `--w-cong`, `--w-phero` | `1.5`, `1.8` | Auction weights: congestion and pheromones | `--w-phero 2.5` |
| **Evaporation** | `--evap-rate` | `0.04` | Pheromone trail decay rate per tick | `--evap-rate 0.08` |
| **Safety Margin**| `--safety-margin` | `1.0` | Obstacle buffer distance for A* planning | `--safety-margin 1.5` |

---

## 4. Mathematical Formulation & Coordination Operators

### 4.1 Auction Bid Cost Formula
Each drone scores every open candidate frontier $f$:
$$\text{Bid}(f) = w_{\text{dist}} \cdot d(p, f) + w_{\text{energy}} \cdot \left(\frac{d(p, f)}{\max(10, E_{\text{battery}})} \times 20\right) + w_{\text{cong}} \cdot C(f) + w_{\text{phero}} \cdot P(f)$$

* $d(p, f)$: Manhattan distance from drone position $p$ to frontier cell $f$.
* $E_{\text{battery}}$: Remaining battery units (higher penalty as charge depletes).
* $C(f)$: Number of neighbor claims within congestion radius ($r \le 6$).
* $P(f)$: Local pheromone intensity at the candidate location.

### 4.2 Multi-Objective Fitness Function
The swarm's performance across an episode is evaluated as:
$$\text{Fitness} = a \cdot S_{\text{frac}} + b \cdot C_{\text{frac}} - c \cdot T_{\text{all}} - d \cdot N_{\text{coll}} - e \cdot E_{\text{tot}} - f \cdot R_{\text{overlap}} + g \cdot \rho_{\text{resilience}}$$

Where:
* $S_{\text{frac}} \in [0, 1]$: Fraction of survivors found ($a = 100.0$).
* $C_{\text{frac}} \in [0, 1]$: Fraction of free map explored ($b = 50.0$).
* $T_{\text{all}}$: Ticks taken to locate all survivors ($c = 0.05$).
* $N_{\text{coll}}$: Number of in-flight collisions ($d = 80.0$, always $0$).
* $E_{\text{tot}}$: Total battery units spent across all UAVs ($e = 0.005$).
* $R_{\text{overlap}} \in [0, 1]$: Path overlap ratio ($f = 15.0$).
* $\rho_{\text{resilience}}$: Resilience score relative to undisturbed baseline ($g = 35.0$).

$$\rho_{\text{resilience}} = 0.5 \left(\frac{C_{\text{perturbed}}}{C_{\text{baseline}}}\right) + 0.5 \left(\frac{S_{\text{perturbed}}}{S_{\text{baseline}}}\right)$$

---

## 5. System Architecture & File Layout

```
optiforge/
├── config.py              # SwarmConfig dataclass, JSON loader/saver, schedule parsers
├── world.py               # 2D environment, static rubble, collapses, sensor queries
├── drone.py               # Autonomous DroneAgent (belief, stigmergy, decision cycle)
├── comms.py               # Decentralized P2P gossip network & dropout simulation
├── planner.py             # Bounded A* search with greedy fallback (zero missed ticks)
├── auction.py             # Vectorized frontier detection & auction bidding ledger
├── pheromone.py           # Digital pheromone diffusion, deposit, and max-merge
├── metrics.py             # Telemetry tracking, fitness, and resilience scoring
├── simulator.py           # SwarmSimulator engine & cascading priority yield arbitration
├── tuner.py               # Particle Swarm Optimization (PSO) with diversity checks
├── visualization.py       # 4-panel dashboard and animated GIF generator
├── evaluate.py            # 3-scenario benchmark suite
├── main.py                # Unified CLI entrypoint with argument groups & wizard
├── changelog.md           # Full history of changes and bug fixes
├── security.md            # Security policy and multi-agent robustness model
├── requirements.txt       # Python dependencies (NumPy, Matplotlib, Pillow)
├── .env.example           # Example environment template
└── tests/
    ├── test_area_division.py # Frontier detection & dispersion tests
    ├── test_collision.py     # Vertex, edge swap, and yield tests
    ├── test_comm_loss.py     # Radio blackout & reconnection tests
    ├── test_failure.py       # Heartbeat detection & recovery tests
    ├── test_metrics.py       # Fitness & resilience formula tests
    └── test_simulation.py    # Integration tests (budget, collisions, recovery)
```

---

## 6. Benchmark Evaluation & Resilience Results

Run the benchmark suite:
```bash
python evaluate.py
```

### Benchmark Results (200 Ticks, Seed 42)

| Metric | Scenario 1: Baseline | Scenario 2: Drone Failures | Scenario 3: Stress Test (Failures + Blackouts + Collapses) |
| :--- | :---: | :---: | :---: |
| **Total Ticks** | 200 | 200 | 200 |
| **Survivors Located** | 6 / 10 (60%) | 6 / 10 (60%) | 6 / 10 (60%) |
| **Map Coverage** | 46.15% | 39.91% | 36.11% |
| **In-Flight Collisions** | **0** | **0** | **0** |
| **Swarm Energy Spent** | 1596.8 units | 1216.0 units | 1172.8 units |
| **Resilience Score** | 1.000 | **0.932** (93.2%) | **0.891** (89.1%) |
| **Multi-Objective Fitness** | 95.12 | 89.61 | 85.88 |
| **Average Tick Compute** | 0.639 ms | 0.618 ms | 0.560 ms |
| **Peak Tick Compute** | 24.32 ms | 13.75 ms | 7.72 ms (strictly under 25ms budget) |

---

## 7. Running the Automated Test Suite

OptiForge includes 28 comprehensive unit and integration tests covering all swarm mechanics:

```bash
# Run all tests across all test suites
python -m unittest discover tests -v
```

All 28 tests pass:
* `test_area_division`: Frontier extraction, auction uniqueness, and swarm dispersion.
* `test_collision`: Vertex conflicts, edge swap prevention, stationary drone cells, cascading yields.
* `test_comm_loss`: Blackout window enforcement, range limits, gossip sync.
* `test_failure`: Scheduled crashes, heartbeat timeouts, claim release, battery depletion.
* `test_metrics`: Coverage, survivor recovery, overlap, resilience, and fitness equations.
* `test_simulation`: Real-time budget compliance, zero collisions, post-failure recovery.

---

## 8. Changelog & Security

* **Changelog**: See [`changelog.md`](file:///c:/Users/aniru/Desktop/optiforge/changelog.md) for a record of all features, optimizations, and bug fixes.
* **Security Policy**: See [`security.md`](file:///c:/Users/aniru/Desktop/optiforge/security.md) for details on the offline execution boundary, multi-agent defenses, and safe JSON serialization.
