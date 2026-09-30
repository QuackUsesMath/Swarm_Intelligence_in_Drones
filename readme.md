# OptiForge: Decentralized Autonomous Drone Swarm for Search-and-Rescue

A complete, fully decentralized multi-agent simulation of an autonomous drone swarm conducting search-and-rescue (SAR) operations in a 2D disaster environment subject to stochastic mid-mission drone failures, communication blackouts, and secondary structural collapses.

---

## 1. System Architecture & File Layout

The codebase is organized into cleanly decoupled, strictly typed, deterministic modules:

```
optiforge/
├── config.py              # Central SwarmConfig dataclass with all parameters and fitness weights
├── world.py               # Ground truth 2D disaster environment, collapses, and sensor queries
├── drone.py               # Autonomous DroneAgent (belief map, stigmergy, decision cycle)
├── comms.py               # Decentralized P2P gossip communication network with dropout simulation
├── planner.py             # Budget-bounded A* search with obstacle inflation and greedy fallback
├── auction.py             # Decentralized auction bidding, conflict resolution, and claim ledger
├── pheromone.py           # Stigmergic pheromone layer (evaporation, deposit, max-merge)
├── metrics.py             # Telemetry tracking, multi-objective fitness, and resilience scoring
├── simulator.py           # SwarmSimulator engine, perturbation dispatch, and priority-yield avoidance
├── tuner.py               # Particle Swarm Optimization (PSO) outer loop with diversity preservation
├── visualization.py       # Matplotlib mission dashboard (4-panel) and animated GIF rendering
├── evaluate.py            # 3-scenario benchmark suite generating metrics comparison tables
├── main.py                # Unified CLI entrypoint for simulation, animation, tuning, and evaluation
├── tests/
│   ├── __init__.py
│   └── test_simulation.py # Unit & integration test suite (zero collisions, compute budget, recovery)
└── README.md              # Architectural rationale, operators, and changelog
```

---

## 2. Theoretical Representation & Core Mechanics

### 2.1 Environmental & State Representation
* **Grid World**: $W \times H$ discrete grid containing static obstacles (collapsed buildings), hidden survivors ($K$), and dynamic collapses.
* **Local Belief Grid**: Each UAV maintains its own private belief tensor $B_i(x, y) \in \{-1 (\text{unknown}), 0 (\text{free}), 1 (\text{obstacle}), 2 (\text{survivor})\}$. No central coordinator or global map exists.
* **Stigmergic Pheromone Field**: Continuous 2D field $P_i(x, y) \ge 0$ deposited along traversed trajectories.
* **Gossip Ledger**: Pairwise asynchronous exchange of observed map deltas, sparse pheromone footprints, heartbeats, and active auction claims.

---

## 3. Swarm Operators & Coordination Rules

### 3.1 Vectorized Frontier-Based Exploration
* Drones extract **frontier cells** (known-free cells $B_i(x, y) = 0$ adjacent to at least one unexplored cell $B_i(nx, ny) = -1$).
* Implemented via fast 2D topological array convolution/boolean masking in NumPy ($<0.1\text{ ms}$ for $60 \times 60$ grids).

### 3.2 Decentralized Auction & Multidimensional Bidding
Each drone evaluates open candidate frontiers and computes a bid cost $J(f)$:
$$J(f) = w_{\text{dist}} \cdot d(p, f) + w_{\text{energy}} \cdot \left(\frac{d(p, f)}{\max(10, E_{\text{battery}})}\right) + w_{\text{cong}} \cdot C(f) + w_{\text{phero}} \cdot P(f)$$
* **Distance $d(p, f)$**: Manhattan/path distance from current location $p$ to candidate frontier $f$.
* **Energy Cost**: Penalizes distant targets when battery reserve $E_{\text{battery}}$ is depleted.
* **Congestion $C(f)$**: Counts active neighbor claims within congestion radius $R_{\text{cong}}$.
* **Pheromone Repulsion $P(f)$**: Stigmergic trail intensity at target, driving dispersion into unvisited regions.
* **Conflict Resolution**: Lowest bid wins: $J_A < J_B$. Ties are broken deterministically by unique drone ID ($\text{ID}_A < \text{ID}_B$). Drones losing an auction immediately yield the target and commit to their next best frontier.

### 3.3 Stigmergic Pheromone Coordination
* **Evaporation**: $P(x, y) \leftarrow P(x, y) \cdot (1 - \lambda_{\text{evap}})$ applied per tick.
* **Deposit**: Traversing a cell deposits $+1.0$ units.
* **Gossip Synchronization**: When in communication range, drones merge pheromone trails using an element-wise maximum operator ($P_A \leftarrow \max(P_A, P_B)$), preserving recent trail memory without destructive overwriting.

### 3.4 Budget-Bounded A* with Greedy Fallback
* Path planning utilizes A* with an **obstacle safety inflation margin** that penalizes trajectories grazing building rubble.
* **Compute Budget Guarantee**: The planner tracks expanded nodes and wall-clock execution time. If execution approaches the per-tick compute budget (default: $25\text{ ms}$), the search terminates and returns a safe greedy gradient descent step toward the target. This guarantees **zero missed simulation ticks**.

### 3.5 Cascading Priority-Based Yield Collision Avoidance
At every simulation tick:
1. **Vertex Conflicts**: Multiple drones proposing the same destination cell.
2. **Edge Swap Conflicts**: Two drones swapping cells head-on ($A \to B$ and $B \to A$).
3. **Stationary Cell Obstructions**: Moving into a cell occupied by a stationary or yielding drone.
* **Resolution**: Lower drone ID holds strict right-of-way. The yielding drone cancels its step and stays in its current cell. Cascading yield checks resolve all dependencies in $O(N^2)$ time, mathematically guaranteeing **zero physical collisions**.

### 3.6 Heartbeat Failure Detection & Dynamic Target Release
* Operational drones broadcast heartbeats containing `(drone_id, tick, pos, target, claims)`.
* If a neighbor remains silent past `heartbeat_timeout` ticks ($t - t_{\text{last}} > \tau$):
  1. The neighbor is marked permanently `FAILED`.
  2. All target claims held by the failed drone are wiped from the consensus ledger.
  3. The abandoned search sector is released back into the open candidate pool for surviving UAVs to claim.

---

## 4. Why Each Technique Was Chosen (Design Rationale)

| Technique | Alternative Considered | Rationale for Selection |
| :--- | :--- | :--- |
| **Decentralized Gossip Protocol** | Central Dispatcher / Cloud Server | Eliminates single points of failure. In disaster environments, central comm hubs collapse; peer-to-peer gossip ensures operational continuity even during comm dropouts. |
| **Market-Based Auction** | Voronoi / Fixed Sector Partitioning | Static sector partitioning fails when drones crash, leaving orphaned zones. Auctioning allows dynamic workload reallocation with deterministic tie-breaking. |
| **Stigmergy (Pheromones)** | Pure Frontier Distance | Pure frontier selection causes UAV clustering and path crossing. Pheromones provide continuous, distributed spatial repulsion with minimal comm bandwidth. |
| **A\* with Greedy Fallback** | Unbounded Dijkstra / RRT\* | Search-and-rescue UAVs operate under strict real-time deadlines. Bounded A\* guarantees that path planning never exceeds hardware compute budgets. |
| **Cascading Priority Yield** | Centralized Space-Time A\* | Distributed right-of-way rules allow local collision avoidance without requiring knowledge of all global agent paths. |
| **PSO Outer Loop** | Grid Search / Manual Heuristics | Complex trade-offs between coverage speed, energy consumption, and collision safety are non-linear; PSO efficiently tunes 6 continuous parameters concurrently. |

---

## 5. Multi-Objective Fitness Function & Resilience Score

$$\text{fitness} = a \cdot S_{\text{frac}} + b \cdot C_{\text{frac}} - c \cdot T_{\text{all}} - d \cdot N_{\text{coll}} - e \cdot E_{\text{tot}} - f \cdot R_{\text{overlap}} + g \cdot \rho_{\text{resilience}}$$

Where:
* $S_{\text{frac}} \in [0, 1]$: Fraction of hidden survivors located.
* $C_{\text{frac}} \in [0, 1]$: Fraction of free map area successfully explored.
* $T_{\text{all}}$: Simulation ticks taken to locate 100% of survivors.
* $N_{\text{coll}}$: Number of physical in-flight collisions ($= 0$).
* $E_{\text{tot}}$: Total battery energy depleted across the swarm.
* $R_{\text{overlap}} \in [0, 1]$: Redundancy ratio of duplicate cell visits ($1 - \frac{\text{unique}}{\text{moves}}$).
* $\rho_{\text{resilience}}$: Resilience score evaluating performance under perturbations against the undisturbed baseline:
$$\rho_{\text{resilience}} = 0.5 \left(\frac{C_{\text{perturbed}}}{C_{\text{baseline}}}\right) + 0.5 \left(\frac{S_{\text{perturbed}}}{S_{\text{baseline}}}\right)$$

---

## 6. Execution Instructions

### Environment Setup
Create a virtual environment and install dependencies:
```bash
# Create virtual environment
python -m venv venv

# Activate on Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Activate on Linux / macOS
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 1. Run Complete Simulation, Benchmarks, & Tuning
```bash
python main.py
```
This performs:
1. Full 200-tick SAR simulation with mid-mission failures, comm dropouts, and dynamic collapses.
2. Generates visual 4-panel dashboard `swarm_mission_dashboard.png`.
3. Renders animated mission execution `swarm_simulation.gif`.
4. Executes the 3-scenario benchmark suite and prints the metrics table.
5. Runs PSO outer-loop optimization with premature convergence detection and outputs `pso_convergence.png`.

### 2. Run Only Scenario Evaluation Benchmarks
```bash
python evaluate.py
```

### 3. Run Automated Test Suite
```bash
python -m unittest tests/test_simulation.py -v
```

---

## 7. "What Changed and Why" Changelog Template

When extending or modifying the simulation engine, use the following standardized changelog format:

```markdown
### [Version / Date] - Title of Change
* **Module Affected**: `planner.py`, `auction.py`, etc.
* **Component**: (e.g., Collision Arbitration, Frontier Extraction, PSO Mutator)
* **What Changed**: Brief, technical description of the exact modification.
* **Why Changed**: Root cause analysis or design motivation (e.g., performance bottleneck, edge case failure, biological inspiration).
* **Verification & Metrics Impact**:
  - Test case added/updated: `test_name`
  - Impact on Coverage / Survivors / Compute Time: (e.g., +4.2% coverage, compute reduced from 41ms to 0.6ms)
  - Collisions: Zero violations maintained.
```
