# Security Policy

## 1. Scope & Execution Boundary

OptiForge is a local, offline simulation framework for decentralized multi-agent search-and-rescue systems. It relies exclusively on standard Python libraries (`dataclasses`, `heapq`, `json`, `argparse`, `unittest`), `numpy`, and `matplotlib`.

* **No Network Listeners**: The gossip communication model is simulated virtually in-memory and does not open network sockets, HTTP servers, or external communication ports.
* **No Arbitrary Code Execution**: Configuration files are strictly parsed through JSON schema deserialization (`json.loads`/`json.load`) into type-checked dataclass fields. No dynamic evaluation (`eval`, `exec`, or `pickle`) is used.
* **Local File Isolation**: All output artifacts (GIFs, PNGs, and JSON exports) are written to validated paths within the local workspace directory.

---

## 2. Multi-Agent Swarm Security & Robustness Model

In decentralized swarm intelligence, agents communicate peer-to-peer without a centralized certifying authority. The architecture implements several defensive patterns:

| Defensive Pattern | Threat Mitigated | Implementation |
| :--- | :--- | :--- |
| **Compute Budget Cutoffs** | Algorithmic Denial-of-Service (DoS) / Infinite Path Loops | A* search enforces strict node expansion limits (`max_a_star_expansions`) and wall-clock execution limits (`time_budget_ms`), falling back to a greedy step if budget is exhausted. |
| **Deterministic Tie-Breaking** | Consensus Deadlocks / Oscillatory Auction Hijacking | Auction bids resolve conflicts by strictly prioritizing lowest bid cost, broken deterministically by immutable drone IDs (`id_A < id_B`). |
| **Cascading Yield Arbitration** | Physical Collisions & Positional Swaps | Iterative resolution ensures vertex, edge-swap, and stationary obstructions are resolved with zero collisions, even under sudden neighbor failure. |
| **Heartbeat Timeout Leases** | Zombie Claims / Orphaned Sectors | If an agent goes silent, its target claims automatically expire after `heartbeat_timeout` ticks, releasing the sector back to surviving drones. |

---

## 3. Reporting a Vulnerability

If you discover a security vulnerability or crash exploit within OptiForge:

1. **Do not create a public issue**.
2. Document the exact command-line invocation, seed, or JSON configuration file that triggers the issue.
3. Submit a report directly to the repository maintainers.
4. Maintainers will review the report, issue a fix, and document the resolution in `changelog.md`.
