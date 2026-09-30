"""
Decentralized gossip communication module.
Facilitates peer-to-peer message passing, supports communication dropouts,
and enforces range limits without any centralized broker.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import numpy as np


@dataclass
class GossipMessage:
    """Message payload exchanged between communicating drones."""
    sender_id: int
    tick: int
    pos: Tuple[int, int]
    battery: float
    status: str
    target: Optional[Tuple[int, int]]
    # Sparse map deltas: (x, y) -> cell_state
    map_deltas: Dict[Tuple[int, int], int]
    # Sparse pheromone deltas: (x, y) -> phero_value
    phero_deltas: Dict[Tuple[int, int], float]
    # Active auction claims
    claims: Dict[Tuple[int, int], Tuple[int, float, int]]


class CommsChannel:
    """Simulates decentralized P2P communication medium with range and dropout constraints."""

    def __init__(self, comm_range: float = 10.0, dropouts: Optional[List[Tuple[int, int]]] = None):
        self.comm_range = comm_range
        self.dropouts = dropouts if dropouts is not None else []

    def is_in_dropout(self, current_tick: int) -> bool:
        """Check whether global communication blackout is active at current tick."""
        for start, end in self.dropouts:
            if start <= current_tick <= end:
                return True
        return False

    def get_communicating_pairs(
        self, drone_positions: Dict[int, Tuple[int, int]], active_ids: Set[int], current_tick: int
    ) -> List[Tuple[int, int]]:
        """
        Determine which pairs of active drones are within communication range,
        subject to dropout blackout.
        """
        if self.is_in_dropout(current_tick):
            return []

        pairs: List[Tuple[int, int]] = []
        r2 = self.comm_range * self.comm_range
        active_list = list(active_ids)
        num_active = len(active_list)

        for i in range(num_active):
            id_a = active_list[i]
            pos_a = drone_positions[id_a]
            for j in range(i + 1, num_active):
                id_b = active_list[j]
                pos_b = drone_positions[id_b]
                dist2 = (pos_a[0] - pos_b[0]) ** 2 + (pos_a[1] - pos_b[1]) ** 2
                if dist2 <= r2:
                    pairs.append((id_a, id_b))

        return pairs
