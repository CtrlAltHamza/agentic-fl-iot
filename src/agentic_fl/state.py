"""Structured contracts shared by the baseline and the future agentic controller."""
from dataclasses import dataclass, field, asdict
from typing import List


@dataclass
class ClientState:
    cid: int
    bandwidth_mbps: float      # uplink bandwidth
    battery: float             # 0..1 remaining charge
    compute_speed: float       # relative speed (1.0 = reference device)
    dropout_prob: float        # chance the device fails in a round
    n_samples: int


@dataclass
class Decision:
    """What a controller decides each round. Baseline = random subset, fixed epochs."""
    selected: List[int]
    local_epochs: int = 1
    compress: bool = False     # placeholder for update compression (agentic phase)


@dataclass
class RoundRecord:
    round: int
    accuracy: float
    loss: float
    comm_mb: float
    energy_j: float
    round_time_s: float
    n_selected: int
    n_dropped: int
    selected: List[int] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)
