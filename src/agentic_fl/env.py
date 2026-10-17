"""IoT environment simulator: heterogeneous bandwidth, battery, speed and failures."""
import numpy as np
from .state import ClientState

TX_POWER_W = 0.8          # radio power while uploading
COMPUTE_POWER_W = 2.0     # CPU power while training
SEC_PER_SAMPLE = 0.002    # reference device time per sample per epoch


def make_clients(sizes, seed=0):
    rng = np.random.default_rng(seed)
    clients = []
    for cid, n in enumerate(sizes):
        clients.append(ClientState(
            cid=cid,
            bandwidth_mbps=float(rng.choice([0.25, 1.0, 5.0], p=[0.3, 0.4, 0.3])),
            battery=float(rng.uniform(0.3, 1.0)),
            compute_speed=float(rng.choice([0.5, 1.0, 2.0], p=[0.3, 0.5, 0.2])),
            dropout_prob=float(rng.choice([0.05, 0.2, 0.4], p=[0.5, 0.3, 0.2])),
            n_samples=int(n)))
    return clients


def simulate_client_round(c: ClientState, model_bytes, epochs, rng):
    """Returns (dropped, time_s, energy_j, bytes_uploaded)."""
    train_t = c.n_samples * epochs * SEC_PER_SAMPLE / c.compute_speed
    up_t = model_bytes * 8 / (c.bandwidth_mbps * 1e6)
    energy = train_t * COMPUTE_POWER_W + up_t * TX_POWER_W
    dropped = rng.random() < c.dropout_prob or c.battery <= 0.05
    c.battery = max(0.0, c.battery - energy / 400.0)   # drain model
    if dropped:
        return True, train_t, train_t * COMPUTE_POWER_W, 0
    return False, train_t + up_t, energy, model_bytes
