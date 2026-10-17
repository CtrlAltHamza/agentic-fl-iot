"""Baseline: FedAvg with random client selection and a fixed local-epoch budget."""
import numpy as np
from .env import simulate_client_round
from .model import SoftmaxModel
from .state import Decision, RoundRecord


class RandomSelector:
    """Non-agentic baseline controller (the component the agent will replace)."""
    def __init__(self, k, epochs=1, seed=0):
        self.k, self.epochs, self.rng = k, epochs, np.random.default_rng(seed)

    def decide(self, rnd, clients, history):
        sel = self.rng.choice(len(clients), size=min(self.k, len(clients)), replace=False)
        return Decision(selected=sorted(int(i) for i in sel), local_epochs=self.epochs)


def run_fedavg(Xtr, ytr, Xte, yte, parts, clients, controller, rounds=40, lr=0.1, seed=0):
    n_feat, n_cls = Xtr.shape[1], int(max(ytr.max(), yte.max())) + 1
    glob = SoftmaxModel(n_feat, n_cls, seed)
    rng = np.random.default_rng(seed + 1)
    history = []
    for r in range(1, rounds + 1):
        d = controller.decide(r, clients, history)
        updates, weights, times, energy, comm, dropped = [], [], [], 0.0, 0.0, 0
        for cid in d.selected:
            c = clients[cid]
            drop, t, e, b = simulate_client_round(c, glob.n_bytes(), d.local_epochs, rng)
            energy += e
            times.append(t)
            if drop or len(parts[cid]) == 0:
                dropped += 1
                continue
            local = SoftmaxModel(n_feat, n_cls)
            local.set(glob.get())
            idx = parts[cid]
            local.train_local(Xtr[idx], ytr[idx], d.local_epochs, lr, seed=seed + r + cid)
            updates.append(local.get())
            weights.append(len(idx))
            comm += b
        if updates:
            w = np.array(weights, dtype=np.float32) / sum(weights)
            glob.set([sum(wi * u[k] for wi, u in zip(w, updates)) for k in range(2)])
        loss, acc = glob.loss_acc(Xte, yte)
        history.append(RoundRecord(r, acc, loss, comm / 1e6, energy,
                                   max(times) if times else 0.0,
                                   len(d.selected), dropped, d.selected))
    return glob, history
