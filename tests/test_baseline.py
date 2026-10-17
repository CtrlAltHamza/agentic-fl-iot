import numpy as np
from src.agentic_fl import data, env, fedavg, metrics


def test_partition_covers_all_samples():
    y = np.random.default_rng(0).integers(0, 6, 1000)
    parts = data.dirichlet_partition(y, 10, 0.3)
    assert sorted(np.concatenate(parts).tolist()) == list(range(1000))


def test_env_energy_positive_and_dropout_flag():
    c = env.make_clients([100], seed=1)[0]
    drop, t, e, b = env.simulate_client_round(c, 4000, 1, np.random.default_rng(0))
    assert e > 0 and t > 0 and (b == 0 if drop else b == 4000)


def test_fedavg_learns_above_chance():
    X, y = data.make_synthetic_har(2000)
    X = data.standardize(X)
    Xtr, ytr, Xte, yte = data.train_test_split(X, y)
    parts = data.dirichlet_partition(ytr, 10, 1.0)
    clients = env.make_clients([len(p) for p in parts])
    _, h = fedavg.run_fedavg(Xtr, ytr, Xte, yte, parts, clients,
                             fedavg.RandomSelector(5), rounds=15)
    assert h[-1].accuracy > 0.3  # chance is ~0.17
    assert metrics.summarize(h)["total_comm_mb"] > 0
