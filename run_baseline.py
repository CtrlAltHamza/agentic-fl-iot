"""End-to-end baseline: data -> non-IID partition -> IoT env -> FedAvg -> results."""
import argparse, json, os
import numpy as np
from src.agentic_fl import data, env, fedavg, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", type=int, default=30)
    ap.add_argument("--per-round", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--alpha", type=float, default=0.1, help="Dirichlet non-IID level")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--target", type=float, default=0.5)
    ap.add_argument("--dataset", choices=["synthetic", "uci-har"], default="synthetic")
    a = ap.parse_args()

    raw_split = data.load_uci_har_split() if a.dataset == "uci-har" else None
    if a.dataset == "uci-har" and raw_split is None:
        raise SystemExit("UCI-HAR not found in data/raw/. See README.")
    if a.dataset == "synthetic":
        X, y = data.make_synthetic_har()

    runs = []
    for s in range(a.seeds):
        if a.dataset == "uci-har":
            Xtr, ytr, Xte, yte = (arr.copy() for arr in raw_split)
        else:
            Xtr, ytr, Xte, yte = data.train_test_split(X, y, seed=s)
        # Fit preprocessing on training features only; apply unchanged transform to test data.
        mean, std = data.fit_standardizer(Xtr)
        Xtr = data.apply_standardizer(Xtr, mean, std)
        Xte = data.apply_standardizer(Xte, mean, std)
        parts = data.dirichlet_partition(ytr, a.clients, a.alpha, seed=s)
        clients = env.make_clients([len(p) for p in parts], seed=s)
        ctrl = fedavg.RandomSelector(a.per_round, epochs=1, seed=s)
        _, hist = fedavg.run_fedavg(Xtr, ytr, Xte, yte, parts, clients, ctrl, a.rounds, seed=s)
        summ = metrics.summarize(hist, a.target)
        summ["seed"] = s
        runs.append({"summary": summ, "rounds": [h.to_dict() for h in hist]})
        print(f"seed {s}: acc={summ['final_accuracy']:.3f} "
              f"rounds_to_{a.target}={summ['rounds_to_target']} "
              f"comm={summ['total_comm_mb']:.2f}MB energy={summ['total_energy_j']:.0f}J "
              f"dropped={summ['total_dropped_updates']}")

    keys = ["final_accuracy", "total_comm_mb", "total_energy_j", "total_sim_time_s", "total_dropped_updates"]
    agg = {k: {"mean": float(np.mean([r["summary"][k] for r in runs])),
               "std": float(np.std([r["summary"][k] for r in runs]))} for k in keys}
    os.makedirs("results", exist_ok=True)
    with open("results/baseline_results.json", "w") as f:
        json.dump({"config": vars(a), "aggregate": agg, "runs": runs}, f, indent=2)
    print("\nAggregate (mean +- std over seeds):")
    for k, v in agg.items():
        print(f"  {k}: {v['mean']:.3f} +- {v['std']:.3f}")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for r in runs:
            plt.plot([x["accuracy"] for x in r["rounds"]], alpha=0.7)
        plt.xlabel("Round"); plt.ylabel("Test accuracy"); plt.title("FedAvg + random selection (baseline)")
        plt.savefig("results/baseline_accuracy.png", dpi=150, bbox_inches="tight")
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
