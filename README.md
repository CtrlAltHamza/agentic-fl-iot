# G3-08 — Agentic Federated Edge Learning for Resource-Constrained IoT

**Course:** AI4015 Agentic AI (Fall 2026)  
**Phase 1 status:** Working non-agentic baseline; adaptive agent controller is planned for the next stage.

## Research objective
Build an agentic orchestration layer for federated learning (FL) that selects IoT clients and local-training budgets using changing battery, bandwidth, compute, dropout, data-volume, and recent training-history signals. The project will compare the agentic controller against random selection and resource-aware non-agentic baselines.

## Current baseline
- NumPy softmax-regression classifier.
- FedAvg aggregation with random client selection and fixed local epochs.
- Simulated heterogeneous IoT clients (battery, bandwidth, compute speed, dropout).
- Dirichlet label-skew partitions to simulate non-IID client data.
- Metrics: test accuracy/loss, simulated communication, energy, round time, dropped updates, and rounds-to-target.
- Synthetic HAR-like data is the default for reproducibility. It is **not real sensor data** and results must be labelled synthetic.
- Optional UCI-HAR loader preserves the official train/test split when the dataset is downloaded by the user.

## Repository layout
| Path | Purpose |
|---|---|
| `src/agentic_fl/data.py` | Synthetic data, optional UCI-HAR loader, train/test split, official UCI-HAR split, train-only normalization, non-IID partition |
| `src/agentic_fl/env.py` | Simulated device/network conditions and resource cost model |
| `src/agentic_fl/state.py` | Structured `ClientState`, `Decision`, and `RoundRecord` contracts |
| `src/agentic_fl/model.py` | Lightweight NumPy softmax regression |
| `src/agentic_fl/fedavg.py` | FedAvg loop and random-selection baseline controller |
| `src/agentic_fl/metrics.py` | Summary metrics |
| `run_baseline.py` | Reproducible end-to-end experiment |
| `tests/` | Deterministic smoke tests |
| `results/` | Baseline results and accuracy plot |

## Setup and run
Python 3.10+ recommended.

```bash
python -m pip install -r requirements.txt
python run_baseline.py
python -m pytest -q
```

Optional UCI-HAR:
1. Download the UCI Human Activity Recognition Using Smartphones dataset from its official repository.
2. Place the extracted folder at `data/raw/UCI HAR Dataset/`.
3. Run `python run_baseline.py --dataset uci-har`.

## Controller contract
A controller implements `decide(round, clients, history) -> Decision`.
The supplied baseline uses random selection and fixed local epochs. The proposed agent will consume the structured client state and prior round records, then select a feasible subset and training budget. Safety constraints will be enforced in code rather than delegated to free-form model output.

## Important limitations
- Current results use synthetic data and a simulated resource model.
- Energy, latency, and bandwidth figures are simulator estimates, not measurements from physical IoT hardware.
- The current implementation is **not yet an agentic system** and does not claim LLM/provider integration.
- The current model is a lightweight baseline, not a state-of-the-art deep neural network.
- Before final evaluation, run multiple seeds, report mean ± standard deviation, and compare under identical random seeds and client partitions.

## Planned next-stage additions
1. Implement resource-aware heuristic selection as a non-agentic comparison.
2. Add an orchestrator/agent that reads state and history, calls typed tools, proposes a client set and local epoch budget, and receives evaluation feedback.
3. Enforce hard constraints (client index validity, battery floor, participant budget, action schema).
4. Add ablations: no history, no energy feature, no data-diversity feature, no reflection/replanning.
5. Compare accuracy, energy, communication, wall-clock latency, dropped updates, fairness/participation, and tool-call cost.

## Team
Fill in all group members and roll numbers before submission. Each member should contribute meaningful commits from their own account and understand the full system.
