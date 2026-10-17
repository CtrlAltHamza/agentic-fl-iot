def summarize(history, target=0.8):
    reach = next((h.round for h in history if h.accuracy >= target), None)
    return {
        "final_accuracy": history[-1].accuracy,
        "best_accuracy": max(h.accuracy for h in history),
        "rounds_to_target": reach,
        "target_accuracy": target,
        "total_comm_mb": sum(h.comm_mb for h in history),
        "total_energy_j": sum(h.energy_j for h in history),
        "total_sim_time_s": sum(h.round_time_s for h in history),
        "total_dropped_updates": sum(h.n_dropped for h in history),
    }
