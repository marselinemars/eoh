import numpy as np


class Diagnostician:
    def compute_metrics(self, trace) -> dict:
        per_item = []
        if isinstance(trace, dict):
            per_item = trace.get("per_item", []) or []

        if len(per_item) == 0:
            return {
                "avg_margin": 0.0,
                "new_bin_rate": 0.0,
                "avg_candidate_bins": 0.0,
                "capacity_variance": 0.0,
                "tie_rate": 1.0,
            }

        best_scores = np.array([float(x.get("best_score", 0.0)) for x in per_item], dtype=float)
        second_scores = np.array([float(x.get("second_best_score", 0.0)) for x in per_item], dtype=float)
        margins = best_scores - second_scores
        new_bins = np.array([1.0 if bool(x.get("new_bin", False)) else 0.0 for x in per_item], dtype=float)
        candidate_bins = np.array([float(x.get("candidate_bins", 0.0)) for x in per_item], dtype=float)
        capacities = np.array([float(x.get("remaining_capacity", 0.0)) for x in per_item], dtype=float)
        ties = np.array(
            [1.0 if candidate_bins[i] > 1 and abs(margins[i]) <= 1e-12 else 0.0 for i in range(len(margins))],
            dtype=float,
        )

        return {
            "avg_margin": float(np.mean(margins)),
            "new_bin_rate": float(np.mean(new_bins)),
            "avg_candidate_bins": float(np.mean(candidate_bins)),
            "capacity_variance": float(np.var(capacities)),
            "tie_rate": float(np.mean(ties)),
        }

    def diagnose(self, metrics) -> str:
        avg_margin = float(metrics.get("avg_margin", 0.0))
        new_bin_rate = float(metrics.get("new_bin_rate", 0.0))
        capacity_variance = float(metrics.get("capacity_variance", 0.0))
        tie_rate = float(metrics.get("tie_rate", 0.0))

        if avg_margin >= 0.5 and new_bin_rate >= 0.35:
            return "OVER_GREEDY"
        if avg_margin <= 1e-3 or tie_rate >= 0.25:
            return "FLAT_SCORING"
        if capacity_variance >= 200.0:
            return "UNSTABLE_CAPACITY"
        return "EXPLORATION_NEEDED"
