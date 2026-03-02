import numpy as np


class Diagnostician:
    def compute_metrics(self, trace) -> dict:
        if isinstance(trace, dict):
            summary = trace.get("summary", None)
            if isinstance(summary, dict):
                n_items = float(summary.get("n_items", 0.0))
                count_margin = float(summary.get("count_margin", 0.0))
                sum_margin = float(summary.get("sum_margin", 0.0))
                new_bin_count = float(summary.get("new_bin_count", 0.0))
                sum_candidate_bins = float(summary.get("sum_candidate_bins", 0.0))
                tie_count = float(summary.get("tie_count", 0.0))
                sum_remaining_cap = float(summary.get("sum_remaining_cap", 0.0))
                sumsq_remaining_cap = float(summary.get("sumsq_remaining_cap", 0.0))

                if n_items > 0:
                    mean_remaining = sum_remaining_cap / n_items
                    capacity_variance = (sumsq_remaining_cap / n_items) - (mean_remaining ** 2)
                    if capacity_variance < 0:
                        capacity_variance = 0.0
                    return {
                        "avg_margin": float(sum_margin / max(count_margin, 1.0)),
                        "new_bin_rate": float(new_bin_count / n_items),
                        "avg_candidate_bins": float(sum_candidate_bins / n_items),
                        "capacity_variance": float(capacity_variance),
                        "tie_rate": float(tie_count / n_items),
                    }

            # Backward compatibility with legacy traces if needed.
            per_item = trace.get("per_item", []) or []
            if len(per_item) > 0:
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

        return {
            "avg_margin": 0.0,
            "new_bin_rate": 0.0,
            "avg_candidate_bins": 0.0,
            "capacity_variance": 0.0,
            "tie_rate": 1.0,
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
