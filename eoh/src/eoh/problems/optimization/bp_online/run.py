import os
import sys
import types
import warnings

import numpy as np

from .get_instance import GetData
from .prompts import GetPrompts


class BPONLINE:
    def __init__(self, paras=None):
        getdate = GetData()
        self.instances, self.lb = getdate.get_instances()
        self.prompts = GetPrompts()
        self.eval_instances_per_gen = None
        self.holdout_instances = 64
        self.profile_train_instances = 8
        self.profile_holdout_instances = 8
        if paras is not None:
            self.eval_instances_per_gen = getattr(paras, "eval_instances_per_gen", None)
            self.holdout_instances = int(getattr(paras, "holdout_instances", 64))
            self.profile_train_instances = int(getattr(paras, "planner_profile_train_instances", 8))
            self.profile_holdout_instances = int(getattr(paras, "planner_profile_holdout_instances", 8))

    def get_problem_context(self):
        return (
            "Online bin packing. Items arrive sequentially and must be assigned immediately. "
            "The heuristic scores feasible bins and the highest score wins. Lower fitness is better."
        )

    def _select_train_instances(self, dataset_items, limit=None):
        if limit is None:
            limit = self.eval_instances_per_gen
        if limit is None:
            return dataset_items
        try:
            cap = int(limit)
        except (TypeError, ValueError):
            return dataset_items
        if cap <= 0:
            return dataset_items
        return dataset_items[:cap]

    def _select_holdout_instances(self, dataset_items, limit=None):
        try:
            k = int(self.holdout_instances if limit is None else limit)
        except (TypeError, ValueError):
            k = 64
        if k <= 0:
            return []
        train_items = self._select_train_instances(dataset_items)
        start = len(train_items)
        holdout_items = dataset_items[start : start + k]
        return holdout_items

    def get_valid_bin_indices(self, item: float, bins: np.ndarray) -> np.ndarray:
        return np.nonzero((bins - item) >= 0)[0]

    def _choice_entropy(self, scores):
        if len(scores) <= 1:
            return 0.0
        arr = np.asarray(scores, dtype=float)
        arr = arr - np.max(arr)
        arr = np.clip(arr, -50.0, 50.0)
        probs = np.exp(arr)
        total = float(np.sum(probs))
        if total <= 0.0 or not np.isfinite(total):
            return 0.0
        probs = probs / total
        probs = probs[probs > 1e-12]
        if len(probs) == 0:
            return 0.0
        entropy = -float(np.sum(probs * np.log(probs)))
        return entropy / float(np.log(len(scores)))

    def _score_margin(self, scores):
        if len(scores) <= 1:
            return 0.0
        ranked = np.sort(np.asarray(scores, dtype=float))
        return float(ranked[-1] - ranked[-2])

    def _run_instance_trace(self, items, capacity, alg):
        bins = np.array([capacity for _ in range(len(items))], dtype=float)
        stage_stats = {
            "early": {"opens": 0, "steps": 0},
            "mid": {"opens": 0, "steps": 0},
            "late": {"opens": 0, "steps": 0},
        }
        entropies = []
        extreme_flags = []
        margins = []

        n_items = max(1, len(items))
        for idx, item in enumerate(items):
            valid_bin_indices = self.get_valid_bin_indices(item, bins)
            valid_bins = bins[valid_bin_indices].copy()
            priorities = np.asarray(alg.score(item, valid_bins), dtype=float)
            best_local = int(np.argmax(priorities))
            chosen_bin = int(valid_bin_indices[best_local])

            stage_ratio = idx / float(n_items)
            stage_key = "early" if stage_ratio < (1.0 / 3.0) else "mid" if stage_ratio < (2.0 / 3.0) else "late"
            stage_stats[stage_key]["steps"] += 1
            if valid_bins[best_local] >= capacity:
                stage_stats[stage_key]["opens"] += 1

            entropies.append(self._choice_entropy(priorities))
            margins.append(self._score_margin(priorities))

            order = np.argsort(valid_bins)
            chosen_rank = int(np.where(order == best_local)[0][0]) if len(order) > 0 else 0
            if len(valid_bins) > 1:
                extreme_threshold = max(1, int(np.ceil(0.2 * len(valid_bins))))
                extreme_flags.append(
                    float(chosen_rank < extreme_threshold or chosen_rank >= (len(valid_bins) - extreme_threshold))
                )
            else:
                extreme_flags.append(1.0)

            bins[chosen_bin] -= item

        used_bins = bins[bins != capacity]
        bins_used = int(len(used_bins))
        residual_ratios = (used_bins / float(capacity)) if bins_used > 0 else np.array([0.0])
        fragmentation = float(np.mean((residual_ratios > 0.05) & (residual_ratios < 0.50)))

        return {
            "bins_used": bins_used,
            "mean_residual_ratio": float(np.mean(residual_ratios)),
            "residual_variance": float(np.var(residual_ratios)),
            "fragmentation_index": fragmentation,
            "resource_opening_rate_early": float(stage_stats["early"]["opens"] / max(1, stage_stats["early"]["steps"])),
            "resource_opening_rate_mid": float(stage_stats["mid"]["opens"] / max(1, stage_stats["mid"]["steps"])),
            "resource_opening_rate_late": float(stage_stats["late"]["opens"] / max(1, stage_stats["late"]["steps"])),
            "choice_entropy": float(np.mean(np.asarray(entropies))) if entropies else 0.0,
            "extreme_option_preference": float(np.mean(np.asarray(extreme_flags))) if extreme_flags else 0.0,
            "score_margin_mean": float(np.mean(np.asarray(margins))) if margins else 0.0,
            "score_margin_variance": float(np.var(np.asarray(margins))) if margins else 0.0,
        }

    def _profile_module(self, alg, split="train", limit=None):
        metric_keys = [
            "mean_residual_ratio",
            "residual_variance",
            "fragmentation_index",
            "resource_opening_rate_early",
            "resource_opening_rate_mid",
            "resource_opening_rate_late",
            "choice_entropy",
            "extreme_option_preference",
            "score_margin_mean",
            "score_margin_variance",
        ]
        collected = {key: [] for key in metric_keys}
        family_bins = {}
        order_sensitivity = []
        fitness_values = []

        for name, dataset in self.instances.items():
            dataset_items = list(dataset.items())
            if split == "train":
                selected_items = self._select_train_instances(dataset_items, limit=limit)
            elif split == "holdout":
                selected_items = self._select_holdout_instances(dataset_items, limit=limit)
            else:
                selected_items = dataset_items

            bins_used_values = []
            for _, instance in selected_items:
                capacity = float(instance["capacity"])
                items = np.array(instance["items"], dtype=float)
                trace = self._run_instance_trace(items, capacity, alg)
                for key in metric_keys:
                    collected[key].append(trace[key])
                bins_used_values.append(trace["bins_used"])
                if len(order_sensitivity) < max(1, min(3, len(selected_items))):
                    reversed_trace = self._run_instance_trace(items[::-1], capacity, alg)
                    denom = max(1.0, float(trace["bins_used"]))
                    order_sensitivity.append(abs(trace["bins_used"] - reversed_trace["bins_used"]) / denom)

            if bins_used_values:
                family_bins[name] = float(np.mean(np.asarray(bins_used_values)))
                lb_value = float(self.lb.get(name, 1.0))
                fitness_values.append((family_bins[name] - lb_value) / max(lb_value, 1.0))

        behavior = {}
        for key in metric_keys:
            values = collected[key]
            behavior[key] = float(np.mean(np.asarray(values))) if values else 0.0
        family_values = list(family_bins.values())
        behavior["family_variance"] = float(np.var(np.asarray(family_values))) if len(family_values) > 1 else 0.0
        behavior["order_sensitivity"] = float(np.mean(np.asarray(order_sensitivity))) if order_sensitivity else 0.0
        behavior["fitness"] = float(np.mean(np.asarray(fitness_values))) if fitness_values else None
        return behavior

    def online_binpack(self, items: tuple, bins: np.ndarray, alg):
        packing = [[] for _ in bins]
        for item in items:
            valid_bin_indices = self.get_valid_bin_indices(item, bins)
            priorities = alg.score(item, bins[valid_bin_indices])
            best_bin = valid_bin_indices[np.argmax(priorities)]
            bins[best_bin] -= item
            packing[best_bin].append(item)
        packing = [bin_items for bin_items in packing if bin_items]
        return packing, bins

    def _evaluate_dataset(self, dataset_items, alg, lb_value):
        num_bins_list = []
        for _, instance in dataset_items:
            capacity = instance["capacity"]
            items = np.array(instance["items"])
            bins = np.array([capacity for _ in range(instance["num_items"])])
            _, bins_packed = self.online_binpack(items, bins, alg)
            num_bins = (bins_packed != capacity).sum()
            num_bins_list.append(-num_bins)
        if len(num_bins_list) == 0:
            return None
        avg_num_bins = -np.mean(np.array(num_bins_list))
        fitness = (avg_num_bins - lb_value) / lb_value
        return float(fitness)

    def evaluateGreedy(self, alg, split="train") -> float:
        fitness_values = []
        for name, dataset in self.instances.items():
            dataset_items = list(dataset.items())
            if split == "train":
                selected_items = self._select_train_instances(dataset_items)
            elif split == "holdout":
                selected_items = self._select_holdout_instances(dataset_items)
            else:
                selected_items = dataset_items
            dataset_fitness = self._evaluate_dataset(selected_items, alg, self.lb[name])
            if dataset_fitness is not None:
                fitness_values.append(dataset_fitness)
        if len(fitness_values) == 0:
            return None
        return float(np.mean(np.array(fitness_values)))

    def _module_from_code(self, code_string):
        heuristic_module = types.ModuleType("heuristic_module")
        exec(code_string, heuristic_module.__dict__)
        sys.modules[heuristic_module.__name__] = heuristic_module
        return heuristic_module

    def evaluate_on_split(self, code_string, split="train"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                heuristic_module = self._module_from_code(code_string)
                return self.evaluateGreedy(heuristic_module, split=split)
        except Exception as e:
            if os.getenv("EOH_VERBOSE_EVAL_ERRORS", "1") == "1":
                print(f"BP evaluate error: {e}")
            return None

    def profile_code(self, code_string, train_instances=None, holdout_instances=None):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                heuristic_module = self._module_from_code(code_string)
                train_profile = self._profile_module(
                    heuristic_module,
                    split="train",
                    limit=train_instances if train_instances is not None else self.profile_train_instances,
                )
                holdout_profile = self._profile_module(
                    heuristic_module,
                    split="holdout",
                    limit=holdout_instances if holdout_instances is not None else self.profile_holdout_instances,
                )
                holdout_gap = 0.0
                if train_profile.get("fitness") is not None and holdout_profile.get("fitness") is not None:
                    holdout_gap = float(holdout_profile["fitness"] - train_profile["fitness"])
                return {
                    "mean_residual_ratio": train_profile.get("mean_residual_ratio", 0.0),
                    "residual_variance": train_profile.get("residual_variance", 0.0),
                    "fragmentation_index": train_profile.get("fragmentation_index", 0.0),
                    "resource_opening_rate_early": train_profile.get("resource_opening_rate_early", 0.0),
                    "resource_opening_rate_mid": train_profile.get("resource_opening_rate_mid", 0.0),
                    "resource_opening_rate_late": train_profile.get("resource_opening_rate_late", 0.0),
                    "choice_entropy": train_profile.get("choice_entropy", 0.0),
                    "extreme_option_preference": train_profile.get("extreme_option_preference", 0.0),
                    "score_margin_mean": train_profile.get("score_margin_mean", 0.0),
                    "score_margin_variance": train_profile.get("score_margin_variance", 0.0),
                    "order_sensitivity": train_profile.get("order_sensitivity", 0.0),
                    "family_variance": train_profile.get("family_variance", 0.0),
                    "holdout_gap": holdout_gap,
                }
        except Exception as e:
            if os.getenv("EOH_VERBOSE_EVAL_ERRORS", "1") == "1":
                print(f"BP profile error: {e}")
            return {
                "mean_residual_ratio": 0.0,
                "residual_variance": 0.0,
                "fragmentation_index": 0.0,
                "resource_opening_rate_early": 0.0,
                "resource_opening_rate_mid": 0.0,
                "resource_opening_rate_late": 0.0,
                "choice_entropy": 0.0,
                "extreme_option_preference": 0.0,
                "score_margin_mean": 0.0,
                "score_margin_variance": 0.0,
                "order_sensitivity": 0.0,
                "family_variance": 0.0,
                "holdout_gap": 0.0,
            }

    def evaluate(self, code_string):
        return self.evaluate_on_split(code_string, split="train")
