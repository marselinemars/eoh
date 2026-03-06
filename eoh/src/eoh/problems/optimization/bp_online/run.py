import numpy as np
import importlib
from .get_instance import GetData
from .prompts import GetPrompts
import types
import warnings
import sys
import os
import concurrent.futures


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _normalized_entropy(counts):
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total <= 1e-12:
        return 0.0
    probs = counts[counts > 0.0] / total
    if len(probs) <= 1:
        return 0.0
    entropy = -np.sum(probs * np.log(probs))
    return float(entropy / np.log(float(len(counts))))

class BPONLINE():
    def __init__(self, paras=None):
        getdate = GetData()
        self.instances, self.lb = getdate.get_instances()
        self.prompts = GetPrompts()
        self.eval_instances_per_gen = None
        self.holdout_instances = 64
        self.eval_parallel_instances = 1
        self.trace_probe_instances = 8
        self.order_probe_instances = 4
        self.holdout_probe_instances = 8
        if paras is not None:
            self.eval_instances_per_gen = getattr(paras, "eval_instances_per_gen", None)
            self.holdout_instances = int(getattr(paras, "holdout_instances", 64))
            self.eval_parallel_instances = max(1, int(getattr(paras, "eval_parallel_instances", 1)))
            self.trace_probe_instances = max(1, int(getattr(paras, "bp_trace_probe_instances", 8)))
            self.order_probe_instances = max(1, int(getattr(paras, "bp_order_probe_instances", 4)))
            self.holdout_probe_instances = max(1, int(getattr(paras, "bp_holdout_probe_instances", 8)))

    def _select_train_instances(self, dataset_items):
        if self.eval_instances_per_gen is None:
            return dataset_items
        try:
            cap = int(self.eval_instances_per_gen)
        except (TypeError, ValueError):
            return dataset_items
        if cap <= 0:
            return dataset_items
        return dataset_items[:cap]

    def _select_holdout_instances(self, dataset_items):
        try:
            k = int(self.holdout_instances)
        except (TypeError, ValueError):
            k = 64
        if k <= 0:
            return []

        train_items = self._select_train_instances(dataset_items)
        start = len(train_items)
        holdout_items = dataset_items[start:start + k]
        return holdout_items

    def get_valid_bin_indices(self,item: float, bins: np.ndarray) -> np.ndarray:
        """Returns indices of bins in which item can fit."""
        return np.nonzero((bins - item) >= 0)[0]


    def _phase_rates(self, opening_events):
        if len(opening_events) == 0:
            return 0.0, 0.0, 0.0
        early, mid, late = np.array_split(np.asarray(opening_events, dtype=float), 3)
        def _mean_or_zero(segment):
            return float(segment.mean()) if len(segment) > 0 else 0.0
        return _mean_or_zero(early), _mean_or_zero(mid), _mean_or_zero(late)

    def online_binpack(self,items: tuple, bins: np.ndarray, alg, collect_trace: bool = False):
        """Performs online binpacking of `items` into `bins`."""
        packing = [[] for _ in bins]
        if collect_trace:
            capacity = float(bins[0]) if len(bins) > 0 else 1.0
            used_mask = np.zeros(len(bins), dtype=bool)
            opening_events = []
            score_margins = []
            rank_bucket_counts = np.zeros(4, dtype=int)
            extreme_pref_hits = 0
            extreme_pref_total = 0

        for item in items:
            valid_bin_indices = self.get_valid_bin_indices(item, bins)
            feasible_bins = bins[valid_bin_indices]
            priorities = np.asarray(alg.score(item, feasible_bins), dtype=float).reshape(-1)
            if len(priorities) != len(valid_bin_indices):
                raise RuntimeError("score output length does not match feasible bins")
            best_bin = valid_bin_indices[np.argmax(priorities)]
            if collect_trace:
                best_local = int(np.argmax(priorities))
                residual_after = feasible_bins - item
                chosen_residual = residual_after[best_local]
                was_new_bin = not used_mask[best_bin]
                opening_events.append(1.0 if was_new_bin else 0.0)
                if len(priorities) >= 2:
                    sorted_scores = np.sort(priorities)
                    score_margins.append(float(sorted_scores[-1] - sorted_scores[-2]))
                if len(priorities) >= 2:
                    order = np.argsort(residual_after)
                    rank = int(np.where(order == best_local)[0][0])
                    frac = rank / float(max(1, len(priorities) - 1))
                    bucket = min(3, int(frac * 4.0))
                    rank_bucket_counts[bucket] += 1
                    if len(priorities) >= 3:
                        extreme_pref_total += 1
                        if rank == 0 or rank == (len(priorities) - 1):
                            extreme_pref_hits += 1
            bins[best_bin] -= item
            packing[best_bin].append(item)
            if collect_trace:
                used_mask[best_bin] = True

        packing = [bin_items for bin_items in packing if bin_items]
        if not collect_trace:
            return packing, bins

        used_residuals = bins[used_mask] / max(1.0, capacity)
        if len(used_residuals) == 0:
            used_residuals = np.asarray([0.0], dtype=float)
        mean_residual_ratio = float(np.mean(used_residuals))
        residual_variance = float(np.var(used_residuals))
        total_residual = float(np.sum(used_residuals))
        if total_residual <= 1e-12:
            fragmentation_index = 0.0
        else:
            shares = used_residuals / total_residual
            fragmentation_index = float(1.0 - np.sum(np.square(shares)))
        early_rate, mid_rate, late_rate = self._phase_rates(opening_events)
        trace = {
            "resource_utilization.mean_residual_ratio": mean_residual_ratio,
            "resource_utilization.residual_variance": residual_variance,
            "resource_utilization.fragmentation_index": fragmentation_index,
            "temporal_behavior.resource_opening_rate_early": early_rate,
            "temporal_behavior.resource_opening_rate_mid": mid_rate,
            "temporal_behavior.resource_opening_rate_late": late_rate,
            "temporal_behavior.phase_shift_index": float(max(early_rate, mid_rate, late_rate) - min(early_rate, mid_rate, late_rate)),
            "decision_pattern.extreme_option_preference": float(extreme_pref_hits / max(1, extreme_pref_total)),
            "decision_pattern.choice_entropy": _normalized_entropy(rank_bucket_counts),
            "decision_pattern.score_margin_mean": float(np.mean(np.asarray(score_margins, dtype=float))) if len(score_margins) > 0 else 0.0,
            "decision_pattern.score_margin_variance": float(np.var(np.asarray(score_margins, dtype=float))) if len(score_margins) > 0 else 0.0,
            "used_bin_count": float(np.sum(used_mask)),
            "opening_event_count": float(np.sum(opening_events)),
        }
        return packing, bins, trace

    def _trace_instance(self, instance, alg, collect_trace=False, items_override=None):
        capacity = instance['capacity']
        items = np.array(instance['items'] if items_override is None else items_override)
        bins = np.array([capacity for _ in range(instance['num_items'])])
        if collect_trace:
            _, bins_packed, trace = self.online_binpack(items, bins, alg, collect_trace=True)
        else:
            _, bins_packed = self.online_binpack(items, bins, alg, collect_trace=False)
            trace = None
        bins_used = int((bins_packed != capacity).sum())
        return bins_used, trace

    def _mean_metrics(self, metrics_list):
        if len(metrics_list) == 0:
            return {}
        keys = sorted({key for metrics in metrics_list for key in metrics.keys()})
        aggregated = {}
        for key in keys:
            values = [_safe_float(metrics.get(key), 0.0) for metrics in metrics_list if key in metrics]
            if len(values) == 0:
                continue
            aggregated[key] = float(np.mean(np.asarray(values, dtype=float)))
        return aggregated

    def _evaluate_dataset(self, dataset_items, alg, lb_value):
        details = self._evaluate_dataset_with_details(dataset_items, alg, lb_value)
        return details.get("fitness")

    def _evaluate_dataset_with_details(self, dataset_items, alg, lb_value):
        indexed_instances = list(enumerate([instance for _, instance in dataset_items]))
        num_bins_list = []
        trace_metrics = []
        metric_reasons = {}
        probe_limit = min(self.trace_probe_instances, len(indexed_instances))
        order_probe_limit = min(self.order_probe_instances, len(indexed_instances))

        def _eval_one(payload):
            idx, instance = payload
            bins_used, trace = self._trace_instance(instance, alg, collect_trace=(idx < probe_limit))
            order_gap = 0.0
            if idx < order_probe_limit:
                reversed_items = list(reversed(instance["items"]))
                reversed_bins_used, _ = self._trace_instance(instance, alg, collect_trace=False, items_override=reversed_items)
                order_gap = abs(float(bins_used) - float(reversed_bins_used)) / max(1.0, float(instance["num_items"]) / float(instance["capacity"]))
            return {
                "bins_used": bins_used,
                "trace": trace,
                "order_gap": float(order_gap),
            }

        if self.eval_parallel_instances > 1 and len(indexed_instances) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.eval_parallel_instances) as executor:
                results = list(executor.map(_eval_one, indexed_instances))
        else:
            results = [_eval_one(payload) for payload in indexed_instances]

        for result in results:
            num_bins_list.append(-result["bins_used"])
            if isinstance(result.get("trace"), dict):
                trace_metrics.append(result["trace"])

        if len(num_bins_list) == 0:
            return {"fitness": None, "trace_metrics": {}, "metric_reasons": {}}

        avg_num_bins = -np.mean(np.asarray(num_bins_list, dtype=float))
        fitness = float((avg_num_bins - lb_value) / lb_value)
        aggregated_trace = self._mean_metrics(trace_metrics)

        if len(results) > 0:
            order_gaps = [result["order_gap"] for result in results[:order_probe_limit]]
            aggregated_trace["robustness.order_sensitivity"] = float(np.mean(np.asarray(order_gaps, dtype=float))) if len(order_gaps) > 0 else 0.0
            metric_reasons["robustness.order_sensitivity"] = (
                f"computed_from_original_vs_reversed_order_on_first_{order_probe_limit}_instances"
            )

        return {
            "fitness": fitness,
            "trace_metrics": aggregated_trace,
            "metric_reasons": metric_reasons,
        }


    # @funsearch.run
    def evaluateGreedy(self, alg, split="train") -> float:
        # algorithm_module = importlib.import_module("ael_alg")
        # alg = importlib.reload(algorithm_module)  
        """Evaluate heuristic function on a set of online binpacking instances."""
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

    def evaluateGreedyWithDetails(self, alg, split="train"):
        family_fitness = {}
        family_traces = []
        family_reasons = {}
        fitness_values = []
        for name, dataset in self.instances.items():
            dataset_items = list(dataset.items())
            if split == "train":
                selected_items = self._select_train_instances(dataset_items)
            elif split == "holdout":
                selected_items = self._select_holdout_instances(dataset_items)
            else:
                selected_items = dataset_items

            dataset_details = self._evaluate_dataset_with_details(selected_items, alg, self.lb[name])
            dataset_fitness = dataset_details.get("fitness")
            if dataset_fitness is not None:
                family_fitness[name] = float(dataset_fitness)
                fitness_values.append(float(dataset_fitness))
            if isinstance(dataset_details.get("trace_metrics"), dict):
                family_traces.append(dataset_details["trace_metrics"])
            if isinstance(dataset_details.get("metric_reasons"), dict):
                for metric_name, reason in dataset_details["metric_reasons"].items():
                    family_reasons[f"{name}:{metric_name}"] = reason

        if len(fitness_values) == 0:
            return None

        details = {
            "fitness": float(np.mean(np.asarray(fitness_values, dtype=float))),
            "split": split,
            "trace_metrics": self._mean_metrics(family_traces),
            "metric_reasons": family_reasons,
            "instance_family_performance": family_fitness,
        }

        if len(family_fitness) <= 1:
            details["trace_metrics"]["robustness.instance_family_variance"] = 0.0
            details["metric_reasons"]["robustness.instance_family_variance"] = (
                "single_instance_family_available; variance is defined as 0.0 until more families are added"
            )
            # TODO: use cross-family variance once multiple bp_online families are exposed in GetData().
        else:
            family_vals = np.asarray(list(family_fitness.values()), dtype=float)
            details["trace_metrics"]["robustness.instance_family_variance"] = float(np.var(family_vals))

        if split == "train":
            holdout_values = []
            for name, dataset in self.instances.items():
                dataset_items = self._select_holdout_instances(list(dataset.items()))[: self.holdout_probe_instances]
                if len(dataset_items) == 0:
                    continue
                holdout_details = self._evaluate_dataset_with_details(dataset_items, alg, self.lb[name])
                holdout_fitness = holdout_details.get("fitness")
                if holdout_fitness is not None:
                    holdout_values.append(float(holdout_fitness))
            if len(holdout_values) > 0:
                holdout_probe = float(np.mean(np.asarray(holdout_values, dtype=float)))
                details["trace_metrics"]["robustness.holdout_gap"] = float(holdout_probe - details["fitness"])
                details["metric_reasons"]["robustness.holdout_gap"] = (
                    f"computed_from_first_{self.holdout_probe_instances}_holdout_instances_per_family"
                )
            else:
                details["trace_metrics"]["robustness.holdout_gap"] = 0.0
                details["metric_reasons"]["robustness.holdout_gap"] = "holdout_probe_unavailable; using 0.0"

        return details



    # def evaluate(self):
    #     try:

    #         for name, dataset in self.instances.items():
    #             # Parallelize the loop
    #             num_bins = Parallel(n_jobs=4,timeout=30)(delayed(self.evaluateGreedy)(instance) for _, instance in dataset.items())
    #             # avg_num_bins = -self.evaluateGreedy(dataset, algorithm)
    #             avg_num_bins = -np.mean(num_bins)
    #             excess = (avg_num_bins - self.lb[name]) / self.lb[name]
    #             #print(name)
    #             #print(f'\t Average number of bins: {avg_num_bins}')
    #             #print(f'\t Lower bound on optimum: {self.lb[name]}')
    #             #print(f'\t Excess: {100 * excess:.2f}%')        
    #         return excess
    #     except Exception as e:
    #         #print("Error:", str(e))  # Print the error message
    #         return None
        
    def evaluate_on_split(self, code_string, split="train"):
        try:
            # Suppress warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Create a new module object
                heuristic_module = types.ModuleType("heuristic_module")
                
                # Execute the code string in the new module's namespace
                exec(code_string, heuristic_module.__dict__)

                # Add the module to sys.modules so it can be imported
                sys.modules[heuristic_module.__name__] = heuristic_module

                fitness = self.evaluateGreedy(heuristic_module, split=split)

                return fitness
        except Exception as e:
            if os.getenv("EOH_VERBOSE_EVAL_ERRORS", "1") == "1":
                print(f"BP evaluate error: {e}")
            return None

    def evaluate(self, code_string):
        return self.evaluate_on_split(code_string, split="train")

    def evaluate_with_details(self, code_string, split="train"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                heuristic_module = types.ModuleType("heuristic_module")
                exec(code_string, heuristic_module.__dict__)
                sys.modules[heuristic_module.__name__] = heuristic_module
                return self.evaluateGreedyWithDetails(heuristic_module, split=split)
        except Exception as e:
            if os.getenv("EOH_VERBOSE_EVAL_ERRORS", "1") == "1":
                print(f"BP detailed evaluate error: {e}")
            return None




