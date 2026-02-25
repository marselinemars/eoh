import numpy as np
import importlib
from .get_instance import GetData
from .prompts import GetPrompts
import types
import warnings
import sys

class BPONLINE():
    def __init__(self):
        getdate = GetData()
        self.instances, self.lb = getdate.get_instances()
        self.prompts = GetPrompts()

    def get_valid_bin_indices(self,item: float, bins: np.ndarray) -> np.ndarray:
        """Returns indices of bins in which item can fit."""
        return np.nonzero((bins - item) >= 0)[0]


    def online_binpack(self,items: tuple, bins: np.ndarray, alg, return_trace=False):
        """Performs online binpacking of `items` into `bins`."""
        # Track which items are added to each bin.
        packing = [[] for _ in bins]
        trace = {
            "tie_events": 0,
            "near_miss": 0,
            "late_large_failures": 0,
            "new_bin_events": 0,
            "n_items": int(len(items)),
        }
        cap = float(bins[0]) if len(bins) > 0 else 1.0
        near_eps = 0.05
        late_start = int(0.7 * len(items))
        large_threshold = 0.7 * cap

        # Add items to bins.
        for idx, item in enumerate(items):
            # Extract bins that have sufficient space to fit item.
            valid_bin_indices = self.get_valid_bin_indices(item, bins)
            if len(valid_bin_indices) == 0:
                continue
            # Score each bin based on heuristic.
            priorities = alg.score(item, bins[valid_bin_indices])
            # Add item to bin with highest priority.
            if len(priorities) > 0:
                max_priority = np.max(priorities)
                tie_count = int(np.sum(priorities == max_priority))
                if tie_count > 1:
                    trace["tie_events"] += 1
            best_bin = valid_bin_indices[np.argmax(priorities)]
            selected_was_new = bool(bins[best_bin] == cap)
            if selected_was_new:
                trace["new_bin_events"] += 1
            bins[best_bin] -= item
            packing[best_bin].append(item)

            if selected_was_new:
                used_valid = valid_bin_indices[bins[valid_bin_indices] < cap]
                if len(used_valid) > 0:
                    post_slack = bins[used_valid] - item
                    feasible = post_slack[post_slack >= 0]
                    if len(feasible) > 0 and float(np.min(feasible)) <= near_eps * cap:
                        trace["near_miss"] += 1

                if idx >= late_start and item >= large_threshold:
                    trace["late_large_failures"] += 1
            
        # Remove unused bins from packing.
        packing = [bin_items for bin_items in packing if bin_items]
        if not return_trace:
            return packing, bins

        used_mask = bins != cap
        used_bins = bins[used_mask]
        bins_used = int(np.sum(used_mask))
        trace.update(
            {
                "bins_used": bins_used,
                "unused_total": float(np.sum(used_bins)) if bins_used > 0 else 0.0,
                "fill_std": float(np.std(cap - used_bins)) if bins_used > 0 else 0.0,
                "utilization_mean": float(np.mean((cap - used_bins) / cap)) if bins_used > 0 else 0.0,
                "capacity": cap,
            }
        )
        return packing, bins, trace


    # @funsearch.run
    def _summarize_trace(self, traces):
        if len(traces) == 0:
            return {}
        keys = [
            "bins_used",
            "unused_total",
            "fill_std",
            "utilization_mean",
            "tie_events",
            "near_miss",
            "late_large_failures",
            "new_bin_events",
            "n_items",
        ]
        out = {}
        for key in keys:
            values = np.array([float(t.get(key, 0.0)) for t in traces], dtype=float)
            out[f"{key}_mean"] = float(np.mean(values))
            out[f"{key}_p90"] = float(np.percentile(values, 90))
        return out


    # @funsearch.run
    def evaluateGreedy(self,alg, return_trace=False) -> float:
        # algorithm_module = importlib.import_module("ael_alg")
        # alg = importlib.reload(algorithm_module)  
        """Evaluate heuristic function on a set of online binpacking instances."""
        # List storing number of bins used for each instance.
        #num_bins = []
        # Perform online binpacking for each instance.
        # for name in instances:
        #     #print(name)

        regime_summary = {}
        fitness = None

        for name, dataset in self.instances.items():
            num_bins_list = []
            regime_traces = []
            for _, instance in dataset.items():

                capacity = instance['capacity']
                items = np.array(instance['items'])

                # items = items/capacity
                # capacity = 1.0

                # Create num_items bins so there will always be space for all items,
                # regardless of packing order. Array has shape (num_items,).
                bins = np.array([capacity for _ in range(instance['num_items'])])
                # Pack items into bins and return remaining capacity in bins_packed, which
                # has shape (num_items,).
                if return_trace:
                    _, bins_packed, trace = self.online_binpack(items, bins, alg, return_trace=True)
                    regime_traces.append(trace)
                else:
                    _, bins_packed = self.online_binpack(items, bins, alg, return_trace=False)
                # If remaining capacity in a bin is equal to initial capacity, then it is
                # unused. Count number of used bins.
                num_bins = (bins_packed != capacity).sum()

                num_bins_list.append(-num_bins)

            # avg_num_bins = -self.evaluateGreedy(dataset, algorithm)
            avg_num_bins = -np.mean(np.array(num_bins_list))
            fitness = (avg_num_bins - self.lb[name]) / self.lb[name]
            if return_trace:
                regime_summary[name] = {
                    "fitness": float(fitness),
                    "n_instances": len(num_bins_list),
                    "stats": self._summarize_trace(regime_traces),
                }


        # Score of heuristic function is negative of average number of bins used
        # across instances (as we want to minimize number of bins).

        if not return_trace:
            return fitness

        all_stats = []
        for regime_name in regime_summary:
            all_stats.append(regime_summary[regime_name]["stats"])
        overall = {}
        if len(all_stats) > 0:
            stat_keys = all_stats[0].keys()
            for key in stat_keys:
                overall[key] = float(np.mean([s[key] for s in all_stats]))

        trace_summary = {
            "problem": "bp_online",
            "regimes": regime_summary,
            "overall": overall,
            "n_regimes": len(regime_summary),
        }
        return fitness, trace_summary



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
        
    def evaluate(self, code_string, return_trace=False):
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

                result = self.evaluateGreedy(heuristic_module, return_trace=return_trace)
                return result
        except Exception as e:
            #print("Error:", str(e))
            return None


