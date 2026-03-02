import numpy as np
import importlib
from .get_instance import GetData
from .prompts import GetPrompts
import types
import warnings
import sys

class BPONLINE():
    EXEMPLAR_INSTANCES = {0, 17, 93, 221, 509}

    def __init__(self, paras=None):
        getdate = GetData()
        self.instances, self.lb = getdate.get_instances()
        self.prompts = GetPrompts()
        self.eval_instances_per_gen = int(getattr(paras, "eval_instances_per_gen", 512))
        self.log_trace_summary = bool(getattr(paras, "log_trace_summary", True))
        self.log_exemplars = bool(getattr(paras, "log_exemplars", False))
        self._full_evaluation_mode = False

    def set_full_evaluation_mode(self, enabled: bool) -> None:
        self._full_evaluation_mode = bool(enabled)

    def get_valid_bin_indices(self,item: float, bins: np.ndarray) -> np.ndarray:
        """Returns indices of bins in which item can fit."""
        return np.nonzero((bins - item) >= 0)[0]


    def online_binpack(self,items: tuple, bins: np.ndarray, alg, collect_per_item: bool = False):
        """Performs online binpacking of `items` into `bins`."""
        # Track which items are added to each bin.
        packing = [[] for _ in bins]
        per_item_trace = [] if collect_per_item else None
        full_capacity = float(np.max(bins)) if len(bins) > 0 else 0.0

        sum_margin = 0.0
        count_margin = 0
        new_bin_count = 0
        item_count = 0
        sum_candidate_bins = 0
        tie_count = 0
        sum_remaining_cap = 0.0
        sumsq_remaining_cap = 0.0

        # Add items to bins.
        for item in items:
            valid_bin_indices = self.get_valid_bin_indices(item, bins)
            # Score each bin based on heuristic.
            priorities = alg.score(item, bins[valid_bin_indices])
            candidate_bin_count = int(len(valid_bin_indices))
            if candidate_bin_count == 0 or len(priorities) == 0:
                continue
            best_score = float(np.max(priorities)) if len(priorities) > 0 else 0.0
            if len(priorities) > 1:
                second_best_score = float(np.partition(priorities, -2)[-2])
            elif len(priorities) == 1:
                second_best_score = best_score
            else:
                second_best_score = 0.0
            # Add item to bin with highest priority.
            best_bin = valid_bin_indices[np.argmax(priorities)]
            was_new_bin = bool(bins[best_bin] == full_capacity)
            bins[best_bin] -= item
            packing[best_bin].append(item)
            margin = best_score - second_best_score
            remaining_capacity = float(bins[best_bin])

            sum_margin += margin
            count_margin += 1
            if was_new_bin:
                new_bin_count += 1
            item_count += 1
            sum_candidate_bins += candidate_bin_count
            if best_score == second_best_score:
                tie_count += 1
            sum_remaining_cap += remaining_capacity
            sumsq_remaining_cap += remaining_capacity ** 2

            if collect_per_item:
                per_item_trace.append(
                    {
                        "item_size": float(item),
                        "candidate_bins": candidate_bin_count,
                        "best_score": best_score,
                        "second_best_score": second_best_score,
                        "new_bin": was_new_bin,
                        "remaining_capacity": remaining_capacity,
                    }
                )
             
        # Remove unused bins from packing.
        packing = [bin_items for bin_items in packing if bin_items]
        summary = {
            "n_items": int(item_count),
            "sum_margin": float(sum_margin),
            "count_margin": int(count_margin),
            "new_bin_count": int(new_bin_count),
            "sum_candidate_bins": int(sum_candidate_bins),
            "tie_count": int(tie_count),
            "sum_remaining_cap": float(sum_remaining_cap),
            "sumsq_remaining_cap": float(sumsq_remaining_cap),
        }
        return packing, bins, summary, (per_item_trace if collect_per_item else None)


    # @funsearch.run
    def evaluateGreedy(self,alg) -> float:
        # algorithm_module = importlib.import_module("ael_alg")
        # alg = importlib.reload(algorithm_module)  
        """Evaluate heuristic function on a set of online binpacking instances."""
        # List storing number of bins used for each instance.
        #num_bins = []
        # Perform online binpacking for each instance.
        # for name in instances:
        #     #print(name)
        sum_margin = 0.0
        count_margin = 0
        new_bin_count = 0
        item_count = 0
        sum_candidate_bins = 0
        tie_count = 0
        sum_remaining_cap = 0.0
        sumsq_remaining_cap = 0.0
        exemplars = {}
        instance_idx = 0
        limit_reached = False
        fitness = None
        for name, dataset in self.instances.items():
            num_bins_list = []
            for _, instance in dataset.items():
                if (not self._full_evaluation_mode) and instance_idx >= self.eval_instances_per_gen:
                    limit_reached = True
                    break

                capacity = instance['capacity']
                items = np.array(instance['items'])

                # items = items/capacity
                # capacity = 1.0

                # Create num_items bins so there will always be space for all items,
                # regardless of packing order. Array has shape (num_items,).
                bins = np.array([capacity for _ in range(instance['num_items'])])
                # Pack items into bins and return remaining capacity in bins_packed, which
                # has shape (num_items,).
                collect_per_item = self.log_exemplars and (instance_idx in self.EXEMPLAR_INSTANCES)
                _, bins_packed, summary, per_item = self.online_binpack(items, bins, alg, collect_per_item=collect_per_item)
                # If remaining capacity in a bin is equal to initial capacity, then it is
                # unused. Count number of used bins.
                num_bins = (bins_packed != capacity).sum()

                sum_margin += float(summary["sum_margin"])
                count_margin += int(summary["count_margin"])
                new_bin_count += int(summary["new_bin_count"])
                item_count += int(summary["n_items"])
                sum_candidate_bins += int(summary["sum_candidate_bins"])
                tie_count += int(summary["tie_count"])
                sum_remaining_cap += float(summary["sum_remaining_cap"])
                sumsq_remaining_cap += float(summary["sumsq_remaining_cap"])

                if collect_per_item and per_item is not None:
                    exemplars[str(instance_idx)] = per_item

                num_bins_list.append(-num_bins)
                instance_idx += 1

            # avg_num_bins = -self.evaluateGreedy(dataset, algorithm)
            if len(num_bins_list) > 0:
                avg_num_bins = -np.mean(np.array(num_bins_list))
                fitness = (avg_num_bins - self.lb[name]) / self.lb[name]

            if limit_reached:
                break


        # Score of heuristic function is negative of average number of bins used
        # across instances (as we want to minimize number of bins).

        trace = {
            "summary": {
                "n_items": int(item_count),
                "sum_margin": float(sum_margin),
                "count_margin": int(count_margin),
                "new_bin_count": int(new_bin_count),
                "sum_candidate_bins": int(sum_candidate_bins),
                "tie_count": int(tie_count),
                "sum_remaining_cap": float(sum_remaining_cap),
                "sumsq_remaining_cap": float(sumsq_remaining_cap),
            }
        }
        if self.log_exemplars and len(exemplars) > 0:
            trace["exemplars"] = exemplars
        return fitness, trace



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
        
    def evaluate(self, code_string):
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

                result = self.evaluateGreedy(heuristic_module)
                return result
        except Exception as e:
            #print("Error:", str(e))
            return None




