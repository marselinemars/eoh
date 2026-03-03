import numpy as np
import importlib
from .get_instance import GetData
from .prompts import GetPrompts
import types
import warnings
import sys
import os

class BPONLINE():
    def __init__(self, paras=None):
        getdate = GetData()
        self.instances, self.lb = getdate.get_instances()
        self.prompts = GetPrompts()
        self.eval_instances_per_gen = None
        self.holdout_instances = 64
        if paras is not None:
            self.eval_instances_per_gen = getattr(paras, "eval_instances_per_gen", None)
            self.holdout_instances = int(getattr(paras, "holdout_instances", 64))

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


    def online_binpack(self,items: tuple, bins: np.ndarray, alg):
        """Performs online binpacking of `items` into `bins`."""
        # Track which items are added to each bin.
        packing = [[] for _ in bins]
        # Add items to bins.
        n = 1
        for item in items:
            # Extract bins that have sufficient space to fit item.
            valid_bin_indices = self.get_valid_bin_indices(item, bins)
            # Score each bin based on heuristic.
            priorities = alg.score(item, bins[valid_bin_indices])
            # Add item to bin with highest priority.
            best_bin = valid_bin_indices[np.argmax(priorities)]
            bins[best_bin] -= item
            packing[best_bin].append(item)
            n=n+1
            
        # Remove unused bins from packing.
        packing = [bin_items for bin_items in packing if bin_items]
        return packing, bins

    def _evaluate_dataset(self, dataset_items, alg, lb_value):
        num_bins_list = []
        for _, instance in dataset_items:
            capacity = instance['capacity']
            items = np.array(instance['items'])

            bins = np.array([capacity for _ in range(instance['num_items'])])
            _, bins_packed = self.online_binpack(items, bins, alg)
            num_bins = (bins_packed != capacity).sum()
            num_bins_list.append(-num_bins)

        if len(num_bins_list) == 0:
            return None

        avg_num_bins = -np.mean(np.array(num_bins_list))
        fitness = (avg_num_bins - lb_value) / lb_value
        return float(fitness)


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




