import numpy as np
import json
import random
import time
import os
import hashlib

from .eoh_interface_EC import InterfaceEC
# main class for eoh
class EOH:

    # initilization
    def __init__(self, paras, problem, select, manage, **kwargs):

        self.prob = problem
        self.select = select
        self.manage = manage
        
        # LLM settings
        self.use_local_llm = paras.llm_use_local
        self.llm_local_url = paras.llm_local_url
        self.api_endpoint = paras.llm_api_endpoint  # currently only API2D + GPT
        self.api_key = paras.llm_api_key
        self.llm_model = paras.llm_model

        # ------------------ RZ: use local LLM ------------------
        # self.use_local_llm = kwargs.get('use_local_llm', False)
        # assert isinstance(self.use_local_llm, bool)
        # if self.use_local_llm:
        #     assert 'url' in kwargs, 'The keyword "url" should be provided when use_local_llm is True.'
        #     assert isinstance(kwargs.get('url'), str)
        #     self.url = kwargs.get('url')
        # -------------------------------------------------------

        # Experimental settings       
        self.pop_size = paras.ec_pop_size  # popopulation size, i.e., the number of algorithms in population
        self.n_pop = paras.ec_n_pop  # number of populations

        self.operators = paras.ec_operators
        self.operator_weights = paras.ec_operator_weights
        if paras.ec_m > self.pop_size or paras.ec_m == 1:
            print("m should not be larger than pop size or smaller than 2, adjust it to m=2")
            paras.ec_m = 2
        self.m = paras.ec_m

        self.debug_mode = paras.exp_debug_mode  # if debug
        self.ndelay = 1  # default

        self.use_seed = paras.exp_use_seed
        self.seed_path = paras.exp_seed_path
        self.load_pop = paras.exp_use_continue
        self.load_pop_path = paras.exp_continue_path
        self.load_pop_id = paras.exp_continue_id

        self.output_path = paras.exp_output_path

        self.exp_n_proc = paras.exp_n_proc
        
        self.timeout = paras.eva_timeout

        self.use_numba = paras.eva_numba_decorator

        self.mode = getattr(paras, "eoh_mode", "baseline")
        if self.mode not in ["baseline", "routed"]:
            print(f"Unknown eoh_mode={self.mode}, fallback to baseline.")
            self.mode = "baseline"
        self.route_improvement_epsilon = float(getattr(paras, "route_improvement_epsilon", 1e-4))
        self.route_stagnation_k = int(getattr(paras, "route_stagnation_k", 3))
        self.route_invalid_rate_threshold = float(getattr(paras, "route_invalid_rate_threshold", 0.5))
        self.route_use_diversity = bool(getattr(paras, "route_use_diversity", True))
        self.log_full_population = bool(getattr(paras, "log_full_population", False))
        self.run_log_path = os.path.join(self.output_path, "results", "run_log.jsonl")
        self.operator_log_path = os.path.join(self.output_path, "results", "operator_events.jsonl")

        print("- EoH parameters loaded -")

        # Set a random seed
        random.seed(2024)

    # add new individual to population
    def add2pop(self, population, offspring):
        for off in offspring:
            for ind in population:
                if ind['objective'] == off['objective']:
                    if (self.debug_mode):
                        print("duplicated result, retrying ... ")
            population.append(off)
    
    def _prepare_run_log(self):
        os.makedirs(os.path.dirname(self.run_log_path), exist_ok=True)
        with open(self.run_log_path, "w", encoding="utf-8") as _:
            pass
        with open(self.operator_log_path, "w", encoding="utf-8") as _:
            pass

    def _write_run_log(self, record):
        with open(self.run_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _write_operator_log(self, record):
        with open(self.operator_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _code_hash(self, code):
        if code is None:
            return None
        return hashlib.sha1(code.encode("utf-8")).hexdigest()[:12]

    def _compact_individual(self, individual):
        return {
            "objective": individual.get("objective"),
            "code_hash": self._code_hash(individual.get("code")),
        }

    def _serialize_population(self, population):
        if self.log_full_population:
            return population
        return [self._compact_individual(individual) for individual in population]

    def _serialize_best(self, population):
        if len(population) == 0:
            return {}
        if self.log_full_population:
            return population[0]
        return self._compact_individual(population[0])

    def _save_population(self, population, generation):
        pop_path = os.path.join(self.output_path, "results", "pops", f"population_generation_{generation}.json")
        best_path = os.path.join(self.output_path, "results", "pops_best", f"population_generation_{generation}.json")
        with open(pop_path, "w", encoding="utf-8") as f:
            json.dump(self._serialize_population(population), f, indent=2)
        with open(best_path, "w", encoding="utf-8") as f:
            json.dump(self._serialize_best(population), f, indent=2)

    def _invalid_rate(self, offspring_list):
        if len(offspring_list) == 0:
            return 1.0
        invalid_count = 0
        for offspring in offspring_list:
            if offspring is None:
                invalid_count += 1
                continue
            if offspring.get("objective") is None or offspring.get("code") is None:
                invalid_count += 1
        return invalid_count / len(offspring_list)

    def _count_valid(self, offspring_list):
        valid_count = 0
        for offspring in offspring_list:
            if offspring is None:
                continue
            if offspring.get("objective") is not None and offspring.get("code") is not None:
                valid_count += 1
        return valid_count

    def _best_objective(self, population):
        if len(population) == 0:
            return None
        return population[0]["objective"]

    def _to_float_or_none(self, value):
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _population_diversity(self, population):
        if not self.route_use_diversity:
            return None
        hashes = set()
        for individual in population:
            code_hash = self._code_hash(individual.get("code"))
            if code_hash is not None:
                hashes.add(code_hash)
        return len(hashes)

    def _diagnose(self, last_improved, stagnation_count, last_invalid_rate):
        if last_invalid_rate > self.route_invalid_rate_threshold:
            return "TOO_MANY_INVALIDS"
        if last_improved:
            return "IMPROVING"
        if stagnation_count >= self.route_stagnation_k:
            return "STAGNATING"
        return "DEFAULT"

    def _route_operator(self, diagnosis_label):
        route_map = {
            "IMPROVING": "e1",
            "STAGNATING": "m2",
            "TOO_MANY_INVALIDS": "m1",
            "DEFAULT": "e2",
        }
        op = route_map.get(diagnosis_label, self.operators[0])
        if op not in self.operators:
            op = self.operators[0]
        return op

    # run eoh 
    def run(self):

        print("- Evolution Start -")

        time_start = time.time()

        # interface for large language model (llm)
        # interface_llm = PromptLLMs(self.api_endpoint,self.api_key,self.llm_model,self.debug_mode)

        # interface for evaluation
        interface_prob = self.prob

        # interface for ec operators
        interface_ec = InterfaceEC(self.pop_size, self.m, self.api_endpoint, self.api_key, self.llm_model, self.use_local_llm, self.llm_local_url,
                                   self.debug_mode, interface_prob, select=self.select,n_p=self.exp_n_proc,
                                   timeout = self.timeout, use_numba=self.use_numba
                                   )

        self._prepare_run_log()

        # initialization
        population = []
        if self.use_seed:
            with open(self.seed_path) as file:
                data = json.load(file)
            population = interface_ec.population_generation_seed(data,self.exp_n_proc)
            self._save_population(population, 0)
            n_start = 0
        else:
            if self.load_pop:  # load population from files
                print("load initial population from " + self.load_pop_path)
                with open(self.load_pop_path) as file:
                    data = json.load(file)
                for individual in data:
                    population.append(individual)
                print("initial population has been loaded!")
                n_start = self.load_pop_id
            else:  # create new population
                print("creating initial population:")
                population = interface_ec.population_generation()
                population = self.manage.population_management(population, self.pop_size)

                # print(len(population))
                # if len(population)<self.pop_size:
                #     for op in [self.operators[0],self.operators[2]]:
                #         _,new_ind = interface_ec.get_algorithm(population, op)
                #         self.add2pop(population, new_ind)
                #         population = self.manage.population_management(population, self.pop_size)
                #         if len(population) >= self.pop_size:
                #             break
                #         print(len(population))
     
                
                print(f"Pop initial: ")
                for off in population:
                    print(" Obj: ", off['objective'], end="|")
                print()
                print("initial population has been created!")
                self._save_population(population, 0)
                n_start = 0

        # main loop
        n_op = len(self.operators)
        prev_best = population[0]["objective"] if len(population) > 0 else None
        stagnation_count = 0
        last_invalid_rate = 0.0
        last_improved = False

        for pop in range(n_start, self.n_pop):
            generation_offspring = []
            chosen_operator = None
            diagnosis_label = "DEFAULT"

            if self.mode == "routed":
                diagnosis_label = self._diagnose(last_improved, stagnation_count, last_invalid_rate)
                chosen_operator = self._route_operator(diagnosis_label)
                print(f" OP: {chosen_operator}, [routed] ", end="|")
                best_before = self._best_objective(population)
                _, offsprings = interface_ec.get_algorithm(population, chosen_operator)
                self.add2pop(population, offsprings)
                generation_offspring.extend(offsprings)
                for off in offsprings:
                    print(" Obj: ", off["objective"], end="|")
                size_act = min(len(population), self.pop_size)
                population = self.manage.population_management(population, size_act)
                best_after = self._best_objective(population)
                n_offspring = len(offsprings)
                n_valid = self._count_valid(offsprings)
                n_invalid = n_offspring - n_valid
                invalid_rate_op = (n_invalid / n_offspring) if n_offspring > 0 else 1.0
                delta_best = None
                if best_before is not None and best_after is not None:
                    delta_best = float(best_before - best_after)
                op_record = {
                    "gen": int(pop + 1),
                    "mode": self.mode,
                    "operator": chosen_operator,
                    "operator_weight": None,
                    "executed": True,
                    "diagnosis_label": diagnosis_label,
                    "n_offspring": int(n_offspring),
                    "n_valid": int(n_valid),
                    "n_invalid": int(n_invalid),
                    "invalid_rate": float(invalid_rate_op),
                    "best_before": self._to_float_or_none(best_before),
                    "best_after": self._to_float_or_none(best_after),
                    "delta_best": self._to_float_or_none(delta_best),
                }
                self._write_operator_log(op_record)
                print()
            else:
                chosen_operator = "schedule:" + ",".join(self.operators)
                for i in range(n_op):
                    op = self.operators[i]
                    print(f" OP: {op}, [{i + 1} / {n_op}] ", end="|")
                    op_w = self.operator_weights[i]
                    offsprings = []
                    best_before = self._best_objective(population)
                    executed = np.random.rand() < op_w
                    if executed:
                        _, offsprings = interface_ec.get_algorithm(population, op)
                    self.add2pop(population, offsprings)
                    generation_offspring.extend(offsprings)
                    for off in offsprings:
                        print(" Obj: ", off["objective"], end="|")
                    size_act = min(len(population), self.pop_size)
                    population = self.manage.population_management(population, size_act)
                    best_after = self._best_objective(population)
                    n_offspring = len(offsprings)
                    n_valid = self._count_valid(offsprings)
                    n_invalid = n_offspring - n_valid
                    invalid_rate_op = (n_invalid / n_offspring) if n_offspring > 0 else 1.0
                    delta_best = None
                    if best_before is not None and best_after is not None:
                        delta_best = float(best_before - best_after)
                    op_record = {
                        "gen": int(pop + 1),
                        "mode": self.mode,
                        "operator": op,
                        "operator_weight": float(op_w),
                        "executed": bool(executed),
                        "diagnosis_label": "DEFAULT",
                        "n_offspring": int(n_offspring),
                        "n_valid": int(n_valid),
                        "n_invalid": int(n_invalid),
                        "invalid_rate": float(invalid_rate_op),
                        "best_before": self._to_float_or_none(best_before),
                        "best_after": self._to_float_or_none(best_after),
                        "delta_best": self._to_float_or_none(delta_best),
                    }
                    self._write_operator_log(op_record)
                    print()

            self._save_population(population, pop + 1)

            invalid_rate = self._invalid_rate(generation_offspring)
            diversity = self._population_diversity(population)
            best_fitness = population[0]["objective"] if len(population) > 0 else None

            if prev_best is None and best_fitness is not None:
                last_improved = True
                stagnation_count = 0
                prev_best = best_fitness
            elif best_fitness is None:
                last_improved = False
                stagnation_count += 1
            else:
                improvement = prev_best - best_fitness
                if improvement > self.route_improvement_epsilon:
                    last_improved = True
                    stagnation_count = 0
                    prev_best = best_fitness
                else:
                    last_improved = False
                    stagnation_count += 1
                    prev_best = min(prev_best, best_fitness)

            last_invalid_rate = invalid_rate
            run_record = {
                "gen": int(pop + 1),
                "mode": self.mode,
                "best_fitness": best_fitness,
                "chosen_operator": chosen_operator,
                "diagnosis_label": diagnosis_label,
                "invalid_rate": invalid_rate,
                "stagnation_count": int(stagnation_count),
                "diversity": diversity,
            }
            self._write_run_log(run_record)

            print(f"--- {pop + 1} of {self.n_pop} populations finished. Time Cost:  {((time.time()-time_start)/60):.1f} m")
            print("Pop Objs: ", end=" ")
            for i in range(len(population)):
                print(str(population[i]['objective']) + " ", end="")
            print()

