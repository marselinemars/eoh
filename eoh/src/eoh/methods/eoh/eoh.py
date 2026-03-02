import numpy as np
import json
import random
import time
import os
import hashlib

from .eoh_interface_EC import InterfaceEC
from .diagnosis import Diagnostician
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
        self.use_ddhs = getattr(paras, "use_ddhs", False)
        self.ddhs_shuffle = getattr(paras, "ddhs_shuffle", False)
        self.final_full_evaluation = bool(getattr(paras, "final_full_evaluation", True))
        self.log_trace_summary = bool(getattr(paras, "log_trace_summary", True))
        self.log_exemplars = bool(getattr(paras, "log_exemplars", False))
        self.log_llm_interactions = bool(getattr(paras, "log_llm_interactions", False))
        self.log_full_population = bool(getattr(paras, "log_full_population", False))
        self.ddhs_mode = "ddhs_shuffle" if self.use_ddhs and self.ddhs_shuffle else ("ddhs" if self.use_ddhs else "baseline")
        self.diagnostician = Diagnostician()
        self.diagnosis_records = []
        self.diagnosis_log_path = os.path.join(self.output_path, "results", "diagnosis_log.json")
        self.llm_log_dir = os.path.join(self.output_path, "logs", "llm_interactions")
        self.ddhs_routing = {
            "OVER_GREEDY": "m2",
            "FLAT_SCORING": "e1",
            "UNSTABLE_CAPACITY": "m3",
            "EXPLORATION_NEEDED": "e2",
        }

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

    def _code_hash(self, code):
        if code is None:
            return None
        return hashlib.sha256(code.encode("utf-8")).hexdigest()

    def _trace_summary(self, trace):
        if isinstance(trace, dict):
            summary = trace.get("summary")
            if isinstance(summary, dict):
                return summary
        return {}

    def _compact_individual(self, individual):
        trace = individual.get("trace")
        metrics = self.diagnostician.compute_metrics(trace)
        diagnosis_label = self.diagnostician.diagnose(metrics)
        metric_summary = self._trace_summary(trace) if self.log_trace_summary else {}
        data = {
            "objective": individual.get("objective"),
            "operator": individual.get("operator"),
            "diagnosis": diagnosis_label,
            "metric_summary": metric_summary,
            "code_hash": self._code_hash(individual.get("code")),
            "code": individual.get("code"),
            "algorithm": individual.get("algorithm"),
        }
        if self.log_full_population:
            data["trace"] = trace
            data["other_inf"] = individual.get("other_inf")
        return data

    def _serialize_population(self, population):
        if self.log_full_population:
            return population
        return [self._compact_individual(individual) for individual in population]

    def _serialize_best(self, individual):
        if self.log_full_population:
            return individual
        return self._compact_individual(individual)

    def _restore_trace_if_missing(self, individual):
        if individual.get("trace") is not None:
            return individual
        metric_summary = individual.get("metric_summary")
        if isinstance(metric_summary, dict):
            individual["trace"] = {"summary": metric_summary}
        else:
            individual["trace"] = None
        return individual
    

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
                                   timeout = self.timeout, use_numba=self.use_numba,
                                   log_llm_interactions=self.log_llm_interactions,
                                   llm_log_dir=self.llm_log_dir,
                                   )

        # initialization
        population = []
        if self.use_seed:
            with open(self.seed_path) as file:
                data = json.load(file)
            population = interface_ec.population_generation_seed(data,self.exp_n_proc)
            filename = self.output_path + "/results/pops/population_generation_0.json"
            with open(filename, 'w') as f:
                json.dump(self._serialize_population(population), f, indent=5)
            n_start = 0
        else:
            if self.load_pop:  # load population from files
                print("load initial population from " + self.load_pop_path)
                with open(self.load_pop_path) as file:
                    data = json.load(file)
                for individual in data:
                    population.append(self._restore_trace_if_missing(individual))
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
                # Save population to a file
                filename = self.output_path + "/results/pops/population_generation_0.json"
                with open(filename, 'w') as f:
                    json.dump(self._serialize_population(population), f, indent=5)
                n_start = 0

        # main loop
        n_op = len(self.operators)

        for pop in range(n_start, self.n_pop):  
            if len(population) == 0:
                print("Warning: population is empty. Stopping evolution early.")
                break
            fitness_before = None
            diagnosis_label = None
            diagnosis_label_routed = None
            ddhs_operator = None
            diagnosis_metrics = {}
            if len(population) > 0:
                fitness_before = population[0].get("objective")

            if self.use_ddhs:
                best_individual = population[0] if len(population) > 0 else None
                best_trace = None if best_individual is None else best_individual.get("trace")
                diagnosis_metrics = self.diagnostician.compute_metrics(best_trace)
                diagnosis_label = self.diagnostician.diagnose(diagnosis_metrics)
                diagnosis_label_routed = diagnosis_label
                if self.ddhs_shuffle:
                    diagnosis_label_routed = random.choice(list(self.ddhs_routing.keys()))
                ddhs_operator = self.ddhs_routing.get(diagnosis_label_routed, "e2")

            #print(f" [{na + 1} / {self.pop_size}] ", end="|")         
            if self.use_ddhs:
                operator_weight_map = {op_name: self.operator_weights[idx] for idx, op_name in enumerate(self.operators)}
                for i in range(n_op):
                    op = ddhs_operator
                    op_w = operator_weight_map.get(op, 1)
                    print(f" OP: {op}, [{i + 1} / {n_op}] ", end="|")
                    offsprings = []
                    if (np.random.rand() < op_w):
                        parents, offsprings = interface_ec.get_algorithm(population, op, generation_idx=pop + 1)
                    self.add2pop(population, offsprings)
                    for off in offsprings:
                        print(" Obj: ", off['objective'], end="|")
                    size_act = min(len(population), self.pop_size)
                    population = self.manage.population_management(population, size_act)
                    print()
            else:
                for i in range(n_op):
                    op = self.operators[i]
                    print(f" OP: {op}, [{i + 1} / {n_op}] ", end="|") 
                    op_w = self.operator_weights[i]
                    offsprings = []
                    if (np.random.rand() < op_w):
                        parents, offsprings = interface_ec.get_algorithm(population, op, generation_idx=pop + 1)
                    self.add2pop(population, offsprings)  # Check duplication, and add the new offspring
                    for off in offsprings:
                        print(" Obj: ", off['objective'], end="|")
                    # if is_add:
                    #     data = {}
                    #     for i in range(len(parents)):
                    #         data[f"parent{i + 1}"] = parents[i]
                    #     data["offspring"] = offspring
                    #     with open(self.output_path + "/results/history/pop_" + str(pop + 1) + "_" + str(
                    #             na) + "_" + op + ".json", "w") as file:
                    #         json.dump(data, file, indent=5)
                    # populatin management
                    size_act = min(len(population), self.pop_size)
                    population = self.manage.population_management(population, size_act)
                    print()


            # Save population to a file
            filename = self.output_path + "/results/pops/population_generation_" + str(pop + 1) + ".json"
            with open(filename, 'w') as f:
                json.dump(self._serialize_population(population), f, indent=5)

            # Save the best one to a file
            filename = self.output_path + "/results/pops_best/population_generation_" + str(pop + 1) + ".json"
            with open(filename, 'w') as f:
                if len(population) > 0:
                    json.dump(self._serialize_best(population[0]), f, indent=5)
                else:
                    json.dump({}, f, indent=5)

            fitness_after = population[0].get("objective") if len(population) > 0 else None
            diagnosis_record = {
                "generation": int(pop + 1),
                "mode": self.ddhs_mode,
                "diagnosis": diagnosis_label,
                "diagnosis_routed": diagnosis_label_routed,
                "chosen_operator": ddhs_operator,
                "fitness_before": fitness_before,
                "fitness_after": fitness_after,
                "metrics": diagnosis_metrics,
            }
            self.diagnosis_records.append(diagnosis_record)
            with open(self.diagnosis_log_path, "w", encoding="utf-8") as fh:
                json.dump(self.diagnosis_records, fh, indent=2)


            print(f"--- {pop + 1} of {self.n_pop} populations finished. Time Cost:  {((time.time()-time_start)/60):.1f} m")
            print("Pop Objs: ", end=" ")
            for i in range(len(population)):
                print(str(population[i]['objective']) + " ", end="")
            print()

        if self.final_full_evaluation and len(population) > 0:
            best_code = population[0].get("code")
            final_full_objective = None
            final_trace_summary = {}
            if best_code is not None:
                if hasattr(interface_prob, "set_full_evaluation_mode"):
                    interface_prob.set_full_evaluation_mode(True)
                try:
                    final_eval = interface_prob.evaluate(best_code)
                    if isinstance(final_eval, tuple) and len(final_eval) == 2:
                        final_full_objective, final_trace = final_eval
                        final_trace_summary = self._trace_summary(final_trace)
                    else:
                        final_full_objective = final_eval
                finally:
                    if hasattr(interface_prob, "set_full_evaluation_mode"):
                        interface_prob.set_full_evaluation_mode(False)

            final_payload = {
                "mode": self.ddhs_mode,
                "best_objective_evolution": population[0].get("objective"),
                "final_full_objective": None if final_full_objective is None else float(final_full_objective),
                "final_full_metric_summary": final_trace_summary if self.log_trace_summary else {},
            }
            final_path = os.path.join(self.output_path, "results", "final_full_evaluation.json")
            with open(final_path, "w", encoding="utf-8") as fh:
                json.dump(final_payload, fh, indent=2)

