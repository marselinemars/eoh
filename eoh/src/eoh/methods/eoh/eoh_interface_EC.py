import numpy as np
import time
from .eoh_evolution import Evolution
import warnings
from joblib import Parallel, delayed
from .evaluator_accelerate import add_numba_decorator
import re
import concurrent.futures
import hashlib

class InterfaceEC():
    def __init__(self, pop_size, m, api_endpoint, api_key, llm_model,llm_use_local,llm_local_url, debug_mode, interface_prob, select,n_p,timeout,use_numba,**kwargs):

        # LLM settings
        self.pop_size = pop_size
        self.interface_eval = interface_prob
        prompts = interface_prob.prompts
        self.proposal_mode = kwargs.get("proposal_mode", "eoh")
        self.dx_history_k = kwargs.get("dx_history_k", 5)
        self.dx_history = []
        self.evol = Evolution(api_endpoint, api_key, llm_model,llm_use_local,llm_local_url, debug_mode,prompts, **kwargs)
        self.m = m
        self.debug = debug_mode

        if not self.debug:
            warnings.filterwarnings("ignore")

        self.select = select
        self.n_p = n_p
        
        self.timeout = timeout
        self.use_numba = use_numba

    def _code_hash(self, code):
        if not isinstance(code, str):
            return None
        return hashlib.sha1(code.encode("utf-8", errors="ignore")).hexdigest()

    def _build_dx_context(self, parents, operator):
        if self.proposal_mode != "dx":
            return {}

        parent_context = []
        if parents:
            for p in parents:
                if p is None:
                    continue
                parent_context.append(
                    {
                        "objective": p.get("objective"),
                        "code_hash": self._code_hash(p.get("code")),
                        "trace_summary": p.get("trace_summary"),
                    }
                )

        recent_history = self.dx_history[-self.dx_history_k :] if self.dx_history_k > 0 else []
        return {
            "operator": operator,
            "parent_trace_summary": parent_context,
            "recent_history": recent_history,
        }

    def _evaluate_candidate(self, code):
        if self.proposal_mode != "dx":
            return self.interface_eval.evaluate(code), None

        try:
            out = self.interface_eval.evaluate(code, return_trace=True)
        except TypeError:
            out = self.interface_eval.evaluate(code)

        if isinstance(out, tuple) and len(out) == 2:
            return out[0], out[1]
        return out, None
        
    def code2file(self,code):
        with open("./ael_alg.py", "w") as file:
        # Write the code to the file
            file.write(code)
        return 
    
    def add2pop(self,population,offspring):
        for ind in population:
            if ind['objective'] == offspring['objective']:
                if self.debug:
                    print("duplicated result, retrying ... ")
                return False
        population.append(offspring)
        return True
    
    def check_duplicate(self,population,code):
        for ind in population:
            if code == ind['code']:
                return True
        return False

    # def population_management(self,pop):
    #     # Delete the worst individual
    #     pop_new = heapq.nsmallest(self.pop_size, pop, key=lambda x: x['objective'])
    #     return pop_new
    
    # def parent_selection(self,pop,m):
    #     ranks = [i for i in range(len(pop))]
    #     probs = [1 / (rank + 1 + len(pop)) for rank in ranks]
    #     parents = random.choices(pop, weights=probs, k=m)
    #     return parents

    def population_generation(self):
        
        n_create = 2
        
        population = []

        for i in range(n_create):
            _,pop = self.get_algorithm([],'i1')
            for p in pop:
                population.append(p)
             
        return population
    
    def population_generation_seed(self,seeds,n_p):

        population = []

        if self.proposal_mode == "dx":
            fitness = Parallel(n_jobs=n_p)(
                delayed(self.interface_eval.evaluate)(seed['code'], return_trace=True) for seed in seeds
            )
        else:
            fitness = Parallel(n_jobs=n_p)(delayed(self.interface_eval.evaluate)(seed['code']) for seed in seeds)

        for i in range(len(seeds)):
            try:
                seed_alg = {
                    'algorithm': seeds[i]['algorithm'],
                    'code': seeds[i]['code'],
                    'objective': None,
                    'other_inf': None,
                    'trace_summary': None,
                }

                trace_summary = None
                fit_i = fitness[i]
                if isinstance(fit_i, tuple) and len(fit_i) == 2:
                    fit_i, trace_summary = fit_i
                obj = np.array(fit_i)
                seed_alg['objective'] = np.round(obj, 5)
                seed_alg['trace_summary'] = trace_summary
                seed_alg['other_inf'] = {
                    "proposal_mode": self.proposal_mode,
                    "trace_summary": trace_summary,
                }
                population.append(seed_alg)

            except Exception as e:
                print("Error in seed algorithm")
                exit()

        print("Initiliazation finished! Get "+str(len(seeds))+" seed algorithms")

        return population
    

    def _get_alg(self,pop,operator):
        offspring = {
            'algorithm': None,
            'code': None,
            'objective': None,
            'other_inf': None,
            'trace_summary': None,
        }
        dx_context = {}
        if operator == "i1":
            parents = None
            dx_context = self._build_dx_context(parents, operator)
            [offspring['code'],offspring['algorithm']] =  self.evol.i1(dx_context=dx_context)
        elif operator == "e1":
            parents = self.select.parent_selection(pop,self.m)
            dx_context = self._build_dx_context(parents, operator)
            [offspring['code'],offspring['algorithm']] = self.evol.e1(parents, dx_context=dx_context)
        elif operator == "e2":
            parents = self.select.parent_selection(pop,self.m)
            dx_context = self._build_dx_context(parents, operator)
            [offspring['code'],offspring['algorithm']] = self.evol.e2(parents, dx_context=dx_context) 
        elif operator == "m1":
            parents = self.select.parent_selection(pop,1)
            dx_context = self._build_dx_context(parents, operator)
            [offspring['code'],offspring['algorithm']] = self.evol.m1(parents[0], dx_context=dx_context)
        elif operator == "m2":
            parents = self.select.parent_selection(pop,1)
            dx_context = self._build_dx_context(parents, operator)
            [offspring['code'],offspring['algorithm']] = self.evol.m2(parents[0], dx_context=dx_context)
        elif operator == "m3":
            parents = self.select.parent_selection(pop,1)
            dx_context = self._build_dx_context(parents, operator)
            [offspring['code'],offspring['algorithm']] = self.evol.m3(parents[0], dx_context=dx_context)
        else:
            print(f"Evolution operator [{operator}] has not been implemented ! \n") 

        return parents, offspring

    def get_offspring(self, pop, operator):

        try:
            p, offspring = self._get_alg(pop, operator)
            
            if self.use_numba:
                
                # Regular expression pattern to match function definitions
                pattern = r"def\s+(\w+)\s*\(.*\):"

                # Search for function definitions in the code
                match = re.search(pattern, offspring['code'])

                function_name = match.group(1)

                code = add_numba_decorator(program=offspring['code'], function_name=function_name)
            else:
                code = offspring['code']

            n_retry= 1
            while self.check_duplicate(pop, offspring['code']):
                
                n_retry += 1
                if self.debug:
                    print("duplicated code, wait 1 second and retrying ... ")
                    
                p, offspring = self._get_alg(pop, operator)

                if self.use_numba:
                    # Regular expression pattern to match function definitions
                    pattern = r"def\s+(\w+)\s*\(.*\):"

                    # Search for function definitions in the code
                    match = re.search(pattern, offspring['code'])

                    function_name = match.group(1)

                    code = add_numba_decorator(program=offspring['code'], function_name=function_name)
                else:
                    code = offspring['code']
                    
                if n_retry > 1:
                    break
                
                
            #self.code2file(offspring['code'])
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self._evaluate_candidate, code)
                fitness, trace_summary = future.result(timeout=self.timeout)
                offspring['objective'] = np.round(fitness, 5)
                offspring['trace_summary'] = trace_summary
                future.cancel()        
                # fitness = self.interface_eval.evaluate(code)

                parent_objectives = []
                if p:
                    for parent in p:
                        if parent is not None and parent.get("objective") is not None:
                            parent_objectives.append(float(parent.get("objective")))

                best_parent = min(parent_objectives) if len(parent_objectives) > 0 else None
                delta_objective = None
                if best_parent is not None and fitness is not None:
                    delta_objective = float(np.round(fitness - best_parent, 5))

                offspring['other_inf'] = {
                    "proposal_mode": self.proposal_mode,
                    "parent_objectives": parent_objectives,
                    "delta_objective_vs_best_parent": delta_objective,
                    "trace_summary": trace_summary,
                }

                if self.proposal_mode == "dx":
                    self.dx_history.append(
                        {
                            "operator": operator,
                            "parent_objectives": parent_objectives,
                            "offspring_objective": float(np.round(fitness, 5)) if fitness is not None else None,
                            "delta_objective_vs_best_parent": delta_objective,
                            "trace_summary": trace_summary,
                        }
                    )
                    if self.dx_history_k > 0 and len(self.dx_history) > self.dx_history_k:
                        self.dx_history = self.dx_history[-self.dx_history_k :]
                

        except Exception as e:

            offspring = {
                'algorithm': None,
                'code': None,
                'objective': None,
                'other_inf': None,
                'trace_summary': None,
            }
            p = None

        # Round the objective values
        return p, offspring
    # def process_task(self,pop, operator):
    #     result =  None, {
    #             'algorithm': None,
    #             'code': None,
    #             'objective': None,
    #             'other_inf': None
    #         }
    #     with concurrent.futures.ThreadPoolExecutor() as executor:
    #         future = executor.submit(self.get_offspring, pop, operator)
    #         try:
    #             result = future.result(timeout=self.timeout)
    #             future.cancel()
    #             #print(result)
    #         except:
    #             future.cancel()
                
    #     return result

    
    def get_algorithm(self, pop, operator):
        results = []
        try:
            results = Parallel(n_jobs=self.n_p,timeout=self.timeout+15)(delayed(self.get_offspring)(pop, operator) for _ in range(self.pop_size))
        except Exception as e:
            if self.debug:
                print(f"Error: {e}")
            print("Parallel time out .")
            
        time.sleep(2)


        out_p = []
        out_off = []

        for p, off in results:
            out_p.append(p)
            out_off.append(off)
            if self.debug:
                print(f">>> check offsprings: \n {off}")
        return out_p, out_off
    # def get_algorithm(self,pop,operator, pop_size, n_p):
        
    #     # perform it pop_size times with n_p processes in parallel
    #     p,offspring = self._get_alg(pop,operator)
    #     while self.check_duplicate(pop,offspring['code']):
    #         if self.debug:
    #             print("duplicated code, wait 1 second and retrying ... ")
    #         time.sleep(1)
    #         p,offspring = self._get_alg(pop,operator)
    #     self.code2file(offspring['code'])
    #     try:
    #         fitness= self.interface_eval.evaluate()
    #     except:
    #         fitness = None
    #     offspring['objective'] =  fitness
    #     #offspring['other_inf'] =  first_gap
    #     while (fitness == None):
    #         if self.debug:
    #             print("warning! error code, retrying ... ")
    #         p,offspring = self._get_alg(pop,operator)
    #         while self.check_duplicate(pop,offspring['code']):
    #             if self.debug:
    #                 print("duplicated code, wait 1 second and retrying ... ")
    #             time.sleep(1)
    #             p,offspring = self._get_alg(pop,operator)
    #         self.code2file(offspring['code'])
    #         try:
    #             fitness= self.interface_eval.evaluate()
    #         except:
    #             fitness = None
    #         offspring['objective'] =  fitness
    #         #offspring['other_inf'] =  first_gap
    #     offspring['objective'] = np.round(offspring['objective'],5) 
    #     #offspring['other_inf'] = np.round(offspring['other_inf'],3)
    #     return p,offspring
