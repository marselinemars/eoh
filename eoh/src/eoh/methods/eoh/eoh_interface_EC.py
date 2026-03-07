import numpy as np
import time
from .eoh_evolution import Evolution
import warnings
from joblib import Parallel, delayed
from .evaluator_accelerate import add_numba_decorator
import re
import concurrent.futures
import os

class InterfaceEC():
    def __init__(self, pop_size, m, api_endpoint, api_key, llm_model,llm_use_local,llm_local_url, debug_mode, interface_prob, select,n_p,timeout,use_numba,**kwargs):

        # LLM settings
        self.pop_size = pop_size
        self.interface_eval = interface_prob
        prompts = interface_prob.prompts
        self.evol = Evolution(api_endpoint, api_key, llm_model,llm_use_local,llm_local_url, debug_mode,prompts, **kwargs)
        self.m = m
        self.debug = debug_mode

        if not self.debug:
            warnings.filterwarnings("ignore")

        self.select = select
        self.n_p = n_p
        self.llm_use_local = llm_use_local
        
        self.timeout = timeout
        self.use_numba = use_numba
        self.max_offspring_retries = int(os.getenv("EOH_OFFSPRING_RETRIES", "4"))
        self.parallel_timeout = max(self.timeout + 15, 120)

    def _apply_numba_if_needed(self, code):
        if not self.use_numba:
            return code
        pattern = r"def\s+(\w+)\s*\(.*\):"
        match = re.search(pattern, code)
        if match is None:
            raise RuntimeError("No function definition found in generated code.")
        function_name = match.group(1)
        return add_numba_decorator(program=code, function_name=function_name)

    def _evaluate_code_with_fallback(self, raw_code):
        candidate_codes = []
        try:
            candidate_codes.append(self._apply_numba_if_needed(raw_code))
        except Exception as exc:
            if self.debug:
                print(f"numba decoration failed, using raw code: {exc}")
        candidate_codes.append(raw_code)

        seen = set()
        for candidate_code in candidate_codes:
            if candidate_code in seen:
                continue
            seen.add(candidate_code)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self.interface_eval.evaluate, candidate_code)
                fitness = future.result(timeout=self.timeout)
                future.cancel()
            if fitness is not None:
                return fitness
        raise RuntimeError("Evaluation returned None for both numba and raw code.")
        
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

        fitness = Parallel(n_jobs=n_p)(delayed(self.interface_eval.evaluate)(seed['code']) for seed in seeds)

        for i in range(len(seeds)):
            try:
                seed_alg = {
                    'algorithm': seeds[i]['algorithm'],
                    'code': seeds[i]['code'],
                    'objective': None,
                    'other_inf': None
                }

                obj = np.array(fitness[i])
                seed_alg['objective'] = np.round(obj, 5)
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
            'other_inf': None
        }
        if operator == "i1":
            parents = None
            [offspring['code'],offspring['algorithm']] =  self.evol.i1()            
        elif operator == "e1":
            parents = self.select.parent_selection(pop,self.m)
            [offspring['code'],offspring['algorithm']] = self.evol.e1(parents)
        elif operator == "e2":
            parents = self.select.parent_selection(pop,self.m)
            [offspring['code'],offspring['algorithm']] = self.evol.e2(parents) 
        elif operator == "m1":
            parents = self.select.parent_selection(pop,1)
            [offspring['code'],offspring['algorithm']] = self.evol.m1(parents[0])   
        elif operator == "m2":
            parents = self.select.parent_selection(pop,1)
            [offspring['code'],offspring['algorithm']] = self.evol.m2(parents[0]) 
        elif operator == "m3":
            parents = self.select.parent_selection(pop,1)
            [offspring['code'],offspring['algorithm']] = self.evol.m3(parents[0]) 
        else:
            print(f"Evolution operator [{operator}] has not been implemented ! \n") 

        return parents, offspring

    def get_offspring(self, pop, operator):
        last_error = None
        for attempt in range(1, self.max_offspring_retries + 1):
            try:
                p, offspring = self._get_alg(pop, operator)

                n_retry = 1
                while self.check_duplicate(pop, offspring['code']):
                    n_retry += 1
                    if self.debug:
                        print("duplicated code, retrying ... ")
                    p, offspring = self._get_alg(pop, operator)
                    if n_retry > 1:
                        break

                fitness = self._evaluate_code_with_fallback(offspring['code'])

                offspring['objective'] = np.round(fitness, 5)
                return p, offspring

            except Exception as e:
                last_error = e
                if self.debug:
                    print(f"offspring attempt {attempt}/{self.max_offspring_retries} failed: {e}")
                continue

        if self.debug:
            print(f"all offspring attempts failed for operator {operator}: {last_error}")
        offspring = {
            'algorithm': None,
            'code': None,
            'objective': None,
            'other_inf': None
        }
        p = None
        return p, offspring

    def get_offspring_from_prompt(self, pop, prompt_content):
        last_error = None
        for attempt in range(1, self.max_offspring_retries + 1):
            try:
                offspring = {
                    'algorithm': None,
                    'code': None,
                    'objective': None,
                    'other_inf': None
                }
                offspring['code'], offspring['algorithm'] = self.evol.generate_from_prompt(prompt_content)

                if self.check_duplicate(pop, offspring['code']):
                    raise RuntimeError("Generated duplicate code.")

                fitness = self._evaluate_code_with_fallback(offspring['code'])

                offspring['objective'] = np.round(fitness, 5)
                return offspring

            except Exception as e:
                last_error = e
                if self.debug:
                    print(f"planner offspring attempt {attempt}/{self.max_offspring_retries} failed: {e}")
                continue

        if self.debug:
            print(f"planner offspring generation failed: {last_error}")
        return {
            'algorithm': None,
            'code': None,
            'objective': None,
            'other_inf': None
        }
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

    
    def _get_algorithm_sequential(self, pop, operator):
        results = []
        for _ in range(self.pop_size):
            results.append(self.get_offspring(pop, operator))
        return results

    def get_algorithm(self, pop, operator):
        results = []
        use_parallel = self.llm_use_local and self.n_p > 1
        if use_parallel:
            try:
                results = Parallel(n_jobs=self.n_p, timeout=self.parallel_timeout)(
                    delayed(self.get_offspring)(pop, operator) for _ in range(self.pop_size)
                )
            except Exception as e:
                if self.debug:
                    print(f"parallel offspring generation failed: {e}")
                print("Parallel generation failed. Retrying sequentially.")
                results = self._get_algorithm_sequential(pop, operator)
        else:
            results = self._get_algorithm_sequential(pop, operator)

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
