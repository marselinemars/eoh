import numpy as np
import time
from .eoh_evolution import Evolution
import warnings
from joblib import Parallel, delayed
from .evaluator_accelerate import add_numba_decorator
import re
import concurrent.futures
import os
import hashlib
import random

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
        
        self.timeout = timeout
        self.use_numba = use_numba
        self.max_offspring_retries = int(os.getenv("EOH_OFFSPRING_RETRIES", "4"))
        self.parallel_backend = os.getenv("EOH_PARALLEL_BACKEND", "loky")
        self.parallel_fallback_sequential = os.getenv("EOH_PARALLEL_FALLBACK_SEQUENTIAL", "1") == "1"
        self.generation_timeout = max(
            int(self.timeout) + 15,
            int(os.getenv("EOH_GENERATION_TIMEOUT_S", "900")),
        )
        self.max_parallel_llm_requests = max(1, int(os.getenv("EOH_MAX_PARALLEL_LLM_REQUESTS", "2")))
        self.max_parallel_i1_requests = max(1, int(os.getenv("EOH_MAX_PARALLEL_I1_REQUESTS", "1")))
        self.controller_parent_mix = None
        self.controller_prompt_modifiers = []
        self.controller_preferred_parent_hashes = set()
        self.controller_experience_context = ""
        self._last_batch_stats = {
            "generation_time_s": 0.0,
            "evaluation_time_s": 0.0,
            "invalid_before_eval_count": 0,
        }

    def _format_error(self, error, default="unknown_error"):
        if error is None:
            return default
        name = type(error).__name__
        text = str(error).strip()
        if text:
            return f"{name}: {text}"
        return name

    def set_controller_context(self, parent_mix=None, prompt_modifiers=None, preferred_parent_hashes=None, experience_context=None):
        if isinstance(parent_mix, dict):
            self.controller_parent_mix = dict(parent_mix)
        else:
            self.controller_parent_mix = None

        if isinstance(prompt_modifiers, list):
            self.controller_prompt_modifiers = [str(x).strip() for x in prompt_modifiers if str(x).strip()][:4]
        else:
            self.controller_prompt_modifiers = []
        if isinstance(preferred_parent_hashes, list):
            self.controller_preferred_parent_hashes = {str(x).strip() for x in preferred_parent_hashes if str(x).strip()}
        else:
            self.controller_preferred_parent_hashes = set()
        self.controller_experience_context = str(experience_context).strip() if experience_context is not None else ""
        self.evol.set_prompt_modifiers(self.controller_prompt_modifiers)
        self.evol.set_experience_context(self.controller_experience_context)

    def _normalize_mix(self, mix):
        keys = ["elite", "diverse", "random"]
        if isinstance(mix, dict) and "preferred" in mix:
            keys.append("preferred")
        vals = {}
        for k in keys:
            try:
                vals[k] = max(0.0, float(mix.get(k, 0.0)))
            except (TypeError, ValueError):
                vals[k] = 0.0
        total = sum(vals.values())
        if total <= 1e-12:
            if "preferred" in keys:
                return {"elite": 0.20, "diverse": 0.10, "random": 0.10, "preferred": 0.60}
            return {"elite": 0.5, "diverse": 0.3, "random": 0.2}
        return {k: vals[k] / total for k in keys}

    def _build_parent_pools(self, pop):
        if len(pop) == 0:
            return {"elite": [], "diverse": [], "random": []}
        sorted_pop = sorted(pop, key=lambda x: x.get("objective", float("inf")))
        elite_count = max(1, int(np.ceil(0.3 * len(sorted_pop))))
        elite_pool = sorted_pop[:elite_count]
        non_elite = sorted_pop[elite_count:]
        if len(non_elite) == 0:
            non_elite = sorted_pop
        seen = set()
        diverse_pool = []
        for ind in non_elite:
            code = ind.get("code")
            if not isinstance(code, str):
                continue
            code_hash = hashlib.sha1(code.encode("utf-8")).hexdigest()
            if code_hash in seen:
                continue
            seen.add(code_hash)
            diverse_pool.append(ind)
        if len(diverse_pool) == 0:
            diverse_pool = non_elite
        preferred_pool = []
        if len(self.controller_preferred_parent_hashes) > 0:
            for ind in sorted_pop:
                code = ind.get("code")
                if not isinstance(code, str):
                    continue
                code_hash = hashlib.sha1(code.encode("utf-8")).hexdigest()[:12]
                if code_hash in self.controller_preferred_parent_hashes:
                    preferred_pool.append(ind)
        return {
            "elite": elite_pool,
            "diverse": diverse_pool,
            "random": sorted_pop,
            "preferred": preferred_pool,
        }

    def _select_parents(self, pop, m):
        if len(pop) == 0:
            return []
        if self.controller_parent_mix is None:
            return self.select.parent_selection(pop, m)
        mix = self._normalize_mix(self.controller_parent_mix)
        pools = self._build_parent_pools(pop)
        parents = []
        keys = ["preferred", "elite", "diverse", "random"] if "preferred" in mix else ["elite", "diverse", "random"]
        for _ in range(m):
            draw = random.random()
            cum = 0.0
            picked_bucket = "random"
            for key in keys:
                cum += mix[key]
                if draw <= cum:
                    picked_bucket = key
                    break
            pool = pools.get(picked_bucket, [])
            if len(pool) == 0:
                pool = pools["random"]
            parents.append(random.choice(pool))
        return parents
        
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
            _,pop = self.get_algorithm([], 'i1')
            for p in pop:
                population.append(p)
             
        return population
    
    def population_generation_seed(self,seeds,n_p):

        population = []

        evaluations = Parallel(n_jobs=n_p)(delayed(self._evaluate_candidate)(seed['code']) for seed in seeds)

        for i in range(len(seeds)):
            try:
                seed_alg = {
                    'algorithm': seeds[i]['algorithm'],
                    'code': seeds[i]['code'],
                    'objective': None,
                    'other_inf': {
                        "lineage": {
                            "created_by": "seed",
                            "parent_ids": [],
                            "parent_hashes": [],
                            "note": "seed_population",
                        }
                    }
                }

                eval_result = evaluations[i]
                fitness = eval_result.get("fitness") if isinstance(eval_result, dict) else eval_result
                obj = np.array(fitness)
                seed_alg['objective'] = np.round(obj, 5)
                if isinstance(eval_result, dict) and isinstance(eval_result.get("details"), dict):
                    seed_alg['other_inf'].update(eval_result.get("details"))
                population.append(seed_alg)

            except Exception as e:
                print("Error in seed algorithm")
                exit()

        print("Initiliazation finished! Get "+str(len(seeds))+" seed algorithms")

        return population
    

    def _get_alg(self,pop,operator):
        self.evol.set_prompt_modifiers(self.controller_prompt_modifiers)
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
            parents = self._select_parents(pop,self.m)
            [offspring['code'],offspring['algorithm']] = self.evol.e1(parents)
        elif operator == "e2":
            parents = self._select_parents(pop,self.m)
            [offspring['code'],offspring['algorithm']] = self.evol.e2(parents) 
        elif operator == "m1":
            parents = self._select_parents(pop,1)
            [offspring['code'],offspring['algorithm']] = self.evol.m1(parents[0])   
        elif operator == "m2":
            parents = self._select_parents(pop,1)
            [offspring['code'],offspring['algorithm']] = self.evol.m2(parents[0]) 
        elif operator == "m3":
            parents = self._select_parents(pop,1)
            [offspring['code'],offspring['algorithm']] = self.evol.m3(parents[0]) 
        else:
            print(f"Evolution operator [{operator}] has not been implemented ! \n") 

        return parents, offspring

    def _get_alg_from_prompt(self, prompt_content, parent_cards=None):
        self.evol.set_prompt_modifiers(self.controller_prompt_modifiers)
        offspring = {
            'algorithm': None,
            'code': None,
            'objective': None,
            'other_inf': None
        }
        [offspring['code'], offspring['algorithm']] = self.evol.generate_from_prompt(prompt_content)
        parents = list(parent_cards) if isinstance(parent_cards, list) else None
        return parents, offspring

    def _evaluate_candidate(self, code):
        if hasattr(self.interface_eval, "evaluate_with_details"):
            details = self.interface_eval.evaluate_with_details(code, split="train")
            if isinstance(details, dict):
                return {
                    "fitness": details.get("fitness"),
                    "details": details,
                }
        fitness = self.interface_eval.evaluate(code)
        return {
            "fitness": fitness,
            "details": None,
        }

    def _empty_batch_stats(self):
        return {
            "generation_time_s": 0.0,
            "evaluation_time_s": 0.0,
            "invalid_before_eval_count": 0,
        }

    def _merge_batch_stats(self, stats_list):
        merged = self._empty_batch_stats()
        for stats in stats_list:
            if not isinstance(stats, dict):
                continue
            merged["generation_time_s"] += float(stats.get("generation_time_s", 0.0) or 0.0)
            merged["evaluation_time_s"] += float(stats.get("evaluation_time_s", 0.0) or 0.0)
            merged["invalid_before_eval_count"] += int(stats.get("invalid_before_eval_count", 0) or 0)
        self._last_batch_stats = merged

    def get_last_batch_stats(self):
        return dict(self._last_batch_stats)

    def get_offspring(self, pop, operator):
        last_error = None
        local_stats = self._empty_batch_stats()
        for attempt in range(1, self.max_offspring_retries + 1):
            eval_started = False
            try:
                gen_start = time.time()
                p, offspring = self._get_alg(pop, operator)
                local_stats["generation_time_s"] += float(time.time() - gen_start)

                if self.use_numba:
                    pattern = r"def\s+(\w+)\s*\(.*\):"
                    match = re.search(pattern, offspring['code'])
                    if match is None:
                        raise RuntimeError("No function definition found in generated code.")
                    function_name = match.group(1)
                    code = add_numba_decorator(program=offspring['code'], function_name=function_name)
                else:
                    code = offspring['code']

                n_retry = 1
                while self.check_duplicate(pop, offspring['code']):
                    n_retry += 1
                    local_stats["invalid_before_eval_count"] += 1
                    if self.debug:
                        print("duplicated code, retrying ... ")
                    gen_start = time.time()
                    p, offspring = self._get_alg(pop, operator)
                    local_stats["generation_time_s"] += float(time.time() - gen_start)
                    if self.use_numba:
                        pattern = r"def\s+(\w+)\s*\(.*\):"
                        match = re.search(pattern, offspring['code'])
                        if match is None:
                            raise RuntimeError("No function definition found in generated code.")
                        function_name = match.group(1)
                        code = add_numba_decorator(program=offspring['code'], function_name=function_name)
                    else:
                        code = offspring['code']
                    if n_retry > 1:
                        break

                eval_started = True
                eval_start = time.time()
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self._evaluate_candidate, code)
                    try:
                        eval_result = future.result(timeout=self.timeout)
                    except concurrent.futures.TimeoutError as exc:
                        future.cancel()
                        raise RuntimeError(f"evaluation_timeout after {self.timeout}s") from exc
                    future.cancel()
                local_stats["evaluation_time_s"] += float(time.time() - eval_start)

                fitness = eval_result.get("fitness") if isinstance(eval_result, dict) else eval_result
                if fitness is None:
                    raise RuntimeError("Evaluation returned None.")

                offspring['objective'] = np.round(fitness, 5)
                if isinstance(eval_result, dict):
                    offspring['other_inf'] = eval_result.get("details")
                return p, offspring, local_stats

            except Exception as e:
                last_error = e
                if not eval_started:
                    local_stats["invalid_before_eval_count"] += 1
                if self.debug:
                    print(f"offspring attempt {attempt}/{self.max_offspring_retries} failed: {e}")
                continue

        if self.debug:
            print(f"all offspring attempts failed for operator {operator}: {last_error}")
        offspring = {
            'algorithm': None,
            'code': None,
            'objective': None,
            'other_inf': {
                "generation_failed": True,
                "failure_reason": self._format_error(last_error, "unknown_generation_failure"),
                "failure_operator": operator,
            }
        }
        p = None
        return p, offspring, local_stats

    def get_offspring_from_prompt(self, pop, prompt_content, parent_cards=None):
        last_error = None
        local_stats = self._empty_batch_stats()
        for attempt in range(1, self.max_offspring_retries + 1):
            eval_started = False
            try:
                gen_start = time.time()
                p, offspring = self._get_alg_from_prompt(prompt_content, parent_cards=parent_cards)
                local_stats["generation_time_s"] += float(time.time() - gen_start)

                if self.use_numba:
                    pattern = r"def\s+(\w+)\s*\(.*\):"
                    match = re.search(pattern, offspring['code'])
                    if match is None:
                        raise RuntimeError("No function definition found in generated code.")
                    function_name = match.group(1)
                    code = add_numba_decorator(program=offspring['code'], function_name=function_name)
                else:
                    code = offspring['code']

                n_retry = 1
                while self.check_duplicate(pop, offspring['code']):
                    n_retry += 1
                    local_stats["invalid_before_eval_count"] += 1
                    if self.debug:
                        print("duplicated code, retrying ... ")
                    gen_start = time.time()
                    p, offspring = self._get_alg_from_prompt(prompt_content, parent_cards=parent_cards)
                    local_stats["generation_time_s"] += float(time.time() - gen_start)
                    if self.use_numba:
                        pattern = r"def\s+(\w+)\s*\(.*\):"
                        match = re.search(pattern, offspring['code'])
                        if match is None:
                            raise RuntimeError("No function definition found in generated code.")
                        function_name = match.group(1)
                        code = add_numba_decorator(program=offspring['code'], function_name=function_name)
                    else:
                        code = offspring['code']
                    if n_retry > 1:
                        break

                eval_started = True
                eval_start = time.time()
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self._evaluate_candidate, code)
                    try:
                        eval_result = future.result(timeout=self.timeout)
                    except concurrent.futures.TimeoutError as exc:
                        future.cancel()
                        raise RuntimeError(f"evaluation_timeout after {self.timeout}s") from exc
                    future.cancel()
                local_stats["evaluation_time_s"] += float(time.time() - eval_start)

                fitness = eval_result.get("fitness") if isinstance(eval_result, dict) else eval_result
                if fitness is None:
                    raise RuntimeError("Evaluation returned None.")

                offspring['objective'] = np.round(fitness, 5)
                if isinstance(eval_result, dict):
                    offspring['other_inf'] = eval_result.get("details")
                return p, offspring, local_stats

            except Exception as e:
                last_error = e
                if not eval_started:
                    local_stats["invalid_before_eval_count"] += 1
                if self.debug:
                    print(f"custom offspring attempt {attempt}/{self.max_offspring_retries} failed: {e}")
                continue

        if self.debug:
            print(f"all custom offspring attempts failed: {last_error}")
        offspring = {
            'algorithm': None,
            'code': None,
            'objective': None,
            'other_inf': {
                "generation_failed": True,
                "failure_reason": self._format_error(last_error, "unknown_custom_generation_failure"),
                "failure_operator": "custom_prompt",
            }
        }
        return None, offspring, local_stats
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

    
    def get_algorithm(self, pop, operator, n_offspring=None, max_parallel_requests=None):
        if n_offspring is None:
            n_targets = int(self.pop_size)
        else:
            try:
                n_targets = int(n_offspring)
            except (TypeError, ValueError):
                n_targets = int(self.pop_size)
        n_targets = max(0, n_targets)
        if n_targets == 0:
            return [], []

        results = []
        parallel_jobs = min(self.n_p, self.max_parallel_llm_requests)
        if max_parallel_requests is not None:
            try:
                parallel_jobs = min(parallel_jobs, max(1, int(max_parallel_requests)))
            except (TypeError, ValueError):
                pass
        if operator == "i1":
            parallel_jobs = min(parallel_jobs, self.max_parallel_i1_requests)
        parallel_jobs = max(1, parallel_jobs)
        try:
            results = Parallel(
                n_jobs=parallel_jobs,
                timeout=self.generation_timeout,
                backend=self.parallel_backend,
                batch_size=1,
            )(delayed(self.get_offspring)(pop, operator) for _ in range(n_targets))
        except Exception as e:
            print(f"Parallel offspring generation failed for operator {operator}: {type(e).__name__}: {e}")
            print("Falling back to sequential offspring generation.")
            results = []
            for _ in range(n_targets):
                results.append(self.get_offspring(pop, operator))
            
        time.sleep(2)


        out_p = []
        out_off = []
        stats_list = []

        for p, off, stats in results:
            out_p.append(p)
            out_off.append(off)
            stats_list.append(stats)
            if self.debug:
                print(f">>> check offsprings: \n {off}")
        self._merge_batch_stats(stats_list)
        return out_p, out_off

    def get_algorithm_from_prompt(self, pop, prompt_content, n_offspring=None, parent_cards=None, max_parallel_requests=None):
        if n_offspring is None:
            n_targets = int(self.pop_size)
        else:
            try:
                n_targets = int(n_offspring)
            except (TypeError, ValueError):
                n_targets = int(self.pop_size)
        n_targets = max(0, n_targets)
        if n_targets == 0:
            return [], []

        parallel_jobs = min(self.n_p, self.max_parallel_llm_requests)
        if max_parallel_requests is not None:
            try:
                parallel_jobs = min(parallel_jobs, max(1, int(max_parallel_requests)))
            except (TypeError, ValueError):
                pass
        parallel_jobs = max(1, parallel_jobs)
        results = []
        try:
            results = Parallel(
                n_jobs=parallel_jobs,
                timeout=self.generation_timeout,
                backend=self.parallel_backend,
                batch_size=1,
            )(delayed(self.get_offspring_from_prompt)(pop, prompt_content, parent_cards=parent_cards) for _ in range(n_targets))
        except Exception as e:
            print(f"Parallel offspring generation failed for custom prompt: {type(e).__name__}: {e}")
            print("Falling back to sequential offspring generation.")
            results = []
            for _ in range(n_targets):
                results.append(self.get_offspring_from_prompt(pop, prompt_content, parent_cards=parent_cards))

        time.sleep(2)

        out_p = []
        out_off = []
        stats_list = []
        for p, off, stats in results:
            out_p.append(p)
            out_off.append(off)
            stats_list.append(stats)
            if self.debug:
                print(f">>> check custom offsprings: \n {off}")
        self._merge_batch_stats(stats_list)
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
