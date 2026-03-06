import concurrent.futures
import hashlib
import json
import os
import random
import time

import numpy as np

from .eoh_interface_EC import InterfaceEC
from .population_planner import (
    PopulationPlanner,
    PopulationPlannerExecutor,
    PopulationProfiler,
    PopulationSelector,
    build_population_summary,
)


class EOH:
    def __init__(self, paras, problem, select, manage, **kwargs):
        self.prob = problem
        self.select = select
        self.manage = manage
        self.use_local_llm = paras.llm_use_local
        self.llm_local_url = paras.llm_local_url
        self.api_endpoint = paras.llm_api_endpoint
        self.api_key = paras.llm_api_key
        self.llm_model = paras.llm_model
        self.pop_size = paras.ec_pop_size
        self.n_pop = paras.ec_n_pop
        self.operators = paras.ec_operators
        self.operator_weights = paras.ec_operator_weights
        self.m = 2 if paras.ec_m > self.pop_size or paras.ec_m == 1 else paras.ec_m
        self.debug_mode = paras.exp_debug_mode
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
        if self.mode not in ["baseline", "routed", "population_planner"]:
            self.mode = "baseline"
        self.route_improvement_epsilon = float(getattr(paras, "route_improvement_epsilon", 1e-12))
        self.route_recent_window = int(getattr(paras, "route_recent_window", 3))
        self.route_structural_plateau_eps = float(getattr(paras, "route_structural_plateau_eps", 1e-5))
        self.route_warmup_gens = max(0, int(getattr(paras, "route_warmup_gens", 2)))
        self.route_e1_cooldown = max(0, int(getattr(paras, "route_e1_cooldown", 3)))
        self.route_e2_recent_k = max(1, int(getattr(paras, "route_e2_recent_k", 3)))
        self.route_use_probabilistic = bool(getattr(paras, "route_use_probabilistic", True))
        self.route_use_diversity = bool(getattr(paras, "route_use_diversity", True))
        self.holdout_eval_interval = max(1, int(getattr(paras, "holdout_eval_interval", 1)))
        self.log_full_population = bool(getattr(paras, "log_full_population", False))
        self.planner_profile_train_instances = int(getattr(paras, "planner_profile_train_instances", 8))
        self.planner_profile_holdout_instances = int(getattr(paras, "planner_profile_holdout_instances", 8))
        self.planner_view_size = max(4, int(getattr(paras, "planner_view_size", 8)))
        results_dir = os.path.join(self.output_path, "results")
        self.run_log_path = os.path.join(results_dir, "run_log.jsonl")
        self.operator_log_path = os.path.join(results_dir, "operator_events.jsonl")
        self.heuristic_cards_log_path = os.path.join(results_dir, "heuristic_cards.jsonl")
        self.population_summary_log_path = os.path.join(results_dir, "population_summary.jsonl")
        self.population_planner_log_path = os.path.join(results_dir, "population_planner_output.jsonl")
        self.executed_interventions_log_path = os.path.join(results_dir, "executed_interventions.jsonl")
        self.offspring_lineage_log_path = os.path.join(results_dir, "offspring_lineage.jsonl")
        print("- EoH parameters loaded -")
        random.seed(2024)

    def _write_jsonl(self, path, record):
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _prepare_run_log(self):
        os.makedirs(os.path.dirname(self.run_log_path), exist_ok=True)
        for path in [
            self.run_log_path,
            self.operator_log_path,
            self.heuristic_cards_log_path,
            self.population_summary_log_path,
            self.population_planner_log_path,
            self.executed_interventions_log_path,
            self.offspring_lineage_log_path,
        ]:
            with open(path, "w", encoding="utf-8") as _:
                pass

    def _code_hash(self, code):
        return hashlib.sha1(code.encode("utf-8")).hexdigest()[:12] if code is not None else None

    def _save_population(self, population, generation):
        pop_path = os.path.join(self.output_path, "results", "pops", f"population_generation_{generation}.json")
        best_path = os.path.join(self.output_path, "results", "pops_best", f"population_generation_{generation}.json")
        os.makedirs(os.path.dirname(pop_path), exist_ok=True)
        os.makedirs(os.path.dirname(best_path), exist_ok=True)
        compact = population if self.log_full_population else [
            {
                "id": item.get("id"),
                "objective": item.get("objective"),
                "code_hash": self._code_hash(item.get("code")),
                "created_by": item.get("created_by"),
                "parent_ids": item.get("parent_ids", []),
            }
            for item in population
        ]
        with open(pop_path, "w", encoding="utf-8") as f:
            json.dump(compact, f, indent=2)
        with open(best_path, "w", encoding="utf-8") as f:
            json.dump(compact[0] if compact else {}, f, indent=2)

    def _best_objective(self, population):
        return population[0]["objective"] if population else None

    def _invalid_rate(self, offspring_list):
        if not offspring_list:
            return 1.0
        invalid = 0
        for offspring in offspring_list:
            if offspring is None or offspring.get("objective") is None or offspring.get("code") is None:
                invalid += 1
        return invalid / len(offspring_list)

    def _count_valid(self, offspring_list):
        return sum(1 for offspring in offspring_list if offspring is not None and offspring.get("objective") is not None and offspring.get("code") is not None)

    def _population_diversity(self, population):
        if not self.route_use_diversity:
            return None
        return len({self._code_hash(item.get("code")) for item in population if item.get("code") is not None})

    def _population_avg_code_length(self, population):
        lengths = [len(item.get("code")) for item in population if isinstance(item.get("code"), str)]
        return float(np.mean(np.array(lengths))) if lengths else None

    def _evaluate_best_on_holdout(self, interface_prob, population, generation_index):
        if not population or ((generation_index + 1) % self.holdout_eval_interval) != 0 or not hasattr(interface_prob, "evaluate_on_split"):
            return None, False
        code = population[0].get("code")
        if code is None:
            return None, True
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(interface_prob.evaluate_on_split, code, "holdout")
                result = future.result(timeout=max(20, self.timeout * 2))
                future.cancel()
            return (float(result), True) if result is not None else (None, True)
        except Exception:
            return None, True

    def _route_operator(self, diagnosis_label, generation_index, no_improve_gens, recent_ops, current_best, best_at_last_e2, e1_cooldown_remaining):
        if (generation_index + 1) <= self.route_warmup_gens or e1_cooldown_remaining > 0:
            return "e2" if "e2" in self.operators else self.operators[0]
        if diagnosis_label == "OVERFIT_RISK":
            return "m3" if "m3" in self.operators else "m1"
        if diagnosis_label == "NEED_PARAM_TUNING":
            return "m2" if "m2" in self.operators else "m1"
        if diagnosis_label == "NEED_STRUCTURAL_EDIT":
            return "m1" if "m1" in self.operators else "e2"
        if diagnosis_label == "NEED_GLOBAL_NOVELTY" and no_improve_gens >= 4 and any(op == "e2" for op in recent_ops[-self.route_e2_recent_k :]):
            if current_best is None or best_at_last_e2 is None or not (current_best < (best_at_last_e2 - self.route_improvement_epsilon)):
                return "e1" if "e1" in self.operators else "e2"
        return "e2" if "e2" in self.operators else self.operators[0]

    def _routed_diagnosis(self, last_improved, no_improve_gens, overfit_risk, recent_ops, recent_deltas, diversity, code_length_growth):
        if overfit_risk:
            return "OVERFIT_RISK"
        if last_improved:
            return "NEED_PARAM_TUNING"
        m2_deltas = [recent_deltas[i] for i, op in enumerate(recent_ops) if op == "m2"]
        if no_improve_gens >= 2 and m2_deltas and all((delta is None) or (delta <= self.route_structural_plateau_eps) for delta in m2_deltas):
            low_div = diversity is not None and diversity <= max(2, self.pop_size // 2)
            growing = code_length_growth is not None and code_length_growth > 0.0
            if low_div or growing:
                return "NEED_STRUCTURAL_EDIT"
        return "NEED_GLOBAL_NOVELTY" if no_improve_gens >= 3 else "NEED_BACKBONE_VARIANT"

    def _reset_counter(self, population):
        seen = 0
        for item in population:
            heuristic_id = str(item.get("id", ""))
            if heuristic_id.startswith("H"):
                try:
                    seen = max(seen, int(heuristic_id[1:]))
                except ValueError:
                    pass
        self._heuristic_counter = seen

    def _next_id(self):
        self._heuristic_counter += 1
        return f"H{self._heuristic_counter}"

    def _stamp(self, individual, generation, created_by, parent_ids=None, planner_goal=None, planner_instruction=None):
        if individual.get("id") in [None, "", "None"]:
            individual["id"] = self._next_id()
        individual["generation"] = int(generation)
        individual["created_by"] = str(created_by)
        individual["parent_ids"] = [str(item) for item in (parent_ids or []) if item is not None]
        if planner_goal is not None:
            individual["planner_goal"] = str(planner_goal)
        if planner_instruction is not None:
            individual["planner_instruction"] = str(planner_instruction)
        return individual

    def _planner_manage(self, population, preserve_ids=None, deprioritize_ids=None):
        preserve_ids = set(str(item) for item in (preserve_ids or []))
        deprioritize_ids = set(str(item) for item in (deprioritize_ids or []))
        filtered = [item for item in population if item.get("objective") is not None and item.get("code") is not None]
        unique = []
        seen_hash = set()
        for item in sorted(filtered, key=lambda x: float(x.get("objective"))):
            code_hash = self._code_hash(item.get("code"))
            if code_hash in seen_hash:
                continue
            seen_hash.add(code_hash)
            unique.append(item)
        preserved = [item for item in unique if str(item.get("id")) in preserve_ids]
        remainder = [item for item in unique if str(item.get("id")) not in preserve_ids]
        preserved = sorted(preserved, key=lambda x: float(x.get("objective")))
        remainder = sorted(remainder, key=lambda x: (str(x.get("id")) in deprioritize_ids, float(x.get("objective"))))
        return (preserved + remainder)[: self.pop_size]

    def _planner_generation(self, gen, population, interface_prob, interface_ec, planner, profiler, selector, best_history, no_improve_gens, last_major_improvement_generation):
        cards = profiler.profile_population(population, generation=gen)
        summary = build_population_summary(cards, gen + 1, best_history, no_improve_gens, last_major_improvement_generation)
        self._write_jsonl(self.heuristic_cards_log_path, {"gen": int(gen + 1), "cards": [card.to_dict() for card in cards]})
        self._write_jsonl(self.population_summary_log_path, {"gen": int(gen + 1), "summary": summary.to_dict()})
        selected_cards = selector.select(cards)
        context = interface_prob.get_problem_context() if hasattr(interface_prob, "get_problem_context") else str(interface_prob)
        planner_result = planner.plan(context, summary, selected_cards, generation_budget=self.pop_size)
        plan = planner_result["plan"]
        self._write_jsonl(self.population_planner_log_path, {"gen": int(gen + 1), "planner_view_ids": [card.id for card in selected_cards], "summary": summary.to_dict(), "plan": plan.to_dict(), "raw_output": planner_result["raw_text"], "fallback_used": bool(planner_result["fallback_used"])})
        executor = PopulationPlannerExecutor(interface_ec, summary, selected_cards)
        remaining = self.pop_size
        pool = list(population)
        offsprings = []
        chosen_modes = []
        for intervention in executor.sort_interventions(plan.interventions):
            parent_cards = [card for card in selected_cards if card.id in intervention.targets]
            before = self._best_objective(self._planner_manage(pool, plan.preserve_ids, plan.deprioritize_ids))
            produced = 0
            status = "executed"
            if intervention.execution_mode == "evaluate":
                status = "evaluated_only"
            elif remaining <= 0:
                status = "budget_exhausted"
            else:
                prompt = executor.build_prompt(intervention)
                request_n = min(int(intervention.offspring_count), remaining)
                for _ in range(request_n):
                    child = interface_ec.get_offspring_from_prompt(pool + offsprings, prompt)
                    if child.get("objective") is None or child.get("code") is None:
                        continue
                    self._stamp(child, gen + 1, intervention.execution_mode, intervention.targets, intervention.goal, intervention.instruction)
                    pool.append(child)
                    offsprings.append(child)
                    child_card = profiler.profile_individual(child, generation=gen + 1, rank=0)
                    parent_fitness = [card.fitness for card in parent_cards]
                    self._write_jsonl(self.offspring_lineage_log_path, {
                        "gen": int(gen + 1),
                        "child_id": child.get("id"),
                        "created_by": child.get("created_by"),
                        "parent_ids": child.get("parent_ids", []),
                        "planner_goal": intervention.goal,
                        "planner_instruction": intervention.instruction,
                        "child_fitness": child.get("objective"),
                        "parent_fitness": parent_fitness,
                        "fitness_delta_to_best_parent": (float(min(parent_fitness) - child.get("objective")) if parent_fitness else None),
                        "mean_residual_delta": (float(child_card.behavior.mean_residual_ratio - np.mean(np.array([card.behavior.mean_residual_ratio for card in parent_cards]))) if parent_cards else None),
                        "fragmentation_delta": (float(child_card.behavior.fragmentation_index - np.mean(np.array([card.behavior.fragmentation_index for card in parent_cards]))) if parent_cards else None),
                        "early_open_rate_delta": (float(child_card.behavior.resource_opening_rate_early - np.mean(np.array([card.behavior.resource_opening_rate_early for card in parent_cards]))) if parent_cards else None),
                        "order_sensitivity_delta": (float(child_card.behavior.order_sensitivity - np.mean(np.array([card.behavior.order_sensitivity for card in parent_cards]))) if parent_cards else None),
                    })
                    produced += 1
                    remaining -= 1
            after = self._best_objective(self._planner_manage(pool, plan.preserve_ids, plan.deprioritize_ids))
            delta = float(before - after) if before is not None and after is not None else None
            self._write_jsonl(self.executed_interventions_log_path, {"gen": int(gen + 1), "targets": list(intervention.targets), "execution_mode": intervention.execution_mode, "goal": intervention.goal, "instruction": intervention.instruction, "offspring_requested": int(intervention.offspring_count), "offspring_generated": int(produced), "priority": intervention.priority, "status": status, "best_before": before, "best_after": after, "delta_best": delta})
            self._write_jsonl(self.operator_log_path, {"gen": int(gen + 1), "mode": self.mode, "operator": intervention.execution_mode, "operator_weight": None, "executed": bool(status == "executed"), "diagnosis_label": summary.current_search_regime, "targets": list(intervention.targets), "goal": intervention.goal, "instruction": intervention.instruction, "priority": intervention.priority, "n_offspring": int(produced), "n_valid": int(produced), "n_invalid": int(max(0, intervention.offspring_count - produced)) if intervention.execution_mode != "evaluate" else 0, "invalid_rate": (float(max(0, intervention.offspring_count - produced) / max(1, intervention.offspring_count)) if intervention.execution_mode != "evaluate" else 0.0), "best_before": before, "best_after": after, "delta_best": delta})
            if produced > 0:
                chosen_modes.append(f"{intervention.execution_mode}x{produced}")
        return self._planner_manage(pool, plan.preserve_ids, plan.deprioritize_ids), offsprings, summary.current_search_regime, ("planner:" + ",".join(chosen_modes) if chosen_modes else "planner:none")

    def run(self):
        print("- Evolution Start -")
        time_start = time.time()
        interface_prob = self.prob
        interface_ec = InterfaceEC(self.pop_size, self.m, self.api_endpoint, self.api_key, self.llm_model, self.use_local_llm, self.llm_local_url, self.debug_mode, interface_prob, select=self.select, n_p=self.exp_n_proc, timeout=self.timeout, use_numba=self.use_numba)
        self._prepare_run_log()
        planner = PopulationPlanner(self.api_endpoint, self.api_key, self.llm_model, self.use_local_llm, self.llm_local_url, debug_mode=self.debug_mode) if self.mode == "population_planner" else None
        profiler = PopulationProfiler(interface_prob, self.planner_profile_train_instances, self.planner_profile_holdout_instances) if self.mode == "population_planner" else None
        selector = PopulationSelector(self.planner_view_size) if self.mode == "population_planner" else None

        if self.use_seed:
            with open(self.seed_path) as file:
                population = interface_ec.population_generation_seed(json.load(file), self.exp_n_proc)
            self._reset_counter(population)
            for item in population:
                self._stamp(item, 0, "seed")
            start_gen = 0
        elif self.load_pop:
            with open(self.load_pop_path) as file:
                population = list(json.load(file))
            self._reset_counter(population)
            for item in population:
                self._stamp(item, 0, "resume")
            start_gen = self.load_pop_id
        else:
            population = interface_ec.population_generation()
            population = self.manage.population_management(population, self.pop_size)
            self._heuristic_counter = 0
            for item in population:
                self._stamp(item, 0, "i1")
            start_gen = 0
        self._save_population(population, 0)

        prev_best = population[0]["objective"] if population else None
        best_history = [float(prev_best)] if prev_best is not None else []
        no_improve_gens = 0
        last_improved = False
        recent_ops = []
        recent_deltas = []
        last_diversity = self._population_diversity(population)
        prev_avg_code_length = self._population_avg_code_length(population)
        last_code_length_growth = None
        last_overfit_risk = False
        last_holdout_fitness = None
        last_train_at_holdout = None
        best_at_last_e2 = prev_best
        e1_cooldown_remaining = 0
        last_major_improvement_generation = 0

        for gen in range(start_gen, self.n_pop):
            generation_offspring = []
            diagnosis_label = "DEFAULT"
            chosen_operator = None
            if self.mode == "population_planner":
                population, generation_offspring, diagnosis_label, chosen_operator = self._planner_generation(gen, population, interface_prob, interface_ec, planner, profiler, selector, best_history, no_improve_gens, last_major_improvement_generation)
            elif self.mode == "routed":
                diagnosis_label = self._routed_diagnosis(last_improved, no_improve_gens, last_overfit_risk, recent_ops, recent_deltas, last_diversity, last_code_length_growth)
                chosen_operator = self._route_operator(diagnosis_label, gen, no_improve_gens, recent_ops, prev_best, best_at_last_e2, e1_cooldown_remaining)
                parents, generation_offspring = interface_ec.get_algorithm(population, chosen_operator)
                for child in generation_offspring:
                    self._stamp(child, gen + 1, chosen_operator, [parent.get("id") for parent in parents] if parents else [])
                population.extend(generation_offspring)
                population = self.manage.population_management(population, min(len(population), self.pop_size))
                recent_ops.append(chosen_operator)
                recent_deltas.append(None)
                recent_ops = recent_ops[-self.route_recent_window :]
                recent_deltas = recent_deltas[-self.route_recent_window :]
                if chosen_operator == "e2" and population:
                    best_at_last_e2 = float(population[0]["objective"])
                if chosen_operator == "e1":
                    e1_cooldown_remaining = self.route_e1_cooldown
                elif e1_cooldown_remaining > 0:
                    e1_cooldown_remaining -= 1
                self._write_jsonl(self.operator_log_path, {"gen": int(gen + 1), "mode": self.mode, "operator": chosen_operator, "operator_weight": None, "executed": True, "diagnosis_label": diagnosis_label, "n_offspring": int(len(generation_offspring)), "n_valid": int(self._count_valid(generation_offspring)), "n_invalid": int(len(generation_offspring) - self._count_valid(generation_offspring)), "invalid_rate": float(self._invalid_rate(generation_offspring))})
            else:
                chosen_operator = "schedule:" + ",".join(self.operators)
                diagnosis_label = "BASELINE_SCHEDULE"
                for idx, op in enumerate(self.operators):
                    executed = np.random.rand() < self.operator_weights[idx]
                    offsprings = []
                    parents = []
                    if executed:
                        parents, offsprings = interface_ec.get_algorithm(population, op)
                        for child in offsprings:
                            self._stamp(child, gen + 1, op, [parent.get("id") for parent in parents] if parents else [])
                    generation_offspring.extend(offsprings)
                    population.extend(offsprings)
                    population = self.manage.population_management(population, min(len(population), self.pop_size))
                    self._write_jsonl(self.operator_log_path, {"gen": int(gen + 1), "mode": self.mode, "operator": op, "operator_weight": float(self.operator_weights[idx]), "executed": bool(executed), "diagnosis_label": diagnosis_label, "n_offspring": int(len(offsprings)), "n_valid": int(self._count_valid(offsprings)), "n_invalid": int(len(offsprings) - self._count_valid(offsprings)), "invalid_rate": float(self._invalid_rate(offsprings))})
                if e1_cooldown_remaining > 0:
                    e1_cooldown_remaining -= 1

            self._save_population(population, gen + 1)
            invalid_rate = self._invalid_rate(generation_offspring)
            diversity = self._population_diversity(population)
            avg_code_length = self._population_avg_code_length(population)
            last_code_length_growth = float(avg_code_length - prev_avg_code_length) if prev_avg_code_length is not None and avg_code_length is not None else None
            prev_avg_code_length = avg_code_length
            best_fitness = population[0]["objective"] if population else None
            if best_fitness is not None:
                best_history.append(float(best_fitness))

            train_delta = None
            if prev_best is None and best_fitness is not None:
                last_improved = True
                no_improve_gens = 0
                prev_best = best_fitness
                last_major_improvement_generation = gen + 1
            elif best_fitness is None:
                last_improved = False
                no_improve_gens += 1
            else:
                train_delta = float(prev_best - best_fitness)
                improved = bool(best_fitness < (prev_best - self.route_improvement_epsilon))
                if improved:
                    last_improved = True
                    no_improve_gens = 0
                    prev_best = best_fitness
                    last_major_improvement_generation = gen + 1
                else:
                    last_improved = False
                    no_improve_gens += 1
                    prev_best = min(prev_best, best_fitness)

            holdout_fitness_eval, holdout_evaluated = self._evaluate_best_on_holdout(interface_prob, population, gen)
            if holdout_evaluated and holdout_fitness_eval is not None:
                holdout_fitness = holdout_fitness_eval
                if last_holdout_fitness is not None and last_train_at_holdout is not None and best_fitness is not None:
                    train_improved = bool(best_fitness < (last_train_at_holdout - self.route_improvement_epsilon))
                    holdout_worsened = bool(holdout_fitness > (last_holdout_fitness + self.route_improvement_epsilon))
                    last_overfit_risk = bool(train_improved and holdout_worsened)
                else:
                    last_overfit_risk = False
                last_holdout_fitness = holdout_fitness
                last_train_at_holdout = best_fitness
            elif holdout_evaluated:
                holdout_fitness = last_holdout_fitness
                last_overfit_risk = False
            else:
                holdout_fitness = last_holdout_fitness

            fitness_gap = float(holdout_fitness - best_fitness) if holdout_fitness is not None and best_fitness is not None else None
            last_diversity = diversity
            self._write_jsonl(self.run_log_path, {"gen": int(gen + 1), "mode": self.mode, "best_fitness": best_fitness, "train_fitness": best_fitness, "holdout_fitness": holdout_fitness, "fitness_gap": fitness_gap, "chosen_operator": chosen_operator, "diagnosis_label": diagnosis_label, "invalid_rate": invalid_rate, "no_improve_gens": int(no_improve_gens), "e1_cooldown_remaining": int(e1_cooldown_remaining), "stagnation_count": int(no_improve_gens), "diversity": diversity, "code_length_growth": last_code_length_growth, "train_delta": train_delta, "overfit_risk": bool(last_overfit_risk), "holdout_evaluated": bool(holdout_evaluated), "last_major_improvement_generation": int(last_major_improvement_generation)})
            print(f"--- {gen + 1} of {self.n_pop} populations finished. Time Cost:  {((time.time() - time_start) / 60):.1f} m")
            print("Pop Objs: ", end=" ")
            for item in population:
                print(str(item["objective"]) + " ", end="")
            print()
