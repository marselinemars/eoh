import numpy as np
import json
import random
import time
import os
import hashlib
import concurrent.futures

from .eoh_interface_EC import InterfaceEC
from .agentic_controller import AgenticController
from .population_planner import (
    PopulationPlanner,
    PopulationPlannerExecutor,
    build_heuristic_cards,
    build_population_summary,
    select_planner_cards,
)
from .population_planner.profiler import resolve_trace_metric
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
        if self.mode not in ["baseline", "routed", "agentic", "planner_population"]:
            print(f"Unknown eoh_mode={self.mode}, fallback to baseline.")
            self.mode = "baseline"
        if self.mode == "agentic":
            self.mode = "routed"
        self.route_controller_enabled = bool(getattr(paras, "route_controller_enabled", True))
        self.route_controller_window = max(2, int(getattr(paras, "route_controller_window", 5)))
        self.route_controller_use_llm = bool(getattr(paras, "route_controller_use_llm", True))
        self.route_controller_use_critic = bool(getattr(paras, "route_controller_use_critic", False))
        self.route_improvement_epsilon = float(getattr(paras, "route_improvement_epsilon", 1e-12))
        self.route_stagnation_k = int(getattr(paras, "route_stagnation_k", 3))
        self.route_invalid_rate_threshold = float(getattr(paras, "route_invalid_rate_threshold", 0.5))
        self.route_use_diversity = bool(getattr(paras, "route_use_diversity", True))
        self.route_recent_window = int(getattr(paras, "route_recent_window", 3))
        self.route_structural_plateau_eps = float(getattr(paras, "route_structural_plateau_eps", 1e-5))
        self.route_warmup_gens = max(0, int(getattr(paras, "route_warmup_gens", 2)))
        self.route_e1_cooldown = max(0, int(getattr(paras, "route_e1_cooldown", 3)))
        self.route_e2_recent_k = max(1, int(getattr(paras, "route_e2_recent_k", 3)))
        self.route_use_probabilistic = bool(getattr(paras, "route_use_probabilistic", True))
        self.route_shuffle_operator_order = bool(getattr(paras, "route_shuffle_operator_order", False))
        self.holdout_eval_interval = max(1, int(getattr(paras, "holdout_eval_interval", 1)))
        self.log_full_population = bool(getattr(paras, "log_full_population", False))
        self.run_log_path = os.path.join(self.output_path, "results", "run_log.jsonl")
        self.operator_log_path = os.path.join(self.output_path, "results", "operator_events.jsonl")
        self.agent_observation_log_path = os.path.join(self.output_path, "results", "agent_observation.jsonl")
        self.agent_diagnosis_log_path = os.path.join(self.output_path, "results", "agent_diagnosis.jsonl")
        self.agent_plan_log_path = os.path.join(self.output_path, "results", "agent_plan.jsonl")
        self.agent_critic_log_path = os.path.join(self.output_path, "results", "agent_critic.jsonl")
        self.llm_diagnoser_raw_log_path = os.path.join(self.output_path, "results", "llm_diagnoser_raw.jsonl")
        self.llm_planner_raw_log_path = os.path.join(self.output_path, "results", "llm_planner_raw.jsonl")
        self.llm_critic_raw_log_path = os.path.join(self.output_path, "results", "llm_critic_raw.jsonl")
        self.heuristic_cards_log_path = os.path.join(self.output_path, "results", "heuristic_cards.jsonl")
        self.population_summary_log_path = os.path.join(self.output_path, "results", "population_summary.jsonl")
        self.population_planner_output_log_path = os.path.join(self.output_path, "results", "population_planner_output.jsonl")
        self.executed_interventions_log_path = os.path.join(self.output_path, "results", "executed_interventions.jsonl")
        self.offspring_lineage_log_path = os.path.join(self.output_path, "results", "offspring_lineage.jsonl")
        self.behavior_metric_debug_log_path = os.path.join(self.output_path, "results", "behavior_metric_debug.jsonl")
        self.planner_population_view_size = max(4, int(getattr(paras, "planner_population_view_size", 8)))
        self.planner_population_json_retries = max(1, int(getattr(paras, "planner_population_json_retries", 3)))

        print("- EoH parameters loaded -")

        self.controller = None
        self.population_planner = None
        self.population_executor = None
        if self.mode == "routed" and self.route_controller_enabled and self.route_controller_use_llm:
            self.controller = AgenticController(
                self.api_endpoint,
                self.api_key,
                self.llm_model,
                self.use_local_llm,
                self.llm_local_url,
                use_critic_agent=self.route_controller_use_critic,
                debug_mode=self.debug_mode,
            )
        if self.mode == "planner_population" and bool(getattr(paras, "planner_population_enabled", True)):
            self.population_planner = PopulationPlanner(
                self.api_endpoint,
                self.api_key,
                self.llm_model,
                self.use_local_llm,
                self.llm_local_url,
                debug_mode=self.debug_mode,
                max_retries=self.planner_population_json_retries,
            )
            self.population_executor = PopulationPlannerExecutor(
                problem_context=self._problem_context_for_planner(),
                func_name=self.prob.prompts.get_func_name(),
                func_inputs=self.prob.prompts.get_func_inputs(),
                func_outputs=self.prob.prompts.get_func_outputs(),
            )

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
        with open(self.agent_observation_log_path, "w", encoding="utf-8") as _:
            pass
        with open(self.agent_diagnosis_log_path, "w", encoding="utf-8") as _:
            pass
        with open(self.agent_plan_log_path, "w", encoding="utf-8") as _:
            pass
        with open(self.agent_critic_log_path, "w", encoding="utf-8") as _:
            pass
        with open(self.llm_diagnoser_raw_log_path, "w", encoding="utf-8") as _:
            pass
        with open(self.llm_planner_raw_log_path, "w", encoding="utf-8") as _:
            pass
        with open(self.llm_critic_raw_log_path, "w", encoding="utf-8") as _:
            pass
        with open(self.heuristic_cards_log_path, "w", encoding="utf-8") as _:
            pass
        with open(self.population_summary_log_path, "w", encoding="utf-8") as _:
            pass
        with open(self.population_planner_output_log_path, "w", encoding="utf-8") as _:
            pass
        with open(self.executed_interventions_log_path, "w", encoding="utf-8") as _:
            pass
        with open(self.offspring_lineage_log_path, "w", encoding="utf-8") as _:
            pass
        with open(self.behavior_metric_debug_log_path, "w", encoding="utf-8") as _:
            pass

    def _write_run_log(self, record):
        with open(self.run_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _write_operator_log(self, record):
        with open(self.operator_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _write_agent_observation_log(self, record):
        with open(self.agent_observation_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _write_agent_diagnosis_log(self, record):
        with open(self.agent_diagnosis_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _write_agent_plan_log(self, record):
        with open(self.agent_plan_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _write_agent_critic_log(self, record):
        with open(self.agent_critic_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _write_llm_diagnoser_raw_log(self, record):
        with open(self.llm_diagnoser_raw_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _write_llm_planner_raw_log(self, record):
        with open(self.llm_planner_raw_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _write_llm_critic_raw_log(self, record):
        with open(self.llm_critic_raw_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _write_heuristic_cards_log(self, record):
        with open(self.heuristic_cards_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _write_population_summary_log(self, record):
        with open(self.population_summary_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _write_population_planner_output_log(self, record):
        with open(self.population_planner_output_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _write_executed_interventions_log(self, record):
        with open(self.executed_interventions_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _write_offspring_lineage_log(self, record):
        with open(self.offspring_lineage_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _write_behavior_metric_debug_log(self, record):
        with open(self.behavior_metric_debug_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _code_hash(self, code):
        if code is None:
            return None
        return hashlib.sha1(code.encode("utf-8")).hexdigest()[:12]

    def _ensure_individual_metadata(self, individual, generation_index, created_by="seed", parent_ids=None, parent_hashes=None, note=""):
        if not isinstance(individual, dict):
            return
        if not isinstance(individual.get("other_inf"), dict):
            individual["other_inf"] = {}
        other_inf = individual["other_inf"]
        code_hash = self._code_hash(individual.get("code")) or "none"
        if not other_inf.get("heuristic_id"):
            other_inf["heuristic_id"] = f"H{int(generation_index)}_{code_hash}"
        lineage = other_inf.get("lineage", {}) if isinstance(other_inf.get("lineage"), dict) else {}
        if "created_by" not in lineage:
            lineage["created_by"] = created_by
        if parent_ids is not None:
            lineage["parent_ids"] = list(parent_ids)
        else:
            lineage.setdefault("parent_ids", [])
        if parent_hashes is not None:
            lineage["parent_hashes"] = list(parent_hashes)
        else:
            lineage.setdefault("parent_hashes", [])
        if note:
            lineage["note"] = note
        other_inf["lineage"] = lineage

    def _ensure_population_metadata(self, population, generation_index, created_by="seed"):
        for individual in population:
            self._ensure_individual_metadata(individual, generation_index, created_by=created_by)

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

    def _population_avg_code_length(self, population):
        lengths = []
        for individual in population:
            code = individual.get("code")
            if isinstance(code, str):
                lengths.append(len(code))
        if len(lengths) == 0:
            return None
        return float(np.mean(np.array(lengths)))

    def _safe_float(self, value, default=0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _behavior_metric_from_individual(self, individual, metric_name):
        if not isinstance(individual, dict):
            return None
        other_inf = individual.get("other_inf", {}) if isinstance(individual.get("other_inf"), dict) else {}
        trace_metrics = other_inf.get("trace_metrics", {}) if isinstance(other_inf.get("trace_metrics"), dict) else {}
        value, status, _ = resolve_trace_metric(trace_metrics, metric_name)
        if status != "ok":
            return None
        return self._to_float_or_none(value)

    def _behavior_delta(self, individual, parent_card, metric_name):
        child = self._behavior_metric_from_individual(individual, metric_name)
        parent = None
        if parent_card is not None:
            try:
                metric_value = parent_card.behavior.get(metric_name)
                parent = None if metric_value is None else float(metric_value)
            except Exception:
                parent = None
        if child is None or parent is None:
            return None
        return float(child - parent)

    def _is_noop_relative_to_parent(self, offspring, parent_card, deltas):
        if parent_card is None or not isinstance(offspring, dict):
            return False
        child_obj = self._to_float_or_none(offspring.get("objective"))
        parent_obj = self._to_float_or_none(parent_card.fitness)
        if child_obj is None or parent_obj is None:
            return False
        if abs(child_obj - parent_obj) > 1e-12:
            return False
        for delta in deltas:
            if delta is not None and abs(float(delta)) > 1e-12:
                return False
        return True

    def _last_major_improvement_generation(self, best_history):
        if len(best_history) < 2:
            return None
        best_gen = None
        for idx in range(1, len(best_history)):
            prev = best_history[idx - 1]
            curr = best_history[idx]
            if prev is None or curr is None:
                continue
            if curr < (prev - self.route_improvement_epsilon):
                best_gen = idx
        return best_gen

    def _top_heuristics_summary(self, population, top_k=5):
        summary = []
        if len(population) == 0:
            return summary
        for rank, individual in enumerate(population[:top_k]):
            algo = individual.get("algorithm")
            if isinstance(algo, str):
                algo_line = " ".join(algo.split())[:180]
            else:
                algo_line = ""
            code = individual.get("code")
            complexity = len(code) if isinstance(code, str) else 0.0
            summary.append(
                {
                    "id": f"g{rank}_{self._code_hash(code)}",
                    "fitness": self._to_float_or_none(individual.get("objective")),
                    "summary": algo_line,
                    "complexity": float(complexity),
                }
            )
        return summary

    def _compute_improve_rate(self, best_history, window):
        if len(best_history) < 2:
            return 0.0
        window = max(1, int(window))
        start = max(1, len(best_history) - window)
        improvements = 0
        total = 0
        for idx in range(start, len(best_history)):
            prev = best_history[idx - 1]
            curr = best_history[idx]
            if prev is None or curr is None:
                continue
            total += 1
            if curr < (prev - self.route_improvement_epsilon):
                improvements += 1
        if total == 0:
            return 0.0
        return float(improvements / total)

    def _planner_metric_debug_record(self, individual, card, shown):
        other_inf = individual.get("other_inf", {}) if isinstance(individual.get("other_inf"), dict) else {}
        raw_trace = other_inf.get("trace_metrics", {}) if isinstance(other_inf.get("trace_metrics"), dict) else {}
        return {
            "heuristic_id": card.id,
            "shown_to_planner": bool(shown),
            "fitness": self._to_float_or_none(individual.get("objective")),
            "raw_evaluator_metrics": raw_trace,
            "stored_profile_metrics": dict(card.behavior),
            "stored_profile_metric_status": dict(card.behavior_status),
            "stored_profile_metric_sources": dict(card.extra.get("behavior_source_keys", {})) if isinstance(card.extra, dict) else {},
            "final_planner_card_metrics": dict(card.behavior),
            "final_planner_card_metric_status": dict(card.behavior_status),
        }

    def _build_op_stats_k(self, op_history, window):
        stats = {op: {"calls": 0, "success_rate": 0.0, "mean_delta": 0.0, "invalid_rate": 0.0} for op in ["e1", "e2", "m1", "m2", "m3"]}
        window_records = op_history[-max(1, int(window)):]
        grouped = {op: [] for op in ["e1", "e2", "m1", "m2", "m3"]}
        for rec in window_records:
            op = rec.get("operator")
            if op in grouped:
                grouped[op].append(rec)
        for op, records in grouped.items():
            if len(records) == 0:
                continue
            calls = len(records)
            total_offspring = 0
            total_valid = 0
            weighted_delta_num = 0.0
            weighted_delta_den = 0.0
            total_invalid = 0
            for rec in records:
                n_valid = int(rec.get("n_valid", 0) or 0)
                n_off = int(rec.get("n_offspring", 0) or 0)
                n_invalid = int(rec.get("n_invalid", max(0, n_off - n_valid)) or 0)
                total_offspring += max(0, n_off)
                total_valid += max(0, n_valid)
                total_invalid += max(0, n_invalid)
                delta = rec.get("delta_best")
                if delta is not None and n_off > 0:
                    weighted_delta_num += float(delta) * float(n_off)
                    weighted_delta_den += float(n_off)
            if total_offspring > 0:
                success_rate = float(total_valid / total_offspring)
                invalid_rate = float(total_invalid / total_offspring)
            else:
                success_rate = 0.0
                invalid_rate = 1.0
            mean_delta = float(weighted_delta_num / weighted_delta_den) if weighted_delta_den > 0 else 0.0
            stats[op] = {
                "calls": int(calls),
                "success_rate": success_rate,
                "mean_delta": mean_delta,
                "invalid_rate": invalid_rate,
            }
        return stats

    def _build_observation_packet(
        self,
        generation_index,
        population,
        best_history,
        no_improve_gens,
        invalid_rate_k,
        op_history,
        last_used_ops,
        eval_instances_per_gen,
        holdout_instances,
    ):
        best_fitness = self._best_objective(population)
        prev_best = best_history[-2] if len(best_history) >= 2 else None
        delta_best = None
        if prev_best is not None and best_fitness is not None:
            delta_best = float(prev_best - best_fitness)
        diversity_abs = self._population_diversity(population)
        if diversity_abs is None or self.pop_size <= 0:
            diversity_score = 0.0
        else:
            diversity_score = float(diversity_abs) / float(max(1, self.pop_size))
        observation = {
            "gen": int(generation_index + 1),
            "population_size": int(len(population)),
            "best_fitness": self._to_float_or_none(best_fitness),
            "best_history": [self._to_float_or_none(v) for v in best_history[-max(2, self.route_controller_window):]],
            "delta_best": self._to_float_or_none(delta_best),
            "improve_rate_k": self._compute_improve_rate(best_history, self.route_controller_window),
            "stagnation_len": int(no_improve_gens),
            "invalid_rate_k": self._safe_float(invalid_rate_k, default=0.0),
            "diversity_score": float(max(0.0, min(1.0, diversity_score))),
            "op_stats_k": self._build_op_stats_k(op_history, self.route_controller_window),
            "last_used_ops": [str(op) for op in last_used_ops[-self.route_controller_window:]],
            "top_heuristics": self._top_heuristics_summary(population, top_k=min(5, len(population))),
            "budget": {
                "instances": int(eval_instances_per_gen),
                "holdout_instances": int(holdout_instances),
            },
            "notes": f"mode={self.mode}",
        }
        return observation

    def _resolve_train_instance_budget(self, interface_prob):
        configured = getattr(interface_prob, "eval_instances_per_gen", None)
        if configured is not None:
            try:
                return max(1, int(configured))
            except (TypeError, ValueError):
                pass
        instances = getattr(interface_prob, "instances", None)
        if isinstance(instances, dict):
            total = 0
            for dataset in instances.values():
                if isinstance(dataset, dict):
                    total += len(dataset)
            if total > 0:
                return int(total)
        return 0

    def _choose_available_operator(self, preferred, fallbacks=None):
        if preferred in self.operators:
            return preferred
        if fallbacks is not None:
            for op in fallbacks:
                if op in self.operators:
                    return op
        return self.operators[0]

    def _sample_operator(self, distribution, fallback_order):
        candidates = []
        for op, weight in distribution.items():
            if op in self.operators and float(weight) > 0.0:
                candidates.append((op, float(weight)))
        if len(candidates) == 0:
            preferred = fallback_order[0] if len(fallback_order) > 0 else self.operators[0]
            fallbacks = fallback_order[1:] if len(fallback_order) > 1 else None
            return self._choose_available_operator(preferred, fallbacks=fallbacks)
        total = sum(weight for _, weight in candidates)
        draw = random.random() * total
        cum = 0.0
        for op, weight in candidates:
            cum += weight
            if draw <= cum:
                return op
        return candidates[-1][0]

    def _sample_routed_operator_batch(self, op_probs, total_offspring):
        budget = max(1, int(total_offspring))
        sampled_ops = []
        for _ in range(budget):
            sampled_ops.append(self.controller.sample_operator(op_probs, self.operators))
        op_counts = {}
        for op in sampled_ops:
            op_counts[op] = op_counts.get(op, 0) + 1
        ordered = [(op, op_counts[op]) for op in self.operators if op in op_counts and op_counts[op] > 0]
        return sampled_ops, ordered

    def _should_structural_edit(self, no_improve_gens, recent_ops, recent_deltas, diversity, code_length_growth):
        if no_improve_gens < 2:
            return False
        if len(recent_ops) == 0:
            return False

        m2_deltas = [recent_deltas[i] for i, op in enumerate(recent_ops) if op == "m2"]
        m2_recent_count = len(m2_deltas)
        if m2_recent_count == 0:
            return False
        no_gain_on_m2 = all((d is None) or (d <= self.route_structural_plateau_eps) for d in m2_deltas)
        low_diversity = (diversity is not None) and (diversity <= max(2, self.pop_size // 2))
        code_is_growing = (code_length_growth is not None) and (code_length_growth > 0.0)
        return (m2_recent_count >= 2) and no_gain_on_m2 and (low_diversity or code_is_growing)

    def _can_try_e1_in_gn(self, no_improve_gens, recent_ops, current_best, best_at_last_e2):
        if no_improve_gens < 4:
            return False
        e2_recent = any(op == "e2" for op in recent_ops[-self.route_e2_recent_k:])
        if not e2_recent:
            return False
        if current_best is None or best_at_last_e2 is None:
            return True
        best_has_not_improved_since_last_e2 = not (
            current_best < (best_at_last_e2 - self.route_improvement_epsilon)
        )
        return best_has_not_improved_since_last_e2

    def _diagnose_routed(
        self,
        last_improved,
        no_improve_gens,
        overfit_risk,
        recent_ops,
        recent_deltas,
        diversity,
        code_length_growth,
    ):
        if overfit_risk:
            return "OVERFIT_RISK"
        if last_improved:
            return "NEED_PARAM_TUNING"
        if self._should_structural_edit(
            no_improve_gens=no_improve_gens,
            recent_ops=recent_ops,
            recent_deltas=recent_deltas,
            diversity=diversity,
            code_length_growth=code_length_growth,
        ):
            return "NEED_STRUCTURAL_EDIT"
        if no_improve_gens >= 3:
            return "NEED_GLOBAL_NOVELTY"
        return "NEED_BACKBONE_VARIANT"

    def _route_operator(
        self,
        diagnosis_label,
        generation_index,
        no_improve_gens,
        recent_ops,
        current_best,
        best_at_last_e2,
        e1_cooldown_remaining,
    ):
        # Warmup: stabilize early search on the strongest observed operator.
        if (generation_index + 1) <= self.route_warmup_gens:
            return self._choose_available_operator("e2", fallbacks=["m1", "m2", "e1"])

        # Cooldown: prevent repeated global novelty collapse.
        # During cooldown, force e2 regardless of diagnosis.
        if e1_cooldown_remaining > 0:
            return self._choose_available_operator("e2", fallbacks=["m1", "m2", "e1"])

        if diagnosis_label == "OVERFIT_RISK":
            return self._choose_available_operator("m3", fallbacks=["m1", "e2", "m2", "e1"])

        if diagnosis_label == "NEED_PARAM_TUNING":
            if self.route_use_probabilistic:
                return self._sample_operator(
                    {"m2": 0.7, "m1": 0.3},
                    fallback_order=["m2", "m1", "e2", "e1"],
                )
            return self._choose_available_operator("m2", fallbacks=["m1", "e2", "e1"])

        if diagnosis_label == "NEED_STRUCTURAL_EDIT":
            return self._choose_available_operator("m1", fallbacks=["e2", "m2", "e1"])

        if diagnosis_label == "NEED_BACKBONE_VARIANT":
            # Enforce BV -> e2.
            return self._choose_available_operator("e2", fallbacks=["m1", "m2", "e1"])

        if diagnosis_label == "NEED_GLOBAL_NOVELTY":
            allow_e1 = self._can_try_e1_in_gn(
                no_improve_gens=no_improve_gens,
                recent_ops=recent_ops,
                current_best=current_best,
                best_at_last_e2=best_at_last_e2,
            )
            if not allow_e1:
                return self._choose_available_operator("e2", fallbacks=["m1", "m2", "e1"])
            if self.route_use_probabilistic:
                return self._sample_operator(
                    {"e2": 0.85, "e1": 0.15},
                    fallback_order=["e2", "e1", "m1", "m2"],
                )
            return self._choose_available_operator("e1", fallbacks=["e2", "m1", "m2"])

        return self._choose_available_operator("e2", fallbacks=["m1", "m2", "e1"])

    def _evaluate_best_on_holdout(self, interface_prob, population, generation_index):
        if len(population) == 0:
            return None, False
        if ((generation_index + 1) % self.holdout_eval_interval) != 0:
            return None, False
        if not hasattr(interface_prob, "evaluate_on_split"):
            return None, False
        code = population[0].get("code")
        if code is None:
            return None, True
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(interface_prob.evaluate_on_split, code, "holdout")
                holdout_fitness = future.result(timeout=max(20, self.timeout * 2))
                future.cancel()
            if holdout_fitness is None:
                return None, True
            return float(holdout_fitness), True
        except Exception:
            return None, True

    def _problem_context_for_planner(self):
        return (
            "Problem: online bin packing. Items arrive sequentially and must be assigned immediately to bins with fixed capacity. "
            "Lower fitness is better. The planner should reason over heuristic behavior, structure, diagnosis, and lineage."
        )

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
            self._ensure_population_metadata(population, generation_index=0, created_by="seed")
            self._save_population(population, 0)
            n_start = 0
        else:
            if self.load_pop:  # load population from files
                print("load initial population from " + self.load_pop_path)
                with open(self.load_pop_path) as file:
                    data = json.load(file)
                for individual in data:
                    population.append(individual)
                self._ensure_population_metadata(population, generation_index=self.load_pop_id, created_by="loaded")
                print("initial population has been loaded!")
                n_start = self.load_pop_id
            else:  # create new population
                print("creating initial population:")
                population = interface_ec.population_generation()
                population = self.manage.population_management(population, self.pop_size)
                self._ensure_population_metadata(population, generation_index=0, created_by="init")

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
        best_history = [float(prev_best)] if prev_best is not None else []
        op_history = []
        last_used_ops = []
        no_improve_gens = 0
        last_invalid_rate = 0.0
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
        eval_instances_budget = self._resolve_train_instance_budget(interface_prob)
        holdout_instances_budget = int(getattr(interface_prob, "holdout_instances", 0) or 0)

        for pop in range(n_start, self.n_pop):
            generation_offspring = []
            chosen_operator = None
            diagnosis_label = "DEFAULT"
            routed_delta_best = None
            active_parent_mix = None
            active_prompt_modifiers = []
            active_op_probs = None
            planner_time_s = 0.0
            generation_time_s = 0.0
            evaluation_time_s = 0.0
            invalid_before_eval_count = 0
            diagnoser_stage_flags = {}
            planner_stage_flags = {}
            critic_stage_flags = {}
            diag_debug = {}
            plan_debug = {}
            critic_debug = {}

            if self.mode == "planner_population":
                interface_ec.set_controller_context(parent_mix=None, prompt_modifiers=None)
                self._ensure_population_metadata(population, generation_index=pop, created_by="survivor")
                cards = build_heuristic_cards(population, generation=pop)
                summary = build_population_summary(
                    cards=cards,
                    generation=int(pop + 1),
                    best_history=best_history,
                    stagnation_length=no_improve_gens,
                    last_major_improvement_generation=self._last_major_improvement_generation(best_history),
                )
                planner_cards = select_planner_cards(cards, max_cards=self.planner_population_view_size)
                planner_card_map = {card.id: card for card in planner_cards}
                card_map = {card.id: card for card in cards}
                diagnosis_label = summary.current_search_regime.upper()
                self._write_heuristic_cards_log(
                    {
                        "gen": int(pop + 1),
                        "mode": self.mode,
                        "cards": [card.to_dict() for card in cards],
                        "shown_card_ids": [card.id for card in planner_cards],
                    }
                )
                debug_records = []
                for individual in population:
                    other_inf = individual.get("other_inf", {}) if isinstance(individual.get("other_inf"), dict) else {}
                    heuristic_id = str(other_inf.get("heuristic_id") or "")
                    card = card_map.get(heuristic_id)
                    if card is None:
                        continue
                    debug_records.append(self._planner_metric_debug_record(individual, card, heuristic_id in planner_card_map))
                self._write_behavior_metric_debug_log(
                    {
                        "gen": int(pop + 1),
                        "mode": self.mode,
                        "heuristics": debug_records,
                    }
                )
                self._write_population_summary_log(
                    {
                        "gen": int(pop + 1),
                        "mode": self.mode,
                        "summary": summary.to_dict(),
                    }
                )
                planner_start = time.time()
                planner_result = self.population_planner.plan(
                    problem_context=self._problem_context_for_planner(),
                    summary=summary,
                    cards=planner_cards,
                )
                planner_time_s += float(time.time() - planner_start)
                planner_output_obj = planner_result["planner_output"]
                planner_llm_meta = planner_result["llm_meta"]
                self._write_population_planner_output_log(
                    {
                        "gen": int(pop + 1),
                        "mode": self.mode,
                        "shown_cards": [card.to_dict() for card in planner_cards],
                        "population_summary": summary.to_dict(),
                        "planner_output": planner_output_obj.to_dict(),
                        "llm_meta": planner_llm_meta,
                    }
                )
                intervention_queue = self.population_executor.build_queue(
                    planner_output=planner_output_obj,
                    cards=planner_cards,
                    summary=summary,
                    generation_budget=self.pop_size,
                )
                chosen_operator = "planner:" + ",".join(
                    f"{item.get('execution_mode')}x{int(item.get('offspring_count', 0))}"
                    for item in intervention_queue
                    if int(item.get("offspring_count", 0) or 0) > 0
                )
                executed_ops = []
                executed_deltas = []
                accepted_code_hashes = {
                    self._code_hash(ind.get("code"))
                    for ind in population
                    if isinstance(ind, dict) and self._code_hash(ind.get("code")) is not None
                }
                for idx_exec, item in enumerate(intervention_queue):
                    execution_mode = str(item.get("execution_mode", ""))
                    targets = [str(x) for x in item.get("targets", [])]
                    operator = item.get("operator")
                    generation_backend = str(item.get("generation_backend", ""))
                    prompt_modifiers = item.get("prompt_modifiers", [])
                    custom_prompt = item.get("custom_prompt")
                    offspring_count = int(item.get("offspring_count", 0) or 0)
                    self._write_executed_interventions_log(
                        {
                            "gen": int(pop + 1),
                            "mode": self.mode,
                            "execution_rank": int(idx_exec + 1),
                            "intervention": item,
                        }
                    )
                    if execution_mode == "evaluate" or offspring_count <= 0 or (operator is None and not (isinstance(custom_prompt, str) and custom_prompt.strip())):
                        continue
                    print(f" INT: {execution_mode}, [{idx_exec + 1} / {len(intervention_queue)}], n={offspring_count} ", end="|")
                    preferred_hashes = [planner_card_map[target].code_hash for target in targets if target in planner_card_map]
                    parent_mix = None
                    if len(preferred_hashes) > 0 and execution_mode in ["rewrite", "tune", "variant"]:
                        parent_mix = {"preferred": 0.75, "elite": 0.15, "diverse": 0.10, "random": 0.0}
                    interface_ec.set_controller_context(
                        parent_mix=parent_mix,
                        prompt_modifiers=prompt_modifiers,
                        preferred_parent_hashes=preferred_hashes,
                    )
                    best_before = self._best_objective(population)
                    target_cards = [planner_card_map[target] for target in targets if target in planner_card_map]
                    llm_parallel_limit = 2
                    if generation_backend in ["dedicated_rewrite", "semantic_explore"]:
                        llm_parallel_limit = 1
                    if isinstance(custom_prompt, str) and custom_prompt.strip():
                        parent_payloads, offsprings = interface_ec.get_algorithm_from_prompt(
                            population,
                            custom_prompt,
                            n_offspring=offspring_count,
                            parent_cards=[{"code": card.code} for card in target_cards if isinstance(card.code, str)],
                            max_parallel_requests=llm_parallel_limit,
                        )
                    else:
                        parent_payloads, offsprings = interface_ec.get_algorithm(
                            population,
                            operator,
                            n_offspring=offspring_count,
                            max_parallel_requests=llm_parallel_limit,
                        )
                    batch_stats = interface_ec.get_last_batch_stats()
                    generation_time_s += float(batch_stats.get("generation_time_s", 0.0) or 0.0)
                    evaluation_time_s += float(batch_stats.get("evaluation_time_s", 0.0) or 0.0)
                    invalid_before_eval_count += int(batch_stats.get("invalid_before_eval_count", 0) or 0)
                    valid_offsprings = []
                    invalid_offspring_count = 0
                    for off_idx, offspring in enumerate(offsprings):
                        primary_parent = target_cards[0] if len(target_cards) > 0 else None
                        parent_hashes = []
                        parent_ids = list(targets)
                        raw_parents = parent_payloads[off_idx] if off_idx < len(parent_payloads) else None
                        if isinstance(raw_parents, list):
                            for parent in raw_parents:
                                parent_hash = self._code_hash(parent.get("code")) if isinstance(parent, dict) else None
                                if parent_hash is not None:
                                    parent_hashes.append(parent_hash)
                        if offspring is None or offspring.get("objective") is None or offspring.get("code") is None:
                            invalid_offspring_count += 1
                            failure_reason = None
                            if isinstance(offspring, dict) and isinstance(offspring.get("other_inf"), dict):
                                failure_reason = offspring["other_inf"].get("failure_reason")
                            self._write_offspring_lineage_log(
                                {
                                    "gen": int(pop + 1),
                                    "mode": self.mode,
                                    "offspring_id": None,
                                    "execution_mode": execution_mode,
                                    "operator": operator,
                                    "generation_backend": generation_backend,
                                    "target_ids": parent_ids,
                                    "fitness": None,
                                    "fitness_delta": None,
                                    "mean_residual_ratio_delta": None,
                                    "fragmentation_delta": None,
                                    "resource_opening_rate_early_delta": None,
                                    "order_sensitivity_delta": None,
                                    "failure_reason": failure_reason,
                                }
                            )
                            continue
                        self._ensure_individual_metadata(
                            offspring,
                            generation_index=int(pop + 1),
                            created_by=execution_mode,
                            parent_ids=parent_ids,
                            parent_hashes=parent_hashes,
                            note=item.get("goal", ""),
                        )
                        offspring["other_inf"]["planner_instruction"] = item.get("instruction", "")
                        lineage_record = {
                            "gen": int(pop + 1),
                            "mode": self.mode,
                            "offspring_id": offspring["other_inf"].get("heuristic_id"),
                            "execution_mode": execution_mode,
                            "operator": operator,
                            "generation_backend": generation_backend,
                            "target_ids": parent_ids,
                            "fitness": self._to_float_or_none(offspring.get("objective")),
                            "fitness_delta": None if primary_parent is None else (
                                None if offspring.get("objective") is None or primary_parent.fitness is None else float(primary_parent.fitness - float(offspring.get("objective")))
                            ),
                            "mean_residual_ratio_delta": self._behavior_delta(offspring, primary_parent, "mean_residual_ratio"),
                            "fragmentation_delta": self._behavior_delta(offspring, primary_parent, "fragmentation_index"),
                            "resource_opening_rate_early_delta": self._behavior_delta(offspring, primary_parent, "resource_opening_rate_early"),
                            "order_sensitivity_delta": self._behavior_delta(offspring, primary_parent, "order_sensitivity"),
                        }
                        offspring_code_hash = self._code_hash(offspring.get("code"))
                        metric_deltas = [
                            lineage_record["mean_residual_ratio_delta"],
                            lineage_record["fragmentation_delta"],
                            lineage_record["resource_opening_rate_early_delta"],
                            lineage_record["order_sensitivity_delta"],
                        ]
                        rejection_reason = None
                        if offspring_code_hash is not None and offspring_code_hash in accepted_code_hashes:
                            rejection_reason = "duplicate_code_in_population_or_batch"
                        elif self._is_noop_relative_to_parent(offspring, primary_parent, metric_deltas):
                            rejection_reason = "no_op_relative_to_parent"
                        if rejection_reason is not None:
                            invalid_offspring_count += 1
                            lineage_record["offspring_id"] = None
                            lineage_record["failure_reason"] = rejection_reason
                            self._write_offspring_lineage_log(lineage_record)
                            continue
                        if offspring_code_hash is not None:
                            accepted_code_hashes.add(offspring_code_hash)
                        self._write_offspring_lineage_log(lineage_record)
                        valid_offsprings.append(offspring)
                    self.add2pop(population, valid_offsprings)
                    generation_offspring.extend(offsprings)
                    for off in valid_offsprings:
                        print(" Obj: ", off["objective"], end="|")
                    if invalid_offspring_count > 0:
                        print(f" Invalid: {invalid_offspring_count}", end="|")
                    size_act = min(len(population), self.pop_size)
                    population = self.manage.population_management(population, size_act)
                    best_after = self._best_objective(population)
                    n_offspring = len(offsprings)
                    n_valid = len(valid_offsprings)
                    n_invalid = n_offspring - n_valid
                    invalid_rate_op = (n_invalid / n_offspring) if n_offspring > 0 else 1.0
                    delta_best = None
                    if best_before is not None and best_after is not None:
                        delta_best = float(best_before - best_after)
                    op_record = {
                        "gen": int(pop + 1),
                        "mode": self.mode,
                        "operator": operator,
                        "execution_mode": execution_mode,
                        "generation_backend": generation_backend,
                        "executed": True,
                        "diagnosis_label": diagnosis_label,
                        "n_offspring": int(n_offspring),
                        "n_valid": int(n_valid),
                        "n_invalid": int(n_invalid),
                        "invalid_rate": float(invalid_rate_op),
                        "best_before": self._to_float_or_none(best_before),
                        "best_after": self._to_float_or_none(best_after),
                        "delta_best": self._to_float_or_none(delta_best),
                        "targets": targets,
                        "prompt_modifiers": prompt_modifiers,
                    }
                    self._write_operator_log(op_record)
                    op_history.append(op_record)
                    executed_ops.append(operator)
                    executed_deltas.append(delta_best)
                    print()
                interface_ec.set_controller_context(parent_mix=None, prompt_modifiers=None, preferred_parent_hashes=None)
                last_used_ops.extend(executed_ops)
                recent_ops.extend(executed_ops)
                recent_deltas.extend(executed_deltas)
                if len(recent_ops) > self.route_recent_window:
                    recent_ops = recent_ops[-self.route_recent_window:]
                    recent_deltas = recent_deltas[-self.route_recent_window:]

            elif self.mode == "routed":
                if self.controller is not None:
                    observation = self._build_observation_packet(
                        generation_index=pop,
                        population=population,
                        best_history=best_history,
                        no_improve_gens=no_improve_gens,
                        invalid_rate_k=last_invalid_rate,
                        op_history=op_history,
                        last_used_ops=last_used_ops,
                        eval_instances_per_gen=eval_instances_budget,
                        holdout_instances=holdout_instances_budget,
                    )
                    self._write_agent_observation_log(
                        {
                            "gen": int(pop + 1),
                            "mode": self.mode,
                            "observation": observation,
                        }
                    )
                    controller_result = self.controller.run(observation)
                    diagnosis = controller_result["diagnosis"]
                    planner_output = controller_result["planner_output"]
                    critic_output = controller_result["critic_output"]
                    controller_debug = controller_result.get("debug", {})
                    diag_debug = controller_debug.get("diagnoser", {}) if isinstance(controller_debug.get("diagnoser", {}), dict) else {}
                    plan_debug = controller_debug.get("planner", {}) if isinstance(controller_debug.get("planner", {}), dict) else {}
                    critic_debug = controller_debug.get("critic", {}) if isinstance(controller_debug.get("critic", {}), dict) else {}
                    diagnoser_stage_flags = diag_debug.get("flags", {}) if isinstance(diag_debug.get("flags", {}), dict) else {}
                    planner_stage_flags = plan_debug.get("flags", {}) if isinstance(plan_debug.get("flags", {}), dict) else {}
                    critic_stage_flags = critic_debug.get("flags", {}) if isinstance(critic_debug.get("flags", {}), dict) else {}
                    final_plan = critic_output["final_plan"]
                    active_op_probs = final_plan["op_probs"]
                    active_parent_mix = final_plan["parent_mix"]
                    active_prompt_modifiers = final_plan.get("prompt_modifiers", [])

                    labels = diagnosis.get("diagnosis_labels", [])
                    diagnosis_label = str(labels[0]) if isinstance(labels, list) and len(labels) > 0 else "AGENTIC"
                    self._write_agent_diagnosis_log(
                        {
                            "gen": int(pop + 1),
                            "mode": self.mode,
                            "diagnosis_label": diagnosis_label,
                            "summary": diagnosis.get("summary"),
                            "factors": diagnosis.get("factors"),
                            "diagnosis_labels": diagnosis.get("diagnosis_labels"),
                            "evidence": diagnosis.get("evidence"),
                            "pure_llm_output": bool(diag_debug.get("flags", {}).get("pure_llm_output", False)),
                            "llm_output_patched": bool(diag_debug.get("flags", {}).get("llm_output_patched", False)),
                            "fully_fallback": bool(diag_debug.get("flags", {}).get("fully_fallback", False)),
                            "fallback_used": bool(diag_debug.get("sanitize", {}).get("fallback_used", False)),
                            "llm_success": bool(diag_debug.get("llm", {}).get("llm_success", False)),
                            "llm_mode": str(diag_debug.get("llm", {}).get("llm_mode", "")),
                            "parse_ok": bool(diag_debug.get("llm", {}).get("parse_ok", False)),
                            "validation_ok": bool(diag_debug.get("llm", {}).get("validation_ok", False)),
                            "retries_used": int(diag_debug.get("llm", {}).get("retries_used", 0) or 0),
                            "repair_used": bool(diag_debug.get("llm", {}).get("repair_used", False)),
                            "failure_reason": str(diag_debug.get("llm", {}).get("failure_reason", "")),
                            "validation_errors": diag_debug.get("llm", {}).get("last_errors", []),
                            "sanitize_patches": diag_debug.get("sanitize", {}).get("patches", []),
                        }
                    )
                    self._write_agent_plan_log(
                        {
                            "gen": int(pop + 1),
                            "mode": self.mode,
                            "diagnosis_used": planner_output.get("diagnosis_used"),
                            "op_probs": planner_output.get("op_probs"),
                            "parent_mix": planner_output.get("parent_mix"),
                            "prompt_modifiers": planner_output.get("prompt_modifiers"),
                            "evaluation_plan": planner_output.get("evaluation_plan"),
                            "rationale": planner_output.get("rationale"),
                            "pure_llm_output": bool(plan_debug.get("flags", {}).get("pure_llm_output", False)),
                            "llm_output_patched": bool(plan_debug.get("flags", {}).get("llm_output_patched", False)),
                            "fully_fallback": bool(plan_debug.get("flags", {}).get("fully_fallback", False)),
                            "fallback_used": bool(plan_debug.get("sanitize", {}).get("fallback_used", False)),
                            "llm_success": bool(plan_debug.get("llm", {}).get("llm_success", False)),
                            "llm_mode": str(plan_debug.get("llm", {}).get("llm_mode", "")),
                            "parse_ok": bool(plan_debug.get("llm", {}).get("parse_ok", False)),
                            "validation_ok": bool(plan_debug.get("llm", {}).get("validation_ok", False)),
                            "retries_used": int(plan_debug.get("llm", {}).get("retries_used", 0) or 0),
                            "repair_used": bool(plan_debug.get("llm", {}).get("repair_used", False)),
                            "failure_reason": str(plan_debug.get("llm", {}).get("failure_reason", "")),
                            "validation_errors": plan_debug.get("llm", {}).get("last_errors", []),
                            "sanitize_patches": plan_debug.get("sanitize", {}).get("patches", []),
                        }
                    )
                    interface_ec.set_controller_context(
                        parent_mix=active_parent_mix,
                        prompt_modifiers=active_prompt_modifiers,
                    )
                    sampled_ops, op_execution_plan = self._sample_routed_operator_batch(
                        active_op_probs,
                        self.pop_size,
                    )
                    if len(op_execution_plan) == 0:
                        fallback_op = self.controller.sample_operator(active_op_probs, self.operators)
                        sampled_ops = [fallback_op]
                        op_execution_plan = [(fallback_op, 1)]
                    if self.route_shuffle_operator_order and len(op_execution_plan) > 1:
                        op_execution_plan = list(op_execution_plan)
                        random.shuffle(op_execution_plan)
                    chosen_operator = "mixture:" + ",".join(f"{op}x{cnt}" for op, cnt in op_execution_plan)
                    self._write_agent_critic_log(
                        {
                            "gen": int(pop + 1),
                            "mode": self.mode,
                            "verdict": critic_output.get("verdict"),
                            "reasons": critic_output.get("reasons"),
                            "final_plan": final_plan,
                            "chosen_operator": chosen_operator,
                            "sampled_ops": sampled_ops,
                            "operator_plan": [{"operator": op, "count": int(cnt)} for op, cnt in op_execution_plan],
                            "shuffle_operator_order": bool(self.route_shuffle_operator_order),
                            "pure_llm_output": bool(critic_debug.get("flags", {}).get("pure_llm_output", False)),
                            "llm_output_patched": bool(critic_debug.get("flags", {}).get("llm_output_patched", False)),
                            "fully_fallback": bool(critic_debug.get("flags", {}).get("fully_fallback", False)),
                            "skipped": bool(critic_debug.get("flags", {}).get("skipped", False)),
                            "llm_success": bool(critic_debug.get("llm", {}).get("llm_success", False)),
                            "llm_mode": str(critic_debug.get("llm", {}).get("llm_mode", "")),
                            "parse_ok": bool(critic_debug.get("llm", {}).get("parse_ok", False)),
                            "validation_ok": bool(critic_debug.get("llm", {}).get("validation_ok", False)),
                            "retries_used": int(critic_debug.get("llm", {}).get("retries_used", 0) or 0),
                            "repair_used": bool(critic_debug.get("llm", {}).get("repair_used", False)),
                            "failure_reason": str(critic_debug.get("llm", {}).get("failure_reason", "")),
                            "validation_errors": critic_debug.get("llm", {}).get("last_errors", []),
                            "sanitize_patches": critic_debug.get("sanitize", {}).get("patches", []),
                        }
                    )
                    self._write_llm_diagnoser_raw_log(
                        {
                            "gen": int(pop + 1),
                            "mode": self.mode,
                            "raw_output": controller_result.get("raw", {}).get("diagnosis_text"),
                            "attempts": diag_debug.get("llm", {}).get("attempts", []),
                            "retries_used": int(diag_debug.get("llm", {}).get("retries_used", 0) or 0),
                            "last_errors": diag_debug.get("llm", {}).get("last_errors", []),
                            "fallback_used": bool(diag_debug.get("sanitize", {}).get("fallback_used", False)),
                            "llm_success": bool(diag_debug.get("llm", {}).get("llm_success", False)),
                            "llm_mode": str(diag_debug.get("llm", {}).get("llm_mode", "")),
                            "parse_ok": bool(diag_debug.get("llm", {}).get("parse_ok", False)),
                            "validation_ok": bool(diag_debug.get("llm", {}).get("validation_ok", False)),
                            "repair_used": bool(diag_debug.get("llm", {}).get("repair_used", False)),
                            "failure_reason": str(diag_debug.get("llm", {}).get("failure_reason", "")),
                            "pure_llm_output": bool(diag_debug.get("flags", {}).get("pure_llm_output", False)),
                            "llm_output_patched": bool(diag_debug.get("flags", {}).get("llm_output_patched", False)),
                            "fully_fallback": bool(diag_debug.get("flags", {}).get("fully_fallback", False)),
                        }
                    )
                    self._write_llm_planner_raw_log(
                        {
                            "gen": int(pop + 1),
                            "mode": self.mode,
                            "raw_output": controller_result.get("raw", {}).get("planner_text"),
                            "attempts": plan_debug.get("llm", {}).get("attempts", []),
                            "retries_used": int(plan_debug.get("llm", {}).get("retries_used", 0) or 0),
                            "last_errors": plan_debug.get("llm", {}).get("last_errors", []),
                            "fallback_used": bool(plan_debug.get("sanitize", {}).get("fallback_used", False)),
                            "llm_success": bool(plan_debug.get("llm", {}).get("llm_success", False)),
                            "llm_mode": str(plan_debug.get("llm", {}).get("llm_mode", "")),
                            "parse_ok": bool(plan_debug.get("llm", {}).get("parse_ok", False)),
                            "validation_ok": bool(plan_debug.get("llm", {}).get("validation_ok", False)),
                            "repair_used": bool(plan_debug.get("llm", {}).get("repair_used", False)),
                            "failure_reason": str(plan_debug.get("llm", {}).get("failure_reason", "")),
                            "pure_llm_output": bool(plan_debug.get("flags", {}).get("pure_llm_output", False)),
                            "llm_output_patched": bool(plan_debug.get("flags", {}).get("llm_output_patched", False)),
                            "fully_fallback": bool(plan_debug.get("flags", {}).get("fully_fallback", False)),
                        }
                    )
                    self._write_llm_critic_raw_log(
                        {
                            "gen": int(pop + 1),
                            "mode": self.mode,
                            "raw_output": controller_result.get("raw", {}).get("critic_text"),
                            "attempts": critic_debug.get("llm", {}).get("attempts", []),
                            "retries_used": int(critic_debug.get("llm", {}).get("retries_used", 0) or 0),
                            "last_errors": critic_debug.get("llm", {}).get("last_errors", []),
                            "llm_success": bool(critic_debug.get("llm", {}).get("llm_success", False)),
                            "llm_mode": str(critic_debug.get("llm", {}).get("llm_mode", "")),
                            "parse_ok": bool(critic_debug.get("llm", {}).get("parse_ok", False)),
                            "validation_ok": bool(critic_debug.get("llm", {}).get("validation_ok", False)),
                            "repair_used": bool(critic_debug.get("llm", {}).get("repair_used", False)),
                            "failure_reason": str(critic_debug.get("llm", {}).get("failure_reason", "")),
                            "pure_llm_output": bool(critic_debug.get("flags", {}).get("pure_llm_output", False)),
                            "llm_output_patched": bool(critic_debug.get("flags", {}).get("llm_output_patched", False)),
                            "fully_fallback": bool(critic_debug.get("flags", {}).get("fully_fallback", False)),
                            "skipped": bool(critic_debug.get("flags", {}).get("skipped", False)),
                        }
                    )
                else:
                    interface_ec.set_controller_context(parent_mix=None, prompt_modifiers=None)
                    diagnosis_label = self._diagnose_routed(
                        last_improved=last_improved,
                        no_improve_gens=no_improve_gens,
                        overfit_risk=last_overfit_risk,
                        recent_ops=recent_ops,
                        recent_deltas=recent_deltas,
                        diversity=last_diversity,
                        code_length_growth=last_code_length_growth,
                    )
                    chosen_operator = self._route_operator(
                        diagnosis_label=diagnosis_label,
                        generation_index=pop,
                        no_improve_gens=no_improve_gens,
                        recent_ops=recent_ops,
                        current_best=prev_best,
                        best_at_last_e2=best_at_last_e2,
                        e1_cooldown_remaining=e1_cooldown_remaining,
                    )
                    sampled_ops = [chosen_operator]
                    op_execution_plan = [(chosen_operator, self.pop_size)]
                executed_ops = []
                executed_deltas = []
                for i_exec, (op_exec, planned_count) in enumerate(op_execution_plan):
                    print(f" OP: {op_exec}, [{i_exec + 1} / {len(op_execution_plan)}], n={int(planned_count)} ", end="|")
                    best_before = self._best_objective(population)
                    _, offsprings = interface_ec.get_algorithm(population, op_exec, n_offspring=planned_count)
                    batch_stats = interface_ec.get_last_batch_stats()
                    generation_time_s += float(batch_stats.get("generation_time_s", 0.0) or 0.0)
                    evaluation_time_s += float(batch_stats.get("evaluation_time_s", 0.0) or 0.0)
                    invalid_before_eval_count += int(batch_stats.get("invalid_before_eval_count", 0) or 0)
                    valid_offsprings = [off for off in offsprings if off is not None and off.get("objective") is not None and off.get("code") is not None]
                    invalid_offspring_count = len(offsprings) - len(valid_offsprings)
                    self.add2pop(population, valid_offsprings)
                    generation_offspring.extend(offsprings)
                    for off in valid_offsprings:
                        print(" Obj: ", off["objective"], end="|")
                    if invalid_offspring_count > 0:
                        print(f" Invalid: {invalid_offspring_count}", end="|")
                    size_act = min(len(population), self.pop_size)
                    population = self.manage.population_management(population, size_act)
                    best_after = self._best_objective(population)
                    n_offspring = len(offsprings)
                    n_valid = len(valid_offsprings)
                    n_invalid = n_offspring - n_valid
                    invalid_rate_op = (n_invalid / n_offspring) if n_offspring > 0 else 1.0
                    delta_best = None
                    if best_before is not None and best_after is not None:
                        delta_best = float(best_before - best_after)
                    routed_delta_best = delta_best
                    op_record = {
                        "gen": int(pop + 1),
                        "mode": self.mode,
                        "operator": op_exec,
                        "operator_weight": None,
                        "executed": True,
                        "execution_rank": int(i_exec + 1),
                        "shuffle_operator_order": bool(self.route_shuffle_operator_order),
                        "diagnosis_label": diagnosis_label,
                        "n_offspring": int(n_offspring),
                        "planned_offspring": int(planned_count),
                        "n_valid": int(n_valid),
                        "n_invalid": int(n_invalid),
                        "invalid_rate": float(invalid_rate_op),
                        "best_before": self._to_float_or_none(best_before),
                        "best_after": self._to_float_or_none(best_after),
                        "delta_best": self._to_float_or_none(delta_best),
                        "op_probs": active_op_probs,
                        "parent_mix": active_parent_mix,
                        "prompt_modifiers": active_prompt_modifiers,
                    }
                    self._write_operator_log(op_record)
                    op_history.append(op_record)
                    executed_ops.append(op_exec)
                    executed_deltas.append(routed_delta_best)
                    if op_exec == "e2" and best_after is not None:
                        best_at_last_e2 = float(best_after)
                    print()
                if self.controller is not None:
                    last_used_ops.extend(sampled_ops)
                else:
                    last_used_ops.extend(executed_ops)
                recent_ops.extend(executed_ops)
                recent_deltas.extend(executed_deltas)
                if len(recent_ops) > self.route_recent_window:
                    recent_ops = recent_ops[-self.route_recent_window:]
                    recent_deltas = recent_deltas[-self.route_recent_window:]
                if any(op == "e1" for op in sampled_ops):
                    e1_cooldown_remaining = self.route_e1_cooldown
                elif e1_cooldown_remaining > 0:
                    e1_cooldown_remaining -= 1
            else:
                interface_ec.set_controller_context(parent_mix=None, prompt_modifiers=None)
                diagnosis_label = "BASELINE_SCHEDULE"
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
                        batch_stats = interface_ec.get_last_batch_stats()
                        generation_time_s += float(batch_stats.get("generation_time_s", 0.0) or 0.0)
                        evaluation_time_s += float(batch_stats.get("evaluation_time_s", 0.0) or 0.0)
                        invalid_before_eval_count += int(batch_stats.get("invalid_before_eval_count", 0) or 0)
                    valid_offsprings = [off for off in offsprings if off is not None and off.get("objective") is not None and off.get("code") is not None]
                    invalid_offspring_count = len(offsprings) - len(valid_offsprings)
                    self.add2pop(population, valid_offsprings)
                    generation_offspring.extend(offsprings)
                    for off in valid_offsprings:
                        print(" Obj: ", off["objective"], end="|")
                    if invalid_offspring_count > 0:
                        print(f" Invalid: {invalid_offspring_count}", end="|")
                    size_act = min(len(population), self.pop_size)
                    population = self.manage.population_management(population, size_act)
                    best_after = self._best_objective(population)
                    n_offspring = len(offsprings)
                    n_valid = len(valid_offsprings)
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
                        "diagnosis_label": "BASELINE_SCHEDULE",
                        "n_offspring": int(n_offspring),
                        "n_valid": int(n_valid),
                        "n_invalid": int(n_invalid),
                        "invalid_rate": float(invalid_rate_op),
                        "best_before": self._to_float_or_none(best_before),
                        "best_after": self._to_float_or_none(best_after),
                        "delta_best": self._to_float_or_none(delta_best),
                    }
                    self._write_operator_log(op_record)
                    op_history.append(op_record)
                    print()
                if e1_cooldown_remaining > 0:
                    e1_cooldown_remaining -= 1

            self._save_population(population, pop + 1)

            invalid_rate = self._invalid_rate(generation_offspring)
            diversity = self._population_diversity(population)
            avg_code_length = self._population_avg_code_length(population)
            if prev_avg_code_length is not None and avg_code_length is not None:
                last_code_length_growth = float(avg_code_length - prev_avg_code_length)
            else:
                last_code_length_growth = None
            prev_avg_code_length = avg_code_length
            best_fitness = population[0]["objective"] if len(population) > 0 else None
            if best_fitness is not None:
                best_history.append(float(best_fitness))
            train_fitness = best_fitness

            train_delta = None
            if prev_best is None and best_fitness is not None:
                last_improved = True
                no_improve_gens = 0
                prev_best = best_fitness
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
                else:
                    last_improved = False
                    no_improve_gens += 1
                    prev_best = min(prev_best, best_fitness)

            holdout_fitness_eval, holdout_evaluated = self._evaluate_best_on_holdout(interface_prob, population, pop)
            if holdout_evaluated and holdout_fitness_eval is not None:
                holdout_fitness = holdout_fitness_eval
                if (
                    last_holdout_fitness is not None
                    and last_train_at_holdout is not None
                    and train_fitness is not None
                ):
                    train_improved = bool(
                        train_fitness < (last_train_at_holdout - self.route_improvement_epsilon)
                    )
                    holdout_worsened = bool(
                        holdout_fitness > (last_holdout_fitness + self.route_improvement_epsilon)
                    )
                    last_overfit_risk = bool(train_improved and holdout_worsened)
                else:
                    last_overfit_risk = False
                last_holdout_fitness = holdout_fitness
                last_train_at_holdout = train_fitness
            elif holdout_evaluated:
                holdout_fitness = last_holdout_fitness
                last_overfit_risk = False
            else:
                holdout_fitness = last_holdout_fitness

            if holdout_fitness is not None and train_fitness is not None:
                fitness_gap = float(holdout_fitness - train_fitness)
            else:
                fitness_gap = None

            last_invalid_rate = invalid_rate
            last_diversity = diversity
            run_record = {
                "gen": int(pop + 1),
                "mode": self.mode,
                "best_fitness": best_fitness,
                "train_fitness": train_fitness,
                "holdout_fitness": self._to_float_or_none(holdout_fitness),
                "fitness_gap": self._to_float_or_none(fitness_gap),
                "chosen_operator": chosen_operator,
                "diagnosis_label": diagnosis_label,
                "invalid_rate": invalid_rate,
                "no_improve_gens": int(no_improve_gens),
                "e1_cooldown_remaining": int(e1_cooldown_remaining),
                "stagnation_count": int(no_improve_gens),
                "diversity": diversity,
                "code_length_growth": self._to_float_or_none(last_code_length_growth),
                "train_delta": self._to_float_or_none(train_delta),
                "overfit_risk": bool(last_overfit_risk),
                "holdout_evaluated": bool(holdout_evaluated),
                "controller_enabled": bool(self.controller is not None),
                "controller_use_critic": bool(self.route_controller_use_critic),
                "shuffle_operator_order": bool(self.route_shuffle_operator_order),
                "parent_mix": active_parent_mix,
                "prompt_modifiers": active_prompt_modifiers,
                "op_probs": active_op_probs,
                "diagnoser_fully_fallback": bool(diagnoser_stage_flags.get("fully_fallback", False)),
                "diagnoser_llm_patched": bool(diagnoser_stage_flags.get("llm_output_patched", False)),
                "diagnoser_pure_llm": bool(diagnoser_stage_flags.get("pure_llm_output", False)),
                "diagnoser_llm_success": bool(diag_debug.get("llm", {}).get("llm_success", False)) if self.mode == "routed" and self.controller is not None else False,
                "diagnoser_parse_ok": bool(diag_debug.get("llm", {}).get("parse_ok", False)) if self.mode == "routed" and self.controller is not None else False,
                "diagnoser_validation_ok": bool(diag_debug.get("llm", {}).get("validation_ok", False)) if self.mode == "routed" and self.controller is not None else False,
                "planner_fully_fallback": bool(planner_stage_flags.get("fully_fallback", False)),
                "planner_llm_patched": bool(planner_stage_flags.get("llm_output_patched", False)),
                "planner_pure_llm": bool(planner_stage_flags.get("pure_llm_output", False)),
                "planner_llm_success": bool(plan_debug.get("llm", {}).get("llm_success", False)) if self.mode == "routed" and self.controller is not None else False,
                "planner_parse_ok": bool(plan_debug.get("llm", {}).get("parse_ok", False)) if self.mode == "routed" and self.controller is not None else False,
                "planner_validation_ok": bool(plan_debug.get("llm", {}).get("validation_ok", False)) if self.mode == "routed" and self.controller is not None else False,
                "critic_fully_fallback": bool(critic_stage_flags.get("fully_fallback", False)),
                "critic_llm_patched": bool(critic_stage_flags.get("llm_output_patched", False)),
                "critic_pure_llm": bool(critic_stage_flags.get("pure_llm_output", False)),
                "critic_skipped": bool(critic_stage_flags.get("skipped", False)),
                "critic_llm_success": bool(critic_debug.get("llm", {}).get("llm_success", False)) if self.mode == "routed" and self.controller is not None else False,
                "critic_parse_ok": bool(critic_debug.get("llm", {}).get("parse_ok", False)) if self.mode == "routed" and self.controller is not None else False,
                "critic_validation_ok": bool(critic_debug.get("llm", {}).get("validation_ok", False)) if self.mode == "routed" and self.controller is not None else False,
                "planner_time_s": float(planner_time_s),
                "generation_time_s": float(generation_time_s),
                "evaluation_time_s": float(evaluation_time_s),
                "invalid_before_eval_count": int(invalid_before_eval_count),
            }
            self._write_run_log(run_record)

            print(f"--- {pop + 1} of {self.n_pop} populations finished. Time Cost:  {((time.time()-time_start)/60):.1f} m")
            print("Pop Objs: ", end=" ")
            for i in range(len(population)):
                print(str(population[i]['objective']) + " ", end="")
            print()

