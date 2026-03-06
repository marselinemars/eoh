import ast
import hashlib

from .diagnosis import diagnose_heuristic
from .models import BehaviorMetrics, HeuristicCard, LineageInfo, StructureMetrics


class PopulationProfiler:
    def __init__(self, interface_prob, train_instances=8, holdout_instances=8):
        self.interface_prob = interface_prob
        self.train_instances = max(1, int(train_instances))
        self.holdout_instances = max(1, int(holdout_instances))
        self._cache = {}

    def _code_hash(self, code):
        return hashlib.sha1(code.encode("utf-8")).hexdigest()

    def _extract_structure(self, code):
        try:
            tree = ast.parse(code)
        except Exception:
            return StructureMetrics(
                complexity=float(len(code or "")),
                parameter_count=0,
                condition_count=0,
                simplicity_index=0.0,
            )

        complexity = 0
        parameter_count = 0
        condition_count = 0
        for node in ast.walk(tree):
            complexity += 1
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                parameter_count += 1
            if isinstance(node, (ast.If, ast.IfExp, ast.Compare)):
                condition_count += 1

        denominator = 1.0 + (0.015 * complexity) + (0.08 * parameter_count) + (0.10 * condition_count)
        simplicity_index = 1.0 / denominator
        return StructureMetrics(
            complexity=float(complexity),
            parameter_count=int(parameter_count),
            condition_count=int(condition_count),
            simplicity_index=float(simplicity_index),
        )

    def _extract_behavior(self, code):
        behavior = self.interface_prob.profile_code(
            code,
            train_instances=self.train_instances,
            holdout_instances=self.holdout_instances,
        )
        return BehaviorMetrics(**behavior)

    def profile_individual(self, individual, generation, rank):
        code = str(individual.get("code", "") or "")
        code_key = self._code_hash(code)
        cached = self._cache.get(code_key)
        if cached is None:
            structure = self._extract_structure(code)
            behavior = self._extract_behavior(code)
            diagnosis = diagnose_heuristic(structure, behavior)
            cached = {
                "structure": structure,
                "behavior": behavior,
                "diagnosis": diagnosis,
            }
            self._cache[code_key] = cached

        return HeuristicCard(
            id=str(individual.get("id")),
            generation=int(generation),
            fitness=float(individual.get("objective")),
            rank=int(rank),
            algorithm_summary=str(individual.get("algorithm") or "Generated heuristic"),
            structure=cached["structure"],
            behavior=cached["behavior"],
            diagnosis=cached["diagnosis"],
            lineage=LineageInfo(
                created_by=str(individual.get("created_by", "unknown")),
                parent_ids=[str(x) for x in individual.get("parent_ids", [])],
            ),
            code=code,
            metadata={
                "code_hash": code_key[:12],
                "planner_instruction": individual.get("planner_instruction"),
                "planner_goal": individual.get("planner_goal"),
            },
        )

    def profile_population(self, population, generation):
        ranked = sorted(
            [item for item in population if item.get("objective") is not None and item.get("code") is not None],
            key=lambda item: float(item.get("objective")),
        )
        cards = []
        for index, individual in enumerate(ranked, start=1):
            cards.append(self.profile_individual(individual, generation=generation, rank=index))
        return cards
