from typing import Dict, List

from .models import HeuristicCard, PlannerIntervention, PopulationPlannerOutput, PopulationSummary
from .prompts import (
    build_explore_modifiers,
    build_rewrite_modifiers,
    build_rewrite_prompt,
    build_tune_modifiers,
    build_variant_modifiers,
)


PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}


class PopulationPlannerExecutor:
    def __init__(self, problem_context: str, func_name: str = "score", func_inputs: List[str] | None = None, func_outputs: List[str] | None = None):
        self.problem_context = problem_context
        self.func_name = func_name
        self.func_inputs = list(func_inputs or ["item", "bins"])
        self.func_outputs = list(func_outputs or ["scores"])

    def execute_rewrite(self, intervention: PlannerIntervention, targets: List[HeuristicCard], summary: PopulationSummary, offspring_count: int) -> Dict:
        primary = targets[0]
        return {
            "execution_mode": "rewrite",
            "generation_backend": "dedicated_rewrite",
            "operator": None,
            "targets": [card.id for card in targets],
            "goal": intervention.goal,
            "instruction": intervention.instruction,
            "offspring_count": offspring_count,
            "prompt_modifiers": build_rewrite_modifiers(intervention.goal, intervention.instruction, primary)[:4],
            "custom_prompt": build_rewrite_prompt(
                problem_context=self.problem_context,
                goal=intervention.goal,
                instruction=intervention.instruction,
                card=primary,
                func_name=self.func_name,
                func_inputs=self.func_inputs,
                func_outputs=self.func_outputs,
            ),
        }

    def execute_tune(self, intervention: PlannerIntervention, targets: List[HeuristicCard], summary: PopulationSummary, offspring_count: int) -> Dict:
        primary = targets[0]
        return {
            "execution_mode": "tune",
            "generation_backend": "semantic_tune",
            "operator": "m2",
            "targets": [card.id for card in targets],
            "goal": intervention.goal,
            "instruction": intervention.instruction,
            "offspring_count": offspring_count,
            "prompt_modifiers": build_tune_modifiers(intervention.goal, intervention.instruction, primary)[:4],
        }

    def execute_variant(self, intervention: PlannerIntervention, targets: List[HeuristicCard], summary: PopulationSummary, offspring_count: int) -> Dict:
        return {
            "execution_mode": "variant",
            "generation_backend": "semantic_variant",
            "operator": "e2",
            "targets": [card.id for card in targets],
            "goal": intervention.goal,
            "instruction": intervention.instruction,
            "offspring_count": offspring_count,
            "prompt_modifiers": build_variant_modifiers(intervention.goal, intervention.instruction, targets)[:4],
        }

    def execute_explore(self, intervention: PlannerIntervention, targets: List[HeuristicCard], summary: PopulationSummary, offspring_count: int) -> Dict:
        return {
            "execution_mode": "explore",
            "generation_backend": "semantic_explore",
            "operator": "e1",
            "targets": [card.id for card in targets],
            "goal": intervention.goal,
            "instruction": intervention.instruction,
            "offspring_count": offspring_count,
            "prompt_modifiers": build_explore_modifiers(intervention.goal, intervention.instruction, summary)[:4],
        }

    def execute_evaluate(self, intervention: PlannerIntervention, targets: List[HeuristicCard], summary: PopulationSummary) -> Dict:
        return {
            "execution_mode": "evaluate",
            "generation_backend": "evaluation_only",
            "operator": None,
            "targets": [card.id for card in targets],
            "goal": intervention.goal,
            "instruction": intervention.instruction,
            "offspring_count": 0,
            "prompt_modifiers": [],
        }

    def _dispatch(self, intervention: PlannerIntervention, targets: List[HeuristicCard], summary: PopulationSummary, offspring_count: int) -> Dict:
        mode = intervention.execution_mode
        if mode == "rewrite":
            return self.execute_rewrite(intervention, targets, summary, offspring_count)
        if mode == "tune":
            return self.execute_tune(intervention, targets, summary, offspring_count)
        if mode == "variant":
            return self.execute_variant(intervention, targets, summary, offspring_count)
        if mode == "explore":
            return self.execute_explore(intervention, targets, summary, offspring_count)
        return self.execute_evaluate(intervention, targets, summary)

    def build_queue(
        self,
        planner_output: PopulationPlannerOutput,
        cards: List[HeuristicCard],
        summary: PopulationSummary,
        generation_budget: int,
    ) -> List[Dict]:
        card_map = {card.id: card for card in cards}
        queue = []
        remaining = max(0, int(generation_budget))
        ordered = sorted(
            planner_output.interventions,
            key=lambda item: (PRIORITY_ORDER.get(item.priority, 99), 0 if item.execution_mode != "evaluate" else 1),
        )
        for intervention in ordered:
            if remaining <= 0 and intervention.execution_mode != "evaluate":
                break
            targets = [card_map[target] for target in intervention.targets if target in card_map]
            if len(targets) == 0:
                continue
            if intervention.execution_mode == "evaluate":
                queue.append(self.execute_evaluate(intervention, targets, summary))
                continue
            count = min(remaining, max(1, int(intervention.offspring_count)))
            queue.append(self._dispatch(intervention, targets, summary, count))
            remaining -= count

        if remaining > 0 and len(cards) > 0:
            fallback = PlannerIntervention(
                targets=[cards[0].id],
                execution_mode="variant",
                goal="Safe fallback variant from the current best heuristic.",
                instruction="Preserve the main motif but alter one scoring component.",
                offspring_count=remaining,
                priority="medium",
            )
            queue.append(self.execute_variant(fallback, [cards[0]], summary, remaining))
        return queue
