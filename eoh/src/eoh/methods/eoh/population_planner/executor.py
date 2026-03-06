from typing import Dict, List

from .models import HeuristicCard, PlannerIntervention, PopulationPlannerOutput, PopulationSummary
from .prompts import (
    build_explore_modifiers,
    build_rewrite_modifiers,
    build_tune_modifiers,
    build_variant_modifiers,
)


PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}
MODE_TO_OPERATOR = {
    "rewrite": "m1",
    "tune": "m2",
    "variant": "e2",
    "explore": "e1",
}


class PopulationPlannerExecutor:
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
            mode = intervention.execution_mode
            if mode == "evaluate":
                queue.append(
                    {
                        "execution_mode": mode,
                        "operator": None,
                        "targets": [card.id for card in targets],
                        "goal": intervention.goal,
                        "instruction": intervention.instruction,
                        "offspring_count": 0,
                        "prompt_modifiers": [],
                    }
                )
                continue
            count = min(remaining, max(1, int(intervention.offspring_count)))
            if mode == "rewrite":
                modifiers = build_rewrite_modifiers(intervention.goal, intervention.instruction, targets[0])
            elif mode == "tune":
                modifiers = build_tune_modifiers(intervention.goal, intervention.instruction, targets[0])
            elif mode == "variant":
                modifiers = build_variant_modifiers(intervention.goal, intervention.instruction, targets)
            else:
                modifiers = build_explore_modifiers(intervention.goal, intervention.instruction, summary)
            queue.append(
                {
                    "execution_mode": mode,
                    "operator": MODE_TO_OPERATOR[mode],
                    "targets": [card.id for card in targets],
                    "goal": intervention.goal,
                    "instruction": intervention.instruction,
                    "offspring_count": count,
                    "prompt_modifiers": modifiers[:4],
                }
            )
            remaining -= count

        if remaining > 0 and len(cards) > 0:
            fallback_card = cards[0]
            queue.append(
                {
                    "execution_mode": "variant",
                    "operator": "e2",
                    "targets": [fallback_card.id],
                    "goal": "Safe fallback variant from the current best heuristic.",
                    "instruction": "Preserve the main motif but alter one scoring component.",
                    "offspring_count": remaining,
                    "prompt_modifiers": build_variant_modifiers(
                        "Safe fallback variant from the current best heuristic.",
                        "Preserve the main motif but alter one scoring component.",
                        [fallback_card],
                    )[:4],
                }
            )
        return queue
