from .prompts import (
    build_explore_prompt,
    build_rewrite_prompt,
    build_tune_prompt,
    build_variant_prompt,
)


PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}


class PopulationPlannerExecutor:
    def __init__(self, interface_ec, summary, selected_cards):
        self.interface_ec = interface_ec
        self.summary = summary
        self.card_map = {card.id: card for card in selected_cards}

    def sort_interventions(self, interventions):
        return sorted(
            interventions,
            key=lambda item: (PRIORITY_ORDER.get(item.priority, 1), -int(item.offspring_count)),
        )

    def build_prompt(self, intervention):
        evolution = self.interface_ec.evol
        targets = [self.card_map[target_id] for target_id in intervention.targets if target_id in self.card_map]
        if intervention.execution_mode == "rewrite" and len(targets) > 0:
            return build_rewrite_prompt(evolution, intervention.goal, intervention.instruction, targets[0])
        if intervention.execution_mode == "tune" and len(targets) > 0:
            return build_tune_prompt(evolution, intervention.goal, intervention.instruction, targets[0])
        if intervention.execution_mode == "variant" and len(targets) > 0:
            return build_variant_prompt(evolution, intervention.goal, intervention.instruction, targets)
        return build_explore_prompt(evolution, intervention.goal, intervention.instruction, self.summary, targets)
