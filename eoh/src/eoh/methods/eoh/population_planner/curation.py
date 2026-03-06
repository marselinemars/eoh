import math

from .models import PopulationSummary


def _label_diversity(score):
    if score < 0.20:
        return "low"
    if score < 0.45:
        return "moderate"
    return "high"


def _behavior_signature(card):
    behavior = card.behavior
    return [
        float(behavior.mean_residual_ratio),
        float(behavior.fragmentation_index),
        float(behavior.resource_opening_rate_early),
        float(behavior.choice_entropy),
        float(behavior.order_sensitivity),
        float(behavior.holdout_gap),
    ]


def _distance(a, b):
    if len(a) != len(b):
        return 0.0
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def _mean_pairwise_distance(cards):
    if len(cards) < 2:
        return 0.0
    distances = []
    signatures = [_behavior_signature(card) for card in cards]
    for i in range(len(signatures)):
        for j in range(i + 1, len(signatures)):
            distances.append(_distance(signatures[i], signatures[j]))
    if len(distances) == 0:
        return 0.0
    return sum(distances) / float(len(distances))


def build_population_summary(cards, generation, best_history, stagnation_length, last_major_improvement_generation):
    ranked = sorted(cards, key=lambda card: (card.fitness, card.rank))
    best_fitness = ranked[0].fitness if ranked else None
    unique_summaries = len(set(card.algorithm_summary for card in cards if card.algorithm_summary))
    structural_score = unique_summaries / float(max(1, len(cards)))
    behavioral_score = _mean_pairwise_distance(cards)

    if generation <= 2:
        regime = "early_search"
    elif stagnation_length >= 3:
        regime = "stagnation_recovery"
    elif last_major_improvement_generation >= max(0, generation - 2):
        regime = "post_breakthrough_refinement"
    else:
        regime = "steady_refinement"

    notes = []
    if ranked:
        notes.append(f"best heuristic={ranked[0].id}")
    if structural_score < 0.40:
        notes.append("population is structurally concentrated")
    if behavioral_score < 0.20:
        notes.append("population is behaviorally redundant")
    if ranked and ranked[0].behavior.holdout_gap > 0.01:
        notes.append("best heuristic shows train-holdout tension")

    return PopulationSummary(
        generation=int(generation),
        best_fitness=best_fitness,
        recent_best_fitness_history=[float(x) for x in best_history[-5:]],
        stagnation_length=int(stagnation_length),
        structural_diversity_estimate=_label_diversity(structural_score),
        behavioral_diversity_estimate=_label_diversity(behavioral_score),
        last_major_improvement_generation=int(last_major_improvement_generation),
        current_search_regime=regime,
        notes=notes[:4],
    )


class PopulationSelector:
    def __init__(self, max_cards=8):
        self.max_cards = max(4, int(max_cards))

    def select(self, cards):
        if len(cards) <= self.max_cards:
            return list(cards)

        ranked = sorted(cards, key=lambda card: (card.fitness, card.rank))
        selected = []
        used_ids = set()

        def add_card(card):
            if card.id in used_ids:
                return
            selected.append(card)
            used_ids.add(card.id)

        for card in ranked[:3]:
            add_card(card)

        remaining = [card for card in ranked if card.id not in used_ids]
        if remaining:
            add_card(max(remaining, key=lambda card: card.structure.simplicity_index))

        remaining = [card for card in ranked if card.id not in used_ids]
        if remaining:
            add_card(min(remaining, key=lambda card: (abs(card.behavior.holdout_gap), card.fitness)))

        remaining = [card for card in ranked if card.id not in used_ids]
        while len(selected) < min(self.max_cards, 6) and remaining:
            chosen = max(
                remaining,
                key=lambda card: min(
                    _distance(_behavior_signature(card), _behavior_signature(existing))
                    for existing in selected
                ),
            )
            add_card(chosen)
            remaining = [card for card in remaining if card.id not in used_ids]

        remaining = [card for card in ranked if card.id not in used_ids]
        if remaining and len(selected) < self.max_cards:
            outlier = max(
                remaining,
                key=lambda card: (
                    abs(card.behavior.order_sensitivity - ranked[0].behavior.order_sensitivity),
                    abs(card.behavior.fragmentation_index - ranked[0].behavior.fragmentation_index),
                ),
            )
            add_card(outlier)

        remaining = [card for card in ranked if card.id not in used_ids]
        while len(selected) < self.max_cards and remaining:
            add_card(remaining.pop(0))

        return selected[: self.max_cards]
