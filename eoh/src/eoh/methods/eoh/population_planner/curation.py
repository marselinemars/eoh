from typing import Dict, List, Optional

from .models import HeuristicCard, PopulationSummary


BEHAVIOR_DISTANCE_KEYS = [
    "fragmentation_index",
    "resource_opening_rate_early",
    "choice_entropy",
    "score_margin_mean",
    "order_sensitivity",
]


def _behavior_distance(a: HeuristicCard, b: HeuristicCard) -> float:
    distance = 0.0
    for key in BEHAVIOR_DISTANCE_KEYS:
        distance += abs(float(a.behavior.get(key, 0.0) or 0.0) - float(b.behavior.get(key, 0.0) or 0.0))
    return float(distance)


def _label_diversity(score: float) -> str:
    if score >= 0.25:
        return "high"
    if score >= 0.12:
        return "moderate"
    return "low"


def _current_regime(best_history: List[float], stagnation_length: int, last_major_improvement_generation: Optional[int]) -> str:
    if stagnation_length >= 4:
        return "stagnation_recovery"
    if last_major_improvement_generation is not None and last_major_improvement_generation >= max(1, len(best_history) - 5):
        return "post_breakthrough_refinement"
    return "early_search"


def build_population_summary(
    cards: List[HeuristicCard],
    generation: int,
    best_history: List[float],
    stagnation_length: int,
    last_major_improvement_generation: Optional[int],
) -> PopulationSummary:
    best_fitness = cards[0].fitness if len(cards) > 0 else None
    structural_values = [float(card.simplicity_index or 0.0) for card in cards]
    if len(structural_values) >= 2:
        structural_diversity = _label_diversity(max(structural_values) - min(structural_values))
    else:
        structural_diversity = "low"

    distances = []
    for idx, card in enumerate(cards):
        for other in cards[idx + 1 :]:
            distances.append(_behavior_distance(card, other))
    behavioral_diversity = _label_diversity(sum(distances) / float(len(distances))) if len(distances) > 0 else "low"
    regime = _current_regime(best_history, stagnation_length, last_major_improvement_generation)
    notes = []
    if len(cards) > 0:
        labels = {}
        for card in cards:
            for label in card.diagnosis.labels:
                labels[label] = labels.get(label, 0) + 1
        if len(labels) > 0:
            dominant = sorted(labels.items(), key=lambda item: item[1], reverse=True)[0][0]
            notes.append(f"dominant diagnosis motif: {dominant}")
        if behavioral_diversity == "low":
            notes.append("population currently appears behaviorally redundant")

    return PopulationSummary(
        generation=int(generation),
        best_fitness=best_fitness,
        recent_best_history=[float(x) for x in best_history[-6:] if x is not None],
        stagnation_length=int(stagnation_length),
        structural_diversity=structural_diversity,
        behavioral_diversity=behavioral_diversity,
        last_major_improvement_generation=last_major_improvement_generation,
        current_search_regime=regime,
        notes=notes,
    )


def select_planner_cards(cards: List[HeuristicCard], max_cards: int = 8) -> List[HeuristicCard]:
    if len(cards) <= max_cards:
        return cards
    selected: Dict[str, HeuristicCard] = {}
    for card in cards[:3]:
        selected[card.id] = card

    best = cards[0]
    distinct_sorted = sorted(cards[1:], key=lambda card: _behavior_distance(best, card), reverse=True)
    for card in distinct_sorted[:2]:
        selected[card.id] = card

    simple_card = max(cards, key=lambda card: (card.simplicity_index, -(card.fitness or 999.0)))
    selected[simple_card.id] = simple_card

    robust_card = min(cards, key=lambda card: (card.behavior.get("order_sensitivity", 0.0) + card.behavior.get("holdout_gap", 0.0), card.fitness or 999.0))
    selected[robust_card.id] = robust_card

    outlier = max(cards, key=lambda card: (card.behavior.get("fragmentation_index", 0.0) + card.behavior.get("resource_opening_rate_early", 0.0)))
    selected[outlier.id] = outlier

    ordered = [selected[card.id] for card in cards if card.id in selected]
    return ordered[:max_cards]
