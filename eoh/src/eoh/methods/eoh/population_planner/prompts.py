from typing import List

from .models import HeuristicCard, PopulationSummary


PLANNER_PROMPT_TEMPLATE = """You are the Population Planner for an agentic heuristic search system.

You are given a curated set of candidate heuristics for an optimization problem.

Each heuristic includes:
- scalar fitness (lower is better)
- an algorithm summary
- structural properties
- behavioral metrics
- a compact diagnosis
- lineage information

Your goal is to decide how the next generation of heuristics should be produced.

Available execution modes:

rewrite
revise an existing heuristic while preserving useful structure

tune
adjust coefficients or small components without major structural changes

variant
generate a related variant from one or more heuristics

explore
generate a more novel heuristic that explores a different direction

evaluate
request additional evaluation rather than generating a new heuristic

Guidelines:
1. Lower fitness matters, but behavior and diagnosis also matter.
2. Preserve strong motifs from good heuristics when possible.
3. If the best heuristic has clear weaknesses, propose targeted improvements.
4. If the population is behaviorally redundant, encourage exploration.
5. Slightly worse heuristics may still be useful if they represent different behaviors.
6. Use the diagnoses as evidence, not as hard rules.
7. Prefer concise, executable interventions.

Problem Context:
{PROBLEM_CONTEXT}

Population Summary:
{POPULATION_SUMMARY}

Candidate Heuristics:
{HEURISTIC_CARDS}

Return ONLY valid JSON:
{{
  "population_assessment": "...",
  "overall_strategy": "...",
  "interventions": [
    {{
      "targets": ["H1"],
      "execution_mode": "rewrite",
      "goal": "...",
      "instruction": "...",
      "offspring_count": 2,
      "priority": "high"
    }}
  ],
  "preserve_ids": ["H1"],
  "deprioritize_ids": ["H6"],
  "rationale": ["..."]
}}
"""


def render_heuristic_card(card: HeuristicCard) -> str:
    b = card.behavior
    d = card.diagnosis
    return f"""Heuristic {card.id}
fitness: {card.fitness:.5f} (rank {card.rank})

summary:
{card.algorithm_summary}

structure:
complexity={card.complexity:.0f}, parameters={card.parameter_count:.0f}, conditions={card.condition_count:.0f}, simplicity_index={card.simplicity_index:.3f}

behavior:
mean_residual_ratio={b.get('mean_residual_ratio')}
fragmentation_index={b.get('fragmentation_index')}
open_rate_early={b.get('resource_opening_rate_early')}
open_rate_mid={b.get('resource_opening_rate_mid')}
open_rate_late={b.get('resource_opening_rate_late')}
choice_entropy={b.get('choice_entropy')}
extreme_option_preference={b.get('extreme_option_preference')}
score_margin_mean={b.get('score_margin_mean')}
order_sensitivity={b.get('order_sensitivity')}
family_variance={b.get('family_variance')}
holdout_gap={b.get('holdout_gap')}

diagnosis:
labels={d.labels}
summary={d.summary}
evidence={d.key_evidence}

lineage:
created_by={card.created_by}
parents={card.parent_ids}""".strip()


def render_population_summary(summary: PopulationSummary) -> str:
    notes = "\n".join(f"- {note}" for note in summary.notes) if len(summary.notes) > 0 else "- none"
    return f"""generation: {summary.generation}
best_fitness: {summary.best_fitness}
recent_best_history: {summary.recent_best_history}
stagnation_length: {summary.stagnation_length} generations
last_major_improvement: {summary.last_major_improvement_generation}
structural_diversity: {summary.structural_diversity}
behavioral_diversity: {summary.behavioral_diversity}
current_search_regime: {summary.current_search_regime}
notes:
{notes}"""


def build_planner_prompt(problem_context: str, summary: PopulationSummary, cards: List[HeuristicCard]) -> str:
    card_block = "\n\n".join(render_heuristic_card(card) for card in cards)
    return PLANNER_PROMPT_TEMPLATE.format(
        PROBLEM_CONTEXT=problem_context,
        POPULATION_SUMMARY=render_population_summary(summary),
        HEURISTIC_CARDS=card_block,
    )


def build_rewrite_modifiers(goal: str, instruction: str, card: HeuristicCard) -> List[str]:
    return [
        f"Goal: {goal}",
        f"Instruction: {instruction}",
        f"Preserve the core motif of heuristic {card.id}.",
        "Keep the function signature unchanged and remain reasonably simple.",
    ]


def build_tune_modifiers(goal: str, instruction: str, card: HeuristicCard) -> List[str]:
    return [
        f"Goal: {goal}",
        f"Instruction: {instruction}",
        f"Preserve the main logic of heuristic {card.id}.",
        "Adjust only coefficients or small local score components.",
    ]


def build_variant_modifiers(goal: str, instruction: str, cards: List[HeuristicCard]) -> List[str]:
    parent_ids = ", ".join(card.id for card in cards)
    return [
        f"Goal: {goal}",
        f"Instruction: {instruction}",
        f"Generate a related but meaningfully different variant inspired by: {parent_ids}.",
        "Preserve useful motifs when appropriate.",
    ]


def build_explore_modifiers(goal: str, instruction: str, summary: PopulationSummary) -> List[str]:
    return [
        f"Goal: {goal}",
        f"Instruction: {instruction}",
        f"Current regime: {summary.current_search_regime}.",
        "Explore a genuinely different heuristic direction while staying relevant to the problem.",
    ]
