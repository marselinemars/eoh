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


REWRITE_PROMPT_TEMPLATE = """You are revising an existing heuristic for an optimization problem.

Your task is NOT to invent a completely new heuristic.
Your task is to produce a targeted correction of the existing heuristic.

You must preserve the useful core motif of the current heuristic, while revising the specific weakness described below.

Problem Context:
{PROBLEM_CONTEXT}

Current Heuristic ID:
{HEURISTIC_ID}

Current Heuristic Summary:
{HEURISTIC_SUMMARY}

Current Heuristic Diagnosis:
{HEURISTIC_DIAGNOSIS_SUMMARY}

Key Evidence About Its Behavior:
{HEURISTIC_KEY_EVIDENCE}

Behavior Summary:
{HEURISTIC_BEHAVIOR_SUMMARY}

Planner Goal:
{GOAL}

Planner Instruction:
{INSTRUCTION}

Rewrite Guidance:
1. Preserve the main useful idea or motif of the current heuristic.
2. Focus on correcting the diagnosed weakness described above.
3. Do NOT introduce unnecessary unrelated logic changes.
4. Prefer a targeted revision over a completely different structure.
5. Keep the function signature, inputs, and outputs unchanged.
6. Keep the heuristic reasonably simple unless the instruction explicitly requires otherwise.
7. If the diagnosed weakness can be improved by softening, rebalancing, simplifying, or stabilizing part of the scoring logic, prefer that over replacing the whole heuristic.
8. Your revision will be rejected if it produces effectively the same ranking behavior as the current heuristic on typical inputs.
9. Do NOT merely rescale the existing score, multiply it by a binary mask, or wrap it in a trivial threshold penalty without changing the ranking logic in a meaningful way.
10. Make one focused corrective change that is specific and testable.

What to optimize for:
- maintain the strengths of the current heuristic
- reduce the diagnosed weakness
- improve robustness or decision quality where relevant
- avoid destroying the successful behavior already present
- produce a materially distinct scoring behavior, not a cosmetic rewrite

STRICT OUTPUT FORMAT (MANDATORY):
1. First line: one sentence wrapped in braces like {{your sentence}}.
2. Then output ONLY Python code (no markdown fences).
3. Code must include: import numpy as np
4. Code must define exactly one function named {FUNC_NAME}.
5. Function inputs must be exactly: ({FUNC_INPUTS}).
6. Function must return: {FUNC_OUTPUTS}.
7. Do NOT output analysis, bullet points, explanations, or prose outside the brace sentence.
8. Do NOT output anything before the brace line or after the Python code.
"""


def render_heuristic_card(card: HeuristicCard) -> str:
    b = card.behavior
    d = card.diagnosis
    fitness_text = "missing" if card.fitness is None else f"{card.fitness:.5f}"
    def fmt_metric(key: str) -> str:
        status = (card.behavior_status or {}).get(key, "missing")
        value = b.get(key)
        if status == "ok" and value is not None:
            return str(value)
        return status
    return f"""Heuristic {card.id}
fitness: {fitness_text} (rank {card.rank})

summary:
{card.algorithm_summary}

structure:
complexity={card.complexity:.0f}, parameters={card.parameter_count:.0f}, conditions={card.condition_count:.0f}, simplicity_index={card.simplicity_index:.3f}

behavior:
mean_residual_ratio={fmt_metric('mean_residual_ratio')}
fragmentation_index={fmt_metric('fragmentation_index')}
open_rate_early={fmt_metric('resource_opening_rate_early')}
open_rate_mid={fmt_metric('resource_opening_rate_mid')}
open_rate_late={fmt_metric('resource_opening_rate_late')}
choice_entropy={fmt_metric('choice_entropy')}
extreme_option_preference={fmt_metric('extreme_option_preference')}
score_margin_mean={fmt_metric('score_margin_mean')}
order_sensitivity={fmt_metric('order_sensitivity')}
family_variance={fmt_metric('family_variance')}
holdout_gap={fmt_metric('holdout_gap')}

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


def _behavior_summary_lines(card: HeuristicCard) -> List[str]:
    b = card.behavior
    def fmt_metric(key: str) -> str:
        status = (card.behavior_status or {}).get(key, "missing")
        value = b.get(key)
        if status == "ok" and value is not None:
            return str(value)
        return status
    return [
        f"mean_residual_ratio={fmt_metric('mean_residual_ratio')}",
        f"fragmentation_index={fmt_metric('fragmentation_index')}",
        f"resource_opening_rate_early={fmt_metric('resource_opening_rate_early')}",
        f"resource_opening_rate_mid={fmt_metric('resource_opening_rate_mid')}",
        f"resource_opening_rate_late={fmt_metric('resource_opening_rate_late')}",
        f"choice_entropy={fmt_metric('choice_entropy')}",
        f"extreme_option_preference={fmt_metric('extreme_option_preference')}",
        f"score_margin_mean={fmt_metric('score_margin_mean')}",
        f"score_margin_variance={fmt_metric('score_margin_variance')}",
        f"order_sensitivity={fmt_metric('order_sensitivity')}",
        f"family_variance={fmt_metric('family_variance')}",
        f"holdout_gap={fmt_metric('holdout_gap')}",
    ]


def build_rewrite_prompt(problem_context: str, goal: str, instruction: str, card: HeuristicCard, func_name: str, func_inputs: List[str], func_outputs: List[str]) -> str:
    evidence = "\n".join(f"- {line}" for line in card.diagnosis.key_evidence[:3]) or "- no evidence"
    behavior_summary = "\n".join(f"- {line}" for line in _behavior_summary_lines(card))
    return REWRITE_PROMPT_TEMPLATE.format(
        PROBLEM_CONTEXT=problem_context,
        HEURISTIC_ID=card.id,
        HEURISTIC_SUMMARY=card.algorithm_summary,
        HEURISTIC_DIAGNOSIS_SUMMARY=card.diagnosis.summary,
        HEURISTIC_KEY_EVIDENCE=evidence,
        HEURISTIC_BEHAVIOR_SUMMARY=behavior_summary,
        GOAL=goal,
        INSTRUCTION=instruction,
        FUNC_NAME=func_name,
        FUNC_INPUTS=", ".join(func_inputs),
        FUNC_OUTPUTS=", ".join(func_outputs),
    ) + "\n\nCurrent heuristic code:\n" + (card.code or "")


def build_rewrite_modifiers(goal: str, instruction: str, card: HeuristicCard) -> List[str]:
    return [
        f"Goal: {goal}",
        f"Instruction: {instruction}",
        f"Preserve the core motif of heuristic {card.id}.",
        f"Address this diagnosis: {card.diagnosis.summary}",
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
