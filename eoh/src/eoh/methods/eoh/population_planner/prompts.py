def build_population_summary_text(summary):
    lines = [
        "Population Summary",
        "",
        f"generation: {summary.generation}",
        f"best_fitness: {summary.best_fitness}",
        f"recent_best_fitness_history: {summary.recent_best_fitness_history}",
        f"stagnation_length: {summary.stagnation_length} generations",
        f"last_major_improvement: generation {summary.last_major_improvement_generation}",
        "",
        f"structural_diversity: {summary.structural_diversity_estimate}",
        f"behavioral_diversity: {summary.behavioral_diversity_estimate}",
        f"current_search_regime: {summary.current_search_regime}",
        "",
        "notes:",
    ]
    if summary.notes:
        for note in summary.notes:
            lines.append(f"- {note}")
    else:
        lines.append("- none")
    return "\n".join(lines)


def build_heuristic_card_text(card):
    behavior = card.behavior
    structure = card.structure
    diagnosis = card.diagnosis
    lineage = card.lineage
    lines = [
        f"Heuristic {card.id}",
        f"fitness: {card.fitness} (rank {card.rank})",
        "",
        "summary:",
        card.algorithm_summary,
        "",
        "structure:",
        (
            f"complexity={structure.complexity}, parameters={structure.parameter_count}, "
            f"conditions={structure.condition_count}, simplicity_index={structure.simplicity_index}"
        ),
        "",
        "behavior:",
        f"mean_residual_ratio={behavior.mean_residual_ratio}",
        f"residual_variance={behavior.residual_variance}",
        f"fragmentation_index={behavior.fragmentation_index}",
        f"open_rate_early={behavior.resource_opening_rate_early}",
        f"open_rate_mid={behavior.resource_opening_rate_mid}",
        f"open_rate_late={behavior.resource_opening_rate_late}",
        f"choice_entropy={behavior.choice_entropy}",
        f"extreme_option_preference={behavior.extreme_option_preference}",
        f"score_margin_mean={behavior.score_margin_mean}",
        f"score_margin_variance={behavior.score_margin_variance}",
        f"order_sensitivity={behavior.order_sensitivity}",
        f"family_variance={behavior.family_variance}",
        f"holdout_gap={behavior.holdout_gap}",
        "",
        "diagnosis:",
        f"labels={diagnosis.labels}",
        f"summary={diagnosis.summary}",
        f"confidence={diagnosis.confidence}",
        f"evidence={diagnosis.key_evidence}",
        "",
        "lineage:",
        f"created_by={lineage.created_by}",
        f"parents={lineage.parent_ids}",
    ]
    return "\n".join(lines)


def build_population_planner_prompt(problem_context, summary, cards):
    cards_text = "\n\n".join(build_heuristic_card_text(card) for card in cards)
    summary_text = build_population_summary_text(summary)
    return (
        "You are the Population Planner for an agentic heuristic search system.\n\n"
        "You are given a population of candidate heuristics for an optimization problem.\n\n"
        "Each heuristic includes:\n"
        "- scalar fitness (lower is better)\n"
        "- an algorithm summary\n"
        "- structural properties\n"
        "- behavioral metrics\n"
        "- a compact diagnosis\n"
        "- lineage information\n\n"
        "Your goal is to decide how the next generation of heuristics should be produced.\n\n"
        "You can propose interventions using the following execution modes:\n\n"
        "rewrite\n"
        "revise an existing heuristic while preserving useful structure\n\n"
        "tune\n"
        "adjust coefficients or small components without major structural changes\n\n"
        "variant\n"
        "generate a related variant from one or more heuristics\n\n"
        "explore\n"
        "generate a more novel heuristic that explores a different direction\n\n"
        "evaluate\n"
        "request additional evaluation rather than generating a new heuristic\n\n"
        "Guidelines:\n"
        "1. Lower fitness is important, but behavior and diagnosis also matter.\n"
        "2. Preserve strong motifs from good heuristics when possible.\n"
        "3. If the best heuristic has clear weaknesses, propose targeted improvements.\n"
        "4. If the population is behaviorally redundant, encourage exploration.\n"
        "5. Slightly worse heuristics may still be useful if they represent different behaviors.\n\n"
        f"Problem Context:\n{problem_context}\n\n"
        f"{summary_text}\n\n"
        f"Candidate Heuristics:\n{cards_text}\n\n"
        "Return ONLY valid JSON:\n"
        "{\n"
        '  "population_assessment": "...",\n'
        '  "overall_strategy": "...",\n'
        '  "interventions": [\n'
        "    {\n"
        '      "targets": ["H1"],\n'
        '      "execution_mode": "rewrite",\n'
        '      "goal": "...",\n'
        '      "instruction": "...",\n'
        '      "offspring_count": 2,\n'
        '      "priority": "high"\n'
        "    }\n"
        "  ],\n"
        '  "preserve_ids": ["H1"],\n'
        '  "deprioritize_ids": ["H6"],\n'
        '  "rationale": ["..."]\n'
        "}"
    )


def _build_generation_header(evolution):
    return (
        evolution.prompt_task
        + "\n"
        + evolution.prompt_inout_inf
        + " "
        + evolution.prompt_other_inf
        + "\n"
        + evolution._strict_output_rules()
    )


def build_rewrite_prompt(evolution, goal, instruction, card):
    return (
        _build_generation_header(evolution)
        + "\nYou are revising an existing heuristic.\n"
        + f"\nGoal:\n{goal}\n"
        + f"\nInstruction:\n{instruction}\n"
        + "\nCurrent heuristic summary:\n"
        + build_heuristic_card_text(card)
        + "\n\nCurrent code:\n"
        + card.code
        + "\n\nProduce a revised heuristic that:\n"
        + "- preserves the core idea\n"
        + "- addresses the diagnosed weakness\n"
        + "- keeps the function signature unchanged\n"
        + "- remains reasonably simple\n"
    )


def build_tune_prompt(evolution, goal, instruction, card):
    return (
        _build_generation_header(evolution)
        + "\nYou are tuning an existing heuristic.\n"
        + f"\nGoal:\n{goal}\n"
        + f"\nInstruction:\n{instruction}\n"
        + "\nCurrent heuristic summary:\n"
        + build_heuristic_card_text(card)
        + "\n\nCurrent code:\n"
        + card.code
        + "\n\nPreserve the main logic of the heuristic.\n"
        + "Only adjust coefficients or small components.\n"
        + "Avoid introducing major new structural logic.\n"
    )


def build_variant_prompt(evolution, goal, instruction, cards):
    parent_text = "\n\n".join(build_heuristic_card_text(card) + "\nCode:\n" + card.code for card in cards)
    return (
        _build_generation_header(evolution)
        + "\nYou are generating a variant of one or more heuristics.\n"
        + f"\nGoal:\n{goal}\n"
        + f"\nInstruction:\n{instruction}\n"
        + "\nParent heuristics:\n"
        + parent_text
        + "\n\nGenerate a new heuristic inspired by these parents but meaningfully different.\n"
        + "Preserve useful motifs when appropriate.\n"
    )


def build_explore_prompt(evolution, goal, instruction, summary, cards):
    context = "\n\n".join(build_heuristic_card_text(card) for card in cards[:3])
    return (
        _build_generation_header(evolution)
        + "\nYou are exploring a new heuristic direction.\n"
        + f"\nGoal:\n{goal}\n"
        + f"\nInstruction:\n{instruction}\n"
        + "\nPopulation context:\n"
        + build_population_summary_text(summary)
        + "\n\nReference heuristics:\n"
        + context
        + "\n\nGenerate a heuristic that explores a new approach while remaining relevant to the problem.\n"
    )
