from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


@dataclass
class MetricSpec:
    name: str
    family: str
    description: str
    dtype: str = "float"
    preferred_target: str = "population"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ActionSpec:
    name: str
    family: str
    description: str
    execution_kind: str  # evolutionary_operator | prompt_modifier | eval_request | search_structure | memory
    operator_id: Optional[str] = None
    default_payload: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MetricRegistry:
    def __init__(self):
        self._metrics: Dict[str, MetricSpec] = {}

    def register(self, spec: MetricSpec):
        self._metrics[spec.name] = spec

    def has(self, name: str) -> bool:
        return name in self._metrics

    def get(self, name: str) -> Optional[MetricSpec]:
        return self._metrics.get(name)

    def list(self) -> List[str]:
        return sorted(self._metrics.keys())

    def by_family(self, family: str) -> List[MetricSpec]:
        return [m for m in self._metrics.values() if m.family == family]


class ActionRegistry:
    def __init__(self):
        self._actions: Dict[str, ActionSpec] = {}

    def register(self, spec: ActionSpec):
        self._actions[spec.name] = spec

    def has(self, name: str) -> bool:
        return name in self._actions

    def get(self, name: str) -> Optional[ActionSpec]:
        return self._actions.get(name)

    def list(self) -> List[str]:
        return sorted(self._actions.keys())

    def by_family(self, family: str) -> List[ActionSpec]:
        return [a for a in self._actions.values() if a.family == family]


def build_default_metric_registry() -> MetricRegistry:
    reg = MetricRegistry()
    metrics = [
        ("resource_utilization.mean_usage_ratio", "resource_utilization"),
        ("resource_utilization.mean_residual_ratio", "resource_utilization"),
        ("resource_utilization.residual_variance", "resource_utilization"),
        ("resource_utilization.fragmentation_index", "resource_utilization"),
        ("resource_utilization.utilization_skew", "resource_utilization"),
        ("decision_pattern.greedy_choice_rate", "decision_pattern"),
        ("decision_pattern.tie_break_bias", "decision_pattern"),
        ("decision_pattern.choice_entropy", "decision_pattern"),
        ("decision_pattern.local_vs_global_preference", "decision_pattern"),
        ("decision_pattern.extreme_option_preference", "decision_pattern"),
        ("decision_pattern.score_margin_mean", "decision_pattern"),
        ("decision_pattern.score_margin_variance", "decision_pattern"),
        ("temporal_behavior.early_stage_aggressiveness", "temporal_behavior"),
        ("temporal_behavior.mid_stage_aggressiveness", "temporal_behavior"),
        ("temporal_behavior.late_stage_aggressiveness", "temporal_behavior"),
        ("temporal_behavior.resource_opening_rate_early", "temporal_behavior"),
        ("temporal_behavior.resource_opening_rate_mid", "temporal_behavior"),
        ("temporal_behavior.resource_opening_rate_late", "temporal_behavior"),
        ("temporal_behavior.commitment_stability", "temporal_behavior"),
        ("temporal_behavior.phase_shift_index", "temporal_behavior"),
        ("robustness.order_sensitivity", "robustness"),
        ("robustness.perturbation_sensitivity", "robustness"),
        ("robustness.instance_family_variance", "robustness"),
        ("robustness.cross_seed_variance", "robustness"),
        ("robustness.performance_stability_index", "robustness"),
        ("robustness.holdout_gap", "robustness"),
        ("robustness.behavior_shift_across_families", "robustness"),
        ("structure.code_length", "structure"),
        ("structure.ast_depth", "structure"),
        ("structure.condition_count", "structure"),
        ("structure.parameter_count", "structure"),
        ("structure.operator_count", "structure"),
        ("structure.expression_redundancy", "structure"),
        ("structure.simplicity_index", "structure"),
        ("structure.motif_similarity_to_best", "structure"),
        ("structure.population_structural_diversity", "structure"),
        ("search.best_fitness", "search"),
        ("search.best_improvement_rate", "search"),
        ("search.stagnation_length", "search"),
        ("search.invalid_rate", "search"),
        ("search.operator_success_rate", "search"),
        ("search.operator_mean_delta", "search"),
        ("search.population_fitness_diversity", "search"),
        ("search.behavioral_diversity", "search"),
        ("search.novelty_yield", "search"),
        ("search.branch_productivity", "search"),
    ]
    for name, family in metrics:
        reg.register(
            MetricSpec(
                name=name,
                family=family,
                description=f"{name} metric in {family} family.",
            )
        )
    return reg


def build_default_action_registry() -> ActionRegistry:
    reg = ActionRegistry()
    actions = [
        ("evolutionary.global_novelty", "evolutionary", "Use e1 operator.", "evolutionary_operator", "e1"),
        ("evolutionary.backbone_variant", "evolutionary", "Use e2 operator.", "evolutionary_operator", "e2"),
        ("evolutionary.structural_modification", "evolutionary", "Use m1 operator.", "evolutionary_operator", "m1"),
        ("evolutionary.parameter_tuning", "evolutionary", "Use m2 operator.", "evolutionary_operator", "m2"),
        ("evolutionary.simplification", "evolutionary", "Use m3 operator.", "evolutionary_operator", "m3"),
        ("evolutionary.hybridize_parents", "evolutionary", "Map to e2 with diverse parents.", "evolutionary_operator", "e2"),
        ("evolutionary.recombine_motifs", "evolutionary", "Map to e2 with motif-aware context.", "evolutionary_operator", "e2"),
        ("corrective.simplify_logic", "corrective", "Reduce nested logic.", "prompt_modifier", None),
        ("corrective.soften_thresholds", "corrective", "Avoid hard thresholds.", "prompt_modifier", None),
        ("corrective.reduce_dominant_term", "corrective", "Reduce dominant term weight.", "prompt_modifier", None),
        ("corrective.balance_competing_terms", "corrective", "Balance term trade-offs.", "prompt_modifier", None),
        ("corrective.smooth_penalty_shape", "corrective", "Prefer smooth penalties.", "prompt_modifier", None),
        ("corrective.reduce_feature_sensitivity", "corrective", "Reduce feature sensitivity.", "prompt_modifier", None),
        ("corrective.increase_robustness_bias", "corrective", "Favor robust behavior.", "prompt_modifier", None),
        ("corrective.preserve_motif_retune_local_component", "corrective", "Preserve motif and retune local part.", "prompt_modifier", None),
        ("corrective.remove_redundant_condition", "corrective", "Remove redundant condition blocks.", "prompt_modifier", None),
        ("corrective.adjust_tie_breaking_behavior", "corrective", "Stabilize tie-breaking.", "prompt_modifier", None),
        ("evaluation.run_holdout_test", "evaluation", "Run holdout evaluation.", "eval_request", None),
        ("evaluation.run_hard_case_batch", "evaluation", "Run hard case batch.", "eval_request", None),
        ("evaluation.run_order_perturbation_test", "evaluation", "Run order perturbation test.", "eval_request", None),
        ("evaluation.run_family_wise_comparison", "evaluation", "Run family wise comparison.", "eval_request", None),
        ("evaluation.compare_top_candidates", "evaluation", "Compare top candidates.", "eval_request", None),
        ("evaluation.run_behavior_trace_detail", "evaluation", "Request detailed behavior traces.", "eval_request", None),
        ("evaluation.retest_under_new_seed", "evaluation", "Retest under new seed.", "eval_request", None),
        ("search_structure.split_exploration_branch", "search_structure", "Allocate exploration branch budget.", "search_structure", None),
        ("search_structure.split_exploitation_branch", "search_structure", "Allocate exploitation branch budget.", "search_structure", None),
        ("search_structure.merge_branches", "search_structure", "Merge branch budgets.", "search_structure", None),
        ("search_structure.refresh_diversity_pool", "search_structure", "Increase diverse parent mix.", "search_structure", None),
        ("search_structure.freeze_motif_family", "search_structure", "Freeze motif family.", "search_structure", None),
        ("search_structure.promote_behaviorally_distinct_cluster", "search_structure", "Promote distinct clusters.", "search_structure", None),
        ("search_structure.allocate_budget_to_branch", "search_structure", "Budget re-allocation.", "search_structure", None),
        ("search_structure.archive_failed_family", "search_structure", "Archive failed family.", "search_structure", None),
        ("memory.store_success_pattern", "memory", "Store success pattern.", "memory", None),
        ("memory.store_failure_pattern", "memory", "Store failure pattern.", "memory", None),
        ("memory.update_diagnosis_action_prior", "memory", "Update diagnosis-action prior.", "memory", None),
        ("memory.record_behavior_cluster", "memory", "Record behavior cluster.", "memory", None),
        ("memory.promote_reusable_motif", "memory", "Promote reusable motif.", "memory", None),
        ("memory.demote_unproductive_action_pattern", "memory", "Demote unproductive pattern.", "memory", None),
    ]
    for name, family, description, kind, op_id in actions:
        reg.register(
            ActionSpec(
                name=name,
                family=family,
                description=description,
                execution_kind=kind,
                operator_id=op_id,
                default_payload={},
            )
        )
    return reg


CORRECTIVE_PROMPT_MODIFIER_MAP: Dict[str, str] = {
    "corrective.simplify_logic": "Keep scoring logic simple; avoid deep nested conditions.",
    "corrective.soften_thresholds": "Prefer smooth penalties over hard thresholds.",
    "corrective.reduce_dominant_term": "Avoid a single dominant score term; balance contributions.",
    "corrective.balance_competing_terms": "Balance competing score terms with moderate weights.",
    "corrective.smooth_penalty_shape": "Use smooth penalty shapes and avoid abrupt cliffs.",
    "corrective.reduce_feature_sensitivity": "Reduce sensitivity to one feature or one range.",
    "corrective.increase_robustness_bias": "Favor robust behavior across item orders.",
    "corrective.preserve_motif_retune_local_component": "Preserve good motif and retune local component only.",
    "corrective.remove_redundant_condition": "Remove redundant conditions and duplicate checks.",
    "corrective.adjust_tie_breaking_behavior": "Use stable tie-breaking to reduce brittle decisions.",
}
