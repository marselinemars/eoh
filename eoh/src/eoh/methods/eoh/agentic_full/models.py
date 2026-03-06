from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clip01(value: Any, default: float = 0.0) -> float:
    v = _as_float(value, default)
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


@dataclass
class HeuristicProfile:
    heuristic_id: str
    code_hash: str
    summary: str
    scalar_fitness: Optional[float]
    complexity_metrics: Dict[str, float]
    behavior_trace_summary: Dict[str, float]
    instance_family_performance: Dict[str, float]
    parent_lineage: List[str]
    generating_action: str
    generation_created: int
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MeasurementPlan:
    plan_id: str
    problem_context: Dict[str, Any]
    measurement_objective: str
    targets: List[Dict[str, Any]]
    requested_metrics: List[str]
    sampling_policy: Dict[str, Any]
    expected_value: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BehaviorEvidenceReport:
    report_id: str
    based_on_measurement_plan: str
    generation: int
    metric_values: Dict[str, Dict[str, Any]]
    metric_interpretations: Dict[str, str]
    comparative_views: Dict[str, Any]
    evidence_quality: Dict[str, Any]
    target_summaries: List[Dict[str, Any]]
    notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DiagnosisReport:
    diagnosis_id: str
    based_on_measurement_plan: str
    search_state_summary: str
    diagnoses: List[Dict[str, Any]]
    hypotheses: List[Dict[str, Any]]
    intervention_goals: List[str]
    uncertainties: List[str]
    recommended_focus: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InterventionPortfolio:
    portfolio_id: str
    based_on_diagnosis: str
    search_regime: str
    generation_objective: str
    interventions: List[Dict[str, Any]]
    branches: List[Dict[str, Any]]
    budget_allocation: Dict[str, float]
    branch_policy: Dict[str, Any]
    parent_candidate_groups_used: List[str]
    success_criteria: List[str]
    rationale: List[str]
    fallback_used: bool = False
    fallback_reason: str = ""
    missing_required_metrics: List[str] = field(default_factory=list)
    source: str = "llm"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReflectionReport:
    reflection_id: str
    based_on_portfolio: str
    outcome_summary: Dict[str, Any]
    behavioral_outcomes: List[Dict[str, Any]]
    supported_hypotheses: List[str]
    rejected_hypotheses: List[str]
    observed_effects: List[str]
    tradeoffs: List[str]
    lessons: List[str]
    memory_updates: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def validate_measurement_plan(payload: Dict[str, Any]) -> List[str]:
    errors = []
    if not isinstance(payload, dict):
        return ["measurement_plan_not_object"]
    for key in [
        "plan_id",
        "problem_context",
        "measurement_objective",
        "targets",
        "requested_metrics",
        "sampling_policy",
        "expected_value",
    ]:
        if key not in payload:
            errors.append(f"missing_{key}")
    if not isinstance(payload.get("targets"), list):
        errors.append("targets_not_list")
    if not isinstance(payload.get("requested_metrics"), list):
        errors.append("requested_metrics_not_list")
    return errors


def validate_behavior_evidence_report(payload: Dict[str, Any]) -> List[str]:
    errors = []
    if not isinstance(payload, dict):
        return ["behavior_evidence_report_not_object"]
    for key in [
        "report_id",
        "based_on_measurement_plan",
        "generation",
        "metric_values",
        "metric_interpretations",
        "comparative_views",
        "evidence_quality",
        "target_summaries",
        "notes",
    ]:
        if key not in payload:
            errors.append(f"missing_{key}")
    if not isinstance(payload.get("metric_values"), dict):
        errors.append("metric_values_not_dict")
    if not isinstance(payload.get("metric_interpretations"), dict):
        errors.append("metric_interpretations_not_dict")
    if not isinstance(payload.get("comparative_views"), dict):
        errors.append("comparative_views_not_dict")
    if not isinstance(payload.get("evidence_quality"), dict):
        errors.append("evidence_quality_not_dict")
    if not isinstance(payload.get("target_summaries"), list):
        errors.append("target_summaries_not_list")
    if not isinstance(payload.get("notes"), list):
        errors.append("notes_not_list")
    return errors


def validate_diagnosis_report(payload: Dict[str, Any]) -> List[str]:
    errors = []
    if not isinstance(payload, dict):
        return ["diagnosis_report_not_object"]
    for key in [
        "diagnosis_id",
        "based_on_measurement_plan",
        "search_state_summary",
        "diagnoses",
        "hypotheses",
        "intervention_goals",
        "uncertainties",
        "recommended_focus",
    ]:
        if key not in payload:
            errors.append(f"missing_{key}")
    for container_key in ["diagnoses", "hypotheses", "intervention_goals", "uncertainties", "recommended_focus"]:
        if not isinstance(payload.get(container_key), list):
            errors.append(f"{container_key}_not_list")
    return errors


def validate_intervention_portfolio(payload: Dict[str, Any]) -> List[str]:
    errors = []
    if not isinstance(payload, dict):
        return ["intervention_portfolio_not_object"]
    for key in [
        "portfolio_id",
        "based_on_diagnosis",
        "search_regime",
        "generation_objective",
        "interventions",
        "branches",
        "budget_allocation",
        "branch_policy",
        "parent_candidate_groups_used",
        "success_criteria",
        "rationale",
        "fallback_used",
        "fallback_reason",
        "missing_required_metrics",
        "source",
    ]:
        if key not in payload:
            errors.append(f"missing_{key}")
    if not isinstance(payload.get("interventions"), list):
        errors.append("interventions_not_list")
    if not isinstance(payload.get("branches"), list):
        errors.append("branches_not_list")
    if not isinstance(payload.get("budget_allocation"), dict):
        errors.append("budget_allocation_not_dict")
    if not isinstance(payload.get("parent_candidate_groups_used"), list):
        errors.append("parent_candidate_groups_used_not_list")
    if not isinstance(payload.get("missing_required_metrics"), list):
        errors.append("missing_required_metrics_not_list")
    return errors


def validate_reflection_report(payload: Dict[str, Any]) -> List[str]:
    errors = []
    if not isinstance(payload, dict):
        return ["reflection_report_not_object"]
    for key in [
        "reflection_id",
        "based_on_portfolio",
        "outcome_summary",
        "behavioral_outcomes",
        "supported_hypotheses",
        "rejected_hypotheses",
        "observed_effects",
        "tradeoffs",
        "lessons",
        "memory_updates",
    ]:
        if key not in payload:
            errors.append(f"missing_{key}")
    for container_key in ["behavioral_outcomes", "supported_hypotheses", "rejected_hypotheses", "observed_effects", "tradeoffs", "lessons", "memory_updates"]:
        if not isinstance(payload.get(container_key), list):
            errors.append(f"{container_key}_not_list")
    return errors


def normalize_op_probs(values: Dict[str, Any]) -> Dict[str, float]:
    keys = ["e1", "e2", "m1", "m2", "m3"]
    cleaned = {k: max(0.0, _as_float(values.get(k), 0.0)) for k in keys}
    total = sum(cleaned.values())
    if total <= 1e-12:
        return {k: 1.0 / len(keys) for k in keys}
    return {k: cleaned[k] / total for k in keys}


def normalize_parent_mix(values: Dict[str, Any]) -> Dict[str, float]:
    keys = ["elite", "diverse", "random"]
    if isinstance(values, dict) and "preferred" in values:
        keys.append("preferred")
    cleaned = {k: max(0.0, _as_float(values.get(k), 0.0)) for k in keys}
    total = sum(cleaned.values())
    if total <= 1e-12:
        fallback = {"elite": 0.5, "diverse": 0.3, "random": 0.2}
        if "preferred" in keys:
            fallback = {"elite": 0.4, "diverse": 0.25, "random": 0.15, "preferred": 0.20}
        return fallback
    return {k: cleaned[k] / total for k in keys}


def normalize_diagnosis_factor(value: Any) -> float:
    return _clip01(value, default=0.5)
