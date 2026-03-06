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
    generation_objective: str
    interventions: List[Dict[str, Any]]
    budget_allocation: Dict[str, float]
    branch_policy: Dict[str, Any]
    success_criteria: List[str]
    rationale: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReflectionReport:
    reflection_id: str
    based_on_portfolio: str
    outcome_summary: Dict[str, Any]
    supported_hypotheses: List[str]
    rejected_hypotheses: List[str]
    observed_effects: List[str]
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
        "generation_objective",
        "interventions",
        "budget_allocation",
        "branch_policy",
        "success_criteria",
        "rationale",
    ]:
        if key not in payload:
            errors.append(f"missing_{key}")
    if not isinstance(payload.get("interventions"), list):
        errors.append("interventions_not_list")
    if not isinstance(payload.get("budget_allocation"), dict):
        errors.append("budget_allocation_not_dict")
    return errors


def validate_reflection_report(payload: Dict[str, Any]) -> List[str]:
    errors = []
    if not isinstance(payload, dict):
        return ["reflection_report_not_object"]
    for key in [
        "reflection_id",
        "based_on_portfolio",
        "outcome_summary",
        "supported_hypotheses",
        "rejected_hypotheses",
        "observed_effects",
        "lessons",
        "memory_updates",
    ]:
        if key not in payload:
            errors.append(f"missing_{key}")
    for container_key in ["supported_hypotheses", "rejected_hypotheses", "observed_effects", "lessons", "memory_updates"]:
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
    cleaned = {k: max(0.0, _as_float(values.get(k), 0.0)) for k in keys}
    total = sum(cleaned.values())
    if total <= 1e-12:
        return {"elite": 0.5, "diverse": 0.3, "random": 0.2}
    return {k: cleaned[k] / total for k in keys}


def normalize_diagnosis_factor(value: Any) -> float:
    return _clip01(value, default=0.5)
