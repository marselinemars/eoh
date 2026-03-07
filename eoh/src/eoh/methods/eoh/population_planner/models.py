from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class HeuristicDiagnosis:
    labels: List[str]
    summary: str
    confidence: float
    key_evidence: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HeuristicCard:
    id: str
    generation: int
    fitness: Optional[float]
    rank: int
    algorithm_summary: str
    complexity: float
    parameter_count: float
    condition_count: float
    simplicity_index: float
    behavior: Dict[str, Optional[float]]
    behavior_status: Dict[str, str]
    diagnosis: HeuristicDiagnosis
    created_by: str
    parent_ids: List[str]
    code_hash: str = ""
    code: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["diagnosis"] = self.diagnosis.to_dict()
        return payload


@dataclass
class PopulationSummary:
    generation: int
    best_fitness: Optional[float]
    recent_best_history: List[float]
    stagnation_length: int
    structural_diversity: str
    behavioral_diversity: str
    last_major_improvement_generation: Optional[int]
    current_search_regime: str
    notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PlannerIntervention:
    targets: List[str]
    execution_mode: str
    goal: str
    instruction: str
    offspring_count: int
    priority: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PopulationPlannerOutput:
    population_assessment: str
    overall_strategy: str
    interventions: List[PlannerIntervention]
    preserve_ids: List[str]
    deprioritize_ids: List[str]
    rationale: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "population_assessment": self.population_assessment,
            "overall_strategy": self.overall_strategy,
            "interventions": [item.to_dict() for item in self.interventions],
            "preserve_ids": list(self.preserve_ids),
            "deprioritize_ids": list(self.deprioritize_ids),
            "rationale": list(self.rationale),
        }


def validate_population_planner_output(payload: Dict[str, Any], available_ids: List[str]) -> List[str]:
    errors = []
    if not isinstance(payload, dict):
        return ["planner_output_not_object"]
    for key in ["population_assessment", "overall_strategy", "interventions", "preserve_ids", "deprioritize_ids", "rationale"]:
        if key not in payload:
            errors.append(f"missing_{key}")
    if not isinstance(payload.get("interventions"), list) or len(payload.get("interventions", [])) == 0:
        errors.append("interventions_not_nonempty_list")
    if not isinstance(payload.get("preserve_ids"), list):
        errors.append("preserve_ids_not_list")
    if not isinstance(payload.get("deprioritize_ids"), list):
        errors.append("deprioritize_ids_not_list")
    if not isinstance(payload.get("rationale"), list) or len(payload.get("rationale", [])) == 0:
        errors.append("rationale_not_nonempty_list")

    valid_modes = {"rewrite", "tune", "variant", "explore", "evaluate"}
    valid_priorities = {"high", "medium", "low"}
    available_id_set = set(str(x) for x in available_ids)
    for intervention in payload.get("interventions", []) if isinstance(payload.get("interventions"), list) else []:
        if not isinstance(intervention, dict):
            errors.append("intervention_item_not_dict")
            continue
        for key in ["targets", "execution_mode", "goal", "instruction", "offspring_count", "priority"]:
            if key not in intervention:
                errors.append(f"intervention_missing_{key}")
        targets = intervention.get("targets", [])
        if not isinstance(targets, list):
            errors.append("intervention_targets_not_list")
        else:
            for target in targets:
                if str(target) not in available_id_set:
                    errors.append(f"unknown_target:{target}")
        if str(intervention.get("execution_mode", "")) not in valid_modes:
            errors.append(f"invalid_execution_mode:{intervention.get('execution_mode')}")
        if str(intervention.get("priority", "")) not in valid_priorities:
            errors.append(f"invalid_priority:{intervention.get('priority')}")
        try:
            if int(intervention.get("offspring_count", -1)) < 0:
                errors.append("offspring_count_negative")
        except (TypeError, ValueError):
            errors.append("offspring_count_not_int")
    return errors
