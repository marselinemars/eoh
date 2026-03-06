from dataclasses import asdict, dataclass, field
from typing import Dict, List


@dataclass
class StructureMetrics:
    complexity: float
    parameter_count: int
    condition_count: int
    simplicity_index: float

    def to_dict(self):
        return asdict(self)


@dataclass
class BehaviorMetrics:
    mean_residual_ratio: float
    residual_variance: float
    fragmentation_index: float
    resource_opening_rate_early: float
    resource_opening_rate_mid: float
    resource_opening_rate_late: float
    choice_entropy: float
    extreme_option_preference: float
    score_margin_mean: float
    score_margin_variance: float
    order_sensitivity: float
    family_variance: float
    holdout_gap: float

    def to_dict(self):
        return asdict(self)


@dataclass
class HeuristicDiagnosis:
    labels: List[str]
    summary: str
    confidence: float
    key_evidence: List[str]

    def to_dict(self):
        return asdict(self)


@dataclass
class LineageInfo:
    created_by: str
    parent_ids: List[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


@dataclass
class HeuristicCard:
    id: str
    generation: int
    fitness: float
    rank: int
    algorithm_summary: str
    structure: StructureMetrics
    behavior: BehaviorMetrics
    diagnosis: HeuristicDiagnosis
    lineage: LineageInfo
    code: str
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass
class PopulationSummary:
    generation: int
    best_fitness: float
    recent_best_fitness_history: List[float]
    stagnation_length: int
    structural_diversity_estimate: str
    behavioral_diversity_estimate: str
    last_major_improvement_generation: int
    current_search_regime: str
    notes: List[str]

    def to_dict(self):
        return asdict(self)


@dataclass
class PlannerIntervention:
    targets: List[str]
    execution_mode: str
    goal: str
    instruction: str
    offspring_count: int
    priority: str

    def to_dict(self):
        return asdict(self)


@dataclass
class PopulationPlan:
    population_assessment: str
    overall_strategy: str
    interventions: List[PlannerIntervention]
    preserve_ids: List[str]
    deprioritize_ids: List[str]
    rationale: List[str]

    def to_dict(self):
        return {
            "population_assessment": self.population_assessment,
            "overall_strategy": self.overall_strategy,
            "interventions": [item.to_dict() for item in self.interventions],
            "preserve_ids": list(self.preserve_ids),
            "deprioritize_ids": list(self.deprioritize_ids),
            "rationale": list(self.rationale),
        }
