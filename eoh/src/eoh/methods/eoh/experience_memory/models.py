from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ExperienceEntry:
    entry_id: str
    run_id: str
    generation_index: int
    candidate_id: str
    candidate_code: str
    operator: str
    fitness: Optional[float]
    validity: bool
    timestamp: str
    problem_tag: str
    parent_ids: List[str] = field(default_factory=list)
    parent_hashes: List[str] = field(default_factory=list)
    heuristic_summary: Optional[str] = None
    structural_stats: Dict[str, Any] = field(default_factory=dict)
    behavior_stats: Dict[str, Any] = field(default_factory=dict)
    diagnosis_text: Optional[str] = None
    parent_fitness: Optional[float] = None
    fitness_delta: Optional[float] = None
    entered_next_population: Optional[bool] = None
    failure_reason: Optional[str] = None
    planner_rationale: Optional[str] = None
    prompt_metadata: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ExperienceEntry":
        return cls(
            entry_id=str(payload.get("entry_id", "")),
            run_id=str(payload.get("run_id", "")),
            generation_index=int(payload.get("generation_index", 0) or 0),
            candidate_id=str(payload.get("candidate_id", "")),
            candidate_code=str(payload.get("candidate_code", "")),
            operator=str(payload.get("operator", "")),
            fitness=payload.get("fitness"),
            validity=bool(payload.get("validity", False)),
            timestamp=str(payload.get("timestamp", "")),
            problem_tag=str(payload.get("problem_tag", "")),
            parent_ids=list(payload.get("parent_ids", []) or []),
            parent_hashes=list(payload.get("parent_hashes", []) or []),
            heuristic_summary=payload.get("heuristic_summary"),
            structural_stats=dict(payload.get("structural_stats", {}) or {}),
            behavior_stats=dict(payload.get("behavior_stats", {}) or {}),
            diagnosis_text=payload.get("diagnosis_text"),
            parent_fitness=payload.get("parent_fitness"),
            fitness_delta=payload.get("fitness_delta"),
            entered_next_population=payload.get("entered_next_population"),
            failure_reason=payload.get("failure_reason"),
            planner_rationale=payload.get("planner_rationale"),
            prompt_metadata=dict(payload.get("prompt_metadata", {}) or {}),
            extra=dict(payload.get("extra", {}) or {}),
        )
