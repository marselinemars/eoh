# Agentic Full Search Architecture (EOH Overlay)

## Scope
This design implements a behavior-aware agentic workflow on top of EOH, while preserving:
- existing EOH operator prompts (`e1/e2/m1/m2/m3`)
- scalar-fitness-based acceptance/selection
- existing evaluation path

The system introduces structured measurement, diagnosis, intervention, reflection, and memory loops.

## Module layout
`eoh/src/eoh/methods/eoh/agentic_full/`
- `models.py`
  - Dataclass models:
    - `HeuristicProfile`
    - `MeasurementPlan`
    - `BehaviorEvidenceReport`
    - `DiagnosisReport`
    - `InterventionPortfolio`
    - `ReflectionReport`
  - Validation helpers for strict JSON artifacts.
- `ontology.py`
  - `MetricRegistry` and `ActionRegistry`
  - Default metric ontology (resource/decision/temporal/robustness/structure/search)
  - Default action ontology (evolutionary/corrective/evaluation/search_structure/memory)
  - Corrective action -> prompt modifier map.
- `profiles.py`
  - Deterministic `HeuristicProfile` builder from EOH population.
  - Structural metrics proxy extraction (`code_length`, `ast_depth`, etc.).
- `observer.py`
  - Procedural `EvidenceBuilder` for MeasurementPlan execution.
  - Budget-aware extraction from observation + profile summaries.
- `executor.py`
  - `InterventionExecutor` maps action ontology -> executable controls:
    - `op_probs`
    - `parent_mix`
    - `prompt_modifiers`
    - `evaluation_plan`
- `memory.py`
  - Persistent JSONL memory store (`memory_updates.jsonl`)
- `controller.py`
  - `AgenticFullController` (LLM planner/analyst/strategist/reflection + deterministic observer/executor/memory)
  - Compatible return contract with existing routed loop.

## Action family execution mapping
### Evolutionary
- `evolutionary.global_novelty` -> `e1`
- `evolutionary.backbone_variant` -> `e2`
- `evolutionary.structural_modification` -> `m1`
- `evolutionary.parameter_tuning` -> `m2`
- `evolutionary.simplification` -> `m3`
- `evolutionary.hybridize_parents` -> `e2` (diverse mix encouraged)
- `evolutionary.recombine_motifs` -> `e2` (motif-aware variant)

### Corrective (v1 implementation)
Corrective actions are translated into prompt modifiers appended to original EOH prompts, e.g.:
- `corrective.simplify_logic` -> "Keep scoring logic simple; avoid deep nested conditions."
- `corrective.soften_thresholds` -> "Prefer smooth penalties over hard thresholds."
- `corrective.adjust_tie_breaking_behavior` -> "Use stable tie-breaking to reduce brittle decisions."

No hardcoded AST rewrites are required for v1.

### Evaluation
Mapped as evaluation requests in portfolio execution metadata (kept explicit for future executor hooks).

### Search structure
Mapped to deterministic control knobs:
- adjust `parent_mix` (`refresh_diversity_pool`, `split_exploitation_branch`)
- adjust `evaluation_plan.instances` (`allocate_budget_to_branch`)

### Memory
Mapped to persistent JSONL updates for reuse and future priors.

## EOH integration points
Updated in `eoh/src/eoh/methods/eoh/eoh.py`:
- Added mode: `agentic_full`
- Instantiates `AgenticFullController` when `eoh_mode='agentic_full'`
- Reuses routed execution core (operator-mixture execution already implemented)
- Writes new artifact logs:
  - `measurement_plan.jsonl`
  - `behavior_evidence.jsonl`
  - `diagnosis_report.jsonl`
  - `intervention_portfolio.jsonl`
  - `reflection_report.jsonl`
  - `heuristic_profiles.jsonl`
  - `memory_updates.jsonl`
- Calls `reflect_generation(...)` after generation outcomes are known.

## Generic per-generation loop
1. Build `HeuristicProfile` set from current population.
2. Measurement Planner (LLM) -> `MeasurementPlan`
3. Observer (procedural) -> `BehaviorEvidenceReport`
4. Analyst (LLM) -> `DiagnosisReport`
5. Strategist (LLM) -> `InterventionPortfolio`
6. Executor (procedural) -> executable EOH controls (`op_probs`, `parent_mix`, prompt modifiers)
7. Run mixed-operator generation and evaluate offspring.
8. Reflection Agent (LLM) compares portfolio vs outcomes -> `ReflectionReport`
9. Memory writes updates to `memory_updates.jsonl`

## Comparison modes
- `baseline`
- `routed` (routing_v1)
- `agentic_full`

Optional ablations supported by control flags:
- disable critic (`route_controller_use_critic=False`)
- operator-order shuffle (`route_shuffle_operator_order=True`)
- future: disable corrective actions / disable measurement planner / evolutionary-only

## Notes on bin packing trace metrics
v1 evidence builder supports ontology names and extracts from:
- search telemetry (`ObservationPacket`)
- structural profile proxies

For richer bin-packing behavioral traces (residual distributions, decision entropy from per-item choices), extend evaluator with per-instance trace hooks and register them under existing metric ontology keys.
