import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..agentic_controller import AgenticController, _to_float
from .executor import InterventionExecutor
from .memory import AgenticMemory
from .models import (
    BehaviorEvidenceReport,
    DiagnosisReport,
    InterventionPortfolio,
    MeasurementPlan,
    ReflectionReport,
    validate_behavior_evidence_report,
    validate_diagnosis_report,
    validate_intervention_portfolio,
    validate_measurement_plan,
    validate_reflection_report,
)
from .ontology import build_default_action_registry, build_default_metric_registry
from .observer import EvidenceBuilder, REQUIRED_BEHAVIOR_METRICS
from .profiles import build_heuristic_profiles


MEASUREMENT_PLANNER_PROMPT = """OUTPUT JSON ONLY. No markdown.
You are Measurement Planner Agent for generic heuristic search.
Problem context:
{PROBLEM_CONTEXT}

Observation:
{OBS_JSON}

Top heuristic profiles summary:
{PROFILE_SUMMARY_JSON}

Select what to measure under current budget. Use ontology metric names.
Return ONLY JSON with keys:
{{
  "plan_id": "string",
  "problem_context": {{}},
  "measurement_objective": "string",
  "targets": [{{"target_id":"string","target_type":"population|cluster|heuristic","selector":{{}},"notes":"string"}}],
  "requested_metrics": ["metric.name"],
  "sampling_policy": {{}},
  "expected_value": ["string"]
}}
Rules:
- requested_metrics must include the main bp_online behavior metrics:
  resource_utilization.mean_residual_ratio
  resource_utilization.residual_variance
  resource_utilization.fragmentation_index
  temporal_behavior.resource_opening_rate_early
  temporal_behavior.resource_opening_rate_mid
  temporal_behavior.resource_opening_rate_late
  temporal_behavior.phase_shift_index
  decision_pattern.extreme_option_preference
  decision_pattern.choice_entropy
  decision_pattern.score_margin_mean
  decision_pattern.score_margin_variance
  robustness.order_sensitivity
  robustness.instance_family_variance
  robustness.holdout_gap
  structure.simplicity_index
  structure.parameter_count
  structure.condition_count
- Keep budget-aware: do not exceed Observation.budget.instances.
"""


ANALYST_PROMPT = """OUTPUT JSON ONLY. No markdown.
You are Analyst Agent. Build rich diagnoses from evidence.
Observation:
{OBS_JSON}

MeasurementPlan:
{PLAN_JSON}

BehaviorEvidenceReport:
{EVIDENCE_JSON}

Return ONLY JSON with keys:
{{
  "diagnosis_id": "string",
  "based_on_measurement_plan": "string",
  "search_state_summary": "string",
  "diagnoses": [{{"label":"string","severity":0.0,"confidence":0.0,"evidence":["string"]}}],
  "hypotheses": [{{"hypothesis":"string","support":0.0,"counter_evidence":["string"]}}],
  "intervention_goals": ["string"],
  "uncertainties": ["string"],
  "recommended_focus": ["string"]
}}
Rules:
- severity/support/confidence in [0,1]
- diagnoses and hypotheses must be non-empty
- Each diagnosis evidence array must cite at least 2 specific metric names or interpreted evidence strings from BehaviorEvidenceReport.
- Prefer behavior-specific labels when supported, such as:
  aggressive_early_fill_with_fragmentation
  brittle_low_margin_tie_behavior
  behaviorally_redundant_population
  complexity_barrier_with_low_margin_decisions
- Do not overuse generic labels like stagnation if more specific behavioral evidence is available.
"""


STRATEGIST_PROMPT = """OUTPUT JSON ONLY. No markdown.
You are Strategist Agent. Plan interventions using action ontology.
DiagnosisReport:
{DIAG_JSON}

BehaviorEvidenceReport:
{EVIDENCE_JSON}

Observation:
{OBS_JSON}

Available action families:
- evolutionary.*
- corrective.*
- evaluation.*
- search_structure.*
- memory.*

Return ONLY JSON:
{{
  "portfolio_id": "string",
  "based_on_diagnosis": "string",
  "generation_objective": "string",
  "interventions": [{{"action":"family.name","weight":0.0,"payload":{{}}}}],
  "budget_allocation": {{"exploration":0.0,"exploitation":0.0,"evaluation":0.0}},
  "branch_policy": {{}},
  "success_criteria": ["string"],
  "rationale": ["string"],
  "fallback_used": false,
  "fallback_reason": "",
  "missing_required_metrics": ["metric.name"],
  "source": "llm"
}}
Rules:
- interventions non-empty and use ontology action names.
- weights in [0,1]
- budget_allocation values in [0,1]
- If evidence_quality.level is strong or partial, do not produce a generic fallback portfolio.
- Use corrective actions when diagnoses imply them:
  complexity barrier -> corrective.simplify_logic
  aggressive early fill / fragmentation -> corrective.soften_thresholds or corrective.reduce_dominant_term
  brittle tie behavior / low score margins -> corrective.adjust_tie_breaking_behavior
  preserve useful motif but retune local behavior -> corrective.preserve_motif_retune_local_component
- Rationale must cite metric names from BehaviorEvidenceReport.
"""


REFLECTION_PROMPT = """OUTPUT JSON ONLY. No markdown.
You are Reflection Agent.
DiagnosisReport:
{DIAG_JSON}

InterventionPortfolio:
{PORT_JSON}

Outcome summary:
{OUTCOME_JSON}

Return ONLY JSON:
{{
  "reflection_id": "string",
  "based_on_portfolio": "string",
  "outcome_summary": {{}},
  "supported_hypotheses": ["string"],
  "rejected_hypotheses": ["string"],
  "observed_effects": ["string"],
  "lessons": ["string"],
  "memory_updates": [{{"kind":"string","payload":{{}}}}]
}}
"""


class AgenticFullController(AgenticController):
    def __init__(
        self,
        api_endpoint: str,
        api_key: str,
        model_llm: str,
        llm_use_local: bool,
        llm_local_url: str,
        output_path: str,
        use_critic_agent: bool = False,
        debug_mode: bool = False,
    ):
        super().__init__(
            api_endpoint=api_endpoint,
            api_key=api_key,
            model_llm=model_llm,
            llm_use_local=llm_use_local,
            llm_local_url=llm_local_url,
            use_critic_agent=use_critic_agent,
            debug_mode=debug_mode,
        )
        self.metric_registry = build_default_metric_registry()
        self.action_registry = build_default_action_registry()
        self.observer = EvidenceBuilder()
        self.executor = InterventionExecutor(self.action_registry)
        self.memory = AgenticMemory(memory_path=f"{output_path}/results/memory_store.jsonl")
        self._pending: Dict[int, Dict[str, Any]] = {}

    def _now(self):
        return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    def _validate_measurement_plan(self, payload: Dict[str, Any]) -> List[str]:
        errors = validate_measurement_plan(payload)
        metrics = payload.get("requested_metrics", [])
        if isinstance(metrics, list):
            invalid = [m for m in metrics if not self.metric_registry.has(str(m))]
            if len(invalid) > 0:
                errors.append("unknown_metrics:" + ",".join(str(x) for x in invalid[:8]))
            fam = {"search": 0, "structure": 0, "robustness": 0, "temporal_behavior": 0}
            for m in metrics:
                ms = self.metric_registry.get(str(m))
                if ms is not None and ms.family in fam:
                    fam[ms.family] += 1
            for needed in ["search", "structure", "robustness", "temporal_behavior"]:
                if fam[needed] == 0:
                    errors.append(f"missing_family_{needed}")
            missing_required = [m for m in REQUIRED_BEHAVIOR_METRICS if m not in metrics]
            if len(missing_required) > 0:
                errors.append("missing_required_behavior_metrics:" + ",".join(missing_required[:8]))
            if len(metrics) < 12:
                errors.append("requested_metrics_too_short")
        return errors

    def _validate_analyst(self, payload: Dict[str, Any]) -> List[str]:
        errors = validate_diagnosis_report(payload)
        for d in payload.get("diagnoses", []) if isinstance(payload.get("diagnoses"), list) else []:
            if not isinstance(d, dict):
                errors.append("diagnosis_item_not_dict")
                continue
            if _to_float(d.get("severity"), None) is None or _to_float(d.get("confidence"), None) is None:
                errors.append("diagnosis_scores_invalid")
            evidence = d.get("evidence", [])
            if not isinstance(evidence, list) or len(evidence) < 2:
                errors.append("diagnosis_evidence_too_short")
                continue
            metric_citations = 0
            for line in evidence:
                line_text = str(line)
                if any(metric_name in line_text for metric_name in REQUIRED_BEHAVIOR_METRICS):
                    metric_citations += 1
            if metric_citations < 1:
                errors.append("diagnosis_missing_metric_citation")
        return errors

    def _validate_strategist(self, payload: Dict[str, Any]) -> List[str]:
        errors = validate_intervention_portfolio(payload)
        interventions = payload.get("interventions", [])
        if isinstance(interventions, list):
            if len(interventions) == 0:
                errors.append("empty_interventions")
            for item in interventions:
                if not isinstance(item, dict):
                    errors.append("intervention_item_not_dict")
                    continue
                action = str(item.get("action", ""))
                if not self.action_registry.has(action):
                    errors.append(f"unknown_action:{action}")
                w = _to_float(item.get("weight"), None)
                if w is None or w < 0.0 or w > 1.0:
                    errors.append("invalid_intervention_weight")
        rationale = payload.get("rationale", [])
        if not isinstance(rationale, list) or len(rationale) < 2:
            errors.append("rationale_too_short")
        else:
            if not any(any(metric_name in str(line) for metric_name in REQUIRED_BEHAVIOR_METRICS) for line in rationale):
                errors.append("rationale_missing_behavior_metric_citation")
        return errors

    def _validate_reflection(self, payload: Dict[str, Any]) -> List[str]:
        return validate_reflection_report(payload)

    def _missing_required_metrics(self, evidence: Dict[str, Any]) -> List[str]:
        if not isinstance(evidence, dict):
            return list(REQUIRED_BEHAVIOR_METRICS)
        metric_values = evidence.get("metric_values", {}) if isinstance(evidence.get("metric_values"), dict) else {}
        missing = []
        for metric_name in REQUIRED_BEHAVIOR_METRICS:
            entry = metric_values.get(metric_name, {})
            if not isinstance(entry, dict) or entry.get("value") is None:
                missing.append(metric_name)
        return missing

    def _evidence_is_sufficient(self, evidence: Dict[str, Any]) -> bool:
        errors = validate_behavior_evidence_report(evidence)
        if len(errors) > 0:
            return False
        evidence_quality = evidence.get("evidence_quality", {}) if isinstance(evidence.get("evidence_quality"), dict) else {}
        if evidence_quality.get("level") in ["strong", "partial"]:
            return True
        return len(self._missing_required_metrics(evidence)) <= 3

    def _fallback_measurement_plan(self, observation: Dict[str, Any], generation_index: int) -> Dict[str, Any]:
        return {
            "plan_id": f"mp_fallback_g{generation_index}",
            "problem_context": {"problem": "bp_online"},
            "measurement_objective": "Track behavior-aware search state, robustness, and structure under budget.",
            "targets": [{"target_id": "population_top", "target_type": "population", "selector": {"top_k": 8}, "notes": "default top population target"}],
            "requested_metrics": REQUIRED_BEHAVIOR_METRICS + [
                "search.best_fitness",
                "search.best_improvement_rate",
                "search.stagnation_length",
                "search.invalid_rate",
                "search.operator_mean_delta",
            ],
            "sampling_policy": {"instances": int(observation.get("budget", {}).get("instances", 0) or 0), "granularity": "population"},
            "expected_value": ["Detect behavior-level failure modes.", "Guide targeted intervention selection."],
        }

    def _fallback_diagnosis(self, evidence: Dict[str, Any], measurement_plan: Dict[str, Any], generation_index: int) -> Dict[str, Any]:
        metric_values = evidence.get("metric_values", {}) if isinstance(evidence.get("metric_values"), dict) else {}
        stagnation = _to_float((metric_values.get("search.stagnation_length") or {}).get("value"), 0.0)
        invalid_rate = _to_float((metric_values.get("search.invalid_rate") or {}).get("value"), 0.0)
        early_open = _to_float((metric_values.get("temporal_behavior.resource_opening_rate_early") or {}).get("value"), 0.0)
        fragmentation = _to_float((metric_values.get("resource_utilization.fragmentation_index") or {}).get("value"), 0.0)
        choice_entropy = _to_float((metric_values.get("decision_pattern.choice_entropy") or {}).get("value"), 0.0)
        score_margin = _to_float((metric_values.get("decision_pattern.score_margin_mean") or {}).get("value"), 0.0)
        simplicity = _to_float((metric_values.get("structure.simplicity_index") or {}).get("value"), 0.0)
        diagnoses = []
        if early_open >= 0.40 and fragmentation >= 0.45:
            diagnoses.append(
                {
                    "label": "aggressive_early_fill_with_fragmentation",
                    "severity": min(1.0, 0.5 * early_open + 0.5 * fragmentation),
                    "confidence": 0.70,
                    "evidence": [
                        f"temporal_behavior.resource_opening_rate_early={early_open:.4f}",
                        f"resource_utilization.fragmentation_index={fragmentation:.4f}",
                    ],
                }
            )
        if choice_entropy <= 0.25 and score_margin <= 0.05:
            diagnoses.append(
                {
                    "label": "brittle_low_margin_tie_behavior",
                    "severity": min(1.0, (1.0 - choice_entropy) * 0.5 + max(0.0, 0.10 - score_margin) * 5.0),
                    "confidence": 0.68,
                    "evidence": [
                        f"decision_pattern.choice_entropy={choice_entropy:.4f}",
                        f"decision_pattern.score_margin_mean={score_margin:.4f}",
                    ],
                }
            )
        if simplicity <= 0.35:
            diagnoses.append(
                {
                    "label": "complexity_barrier_with_low_margin_decisions",
                    "severity": min(1.0, 1.0 - simplicity),
                    "confidence": 0.62,
                    "evidence": [
                        f"structure.simplicity_index={simplicity:.4f}",
                        f"search.stagnation_length={stagnation:.4f}",
                    ],
                }
            )
        if len(diagnoses) == 0:
            diagnoses.append(
                {
                    "label": "search_stagnation_without_behavioral_separation" if stagnation >= 2 else "search_active",
                    "severity": min(1.0, stagnation / 6.0),
                    "confidence": 0.55,
                    "evidence": [
                        f"search.stagnation_length={stagnation:.4f}",
                        f"search.invalid_rate={invalid_rate:.4f}",
                    ],
                }
            )
        return {
            "diagnosis_id": f"diag_fallback_g{generation_index}",
            "based_on_measurement_plan": measurement_plan.get("plan_id", "unknown"),
            "search_state_summary": "Deterministic diagnosis grounded in behavior evidence.",
            "diagnoses": diagnoses,
            "hypotheses": [
                {
                    "hypothesis": "Behavior-aware corrective edits are needed before more exploration.",
                    "support": min(1.0, 0.5 + stagnation / 10.0),
                    "counter_evidence": [],
                }
            ],
            "intervention_goals": ["restore improvement trend", "reduce invalid pressure"],
            "uncertainties": ["llm_diagnosis_unavailable; using procedural grounding"],
            "recommended_focus": ["evolutionary.backbone_variant", "corrective.simplify_logic", "corrective.soften_thresholds"],
        }

    def _diagnosis_driven_portfolio(self, diagnosis: Dict[str, Any], evidence: Dict[str, Any], generation_index: int) -> Dict[str, Any]:
        labels = [
            str(item.get("label", ""))
            for item in diagnosis.get("diagnoses", [])
            if isinstance(item, dict)
        ]
        interventions = [
            {"action": "evolutionary.backbone_variant", "weight": 0.35, "payload": {}},
            {"action": "evolutionary.parameter_tuning", "weight": 0.20, "payload": {}},
        ]
        rationale = []
        if "aggressive_early_fill_with_fragmentation" in labels:
            interventions.extend(
                [
                    {"action": "corrective.soften_thresholds", "weight": 0.70, "payload": {}},
                    {"action": "corrective.reduce_dominant_term", "weight": 0.55, "payload": {}},
                ]
            )
            rationale.append(
                "Applied corrective.soften_thresholds due to temporal_behavior.resource_opening_rate_early and resource_utilization.fragmentation_index."
            )
        if "brittle_low_margin_tie_behavior" in labels:
            interventions.extend(
                [
                    {"action": "corrective.adjust_tie_breaking_behavior", "weight": 0.75, "payload": {}},
                    {"action": "corrective.preserve_motif_retune_local_component", "weight": 0.45, "payload": {}},
                ]
            )
            rationale.append(
                "Applied corrective.adjust_tie_breaking_behavior due to decision_pattern.choice_entropy and decision_pattern.score_margin_mean."
            )
        if "complexity_barrier_with_low_margin_decisions" in labels:
            interventions.append({"action": "corrective.simplify_logic", "weight": 0.80, "payload": {}})
            rationale.append(
                "Applied corrective.simplify_logic due to structure.simplicity_index and search.stagnation_length."
            )
        if len(rationale) == 0:
            rationale.append(
                "Evidence remained usable, so portfolio was synthesized from diagnosis instead of generic fallback."
            )
        missing_required = self._missing_required_metrics(evidence)
        return {
            "portfolio_id": f"port_diagnosis_driven_g{generation_index}",
            "based_on_diagnosis": diagnosis.get("diagnosis_id", "unknown"),
            "generation_objective": "apply targeted corrective and evolutionary interventions",
            "interventions": interventions,
            "budget_allocation": {"exploration": 0.35, "exploitation": 0.40, "evaluation": 0.25},
            "branch_policy": {"exploration_branch": 0.40, "exploitation_branch": 0.60},
            "success_criteria": ["positive delta_best", "improved behavior metrics", "stable invalid_rate"],
            "rationale": rationale,
            "fallback_used": False,
            "fallback_reason": "",
            "missing_required_metrics": missing_required,
            "source": "diagnosis_driven_repair",
        }

    def _fallback_portfolio(self, diagnosis: Dict[str, Any], generation_index: int, missing_required_metrics: Optional[List[str]] = None, reason: str = "partial_behavior_evidence") -> Dict[str, Any]:
        missing_required_metrics = list(missing_required_metrics or [])
        return {
            "portfolio_id": f"port_fallback_g{generation_index}",
            "based_on_diagnosis": diagnosis.get("diagnosis_id", "unknown"),
            "generation_objective": "recover progress while preserving validity",
            "interventions": [
                {"action": "evolutionary.backbone_variant", "weight": 0.55, "payload": {}},
                {"action": "evolutionary.parameter_tuning", "weight": 0.25, "payload": {}},
                {"action": "evolutionary.structural_modification", "weight": 0.12, "payload": {}},
                {"action": "evolutionary.simplification", "weight": 0.08, "payload": {}},
                {"action": "corrective.simplify_logic", "weight": 0.5, "payload": {}},
                {"action": "corrective.soften_thresholds", "weight": 0.4, "payload": {}},
                {"action": "search_structure.refresh_diversity_pool", "weight": 0.3, "payload": {}},
            ],
            "budget_allocation": {"exploration": 0.45, "exploitation": 0.40, "evaluation": 0.15},
            "branch_policy": {"exploration_branch": 0.45, "exploitation_branch": 0.55},
            "success_criteria": ["positive delta_best", "stable invalid_rate"],
            "rationale": ["Fallback intervention portfolio under partial or invalid evidence."],
            "fallback_used": True,
            "fallback_reason": reason,
            "missing_required_metrics": missing_required_metrics,
            "source": "fallback",
        }

    def _fallback_reflection(self, diagnosis: Dict[str, Any], portfolio: Dict[str, Any], generation_index: int, outcome_summary: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "reflection_id": f"refl_fallback_g{generation_index}",
            "based_on_portfolio": portfolio.get("portfolio_id", "unknown"),
            "outcome_summary": outcome_summary,
            "supported_hypotheses": [],
            "rejected_hypotheses": [],
            "observed_effects": ["fallback_reflection_generated"],
            "lessons": ["Need richer outcome evidence for stronger causal conclusions."],
            "memory_updates": [
                {
                    "kind": "memory.update_diagnosis_action_prior",
                    "payload": {
                        "diagnosis_id": diagnosis.get("diagnosis_id", "unknown"),
                        "portfolio_id": portfolio.get("portfolio_id", "unknown"),
                        "delta_best": outcome_summary.get("delta_best"),
                    },
                }
            ],
        }

    def run(
        self,
        observation: Dict[str, Any],
        population: Optional[List[Dict[str, Any]]] = None,
        interface_prob: Any = None,
        generation_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        gen = int(generation_index + 1) if generation_index is not None else int(observation.get("gen", 0) or 0)
        population = population or []
        profiles = build_heuristic_profiles(population, generation_index=gen, source_action="unknown")
        profile_summary = [
            {
                "heuristic_id": p.heuristic_id,
                "fitness": p.scalar_fitness,
                "simplicity": p.complexity_metrics.get("structure.simplicity_index"),
                "code_length": p.complexity_metrics.get("structure.code_length"),
                "choice_entropy": p.behavior_trace_summary.get("decision_pattern.choice_entropy"),
                "early_opening_rate": p.behavior_trace_summary.get("temporal_behavior.resource_opening_rate_early"),
            }
            for p in profiles[:8]
        ]

        measurement_prompt = MEASUREMENT_PLANNER_PROMPT.format(
            PROBLEM_CONTEXT=json.dumps({"problem": "bp_online", "mode": "agentic_full"}, ensure_ascii=True),
            OBS_JSON=json.dumps(observation, ensure_ascii=True),
            PROFILE_SUMMARY_JSON=json.dumps(profile_summary, ensure_ascii=True),
        )
        measurement_raw, measurement_llm = self.call_llm_json(
            prompt=measurement_prompt,
            schema_name="measurement_plan",
            validator=self._validate_measurement_plan,
            corrective_schema_hint='{"plan_id":"...","requested_metrics":["metric.name", "..."]}',
            mode="json",
            tool_name="measurement_planner",
        )
        if not isinstance(measurement_raw, dict):
            measurement_raw = self._fallback_measurement_plan(observation, gen)
        measurement_plan = MeasurementPlan(**measurement_raw)

        evidence_obj: BehaviorEvidenceReport = self.observer.build(
            measurement_plan=measurement_plan,
            observation=observation,
            profiles=[p.to_dict() for p in profiles],
            generation=gen,
        )

        analyst_prompt = ANALYST_PROMPT.format(
            OBS_JSON=json.dumps(observation, ensure_ascii=True),
            PLAN_JSON=json.dumps(measurement_plan.to_dict(), ensure_ascii=True),
            EVIDENCE_JSON=json.dumps(evidence_obj.to_dict(), ensure_ascii=True),
        )
        diagnosis_raw, diagnosis_llm = self.call_llm_json(
            prompt=analyst_prompt,
            schema_name="diagnosis_report",
            validator=self._validate_analyst,
            corrective_schema_hint='{"diagnosis_id":"...","diagnoses":[...],"hypotheses":[...]}',
            mode="json",
            tool_name="analyst",
        )
        if not isinstance(diagnosis_raw, dict):
            diagnosis_raw = self._fallback_diagnosis(evidence_obj.to_dict(), measurement_plan.to_dict(), gen)
        diagnosis_report = DiagnosisReport(**diagnosis_raw)

        strategist_prompt = STRATEGIST_PROMPT.format(
            DIAG_JSON=json.dumps(diagnosis_report.to_dict(), ensure_ascii=True),
            EVIDENCE_JSON=json.dumps(evidence_obj.to_dict(), ensure_ascii=True),
            OBS_JSON=json.dumps(observation, ensure_ascii=True),
        )
        portfolio_raw, strategist_llm = self.call_llm_json(
            prompt=strategist_prompt,
            schema_name="intervention_portfolio",
            validator=self._validate_strategist,
            corrective_schema_hint='{"portfolio_id":"...","interventions":[{"action":"evolutionary.backbone_variant","weight":0.5,"payload":{}}]}',
            mode="json",
            tool_name="strategist",
        )
        evidence_dict = evidence_obj.to_dict()
        missing_required_metrics = self._missing_required_metrics(evidence_dict)
        if not isinstance(portfolio_raw, dict):
            if self._evidence_is_sufficient(evidence_dict):
                portfolio_raw = self._diagnosis_driven_portfolio(diagnosis_report.to_dict(), evidence_dict, gen)
            else:
                portfolio_raw = self._fallback_portfolio(
                    diagnosis_report.to_dict(),
                    gen,
                    missing_required_metrics=missing_required_metrics,
                    reason="missing_or_invalid_behavior_evidence",
                )
        portfolio = InterventionPortfolio(**portfolio_raw)

        execution_plan = self.executor.apply(portfolio.to_dict(), observation)

        # Compatibility payload for existing routed execution path.
        diagnosis_compat = {
            "summary": diagnosis_report.search_state_summary,
            "factors": {
                "exploration_need": 0.5,
                "exploitation_need": 0.5,
                "diversity_need": 0.5,
                "invalid_risk": _to_float(observation.get("invalid_rate_k"), 0.0),
                "overfit_risk": 0.3,
                "confidence": 0.6,
            },
            "diagnosis_labels": [d.get("label", "DIAGNOSIS") for d in diagnosis_report.diagnoses[:4]] or ["DIAGNOSIS"],
            "evidence": [h.get("hypothesis", "") for h in diagnosis_report.hypotheses[:4]] or ["evidence_pending"],
        }
        planner_compat = {
            "diagnosis_used": diagnosis_report.search_state_summary,
            "op_probs": execution_plan["op_probs"],
            "parent_mix": execution_plan["parent_mix"],
            "prompt_modifiers": execution_plan["prompt_modifiers"],
            "evaluation_plan": execution_plan["evaluation_plan"],
            "rationale": portfolio.rationale[:6] if isinstance(portfolio.rationale, list) else [],
        }
        critic_compat = {
            "verdict": "skipped",
            "reasons": ["agentic_full_executor_portfolio_applied"],
            "final_plan": {
                "op_probs": execution_plan["op_probs"],
                "parent_mix": execution_plan["parent_mix"],
                "prompt_modifiers": execution_plan["prompt_modifiers"],
                "evaluation_plan": execution_plan["evaluation_plan"],
            },
        }

        # Pending context for post-generation reflection.
        self._pending[gen] = {
            "diagnosis_report": diagnosis_report.to_dict(),
            "intervention_portfolio": portfolio.to_dict(),
            "measurement_plan": measurement_plan.to_dict(),
            "evidence": evidence_obj.to_dict(),
            "execution_plan": execution_plan,
        }

        return {
            "time": self._now(),
            "observation": observation,
            "diagnosis": diagnosis_compat,
            "planner_output": planner_compat,
            "critic_output": critic_compat,
            "raw": {
                "diagnosis_text": diagnosis_llm["attempts"][-1]["raw_output"] if diagnosis_llm.get("attempts") else None,
                "planner_text": strategist_llm["attempts"][-1]["raw_output"] if strategist_llm.get("attempts") else None,
                "critic_text": None,
                "measurement_text": measurement_llm["attempts"][-1]["raw_output"] if measurement_llm.get("attempts") else None,
            },
            "artifacts": {
                "measurement_plan": measurement_plan.to_dict(),
                "behavior_evidence": evidence_obj.to_dict(),
                "diagnosis_report": diagnosis_report.to_dict(),
                "intervention_portfolio": portfolio.to_dict(),
                "heuristic_profiles": [p.to_dict() for p in profiles],
                "execution_plan": execution_plan,
                "memory_updates": [],
            },
            "debug": {
                "diagnoser": {
                    "llm": diagnosis_llm,
                    "sanitize": {"fallback_used": False, "patched": False, "patches": []},
                    "flags": {"pure_llm_output": diagnosis_llm.get("success", False), "llm_output_patched": False, "fully_fallback": not diagnosis_llm.get("success", False), "skipped": False},
                },
                "planner": {
                    "llm": strategist_llm,
                    "sanitize": {"fallback_used": False, "patched": False, "patches": []},
                    "flags": {"pure_llm_output": strategist_llm.get("success", False), "llm_output_patched": False, "fully_fallback": not strategist_llm.get("success", False), "skipped": False},
                },
                "critic": {
                    "llm": {"success": False, "llm_success": False, "llm_mode": "disabled", "attempts": [], "retries_used": 0, "repair_used": False, "parse_ok": False, "validation_ok": False, "failure_reason": "critic_disabled", "last_errors": []},
                    "sanitize": {"fallback_used": False, "patched": False, "patches": []},
                    "flags": {"pure_llm_output": False, "llm_output_patched": False, "fully_fallback": False, "skipped": True},
                    "guardrail_meta": {"skipped": True},
                },
                "measurement_planner": {"llm": measurement_llm},
                "analyst": {"llm": diagnosis_llm},
                "strategist": {
                    "llm": strategist_llm,
                    "fallback_used": bool(portfolio.fallback_used),
                    "fallback_reason": str(portfolio.fallback_reason),
                    "missing_required_metrics": list(portfolio.missing_required_metrics),
                    "source": str(portfolio.source),
                },
            },
        }

    def reflect_generation(self, generation: int, outcome_summary: Dict[str, Any]) -> Dict[str, Any]:
        pending = self._pending.pop(int(generation), None)
        if pending is None:
            return {}

        reflection_prompt = REFLECTION_PROMPT.format(
            DIAG_JSON=json.dumps(pending["diagnosis_report"], ensure_ascii=True),
            PORT_JSON=json.dumps(pending["intervention_portfolio"], ensure_ascii=True),
            OUTCOME_JSON=json.dumps(outcome_summary, ensure_ascii=True),
        )
        reflection_raw, reflection_llm = self.call_llm_json(
            prompt=reflection_prompt,
            schema_name="reflection_report",
            validator=self._validate_reflection,
            corrective_schema_hint='{"reflection_id":"...","memory_updates":[...]}',
            mode="json",
            tool_name="reflection_agent",
        )
        if not isinstance(reflection_raw, dict):
            reflection_raw = self._fallback_reflection(
                diagnosis=pending["diagnosis_report"],
                portfolio=pending["intervention_portfolio"],
                generation_index=int(generation),
                outcome_summary=outcome_summary,
            )

        reflection = ReflectionReport(**reflection_raw)
        updates = reflection.memory_updates if isinstance(reflection.memory_updates, list) else []
        self.memory.append_many(updates)
        return {
            "reflection_report": reflection.to_dict(),
            "reflection_llm": reflection_llm,
            "memory_updates": updates,
        }
