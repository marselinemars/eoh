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

Search regime:
{SEARCH_REGIME_JSON}

Relevant memory retrievals:
{MEMORY_JSON}

Available parent candidate summaries:
{PARENT_CANDIDATES_JSON}

Available action families:
- evolutionary.*
- corrective.*
- evaluation.*
- search_structure.*
- memory.*

Action ontology summary:
{ACTION_ONTOLOGY_JSON}

Return ONLY JSON:
{{
  "portfolio_id": "string",
  "based_on_diagnosis": "string",
  "search_regime": "string",
  "generation_objective": "string",
  "interventions": [{{"action":"family.name","weight":0.0,"payload":{{}},"intended_behavioral_target":"string","expected_behavioral_change":"string","expected_structural_change":"string","expected_fitness_effect":"string"}}],
  "branches": [{{"branch_id":"string","objective":"string","budget_share":0.0,"parent_candidate_groups":["string"],"interventions":[{{"action":"family.name","weight":0.0,"payload":{{}},"intended_behavioral_target":"string","expected_behavioral_change":"string","expected_structural_change":"string","expected_fitness_effect":"string"}}],"success_criteria":["string"],"rationale":["string"]}}],
  "budget_allocation": {{"exploration":0.0,"exploitation":0.0,"evaluation":0.0}},
  "branch_policy": {{}},
  "parent_candidate_groups_used": ["string"],
  "success_criteria": ["string"],
  "rationale": ["string"],
  "fallback_used": false,
  "fallback_reason": "",
  "missing_required_metrics": ["metric.name"],
  "source": "llm"
}}
Rules:
- interventions non-empty and use ontology action names.
- search_regime must match the provided regime.
- weights in [0,1]
- budget_allocation values in [0,1]
- If evidence_quality.level is strong or partial, do not produce a generic fallback portfolio.
- Use the full action ontology; combine evolutionary, corrective, evaluation, search_structure, and memory actions when justified.
- Use memory retrievals as soft priors, not commands.
- If multiple plausible improvement directions exist, you may create 2-3 branches with different objectives or parent groups. Do not branch unless the evidence justifies it.
- Every intervention must explain what behavior it is trying to change and what structural effect it expects.
- Parent candidate groups used must be chosen from the provided candidate summaries.
- Rationale must cite metric names from BehaviorEvidenceReport.
"""


REFLECTION_PROMPT = """OUTPUT JSON ONLY. No markdown.
You are Reflection Agent.
DiagnosisReport:
{DIAG_JSON}

InterventionPortfolio:
{PORT_JSON}

Prior behavior evidence:
{EVIDENCE_JSON}

Outcome summary:
{OUTCOME_JSON}

Return ONLY JSON:
{{
  "reflection_id": "string",
  "based_on_portfolio": "string",
  "outcome_summary": {{}},
  "behavioral_outcomes": [{{"target":"string","observed_change":"string","supported":true,"metrics":["string"]}}],
  "supported_hypotheses": ["string"],
  "rejected_hypotheses": ["string"],
  "observed_effects": ["string"],
  "tradeoffs": ["string"],
  "lessons": ["string"],
  "memory_updates": [{{"kind":"string","payload":{{}}}}]
}}

Rules:
- Judge outcomes using both fitness changes and behavior changes.
- If an intervention changed the intended behavior but did not yet improve best fitness, record that explicitly.
- Cite metric names in behavioral_outcomes or observed_effects whenever possible.
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
        if str(payload.get("search_regime", "")).strip() == "":
            errors.append("missing_search_regime_value")
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
                for required_text in [
                    "intended_behavioral_target",
                    "expected_behavioral_change",
                    "expected_structural_change",
                    "expected_fitness_effect",
                ]:
                    if str(item.get(required_text, "")).strip() == "":
                        errors.append(f"missing_{required_text}")
        branches = payload.get("branches", [])
        if isinstance(branches, list):
            for branch in branches:
                if not isinstance(branch, dict):
                    errors.append("branch_item_not_dict")
                    continue
                if str(branch.get("branch_id", "")).strip() == "":
                    errors.append("branch_missing_id")
                budget_share = _to_float(branch.get("budget_share"), None)
                if budget_share is None or budget_share < 0.0 or budget_share > 1.0:
                    errors.append("invalid_branch_budget_share")
                if not isinstance(branch.get("interventions"), list) or len(branch.get("interventions")) == 0:
                    errors.append("branch_missing_interventions")
                if not isinstance(branch.get("rationale"), list) or len(branch.get("rationale")) == 0:
                    errors.append("branch_missing_rationale")
        rationale = payload.get("rationale", [])
        if not isinstance(rationale, list) or len(rationale) < 2:
            errors.append("rationale_too_short")
        else:
            if not any(any(metric_name in str(line) for metric_name in REQUIRED_BEHAVIOR_METRICS) for line in rationale):
                errors.append("rationale_missing_behavior_metric_citation")
        if not isinstance(payload.get("parent_candidate_groups_used"), list):
            errors.append("parent_candidate_groups_used_not_list")
        return errors

    def _validate_reflection(self, payload: Dict[str, Any]) -> List[str]:
        errors = validate_reflection_report(payload)
        outcomes = payload.get("behavioral_outcomes", [])
        if isinstance(outcomes, list):
            for item in outcomes:
                if not isinstance(item, dict):
                    errors.append("behavioral_outcome_not_dict")
                    continue
                if str(item.get("target", "")).strip() == "":
                    errors.append("behavioral_outcome_missing_target")
                if not isinstance(item.get("metrics", []), list):
                    errors.append("behavioral_outcome_metrics_not_list")
        return errors

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

    def _metric_value(self, evidence: Dict[str, Any], metric_name: str, default: Optional[float] = None) -> Optional[float]:
        if not isinstance(evidence, dict):
            return default
        metric_values = evidence.get("metric_values", {}) if isinstance(evidence.get("metric_values"), dict) else {}
        entry = metric_values.get(metric_name, {})
        if not isinstance(entry, dict):
            return default
        value = _to_float(entry.get("value"), None)
        if value is None:
            return default
        return float(value)

    def _compute_search_regime(self, observation: Dict[str, Any], evidence: Dict[str, Any]) -> Dict[str, Any]:
        gen = int(observation.get("gen", 0) or 0)
        best_history = observation.get("best_history", []) if isinstance(observation.get("best_history"), list) else []
        current_best = _to_float(observation.get("best_fitness"), None)
        initial_best = _to_float(best_history[0], current_best) if len(best_history) > 0 else current_best
        stagnation = int(observation.get("stagnation_len", 0) or 0)
        improve_rate = _to_float(observation.get("improve_rate_k"), 0.0)
        holdout_gap = self._metric_value(evidence, "robustness.holdout_gap", 0.0) or 0.0
        order_sensitivity = self._metric_value(evidence, "robustness.order_sensitivity", 0.0) or 0.0
        relative_gain = 0.0
        if current_best is not None and initial_best is not None:
            denom = max(abs(initial_best), 1e-8)
            relative_gain = max(0.0, float(initial_best - current_best) / denom)
        breakthrough = bool(relative_gain >= 0.05 or (gen >= 2 and improve_rate >= 0.34 and stagnation <= 1))
        if breakthrough and (holdout_gap >= 0.05 or order_sensitivity >= 0.08):
            regime = "robustness_refinement"
            rationale = "Breakthrough exists but robustness metrics show fragility."
        elif breakthrough and stagnation >= 3:
            regime = "stagnation_recovery"
            rationale = "Breakthrough exists but recent progress has stalled."
        elif breakthrough:
            regime = "post_breakthrough_refinement"
            rationale = "A meaningful gain exists; preserve the motif and refine its remaining weaknesses."
        else:
            regime = "pre_breakthrough"
            rationale = "No clear breakthrough yet; prioritize broad motif discovery and exploratory search."
        return {
            "name": regime,
            "breakthrough_detected": breakthrough,
            "relative_gain": float(relative_gain),
            "stagnation_len": int(stagnation),
            "improve_rate_k": float(improve_rate),
            "rationale": rationale,
        }

    def _build_parent_candidate_summaries(self, profiles) -> List[Dict[str, Any]]:
        if not isinstance(profiles, list) or len(profiles) == 0:
            return []
        sorted_profiles = sorted(
            profiles,
            key=lambda p: float(p.scalar_fitness if p.scalar_fitness is not None else float("inf")),
        )
        best = sorted_profiles[0]
        best_behavior = best.behavior_trace_summary if isinstance(best.behavior_trace_summary, dict) else {}

        def _metric(profile, name, default=0.0):
            source = profile.behavior_trace_summary if isinstance(profile.behavior_trace_summary, dict) else {}
            if name.startswith("structure."):
                source = profile.complexity_metrics if isinstance(profile.complexity_metrics, dict) else {}
            return _to_float(source.get(name), default)

        def _distance_from_best(profile) -> float:
            names = [
                "resource_utilization.fragmentation_index",
                "temporal_behavior.resource_opening_rate_early",
                "decision_pattern.choice_entropy",
                "decision_pattern.score_margin_mean",
                "robustness.order_sensitivity",
            ]
            dist = 0.0
            for name in names:
                dist += abs(_to_float(profile.behavior_trace_summary.get(name), 0.0) - _to_float(best_behavior.get(name), 0.0))
            return float(dist)

        candidates = []
        seen = set()

        def _add_candidate(group_name: str, profile, reason: str):
            if profile is None:
                return
            code_hash = str(profile.code_hash)
            if not code_hash or group_name in seen:
                return
            seen.add(group_name)
            candidates.append(
                {
                    "group_name": group_name,
                    "heuristic_id": profile.heuristic_id,
                    "code_hash": code_hash,
                    "fitness": profile.scalar_fitness,
                    "reason": reason,
                    "summary": profile.summary,
                    "behavior_snippet": {
                        "fragmentation": _metric(profile, "resource_utilization.fragmentation_index"),
                        "early_opening": _metric(profile, "temporal_behavior.resource_opening_rate_early"),
                        "choice_entropy": _metric(profile, "decision_pattern.choice_entropy"),
                        "order_sensitivity": _metric(profile, "robustness.order_sensitivity"),
                        "simplicity": _metric(profile, "structure.simplicity_index"),
                    },
                }
            )

        _add_candidate("current_best", best, "Best current scalar fitness.")
        distinct = max(sorted_profiles[1:] or [best], key=_distance_from_best)
        _add_candidate("behaviorally_distinct_runner_up", distinct, "Behaviorally distinct from current_best.")
        robust = min(sorted_profiles, key=lambda p: (_metric(p, "robustness.order_sensitivity", 1.0) + _metric(p, "robustness.holdout_gap", 1.0), _metric(p, "resource_utilization.fragmentation_index", 1.0)))
        _add_candidate("robust_runner_up", robust, "Lowest order_sensitivity plus holdout_gap.")
        low_frag = min(sorted_profiles, key=lambda p: (_metric(p, "resource_utilization.fragmentation_index", 1.0), p.scalar_fitness if p.scalar_fitness is not None else float("inf")))
        _add_candidate("low_fragmentation_candidate", low_frag, "Low residual fragmentation.")
        low_order = min(sorted_profiles, key=lambda p: (_metric(p, "robustness.order_sensitivity", 1.0), p.scalar_fitness if p.scalar_fitness is not None else float("inf")))
        _add_candidate("low_order_sensitivity_candidate", low_order, "Low order sensitivity.")
        simple = max(sorted_profiles, key=lambda p: (_metric(p, "structure.simplicity_index", 0.0), -_metric(p, "structure.parameter_count", 999.0)))
        _add_candidate("simple_candidate", simple, "High simplicity index.")
        return candidates

    def _memory_query_text(
        self,
        diagnosis: Dict[str, Any],
        evidence: Dict[str, Any],
        search_regime: Dict[str, Any],
    ) -> str:
        labels = []
        for item in diagnosis.get("diagnoses", []) if isinstance(diagnosis.get("diagnoses"), list) else []:
            if isinstance(item, dict):
                labels.append(str(item.get("label", "")))
        focus = diagnosis.get("recommended_focus", []) if isinstance(diagnosis.get("recommended_focus"), list) else []
        evidence_lines = []
        for item in diagnosis.get("diagnoses", []) if isinstance(diagnosis.get("diagnoses"), list) else []:
            if isinstance(item, dict):
                evidence_lines.extend(str(line) for line in item.get("evidence", [])[:2])
        return " ".join(
            [search_regime.get("name", "")]
            + labels[:4]
            + [str(x) for x in focus[:4]]
            + evidence_lines[:6]
            + list((evidence.get("metric_interpretations", {}) or {}).values())[:4]
        )

    def _retrieve_memory_context(
        self,
        diagnosis: Dict[str, Any],
        evidence: Dict[str, Any],
        search_regime: Dict[str, Any],
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        query = self._memory_query_text(diagnosis, evidence, search_regime)
        retrieved = self.memory.retrieve(query, limit=limit)
        out = []
        for item in retrieved:
            out.append(
                {
                    "kind": item.get("kind", ""),
                    "score": item.get("_retrieval_score", 0.0),
                    "payload": item.get("payload", {}) if isinstance(item.get("payload"), dict) else {},
                }
            )
        return out

    def _action_ontology_summary(self) -> Dict[str, List[str]]:
        return {
            family: [spec.name for spec in self.action_registry.by_family(family)]
            for family in ["evolutionary", "corrective", "evaluation", "search_structure", "memory"]
        }

    def _family_soft_priors(self, search_regime: Dict[str, Any], evidence: Dict[str, Any]) -> Dict[str, float]:
        priors = {
            "evolutionary": 1.0,
            "corrective": 1.0,
            "evaluation": 0.8,
            "search_structure": 0.8,
            "memory": 0.4,
        }
        regime_name = str(search_regime.get("name", ""))
        if regime_name == "pre_breakthrough":
            priors["evolutionary"] += 0.30
            priors["search_structure"] += 0.15
        elif regime_name == "post_breakthrough_refinement":
            priors["corrective"] += 0.35
            priors["evaluation"] += 0.15
        elif regime_name == "stagnation_recovery":
            priors["evolutionary"] += 0.20
            priors["search_structure"] += 0.20
            priors["corrective"] += 0.15
        elif regime_name == "robustness_refinement":
            priors["evaluation"] += 0.35
            priors["corrective"] += 0.20
        quality = evidence.get("evidence_quality", {}) if isinstance(evidence.get("evidence_quality"), dict) else {}
        if quality.get("level") == "strong":
            priors["corrective"] += 0.15
        return priors

    def _contextual_resynthesis_portfolio(
        self,
        diagnosis: Dict[str, Any],
        evidence: Dict[str, Any],
        search_regime: Dict[str, Any],
        parent_candidates: List[Dict[str, Any]],
        memory_context: List[Dict[str, Any]],
        generation_index: int,
        missing_required_metrics: Optional[List[str]] = None,
        source: str = "diagnosis_driven_resynthesis",
    ) -> Dict[str, Any]:
        missing_required_metrics = list(missing_required_metrics or [])
        diagnosis_lines = []
        labels = []
        for item in diagnosis.get("diagnoses", []) if isinstance(diagnosis.get("diagnoses"), list) else []:
            if isinstance(item, dict):
                label = str(item.get("label", ""))
                if label:
                    labels.append(label)
                diagnosis_lines.append(label)
                diagnosis_lines.extend(str(line) for line in item.get("evidence", []))
        hypotheses = [str(item.get("hypothesis", "")) for item in diagnosis.get("hypotheses", []) if isinstance(item, dict)]
        goals = [str(x) for x in diagnosis.get("intervention_goals", []) if str(x).strip()]
        interpretations = list((evidence.get("metric_interpretations", {}) or {}).values())
        memory_text = json.dumps(memory_context, ensure_ascii=True)
        context_text = " ".join(diagnosis_lines + hypotheses + goals + interpretations + [memory_text, search_regime.get("name", "")]).lower()
        priors = self._family_soft_priors(search_regime, evidence)

        scored_actions = []
        for action_name in self.action_registry.list():
            spec = self.action_registry.get(action_name)
            if spec is None:
                continue
            score = priors.get(spec.family, 0.5)
            action_text = f"{action_name} {spec.description}".lower()
            for token in context_text.split():
                if token and token in action_text:
                    score += 0.15
            for mem in memory_context:
                payload = mem.get("payload", {}) if isinstance(mem.get("payload"), dict) else {}
                used_actions = payload.get("actions", [])
                if action_name in used_actions:
                    score += 0.20
                if payload.get("fitness_delta", 0.0) and float(payload.get("fitness_delta", 0.0) or 0.0) > 0.0:
                    score += 0.05
            scored_actions.append((float(score), action_name, spec))
        scored_actions.sort(key=lambda item: item[0], reverse=True)

        selected = []
        used_families = set()
        for score, action_name, spec in scored_actions:
            if len(selected) >= 5:
                break
            if spec.family in used_families and score < (selected[0]["weight"] if selected else 0.0):
                continue
            selected.append(
                {
                    "action": action_name,
                    "weight": min(0.95, max(0.15, score / 3.0)),
                    "payload": {},
                    "intended_behavioral_target": goals[0] if len(goals) > 0 else (labels[0] if len(labels) > 0 else search_regime.get("name", "")),
                    "expected_behavioral_change": interpretations[0] if len(interpretations) > 0 else "Shift the diagnosed behavior in a measurable direction.",
                    "expected_structural_change": "Small-to-moderate structural adjustment consistent with selected action family.",
                    "expected_fitness_effect": "Seek improvement or maintain breakthrough fitness while changing the target behavior.",
                }
            )
            used_families.add(spec.family)

        selected_action_names = {item["action"] for item in selected}
        for required_family in ["evolutionary", "corrective"]:
            if any(self.action_registry.get(name).family == required_family for name in selected_action_names if self.action_registry.get(name) is not None):
                continue
            for score, action_name, spec in scored_actions:
                if spec.family != required_family or action_name in selected_action_names:
                    continue
                selected.append(
                    {
                        "action": action_name,
                        "weight": min(0.85, max(0.15, score / 3.5)),
                        "payload": {},
                        "intended_behavioral_target": goals[0] if len(goals) > 0 else (labels[0] if len(labels) > 0 else search_regime.get("name", "")),
                        "expected_behavioral_change": interpretations[0] if len(interpretations) > 0 else "Shift the diagnosed behavior in a measurable direction.",
                        "expected_structural_change": "Small-to-moderate structural adjustment consistent with selected action family.",
                        "expected_fitness_effect": "Seek improvement or maintain breakthrough fitness while changing the target behavior.",
                    }
                )
                selected_action_names.add(action_name)
                break

        branch_candidates = []
        if len(goals) > 1 or len(labels) > 1:
            focus_texts = goals[:2] if len(goals) > 1 else labels[:2]
            for idx, focus in enumerate(focus_texts[:2], start=1):
                branch_actions = []
                picked_groups = []
                for candidate in parent_candidates:
                    if focus.lower().find("robust") >= 0 and "robust" in candidate.get("group_name", ""):
                        picked_groups.append(candidate.get("group_name"))
                    elif focus.lower().find("fragment") >= 0 and "fragmentation" in candidate.get("group_name", ""):
                        picked_groups.append(candidate.get("group_name"))
                if len(picked_groups) == 0 and len(parent_candidates) > 0:
                    picked_groups.append(parent_candidates[min(idx - 1, len(parent_candidates) - 1)].get("group_name"))
                for item in selected[:3]:
                    payload = dict(item.get("payload", {}))
                    payload["parent_candidate_groups"] = picked_groups
                    payload["parent_candidate_hashes"] = [
                        c.get("code_hash")
                        for c in parent_candidates
                        if c.get("group_name") in picked_groups and c.get("code_hash")
                    ]
                    branch_actions.append(
                        {
                            "action": item["action"],
                            "weight": max(0.10, item["weight"] * 0.8),
                            "payload": payload,
                            "intended_behavioral_target": focus,
                            "expected_behavioral_change": item["expected_behavioral_change"],
                            "expected_structural_change": item["expected_structural_change"],
                            "expected_fitness_effect": item["expected_fitness_effect"],
                        }
                    )
                branch_candidates.append(
                    {
                        "branch_id": f"branch_{idx}",
                        "objective": focus,
                        "budget_share": 0.5,
                        "parent_candidate_groups": picked_groups,
                        "parent_candidate_hashes": [
                            c.get("code_hash")
                            for c in parent_candidates
                            if c.get("group_name") in picked_groups and c.get("code_hash")
                        ],
                        "interventions": branch_actions,
                        "success_criteria": [
                            f"Behavior target shifts: {focus}",
                            "No invalid spike",
                        ],
                        "rationale": [
                            f"Branch objective derived from diagnosis/intervention_goals: {focus}",
                        ],
                    }
                )

        parent_groups_used = []
        if len(parent_candidates) > 0:
            parent_groups_used = [c.get("group_name") for c in parent_candidates[:3] if c.get("group_name")]

        rationale = [
            f"Resynthesized from search_regime={search_regime.get('name')} with evidence_quality={((evidence.get('evidence_quality') or {}).get('level', 'unknown'))}.",
            "Portfolio selected by ontology/memory/evidence overlap instead of generic fallback.",
        ]
        for diag in diagnosis.get("diagnoses", [])[:2] if isinstance(diagnosis.get("diagnoses"), list) else []:
            if isinstance(diag, dict) and isinstance(diag.get("evidence"), list) and len(diag["evidence"]) > 0:
                rationale.append(str(diag["evidence"][0]))

        return {
            "portfolio_id": f"port_contextual_g{generation_index}",
            "based_on_diagnosis": diagnosis.get("diagnosis_id", "unknown"),
            "search_regime": search_regime.get("name", "pre_breakthrough"),
            "generation_objective": search_regime.get("rationale", "Refine search using current evidence."),
            "interventions": selected,
            "branches": branch_candidates,
            "budget_allocation": {"exploration": 0.30, "exploitation": 0.45, "evaluation": 0.25},
            "branch_policy": {
                "mode": "parallel" if len(branch_candidates) > 1 else "single_path",
                "branch_count": len(branch_candidates),
            },
            "parent_candidate_groups_used": [x for x in parent_groups_used if x],
            "success_criteria": [
                "Positive fitness delta or improved targeted behavior metrics.",
                "No invalid spike.",
            ],
            "rationale": rationale[:6],
            "fallback_used": False,
            "fallback_reason": "",
            "missing_required_metrics": missing_required_metrics,
            "source": source,
        }

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

    def _diagnosis_driven_portfolio(
        self,
        diagnosis: Dict[str, Any],
        evidence: Dict[str, Any],
        search_regime: Dict[str, Any],
        parent_candidates: List[Dict[str, Any]],
        memory_context: List[Dict[str, Any]],
        generation_index: int,
    ) -> Dict[str, Any]:
        return self._contextual_resynthesis_portfolio(
            diagnosis=diagnosis,
            evidence=evidence,
            search_regime=search_regime,
            parent_candidates=parent_candidates,
            memory_context=memory_context,
            generation_index=generation_index,
            missing_required_metrics=self._missing_required_metrics(evidence),
            source="diagnosis_driven_resynthesis",
        )

    def _fallback_portfolio(self, diagnosis: Dict[str, Any], generation_index: int, missing_required_metrics: Optional[List[str]] = None, reason: str = "partial_behavior_evidence") -> Dict[str, Any]:
        missing_required_metrics = list(missing_required_metrics or [])
        return {
            "portfolio_id": f"port_fallback_g{generation_index}",
            "based_on_diagnosis": diagnosis.get("diagnosis_id", "unknown"),
            "search_regime": "stagnation_recovery",
            "generation_objective": "recover progress while preserving validity",
            "interventions": [
                {
                    "action": "evolutionary.backbone_variant",
                    "weight": 0.55,
                    "payload": {},
                    "intended_behavioral_target": "Recover movement under weak evidence.",
                    "expected_behavioral_change": "Increase variation around current motifs.",
                    "expected_structural_change": "Moderate structural change.",
                    "expected_fitness_effect": "Open a new path to improvement.",
                },
                {
                    "action": "evolutionary.parameter_tuning",
                    "weight": 0.25,
                    "payload": {},
                    "intended_behavioral_target": "Refine score balance safely.",
                    "expected_behavioral_change": "Small local behavior shift.",
                    "expected_structural_change": "Low structural disruption.",
                    "expected_fitness_effect": "Exploit safe local gains.",
                },
                {
                    "action": "corrective.simplify_logic",
                    "weight": 0.40,
                    "payload": {},
                    "intended_behavioral_target": "Reduce brittle complexity.",
                    "expected_behavioral_change": "More stable decisions.",
                    "expected_structural_change": "Simpler logic.",
                    "expected_fitness_effect": "Improve robustness and validity.",
                },
            ],
            "branches": [],
            "budget_allocation": {"exploration": 0.45, "exploitation": 0.40, "evaluation": 0.15},
            "branch_policy": {"mode": "single_path", "branch_count": 0},
            "parent_candidate_groups_used": [],
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
            "behavioral_outcomes": outcome_summary.get("behavioral_outcomes", []) if isinstance(outcome_summary.get("behavioral_outcomes"), list) else [],
            "supported_hypotheses": [],
            "rejected_hypotheses": [],
            "observed_effects": ["fallback_reflection_generated"],
            "tradeoffs": outcome_summary.get("behavioral_tradeoffs", []) if isinstance(outcome_summary.get("behavioral_tradeoffs"), list) else [],
            "lessons": ["Need richer outcome evidence for stronger causal conclusions."],
            "memory_updates": [
                {
                    "kind": "memory.update_diagnosis_action_prior",
                    "payload": {
                        "diagnosis_id": diagnosis.get("diagnosis_id", "unknown"),
                        "portfolio_id": portfolio.get("portfolio_id", "unknown"),
                        "search_regime": portfolio.get("search_regime", ""),
                        "actions": [item.get("action") for item in portfolio.get("interventions", []) if isinstance(item, dict)],
                        "delta_best": outcome_summary.get("delta_best"),
                        "behavioral_outcomes": outcome_summary.get("behavioral_outcomes", []),
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
        evidence_dict = evidence_obj.to_dict()
        search_regime = self._compute_search_regime(observation, evidence_dict)
        parent_candidate_summaries = self._build_parent_candidate_summaries(profiles)

        analyst_prompt = ANALYST_PROMPT.format(
            OBS_JSON=json.dumps(observation, ensure_ascii=True),
            PLAN_JSON=json.dumps(measurement_plan.to_dict(), ensure_ascii=True),
            EVIDENCE_JSON=json.dumps(evidence_dict, ensure_ascii=True),
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
            diagnosis_raw = self._fallback_diagnosis(evidence_dict, measurement_plan.to_dict(), gen)
        diagnosis_report = DiagnosisReport(**diagnosis_raw)
        memory_context = self._retrieve_memory_context(
            diagnosis=diagnosis_report.to_dict(),
            evidence=evidence_dict,
            search_regime=search_regime,
            limit=5,
        )

        strategist_prompt = STRATEGIST_PROMPT.format(
            DIAG_JSON=json.dumps(diagnosis_report.to_dict(), ensure_ascii=True),
            EVIDENCE_JSON=json.dumps(evidence_dict, ensure_ascii=True),
            OBS_JSON=json.dumps(observation, ensure_ascii=True),
            SEARCH_REGIME_JSON=json.dumps(search_regime, ensure_ascii=True),
            MEMORY_JSON=json.dumps(memory_context, ensure_ascii=True),
            PARENT_CANDIDATES_JSON=json.dumps(parent_candidate_summaries, ensure_ascii=True),
            ACTION_ONTOLOGY_JSON=json.dumps(self._action_ontology_summary(), ensure_ascii=True),
        )
        portfolio_raw, strategist_llm = self.call_llm_json(
            prompt=strategist_prompt,
            schema_name="intervention_portfolio",
            validator=self._validate_strategist,
            corrective_schema_hint='{"portfolio_id":"...","interventions":[{"action":"evolutionary.backbone_variant","weight":0.5,"payload":{}}]}',
            mode="json",
            tool_name="strategist",
        )
        missing_required_metrics = self._missing_required_metrics(evidence_dict)
        if not isinstance(portfolio_raw, dict):
            if self._evidence_is_sufficient(evidence_dict):
                portfolio_raw = self._diagnosis_driven_portfolio(
                    diagnosis_report.to_dict(),
                    evidence_dict,
                    search_regime,
                    parent_candidate_summaries,
                    memory_context,
                    gen,
                )
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
            "evidence": evidence_dict,
            "execution_plan": execution_plan,
            "search_regime": search_regime,
            "memory_context": memory_context,
            "parent_candidate_summaries": parent_candidate_summaries,
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
                "search_regime": search_regime,
                "memory_context": memory_context,
                "parent_candidate_summaries": parent_candidate_summaries,
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
                    "sanitize": {
                        "fallback_used": bool(portfolio.fallback_used),
                        "patched": bool((not strategist_llm.get("success", False)) and (not portfolio.fallback_used)),
                        "patches": [str(portfolio.source)] if str(portfolio.source) not in ["", "llm"] else [],
                    },
                    "flags": {
                        "pure_llm_output": strategist_llm.get("success", False),
                        "llm_output_patched": bool((not strategist_llm.get("success", False)) and (not portfolio.fallback_used)),
                        "fully_fallback": bool(portfolio.fallback_used),
                        "skipped": False,
                    },
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
                    "search_regime": search_regime,
                    "parent_candidate_summaries": parent_candidate_summaries,
                    "memory_context": memory_context,
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
            EVIDENCE_JSON=json.dumps(pending["evidence"], ensure_ascii=True),
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
        if len(updates) == 0:
            updates = [
                {
                    "kind": "memory.update_diagnosis_action_prior",
                    "payload": {
                        "diagnosis_id": pending["diagnosis_report"].get("diagnosis_id", "unknown"),
                        "portfolio_id": pending["intervention_portfolio"].get("portfolio_id", "unknown"),
                        "search_regime": pending.get("search_regime", {}).get("name", ""),
                        "actions": [
                            item.get("action")
                            for item in pending["intervention_portfolio"].get("interventions", [])
                            if isinstance(item, dict)
                        ],
                        "behavioral_outcomes": reflection.behavioral_outcomes,
                        "fitness_delta": outcome_summary.get("train_delta"),
                    },
                }
            ]
        self.memory.append_many(updates)
        return {
            "reflection_report": reflection.to_dict(),
            "reflection_llm": reflection_llm,
            "memory_updates": updates,
        }
