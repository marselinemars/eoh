from typing import Any, Dict, List

from .models import BehaviorEvidenceReport, MeasurementPlan


REQUIRED_BEHAVIOR_METRICS = [
    "resource_utilization.mean_residual_ratio",
    "resource_utilization.residual_variance",
    "resource_utilization.fragmentation_index",
    "temporal_behavior.resource_opening_rate_early",
    "temporal_behavior.resource_opening_rate_mid",
    "temporal_behavior.resource_opening_rate_late",
    "temporal_behavior.phase_shift_index",
    "decision_pattern.extreme_option_preference",
    "decision_pattern.choice_entropy",
    "decision_pattern.score_margin_mean",
    "decision_pattern.score_margin_variance",
    "robustness.order_sensitivity",
    "robustness.instance_family_variance",
    "robustness.holdout_gap",
    "structure.simplicity_index",
    "structure.parameter_count",
    "structure.condition_count",
]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


class EvidenceBuilder:
    """Deterministic evidence extractor with behavior-aware comparisons."""

    def __init__(self):
        pass

    def _profile_metric(self, profile: Dict[str, Any], metric_name: str) -> Any:
        if metric_name.startswith("structure."):
            metrics = profile.get("complexity_metrics", {}) if isinstance(profile.get("complexity_metrics"), dict) else {}
            return metrics.get(metric_name)
        metrics = profile.get("behavior_trace_summary", {}) if isinstance(profile.get("behavior_trace_summary"), dict) else {}
        return metrics.get(metric_name)

    def _profile_metric_reason(self, profile: Dict[str, Any], metric_name: str) -> str:
        extra = profile.get("extra", {}) if isinstance(profile.get("extra"), dict) else {}
        reasons = extra.get("metric_reasons", {}) if isinstance(extra.get("metric_reasons"), dict) else {}
        return str(reasons.get(metric_name, ""))

    def _collect_profile_values(self, profiles: List[Dict[str, Any]], metric_name: str) -> List[float]:
        values = []
        for profile in profiles:
            raw = self._profile_metric(profile, metric_name)
            if raw is None:
                continue
            values.append(_safe_float(raw, 0.0))
        return values

    def _metric_from_observation(self, metric_name: str, observation: Dict[str, Any]) -> Dict[str, Any]:
        op_stats = observation.get("op_stats_k", {}) if isinstance(observation.get("op_stats_k"), dict) else {}
        mapping = {
            "search.best_fitness": observation.get("best_fitness"),
            "search.best_improvement_rate": observation.get("improve_rate_k"),
            "search.stagnation_length": observation.get("stagnation_len"),
            "search.invalid_rate": observation.get("invalid_rate_k"),
            "search.population_fitness_diversity": observation.get("diversity_score"),
            "search.behavioral_diversity": observation.get("diversity_score"),
        }
        if metric_name == "search.operator_success_rate":
            return {
                "value": {k: _safe_float(v.get("success_rate"), 0.0) for k, v in op_stats.items() if isinstance(v, dict)},
                "source": "observation.op_stats_k",
                "status": "ok",
                "reason": "",
            }
        if metric_name == "search.operator_mean_delta":
            return {
                "value": {k: _safe_float(v.get("mean_delta"), 0.0) for k, v in op_stats.items() if isinstance(v, dict)},
                "source": "observation.op_stats_k",
                "status": "ok",
                "reason": "",
            }
        if metric_name in mapping:
            raw = mapping[metric_name]
            if raw is None:
                return {"value": None, "source": "observation", "status": "missing", "reason": "observation_field_unavailable"}
            return {"value": _safe_float(raw, 0.0), "source": "observation", "status": "ok", "reason": ""}
        return {"value": None, "source": "observation", "status": "unsupported", "reason": "metric_not_in_observation"}

    def _build_metric_entry(self, metric_name: str, observation: Dict[str, Any], profiles: List[Dict[str, Any]]) -> Dict[str, Any]:
        obs_entry = self._metric_from_observation(metric_name, observation)
        if obs_entry.get("value") is not None:
            return obs_entry

        if len(profiles) == 0:
            return {
                "value": None,
                "source": "profiles",
                "status": "missing",
                "reason": "population_profiles_empty",
            }

        best_profile = profiles[0]
        values = self._collect_profile_values(profiles, metric_name)
        top_k_values = self._collect_profile_values(profiles[: min(3, len(profiles))], metric_name)
        best_raw = self._profile_metric(best_profile, metric_name)
        reason = self._profile_metric_reason(best_profile, metric_name)

        if best_raw is None and len(values) == 0:
            if metric_name in REQUIRED_BEHAVIOR_METRICS:
                return {
                    "value": None,
                    "source": "profiles",
                    "status": "missing",
                    "reason": reason or "metric_missing_from_behavior_trace_summary",
                }
            return {
                "value": None,
                "source": "profiles",
                "status": "unsupported",
                "reason": reason or "metric_not_supported_by_profiles",
            }

        entry = {
            "value": _safe_float(best_raw, 0.0) if best_raw is not None else (_safe_float(values[0], 0.0) if len(values) > 0 else None),
            "source": "profiles.best",
            "status": "ok" if best_raw is not None else "partial",
            "reason": reason,
        }
        if len(values) > 0:
            entry["population_mean"] = float(sum(values) / len(values))
        if len(top_k_values) > 0:
            entry["top_k_mean"] = float(sum(top_k_values) / len(top_k_values))
        if best_raw is not None:
            entry["best"] = _safe_float(best_raw, 0.0)
        return entry

    def _interpret_metric(self, metric_name: str, entry: Dict[str, Any]) -> str:
        value = entry.get("value")
        if value is None:
            reason = entry.get("reason", "metric_unavailable")
            return f"metric unavailable: {reason}"
        value = _safe_float(value, 0.0)
        if metric_name == "temporal_behavior.resource_opening_rate_early":
            return "high early bin opening" if value >= 0.45 else ("moderate early bin opening" if value >= 0.20 else "low early bin opening")
        if metric_name == "temporal_behavior.resource_opening_rate_mid":
            return "high mid-stage opening" if value >= 0.45 else ("moderate mid-stage opening" if value >= 0.20 else "low mid-stage opening")
        if metric_name == "temporal_behavior.resource_opening_rate_late":
            return "high late bin opening" if value >= 0.30 else ("moderate late bin opening" if value >= 0.10 else "low late bin opening")
        if metric_name == "temporal_behavior.phase_shift_index":
            return "strong phase shift" if value >= 0.20 else ("moderate phase shift" if value >= 0.08 else "stable phase behavior")
        if metric_name == "decision_pattern.choice_entropy":
            return "high placement entropy" if value >= 0.60 else ("moderate placement entropy" if value >= 0.30 else "low placement entropy")
        if metric_name == "decision_pattern.extreme_option_preference":
            return "strong extreme-option preference" if value >= 0.55 else ("moderate extreme-option preference" if value >= 0.30 else "weak extreme-option preference")
        if metric_name == "decision_pattern.score_margin_mean":
            return "large score margins" if value >= 0.15 else ("moderate score margins" if value >= 0.05 else "small score margins / brittle ranking")
        if metric_name == "resource_utilization.fragmentation_index":
            return "high residual fragmentation" if value >= 0.55 else ("moderate residual fragmentation" if value >= 0.30 else "low residual fragmentation")
        if metric_name == "resource_utilization.residual_variance":
            return "high residual variance" if value >= 0.04 else ("moderate residual variance" if value >= 0.01 else "low residual variance")
        if metric_name == "robustness.order_sensitivity":
            return "high order sensitivity" if value >= 0.15 else ("moderate order sensitivity" if value >= 0.05 else "low order sensitivity")
        if metric_name == "robustness.holdout_gap":
            return "large holdout gap" if value >= 0.01 else ("small holdout gap" if value >= 0.002 else "negligible holdout gap")
        if metric_name == "structure.simplicity_index":
            return "high structural simplicity" if value >= 0.65 else ("moderate structural simplicity" if value >= 0.40 else "low structural simplicity")
        if metric_name == "structure.parameter_count":
            return "many numeric parameters" if value >= 12 else ("moderate numeric parameter count" if value >= 5 else "few numeric parameters")
        if metric_name == "structure.condition_count":
            return "many branch conditions" if value >= 8 else ("moderate branch count" if value >= 4 else "few branch conditions")
        return f"observed {metric_name}={value:.4f}"

    def _build_comparative_views(self, requested_metrics: List[str], profiles: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(profiles) == 0:
            return {
                "current_best_vs_population_mean": {},
                "top_k_vs_population_mean": {},
                "instance_family_groups": {},
            }
        best_profile = profiles[0]
        current_best = {}
        top_k = {}
        for metric_name in requested_metrics:
            values = self._collect_profile_values(profiles, metric_name)
            top_values = self._collect_profile_values(profiles[: min(3, len(profiles))], metric_name)
            best_raw = self._profile_metric(best_profile, metric_name)
            if best_raw is None or len(values) == 0:
                continue
            current_best[metric_name] = float(_safe_float(best_raw, 0.0) - (sum(values) / len(values)))
            if len(top_values) > 0:
                top_k[metric_name] = float((sum(top_values) / len(top_values)) - (sum(values) / len(values)))

        family_groups = {}
        for profile in profiles:
            family_perf = profile.get("instance_family_performance", {}) if isinstance(profile.get("instance_family_performance"), dict) else {}
            for family_name, score in family_perf.items():
                family_groups.setdefault(family_name, []).append(_safe_float(score, 0.0))
        family_groups = {
            family_name: {
                "mean_fitness": float(sum(values) / len(values)),
                "count": int(len(values)),
            }
            for family_name, values in family_groups.items()
            if len(values) > 0
        }

        return {
            "current_best_vs_population_mean": current_best,
            "top_k_vs_population_mean": top_k,
            "instance_family_groups": family_groups,
        }

    def _build_evidence_quality(self, metric_values: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        missing_required = []
        missing_reasons = {}
        available_required = []
        for metric_name in REQUIRED_BEHAVIOR_METRICS:
            entry = metric_values.get(metric_name, {})
            if entry.get("value") is None:
                missing_required.append(metric_name)
                missing_reasons[metric_name] = str(entry.get("reason", "missing"))
            else:
                available_required.append(metric_name)
        if len(available_required) >= 14:
            level = "strong"
        elif len(available_required) >= 8:
            level = "partial"
        else:
            level = "weak"
        return {
            "level": level,
            "required_metrics_available": available_required,
            "missing_required_metrics": missing_required,
            "missing_reasons": missing_reasons,
            "available_required_count": int(len(available_required)),
            "required_metric_count": int(len(REQUIRED_BEHAVIOR_METRICS)),
        }

    def build(
        self,
        measurement_plan: MeasurementPlan,
        observation: Dict[str, Any],
        profiles: List[Dict[str, Any]],
        generation: int,
    ) -> BehaviorEvidenceReport:
        metric_values: Dict[str, Dict[str, Any]] = {}
        metric_interpretations: Dict[str, str] = {}
        for metric_name in measurement_plan.requested_metrics:
            entry = self._build_metric_entry(metric_name, observation, profiles)
            metric_values[metric_name] = entry
            metric_interpretations[metric_name] = self._interpret_metric(metric_name, entry)

        comparative_views = self._build_comparative_views(measurement_plan.requested_metrics, profiles)
        evidence_quality = self._build_evidence_quality(metric_values)

        target_summaries = []
        for target in measurement_plan.targets:
            target_summaries.append(
                {
                    "target_id": target.get("target_id", "population"),
                    "target_type": target.get("target_type", "population"),
                    "notes": target.get("notes", ""),
                }
            )

        notes = []
        if observation.get("notes"):
            notes.append(str(observation.get("notes")))
        if len(evidence_quality["missing_required_metrics"]) > 0:
            notes.append(
                "missing_required_metrics=" + ",".join(evidence_quality["missing_required_metrics"])
            )

        return BehaviorEvidenceReport(
            report_id=f"evidence_g{generation}_{measurement_plan.plan_id}",
            based_on_measurement_plan=measurement_plan.plan_id,
            generation=int(generation),
            metric_values=metric_values,
            metric_interpretations=metric_interpretations,
            comparative_views=comparative_views,
            evidence_quality=evidence_quality,
            target_summaries=target_summaries,
            notes=notes,
        )
