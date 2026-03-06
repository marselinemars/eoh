from typing import Any, Dict, List

from .models import BehaviorEvidenceReport, MeasurementPlan


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


class EvidenceBuilder:
    """Procedural evidence extractor.

    This is intentionally deterministic. It can be extended with detailed
    instance-level traces when the evaluator exposes richer hooks.
    """

    def __init__(self):
        pass

    def _from_observation(self, metric_name: str, observation: Dict[str, Any]) -> Dict[str, Any]:
        op_stats = observation.get("op_stats_k", {}) if isinstance(observation.get("op_stats_k"), dict) else {}
        mapping = {
            "search.best_fitness": observation.get("best_fitness"),
            "search.best_improvement_rate": observation.get("improve_rate_k"),
            "search.stagnation_length": observation.get("stagnation_len"),
            "search.invalid_rate": observation.get("invalid_rate_k"),
            "search.population_fitness_diversity": observation.get("diversity_score"),
            "search.behavioral_diversity": observation.get("diversity_score"),
            "robustness.holdout_gap": observation.get("holdout_gap"),
            "temporal_behavior.resource_opening_rate_early": observation.get("resource_opening_rate_early"),
            "temporal_behavior.resource_opening_rate_late": observation.get("resource_opening_rate_late"),
            "decision_pattern.extreme_option_preference": observation.get("extreme_option_preference"),
            "decision_pattern.choice_entropy": observation.get("choice_entropy"),
            "resource_utilization.mean_residual_ratio": observation.get("mean_residual_ratio"),
            "resource_utilization.residual_variance": observation.get("residual_variance"),
            "resource_utilization.fragmentation_index": observation.get("fragmentation_index"),
            "robustness.order_sensitivity": observation.get("order_sensitivity"),
            "robustness.instance_family_variance": observation.get("instance_family_variance"),
        }
        if metric_name == "search.operator_success_rate":
            result = {k: _safe_float(v.get("success_rate"), 0.0) for k, v in op_stats.items() if isinstance(v, dict)}
            return {"value": result, "source": "observation.op_stats_k"}
        if metric_name == "search.operator_mean_delta":
            result = {k: _safe_float(v.get("mean_delta"), 0.0) for k, v in op_stats.items() if isinstance(v, dict)}
            return {"value": result, "source": "observation.op_stats_k"}
        if metric_name in mapping:
            raw = mapping[metric_name]
            if raw is None:
                return {"value": None, "source": "missing"}
            return {"value": _safe_float(raw, 0.0), "source": "observation"}
        return {"value": None, "source": "unsupported"}

    def _from_profiles(self, metric_name: str, profiles: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(profiles) == 0:
            return {"value": None, "source": "profiles.empty"}

        if metric_name.startswith("structure."):
            values = []
            for p in profiles:
                cm = p.get("complexity_metrics", {}) if isinstance(p.get("complexity_metrics"), dict) else {}
                if metric_name in cm:
                    values.append(_safe_float(cm.get(metric_name), 0.0))
            if len(values) == 0:
                return {"value": None, "source": "profiles.missing_metric"}
            return {"value": float(sum(values) / len(values)), "source": "profiles.complexity"}

        if metric_name == "search.population_fitness_diversity":
            vals = [_safe_float(p.get("scalar_fitness"), 0.0) for p in profiles]
            if len(vals) <= 1:
                return {"value": 0.0, "source": "profiles.fitness"}
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            return {"value": float(var), "source": "profiles.fitness"}

        return {"value": None, "source": "profiles.unsupported"}

    def build(
        self,
        measurement_plan: MeasurementPlan,
        observation: Dict[str, Any],
        profiles: List[Dict[str, Any]],
        generation: int,
    ) -> BehaviorEvidenceReport:
        metric_values: Dict[str, Dict[str, Any]] = {}
        for metric in measurement_plan.requested_metrics:
            val = self._from_observation(metric, observation)
            if val.get("source") in ["missing", "unsupported"]:
                prof_val = self._from_profiles(metric, profiles)
                if prof_val.get("value") is not None:
                    val = prof_val
            metric_values[metric] = val

        target_summaries = []
        for t in measurement_plan.targets:
            target_summaries.append(
                {
                    "target_id": t.get("target_id", "population"),
                    "target_type": t.get("target_type", "population"),
                    "notes": t.get("notes", ""),
                }
            )

        notes = []
        if observation.get("notes"):
            notes.append(str(observation.get("notes")))
        missing_metrics = [k for k, v in metric_values.items() if v.get("value") is None]
        if len(missing_metrics) > 0:
            notes.append("missing_metrics=" + ",".join(missing_metrics))

        return BehaviorEvidenceReport(
            report_id=f"evidence_g{generation}_{measurement_plan.plan_id}",
            based_on_measurement_plan=measurement_plan.plan_id,
            generation=int(generation),
            metric_values=metric_values,
            target_summaries=target_summaries,
            notes=notes,
        )
