import hashlib
import re
from typing import Any, Dict, List

from .models import HeuristicProfile


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _ast_proxy_depth(code: str) -> float:
    if not isinstance(code, str) or not code:
        return 0.0
    indent_levels = [len(line) - len(line.lstrip(" ")) for line in code.splitlines() if line.strip()]
    if len(indent_levels) == 0:
        return 0.0
    return float(max(indent_levels) / 4.0)


def _condition_count(code: str) -> float:
    if not isinstance(code, str):
        return 0.0
    return float(len(re.findall(r"\b(if|elif)\b", code)))


def _parameter_count(code: str) -> float:
    if not isinstance(code, str):
        return 0.0
    constants = re.findall(r"(?<![\w.])[-+]?\d+\.\d+|(?<![\w.])[-+]?\d+", code)
    return float(len(constants))


def _operator_count(code: str) -> float:
    if not isinstance(code, str):
        return 0.0
    return float(len(re.findall(r"[+\-*/]", code)))


def _redundancy_index(code: str) -> float:
    if not isinstance(code, str):
        return 0.0
    lines = [line.strip() for line in code.splitlines() if line.strip()]
    if len(lines) == 0:
        return 0.0
    unique = len(set(lines))
    return float(max(0.0, 1.0 - (unique / max(1.0, float(len(lines))))))


def _simplicity_index(code: str) -> float:
    code_len = max(1.0, float(len(code) if isinstance(code, str) else 0))
    depth = _ast_proxy_depth(code)
    cond = _condition_count(code)
    params = _parameter_count(code)
    complexity = 0.35 * min(1.0, code_len / 1600.0) + 0.30 * min(1.0, depth / 16.0) + 0.20 * min(1.0, cond / 24.0) + 0.15 * min(1.0, params / 32.0)
    return float(max(0.0, 1.0 - complexity))


def _code_hash(code: str) -> str:
    if not isinstance(code, str):
        return "none"
    return hashlib.sha1(code.encode("utf-8")).hexdigest()[:12]


def build_heuristic_profiles(population: List[Dict[str, Any]], generation_index: int, source_action: str) -> List[HeuristicProfile]:
    profiles: List[HeuristicProfile] = []
    for rank, individual in enumerate(population):
        code = individual.get("code")
        algo = individual.get("algorithm")
        other_inf = individual.get("other_inf", {}) if isinstance(individual.get("other_inf"), dict) else {}
        trace_metrics = other_inf.get("trace_metrics", {}) if isinstance(other_inf.get("trace_metrics"), dict) else {}
        metric_reasons = other_inf.get("metric_reasons", {}) if isinstance(other_inf.get("metric_reasons"), dict) else {}
        instance_family_performance = other_inf.get("instance_family_performance", {}) if isinstance(other_inf.get("instance_family_performance"), dict) else {}
        complexity = {
            "structure.code_length": float(len(code)) if isinstance(code, str) else 0.0,
            "structure.ast_depth": _ast_proxy_depth(code),
            "structure.condition_count": _condition_count(code),
            "structure.parameter_count": _parameter_count(code),
            "structure.operator_count": _operator_count(code),
            "structure.expression_redundancy": _redundancy_index(code),
            "structure.simplicity_index": _simplicity_index(code),
        }
        profile = HeuristicProfile(
            heuristic_id=f"g{generation_index}_r{rank}_{_code_hash(code)}",
            code_hash=_code_hash(code),
            summary=str(algo) if isinstance(algo, str) else "",
            scalar_fitness=_safe_float(individual.get("objective"), default=0.0),
            complexity_metrics=complexity,
            behavior_trace_summary={k: _safe_float(v, 0.0) for k, v in trace_metrics.items()},
            instance_family_performance={k: _safe_float(v, 0.0) for k, v in instance_family_performance.items()},
            parent_lineage=[],
            generating_action=source_action,
            generation_created=int(generation_index),
            extra={"metric_reasons": metric_reasons},
        )
        profiles.append(profile)
    return profiles
