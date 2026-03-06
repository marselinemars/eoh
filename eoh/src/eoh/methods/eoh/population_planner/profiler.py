import hashlib
import re
from typing import Any, Dict, List

from .diagnosis import diagnose_heuristic
from .models import HeuristicCard


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _code_hash(code: str) -> str:
    if not isinstance(code, str):
        return "none"
    return hashlib.sha1(code.encode("utf-8")).hexdigest()[:12]


def _condition_count(code: str) -> float:
    return float(len(re.findall(r"\b(if|elif)\b", code or "")))


def _parameter_count(code: str) -> float:
    constants = re.findall(r"(?<![\w.])[-+]?\d+\.\d+|(?<![\w.])[-+]?\d+", code or "")
    return float(len(constants))


def _complexity(code: str) -> float:
    return float(len(code or ""))


def _simplicity_index(code: str) -> float:
    length_penalty = min(1.0, float(len(code or "")) / 1600.0)
    cond_penalty = min(1.0, _condition_count(code) / 24.0)
    param_penalty = min(1.0, _parameter_count(code) / 24.0)
    score = 1.0 - (0.45 * length_penalty + 0.30 * cond_penalty + 0.25 * param_penalty)
    return float(max(0.0, min(1.0, score)))


def _clean_summary(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    return " ".join(text.split())[:220]


BEHAVIOR_KEYS = [
    "mean_residual_ratio",
    "residual_variance",
    "fragmentation_index",
    "resource_opening_rate_early",
    "resource_opening_rate_mid",
    "resource_opening_rate_late",
    "choice_entropy",
    "extreme_option_preference",
    "score_margin_mean",
    "score_margin_variance",
    "order_sensitivity",
    "family_variance",
    "holdout_gap",
]


def build_heuristic_cards(population: List[Dict[str, Any]], generation: int) -> List[HeuristicCard]:
    cards: List[HeuristicCard] = []
    for rank, individual in enumerate(population):
        code = individual.get("code")
        other_inf = individual.get("other_inf", {}) if isinstance(individual.get("other_inf"), dict) else {}
        trace = other_inf.get("trace_metrics", {}) if isinstance(other_inf.get("trace_metrics"), dict) else {}
        lineage = other_inf.get("lineage", {}) if isinstance(other_inf.get("lineage"), dict) else {}
        heuristic_id = str(other_inf.get("heuristic_id") or f"H{rank+1}_{_code_hash(code)}")
        behavior = {key: _safe_float(trace.get(key), 0.0) for key in BEHAVIOR_KEYS}
        card = HeuristicCard(
            id=heuristic_id,
            generation=int(generation),
            fitness=_safe_float(individual.get("objective"), 0.0),
            rank=int(rank + 1),
            algorithm_summary=_clean_summary(individual.get("algorithm")),
            complexity=_complexity(code),
            parameter_count=_parameter_count(code),
            condition_count=_condition_count(code),
            simplicity_index=_simplicity_index(code),
            behavior=behavior,
            diagnosis=None,  # type: ignore[arg-type]
            created_by=str(lineage.get("created_by") or "seed"),
            parent_ids=[str(x) for x in lineage.get("parent_ids", [])] if isinstance(lineage.get("parent_ids"), list) else [],
            code_hash=_code_hash(code),
            code=code,
            extra={
                "parent_hashes": lineage.get("parent_hashes", []),
                "lineage_note": lineage.get("note", ""),
            },
        )
        card.diagnosis = diagnose_heuristic(card)
        cards.append(card)
    return cards
