from typing import List

from .models import HeuristicCard, HeuristicDiagnosis


def _metric(card: HeuristicCard, key: str):
    value = (card.behavior or {}).get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def diagnose_heuristic(card: HeuristicCard) -> HeuristicDiagnosis:
    labels: List[str] = []
    evidence: List[str] = []

    early_open = _metric(card, "resource_opening_rate_early")
    fragmentation = _metric(card, "fragmentation_index")
    entropy = _metric(card, "choice_entropy")
    margin = _metric(card, "score_margin_mean")
    order = _metric(card, "order_sensitivity")
    holdout_gap = _metric(card, "holdout_gap")
    simplicity = float(card.simplicity_index or 0.0)

    if early_open is not None and early_open >= 0.35:
        labels.append("aggressive_early_commitment")
        evidence.append(f"resource_opening_rate_early={early_open:.4f}")
    if fragmentation is not None and fragmentation >= 0.45:
        labels.append("high_fragmentation")
        evidence.append(f"fragmentation_index={fragmentation:.4f}")
    if entropy is not None and margin is not None and entropy <= 0.25 and margin <= 0.05:
        labels.append("low_margin_decisions")
        evidence.append(f"choice_entropy={entropy:.4f}")
        evidence.append(f"score_margin_mean={margin:.4f}")
    if (order is not None and order >= 0.08) or (holdout_gap is not None and holdout_gap >= 0.05):
        labels.append("brittle_order_sensitivity")
        if order is not None:
            evidence.append(f"order_sensitivity={order:.4f}")
        if holdout_gap is not None:
            evidence.append(f"holdout_gap={holdout_gap:.4f}")
    if simplicity >= 0.60:
        labels.append("simple_promising_motif")
        evidence.append(f"simplicity_index={simplicity:.4f}")

    missing_keys = [key for key, status in (card.behavior_status or {}).items() if status != "ok"]
    if len(labels) == 0 and len(missing_keys) >= 4:
        labels.append("missing_behavior_signal")
        evidence.append("behavior metrics unavailable for diagnosis")

    if len(labels) == 0:
        labels = ["behaviorally_neutral_candidate"]
        evidence = evidence or [f"fitness={card.fitness}"]

    confidence = 0.55 + 0.08 * min(3, len(labels))
    confidence = max(0.0, min(0.95, confidence))
    summary = " / ".join(labels[:3]).replace("_", " ")

    return HeuristicDiagnosis(
        labels=labels[:4],
        summary=summary,
        confidence=confidence,
        key_evidence=evidence[:3],
    )
