from typing import List

from .models import HeuristicCard, HeuristicDiagnosis


def diagnose_heuristic(card: HeuristicCard) -> HeuristicDiagnosis:
    behavior = card.behavior or {}
    labels: List[str] = []
    evidence: List[str] = []

    early_open = float(behavior.get("resource_opening_rate_early", 0.0) or 0.0)
    fragmentation = float(behavior.get("fragmentation_index", 0.0) or 0.0)
    entropy = float(behavior.get("choice_entropy", 0.0) or 0.0)
    margin = float(behavior.get("score_margin_mean", 0.0) or 0.0)
    order = float(behavior.get("order_sensitivity", 0.0) or 0.0)
    holdout_gap = float(behavior.get("holdout_gap", 0.0) or 0.0)
    simplicity = float(card.simplicity_index or 0.0)

    if early_open >= 0.35:
        labels.append("aggressive_early_commitment")
        evidence.append(f"resource_opening_rate_early={early_open:.4f}")
    if fragmentation >= 0.45:
        labels.append("high_fragmentation")
        evidence.append(f"fragmentation_index={fragmentation:.4f}")
    if entropy <= 0.25 and margin <= 0.05:
        labels.append("low_margin_decisions")
        evidence.append(f"choice_entropy={entropy:.4f}")
        evidence.append(f"score_margin_mean={margin:.4f}")
    if order >= 0.08 or holdout_gap >= 0.05:
        labels.append("brittle_order_sensitivity")
        evidence.append(f"order_sensitivity={order:.4f}")
        evidence.append(f"holdout_gap={holdout_gap:.4f}")
    if simplicity >= 0.60:
        labels.append("simple_promising_motif")
        evidence.append(f"simplicity_index={simplicity:.4f}")

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
