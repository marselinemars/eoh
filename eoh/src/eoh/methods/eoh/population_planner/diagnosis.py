from .models import HeuristicDiagnosis, StructureMetrics


def _clip01(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def diagnose_heuristic(structure: StructureMetrics, behavior) -> HeuristicDiagnosis:
    labels = []
    evidence = []

    if behavior.resource_opening_rate_early >= 0.35:
        labels.append("aggressive_early_commitment")
        evidence.append(f"high early open rate={behavior.resource_opening_rate_early:.3f}")
    if behavior.fragmentation_index >= 0.35:
        labels.append("high_fragmentation")
        evidence.append(f"fragmentation index={behavior.fragmentation_index:.3f}")
    if behavior.order_sensitivity >= 0.08:
        labels.append("order_sensitive")
        evidence.append(f"order sensitivity={behavior.order_sensitivity:.3f}")
    if behavior.holdout_gap >= 0.01:
        labels.append("train_holdout_gap")
        evidence.append(f"holdout gap={behavior.holdout_gap:.4f}")
    if behavior.choice_entropy <= 0.20:
        labels.append("narrow_decision_policy")
        evidence.append(f"low choice entropy={behavior.choice_entropy:.3f}")
    if structure.simplicity_index >= 0.70:
        labels.append("structurally_simple")
        evidence.append(f"simplicity index={structure.simplicity_index:.3f}")
    if structure.complexity >= 420:
        labels.append("structurally_heavy")
        evidence.append(f"complexity={structure.complexity:.1f}")
    if behavior.score_margin_mean <= 0.02:
        labels.append("low_margin_selector")
        evidence.append(f"score margin mean={behavior.score_margin_mean:.4f}")

    if len(labels) == 0:
        labels = ["balanced_profile"]
        evidence.append(f"residual variance={behavior.residual_variance:.4f}")

    summary_parts = []
    if "aggressive_early_commitment" in labels:
        summary_parts.append("strong early commitment")
    if "high_fragmentation" in labels:
        summary_parts.append("fragmentation risk")
    if "order_sensitive" in labels:
        summary_parts.append("brittle to sequence order")
    if "structurally_simple" in labels:
        summary_parts.append("simple update rule")
    if len(summary_parts) == 0:
        summary_parts.append("behaviorally balanced")

    confidence = 0.45
    confidence += min(0.20, abs(behavior.resource_opening_rate_early - 0.30))
    confidence += min(0.15, abs(behavior.fragmentation_index - 0.25))
    confidence += min(0.15, abs(behavior.order_sensitivity - 0.05))
    confidence = _clip01(confidence)

    return HeuristicDiagnosis(
        labels=labels[:4],
        summary=f"{labels[0]}: " + ", ".join(summary_parts[:3]),
        confidence=confidence,
        key_evidence=evidence[:3],
    )
