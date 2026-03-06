from typing import Any, Dict, List

from .models import normalize_op_probs, normalize_parent_mix
from .ontology import ActionRegistry, CORRECTIVE_PROMPT_MODIFIER_MAP


class InterventionExecutor:
    """Translate intervention ontology into executable EOH control knobs."""

    def __init__(self, action_registry: ActionRegistry):
        self.action_registry = action_registry

    def _default_plan(self, budget: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "op_probs": {"e1": 0.02, "e2": 0.50, "m1": 0.16, "m2": 0.24, "m3": 0.08},
            "parent_mix": {"elite": 0.5, "diverse": 0.3, "random": 0.2},
            "prompt_modifiers": [],
            "evaluation_plan": {
                "instances": int(budget.get("instances", 0) or 0),
                "holdout_instances": int(budget.get("holdout_instances", 0) or 0),
            },
        }

    def apply(self, portfolio: Dict[str, Any], observation: Dict[str, Any]) -> Dict[str, Any]:
        budget = observation.get("budget", {}) if isinstance(observation.get("budget"), dict) else {}
        plan = self._default_plan(budget)
        applied_actions = []
        interventions = portfolio.get("interventions", [])
        if not isinstance(interventions, list):
            interventions = []

        for action_item in interventions:
            if not isinstance(action_item, dict):
                continue
            action_name = str(action_item.get("action", "")).strip()
            weight = float(action_item.get("weight", 0.0) or 0.0)
            payload = action_item.get("payload", {}) if isinstance(action_item.get("payload"), dict) else {}
            spec = self.action_registry.get(action_name)
            if spec is None:
                continue

            if spec.execution_kind == "evolutionary_operator":
                op = spec.operator_id
                if op in plan["op_probs"]:
                    plan["op_probs"][op] = plan["op_probs"].get(op, 0.0) + max(0.0, weight)
                    applied_actions.append({"action": action_name, "effect": f"boost_{op}", "weight": weight})

            elif spec.execution_kind == "prompt_modifier":
                modifier = CORRECTIVE_PROMPT_MODIFIER_MAP.get(action_name)
                if modifier is not None:
                    plan["prompt_modifiers"].append(modifier)
                    applied_actions.append({"action": action_name, "effect": "prompt_modifier"})

            elif spec.execution_kind == "search_structure":
                if action_name == "search_structure.refresh_diversity_pool":
                    plan["parent_mix"]["diverse"] = plan["parent_mix"].get("diverse", 0.0) + max(0.1, weight)
                    applied_actions.append({"action": action_name, "effect": "increase_diverse_parent_mix"})
                elif action_name == "search_structure.split_exploitation_branch":
                    plan["parent_mix"]["elite"] = plan["parent_mix"].get("elite", 0.0) + max(0.1, weight)
                    applied_actions.append({"action": action_name, "effect": "increase_elite_parent_mix"})
                elif action_name == "search_structure.allocate_budget_to_branch":
                    if "instances" in payload:
                        try:
                            wanted = int(payload["instances"])
                            max_budget = int(budget.get("instances", wanted) or wanted)
                            plan["evaluation_plan"]["instances"] = max(1, min(wanted, max_budget))
                            applied_actions.append({"action": action_name, "effect": "set_instances", "instances": plan["evaluation_plan"]["instances"]})
                        except Exception:
                            pass

            elif spec.execution_kind == "eval_request":
                applied_actions.append({"action": action_name, "effect": "evaluation_request"})
            elif spec.execution_kind == "memory":
                applied_actions.append({"action": action_name, "effect": "memory_request"})

        plan["prompt_modifiers"] = [m for i, m in enumerate(plan["prompt_modifiers"]) if m and m not in plan["prompt_modifiers"][:i]][:4]
        plan["op_probs"] = normalize_op_probs(plan["op_probs"])
        plan["parent_mix"] = normalize_parent_mix(plan["parent_mix"])

        max_instances = int(budget.get("instances", plan["evaluation_plan"]["instances"]) or plan["evaluation_plan"]["instances"])
        max_holdout = int(budget.get("holdout_instances", plan["evaluation_plan"]["holdout_instances"]) or plan["evaluation_plan"]["holdout_instances"])
        plan["evaluation_plan"]["instances"] = max(1, min(int(plan["evaluation_plan"]["instances"]), max_instances)) if max_instances > 0 else int(plan["evaluation_plan"]["instances"])
        plan["evaluation_plan"]["holdout_instances"] = max(0, min(int(plan["evaluation_plan"]["holdout_instances"]), max_holdout))
        plan["applied_actions"] = applied_actions
        return plan
