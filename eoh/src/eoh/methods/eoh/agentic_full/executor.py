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
            "parent_mix": {"elite": 0.45, "diverse": 0.25, "random": 0.20, "preferred": 0.10},
            "prompt_modifiers": [],
            "evaluation_plan": {
                "instances": int(budget.get("instances", 0) or 0),
                "holdout_instances": int(budget.get("holdout_instances", 0) or 0),
            },
            "preferred_parent_hashes": [],
            "selected_parent_groups": [],
            "branch_execution": [],
        }

    def _collect_interventions(self, portfolio: Dict[str, Any]) -> List[Dict[str, Any]]:
        collected = []
        global_items = portfolio.get("interventions", [])
        if isinstance(global_items, list):
            for item in global_items:
                if isinstance(item, dict):
                    enriched = dict(item)
                    enriched["_branch_id"] = None
                    enriched["_branch_budget_share"] = 1.0
                    collected.append(enriched)

        branches = portfolio.get("branches", [])
        if isinstance(branches, list):
            for branch in branches:
                if not isinstance(branch, dict):
                    continue
                branch_id = str(branch.get("branch_id", "")).strip() or "branch"
                branch_budget_share = float(branch.get("budget_share", 0.0) or 0.0)
                branch_budget_share = max(0.0, min(1.0, branch_budget_share))
                branch_parent_groups = branch.get("parent_candidate_groups", [])
                branch_parent_hashes = branch.get("parent_candidate_hashes", [])
                branch_objective = str(branch.get("objective", "")).strip()
                for item in branch.get("interventions", []) if isinstance(branch.get("interventions"), list) else []:
                    if not isinstance(item, dict):
                        continue
                    enriched = dict(item)
                    payload = dict(item.get("payload", {})) if isinstance(item.get("payload"), dict) else {}
                    if isinstance(branch_parent_groups, list) and len(branch_parent_groups) > 0:
                        payload.setdefault("parent_candidate_groups", list(branch_parent_groups))
                    if isinstance(branch_parent_hashes, list) and len(branch_parent_hashes) > 0:
                        payload.setdefault("parent_candidate_hashes", list(branch_parent_hashes))
                    if branch_objective:
                        payload.setdefault("branch_objective", branch_objective)
                    enriched["payload"] = payload
                    enriched["_branch_id"] = branch_id
                    enriched["_branch_budget_share"] = branch_budget_share if branch_budget_share > 0.0 else 1.0
                    collected.append(enriched)
        return collected

    def apply(self, portfolio: Dict[str, Any], observation: Dict[str, Any]) -> Dict[str, Any]:
        budget = observation.get("budget", {}) if isinstance(observation.get("budget"), dict) else {}
        plan = self._default_plan(budget)
        applied_actions = []
        interventions = self._collect_interventions(portfolio)
        parent_group_set = set()
        preferred_hashes = []

        branches = portfolio.get("branches", [])
        if isinstance(branches, list):
            for branch in branches:
                if not isinstance(branch, dict):
                    continue
                plan["branch_execution"].append(
                    {
                        "branch_id": str(branch.get("branch_id", "")).strip(),
                        "objective": str(branch.get("objective", "")).strip(),
                        "budget_share": float(branch.get("budget_share", 0.0) or 0.0),
                        "parent_candidate_groups": branch.get("parent_candidate_groups", []),
                        "intervention_count": len(branch.get("interventions", [])) if isinstance(branch.get("interventions"), list) else 0,
                    }
                )

        for action_item in interventions:
            if not isinstance(action_item, dict):
                continue
            action_name = str(action_item.get("action", "")).strip()
            branch_budget_share = float(action_item.get("_branch_budget_share", 1.0) or 1.0)
            branch_budget_share = max(0.0, min(1.0, branch_budget_share))
            weight = float(action_item.get("weight", 0.0) or 0.0) * branch_budget_share
            payload = action_item.get("payload", {}) if isinstance(action_item.get("payload"), dict) else {}
            spec = self.action_registry.get(action_name)
            if spec is None:
                continue
            branch_id = action_item.get("_branch_id")
            for group_name in payload.get("parent_candidate_groups", []) if isinstance(payload.get("parent_candidate_groups"), list) else []:
                parent_group_set.add(str(group_name))
            for code_hash in payload.get("parent_candidate_hashes", []) if isinstance(payload.get("parent_candidate_hashes"), list) else []:
                code_hash = str(code_hash).strip()
                if code_hash:
                    preferred_hashes.append(code_hash)

            if spec.execution_kind == "evolutionary_operator":
                op = spec.operator_id
                if op in plan["op_probs"]:
                    plan["op_probs"][op] = plan["op_probs"].get(op, 0.0) + max(0.0, weight)
                    applied_actions.append({"action": action_name, "effect": f"boost_{op}", "weight": weight, "branch_id": branch_id})

            elif spec.execution_kind == "prompt_modifier":
                modifier = CORRECTIVE_PROMPT_MODIFIER_MAP.get(action_name)
                if modifier is not None:
                    plan["prompt_modifiers"].append(modifier)
                    applied_actions.append({"action": action_name, "effect": "prompt_modifier", "branch_id": branch_id})

            elif spec.execution_kind == "search_structure":
                if action_name == "search_structure.refresh_diversity_pool":
                    plan["parent_mix"]["diverse"] = plan["parent_mix"].get("diverse", 0.0) + max(0.1, weight)
                    applied_actions.append({"action": action_name, "effect": "increase_diverse_parent_mix", "branch_id": branch_id})
                elif action_name == "search_structure.split_exploitation_branch":
                    plan["parent_mix"]["elite"] = plan["parent_mix"].get("elite", 0.0) + max(0.1, weight)
                    applied_actions.append({"action": action_name, "effect": "increase_elite_parent_mix", "branch_id": branch_id})
                elif action_name == "search_structure.split_exploration_branch":
                    plan["parent_mix"]["preferred"] = plan["parent_mix"].get("preferred", 0.0) + max(0.05, weight)
                    plan["parent_mix"]["diverse"] = plan["parent_mix"].get("diverse", 0.0) + max(0.05, weight)
                    applied_actions.append({"action": action_name, "effect": "increase_branch_exploration_mix", "branch_id": branch_id})
                elif action_name == "search_structure.allocate_budget_to_branch":
                    if "instances" in payload:
                        try:
                            wanted = int(payload["instances"])
                            max_budget = int(budget.get("instances", wanted) or wanted)
                            plan["evaluation_plan"]["instances"] = max(1, min(wanted, max_budget))
                            applied_actions.append({"action": action_name, "effect": "set_instances", "instances": plan["evaluation_plan"]["instances"], "branch_id": branch_id})
                        except Exception:
                            pass

            elif spec.execution_kind == "eval_request":
                applied_actions.append({"action": action_name, "effect": "evaluation_request", "branch_id": branch_id})
            elif spec.execution_kind == "memory":
                applied_actions.append({"action": action_name, "effect": "memory_request", "branch_id": branch_id})

        plan["prompt_modifiers"] = [m for i, m in enumerate(plan["prompt_modifiers"]) if m and m not in plan["prompt_modifiers"][:i]][:4]
        if len(parent_group_set) > 0:
            plan["parent_mix"]["preferred"] = plan["parent_mix"].get("preferred", 0.0) + 0.20
        plan["preferred_parent_hashes"] = [x for i, x in enumerate(preferred_hashes) if x and x not in preferred_hashes[:i]][:12]
        plan["selected_parent_groups"] = sorted(parent_group_set)
        plan["op_probs"] = normalize_op_probs(plan["op_probs"])
        plan["parent_mix"] = normalize_parent_mix(plan["parent_mix"])

        max_instances = int(budget.get("instances", plan["evaluation_plan"]["instances"]) or plan["evaluation_plan"]["instances"])
        max_holdout = int(budget.get("holdout_instances", plan["evaluation_plan"]["holdout_instances"]) or plan["evaluation_plan"]["holdout_instances"])
        plan["evaluation_plan"]["instances"] = max(1, min(int(plan["evaluation_plan"]["instances"]), max_instances)) if max_instances > 0 else int(plan["evaluation_plan"]["instances"])
        plan["evaluation_plan"]["holdout_instances"] = max(0, min(int(plan["evaluation_plan"]["holdout_instances"]), max_holdout))
        plan["applied_actions"] = applied_actions
        return plan
