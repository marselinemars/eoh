import json

from ....llm.interface_LLM import InterfaceLLM
from .models import PlannerIntervention, PopulationPlan
from .prompts import build_population_planner_prompt


ALLOWED_MODES = {"rewrite", "tune", "variant", "explore", "evaluate"}
ALLOWED_PRIORITIES = {"high", "medium", "low"}


def _try_parse_json_object(text):
    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return None
    return None


class PopulationPlanner:
    def __init__(self, api_endpoint, api_key, model_llm, llm_use_local, llm_local_url, debug_mode=False):
        self.debug_mode = bool(debug_mode)
        self.interface_llm = InterfaceLLM(
            api_endpoint,
            api_key,
            model_llm,
            llm_use_local,
            llm_local_url,
            self.debug_mode,
        )

    def _fallback_plan(self, summary, cards, generation_budget):
        ranked = sorted(cards, key=lambda card: (card.fitness, card.rank))
        best = ranked[0] if ranked else None
        secondary = ranked[1] if len(ranked) > 1 else best
        interventions = []
        if best is not None:
            rewrite_budget = max(1, generation_budget // 2)
            interventions.append(
                PlannerIntervention(
                    targets=[best.id],
                    execution_mode="rewrite",
                    goal="Preserve the best motif while reducing its diagnosed weakness.",
                    instruction="Keep the core packing idea but reduce the main diagnosis risk and avoid unnecessary complexity.",
                    offspring_count=rewrite_budget,
                    priority="high",
                )
            )
        if secondary is not None:
            interventions.append(
                PlannerIntervention(
                    targets=[secondary.id],
                    execution_mode="variant",
                    goal="Keep a nearby alternative active.",
                    instruction="Generate a meaningfully different variant that keeps useful structure but changes the behavior profile.",
                    offspring_count=max(1, generation_budget - sum(item.offspring_count for item in interventions)),
                    priority="medium",
                )
            )
        return PopulationPlan(
            population_assessment=f"Fallback planner for regime={summary.current_search_regime}.",
            overall_strategy="Preserve the strongest heuristic while spending remaining budget on a nearby variant.",
            interventions=interventions,
            preserve_ids=[best.id] if best is not None else [],
            deprioritize_ids=[],
            rationale=[
                "Fallback used because planner output was missing or invalid.",
                "Budget is concentrated on the best card and one nearby alternative.",
            ],
        )

    def _sanitize(self, obj, cards, generation_budget, summary):
        valid_ids = {card.id for card in cards}
        if not isinstance(obj, dict):
            return self._fallback_plan(summary, cards, generation_budget), True

        interventions = []
        raw_interventions = obj.get("interventions", [])
        if not isinstance(raw_interventions, list):
            raw_interventions = []
        for item in raw_interventions:
            if not isinstance(item, dict):
                continue
            mode = str(item.get("execution_mode", "")).strip().lower()
            if mode not in ALLOWED_MODES:
                continue
            targets = [str(x) for x in item.get("targets", []) if str(x) in valid_ids]
            if mode != "explore" and len(targets) == 0:
                continue
            priority = str(item.get("priority", "medium")).strip().lower()
            if priority not in ALLOWED_PRIORITIES:
                priority = "medium"
            try:
                offspring_count = int(item.get("offspring_count", 1))
            except (TypeError, ValueError):
                offspring_count = 1
            if mode == "evaluate":
                offspring_count = 0
            offspring_count = max(0, offspring_count)
            interventions.append(
                PlannerIntervention(
                    targets=targets,
                    execution_mode=mode,
                    goal=str(item.get("goal", "")).strip() or "Improve the next generation.",
                    instruction=str(item.get("instruction", "")).strip() or "Make a useful improvement.",
                    offspring_count=offspring_count,
                    priority=priority,
                )
            )

        if len(interventions) == 0:
            return self._fallback_plan(summary, cards, generation_budget), True

        total_requested = sum(item.offspring_count for item in interventions)
        if total_requested <= 0:
            return self._fallback_plan(summary, cards, generation_budget), True
        if total_requested > generation_budget:
            scale = generation_budget / float(total_requested)
            adjusted = []
            assigned = 0
            for intervention in interventions:
                if intervention.execution_mode == "evaluate":
                    adjusted.append(intervention)
                    continue
                count = max(1, int(round(intervention.offspring_count * scale)))
                adjusted.append(
                    PlannerIntervention(
                        targets=intervention.targets,
                        execution_mode=intervention.execution_mode,
                        goal=intervention.goal,
                        instruction=intervention.instruction,
                        offspring_count=count,
                        priority=intervention.priority,
                    )
                )
                assigned += count
            while assigned > generation_budget:
                for idx in range(len(adjusted) - 1, -1, -1):
                    item = adjusted[idx]
                    if item.execution_mode != "evaluate" and item.offspring_count > 1 and assigned > generation_budget:
                        adjusted[idx] = PlannerIntervention(
                            targets=item.targets,
                            execution_mode=item.execution_mode,
                            goal=item.goal,
                            instruction=item.instruction,
                            offspring_count=item.offspring_count - 1,
                            priority=item.priority,
                        )
                        assigned -= 1
            interventions = adjusted

        preserve_ids = [str(x) for x in obj.get("preserve_ids", []) if str(x) in valid_ids]
        deprioritize_ids = [str(x) for x in obj.get("deprioritize_ids", []) if str(x) in valid_ids]
        plan = PopulationPlan(
            population_assessment=str(obj.get("population_assessment", "")).strip() or "Population assessed from heuristic cards.",
            overall_strategy=str(obj.get("overall_strategy", "")).strip() or "Balance refinement and exploration.",
            interventions=interventions,
            preserve_ids=preserve_ids[:3],
            deprioritize_ids=deprioritize_ids[:3],
            rationale=[str(x) for x in obj.get("rationale", [])][:6] if isinstance(obj.get("rationale", []), list) else [],
        )
        if len(plan.rationale) == 0:
            plan.rationale = ["Planner rationale unavailable; sanitized output used."]
        return plan, False

    def plan(self, problem_context, summary, cards, generation_budget):
        prompt = build_population_planner_prompt(problem_context, summary, cards)
        raw_text = self.interface_llm.get_response(prompt, request_mode="json")
        raw_obj = _try_parse_json_object(raw_text)
        plan, fallback_used = self._sanitize(raw_obj, cards, generation_budget, summary)
        return {
            "plan": plan,
            "raw_text": raw_text,
            "fallback_used": fallback_used,
            "prompt": prompt,
        }
