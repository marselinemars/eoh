import json
from typing import Any, Dict, List, Optional

from ....llm.interface_LLM import InterfaceLLM
from .models import PlannerIntervention, PopulationPlannerOutput, validate_population_planner_output
from .prompts import build_planner_prompt


def _try_parse_json_object(text):
    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        return None
    try:
        obj = json.loads(text.strip())
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    if isinstance(text, str):
        start = text.find("{")
        end = text.rfind("}")
        if 0 <= start < end:
            try:
                obj = json.loads(text[start : end + 1])
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
    return None


class PopulationPlanner:
    def __init__(self, api_endpoint, api_key, model_llm, llm_use_local, llm_local_url, debug_mode=False, max_retries=3):
        self.interface_llm = InterfaceLLM(api_endpoint, api_key, model_llm, llm_use_local, llm_local_url, debug_mode)
        self.max_retries = max(1, int(max_retries))

    def _validate(self, payload: Dict[str, Any], available_ids: List[str]) -> List[str]:
        return validate_population_planner_output(payload, available_ids)

    def _retry_prompt(self, base_prompt: str, errors: List[str]) -> str:
        bullets = "\n".join(f"- {err}" for err in errors[:10]) or "- invalid_json"
        return base_prompt + "\n\nYour previous output was invalid. Output ONLY valid JSON. Fix these errors:\n" + bullets

    def _fallback(self, available_ids: List[str]) -> Dict[str, Any]:
        keep = available_ids[:2]
        target = available_ids[:1]
        return {
            "population_assessment": "Fallback planner output under parse/validation failure.",
            "overall_strategy": "Preserve the best visible motif and generate one safe variant and one local tune.",
            "interventions": [
                {
                    "targets": target,
                    "execution_mode": "variant",
                    "goal": "Create a nearby motivated variant from the current best heuristic.",
                    "instruction": "Preserve the strong motif but change one meaningful scoring component.",
                    "offspring_count": 1,
                    "priority": "high",
                },
                {
                    "targets": target,
                    "execution_mode": "tune",
                    "goal": "Refine local balance in the current best heuristic.",
                    "instruction": "Adjust only coefficients or local term balance.",
                    "offspring_count": 1,
                    "priority": "medium",
                },
            ],
            "preserve_ids": keep,
            "deprioritize_ids": [],
            "rationale": ["Fallback keeps the best visible motif while still creating some variation."],
        }

    def plan(self, problem_context: str, summary, cards, planner_context: str = "- none") -> Dict[str, Any]:
        available_ids = [card.id for card in cards]
        prompt = build_planner_prompt(problem_context, summary, cards, planner_context=planner_context)
        attempts = []
        current_prompt = prompt
        raw_output = None
        for _ in range(self.max_retries):
            raw_output = self.interface_llm.get_response(current_prompt, request_mode="json", tool_name="population_planner")
            parsed = _try_parse_json_object(raw_output)
            if isinstance(parsed, dict):
                errors = self._validate(parsed, available_ids)
                attempts.append({"raw_output": raw_output, "errors": errors})
                if len(errors) == 0:
                    interventions = [PlannerIntervention(**item) for item in parsed["interventions"]]
                    return {
                        "planner_output": PopulationPlannerOutput(
                            population_assessment=str(parsed["population_assessment"]),
                            overall_strategy=str(parsed["overall_strategy"]),
                            interventions=interventions,
                            preserve_ids=[str(x) for x in parsed["preserve_ids"]],
                            deprioritize_ids=[str(x) for x in parsed["deprioritize_ids"]],
                            rationale=[str(x) for x in parsed["rationale"]],
                        ),
                        "llm_meta": {
                            "success": True,
                            "attempts": attempts,
                            "raw_output": raw_output,
                        },
                    }
                current_prompt = self._retry_prompt(prompt, errors)
            else:
                attempts.append({"raw_output": raw_output, "errors": ["invalid_json_or_not_object"]})
                current_prompt = self._retry_prompt(prompt, ["invalid_json_or_not_object"])

        fallback = self._fallback(available_ids)
        interventions = [PlannerIntervention(**item) for item in fallback["interventions"]]
        return {
            "planner_output": PopulationPlannerOutput(
                population_assessment=fallback["population_assessment"],
                overall_strategy=fallback["overall_strategy"],
                interventions=interventions,
                preserve_ids=fallback["preserve_ids"],
                deprioritize_ids=fallback["deprioritize_ids"],
                rationale=fallback["rationale"],
            ),
            "llm_meta": {
                "success": False,
                "attempts": attempts,
                "raw_output": raw_output,
            },
        }
