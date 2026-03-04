import json
import os
import random
from datetime import datetime

from ...llm.interface_LLM import InterfaceLLM


PROJECT_CONTEXT_BLOCK = """PROJECT CONTEXT (READ CAREFULLY)

You are part of an agentic controller for an Evolution of Heuristics (EOH) system.
EOH evolves heuristic scoring functions for ONLINE BIN PACKING (bp_online).

Task:
- Items arrive sequentially.
- Each item must be placed immediately into an existing bin or a new bin.
- Each bin has fixed capacity.
- A heuristic scoring function scores feasible bins for each item; the best-scoring bin is chosen.

Objective / Fitness:
- Fitness is a scalar; LOWER is BETTER (fewer bins / better packing).
- Selection/acceptance uses ONLY scalar fitness and validity. Do not change acceptance rules.

Your job is to control the PROPOSAL distribution:
- choose which evolutionary operator to use next
- choose which parents to use (elite vs diverse vs random)
- optionally add small prompt constraints for the generator (no major changes)

Available operators (you MUST choose only among these):
- e1: Global Novelty - generate a totally different algorithm (high exploration, risky)
- e2: Backbone Variant - extract common backbone from good heuristics and propose a motivated variant (structured exploration, often productive)
- m1: Structural Modification - moderate structural edit of one heuristic (medium exploration, can increase invalid risk)
- m2: Parameter Tuning - adjust parameters in one heuristic (local exploitation, relatively safe)
- m3: Simplification/Generalization - simplify to reduce overfitting and improve robustness (safe when complexity grows)

Goal of the controller:
Maximize search efficiency and final best fitness by adapting operator usage over time.

INPUT JSON: ObservationPacket

You will receive a JSON object called ObservationPacket containing telemetry and summaries.
If some fields are missing, be robust and infer from available information.
"""


OBSERVATION_PACKET_DEF = """ObservationPacket:
{OBSERVATION_PACKET_JSON}
"""


DIAGNOSER_PROMPT_TEMPLATE = """{PROJECT_CONTEXT}

ROLE: Agent 1 - DIAGNOSER

Task:
Read ObservationPacket and infer the current search state.
Output continuous diagnosis factors in [0,1] and short evidence-based labels.

Interpretation guide:
- exploration_need up when stagnation_len is high, improve_rate_k is low, or diversity_score is low.
- exploitation_need up when improve_rate_k is high and improvements are consistent.
- diversity_need up when diversity_score is low or population seems collapsed.
- invalid_risk up when invalid_rate_k is high or risky operators have high invalid_rate.
- overfit_risk up when complexity of top heuristics is rising and improvements are unstable or diminishing.

You MUST:
- Ground your diagnosis in telemetry.
- Keep it concise and falsifiable.
- Do NOT propose operators or actions.

OUTPUT FORMAT (MANDATORY):
Return ONLY valid JSON with EXACT keys below (no markdown, no extra keys):
{{
  "summary": string,
  "factors": {{
    "exploration_need": number,
    "exploitation_need": number,
    "diversity_need": number,
    "invalid_risk": number,
    "overfit_risk": number,
    "confidence": number
  }},
  "diagnosis_labels": [string, ...],
  "evidence": [string, ...]
}}

Rules:
- All factor values must be in [0,1].
- confidence should be lower with low/contradictory evidence.

ObservationPacket:
{OBS_JSON}
"""


PLANNER_PROMPT_TEMPLATE = """{PROJECT_CONTEXT}

ROLE: Agent 2 - PLANNER

Task:
Given the diagnosis and observation packet, produce a plan for the NEXT generation.

Your plan must define:
- operator probabilities (mixture over e1,e2,m1,m2,m3)
- parent selection mix (elite/diverse/random)
- prompt modifiers (short constraints appended to existing EOH generator prompts)
- evaluation plan (keep instances fixed unless strongly justified)

Decision principles:
1) Prefer operators with better recent mean_delta and success_rate.
2) Penalize operators with high invalid_rate.
3) If exploration_need high, emphasize e2 and m1.
4) Use e1 rarely.
5) If exploitation_need high, emphasize m2 and some m3.
6) If invalid_risk high, suppress e1 and reduce m1.
7) e2 is often productive; do not starve it.

OUTPUT FORMAT (MANDATORY):
Return ONLY valid JSON with EXACT keys below (no markdown, no extra keys):
{{
  "diagnosis_used": string,
  "op_probs": {{"e1": number, "e2": number, "m1": number, "m2": number, "m3": number}},
  "parent_mix": {{"elite": number, "diverse": number, "random": number}},
  "prompt_modifiers": [string, ...],
  "evaluation_plan": {{"instances": number, "holdout_instances": number}},
  "rationale": [string, ...]
}}

Hard rules:
- op_probs must sum to 1.0
- parent_mix must sum to 1.0
- All probabilities must be in [0,1]
- If invalid_risk >= 0.7, set e1 = 0 and keep m1 small.
- prompt_modifiers should be short and safe (1-4 items).
- instances must equal ObservationPacket.budget.instances unless justified.
- holdout_instances <= ObservationPacket.budget.holdout_instances.

Inputs:
DiagnosisOutput:
{DIAG_JSON}

ObservationPacket:
{OBS_JSON}
"""


CRITIC_PROMPT_TEMPLATE = """{PROJECT_CONTEXT}

ROLE: Agent 3 - CRITIC / SAFETY

Task:
Validate the PlannerOutput and revise it if it violates guardrails or is inconsistent with telemetry.
Return a final plan.

GUARDRAILS (MUST ENFORCE):
1) e1 cap: e1 <= 0.05 always.
2) e1 cooldown: if "e1" appears in last_used_ops within last 3 generations, force e1 = 0.
3) e2 floor: e2 >= 0.25 unless op_stats_k.e2.invalid_rate > 0.40.
4) Invalid spike rule:
   - if invalid_rate_k >= 0.30 or Diagnosis.invalid_risk >= 0.70:
     set e1 = 0
     reduce m1
     increase m2 and/or m3
5) Normalize op_probs and parent_mix.
6) Keep within budget.

QUALITY CHECKS:
- Shift weight away from weak operators to stronger ones (usually e2).
- If diversity_score > 0.7, avoid e1.
- If stagnation is high, prefer e2 before aggressive e1.

OUTPUT FORMAT (MANDATORY):
Return ONLY valid JSON with EXACT keys below (no markdown, no extra keys):
{{
  "verdict": "approve"|"revise",
  "reasons": [string, ...],
  "final_plan": {{
    "op_probs": {{"e1": number, "e2": number, "m1": number, "m2": number, "m3": number}},
    "parent_mix": {{"elite": number, "diverse": number, "random": number}},
    "prompt_modifiers": [string, ...],
    "evaluation_plan": {{"instances": number, "holdout_instances": number}}
  }}
}}

Inputs:
PlannerOutput:
{PLAN_JSON}

DiagnosisOutput:
{DIAG_JSON}

ObservationPacket:
{OBS_JSON}
"""


def _clip01(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _normalize_prob_dict(values, keys):
    cleaned = {}
    for key in keys:
        cleaned[key] = max(0.0, float(values.get(key, 0.0)))
    total = sum(cleaned.values())
    if total <= 1e-12:
        uniform = 1.0 / float(len(keys))
        return {k: uniform for k in keys}
    return {k: cleaned[k] / total for k in keys}


def _json_candidate_blocks(text):
    if not isinstance(text, str):
        return []
    blocks = []
    depth = 0
    start = None
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    blocks.append(text[start:i + 1])
                    start = None
    return blocks


def _try_parse_json_object(text):
    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        return None
    text = text.strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    for block in _json_candidate_blocks(text):
        try:
            obj = json.loads(block)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


class AgenticController:
    def __init__(
        self,
        api_endpoint,
        api_key,
        model_llm,
        llm_use_local,
        llm_local_url,
        debug_mode=False,
    ):
        self.debug_mode = bool(debug_mode)
        self.interface_llm = InterfaceLLM(
            api_endpoint,
            api_key,
            model_llm,
            llm_use_local,
            llm_local_url,
            self.debug_mode,
        )
        self.max_json_retries = int(os.getenv("EOH_CONTROLLER_JSON_RETRIES", "2"))

    def _query_json(self, prompt):
        last_raw = None
        for _ in range(max(1, self.max_json_retries)):
            raw = self.interface_llm.get_response(prompt)
            last_raw = raw
            parsed = _try_parse_json_object(raw)
            if parsed is not None:
                return parsed, raw
        return None, last_raw

    def _fallback_diagnosis(self, observation):
        improve_rate = float(observation.get("improve_rate_k", 0.0) or 0.0)
        stagnation = int(observation.get("stagnation_len", 0) or 0)
        invalid_rate = float(observation.get("invalid_rate_k", 0.0) or 0.0)
        diversity = float(observation.get("diversity_score", 0.5) or 0.5)
        top = observation.get("top_heuristics", []) or []
        complexity = [float(h.get("complexity", 0.0) or 0.0) for h in top if isinstance(h, dict)]
        avg_complexity = (sum(complexity) / float(len(complexity))) if complexity else 0.0

        exploration_need = _clip01(0.55 * min(1.0, stagnation / 5.0) + 0.45 * (1.0 - improve_rate))
        exploitation_need = _clip01(0.7 * improve_rate + 0.3 * max(0.0, 1.0 - invalid_rate))
        diversity_need = _clip01(max(0.0, 0.7 - diversity))
        invalid_risk = _clip01(invalid_rate)
        overfit_risk = _clip01(0.4 * min(1.0, avg_complexity) + 0.6 * min(1.0, stagnation / 6.0))
        confidence = _clip01(0.6)
        labels = []
        if exploration_need >= 0.65:
            labels.append("NEED_EXPLORATION")
        if invalid_risk >= 0.7:
            labels.append("INVALID_SPIKE")
        if overfit_risk >= 0.6:
            labels.append("OVERFIT_RISK")
        if len(labels) == 0:
            labels = ["BALANCED_SEARCH"]
        return {
            "summary": "Fallback diagnosis based on telemetry heuristics.",
            "factors": {
                "exploration_need": exploration_need,
                "exploitation_need": exploitation_need,
                "diversity_need": diversity_need,
                "invalid_risk": invalid_risk,
                "overfit_risk": overfit_risk,
                "confidence": confidence,
            },
            "diagnosis_labels": labels,
            "evidence": [
                f"stagnation_len={stagnation}",
                f"improve_rate_k={improve_rate}",
                f"invalid_rate_k={invalid_rate}",
                f"diversity_score={diversity}",
            ],
        }

    def _sanitize_diagnosis(self, diagnosis, observation):
        if not isinstance(diagnosis, dict):
            diagnosis = self._fallback_diagnosis(observation)
        factors = diagnosis.get("factors", {}) if isinstance(diagnosis.get("factors"), dict) else {}
        diagnosis["factors"] = {
            "exploration_need": _clip01(factors.get("exploration_need", 0.5)),
            "exploitation_need": _clip01(factors.get("exploitation_need", 0.5)),
            "diversity_need": _clip01(factors.get("diversity_need", 0.5)),
            "invalid_risk": _clip01(factors.get("invalid_risk", 0.5)),
            "overfit_risk": _clip01(factors.get("overfit_risk", 0.3)),
            "confidence": _clip01(factors.get("confidence", 0.5)),
        }
        diagnosis["summary"] = str(diagnosis.get("summary", "Diagnosis generated from telemetry."))
        labels = diagnosis.get("diagnosis_labels", [])
        diagnosis["diagnosis_labels"] = [str(x) for x in labels][:5] if isinstance(labels, list) else []
        evidence = diagnosis.get("evidence", [])
        diagnosis["evidence"] = [str(x) for x in evidence][:8] if isinstance(evidence, list) else []
        return diagnosis

    def _fallback_plan(self, diagnosis, observation):
        factors = diagnosis.get("factors", {})
        exploration_need = float(factors.get("exploration_need", 0.5))
        exploitation_need = float(factors.get("exploitation_need", 0.5))
        invalid_risk = float(factors.get("invalid_risk", 0.5))
        diversity_need = float(factors.get("diversity_need", 0.5))
        stagnation = int(observation.get("stagnation_len", 0) or 0)
        budget = observation.get("budget", {}) if isinstance(observation.get("budget"), dict) else {}

        op_probs = {
            "e1": 0.01 if (diversity_need > 0.7 and invalid_risk < 0.7 and stagnation >= 4) else 0.0,
            "e2": 0.45 + 0.25 * exploration_need,
            "m1": 0.1 + 0.15 * exploration_need,
            "m2": 0.2 + 0.25 * exploitation_need,
            "m3": 0.1 + 0.15 * invalid_risk,
        }
        if invalid_risk >= 0.7:
            op_probs["e1"] = 0.0
            op_probs["m1"] = min(op_probs["m1"], 0.08)
            op_probs["m2"] += 0.08
            op_probs["m3"] += 0.08

        parent_mix = {
            "elite": 0.5 + 0.2 * exploitation_need,
            "diverse": 0.25 + 0.2 * exploration_need,
            "random": 0.25 + 0.1 * diversity_need,
        }
        modifiers = [
            "Keep scoring function simple; avoid deep nesting.",
            "Avoid adding many new constants.",
        ]
        if invalid_risk >= 0.7:
            modifiers = [
                "Prioritize validity and numerical stability.",
                "Prefer smooth penalties over hard thresholds.",
            ]
        plan = {
            "diagnosis_used": diagnosis.get("summary", "fallback"),
            "op_probs": op_probs,
            "parent_mix": parent_mix,
            "prompt_modifiers": modifiers,
            "evaluation_plan": {
                "instances": int(budget.get("instances", 0) or 0),
                "holdout_instances": int(budget.get("holdout_instances", 0) or 0),
            },
            "rationale": ["Fallback planner policy from diagnosis factors."],
        }
        return plan

    def _sanitize_plan(self, plan, diagnosis, observation):
        if not isinstance(plan, dict):
            plan = self._fallback_plan(diagnosis, observation)
        budget = observation.get("budget", {}) if isinstance(observation.get("budget"), dict) else {}
        budget_instances = int(budget.get("instances", 0) or 0)
        budget_holdout = int(budget.get("holdout_instances", 0) or 0)

        op_probs = plan.get("op_probs", {}) if isinstance(plan.get("op_probs"), dict) else {}
        parent_mix = plan.get("parent_mix", {}) if isinstance(plan.get("parent_mix"), dict) else {}
        plan["op_probs"] = _normalize_prob_dict(op_probs, ["e1", "e2", "m1", "m2", "m3"])
        plan["parent_mix"] = _normalize_prob_dict(parent_mix, ["elite", "diverse", "random"])

        modifiers = plan.get("prompt_modifiers", [])
        if not isinstance(modifiers, list):
            modifiers = []
        plan["prompt_modifiers"] = [str(x).strip() for x in modifiers if str(x).strip()][:4]

        evaluation_plan = plan.get("evaluation_plan", {}) if isinstance(plan.get("evaluation_plan"), dict) else {}
        instances = int(evaluation_plan.get("instances", budget_instances) or budget_instances)
        holdout_instances = int(evaluation_plan.get("holdout_instances", budget_holdout) or 0)
        if budget_instances > 0:
            instances = max(1, min(instances, budget_instances))
        if budget_holdout >= 0:
            holdout_instances = max(0, min(holdout_instances, budget_holdout))
        plan["evaluation_plan"] = {
            "instances": instances,
            "holdout_instances": holdout_instances,
        }
        plan["diagnosis_used"] = str(plan.get("diagnosis_used", diagnosis.get("summary", "diagnosis")))
        rationale = plan.get("rationale", [])
        plan["rationale"] = [str(x) for x in rationale][:8] if isinstance(rationale, list) else []
        return plan

    def _ensure_guardrails(self, plan, diagnosis, observation):
        plan = self._sanitize_plan(plan, diagnosis, observation)
        reasons = []
        op_probs = dict(plan["op_probs"])
        parent_mix = dict(plan["parent_mix"])
        last_used_ops = observation.get("last_used_ops", [])
        if not isinstance(last_used_ops, list):
            last_used_ops = []
        recent_3 = [str(x) for x in last_used_ops[-3:]]
        op_stats_k = observation.get("op_stats_k", {}) if isinstance(observation.get("op_stats_k"), dict) else {}
        e2_invalid = 0.0
        if isinstance(op_stats_k.get("e2"), dict):
            e2_invalid = float(op_stats_k["e2"].get("invalid_rate", 0.0) or 0.0)
        invalid_rate_k = float(observation.get("invalid_rate_k", 0.0) or 0.0)
        diag_invalid_risk = float(diagnosis.get("factors", {}).get("invalid_risk", 0.0) or 0.0)
        diversity_score = float(observation.get("diversity_score", 0.0) or 0.0)
        stagnation_len = int(observation.get("stagnation_len", 0) or 0)

        if op_probs.get("e1", 0.0) > 0.05:
            op_probs["e1"] = 0.05
            reasons.append("Capped e1 at 0.05.")
        if "e1" in recent_3 and op_probs.get("e1", 0.0) > 0.0:
            op_probs["e1"] = 0.0
            reasons.append("Applied e1 cooldown from last 3 generations.")
        if e2_invalid <= 0.40 and op_probs.get("e2", 0.0) < 0.25:
            op_probs["e2"] = 0.25
            reasons.append("Raised e2 to floor 0.25.")

        if invalid_rate_k >= 0.30 or diag_invalid_risk >= 0.70:
            if op_probs.get("e1", 0.0) > 0.0:
                reasons.append("Disabled e1 due to invalid spike rule.")
            op_probs["e1"] = 0.0
            if op_probs.get("m1", 0.0) > 0.10:
                op_probs["m1"] = 0.10
                reasons.append("Reduced m1 due to invalid spike rule.")
            op_probs["m2"] = op_probs.get("m2", 0.0) + 0.08
            op_probs["m3"] = op_probs.get("m3", 0.0) + 0.08

        if diversity_score > 0.70 and op_probs.get("e1", 0.0) > 0.0:
            op_probs["e1"] = min(op_probs["e1"], 0.02)
            reasons.append("Lowered e1 because diversity is already high.")

        if stagnation_len >= 3 and op_probs.get("e2", 0.0) < max(op_probs.values()):
            op_probs["e2"] = max(op_probs["e2"], 0.35)
            reasons.append("Increased e2 for structured exploration during stagnation.")

        op_probs = _normalize_prob_dict(op_probs, ["e1", "e2", "m1", "m2", "m3"])
        parent_mix = _normalize_prob_dict(parent_mix, ["elite", "diverse", "random"])

        budget = observation.get("budget", {}) if isinstance(observation.get("budget"), dict) else {}
        budget_instances = int(budget.get("instances", 0) or 0)
        budget_holdout = int(budget.get("holdout_instances", 0) or 0)
        instances = int(plan["evaluation_plan"].get("instances", budget_instances) or budget_instances)
        holdout_instances = int(plan["evaluation_plan"].get("holdout_instances", 0) or 0)
        if budget_instances > 0 and instances > budget_instances:
            instances = budget_instances
            reasons.append("Clamped instances to budget.")
        if budget_holdout >= 0 and holdout_instances > budget_holdout:
            holdout_instances = budget_holdout
            reasons.append("Clamped holdout_instances to budget.")

        verdict = "approve" if len(reasons) == 0 else "revise"
        final_plan = {
            "op_probs": op_probs,
            "parent_mix": parent_mix,
            "prompt_modifiers": plan.get("prompt_modifiers", [])[:4],
            "evaluation_plan": {
                "instances": instances,
                "holdout_instances": holdout_instances,
            },
        }
        return verdict, reasons, final_plan

    def _fallback_critic(self, planner_output, diagnosis, observation):
        verdict, reasons, final_plan = self._ensure_guardrails(planner_output, diagnosis, observation)
        return {"verdict": verdict, "reasons": reasons, "final_plan": final_plan}

    def _sanitize_critic(self, critic_output, planner_output, diagnosis, observation):
        if not isinstance(critic_output, dict):
            critic_output = self._fallback_critic(planner_output, diagnosis, observation)
        final_plan = critic_output.get("final_plan", {}) if isinstance(critic_output.get("final_plan"), dict) else {}
        verdict, reasons, guarded_plan = self._ensure_guardrails(final_plan, diagnosis, observation)
        critic_output["verdict"] = str(critic_output.get("verdict", verdict)).lower()
        if critic_output["verdict"] not in ["approve", "revise"]:
            critic_output["verdict"] = verdict
        critic_reasons = critic_output.get("reasons", [])
        if not isinstance(critic_reasons, list):
            critic_reasons = []
        merged_reasons = [str(x) for x in critic_reasons][:8]
        for r in reasons:
            if r not in merged_reasons:
                merged_reasons.append(r)
        critic_output["reasons"] = merged_reasons[:10]
        critic_output["final_plan"] = guarded_plan
        return critic_output

    def _build_prompt(self, template, **kwargs):
        base = kwargs.get("PROJECT_CONTEXT", PROJECT_CONTEXT_BLOCK)
        return template.format(PROJECT_CONTEXT=base, **kwargs)

    def _now(self):
        return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    def run(self, observation):
        observation_json = json.dumps(observation, ensure_ascii=True)

        diagnoser_prompt = self._build_prompt(
            DIAGNOSER_PROMPT_TEMPLATE,
            OBS_JSON=observation_json,
        )
        diagnosis_raw, diagnosis_text = self._query_json(diagnoser_prompt)
        diagnosis = self._sanitize_diagnosis(diagnosis_raw, observation)

        planner_prompt = self._build_prompt(
            PLANNER_PROMPT_TEMPLATE,
            DIAG_JSON=json.dumps(diagnosis, ensure_ascii=True),
            OBS_JSON=observation_json,
        )
        planner_raw, planner_text = self._query_json(planner_prompt)
        planner_output = self._sanitize_plan(planner_raw, diagnosis, observation)

        critic_prompt = self._build_prompt(
            CRITIC_PROMPT_TEMPLATE,
            PLAN_JSON=json.dumps(planner_output, ensure_ascii=True),
            DIAG_JSON=json.dumps(diagnosis, ensure_ascii=True),
            OBS_JSON=observation_json,
        )
        critic_raw, critic_text = self._query_json(critic_prompt)
        critic_output = self._sanitize_critic(critic_raw, planner_output, diagnosis, observation)

        result = {
            "time": self._now(),
            "observation": observation,
            "diagnosis": diagnosis,
            "planner_output": planner_output,
            "critic_output": critic_output,
            "raw": {
                "diagnosis_text": diagnosis_text,
                "planner_text": planner_text,
                "critic_text": critic_text,
            },
        }
        return result

    def sample_operator(self, op_probs, available_ops):
        active = []
        for op in ["e1", "e2", "m1", "m2", "m3"]:
            if op in available_ops:
                w = float(op_probs.get(op, 0.0))
                if w > 0.0:
                    active.append((op, w))
        if len(active) == 0:
            return available_ops[0]
        total = sum(w for _, w in active)
        draw = random.random() * total
        csum = 0.0
        for op, w in active:
            csum += w
            if draw <= csum:
                return op
        return active[-1][0]
