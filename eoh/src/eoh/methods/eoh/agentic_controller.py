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
- e2: Backbone Variant - structured exploration around good ideas
- m1: Structural Modification - moderate structural edits
- m2: Parameter Tuning - local exploitation and refinement
- m3: Simplification/Generalization - simplify to reduce overfitting

Goal of the controller:
Maximize search efficiency and final best fitness by adapting operator usage over time.
"""


DIAGNOSER_PROMPT_TEMPLATE = """OUTPUT JSON ONLY. Do NOT include any text outside JSON.
JSON SCHEMA (EXACT KEYS):
{{
  "summary": "string",
  "factors": {{
    "exploration_need": 0.0,
    "exploitation_need": 0.0,
    "diversity_need": 0.0,
    "invalid_risk": 0.0,
    "overfit_risk": 0.0,
    "confidence": 0.0
  }},
  "diagnosis_labels": ["string", "string"],
  "evidence": ["string", "string"]
}}
VALID EXAMPLE:
{{"summary":"Stagnation and low diversity.","factors":{{"exploration_need":0.72,"exploitation_need":0.31,"diversity_need":0.68,"invalid_risk":0.12,"overfit_risk":0.26,"confidence":0.83}},"diagnosis_labels":["STAGNATION","LOW_DIVERSITY"],"evidence":["stagnation_len=4 and improve_rate_k=0.0 indicate plateau.","diversity_score=0.30 and invalid_rate_k=0.08 support safe exploration.","Top mean_delta operator is e2 from op_stats_k.e2.mean_delta=0.0024."]}}
{PROJECT_CONTEXT}
ROLE: Agent 1 - DIAGNOSER
Task: infer search state from ObservationPacket.
Hard constraints:
- Reference stagnation_len, diversity_score, invalid_rate_k, and top operator by mean_delta in summary/evidence.
- diagnosis_labels length >= 2 and evidence length >= 2.
- Do not propose actions/operators.
ObservationPacket:
{OBS_JSON}
"""


PLANNER_PROMPT_TEMPLATE = """OUTPUT JSON ONLY. Do NOT include any text outside JSON.
JSON SCHEMA (EXACT KEYS):
{{
  "diagnosis_used": "string",
  "op_probs": {{"e1": 0.0, "e2": 0.0, "m1": 0.0, "m2": 0.0, "m3": 0.0}},
  "parent_mix": {{"elite": 0.0, "diverse": 0.0, "random": 0.0}},
  "prompt_modifiers": ["string"],
  "evaluation_plan": {{"instances": 0, "holdout_instances": 0}},
  "rationale": ["string", "string"]
}}
VALID EXAMPLE:
{{"diagnosis_used":"Stagnation with low invalid risk.","op_probs":{{"e1":0.02,"e2":0.56,"m1":0.16,"m2":0.18,"m3":0.08}},"parent_mix":{{"elite":0.48,"diverse":0.32,"random":0.20}},"prompt_modifiers":["Keep scoring function simple.","Avoid many new constants."],"evaluation_plan":{{"instances":256,"holdout_instances":0}},"rationale":["stagnation_len=4 and improve_rate_k=0.0 justify structured exploration.","op_stats_k.e2.mean_delta is highest while invalid_rate_k remains low."]}}
{PROJECT_CONTEXT}
ROLE: Agent 2 - PLANNER
Task: build next-generation action plan from DiagnosisOutput + ObservationPacket.
Hard constraints:
- op_probs and parent_mix each sum to exactly 1.0.
- Uniform op_probs are banned; require spread >= 0.15.
- Largest op_probs entry must be highest recent mean_delta operator unless invalid_rate > 0.40.
- rationale must cite at least two telemetry fields.
- prompt_modifiers size 1-4.
DiagnosisOutput:
{DIAG_JSON}
ObservationPacket:
{OBS_JSON}
"""


CRITIC_PROMPT_TEMPLATE = """OUTPUT JSON ONLY. Do NOT include any text outside JSON.
JSON SCHEMA (EXACT KEYS):
{{
  "verdict": "approve",
  "reasons": ["string"],
  "final_plan": {{
    "op_probs": {{"e1": 0.0, "e2": 0.0, "m1": 0.0, "m2": 0.0, "m3": 0.0}},
    "parent_mix": {{"elite": 0.0, "diverse": 0.0, "random": 0.0}},
    "prompt_modifiers": ["string"],
    "evaluation_plan": {{"instances": 0, "holdout_instances": 0}}
  }}
}}
VALID EXAMPLE:
{{"verdict":"revise","reasons":["Applied e1 cooldown.","Raised e2 to floor 0.25."],"final_plan":{{"op_probs":{{"e1":0.00,"e2":0.48,"m1":0.12,"m2":0.24,"m3":0.16}},"parent_mix":{{"elite":0.50,"diverse":0.30,"random":0.20}},"prompt_modifiers":["Keep logic shallow."],"evaluation_plan":{{"instances":256,"holdout_instances":0}}}}}}
{PROJECT_CONTEXT}
ROLE: Agent 3 - CRITIC / SAFETY

Task:
Validate PlannerOutput and revise with guardrails.

Guardrails (must enforce):
1) e1 <= 0.05.
2) If "e1" in last 3 last_used_ops, set e1=0.
3) e2 >= 0.25 unless op_stats_k.e2.invalid_rate > 0.40.
4) If invalid_rate_k >= 0.30 or Diagnosis.invalid_risk >= 0.70:
   - set e1=0
   - reduce m1
   - increase m2 and/or m3
5) Normalize op_probs and parent_mix.
6) Keep evaluation_plan inside budget.

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


def _to_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _sum_close_to_one(values, tol=1e-3):
    return abs(sum(values) - 1.0) <= tol


def _normalize_prob_dict(values, keys):
    cleaned = {}
    for key in keys:
        cleaned[key] = max(0.0, _to_float(values.get(key), 0.0))
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
                    blocks.append(text[start : i + 1])
                    start = None
    return blocks


def _try_parse_json_object(text):
    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    if not stripped:
        return None
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    for block in _json_candidate_blocks(stripped):
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
        self.max_json_retries = max(1, int(os.getenv("EOH_CONTROLLER_JSON_RETRIES", "3")))

    def _now(self):
        return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    def _build_prompt(self, template, **kwargs):
        return template.format(PROJECT_CONTEXT=PROJECT_CONTEXT_BLOCK, **kwargs)

    def _top_operator_by_mean_delta(self, observation):
        op_stats = observation.get("op_stats_k", {}) if isinstance(observation.get("op_stats_k"), dict) else {}
        best_op = None
        best_delta = None
        for op in ["e1", "e2", "m1", "m2", "m3"]:
            s = op_stats.get(op, {}) if isinstance(op_stats.get(op), dict) else {}
            d = _to_float(s.get("mean_delta"), None)
            if d is None:
                continue
            if best_delta is None or d > best_delta:
                best_delta = d
                best_op = op
        return best_op

    def _best_valid_operator_by_delta(self, observation):
        op_stats = observation.get("op_stats_k", {}) if isinstance(observation.get("op_stats_k"), dict) else {}
        best_op = None
        best_delta = None
        for op in ["e1", "e2", "m1", "m2", "m3"]:
            s = op_stats.get(op, {}) if isinstance(op_stats.get(op), dict) else {}
            inv = _to_float(s.get("invalid_rate"), None)
            d = _to_float(s.get("mean_delta"), None)
            if d is None:
                continue
            if inv is not None and inv > 0.40:
                continue
            if best_delta is None or d > best_delta:
                best_delta = d
                best_op = op
        return best_op

    def _validate_diagnosis_output(self, obj, observation):
        errors = []
        if not isinstance(obj, dict):
            return ["output is not a JSON object"]
        required = ["summary", "factors", "diagnosis_labels", "evidence"]
        for key in required:
            if key not in obj:
                errors.append(f"missing key: {key}")

        factors = obj.get("factors")
        if not isinstance(factors, dict):
            errors.append("factors must be object")
        else:
            for key in [
                "exploration_need",
                "exploitation_need",
                "diversity_need",
                "invalid_risk",
                "overfit_risk",
                "confidence",
            ]:
                if key not in factors:
                    errors.append(f"factors missing: {key}")
                else:
                    v = _to_float(factors.get(key), None)
                    if v is None:
                        errors.append(f"factor {key} not numeric")
                    elif v < 0.0 or v > 1.0:
                        errors.append(f"factor {key} out of [0,1]")

        labels = obj.get("diagnosis_labels")
        if not isinstance(labels, list):
            errors.append("diagnosis_labels must be list")
        elif len(labels) < 2:
            errors.append("diagnosis_labels must contain at least 2 items")

        evidence = obj.get("evidence")
        if not isinstance(evidence, list):
            errors.append("evidence must be list")
        elif len(evidence) < 2:
            errors.append("evidence must contain at least 2 items")

        text = (str(obj.get("summary", "")) + " " + " ".join(str(x) for x in (evidence if isinstance(evidence, list) else []))).lower()
        for field_name in ["stagnation_len", "diversity_score", "invalid_rate_k"]:
            if field_name.lower() not in text:
                errors.append(f"missing telemetry reference: {field_name}")

        top_op = self._top_operator_by_mean_delta(observation)
        if top_op is not None:
            if top_op.lower() not in text and f"op_stats_k.{top_op}.mean_delta".lower() not in text:
                errors.append(f"missing top mean_delta operator reference: {top_op}")
        return errors

    def _validate_planner_output(self, obj, diagnosis, observation):
        errors = []
        if not isinstance(obj, dict):
            return ["output is not a JSON object"]
        required = ["diagnosis_used", "op_probs", "parent_mix", "prompt_modifiers", "evaluation_plan", "rationale"]
        for key in required:
            if key not in obj:
                errors.append(f"missing key: {key}")

        op_probs = obj.get("op_probs")
        if not isinstance(op_probs, dict):
            errors.append("op_probs must be object")
        else:
            vals = []
            for op in ["e1", "e2", "m1", "m2", "m3"]:
                v = _to_float(op_probs.get(op), None)
                if v is None:
                    errors.append(f"op_probs missing/invalid: {op}")
                    vals.append(0.0)
                else:
                    if v < 0.0 or v > 1.0:
                        errors.append(f"op_probs[{op}] out of [0,1]")
                    vals.append(v)
            if not _sum_close_to_one(vals):
                errors.append("op_probs must sum to 1")
            if max(vals) - min(vals) < 0.15:
                errors.append("op_probs must be non-uniform (spread >= 0.15)")

            preferred = self._best_valid_operator_by_delta(observation)
            if preferred is not None:
                argmax_op = max(["e1", "e2", "m1", "m2", "m3"], key=lambda k: _to_float(op_probs.get(k), 0.0))
                if argmax_op != preferred:
                    errors.append(
                        f"largest op prob must target highest mean_delta valid operator ({preferred}), got {argmax_op}"
                    )

        parent_mix = obj.get("parent_mix")
        if not isinstance(parent_mix, dict):
            errors.append("parent_mix must be object")
        else:
            vals = []
            for key in ["elite", "diverse", "random"]:
                v = _to_float(parent_mix.get(key), None)
                if v is None:
                    errors.append(f"parent_mix missing/invalid: {key}")
                    vals.append(0.0)
                else:
                    if v < 0.0 or v > 1.0:
                        errors.append(f"parent_mix[{key}] out of [0,1]")
                    vals.append(v)
            if not _sum_close_to_one(vals):
                errors.append("parent_mix must sum to 1")

        prompt_modifiers = obj.get("prompt_modifiers")
        if not isinstance(prompt_modifiers, list) or len(prompt_modifiers) == 0:
            errors.append("prompt_modifiers must be non-empty list")
        elif len(prompt_modifiers) > 4:
            errors.append("prompt_modifiers max length is 4")

        evaluation_plan = obj.get("evaluation_plan")
        if not isinstance(evaluation_plan, dict):
            errors.append("evaluation_plan must be object")
        else:
            budget = observation.get("budget", {}) if isinstance(observation.get("budget"), dict) else {}
            budget_instances = int(budget.get("instances", 0) or 0)
            budget_holdout = int(budget.get("holdout_instances", 0) or 0)
            inst = evaluation_plan.get("instances")
            hold = evaluation_plan.get("holdout_instances")
            if _to_float(inst, None) is None:
                errors.append("evaluation_plan.instances must be numeric")
            if _to_float(hold, None) is None:
                errors.append("evaluation_plan.holdout_instances must be numeric")
            if _to_float(inst, 0) > budget_instances > 0:
                errors.append("evaluation_plan.instances exceeds budget.instances")
            if _to_float(hold, 0) > budget_holdout >= 0:
                errors.append("evaluation_plan.holdout_instances exceeds budget.holdout_instances")

        rationale = obj.get("rationale")
        if not isinstance(rationale, list) or len(rationale) < 2:
            errors.append("rationale must have at least 2 lines")
        else:
            text = " ".join(str(x) for x in rationale).lower()
            citation_tokens = [
                "stagnation_len",
                "invalid_rate_k",
                "diversity_score",
                "improve_rate_k",
                "delta_best",
                "op_stats_k.e1.mean_delta",
                "op_stats_k.e2.mean_delta",
                "op_stats_k.m1.mean_delta",
                "op_stats_k.m2.mean_delta",
                "op_stats_k.m3.mean_delta",
            ]
            cited = sum(1 for token in citation_tokens if token in text)
            if cited < 2:
                errors.append("rationale must cite at least 2 telemetry fields")

        invalid_risk = _to_float((diagnosis.get("factors", {}) if isinstance(diagnosis, dict) else {}).get("invalid_risk"), None)
        if invalid_risk is not None and invalid_risk >= 0.70 and isinstance(op_probs, dict):
            e1 = _to_float(op_probs.get("e1"), 0.0)
            m1 = _to_float(op_probs.get("m1"), 1.0)
            if e1 > 1e-6:
                errors.append("invalid_risk>=0.7 requires e1=0")
            if m1 > 0.15:
                errors.append("invalid_risk>=0.7 requires small m1")
        return errors

    def _validate_critic_output(self, obj, _planner_output, _diagnosis, _observation):
        errors = []
        if not isinstance(obj, dict):
            return ["output is not a JSON object"]
        if obj.get("verdict") not in ["approve", "revise"]:
            errors.append("verdict must be approve|revise")
        final_plan = obj.get("final_plan")
        if not isinstance(final_plan, dict):
            errors.append("final_plan must be object")
        return errors

    def _resolve_llm_mode(self, mode, tool_name=None):
        if mode == "tool" and tool_name:
            return "tool_call"
        if mode == "json":
            return "json_mode"
        return "plain_prompt"

    def _build_retry_prompt(self, prompt, corrective_schema_hint, errors):
        error_text = "\n".join(f"- {e}" for e in (errors or [])[:8]) or "- unknown_error"
        return (
            prompt
            + "\n\nYour previous output was invalid.\n"
            + "Output ONLY valid JSON matching the schema.\n"
            + f"Schema reminder: {corrective_schema_hint}\n"
            + "Fix these errors:\n"
            + error_text
        )

    def _build_repair_prompt(self, schema_name, corrective_schema_hint, raw_output, errors):
        error_text = "\n".join(f"- {e}" for e in (errors or [])[:10]) or "- invalid_json_or_schema"
        return (
            "OUTPUT JSON ONLY. Do NOT include any text outside JSON.\n"
            + f"Target schema ({schema_name}): {corrective_schema_hint}\n"
            + "Fix these errors exactly:\n"
            + error_text
            + "\nRaw output to repair:\n"
            + "<<<RAW_OUTPUT>>>\n"
            + str(raw_output if raw_output is not None else "")
            + "\n<<<END_RAW_OUTPUT>>>\n"
            + "Return ONLY corrected JSON object."
        )

    def _attempt_llm_json(self, prompt, validator, mode, tool_name, corrective_schema_hint, phase, cycle):
        raw = None
        parsed = None
        errors = []
        api_error = None
        parse_ok = False
        validation_ok = False
        try:
            raw = self.interface_llm.get_response(
                prompt,
                request_mode=mode,
                tool_name=tool_name,
                json_schema=None,
                stop=None,
            )
        except Exception as exc:
            api_error = f"api_error: {exc}"
            errors.append(api_error)

        if api_error is None:
            parsed = _try_parse_json_object(raw)
            parse_ok = parsed is not None
            if not parse_ok:
                errors.append("invalid_json_or_not_object")
            else:
                try:
                    errors.extend(validator(parsed))
                except Exception as exc:
                    errors.append(f"validator_exception: {exc}")
                validation_ok = len(errors) == 0

        return {
            "attempt": int(cycle),
            "phase": str(phase),
            "llm_mode": self._resolve_llm_mode(mode, tool_name),
            "request_mode": mode,
            "tool_name": tool_name,
            "schema_hint": corrective_schema_hint,
            "raw_output": raw,
            "parse_ok": bool(parse_ok),
            "validation_ok": bool(validation_ok),
            "errors": list(errors),
            "api_error": api_error,
            "parsed_object": parsed,
        }

    def call_llm_json(
        self,
        prompt,
        schema_name,
        validator,
        corrective_schema_hint,
        mode="json",
        tool_name=None,
    ):
        attempts = []
        current_prompt = prompt
        repair_used = False
        llm_mode = self._resolve_llm_mode(mode, tool_name)

        for cycle in range(1, self.max_json_retries + 1):
            primary = self._attempt_llm_json(
                current_prompt,
                validator,
                mode,
                tool_name,
                corrective_schema_hint,
                phase="primary",
                cycle=cycle,
            )
            attempts.append({k: v for k, v in primary.items() if k != "parsed_object"})
            if primary["parse_ok"] and primary["validation_ok"] and primary["parsed_object"] is not None:
                return primary["parsed_object"], {
                    "success": True,
                    "llm_success": True,
                    "schema": schema_name,
                    "llm_mode": llm_mode,
                    "attempts": attempts,
                    "retries_used": max(0, len(attempts) - 1),
                    "repair_used": repair_used,
                    "parse_ok": True,
                    "validation_ok": True,
                    "failure_reason": "",
                    "last_errors": [],
                }

            repair_used = True
            repair_prompt = self._build_repair_prompt(
                schema_name=schema_name,
                corrective_schema_hint=corrective_schema_hint,
                raw_output=primary.get("raw_output"),
                errors=primary.get("errors"),
            )
            repair = self._attempt_llm_json(
                repair_prompt,
                validator,
                mode,
                tool_name,
                corrective_schema_hint,
                phase="repair",
                cycle=cycle,
            )
            attempts.append({k: v for k, v in repair.items() if k != "parsed_object"})
            if repair["parse_ok"] and repair["validation_ok"] and repair["parsed_object"] is not None:
                return repair["parsed_object"], {
                    "success": True,
                    "llm_success": True,
                    "schema": schema_name,
                    "llm_mode": llm_mode,
                    "attempts": attempts,
                    "retries_used": max(0, len(attempts) - 1),
                    "repair_used": True,
                    "parse_ok": True,
                    "validation_ok": True,
                    "failure_reason": "",
                    "last_errors": [],
                }

            current_prompt = self._build_retry_prompt(
                prompt=prompt,
                corrective_schema_hint=corrective_schema_hint,
                errors=repair.get("errors") or primary.get("errors"),
            )

        last_errors = attempts[-1]["errors"] if attempts else ["no_attempts"]
        failure_reason = str(last_errors[0]) if len(last_errors) > 0 else "unknown_failure"
        return None, {
            "success": False,
            "llm_success": False,
            "schema": schema_name,
            "llm_mode": llm_mode,
            "attempts": attempts,
            "retries_used": max(0, len(attempts) - 1),
            "repair_used": repair_used,
            "parse_ok": bool(attempts[-1]["parse_ok"]) if attempts else False,
            "validation_ok": bool(attempts[-1]["validation_ok"]) if attempts else False,
            "failure_reason": failure_reason,
            "last_errors": last_errors,
        }

    def _fallback_diagnosis(self, observation):
        improve_rate = _to_float(observation.get("improve_rate_k"), 0.0) or 0.0
        stagnation = int(observation.get("stagnation_len", 0) or 0)
        invalid_rate = _to_float(observation.get("invalid_rate_k"), 0.0) or 0.0
        diversity = _to_float(observation.get("diversity_score"), 0.5) or 0.5
        top = observation.get("top_heuristics", []) or []
        complexity = [_to_float(h.get("complexity"), 0.0) for h in top if isinstance(h, dict)]
        complexity = [x for x in complexity if x is not None]
        avg_complexity = (sum(complexity) / float(len(complexity))) if complexity else 0.0

        exploration_need = _clip01(0.55 * min(1.0, stagnation / 5.0) + 0.45 * (1.0 - improve_rate))
        exploitation_need = _clip01(0.7 * improve_rate + 0.3 * max(0.0, 1.0 - invalid_rate))
        diversity_need = _clip01(max(0.0, 0.7 - diversity))
        invalid_risk = _clip01(invalid_rate)
        overfit_risk = _clip01(0.4 * min(1.0, avg_complexity) + 0.6 * min(1.0, stagnation / 6.0))
        top_op = self._top_operator_by_mean_delta(observation) or "e2"
        return {
            "summary": "Fallback diagnosis based on telemetry heuristics.",
            "factors": {
                "exploration_need": exploration_need,
                "exploitation_need": exploitation_need,
                "diversity_need": diversity_need,
                "invalid_risk": invalid_risk,
                "overfit_risk": overfit_risk,
                "confidence": 0.55,
            },
            "diagnosis_labels": ["TELEMETRY_FALLBACK", "NEED_EXPLORATION" if exploration_need >= 0.5 else "NEED_EXPLOITATION"],
            "evidence": [
                f"stagnation_len={stagnation}",
                f"diversity_score={diversity}",
                f"invalid_rate_k={invalid_rate}",
                f"top_op_by_mean_delta={top_op}",
            ],
        }

    def _fallback_plan(self, diagnosis, observation):
        factors = diagnosis.get("factors", {}) if isinstance(diagnosis, dict) else {}
        exploration_need = _to_float(factors.get("exploration_need"), 0.5) or 0.5
        exploitation_need = _to_float(factors.get("exploitation_need"), 0.5) or 0.5
        invalid_risk = _to_float(factors.get("invalid_risk"), 0.5) or 0.5
        diversity_need = _to_float(factors.get("diversity_need"), 0.5) or 0.5
        budget = observation.get("budget", {}) if isinstance(observation.get("budget"), dict) else {}

        op_probs = {
            "e1": 0.0 if invalid_risk >= 0.70 else (0.01 if diversity_need > 0.70 else 0.0),
            "e2": 0.45 + 0.25 * exploration_need,
            "m1": 0.10 + 0.15 * exploration_need,
            "m2": 0.20 + 0.25 * exploitation_need,
            "m3": 0.10 + 0.15 * invalid_risk,
        }
        if invalid_risk >= 0.70:
            op_probs["m1"] = min(op_probs["m1"], 0.08)
            op_probs["m2"] += 0.08
            op_probs["m3"] += 0.08

        return {
            "diagnosis_used": diagnosis.get("summary", "fallback"),
            "op_probs": op_probs,
            "parent_mix": {
                "elite": 0.50 + 0.20 * exploitation_need,
                "diverse": 0.25 + 0.20 * exploration_need,
                "random": 0.25 + 0.10 * diversity_need,
            },
            "prompt_modifiers": [
                "Keep scoring function simple; avoid deep nesting.",
                "Avoid adding many new constants.",
            ],
            "evaluation_plan": {
                "instances": int(budget.get("instances", 0) or 0),
                "holdout_instances": int(budget.get("holdout_instances", 0) or 0),
            },
            "rationale": [
                "Fallback planner policy from diagnosis factors.",
                "Uses stagnation_len and invalid_rate_k to bias e2/m2/m3.",
            ],
        }

    def _sanitize_diagnosis(self, diagnosis, observation):
        patches = []
        fallback_used = False
        source_obj = diagnosis
        if not isinstance(source_obj, dict):
            source_obj = self._fallback_diagnosis(observation)
            fallback_used = True
            patches.append("fallback_applied_non_dict")

        factors = source_obj.get("factors", {}) if isinstance(source_obj.get("factors"), dict) else {}
        if not isinstance(source_obj.get("factors"), dict):
            patches.append("missing_factors_object")
        out = {
            "summary": str(source_obj.get("summary", "Diagnosis generated from telemetry.")),
            "factors": {
                "exploration_need": _clip01(factors.get("exploration_need", 0.5)),
                "exploitation_need": _clip01(factors.get("exploitation_need", 0.5)),
                "diversity_need": _clip01(factors.get("diversity_need", 0.5)),
                "invalid_risk": _clip01(factors.get("invalid_risk", 0.5)),
                "overfit_risk": _clip01(factors.get("overfit_risk", 0.3)),
                "confidence": _clip01(factors.get("confidence", 0.5)),
            },
            "diagnosis_labels": [str(x) for x in source_obj.get("diagnosis_labels", [])][:6]
            if isinstance(source_obj.get("diagnosis_labels"), list)
            else [],
            "evidence": [str(x) for x in source_obj.get("evidence", [])][:10]
            if isinstance(source_obj.get("evidence"), list)
            else [],
        }
        if len(out["diagnosis_labels"]) < 2:
            fallback = self._fallback_diagnosis(observation)
            out["diagnosis_labels"] = fallback["diagnosis_labels"]
            patches.append("diagnosis_labels_too_short")
        if len(out["evidence"]) < 2:
            fallback = self._fallback_diagnosis(observation)
            out["evidence"] = fallback["evidence"]
            patches.append("evidence_too_short")
        return out, {"fallback_used": fallback_used, "patched": len(patches) > 0, "patches": patches}

    def _sanitize_plan(self, plan, diagnosis, observation):
        patches = []
        fallback_used = False
        source_obj = plan
        if not isinstance(source_obj, dict):
            source_obj = self._fallback_plan(diagnosis, observation)
            fallback_used = True
            patches.append("fallback_applied_non_dict")

        budget = observation.get("budget", {}) if isinstance(observation.get("budget"), dict) else {}
        budget_instances = int(budget.get("instances", 0) or 0)
        budget_holdout = int(budget.get("holdout_instances", 0) or 0)

        op_probs = source_obj.get("op_probs", {}) if isinstance(source_obj.get("op_probs"), dict) else {}
        parent_mix = source_obj.get("parent_mix", {}) if isinstance(source_obj.get("parent_mix"), dict) else {}
        if not isinstance(source_obj.get("op_probs"), dict):
            patches.append("missing_op_probs_object")
        if not isinstance(source_obj.get("parent_mix"), dict):
            patches.append("missing_parent_mix_object")

        op_probs_out = _normalize_prob_dict(op_probs, ["e1", "e2", "m1", "m2", "m3"])
        parent_mix_out = _normalize_prob_dict(parent_mix, ["elite", "diverse", "random"])

        vals = [op_probs_out[k] for k in ["e1", "e2", "m1", "m2", "m3"]]
        if (max(vals) - min(vals)) < 0.15:
            fallback = self._fallback_plan(diagnosis, observation)
            op_probs_out = _normalize_prob_dict(fallback["op_probs"], ["e1", "e2", "m1", "m2", "m3"])
            patches.append("uniform_op_probs_replaced")

        prompt_modifiers = source_obj.get("prompt_modifiers", [])
        if not isinstance(prompt_modifiers, list):
            prompt_modifiers = []
            patches.append("prompt_modifiers_not_list")
        prompt_modifiers = [str(x).strip() for x in prompt_modifiers if str(x).strip()][:4]
        if len(prompt_modifiers) == 0:
            prompt_modifiers = ["Keep scoring function simple; avoid deep nesting."]
            patches.append("prompt_modifiers_empty_defaulted")

        eval_plan = source_obj.get("evaluation_plan", {}) if isinstance(source_obj.get("evaluation_plan"), dict) else {}
        if not isinstance(source_obj.get("evaluation_plan"), dict):
            patches.append("missing_evaluation_plan_object")
        instances = int(_to_float(eval_plan.get("instances"), budget_instances) or budget_instances)
        holdout_instances = int(_to_float(eval_plan.get("holdout_instances"), budget_holdout) or 0)
        if budget_instances > 0:
            instances = max(1, min(instances, budget_instances))
        if budget_holdout >= 0:
            holdout_instances = max(0, min(holdout_instances, budget_holdout))

        rationale = source_obj.get("rationale", [])
        if not isinstance(rationale, list):
            rationale = []
            patches.append("rationale_not_list")
        rationale = [str(x) for x in rationale][:8]
        if len(rationale) < 2:
            rationale.extend(
                [
                    "Reference: stagnation_len and invalid_rate_k from ObservationPacket.",
                    "Reference: op_stats_k.e2.mean_delta from ObservationPacket.",
                ]
            )
            rationale = rationale[:8]
            patches.append("rationale_too_short_defaulted")

        out = {
            "diagnosis_used": str(source_obj.get("diagnosis_used", diagnosis.get("summary", "diagnosis"))),
            "op_probs": op_probs_out,
            "parent_mix": parent_mix_out,
            "prompt_modifiers": prompt_modifiers,
            "evaluation_plan": {
                "instances": instances,
                "holdout_instances": holdout_instances,
            },
            "rationale": rationale,
        }
        return out, {"fallback_used": fallback_used, "patched": len(patches) > 0, "patches": patches}

    def _ensure_guardrails(self, plan, diagnosis, observation):
        plan, sanitize_meta = self._sanitize_plan(plan, diagnosis, observation)
        reasons = []
        op_probs = dict(plan["op_probs"])
        parent_mix = dict(plan["parent_mix"])
        last_used_ops = observation.get("last_used_ops", []) if isinstance(observation.get("last_used_ops"), list) else []
        recent_3 = [str(x) for x in last_used_ops[-3:]]
        op_stats_k = observation.get("op_stats_k", {}) if isinstance(observation.get("op_stats_k"), dict) else {}
        e2_invalid = 0.0
        if isinstance(op_stats_k.get("e2"), dict):
            e2_invalid = _to_float(op_stats_k["e2"].get("invalid_rate"), 0.0) or 0.0
        invalid_rate_k = _to_float(observation.get("invalid_rate_k"), 0.0) or 0.0
        diag_invalid_risk = _to_float((diagnosis.get("factors", {}) if isinstance(diagnosis, dict) else {}).get("invalid_risk"), 0.0) or 0.0
        diversity_score = _to_float(observation.get("diversity_score"), 0.0) or 0.0
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
        meta = {
            "sanitize_meta": sanitize_meta,
            "guardrail_reasons": list(reasons),
            "guardrails_applied": len(reasons) > 0,
        }
        return verdict, reasons, final_plan, meta

    def _classify_stage(self, llm_meta, sanitize_meta):
        fully_fallback = bool(sanitize_meta.get("fallback_used", False))
        llm_patched = bool(sanitize_meta.get("patched", False) and not fully_fallback)
        pure_llm = bool((not fully_fallback) and (not llm_patched) and llm_meta.get("success", False))
        return {
            "pure_llm_output": pure_llm,
            "llm_output_patched": llm_patched,
            "fully_fallback": fully_fallback,
        }

    def run(self, observation):
        observation_json = json.dumps(observation, ensure_ascii=True)

        diagnoser_prompt = self._build_prompt(DIAGNOSER_PROMPT_TEMPLATE, OBS_JSON=observation_json)
        diagnosis_raw, diagnosis_llm_meta = self.call_llm_json(
            prompt=diagnoser_prompt,
            schema_name="diagnosis",
            validator=lambda obj: self._validate_diagnosis_output(obj, observation),
            corrective_schema_hint='{"summary":str,"factors":{...},"diagnosis_labels":[...],"evidence":[...]}',
            mode="json",
            tool_name="diagnose_state",
        )
        diagnosis, diagnosis_sanitize_meta = self._sanitize_diagnosis(diagnosis_raw, observation)
        diagnosis_flags = self._classify_stage(diagnosis_llm_meta, diagnosis_sanitize_meta)

        planner_prompt = self._build_prompt(
            PLANNER_PROMPT_TEMPLATE,
            DIAG_JSON=json.dumps(diagnosis, ensure_ascii=True),
            OBS_JSON=observation_json,
        )
        planner_raw, planner_llm_meta = self.call_llm_json(
            prompt=planner_prompt,
            schema_name="planner",
            validator=lambda obj: self._validate_planner_output(obj, diagnosis, observation),
            corrective_schema_hint='{"diagnosis_used":str,"op_probs":{"e1":..},"parent_mix":{"elite":..},"prompt_modifiers":[..],"evaluation_plan":{"instances":..},"rationale":[..]}',
            mode="json",
            tool_name="plan_action",
        )
        planner_output, planner_sanitize_meta = self._sanitize_plan(planner_raw, diagnosis, observation)
        planner_flags = self._classify_stage(planner_llm_meta, planner_sanitize_meta)

        critic_prompt = self._build_prompt(
            CRITIC_PROMPT_TEMPLATE,
            PLAN_JSON=json.dumps(planner_output, ensure_ascii=True),
            DIAG_JSON=json.dumps(diagnosis, ensure_ascii=True),
            OBS_JSON=observation_json,
        )
        critic_raw, critic_llm_meta = self.call_llm_json(
            prompt=critic_prompt,
            schema_name="critic",
            validator=lambda obj: self._validate_critic_output(obj, planner_output, diagnosis, observation),
            corrective_schema_hint='{"verdict":"approve|revise","reasons":[...],"final_plan":{"op_probs":{...},"parent_mix":{...}}}',
            mode="json",
            tool_name="critic_plan",
        )
        if not isinstance(critic_raw, dict):
            critic_raw = {"verdict": "revise", "reasons": ["critic_fallback_non_dict"], "final_plan": planner_output}
        verdict, reasons, final_plan, critic_guardrail_meta = self._ensure_guardrails(
            critic_raw.get("final_plan", {}),
            diagnosis,
            observation,
        )
        critic_output = {
            "verdict": str(critic_raw.get("verdict", verdict)).lower() if str(critic_raw.get("verdict", verdict)).lower() in ["approve", "revise"] else verdict,
            "reasons": [str(x) for x in (critic_raw.get("reasons", []) if isinstance(critic_raw.get("reasons"), list) else [])][:10],
            "final_plan": final_plan,
        }
        for r in reasons:
            if r not in critic_output["reasons"]:
                critic_output["reasons"].append(r)
        critic_output["reasons"] = critic_output["reasons"][:12]
        critic_sanitize_meta = {
            "fallback_used": False,
            "patched": bool(critic_guardrail_meta.get("guardrails_applied", False)),
            "patches": critic_guardrail_meta.get("guardrail_reasons", []),
        }
        critic_flags = self._classify_stage(critic_llm_meta, critic_sanitize_meta)

        result = {
            "time": self._now(),
            "observation": observation,
            "diagnosis": diagnosis,
            "planner_output": planner_output,
            "critic_output": critic_output,
            "raw": {
                "diagnosis_text": diagnosis_llm_meta["attempts"][-1]["raw_output"] if diagnosis_llm_meta.get("attempts") else None,
                "planner_text": planner_llm_meta["attempts"][-1]["raw_output"] if planner_llm_meta.get("attempts") else None,
                "critic_text": critic_llm_meta["attempts"][-1]["raw_output"] if critic_llm_meta.get("attempts") else None,
            },
            "debug": {
                "diagnoser": {
                    "llm": diagnosis_llm_meta,
                    "sanitize": diagnosis_sanitize_meta,
                    "flags": diagnosis_flags,
                },
                "planner": {
                    "llm": planner_llm_meta,
                    "sanitize": planner_sanitize_meta,
                    "flags": planner_flags,
                },
                "critic": {
                    "llm": critic_llm_meta,
                    "sanitize": critic_sanitize_meta,
                    "flags": critic_flags,
                    "guardrail_meta": critic_guardrail_meta,
                },
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
