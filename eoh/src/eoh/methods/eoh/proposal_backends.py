import json
import re


def _extract_first_json_object(text):
    if not isinstance(text, str):
        return None

    fenced = re.findall(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        for chunk in fenced:
            try:
                return json.loads(chunk)
            except Exception:
                pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except Exception:
        return None


class EOHProposalBackend:
    mode = "eoh"

    def build_prompt(self, operator, base_prompt, context):
        return base_prompt

    def parse_response(self, response):
        # EOH mode relies on legacy regex parsing in Evolution.
        return None


class DXProposalBackend:
    mode = "dx"

    def build_prompt(self, operator, base_prompt, context):
        schema = {
            "diagnosis": {
                "label": "string",
                "confidence": 0.0,
                "evidence": ["string"],
            },
            "plan": {
                "goal": "string",
                "actions": [
                    {
                        "name": "string",
                        "args": {},
                        "expected_effect": "string",
                        "risk": "string",
                    }
                ],
            },
            "rollback": {
                "condition": "string",
                "fallback_action": {"name": "noop", "args": {}},
            },
            "proposal": {
                "algorithm": "one-sentence description",
                "code": "python function text",
            },
        }
        dx_header = (
            "You are a diagnosis-driven heuristic editor.\n"
            "Use provided trace statistics and history as evidence.\n"
            "Return valid JSON only, no markdown.\n"
            "Do not change objective definition.\n"
            "Use 1-2 actions only.\n"
            "Include proposal.algorithm and proposal.code so the code can be executed."
        )
        ctx_text = ""
        if isinstance(context, dict) and len(context) > 0:
            safe_context = {
                "operator": context.get("operator"),
                "parent_trace_summary": context.get("parent_trace_summary"),
                "recent_history": context.get("recent_history"),
            }
            ctx_text = "\n\nDX context (JSON):\n" + json.dumps(safe_context, ensure_ascii=True)
        schema_text = "\n\nRequired JSON schema:\n" + json.dumps(schema, ensure_ascii=True)
        return dx_header + ctx_text + schema_text + "\n\n" + base_prompt

    def _validate_payload(self, payload):
        if not isinstance(payload, dict):
            return False, "payload is not an object"

        diagnosis = payload.get("diagnosis")
        plan = payload.get("plan")
        rollback = payload.get("rollback")
        proposal = payload.get("proposal")

        if not isinstance(diagnosis, dict):
            return False, "missing diagnosis object"
        if not isinstance(plan, dict):
            return False, "missing plan object"
        if not isinstance(rollback, dict):
            return False, "missing rollback object"
        if not isinstance(proposal, dict):
            return False, "missing proposal object"

        label = diagnosis.get("label")
        confidence = diagnosis.get("confidence")
        evidence = diagnosis.get("evidence")
        if not isinstance(label, str) or len(label.strip()) == 0:
            return False, "invalid diagnosis.label"
        if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
            return False, "invalid diagnosis.confidence"
        if not isinstance(evidence, list) or len(evidence) == 0:
            return False, "invalid diagnosis.evidence"

        actions = plan.get("actions")
        if not isinstance(actions, list) or len(actions) < 1 or len(actions) > 2:
            return False, "plan.actions must contain 1-2 actions"
        for act in actions:
            if not isinstance(act, dict):
                return False, "action must be object"
            if not isinstance(act.get("name"), str):
                return False, "action.name missing"
            if not isinstance(act.get("args"), dict):
                return False, "action.args missing"
            if not isinstance(act.get("expected_effect"), str):
                return False, "action.expected_effect missing"
            if not isinstance(act.get("risk"), str):
                return False, "action.risk missing"

        if not isinstance(rollback.get("condition"), str):
            return False, "rollback.condition missing"
        fallback_action = rollback.get("fallback_action")
        if not isinstance(fallback_action, dict):
            return False, "rollback.fallback_action missing"

        algorithm = proposal.get("algorithm")
        code = proposal.get("code")
        if not isinstance(algorithm, str) or len(algorithm.strip()) == 0:
            return False, "proposal.algorithm missing"
        if not isinstance(code, str) or len(code.strip()) == 0:
            return False, "proposal.code missing"
        return True, None

    def parse_response(self, response):
        payload = _extract_first_json_object(response)
        if payload is None:
            return None, None, {"used_json": False, "parse_error": "no_json_object", "parsed_json": None}

        ok, err = self._validate_payload(payload)
        if not ok:
            return None, None, {"used_json": False, "parse_error": err, "parsed_json": payload}

        algorithm = payload["proposal"]["algorithm"]
        code = payload["proposal"]["code"]
        meta = {"used_json": True, "parse_error": None, "parsed_json": payload}
        return code, algorithm, meta


def get_proposal_backend(mode):
    if mode == "eoh":
        return EOHProposalBackend()
    if mode == "dx":
        return DXProposalBackend()
    raise ValueError(f"Unknown proposal_mode: {mode}")
