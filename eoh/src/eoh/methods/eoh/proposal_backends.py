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
            "diagnosis": {"label": "string", "confidence": 0.0, "evidence": ["string"]},
            "action": {"name": "edit_score|noop", "expected_effect": "string", "risk": "string"},
            "acceptance": {
                "improve": ["string"],
                "must_not_worsen": ["string"],
                "degeneracy_reject": ["string"],
            },
            "proposal": {"algorithm": "{one-sentence description}"},
            "patch": "unified diff string or null",
        }
        wrapper = (
            "You are a diagnosis-driven heuristic editor for online bin packing.\n"
            "You MUST ground decisions in the provided trace statistics and history.\n"
            "Output VALID JSON ONLY (no markdown).\n"
            "Do NOT change the objective definition.\n\n"
            "HARD CONSTRAINTS:\n"
            "1) Propose exactly ONE action: either \"edit_score\" or \"noop\".\n"
            "2) If action=\"edit_score\", you MUST provide a unified diff patch in `patch` that edits ONLY the `score(item, bins)` function (no other files).\n"
            "3) Your edit must not introduce global-only scalars that do not vary per-bin (e.g., var(remaining) used alone). Global statistics are allowed only as multipliers of per-bin terms.\n"
            "4) Avoid score clamping like `np.maximum(scores, 0)` because it increases ties.\n"
            "5) The score must be well-defined for all feasible bins (no div by 0, log of nonpositive, NaNs)."
        )
        safe_context = {}
        if isinstance(context, dict):
            safe_context = {
                "operator": context.get("operator"),
                "parent_trace_summary": context.get("parent_trace_summary"),
                "recent_history": context.get("recent_history"),
                "observer_summary": context.get("observer_summary"),
                "degeneracy_flags": context.get("degeneracy_flags"),
                "priority": context.get("priority"),
                "current_metrics": context.get("current_metrics"),
            }
        ctx_text = "\n\nDX context (JSON):\n" + json.dumps(safe_context, ensure_ascii=True)
        schema_text = "\n\nREQUIRED JSON schema:\n" + json.dumps(schema, ensure_ascii=True)
        evidence_text = (
            "\n\nIMPORTANT: In evidence, include 3-6 short bullet facts that quote actual numbers from DX context (not generic claims)."
        )
        return wrapper + schema_text + ctx_text + evidence_text + "\n\n" + base_prompt

    def _validate_payload(self, payload):
        if not isinstance(payload, dict):
            return False, "payload is not an object"

        diagnosis = payload.get("diagnosis")
        action = payload.get("action")
        acceptance = payload.get("acceptance")
        proposal = payload.get("proposal")
        patch = payload.get("patch")

        if not isinstance(diagnosis, dict):
            return False, "missing diagnosis object"
        if not isinstance(action, dict):
            return False, "missing action object"
        if not isinstance(acceptance, dict):
            return False, "missing acceptance object"
        if not isinstance(proposal, dict):
            return False, "missing proposal object"

        label = diagnosis.get("label")
        confidence = diagnosis.get("confidence")
        evidence = diagnosis.get("evidence")
        if not isinstance(label, str) or len(label.strip()) == 0:
            return False, "invalid diagnosis.label"
        if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
            return False, "invalid diagnosis.confidence"
        if not isinstance(evidence, list) or len(evidence) < 3 or len(evidence) > 6:
            return False, "diagnosis.evidence must contain 3-6 items"

        action_name = action.get("name")
        if action_name not in ("edit_score", "noop"):
            return False, "action.name must be edit_score|noop"
        if not isinstance(action.get("expected_effect"), str):
            return False, "action.expected_effect missing"
        if not isinstance(action.get("risk"), str):
            return False, "action.risk missing"

        improve = acceptance.get("improve")
        must_not_worsen = acceptance.get("must_not_worsen")
        degeneracy_reject = acceptance.get("degeneracy_reject")
        if not isinstance(improve, list):
            return False, "acceptance.improve missing"
        if not isinstance(must_not_worsen, list):
            return False, "acceptance.must_not_worsen missing"
        if not isinstance(degeneracy_reject, list):
            return False, "acceptance.degeneracy_reject missing"

        algorithm = proposal.get("algorithm")
        if not isinstance(algorithm, str) or len(algorithm.strip()) == 0:
            return False, "proposal.algorithm missing"
        if patch is not None and not isinstance(patch, str):
            return False, "patch must be string or null"

        if action_name == "edit_score" and (not isinstance(patch, str) or len(patch.strip()) == 0):
            return False, "edit_score requires non-empty patch"
        if action_name == "noop" and patch not in (None, ""):
            return False, "noop requires patch to be null/empty"
        return True, None

    def parse_response(self, response):
        payload = _extract_first_json_object(response)
        if payload is None:
            return None, {"used_json": False, "parse_error": "no_json_object", "parsed_json": None}

        ok, err = self._validate_payload(payload)
        if not ok:
            return None, {"used_json": False, "parse_error": err, "parsed_json": payload}

        meta = {"used_json": True, "parse_error": None, "parsed_json": payload}
        return payload, meta


def get_proposal_backend(mode):
    if mode == "eoh":
        return EOHProposalBackend()
    if mode == "dx":
        return DXProposalBackend()
    raise ValueError(f"Unknown proposal_mode: {mode}")
