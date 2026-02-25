import json
import re


def _extract_first_json_object(text):
    if not isinstance(text, str):
        return None

    # Try fenced json blocks first.
    fenced = re.findall(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        for chunk in fenced:
            try:
                return json.loads(chunk)
            except Exception:
                pass

    # Fallback: greedy first object extraction.
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
        # Phase-1 scaffold: add diagnosis/planning contract while keeping legacy
        # code-generation prompt intact so baseline behavior remains stable.
        dx_header = (
            "You are a diagnosis-driven heuristic editor.\n"
            "Use any available trace/evidence context if provided.\n"
            "Propose one focused improvement.\n"
            "Output may include analysis text, but must include a valid Python function implementation."
        )
        return dx_header + "\n\n" + base_prompt

    def parse_response(self, response):
        payload = _extract_first_json_object(response)
        if payload is None:
            return None
        if not isinstance(payload, dict):
            return None

        # Accept direct shape: {"algorithm":"...","code":"..."}
        algorithm = payload.get("algorithm")
        code = payload.get("code")
        if isinstance(algorithm, str) and isinstance(code, str):
            return code, algorithm

        # Accept nested shape with plan + patch spec for later phases.
        plan = payload.get("plan")
        diagnosis = payload.get("diagnosis")
        patch_spec = payload.get("patch_spec")
        if (
            isinstance(diagnosis, dict)
            and isinstance(plan, dict)
            and isinstance(patch_spec, dict)
        ):
            # No patch applier in Phase-1; fall back to legacy parsing path.
            return None

        return None


def get_proposal_backend(mode):
    if mode == "eoh":
        return EOHProposalBackend()
    if mode == "dx":
        return DXProposalBackend()
    raise ValueError(f"Unknown proposal_mode: {mode}")
