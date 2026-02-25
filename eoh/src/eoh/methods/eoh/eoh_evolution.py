import json
import os
import re
from datetime import datetime, timezone

from ...llm.interface_LLM import InterfaceLLM
from .dx_utils import (
    DXArtifactLogger,
    apply_unified_diff,
    extract_score_range,
    patch_is_within_score,
    sanity_check_score,
    sha1_text,
)
from .proposal_backends import get_proposal_backend


class Evolution:
    def __init__(self, api_endpoint, api_key, model_LLM, llm_use_local, llm_local_url, debug_mode, prompts, **kwargs):
        self.prompt_task = prompts.get_task()
        self.prompt_func_name = prompts.get_func_name()
        self.prompt_func_inputs = prompts.get_func_inputs()
        self.prompt_func_outputs = prompts.get_func_outputs()
        self.prompt_inout_inf = prompts.get_inout_inf()
        self.prompt_other_inf = prompts.get_other_inf()

        if len(self.prompt_func_inputs) > 1:
            self.joined_inputs = ", ".join("'" + s + "'" for s in self.prompt_func_inputs)
        else:
            self.joined_inputs = "'" + self.prompt_func_inputs[0] + "'"

        if len(self.prompt_func_outputs) > 1:
            self.joined_outputs = ", ".join("'" + s + "'" for s in self.prompt_func_outputs)
        else:
            self.joined_outputs = "'" + self.prompt_func_outputs[0] + "'"

        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_LLM = model_LLM
        self.debug_mode = debug_mode
        self.proposal_mode = kwargs.get("proposal_mode", "eoh")
        self.proposal_backend = get_proposal_backend(self.proposal_mode)
        self.dx_call_mode = kwargs.get("dx_call_mode", "single")
        self.dx_observer_threshold_chars = kwargs.get("dx_observer_threshold_chars", 2200)
        self.dx_max_retries = int(kwargs.get("dx_max_retries", 2))
        self.dx_artifacts_mode = kwargs.get("dx_artifacts_mode", "compact")
        self.last_proposal_info = None

        output_path = kwargs.get("output_path", ".")
        self.run_id = kwargs.get("run_id") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.results_root = os.path.join(output_path or ".", "results")
        self.dx_artifacts = DXArtifactLogger(self.results_root, self.run_id, mode=self.dx_artifacts_mode)

        self.interface_llm = InterfaceLLM(
            self.api_endpoint,
            self.api_key,
            self.model_LLM,
            llm_use_local,
            llm_local_url,
            self.debug_mode,
        )

    def _safe_json_load(self, text):
        if not isinstance(text, str):
            return None
        try:
            return json.loads(text)
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

    def _default_score_code(self):
        return (
            "import numpy as np\n\n"
            "def score(item, bins):\n"
            "    remaining = bins - item\n"
            "    scores = -remaining\n"
            "    return scores\n"
        )

    def _current_code(self, dx_context):
        if isinstance(dx_context, dict):
            code = dx_context.get("current_code")
            if isinstance(code, str) and len(code.strip()) > 0:
                return code
        return self._default_score_code()

    def _need_observer_precall(self, dx_context):
        if self.proposal_mode != "dx":
            return False
        if self.dx_call_mode == "single":
            return False
        if not isinstance(dx_context, dict):
            return False
        if self.dx_call_mode == "observer_planner":
            return True
        has_signal = bool(dx_context.get("parent_trace_summary")) or bool(dx_context.get("recent_history"))
        if not has_signal:
            return False
        try:
            context_chars = len(json.dumps(dx_context, ensure_ascii=True))
        except Exception:
            context_chars = 0
        return context_chars >= int(self.dx_observer_threshold_chars)

    def _observer_prompt(self, dx_context):
        return (
            "You are the Observer agent.\n"
            "Compress the diagnostic context into compact JSON for a planner.\n"
            "Output JSON only with keys:\n"
            "- observation_summary\n"
            "- anomalies\n"
            "- regime_ranking\n"
            "- degeneracy_flags\n\n"
            "degeneracy_flags must include any of:\n"
            "- \"tie_rate_high\" if tie_events_mean / n_items_mean > 0.25\n"
            "- \"score_flat_suspected\" if tie_events_mean is near n_items_mean\n"
            "- \"single_regime_only\" if n_regimes == 1\n\n"
            "Input context:\n"
            + json.dumps(dx_context, ensure_ascii=True)
        )

    def _run_observer_precall(self, dx_context):
        info = {
            "observer_used": False,
            "observer_parse_error": None,
            "observer_summary": None,
            "observer_prompt": None,
            "observer_raw_response": None,
            "observer_event_id": None,
        }
        if not self._need_observer_precall(dx_context):
            return dx_context, info

        observer_prompt = self._observer_prompt(dx_context)
        observer_response = self.interface_llm.get_response(observer_prompt)
        observer_json = self._safe_json_load(observer_response)

        info["observer_used"] = True
        info["observer_prompt"] = observer_prompt
        info["observer_raw_response"] = observer_response

        required_keys = {"observation_summary", "anomalies", "regime_ranking", "degeneracy_flags"}
        valid_observer_json = isinstance(observer_json, dict) and required_keys.issubset(set(observer_json.keys()))
        parsed_payload = observer_json if valid_observer_json else {"parse_error": "invalid_observer_json"}
        if valid_observer_json:
            compact = {
                "observation_summary": observer_json.get("observation_summary"),
                "anomalies": observer_json.get("anomalies"),
                "regime_ranking": observer_json.get("regime_ranking"),
                "degeneracy_flags": observer_json.get("degeneracy_flags") or dx_context.get("degeneracy_flags", []),
            }
            info["observer_summary"] = compact
            new_context = dict(dx_context) if isinstance(dx_context, dict) else {}
            new_context["observer_summary"] = compact
            effective_context = new_context
        else:
            info["observer_parse_error"] = "invalid_observer_json"
            effective_context = dx_context

        call_meta = self.dx_artifacts.log_call(
            agent="observer",
            prompt_text=observer_prompt,
            response_text=observer_response,
            parsed_json=parsed_payload,
            patch_text=None,
            metrics_snapshot=(dx_context or {}).get("current_metrics"),
            code_hash_before=sha1_text((dx_context or {}).get("current_code")),
            code_hash_after=sha1_text((effective_context or {}).get("current_code")),
            extra={"parse_error": info["observer_parse_error"]},
        )
        info["observer_event_id"] = call_meta["event_id"]
        return effective_context, info

    def _prepare_prompt(self, operator, base_prompt, dx_context):
        effective_context, observer_info = self._run_observer_precall(dx_context)
        prompt_content = self.proposal_backend.build_prompt(operator, base_prompt, effective_context)
        return prompt_content, observer_info, effective_context

    def _targeted_edit_prompt(self, dx_context):
        context = dx_context if isinstance(dx_context, dict) else {}
        current_code = self._current_code(context)
        current_metrics = context.get("current_metrics") or {}
        degeneracy_flags = context.get("degeneracy_flags") or []
        priority = context.get("priority") or "improve bins_used_mean while preserving score differentiation"
        top_metric = context.get("top_metric")
        recent_history = context.get("recent_history") or []
        observer_summary = context.get("observer_summary")

        prompt = (
            "Task: Edit the existing `score(item, bins)` function to address the diagnosis.\n"
            "You are NOT designing from scratch. You are making a targeted revision.\n\n"
            "Use these signals as primary evidence (if present):\n"
            "- bins_used_mean (main objective proxy)\n"
            "- late_large_failures_mean (catastrophic late failures)\n"
            "- near_miss_mean (wasted opportunities)\n"
            "- tie_events_mean and n_items_mean (degeneracy / flat scoring)\n"
            "- utilization_mean_mean (packing efficiency)\n"
            "- fill_std_mean (over-consolidation risk)\n\n"
            "If degeneracy_flags includes \"tie_rate_high\" or \"score_flat_suspected\":\n"
            "your first priority is to make scores vary across bins (reduce ties) while keeping best-fit behavior.\n\n"
            "Output exactly one action. If you cannot produce a safe improvement, output noop.\n\n"
            f"Priority: {priority}\n"
            f"Top metric: {top_metric}\n"
            f"Current metrics: {json.dumps(current_metrics, ensure_ascii=True)}\n"
            f"Degeneracy flags: {json.dumps(degeneracy_flags, ensure_ascii=True)}\n"
            f"Observer summary: {json.dumps(observer_summary, ensure_ascii=True)}\n"
            f"Recent history: {json.dumps(recent_history, ensure_ascii=True)}\n\n"
            "Current code (edit this function only):\n"
            "```python\n"
            f"{current_code}\n"
            "```"
        )
        return prompt

    def get_prompt_i1(self):
        prompt_content = (
            self.prompt_task
            + "\n"
            "First, describe your new algorithm and main steps in one sentence. "
            "The description must be inside a brace. Next, implement it in Python as a function named "
            + self.prompt_func_name
            + ". This function should accept "
            + str(len(self.prompt_func_inputs))
            + " input(s): "
            + self.joined_inputs
            + ". The function should return "
            + str(len(self.prompt_func_outputs))
            + " output(s): "
            + self.joined_outputs
            + ". "
            + self.prompt_inout_inf
            + " "
            + self.prompt_other_inf
            + "\n"
            + "Do not give additional explanations."
        )
        return prompt_content

    def get_prompt_e1(self, indivs):
        prompt_indiv = ""
        for i in range(len(indivs)):
            prompt_indiv = (
                prompt_indiv
                + "No."
                + str(i + 1)
                + " algorithm and the corresponding code are: \n"
                + indivs[i]["algorithm"]
                + "\n"
                + indivs[i]["code"]
                + "\n"
            )

        prompt_content = (
            self.prompt_task
            + "\n"
            "I have "
            + str(len(indivs))
            + " existing algorithms with their codes as follows: \n"
            + prompt_indiv
            + "Please help me create a new algorithm that has a totally different form from the given ones. \n"
            "First, describe your new algorithm and main steps in one sentence. "
            "The description must be inside a brace. Next, implement it in Python as a function named "
            + self.prompt_func_name
            + ". This function should accept "
            + str(len(self.prompt_func_inputs))
            + " input(s): "
            + self.joined_inputs
            + ". The function should return "
            + str(len(self.prompt_func_outputs))
            + " output(s): "
            + self.joined_outputs
            + ". "
            + self.prompt_inout_inf
            + " "
            + self.prompt_other_inf
            + "\n"
            + "Do not give additional explanations."
        )
        return prompt_content

    def get_prompt_e2(self, indivs):
        prompt_indiv = ""
        for i in range(len(indivs)):
            prompt_indiv = (
                prompt_indiv
                + "No."
                + str(i + 1)
                + " algorithm and the corresponding code are: \n"
                + indivs[i]["algorithm"]
                + "\n"
                + indivs[i]["code"]
                + "\n"
            )

        prompt_content = (
            self.prompt_task
            + "\n"
            "I have "
            + str(len(indivs))
            + " existing algorithms with their codes as follows: \n"
            + prompt_indiv
            + "Please help me create a new algorithm that has a totally different form from the given ones but can be motivated from them. \n"
            "Firstly, identify the common backbone idea in the provided algorithms. Secondly, based on the backbone idea describe your new algorithm in one sentence. "
            "The description must be inside a brace. Thirdly, implement it in Python as a function named "
            + self.prompt_func_name
            + ". This function should accept "
            + str(len(self.prompt_func_inputs))
            + " input(s): "
            + self.joined_inputs
            + ". The function should return "
            + str(len(self.prompt_func_outputs))
            + " output(s): "
            + self.joined_outputs
            + ". "
            + self.prompt_inout_inf
            + " "
            + self.prompt_other_inf
            + "\n"
            + "Do not give additional explanations."
        )
        return prompt_content

    def get_prompt_m1(self, indiv1):
        prompt_content = (
            self.prompt_task
            + "\n"
            "I have one algorithm with its code as follows. "
            "Algorithm description: "
            + indiv1["algorithm"]
            + "\n"
            "Code:\n"
            + indiv1["code"]
            + "\n"
            "Please assist me in creating a new algorithm that has a different form but can be a modified version of the algorithm provided. \n"
            "First, describe your new algorithm and main steps in one sentence. "
            "The description must be inside a brace. Next, implement it in Python as a function named "
            + self.prompt_func_name
            + ". This function should accept "
            + str(len(self.prompt_func_inputs))
            + " input(s): "
            + self.joined_inputs
            + ". The function should return "
            + str(len(self.prompt_func_outputs))
            + " output(s): "
            + self.joined_outputs
            + ". "
            + self.prompt_inout_inf
            + " "
            + self.prompt_other_inf
            + "\n"
            + "Do not give additional explanations."
        )
        return prompt_content

    def get_prompt_m2(self, indiv1):
        prompt_content = (
            self.prompt_task
            + "\n"
            "I have one algorithm with its code as follows. "
            "Algorithm description: "
            + indiv1["algorithm"]
            + "\n"
            "Code:\n"
            + indiv1["code"]
            + "\n"
            "Please identify the main algorithm parameters and assist me in creating a new algorithm that has a different parameter settings of the score function provided. \n"
            "First, describe your new algorithm and main steps in one sentence. "
            "The description must be inside a brace. Next, implement it in Python as a function named "
            + self.prompt_func_name
            + ". This function should accept "
            + str(len(self.prompt_func_inputs))
            + " input(s): "
            + self.joined_inputs
            + ". The function should return "
            + str(len(self.prompt_func_outputs))
            + " output(s): "
            + self.joined_outputs
            + ". "
            + self.prompt_inout_inf
            + " "
            + self.prompt_other_inf
            + "\n"
            + "Do not give additional explanations."
        )
        return prompt_content

    def get_prompt_m3(self, indiv1):
        prompt_content = (
            "First, you need to identify the main components in the function below. "
            "Next, analyze whether any of these components can be overfit to the in-distribution instances. "
            "Then, based on your analysis, simplify the components to enhance the generalization to potential out-of-distribution instances. "
            "Finally, provide the revised code, keeping the function name, inputs, and outputs unchanged. \n"
            + indiv1["code"]
            + "\n"
            + self.prompt_inout_inf
            + "\n"
            + "Do not give additional explanations."
        )
        return prompt_content

    def _legacy_extract(self, response):
        algorithm = re.findall(r"\{(.*)\}", response, re.DOTALL)
        if len(algorithm) == 0:
            if "python" in response:
                algorithm = re.findall(r"^.*?(?=python)", response, re.DOTALL)
            elif "import" in response:
                algorithm = re.findall(r"^.*?(?=import)", response, re.DOTALL)
            else:
                algorithm = re.findall(r"^.*?(?=def)", response, re.DOTALL)

        code = re.findall(r"import.*return", response, re.DOTALL)
        if len(code) == 0:
            code = re.findall(r"def.*return", response, re.DOTALL)

        if len(algorithm) == 0 or len(code) == 0:
            return None
        return code[0], algorithm[0]

    def _looks_like_complete_return(self, code):
        if not isinstance(code, str):
            return False
        return bool(re.search(r"return\s+[^\s].*", code))

    def _apply_dx_action(self, payload, current_code):
        action = payload.get("action", {})
        action_name = action.get("name")
        algorithm = payload.get("proposal", {}).get("algorithm", "{noop}")

        if action_name == "noop":
            return True, current_code, algorithm, None, {
                "action_applied": "noop",
                "patch_applied": False,
                "sanity_check": "skipped",
            }

        patch_text = payload.get("patch")
        if not isinstance(patch_text, str) or len(patch_text.strip()) == 0:
            return False, None, algorithm, "empty_patch_for_edit_score", {"action_applied": "rejected"}

        try:
            score_range = extract_score_range(current_code)
        except Exception as exc:
            return False, None, algorithm, f"score_range_error:{exc}", {"action_applied": "rejected"}

        try:
            updated_code, changed_old_lines, insert_positions = apply_unified_diff(current_code, patch_text)
        except Exception as exc:
            return False, None, algorithm, f"patch_apply_error:{exc}", {"action_applied": "rejected"}

        if not patch_is_within_score(score_range, changed_old_lines, insert_positions):
            return False, None, algorithm, "patch_outside_score_function", {"action_applied": "rejected"}

        lowered = updated_code.replace(" ", "")
        if "np.maximum(scores,0)" in lowered or "np.maximum(scores,0.0)" in lowered:
            return False, None, algorithm, "forbidden_score_clamp", {"action_applied": "rejected"}

        try:
            ok, sanity_error = sanity_check_score(updated_code)
        except Exception as exc:
            return False, None, algorithm, f"sanity_exec_error:{exc}", {"action_applied": "rejected"}
        if not ok:
            return False, None, algorithm, f"sanity_failed:{sanity_error}", {"action_applied": "rejected"}

        return True, updated_code, algorithm, None, {
            "action_applied": "edit_score",
            "patch_applied": True,
            "sanity_check": "ok",
            "score_line_range": list(score_range),
        }

    def _dx_noop_result(self, current_code):
        return current_code, "{noop: keep score(item, bins) unchanged}"

    def _get_alg(self, prompt_content, observer_info=None, dx_context=None):
        if self.proposal_mode != "dx":
            response = self.interface_llm.get_response(prompt_content)
            proposal_meta = {
                "proposal_mode": self.proposal_mode,
                "prompt": prompt_content,
                "raw_response": response,
                "used_json": False,
                "parse_error": None,
                "parsed_json": None,
                "fallback_used": False,
                "retry_count": 0,
                "observer_used": False,
                "observer_parse_error": None,
                "observer_summary": None,
                "observer_prompt": None,
                "observer_raw_response": None,
            }
            if isinstance(observer_info, dict):
                proposal_meta.update(observer_info)

            parsed = self._legacy_extract(response)
            n_retry = 1
            while parsed is None:
                if self.debug_mode:
                    print("Error: algorithm or code not identified, wait 1 seconds and retrying ... ")
                response = self.interface_llm.get_response(prompt_content)
                proposal_meta["raw_response"] = response
                proposal_meta["retry_count"] = n_retry
                parsed = self._legacy_extract(response)
                if n_retry > 3:
                    break
                n_retry += 1

            if parsed is None:
                self.last_proposal_info = proposal_meta
                raise ValueError("Could not parse algorithm or code from LLM response.")

            code, algorithm = parsed
            if proposal_meta.get("used_json") and self._looks_like_complete_return(code):
                code_all = code
            else:
                code_all = code + " " + ", ".join(s for s in self.prompt_func_outputs)
            self.last_proposal_info = proposal_meta
            return [code_all, algorithm, proposal_meta]

        context = dx_context if isinstance(dx_context, dict) else {}
        current_code = self._current_code(context)
        current_code_hash = sha1_text(current_code)
        current_metrics = context.get("current_metrics")

        proposal_meta = {
            "proposal_mode": self.proposal_mode,
            "prompt": prompt_content,
            "raw_response": None,
            "used_json": False,
            "parse_error": None,
            "parsed_json": None,
            "fallback_used": False,
            "retry_count": 0,
            "observer_used": False,
            "observer_parse_error": None,
            "observer_summary": None,
            "observer_prompt": None,
            "observer_raw_response": None,
            "action_applied": None,
            "patch_applied": False,
            "planner_event_ids": [],
            "code_hash_before": current_code_hash,
            "code_hash_after": current_code_hash,
        }
        if isinstance(observer_info, dict):
            proposal_meta.update(observer_info)

        max_attempts = max(1, self.dx_max_retries + 1)
        last_error = None

        for attempt in range(max_attempts):
            response = self.interface_llm.get_response(prompt_content)
            payload, parse_meta = self.proposal_backend.parse_response(response)
            parsed_payload = payload if payload is not None else {"parse_error": parse_meta.get("parse_error")}

            proposal_meta["raw_response"] = response
            proposal_meta["used_json"] = bool(parse_meta.get("used_json"))
            proposal_meta["parse_error"] = parse_meta.get("parse_error")
            proposal_meta["parsed_json"] = parse_meta.get("parsed_json")
            proposal_meta["retry_count"] = attempt

            accepted = False
            algorithm = None
            candidate_code = None
            action_result = {"action_applied": "rejected", "patch_applied": False}
            patch_text = None

            if payload is not None:
                patch_text = payload.get("patch")
                accepted, candidate_code, algorithm, action_error, action_result = self._apply_dx_action(payload, current_code)
                if not accepted:
                    last_error = action_error
                    proposal_meta["parse_error"] = action_error
            else:
                last_error = parse_meta.get("parse_error")

            call_meta = self.dx_artifacts.log_call(
                agent="planner",
                prompt_text=prompt_content,
                response_text=response,
                parsed_json=parsed_payload,
                patch_text=patch_text,
                metrics_snapshot=current_metrics,
                code_hash_before=current_code_hash,
                code_hash_after=sha1_text(candidate_code) if accepted else current_code_hash,
                extra={
                    "attempt": attempt,
                    "accepted": accepted,
                    "parse_error": proposal_meta.get("parse_error"),
                    "action_applied": action_result.get("action_applied"),
                    "proposal_algorithm": (payload or {}).get("proposal", {}).get("algorithm") if isinstance(payload, dict) else None,
                },
            )
            proposal_meta["planner_event_ids"].append(call_meta["event_id"])

            if accepted:
                proposal_meta["action_applied"] = action_result.get("action_applied")
                proposal_meta["patch_applied"] = bool(action_result.get("patch_applied"))
                proposal_meta["fallback_used"] = False
                proposal_meta["code_hash_after"] = sha1_text(candidate_code)
                self.last_proposal_info = proposal_meta
                return [candidate_code, algorithm, proposal_meta]

        fallback_code, fallback_algorithm = self._dx_noop_result(current_code)
        proposal_meta["fallback_used"] = True
        proposal_meta["action_applied"] = "noop_fallback"
        proposal_meta["patch_applied"] = False
        proposal_meta["parse_error"] = last_error or proposal_meta.get("parse_error")
        proposal_meta["retry_count"] = max_attempts - 1
        proposal_meta["code_hash_after"] = sha1_text(fallback_code)
        self.last_proposal_info = proposal_meta
        return [fallback_code, fallback_algorithm, proposal_meta]

    def _build_base_prompt(self, operator, parents, dx_context):
        if self.proposal_mode == "dx":
            return self._targeted_edit_prompt(dx_context)
        if operator == "i1":
            return self.get_prompt_i1()
        if operator == "e1":
            return self.get_prompt_e1(parents)
        if operator == "e2":
            return self.get_prompt_e2(parents)
        if operator == "m1":
            return self.get_prompt_m1(parents)
        if operator == "m2":
            return self.get_prompt_m2(parents)
        if operator == "m3":
            return self.get_prompt_m3(parents)
        return self.get_prompt_i1()

    def i1(self, dx_context=None):
        base_prompt = self._build_base_prompt("i1", None, dx_context)
        prompt_content, observer_info, effective_context = self._prepare_prompt("i1", base_prompt, dx_context or {})

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ i1 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        code_all, algorithm, proposal_info = self._get_alg(prompt_content, observer_info=observer_info, dx_context=effective_context)
        return [code_all, algorithm, proposal_info]

    def e1(self, parents, dx_context=None):
        base_prompt = self._build_base_prompt("e1", parents, dx_context)
        prompt_content, observer_info, effective_context = self._prepare_prompt("e1", base_prompt, dx_context or {})
        code_all, algorithm, proposal_info = self._get_alg(prompt_content, observer_info=observer_info, dx_context=effective_context)
        return [code_all, algorithm, proposal_info]

    def e2(self, parents, dx_context=None):
        base_prompt = self._build_base_prompt("e2", parents, dx_context)
        prompt_content, observer_info, effective_context = self._prepare_prompt("e2", base_prompt, dx_context or {})
        code_all, algorithm, proposal_info = self._get_alg(prompt_content, observer_info=observer_info, dx_context=effective_context)
        return [code_all, algorithm, proposal_info]

    def m1(self, parents, dx_context=None):
        base_prompt = self._build_base_prompt("m1", parents, dx_context)
        prompt_content, observer_info, effective_context = self._prepare_prompt("m1", base_prompt, dx_context or {})
        code_all, algorithm, proposal_info = self._get_alg(prompt_content, observer_info=observer_info, dx_context=effective_context)
        return [code_all, algorithm, proposal_info]

    def m2(self, parents, dx_context=None):
        base_prompt = self._build_base_prompt("m2", parents, dx_context)
        prompt_content, observer_info, effective_context = self._prepare_prompt("m2", base_prompt, dx_context or {})
        code_all, algorithm, proposal_info = self._get_alg(prompt_content, observer_info=observer_info, dx_context=effective_context)
        return [code_all, algorithm, proposal_info]

    def m3(self, parents, dx_context=None):
        base_prompt = self._build_base_prompt("m3", parents, dx_context)
        prompt_content, observer_info, effective_context = self._prepare_prompt("m3", base_prompt, dx_context or {})
        code_all, algorithm, proposal_info = self._get_alg(prompt_content, observer_info=observer_info, dx_context=effective_context)
        return [code_all, algorithm, proposal_info]
