# This file includes classe to get response from deployed local LLM
import json
import os
import time
import requests


class InterfaceLocalLLM:
    """Language model that predicts continuation of provided source code.
    """

    def __init__(self, url):
        self._url = url  # 'http://127.0.0.1:11045/completions'
        self._timeout_s = int(os.getenv("EOH_LOCAL_LLM_TIMEOUT_S", "180"))
        self._retry_sleep_s = float(os.getenv("EOH_LOCAL_LLM_RETRY_SLEEP_S", "1.0"))
        self._do_sample = os.getenv("EOH_LOCAL_LLM_DO_SAMPLE", "0") == "1"
        self._temperature = float(os.getenv("EOH_LOCAL_LLM_TEMPERATURE", "0.2"))
        self._top_p = float(os.getenv("EOH_LOCAL_LLM_TOP_P", "0.95"))
        self._max_new_tokens = int(os.getenv("EOH_LOCAL_LLM_MAX_NEW_TOKENS", "1200"))

    def get_response(self, content: str, request_mode="code", tool_name=None, json_schema=None, stop=None) -> str:
        n_try = 0
        while True:
            try:
                n_try += 1
                response = self._do_request(
                    content,
                    request_mode=request_mode,
                    tool_name=tool_name,
                    json_schema=json_schema,
                    stop=stop,
                )
                return response
            except Exception as exc:
                print(f"Local LLM request failed (try={n_try}): {exc}")
                time.sleep(self._retry_sleep_s)
                continue

    def _is_controller_json_prompt(self, content: str) -> bool:
        if not isinstance(content, str):
            return False
        markers = [
            "ROLE: Agent 1 - DIAGNOSER",
            "ROLE: Agent 2 - PLANNER",
            "ROLE: Agent 3 - CRITIC / SAFETY",
            "Return ONLY valid JSON",
            "\"op_probs\"",
            "\"diagnosis_labels\"",
        ]
        return any(marker in content for marker in markers)

    def _do_request(self, content: str, request_mode="code", tool_name=None, json_schema=None, stop=None) -> str:
        content = content.strip('\n').strip()
        if request_mode == "json":
            response_mode = "json"
        elif request_mode == "tool":
            response_mode = "json"
        elif request_mode == "code":
            response_mode = "code"
        else:
            response_mode = "json" if self._is_controller_json_prompt(content) else "code"
        # repeat the prompt for batch inference (inorder to decease the sample delay)
        params = {
            'eoh_response_mode': response_mode,
            'do_sample': self._do_sample,
            'temperature': self._temperature,
            'top_k': None,
            'top_p': self._top_p,
            'max_new_tokens': self._max_new_tokens,
            'add_special_tokens': False,
            'skip_special_tokens': True,
        }
        if tool_name is not None:
            params["eoh_tool_name"] = str(tool_name)
        if json_schema is not None:
            params["eoh_json_schema"] = json_schema
        if stop is not None:
            params["stop"] = stop
        data = {
            'prompt': content,
            'repeat_prompt': 1,
            'params': params
        }
        headers = {'Content-Type': 'application/json'}
        response = requests.post(self._url, data=json.dumps(data), headers=headers, timeout=self._timeout_s)
        print(response)
        if response.status_code == 200:
            response = response.json()['content'][0]
            return response
        raise RuntimeError(f"status={response.status_code} body={response.text[:300]}")
