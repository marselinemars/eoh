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

    def get_response(self, content: str) -> str:
        n_try = 0
        while True:
            try:
                n_try += 1
                response = self._do_request(content)
                return response
            except Exception as exc:
                print(f"Local LLM request failed (try={n_try}): {exc}")
                time.sleep(self._retry_sleep_s)
                continue

    def _do_request(self, content: str) -> str:
        content = content.strip('\n').strip()
        # repeat the prompt for batch inference (inorder to decease the sample delay)
        data = {
            'prompt': content,
            'repeat_prompt': 1,
            'params': {
                'do_sample': True,
                'temperature': None,
                'top_k': None,
                'top_p': None,
                'add_special_tokens': False,
                'skip_special_tokens': True,
            }
        }
        headers = {'Content-Type': 'application/json'}
        response = requests.post(self._url, data=json.dumps(data), headers=headers, timeout=self._timeout_s)
        print(response)
        if response.status_code == 200:
            response = response.json()['content'][0]
            return response
        raise RuntimeError(f"status={response.status_code} body={response.text[:300]}")
