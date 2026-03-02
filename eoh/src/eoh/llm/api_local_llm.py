# This file includes classe to get response from deployed local LLM
import json
from typing import Collection
import requests


class InterfaceLocalLLM:
    """Language model that predicts continuation of provided source code.
    """

    def __init__(self, url):
        self._url = url  # 'http://127.0.0.1:11045/completions'

    def get_response(self, content: str) -> str:
        while True:
            try:
                response = self._do_request(content)
                return response
            except Exception:
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
        response = requests.post(self._url, data=json.dumps(data), headers=headers)
        if response.status_code != 200:
            raise RuntimeError(f"Local LLM request failed ({response.status_code}): {response.text[:300]}")

        payload = response.json()
        content_field = payload.get('content')
        if isinstance(content_field, list) and len(content_field) > 0:
            return str(content_field[0])
        if isinstance(content_field, str):
            return content_field

        choices = payload.get('choices')
        if isinstance(choices, list) and len(choices) > 0:
            first = choices[0]
            if isinstance(first, dict):
                msg = first.get('message')
                if isinstance(msg, dict) and isinstance(msg.get('content'), str):
                    return msg['content']
                if isinstance(first.get('text'), str):
                    return first['text']

        raise RuntimeError("Unsupported local LLM response schema: expected content/choices.")
