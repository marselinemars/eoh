import json
from urllib.parse import urlparse

import requests


class InterfaceAPI:
    def __init__(self, api_endpoint, api_key, model_LLM, debug_mode):
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_LLM = model_LLM
        self.debug_mode = debug_mode
        self.n_trial = 5
        self.timeout_s = 180
        self.chat_url = self._resolve_chat_url(api_endpoint)

    def _resolve_chat_url(self, api_endpoint):
        endpoint = str(api_endpoint or "").strip()
        if endpoint.startswith("http://") or endpoint.startswith("https://"):
            parsed = urlparse(endpoint)
            base = endpoint.rstrip("/")
            if base.endswith("/chat/completions"):
                return base
            if base.endswith("/v1"):
                return base + "/chat/completions"
            return base + "/v1/chat/completions"
        return "https://" + endpoint.rstrip("/") + "/v1/chat/completions"

    def get_response(self, prompt_content):
        payload = {
            "model": self.model_LLM,
            "messages": [
                {"role": "user", "content": prompt_content}
            ],
        }
        headers = {
            "Authorization": "Bearer " + self.api_key,
            "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
            "Content-Type": "application/json",
            "x-api2d-no-cache": "1",
        }

        response_text = None
        for _ in range(self.n_trial):
            try:
                res = requests.post(
                    self.chat_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout_s,
                )
                if res.status_code != 200:
                    raise RuntimeError(f"http_status_{res.status_code}: {res.text[:300]}")
                data = res.json()
                response_text = data["choices"][0]["message"]["content"]
                break
            except Exception as exc:
                if self.debug_mode:
                    print(f"Error in API. Restarting the process... {exc}")
                continue

        return response_text
