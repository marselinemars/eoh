import json
import os
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
        self.enable_model_fallback = os.getenv("EOH_API_MODEL_FALLBACK", "1") == "1"
        self.base_url = self._resolve_base_url(api_endpoint)
        self.chat_url = self.base_url + "/chat/completions"
        self.models_url = self.base_url + "/models"
        self.available_models = []
        self.active_model = self._resolve_model_name(model_LLM)

    def _resolve_base_url(self, api_endpoint):
        endpoint = str(api_endpoint or "").strip()
        if endpoint.startswith("http://") or endpoint.startswith("https://"):
            base = endpoint.rstrip("/")
            if base.endswith("/chat/completions"):
                return base[: -len("/chat/completions")]
            if base.endswith("/v1"):
                return base
            return base + "/v1"
        parsed = urlparse("https://" + endpoint.rstrip("/"))
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path.rstrip('/')}/v1"

    def _headers(self):
        return {
            "Authorization": "Bearer " + self.api_key,
            "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
            "Content-Type": "application/json",
            "x-api2d-no-cache": "1",
        }

    def _fetch_models(self):
        if not self.enable_model_fallback:
            return []
        try:
            res = requests.get(self.models_url, headers=self._headers(), timeout=min(30, self.timeout_s))
            if res.status_code != 200:
                if self.debug_mode:
                    print(f"/models lookup failed: {res.status_code} {res.text[:200]}")
                return []
            data = res.json()
            models = [item.get("id") for item in data.get("data", []) if item.get("id")]
            self.available_models = models
            return models
        except Exception as exc:
            if self.debug_mode:
                print(f"/models lookup failed: {exc}")
            return []

    def _resolve_model_name(self, requested_model):
        if not self.enable_model_fallback:
            return requested_model
        models = self._fetch_models()
        if not models:
            return requested_model
        if requested_model in models:
            return requested_model
        if self.debug_mode:
            print(f"Requested model '{requested_model}' unavailable. Falling back to '{models[0]}'.")
        return models[0]

    def _is_model_error(self, response):
        try:
            payload = response.json()
        except Exception:
            payload = {}
        body_text = response.text.lower() if isinstance(response.text, str) else ""
        error_text = json.dumps(payload).lower()
        text = body_text + " " + error_text
        markers = [
            "model",
            "not found",
            "does not exist",
            "unknown model",
            "invalid model",
        ]
        return any(marker in text for marker in markers)

    def _fallback_to_available_model(self):
        models = self.available_models or self._fetch_models()
        if not models:
            return False
        fallback = models[0]
        if self.active_model == fallback:
            return False
        if self.debug_mode:
            print(f"Retrying with fallback model '{fallback}'.")
        self.active_model = fallback
        return True

    def get_response(self, prompt_content):
        response_text = None
        for _ in range(self.n_trial):
            try:
                payload = {
                    "model": self.active_model,
                    "messages": [
                        {"role": "user", "content": prompt_content}
                    ],
                }
                res = requests.post(
                    self.chat_url,
                    headers=self._headers(),
                    json=payload,
                    timeout=self.timeout_s,
                )
                if res.status_code != 200:
                    if self.enable_model_fallback and self._is_model_error(res) and self._fallback_to_available_model():
                        continue
                    raise RuntimeError(f"http_status_{res.status_code}: {res.text[:300]}")
                data = res.json()
                response_text = data["choices"][0]["message"]["content"]
                break
            except Exception as exc:
                if self.debug_mode:
                    print(f"Error in API. Restarting the process... {exc}")
                continue
        return response_text
