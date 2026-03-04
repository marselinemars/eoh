import http.client
import json


class InterfaceAPI:
    def __init__(self, api_endpoint, api_key, model_LLM, debug_mode):
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_LLM = model_LLM
        self.debug_mode = debug_mode
        self.n_trial = 5

    def get_response(self, prompt_content, request_mode="code", tool_name=None, json_schema=None, stop=None):
        payload = {
            "model": self.model_LLM,
            "messages": [
                {"role": "user", "content": prompt_content}
            ],
        }
        if request_mode == "json":
            payload["response_format"] = {"type": "json_object"}
        if stop is not None:
            payload["stop"] = stop

        # Optional future path for tool calling support.
        if request_mode == "tool" and tool_name and isinstance(json_schema, dict):
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": str(tool_name),
                        "parameters": json_schema,
                    },
                }
            ]
            payload["tool_choice"] = {"type": "function", "function": {"name": str(tool_name)}}

        headers = {
            "Authorization": "Bearer " + self.api_key,
            "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
            "Content-Type": "application/json",
            "x-api2d-no-cache": 1,
        }
        
        response = None
        active_payload = dict(payload)
        n_trial = 0
        while n_trial < self.n_trial:
            n_trial += 1
            try:
                payload_explanation = json.dumps(active_payload)
                conn = http.client.HTTPSConnection(self.api_endpoint)
                conn.request("POST", "/v1/chat/completions", payload_explanation, headers)
                res = conn.getresponse()
                data = res.read()
                if res.status != 200:
                    if request_mode == "json" and "response_format" in active_payload:
                        active_payload = dict(active_payload)
                        active_payload.pop("response_format", None)
                        continue
                    raise RuntimeError(f"http_status_{res.status}: {data[:200]!r}")
                json_data = json.loads(data)
                choice = json_data["choices"][0]["message"]
                if request_mode == "tool" and isinstance(choice.get("tool_calls"), list) and len(choice["tool_calls"]) > 0:
                    args = choice["tool_calls"][0].get("function", {}).get("arguments")
                    response = args if isinstance(args, str) else json.dumps(args)
                else:
                    response = choice.get("content")
                break
            except:
                if self.debug_mode:
                    print("Error in API. Restarting the process...")
                continue
            

        return response
