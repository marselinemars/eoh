from ..llm.api_general import InterfaceAPI
from ..llm.api_local_llm import InterfaceLocalLLM
import os
import json
from pathlib import Path
from datetime import datetime


class InterfaceLLM:
    def __init__(self, api_endpoint, api_key, model_LLM, llm_use_local, llm_local_url, debug_mode):
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_LLM = model_LLM
        self.debug_mode = debug_mode
        self.llm_use_local = llm_use_local
        self.llm_local_url = llm_local_url

        self.log_llm_io = os.getenv("EOH_LOG_LLM_IO", "0") == "1"
        self.llm_io_dir = Path(os.getenv("EOH_LLM_IO_DIR", "./results/llm_io"))
        self.llm_io_path = self.llm_io_dir / "llm_interactions.jsonl"
        self._request_id = 0
        self.last_request_id = None
        if self.log_llm_io:
            self.llm_io_dir.mkdir(parents=True, exist_ok=True)

        print("- check LLM API")

        if self.llm_use_local:
            print("local llm delopyment is used ...")
            if self.llm_local_url is None or self.llm_local_url == "xxx":
                raise RuntimeError("Stop with empty url for local llm.")
            self.interface_llm = InterfaceLocalLLM(self.llm_local_url)
        else:
            print("remote llm api is used ...")
            self.interface_llm = InterfaceAPI(
                self.api_endpoint,
                self.api_key,
                self.model_LLM,
                self.debug_mode,
            )

        res = self.interface_llm.get_response("1+1=?")
        self._log_interaction("1+1=?", res, stage="startup_probe")
        if res is None:
            active_model = getattr(self.interface_llm, "active_model", self.model_LLM)
            raise RuntimeError(
                f"Error in LLM API, wrong endpoint, key, model or local deployment. "
                f"requested_model={self.model_LLM} active_model={active_model} "
                f"base_url={getattr(self.interface_llm, 'base_url', self.api_endpoint)}"
            )

    def _log_interaction(self, prompt_content, response, stage, error=None):
        if not self.log_llm_io:
            return
        self._request_id += 1
        self.last_request_id = self._request_id
        record = {
            "time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "request_id": self._request_id,
            "stage": stage,
            "llm_use_local": self.llm_use_local,
            "model": self.model_LLM,
            "prompt": prompt_content,
            "response": response,
            "prompt_chars": len(prompt_content) if prompt_content is not None else 0,
            "response_chars": len(response) if isinstance(response, str) else 0,
            "error": error,
        }
        with self.llm_io_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def get_response(self, prompt_content):
        try:
            response = self.interface_llm.get_response(prompt_content)
            self._log_interaction(prompt_content, response, stage="generation")
        except Exception as exc:
            self._log_interaction(prompt_content, None, stage="generation", error=str(exc))
            raise
        return response
