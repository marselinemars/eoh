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
                print(">> Stop with empty url for local llm !")
                exit()
            self.interface_llm = InterfaceLocalLLM(self.llm_local_url)
        else:
            print("remote llm api is used ...")
            if (
                self.api_key is None
                or self.api_endpoint is None
                or self.api_key == "xxx"
                or self.api_endpoint == "xxx"
            ):
                print(">> Stop with wrong API setting: Set api_endpoint and api_key !")
                exit()
            self.interface_llm = InterfaceAPI(
                self.api_endpoint,
                self.api_key,
                self.model_LLM,
                self.debug_mode,
            )

        res = self.interface_llm.get_response("1+1=?", request_mode="code")
        self._log_interaction("1+1=?", res, stage="startup_probe", request_mode="code")
        if res is None:
            print(">> Error in LLM API, wrong endpoint, key, model or local deployment!")
            exit()

    def _log_interaction(
        self,
        prompt_content,
        response,
        stage,
        error=None,
        request_mode="code",
        tool_name=None,
        json_schema=None,
        stop=None,
    ):
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
            "request_mode": request_mode,
            "tool_name": tool_name,
            "json_schema": json_schema,
            "stop": stop,
        }
        with self.llm_io_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def get_response(self, prompt_content, request_mode="code", tool_name=None, json_schema=None, stop=None):
        try:
            response = self.interface_llm.get_response(
                prompt_content,
                request_mode=request_mode,
                tool_name=tool_name,
                json_schema=json_schema,
                stop=stop,
            )
            self._log_interaction(
                prompt_content,
                response,
                stage="generation",
                request_mode=request_mode,
                tool_name=tool_name,
                json_schema=json_schema,
                stop=stop,
            )
        except Exception as exc:
            self._log_interaction(
                prompt_content,
                None,
                stage="generation",
                error=str(exc),
                request_mode=request_mode,
                tool_name=tool_name,
                json_schema=json_schema,
                stop=stop,
            )
            raise
        return response
