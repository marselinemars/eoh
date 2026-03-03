import json
import os
import sys
import threading
import re
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Tuple

import requests

DEFAULT_HPC_BASE = "http://vllm-nodeport.vllm-ns.svc.cluster.local:8000/v1"
DEFAULT_HPC_KEY = "my-key-ensia-2022-1030"
DEFAULT_HPC_MODEL = "QuantTrio/Qwen3-VL-235B-A22B-Instruct-AWQ"
DEFAULT_BRIDGE_SYSTEM_MESSAGE = (
    "Return ONLY: (1) one short algorithm sentence in {...} and (2) valid Python code. "
    "No reasoning, no thinking process, no markdown fences, no extra commentary."
)


@dataclass
class HPCBridgeConfig:
    base_url: str
    api_key: str
    model: str = "auto"
    port: int = 18000


def ensure_eoh_src_on_path(start_dir: Path | None = None) -> Tuple[Path, Path]:
    cwd = (start_dir or Path.cwd()).resolve()
    candidates = [cwd, cwd.parent, cwd.parent.parent]

    project_root = None
    eoh_src = None
    for base in candidates:
        p = base / "eoh" / "src"
        if p.exists():
            project_root = base
            eoh_src = p
            break

    if eoh_src is None:
        raise RuntimeError(f"Could not find eoh/src from cwd={cwd}")

    if str(eoh_src) not in sys.path:
        sys.path.insert(0, str(eoh_src))

    return project_root, eoh_src


def config_from_env() -> HPCBridgeConfig:
    base_url = os.getenv("ENSIA_VLLM_BASE", DEFAULT_HPC_BASE)
    api_key = os.getenv("ENSIA_VLLM_API_KEY", DEFAULT_HPC_KEY)
    model = os.getenv("ENSIA_VLLM_MODEL", DEFAULT_HPC_MODEL)
    port = int(os.getenv("EOH_BRIDGE_PORT", "18000"))
    return HPCBridgeConfig(base_url=base_url, api_key=api_key, model=model, port=port)


def _fetch_model_ids(cfg: HPCBridgeConfig, timeout_s: int = 60) -> list[str]:
    headers = {}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    resp = requests.get(f"{cfg.base_url}/models", headers=headers, timeout=timeout_s)
    if resp.status_code != 200:
        raise RuntimeError(f"/models failed: {resp.status_code} {resp.text[:300]}")
    payload = resp.json()
    model_ids = [m.get("id") for m in payload.get("data", []) if m.get("id")]
    return model_ids


def resolve_model_id(cfg: HPCBridgeConfig, timeout_s: int = 60) -> str:
    model_ids = _fetch_model_ids(cfg, timeout_s=timeout_s)
    if not model_ids:
        raise RuntimeError("No model IDs returned by /models.")

    if cfg.model == "auto":
        return model_ids[0]

    if cfg.model in model_ids:
        return cfg.model

    print(
        f"Requested model '{cfg.model}' is unavailable. "
        f"Falling back to available model '{model_ids[0]}'."
    )
    return model_ids[0]


def _make_handler(base_url: str, api_key: str, model_id: str):
    bridge_max_tokens = int(os.getenv("EOH_BRIDGE_MAX_TOKENS", "1200"))
    bridge_system_message = os.getenv("EOH_BRIDGE_SYSTEM_MESSAGE", DEFAULT_BRIDGE_SYSTEM_MESSAGE)
    disable_thinking = os.getenv("EOH_BRIDGE_DISABLE_THINKING", "1") == "1"
    repair_retries = int(os.getenv("EOH_BRIDGE_REPAIR_RETRIES", "1"))

    class BridgeHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def _send(self, code: int, payload: dict):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _extract_description(self, text: str):
            m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
            return m.group(0).strip() if m else None

        def _extract_code(self, text: str):
            code_blocks = re.findall(r"```(?:python|py)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
            for block in code_blocks:
                if "def score" in block:
                    return block.strip()
            if code_blocks:
                return code_blocks[0].strip()
            m = re.search(r"(import[\s\S]*?def\s+score\s*\(.*)", text, re.DOTALL)
            if m:
                return m.group(1).strip()
            m = re.search(r"(def\s+score\s*\(.*)", text, re.DOTALL)
            if m:
                return m.group(1).strip()
            return None

        def _looks_valid_final(self, text: str):
            t = text or ""
            return ("def score" in t) and ("{" in t and "}" in t)

        def _sanitize_to_final(self, text: str):
            if not isinstance(text, str):
                return ""
            desc = self._extract_description(text)
            code = self._extract_code(text)
            if code is None:
                return text.strip()
            parts = []
            if desc:
                parts.append(desc)
            parts.append(code)
            return "\n".join(parts).strip()

        def do_POST(self):
            if self.path != "/completions":
                self._send(404, {"error": "not found"})
                return

            try:
                length = int(self.headers.get("Content-Length", "0"))
                req = json.loads(self.rfile.read(length).decode("utf-8"))
                prompt = req.get("prompt", "")
                params = req.get("params", {}) or {}
                temperature = params.get("temperature", 0.2)
                if temperature is None:
                    temperature = 0.2
                max_new_tokens = params.get("max_new_tokens", bridge_max_tokens)
                if max_new_tokens is None:
                    max_new_tokens = bridge_max_tokens

                headers = {
                    "Content-Type": "application/json",
                }
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"

                messages = [{"role": "user", "content": prompt}]
                if bridge_system_message:
                    messages = [
                        {"role": "system", "content": bridge_system_message},
                        {"role": "user", "content": prompt},
                    ]

                chat_payload = {
                    "model": model_id,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_new_tokens,
                }
                if disable_thinking:
                    # Some Qwen/vLLM deployments accept one of these flags.
                    # Unknown keys are typically ignored by compliant servers.
                    chat_payload["thinking"] = False
                    chat_payload["chat_template_kwargs"] = {"enable_thinking": False}
                    chat_payload["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
                r = requests.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json=chat_payload,
                    timeout=600,
                )
                if r.status_code == 200:
                    data = r.json()
                    text = data.get("choices", [{}])[0].get("message", {}).get("content")
                    if isinstance(text, str) and text:
                        text = self._sanitize_to_final(text)
                        if not self._looks_valid_final(text):
                            for _ in range(repair_retries):
                                repair_prompt = (
                                    "Rewrite the following draft into final output ONLY:\n"
                                    "1) one short sentence in braces {...}\n"
                                    "2) valid Python code defining score(item, bins) and returning scores\n"
                                    "No reasoning text.\n\nDraft:\n"
                                    f"{text}"
                                )
                                repair_payload = {
                                    "model": model_id,
                                    "messages": [
                                        {"role": "system", "content": bridge_system_message},
                                        {"role": "user", "content": repair_prompt},
                                    ],
                                    "temperature": 0.0,
                                    "max_tokens": max_new_tokens,
                                }
                                if disable_thinking:
                                    repair_payload["thinking"] = False
                                    repair_payload["chat_template_kwargs"] = {"enable_thinking": False}
                                rr = requests.post(
                                    f"{base_url}/chat/completions",
                                    headers=headers,
                                    json=repair_payload,
                                    timeout=600,
                                )
                                if rr.status_code == 200:
                                    rd = rr.json()
                                    rtext = rd.get("choices", [{}])[0].get("message", {}).get("content", "")
                                    text = self._sanitize_to_final(rtext)
                                    if self._looks_valid_final(text):
                                        break
                        self._send(200, {"content": [text]})
                        return

                # fallback to completion-style endpoint if chat path is unavailable
                completion_prompt = prompt
                if bridge_system_message:
                    completion_prompt = (
                        f"System: {bridge_system_message}\n\n"
                        f"User: {prompt}\n\n"
                        "Assistant:"
                    )
                comp_payload = {
                    "model": model_id,
                    "prompt": completion_prompt,
                    "temperature": temperature,
                    "max_tokens": max_new_tokens,
                }
                if disable_thinking:
                    comp_payload["thinking"] = False
                r2 = requests.post(
                    f"{base_url}/completions",
                    headers=headers,
                    json=comp_payload,
                    timeout=600,
                )
                if r2.status_code != 200:
                    self._send(
                        502,
                        {
                            "error": "upstream",
                            "chat_status": r.status_code,
                            "comp_status": r2.status_code,
                            "chat_text": r.text[:300],
                            "comp_text": r2.text[:300],
                        },
                    )
                    return

                data2 = r2.json()
                text2 = data2.get("choices", [{}])[0].get("text", "")
                text2 = self._sanitize_to_final(text2)
                self._send(200, {"content": [text2]})
            except Exception as exc:
                self._send(500, {"error": str(exc)})

    return BridgeHandler


def start_hpc_bridge(cfg: HPCBridgeConfig):
    model_id = resolve_model_id(cfg)
    handler = _make_handler(cfg.base_url, cfg.api_key, model_id)
    server = HTTPServer(("127.0.0.1", cfg.port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    bridge_url = f"http://127.0.0.1:{cfg.port}/completions"
    return server, thread, bridge_url, model_id


def test_bridge(bridge_url: str):
    test_payload = {
        "prompt": "Reply with exactly: OK",
        "repeat_prompt": 1,
        "params": {"do_sample": True},
    }
    resp = requests.post(bridge_url, json=test_payload, timeout=180)
    return resp.status_code, resp.json()


def stop_hpc_bridge(server: HTTPServer | None):
    if server is None:
        return
    server.shutdown()
    server.server_close()
