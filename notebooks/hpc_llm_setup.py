import json
import os
import sys
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Tuple

import requests


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
    base_url = os.getenv(
        "ENSIA_VLLM_BASE",
        "http://vllm-nodeport.vllm-ns.svc.cluster.local:8000/v1",
    )
    api_key = os.getenv("ENSIA_VLLM_API_KEY", "")
    model = os.getenv("ENSIA_VLLM_MODEL", "auto")
    port = int(os.getenv("EOH_BRIDGE_PORT", "18000"))
    return HPCBridgeConfig(base_url=base_url, api_key=api_key, model=model, port=port)


def resolve_model_id(cfg: HPCBridgeConfig, timeout_s: int = 60) -> str:
    if not cfg.api_key:
        raise RuntimeError("Missing ENSIA_VLLM_API_KEY.")
    if cfg.model != "auto":
        return cfg.model

    headers = {"Authorization": f"Bearer {cfg.api_key}"}
    resp = requests.get(f"{cfg.base_url}/models", headers=headers, timeout=timeout_s)
    if resp.status_code != 200:
        raise RuntimeError(f"/models failed: {resp.status_code} {resp.text[:300]}")

    payload = resp.json()
    model_ids = [m.get("id") for m in payload.get("data", []) if m.get("id")]
    if not model_ids:
        raise RuntimeError("No model IDs returned by /models.")
    return model_ids[0]


def _make_handler(base_url: str, api_key: str, model_id: str):
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

        def do_POST(self):
            if self.path != "/completions":
                self._send(404, {"error": "not found"})
                return

            try:
                length = int(self.headers.get("Content-Length", "0"))
                req = json.loads(self.rfile.read(length).decode("utf-8"))
                prompt = req.get("prompt", "")
                params = req.get("params", {}) or {}

                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }

                chat_payload = {
                    "model": model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": params.get("temperature", 0.2),
                }
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
                        self._send(200, {"content": [text]})
                        return

                # fallback to completion-style endpoint if chat path is unavailable
                comp_payload = {
                    "model": model_id,
                    "prompt": prompt,
                    "temperature": params.get("temperature", 0.2),
                    "max_tokens": params.get("max_new_tokens", 512),
                }
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
