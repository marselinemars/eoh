import hashlib
import json
import os
import time
from datetime import datetime, timezone


def _utc_run_id():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def resolve_events_path(output_path, events_path):
    if events_path is None:
        return None
    if os.path.isabs(events_path):
        return events_path
    base = output_path or "."
    return os.path.normpath(os.path.join(base, events_path))


def code_info(code, include_code=False):
    if code is None or not isinstance(code, str):
        return {}
    digest = hashlib.sha1(code.encode("utf-8", errors="ignore")).hexdigest()
    info = {
        "code_hash": digest,
        "code_len": len(code),
    }
    if include_code:
        info["code"] = code
    return info


def build_run_meta(paras, method_name, problem_name, events_path):
    meta = {
        "method": method_name,
        "problem": problem_name,
        "events_path": events_path,
        "ec_pop_size": getattr(paras, "ec_pop_size", None),
        "ec_n_pop": getattr(paras, "ec_n_pop", None),
        "ec_operators": getattr(paras, "ec_operators", None),
        "ec_m": getattr(paras, "ec_m", None),
        "ec_operator_weights": getattr(paras, "ec_operator_weights", None),
        "exp_n_proc": getattr(paras, "exp_n_proc", None),
        "exp_debug_mode": getattr(paras, "exp_debug_mode", None),
        "exp_use_seed": getattr(paras, "exp_use_seed", None),
        "exp_seed_path": getattr(paras, "exp_seed_path", None),
        "exp_use_continue": getattr(paras, "exp_use_continue", None),
        "exp_continue_path": getattr(paras, "exp_continue_path", None),
        "exp_output_path": getattr(paras, "exp_output_path", None),
        "exp_log_llm_raw": getattr(paras, "exp_log_llm_raw", None),
        "eva_timeout": getattr(paras, "eva_timeout", None),
        "eva_numba_decorator": getattr(paras, "eva_numba_decorator", None),
        "llm_use_local": getattr(paras, "llm_use_local", None),
        "llm_local_url": getattr(paras, "llm_local_url", None),
        "llm_api_endpoint": getattr(paras, "llm_api_endpoint", None),
        "llm_model": getattr(paras, "llm_model", None),
        "proposal_mode": getattr(paras, "proposal_mode", None),
        "dx_history_k": getattr(paras, "dx_history_k", None),
        "dx_call_mode": getattr(paras, "dx_call_mode", None),
        "dx_observer_threshold_chars": getattr(paras, "dx_observer_threshold_chars", None),
        "dx_max_retries": getattr(paras, "dx_max_retries", None),
        "dx_artifacts_mode": getattr(paras, "dx_artifacts_mode", None),
        "bp_add_synthetic_regimes": getattr(paras, "bp_add_synthetic_regimes", None),
    }
    return meta


class EventLogger:
    def __init__(self, enabled, path, run_id=None):
        self.enabled = bool(enabled and path)
        self.path = path
        self.run_id = run_id or _utc_run_id()
        self._event_id = 0
        self._fh = None

        if self.enabled:
            folder = os.path.dirname(self.path)
            if folder:
                os.makedirs(folder, exist_ok=True)
            self._fh = open(self.path, "a", encoding="utf-8")

    def log(self, record):
        if not self.enabled:
            return
        payload = dict(record)
        payload["event_id"] = self._event_id
        payload["run_id"] = self.run_id
        payload["ts"] = time.time()
        self._event_id += 1
        line = json.dumps(payload, ensure_ascii=True)
        self._fh.write(line + "\n")
        self._fh.flush()

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None
