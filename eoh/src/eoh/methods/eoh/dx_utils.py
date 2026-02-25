import ast
import hashlib
import json
import os
import re
import time

import numpy as np


HUNK_RE = re.compile(r"^@@\s*-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s*@@")


def sha1_text(text):
    if not isinstance(text, str):
        return None
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def extract_score_range(code):
    tree = ast.parse(code)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "score":
            start = int(node.lineno)
            end = int(getattr(node, "end_lineno", node.lineno))
            return start, end
    raise ValueError("score() function not found")


def _strip_line_end(text):
    return text.rstrip("\r\n")


def _parse_hunks(patch_text):
    lines = patch_text.splitlines()
    hunks = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("---") or line.startswith("+++") or line.startswith("diff --git"):
            i += 1
            continue
        m = HUNK_RE.match(line)
        if not m:
            i += 1
            continue

        old_start = int(m.group(1))
        old_count = int(m.group(2)) if m.group(2) else 1
        new_start = int(m.group(3))
        new_count = int(m.group(4)) if m.group(4) else 1
        i += 1
        body = []
        while i < len(lines) and not HUNK_RE.match(lines[i]):
            cur = lines[i]
            if cur.startswith(("diff --git", "---", "+++")):
                break
            body.append(cur)
            i += 1
        hunks.append((old_start, old_count, new_start, new_count, body))
    return hunks


def apply_unified_diff(source_text, patch_text):
    if not isinstance(patch_text, str) or len(patch_text.strip()) == 0:
        raise ValueError("empty patch")

    hunks = _parse_hunks(patch_text)
    if len(hunks) == 0:
        raise ValueError("no valid hunks in patch")

    source_lines = source_text.splitlines(keepends=True)
    newline = "\n"
    for line in source_lines:
        if line.endswith("\r\n"):
            newline = "\r\n"
            break
        if line.endswith("\n"):
            newline = "\n"
            break

    out = []
    cursor = 0
    changed_old_lines = set()
    insert_positions = set()

    for old_start, _, _, _, body in hunks:
        old_idx = max(old_start - 1, 0)
        if old_idx < cursor:
            raise ValueError("overlapping or unsorted hunks")
        out.extend(source_lines[cursor:old_idx])
        cursor = old_idx
        old_line_no = old_start

        for pline in body:
            if pline == r"\ No newline at end of file":
                continue
            if len(pline) == 0:
                raise ValueError("invalid hunk line")
            tag = pline[0]
            content = pline[1:]
            if tag == " ":
                if cursor >= len(source_lines):
                    raise ValueError("context past end of file")
                if _strip_line_end(source_lines[cursor]) != content:
                    raise ValueError("context mismatch while applying patch")
                out.append(source_lines[cursor])
                cursor += 1
                old_line_no += 1
            elif tag == "-":
                if cursor >= len(source_lines):
                    raise ValueError("delete past end of file")
                if _strip_line_end(source_lines[cursor]) != content:
                    raise ValueError("delete mismatch while applying patch")
                changed_old_lines.add(old_line_no)
                cursor += 1
                old_line_no += 1
            elif tag == "+":
                insert_positions.add(old_line_no)
                out.append(content + newline)
            else:
                raise ValueError("invalid patch line prefix")

    out.extend(source_lines[cursor:])
    return "".join(out), changed_old_lines, insert_positions


def patch_is_within_score(score_range, changed_old_lines, insert_positions):
    start, end = score_range
    for line_no in changed_old_lines:
        if line_no < start or line_no > end:
            return False
    for pos in insert_positions:
        if pos < start or pos > (end + 1):
            return False
    return True


def sanity_check_score(code_text, n_trials=8, eps=1e-10):
    namespace = {"np": np}
    exec(code_text, namespace)
    fn = namespace.get("score")
    if fn is None or not callable(fn):
        return False, "score_missing"

    rng = np.random.default_rng(2026)
    has_spread = False
    for _ in range(n_trials):
        item = int(rng.integers(1, 60))
        bins = rng.uniform(item + 0.05, 100.0, size=int(rng.integers(8, 20)))
        scores = fn(item, bins)
        arr = np.asarray(scores, dtype=float)
        if arr.shape != bins.shape:
            return False, "shape_mismatch"
        if not np.all(np.isfinite(arr)):
            return False, "non_finite_scores"
        if float(np.var(arr)) > eps:
            has_spread = True

    if not has_spread:
        return False, "flat_scores"
    return True, None


class DXArtifactLogger:
    def __init__(self, results_root, run_id, mode="compact"):
        self.results_root = os.path.normpath(results_root)
        self.run_id = run_id
        self.mode = mode if mode in ("compact", "full") else "compact"
        self._call_id = 0
        os.makedirs(self.results_root, exist_ok=True)

        if self.mode == "full":
            self.prompts_dir = os.path.join(self.results_root, "prompts", self.run_id)
            self.responses_dir = os.path.join(self.results_root, "responses", self.run_id)
            self.json_dir = os.path.join(self.results_root, "json", self.run_id)
            self.patches_dir = os.path.join(self.results_root, "patches", self.run_id)
            self.log_path = os.path.join(self.results_root, "log.jsonl")
            os.makedirs(self.prompts_dir, exist_ok=True)
            os.makedirs(self.responses_dir, exist_ok=True)
            os.makedirs(self.json_dir, exist_ok=True)
            os.makedirs(self.patches_dir, exist_ok=True)
        else:
            self.prompts_dir = None
            self.responses_dir = None
            self.json_dir = None
            self.patches_dir = None
            self.log_path = os.path.join(self.results_root, "dx_calls.jsonl")

    def log_call(
        self,
        agent,
        prompt_text,
        response_text,
        parsed_json,
        patch_text,
        metrics_snapshot,
        code_hash_before,
        code_hash_after,
        extra=None,
    ):
        event_id = self._call_id
        self._call_id += 1
        parsed_payload = parsed_json if parsed_json is not None else {}
        if self.mode == "full":
            base_name = f"{event_id}_{agent}"
            prompt_path = os.path.join(self.prompts_dir, base_name + ".txt")
            response_path = os.path.join(self.responses_dir, base_name + ".txt")
            json_path = os.path.join(self.json_dir, base_name + ".json")
            patch_path = os.path.join(self.patches_dir, base_name + ".patch")

            with open(prompt_path, "w", encoding="utf-8") as fh:
                fh.write(prompt_text if isinstance(prompt_text, str) else "")
            with open(response_path, "w", encoding="utf-8") as fh:
                fh.write(response_text if isinstance(response_text, str) else "")
            with open(json_path, "w", encoding="utf-8") as fh:
                json.dump(parsed_payload, fh, ensure_ascii=True, indent=2)

            if isinstance(patch_text, str) and len(patch_text.strip()) > 0:
                with open(patch_path, "w", encoding="utf-8") as fh:
                    fh.write(patch_text)
                patch_ref = patch_path
            else:
                patch_ref = None

            record = {
                "ts": time.time(),
                "run_id": self.run_id,
                "event_id": event_id,
                "agent": agent,
                "prompt_path": prompt_path,
                "response_path": response_path,
                "json_path": json_path,
                "patch_path": patch_ref,
                "code_hash_before": code_hash_before,
                "code_hash_after": code_hash_after,
                "metrics_snapshot": metrics_snapshot,
            }
            if isinstance(extra, dict):
                record.update(extra)
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=True) + "\n")

            return {
                "event_id": event_id,
                "prompt_path": prompt_path,
                "response_path": response_path,
                "json_path": json_path,
                "patch_path": patch_ref,
            }

        record = {
            "ts": time.time(),
            "run_id": self.run_id,
            "event_id": event_id,
            "agent": agent,
            "prompt": prompt_text if isinstance(prompt_text, str) else "",
            "response": response_text if isinstance(response_text, str) else "",
            "parsed_json": parsed_payload,
            "patch": patch_text if isinstance(patch_text, str) else None,
            "code_hash_before": code_hash_before,
            "code_hash_after": code_hash_after,
            "metrics_snapshot": metrics_snapshot,
        }
        if isinstance(extra, dict):
            record.update(extra)
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=True) + "\n")
        return {"event_id": event_id}
