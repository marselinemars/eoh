import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import ExperienceEntry


def _tokenize(text: str) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    return re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{2,}", text.lower())


def _safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


class ExperienceMemory:
    def __init__(
        self,
        store_path: str,
        max_entries: int = 5000,
        min_score: float = 0.0,
        include_failures: bool = True,
        read_enabled: bool = True,
        write_enabled: bool = True,
        read_only: bool = False,
    ) -> None:
        self.store_path = Path(store_path)
        self.max_entries = max(1, int(max_entries))
        self.min_score = float(min_score)
        self.include_failures = bool(include_failures)
        self.read_enabled = bool(read_enabled)
        self.write_enabled = bool(write_enabled)
        self.read_only = bool(read_only)
        self.entries: List[ExperienceEntry] = []
        self._entry_ids = set()

    def load(self) -> List[ExperienceEntry]:
        self.entries = []
        self._entry_ids = set()
        if not self.read_enabled:
            return self.entries
        if not self.store_path.exists():
            return self.entries
        with self.store_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    entry = ExperienceEntry.from_dict(obj)
                except Exception:
                    continue
                if not entry.entry_id or entry.entry_id in self._entry_ids:
                    continue
                self.entries.append(entry)
                self._entry_ids.add(entry.entry_id)
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries :]
            self._entry_ids = {entry.entry_id for entry in self.entries}
        return list(self.entries)

    def reset(self) -> None:
        self.entries = []
        self._entry_ids = set()
        if self.store_path.exists():
            self.store_path.unlink()

    def _append_line(self, entry: ExperienceEntry) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        with self.store_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry.to_dict(), ensure_ascii=True) + "\n")

    def _rewrite(self) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        with self.store_path.open("w", encoding="utf-8") as f:
            for entry in self.entries:
                f.write(json.dumps(entry.to_dict(), ensure_ascii=True) + "\n")

    def prune(self) -> None:
        if len(self.entries) <= self.max_entries:
            return
        self.entries = self.entries[-self.max_entries :]
        self._entry_ids = {entry.entry_id for entry in self.entries}
        if self.write_enabled and not self.read_only:
            self._rewrite()

    def add_entry(self, entry: ExperienceEntry) -> bool:
        if not self.write_enabled or self.read_only:
            return False
        if not entry.entry_id or entry.entry_id in self._entry_ids:
            return False
        self.entries.append(entry)
        self._entry_ids.add(entry.entry_id)
        self._append_line(entry)
        self.prune()
        return True

    def add_entries(self, entries: List[ExperienceEntry]) -> int:
        written = 0
        for entry in entries:
            if self.add_entry(entry):
                written += 1
        return written

    def _entry_text(self, entry: ExperienceEntry) -> str:
        parts = [
            entry.problem_tag,
            entry.operator,
            entry.heuristic_summary or "",
            entry.diagnosis_text or "",
            entry.failure_reason or "",
        ]
        behavior = " ".join(f"{k} {v}" for k, v in sorted((entry.behavior_stats or {}).items()))
        parts.append(behavior)
        code = entry.candidate_code or ""
        parts.append(code[:600])
        return "\n".join(part for part in parts if part)

    def _score_entry(self, entry: ExperienceEntry, query: Dict[str, Any]) -> Dict[str, Any]:
        score = 0.0
        reasons = []
        problem_tag = str(query.get("problem_tag", "") or "").strip()
        operator = str(query.get("operator", "") or "").strip()
        include_failures = bool(query.get("include_failures", self.include_failures))
        prefer_success = bool(query.get("prefer_success", True))

        if not include_failures and not entry.validity:
            return {"score": float("-inf"), "reasons": ["filtered_failure"]}
        if problem_tag and entry.problem_tag == problem_tag:
            score += 2.0
            reasons.append("problem_tag_match:+2.0")
        if operator and entry.operator == operator:
            score += 1.0
            reasons.append("operator_match:+1.0")

        query_text = str(query.get("text", "") or "")
        query_tokens = Counter(_tokenize(query_text))
        entry_tokens = Counter(_tokenize(self._entry_text(entry)))
        if query_tokens and entry_tokens:
            overlap = sum(min(query_tokens[token], entry_tokens[token]) for token in query_tokens.keys() & entry_tokens.keys())
            norm = max(1.0, float(sum(query_tokens.values())))
            token_score = min(3.0, 3.0 * overlap / norm)
            if token_score > 0:
                score += token_score
                reasons.append(f"token_overlap:+{token_score:.2f}")

        if prefer_success and entry.validity:
            fit = _safe_float(entry.fitness)
            if fit is not None:
                bonus = max(0.0, 1.5 - min(1.5, fit))
                if bonus > 0:
                    score += bonus
                    reasons.append(f"fitness_bonus:+{bonus:.2f}")
            if entry.entered_next_population:
                score += 0.5
                reasons.append("entered_next_population:+0.5")
        if not entry.validity:
            score += 0.3
            reasons.append("failure_lesson:+0.3")
        return {"score": score, "reasons": reasons}

    def query(
        self,
        context: Dict[str, Any],
        top_k: int = 5,
        min_score: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        if not self.read_enabled:
            return []
        threshold = self.min_score if min_score is None else float(min_score)
        scored = []
        for entry in self.entries:
            result = self._score_entry(entry, context)
            score = float(result.get("score", float("-inf")))
            if score < threshold:
                continue
            scored.append(
                {
                    "entry": entry,
                    "score": score,
                    "reasons": list(result.get("reasons", [])),
                }
            )
        scored.sort(
            key=lambda item: (
                -float(item["score"]),
                _safe_float(item["entry"].fitness) if item["entry"].fitness is not None else float("inf"),
                item["entry"].timestamp,
            )
        )
        return scored[: max(0, int(top_k))]

    def query_seed_entries(self, problem_tag: str, top_n: int = 2) -> List[Dict[str, Any]]:
        candidates = []
        for entry in self.entries:
            if not entry.validity:
                continue
            if problem_tag and entry.problem_tag != problem_tag:
                continue
            fit = _safe_float(entry.fitness)
            if fit is None:
                continue
            candidates.append(
                {
                    "entry": entry,
                    "score": (2.0 if entry.entered_next_population else 0.0) - fit,
                    "reasons": ["seed_candidate"],
                }
            )
        candidates.sort(key=lambda item: (-float(item["score"]), float(item["entry"].fitness)))
        return candidates[: max(0, int(top_n))]

    def summarize_entry(self, entry: ExperienceEntry) -> str:
        status = "success" if entry.validity else "failure"
        pieces = [f"{status}", f"op={entry.operator}"]
        if entry.fitness is not None:
            pieces.append(f"fitness={entry.fitness}")
        if entry.heuristic_summary:
            pieces.append(f"summary={entry.heuristic_summary[:140]}")
        if entry.diagnosis_text:
            pieces.append(f"diagnosis={entry.diagnosis_text[:140]}")
        if entry.failure_reason:
            pieces.append(f"failure={entry.failure_reason[:120]}")
        return "; ".join(pieces)

    def summarize_entries(self, entries: List[Dict[str, Any]], max_success: int = 2, max_failure: int = 1) -> str:
        if not entries:
            return ""
        success_lines = []
        failure_lines = []
        lessons = []
        for item in entries:
            entry = item["entry"]
            line = f"- score={item['score']:.2f}; {self.summarize_entry(entry)}"
            if entry.validity and len(success_lines) < max_success:
                success_lines.append(line)
            elif not entry.validity and len(failure_lines) < max_failure:
                failure_lines.append(line)
        if success_lines:
            lessons.append("Relevant past success cases:\n" + "\n".join(success_lines))
        if failure_lines:
            lessons.append("Relevant past failure cases:\n" + "\n".join(failure_lines))
            lessons.append("Lesson: avoid repeating the failure pattern above when the current context is similar.")
        return "\n".join(lessons)
