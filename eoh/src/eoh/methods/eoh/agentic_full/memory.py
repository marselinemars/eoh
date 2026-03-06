import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


class AgenticMemory:
    """Simple persistent JSONL memory store for reusable patterns."""

    def __init__(self, memory_path: str):
        self.memory_path = Path(memory_path)
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.memory_path.exists():
            self.memory_path.write_text("", encoding="utf-8")

    def _now(self) -> str:
        return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    def append(self, record: Dict[str, Any]):
        payload = dict(record)
        payload["time"] = self._now()
        with self.memory_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

    def append_many(self, records: List[Dict[str, Any]]):
        for rec in records:
            self.append(rec)

    def _read_all(self) -> List[Dict[str, Any]]:
        if not self.memory_path.exists():
            return []
        lines = self.memory_path.read_text(encoding="utf-8").splitlines()
        out = []
        for line in lines:
            try:
                out.append(json.loads(line))
            except Exception:
                continue
        return out

    def recent(self, n: int = 20) -> List[Dict[str, Any]]:
        records = self._read_all()
        return records[-max(1, int(n)) :]

    def _record_text(self, record: Dict[str, Any]) -> str:
        fragments = []
        for key in [
            "kind",
            "diagnosis_id",
            "portfolio_id",
            "search_regime",
            "summary",
            "label",
            "payload",
            "lessons",
            "observed_effects",
            "supported_hypotheses",
        ]:
            value = record.get(key)
            if value is None:
                continue
            if isinstance(value, (dict, list)):
                fragments.append(json.dumps(value, ensure_ascii=True, sort_keys=True))
            else:
                fragments.append(str(value))
        return " ".join(fragments)

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-z0-9_\\.]+", str(text).lower())

    def retrieve(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        query_tokens = set(self._tokenize(query))
        if len(query_tokens) == 0:
            return []
        scored = []
        all_records = self._read_all()
        total = max(1, len(all_records))
        for idx, record in enumerate(all_records):
            record_text = self._record_text(record)
            record_tokens = set(self._tokenize(record_text))
            if len(record_tokens) == 0:
                continue
            overlap = len(query_tokens.intersection(record_tokens))
            if overlap <= 0:
                continue
            recency_bonus = float(idx + 1) / float(total)
            score = float(overlap) + 0.1 * recency_bonus
            enriched = dict(record)
            enriched["_retrieval_score"] = score
            scored.append(enriched)
        scored.sort(key=lambda item: item.get("_retrieval_score", 0.0), reverse=True)
        return scored[: max(1, int(limit))]
