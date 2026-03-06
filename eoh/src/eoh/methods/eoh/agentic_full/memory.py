import json
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

    def recent(self, n: int = 20) -> List[Dict[str, Any]]:
        if not self.memory_path.exists():
            return []
        lines = self.memory_path.read_text(encoding="utf-8").splitlines()
        out = []
        for line in lines[-max(1, int(n)):]:
            try:
                out.append(json.loads(line))
            except Exception:
                continue
        return out
