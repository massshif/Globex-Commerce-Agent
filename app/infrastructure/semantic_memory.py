"""Buyer scoped semantic memory with auditable facts and optimistic updates."""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class MemoryFact:
    memory_id: int
    user_id: str
    fact: str
    vector: tuple[float, ...]
    source: str
    version: int
    created_at: str


class MemoryConflict(ValueError):
    pass


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        return 0.0
    den = math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(x*x for x in b))
    return sum(x*y for x, y in zip(a, b)) / den if den else 0.0


class SemanticMemory:
    def __init__(self, database: Path, embed: Callable[[str], Sequence[float]] | None = None) -> None:
        database.parent.mkdir(parents=True, exist_ok=True)
        self._database = database
        self._embed = embed or (lambda text: [float(int.from_bytes(hashlib.sha256(text.encode()).digest()[i:i+4], "big"))
                                             for i in range(0, 16, 4)])
        with sqlite3.connect(database) as db:
            db.execute("""CREATE TABLE IF NOT EXISTS semantic_memories (
                memory_id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL,
                fact TEXT NOT NULL, vector TEXT NOT NULL, source TEXT NOT NULL,
                version INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL)""")

    def remember(self, user_id: str, fact: str, *, source: str = "explicit_input") -> MemoryFact:
        if not user_id or not fact.strip():
            raise ValueError("user_id and fact are required")
        vector = tuple(float(v) for v in self._embed(fact))
        with sqlite3.connect(self._database) as db:
            cur = db.execute("INSERT INTO semantic_memories(user_id,fact,vector,source,version,created_at) VALUES(?,?,?,?,1,?)",
                             (user_id, fact.strip(), json.dumps(vector), source, _now()))
            return MemoryFact(cur.lastrowid, user_id, fact.strip(), vector, source, 1, _now())

    def recall(self, user_id: str, query: str, *, top_k: int = 5, threshold: float = -1.0) -> list[MemoryFact]:
        q = tuple(float(v) for v in self._embed(query))
        with sqlite3.connect(self._database) as db:
            rows = db.execute("SELECT * FROM semantic_memories WHERE user_id=?", (user_id,)).fetchall()
        facts = [MemoryFact(r[0], r[1], r[2], tuple(json.loads(r[3])), r[4], r[5], r[6]) for r in rows]
        return [f for _, f in sorted((( _cosine(f.vector, q), f) for f in facts), key=lambda x: x[0], reverse=True) if _ >= threshold][:top_k]

    def update(self, memory_id: int, user_id: str, fact: str, *, expected_version: int, source: str = "explicit_input") -> MemoryFact:
        vector = tuple(float(v) for v in self._embed(fact))
        with sqlite3.connect(self._database) as db:
            row = db.execute("SELECT * FROM semantic_memories WHERE memory_id=? AND user_id=?", (memory_id, user_id)).fetchone()
            if row is None or row[5] != expected_version:
                raise MemoryConflict("memory version changed or does not belong to user")
            new_version = expected_version + 1
            db.execute("UPDATE semantic_memories SET fact=?, vector=?, source=?, version=?, created_at=? WHERE memory_id=? AND version=?",
                       (fact.strip(), json.dumps(vector), source, new_version, _now(), memory_id, expected_version))
        return MemoryFact(memory_id, user_id, fact.strip(), vector, source, new_version, _now())

    def delete(self, memory_id: int, user_id: str, *, expected_version: int) -> None:
        with sqlite3.connect(self._database) as db:
            cur = db.execute("DELETE FROM semantic_memories WHERE memory_id=? AND user_id=? AND version=?", (memory_id, user_id, expected_version))
            if cur.rowcount != 1:
                raise MemoryConflict("memory version changed or does not belong to user")
