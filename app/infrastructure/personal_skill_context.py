"""Versioned personal/public skills and optional append-only catalog snapshots.

The database is authoritative. Files named SKILL.md are never scanned as a
source of personal skills. A caller must check the current revision before use.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class SkillRevision:
    id: str
    version: int
    content_hash: str
    content: str
    owner_id: str
    visibility: str
    status: str
    updated_at: str


class SkillConflict(ValueError):
    pass


class PersonalSkillContext:
    def __init__(self, database: Path, *, append_only_catalog: bool = False) -> None:
        database.parent.mkdir(parents=True, exist_ok=True)
        self._database = database
        self.append_only_catalog = append_only_catalog
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS skill_revisions (
                    id TEXT NOT NULL, version INTEGER NOT NULL,
                    content_hash TEXT NOT NULL, content TEXT NOT NULL,
                    owner_id TEXT NOT NULL, visibility TEXT NOT NULL,
                    status TEXT NOT NULL, updated_at TEXT NOT NULL,
                    PRIMARY KEY (id, version)
                );
                CREATE TABLE IF NOT EXISTS skill_catalog_snapshots (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL, catalog_hash TEXT NOT NULL,
                    catalog_json TEXT NOT NULL, created_at TEXT NOT NULL
                );
            """)

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self._database, timeout=5)
        db.row_factory = sqlite3.Row
        return db

    @staticmethod
    def _revision(row: sqlite3.Row) -> SkillRevision:
        return SkillRevision(**dict(row))

    def save(self, skill_id: str, owner_id: str, content: str, *,
             visibility: str = "personal", expected_version: int | None = None) -> SkillRevision:
        if not skill_id or not owner_id or not content.strip():
            raise ValueError("id, owner_id and content are required")
        if visibility not in {"personal", "public"}:
            raise ValueError("visibility must be personal or public")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT * FROM skill_revisions WHERE id=? ORDER BY version DESC LIMIT 1", (skill_id,)).fetchone()
            current = old["version"] if old else 0
            if expected_version is not None and expected_version != current:
                raise SkillConflict(f"skill version changed: expected {expected_version}, current {current}")
            if old and (old["owner_id"] != owner_id or old["visibility"] != visibility):
                raise SkillConflict("skill owner or visibility cannot change")
            version = current + 1
            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
            status = "approved" if visibility == "personal" else "pending_review"
            stamp = _now()
            db.execute("INSERT INTO skill_revisions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                       (skill_id, version, digest, content, owner_id, visibility, status, stamp))
        return SkillRevision(skill_id, version, digest, content, owner_id, visibility, status, stamp)

    def approve(self, skill_id: str, version: int) -> None:
        with self._connect() as db:
            row = db.execute("SELECT visibility FROM skill_revisions WHERE id=? AND version=?", (skill_id, version)).fetchone()
            if not row or row["visibility"] != "public":
                raise ValueError("public revision not found")
            db.execute("UPDATE skill_revisions SET status='approved' WHERE id=? AND version=?", (skill_id, version))

    def current(self, skill_id: str, user_id: str) -> SkillRevision | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM skill_revisions WHERE id=? ORDER BY version DESC LIMIT 1", (skill_id,)).fetchone()
        if row is None or (row["visibility"] == "personal" and row["owner_id"] != user_id):
            return None
        return self._revision(row)

    def load(self, skill_id: str, user_id: str, *, expected_version: int | None = None) -> SkillRevision:
        item = self.current(skill_id, user_id)
        if item is None or item.status != "approved":
            raise ValueError("approved skill not found")
        if expected_version is not None and item.version != expected_version:
            raise SkillConflict(f"skill version changed: expected {expected_version}, current {item.version}")
        return item

    def catalog(self, user_id: str) -> list[SkillRevision]:
        with self._connect() as db:
            rows = db.execute("""SELECT s.* FROM skill_revisions s JOIN
                (SELECT id, MAX(version) AS version FROM skill_revisions GROUP BY id) c
                ON s.id=c.id AND s.version=c.version
                WHERE (s.visibility='personal' AND s.owner_id=?)
                   OR (s.visibility='public' AND s.status='approved') ORDER BY s.id""", (user_id,)).fetchall()
        return [self._revision(row) for row in rows]

    def snapshot_catalog_if_changed(self, user_id: str, *, missing: bool = False) -> bool:
        if not self.append_only_catalog:
            return False
        catalog = [{"id": s.id, "version": s.version, "content_hash": s.content_hash}
                   for s in self.catalog(user_id)]
        raw = json.dumps(catalog, ensure_ascii=False, sort_keys=True)
        digest = hashlib.sha256(raw.encode()).hexdigest()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            last = db.execute("SELECT catalog_hash FROM skill_catalog_snapshots WHERE user_id=? ORDER BY sequence DESC LIMIT 1", (user_id,)).fetchone()
            if last and last["catalog_hash"] == digest and not missing:
                return False
            db.execute("INSERT INTO skill_catalog_snapshots (user_id,catalog_hash,catalog_json,created_at) VALUES (?,?,?,?)",
                       (user_id, digest, raw, _now()))
        return True

    def catalog_history(self, user_id: str) -> list[dict]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM skill_catalog_snapshots WHERE user_id=? ORDER BY sequence", (user_id,)).fetchall()
        return [{**dict(row), "catalog": json.loads(row["catalog_json"])} for row in rows]
