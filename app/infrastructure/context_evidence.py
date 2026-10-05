"""Durable, buyer-isolated source records for context compression."""
from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Callable

from agentscope.tool import ToolBase, ToolChunk, ToolMiddlewareBase

from app.infrastructure.context import ShoppingContext


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", value)[:100] or "anonymous"


class ContextEvidenceStore:
    def __init__(self, data_dir: Path) -> None:
        self.root = data_dir / "context_evidence"
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, buyer_id: str, session_id: str, evidence_id: str) -> Path:
        return self.root / _safe(buyer_id) / _safe(session_id) / f"{_safe(evidence_id)}.json"

    def save(self, buyer_id: str, session_id: str, tool: str, args: dict, result: str) -> str:
        evidence_id = f"ev-{uuid.uuid4().hex}"
        path = self._path(buyer_id, session_id, evidence_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"evidence_id": evidence_id, "buyer_id": buyer_id, "session_id": session_id,
                   "tool": tool, "args": args, "result": result}
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(temp, path)
        return evidence_id

    def get(self, buyer_id: str, session_id: str, evidence_id: str) -> dict | None:
        path = self._path(buyer_id, session_id, evidence_id)
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("buyer_id") != buyer_id or payload.get("session_id") != session_id:
            return None
        return payload

    def list_recent(self, buyer_id: str, session_id: str, page: int = 1, limit: int = 5) -> list[dict]:
        limit = max(1, min(limit, 5))
        page = max(1, page)
        folder = self.root / _safe(buyer_id) / _safe(session_id)
        if not folder.is_dir():
            return []
        paths = sorted(folder.glob("ev-*.json"), key=lambda path: path.stat().st_mtime_ns, reverse=True)
        return [json.loads(path.read_text(encoding="utf-8"))
                for path in paths[(page - 1) * limit:page * limit]]


class EvidenceToolMiddleware(ToolMiddlewareBase):
    """Archive the complete final result before AgentScope truncates context."""

    def __init__(self, store: ContextEvidenceStore) -> None:
        self.store = store

    async def on_tool_call(
        self, tool: ToolBase, input_kwargs: dict[str, Any],
        next_handler: Callable[..., AsyncGenerator[ToolChunk, None]],
    ) -> AsyncGenerator[ToolChunk, None]:
        snapshot = ShoppingContext.current()
        async for chunk in next_handler(**input_kwargs):
            if snapshot is not None and chunk.is_last:
                text = "\n".join(str(getattr(block, "text", "")) for block in chunk.content)
                evidence_id = self.store.save(snapshot.buyer_id, snapshot.shopping_session_id,
                                              tool.name, input_kwargs, text)
                chunk.metadata["evidence_id"] = evidence_id
            yield chunk
