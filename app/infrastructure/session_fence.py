"""Per-session revision and fencing for AgentState persistence.

The fence is deliberately outside AgentScope: AgentScope owns the in-memory
AgentState, while this component prevents concurrent turns for one session from
overwriting one another when API and worker tasks share a process.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field


@dataclass
class SessionFence:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    revision: int = 0

    def next_revision(self) -> int:
        self.revision += 1
        return self.revision

