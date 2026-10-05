from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ContextUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_tokens: int | None = None
    latency_ms: float | None = None
    tool_calls: int = 0

    def to_dict(self) -> dict[str, Any]:
        # Missing gateway usage stays null; never infer cache hits or billing.
        return asdict(self)


def record_call(*, input_tokens=None, output_tokens=None, cached_tokens=None,
                latency_ms=None, tool_calls=0) -> dict[str, Any]:
    return ContextUsage(input_tokens, output_tokens, cached_tokens, latency_ms, tool_calls).to_dict()
