"""Best effort context usage telemetry; unknown gateway fields remain unknown."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ContextUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_tokens: int | None = None
    latency_ms: float | None = None

    def attributes(self) -> dict[str, Any]:
        return {
            "context.input_tokens": self.input_tokens,
            "context.output_tokens": self.output_tokens,
            "context.cached_tokens": self.cached_tokens,
            "context.latency_ms": self.latency_ms,
        }


def usage_from_response(response: Any) -> ContextUsage:
    usage = getattr(response, "usage", None)
    if usage is None:
        return ContextUsage()
    def get(name: str):
        try:
            return getattr(usage, name, None)
        except Exception:
            return None
    return ContextUsage(get("input_tokens") or get("prompt_tokens"),
                        get("output_tokens") or get("completion_tokens"),
                        get("cached_tokens") or get("prompt_cache_hit_tokens"))
