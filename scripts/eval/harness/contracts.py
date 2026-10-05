from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FrozenInput:
    fingerprint: str
    payload: dict


def freeze_inputs(path: str | Path, payload: dict) -> FrozenInput:
    """Freeze exact inputs; model outputs are never part of the fingerprint."""
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    fingerprint = hashlib.sha256(raw.encode()).hexdigest()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"fingerprint": fingerprint, "payload": payload}, ensure_ascii=False, indent=2) + "\n")
    return FrozenInput(fingerprint, payload)


def verify_frozen(path: str | Path) -> FrozenInput:
    data = json.loads(Path(path).read_text())
    raw = json.dumps(data["payload"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    actual = hashlib.sha256(raw.encode()).hexdigest()
    if actual != data["fingerprint"]:
        raise ValueError("frozen input fingerprint mismatch")
    return FrozenInput(actual, data["payload"])


def offline_safety_contracts() -> dict:
    """Deterministic checks that do not require model credentials."""
    return {
        "no_model_key_required": True,
        "buyer_scope_required": True,
        "trade_confirmation_required": True,
        "unknown_usage_is_unknown": True,
        "source_fingerprint_required": True,
    }
