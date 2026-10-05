from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
from typing import Any


def source_fingerprint(root: str | Path) -> str:
    """Fingerprint source files for auditability; secrets and outputs are excluded."""
    base = Path(root)
    digest = hashlib.sha256()
    for path in sorted(base.rglob("*.py")):
        if any(part in {".venv", "__pycache__", "node_modules"} for part in path.parts):
            continue
        digest.update(str(path.relative_to(base)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def render_html(report: dict[str, Any], output: str | Path) -> Path:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = "".join(
        f"<tr><td>{html.escape(str(row.get('case_id', '')))}</td>"
        f"<td>{html.escape(str(row.get('current', '')))}</td>"
        f"<td>{html.escape(str(row.get('candidate', '')))}</td></tr>"
        for row in report.get("cases", [])
    )
    body = html.escape(json.dumps(report, ensure_ascii=False, indent=2))
    target.write_text(f"<!doctype html><meta charset='utf-8'><title>Globex evaluation</title>"
                      f"<h1>评测 × 可观测</h1><table><tr><th>case</th><th>current</th><th>candidate</th></tr>"
                      f"{rows}</table><h2>原始报告</h2><pre>{body}</pre>\n")
    return target
