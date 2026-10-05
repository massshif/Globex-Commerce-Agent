"""Run offline safety contracts and optionally paired current/candidate cases.

Real model evaluation is opt-in and requires an explicit evaluator; this command
never uploads buyer text or silently publishes a model result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contracts import freeze_inputs, offline_safety_contracts, verify_frozen
from .report import render_html, source_fingerprint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", default="eval/harness-input.json")
    parser.add_argument("--report", default="eval/harness-report.html")
    args = parser.parse_args()
    path = Path(args.inputs)
    if path.exists():
        frozen = verify_frozen(path)
    else:
        frozen = freeze_inputs(path, {"cases": [], "version": 1})
    report = {"frozen_input": frozen.fingerprint, "source_fingerprint": source_fingerprint(Path.cwd()),
              "contracts": offline_safety_contracts(), "cases": [],
              "publish": False, "reason": "offline contracts only; no model result"}
    render_html(report, args.report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
