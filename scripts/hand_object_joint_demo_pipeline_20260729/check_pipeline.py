#!/usr/bin/env python3
"""Validate the recorded pipeline index and accepted artifacts."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> None:
    pipeline = json.loads((ROOT / "pipeline.json").read_text(encoding="utf-8"))
    workspace = Path(pipeline["workspace"])
    missing_scripts = []
    for stage in pipeline["stages"]:
        for value in stage["scripts"]:
            if not (ROOT / value).is_file():
                missing_scripts.append(value)
    missing_artifacts = [
        value for value in pipeline["accepted_artifacts"] if not (workspace / value).is_file()
    ]
    result = {
        "pipeline": pipeline["name"],
        "stage_count": len(pipeline["stages"]),
        "missing_scripts": missing_scripts,
        "missing_accepted_artifacts": missing_artifacts,
        "status": "ready" if not missing_scripts and not missing_artifacts else "incomplete",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["status"] == "ready" else 1)


if __name__ == "__main__":
    main()
