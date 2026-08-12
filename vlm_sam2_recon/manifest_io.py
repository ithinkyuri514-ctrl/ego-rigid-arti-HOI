"""Read and write unified project manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schemas import ProjectManifest


def read_json(path: str | Path) -> dict[str, Any] | list[Any]:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: str | Path, data: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp_path.replace(path)
    return path


def read_project_manifest(path: str | Path) -> ProjectManifest:
    data = read_json(path)
    if not isinstance(data, dict):
        raise TypeError(f"Project manifest must be a JSON object: {path}")
    manifest = ProjectManifest.from_dict(data)
    if manifest is None:
        raise ValueError(f"Empty project manifest: {path}")
    return manifest


def write_project_manifest(path: str | Path, manifest: ProjectManifest) -> Path:
    manifest.touch()
    return write_json(path, manifest.to_dict())


def validate_or_raise(manifest: ProjectManifest) -> None:
    issues = manifest.validate()
    if issues:
        detail = "\n".join(f"- {item}" for item in issues)
        raise ValueError(f"Project manifest validation failed:\n{detail}")
