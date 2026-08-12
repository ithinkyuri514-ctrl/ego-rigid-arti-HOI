#!/usr/bin/env python3
"""Check native36 handoff code, tools, repositories, imports, and checkpoints."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_TOOLS = ("git", "ffmpeg", "ffprobe", "cmake", "ninja")
COMMON_IMPORTS = ("numpy", "scipy", "cv2", "PIL", "trimesh", "open3d", "torch", "torchvision", "yaml", "yacs", "viser", "requests")
DEFAULT_ROOTS = {
    "Qwen3-VL": "/code/Qwen3-VL",
    "SAM2": "/code/sam2",
    "SAM3D Objects": "/code/sam-3d-objects",
    "DiffuEraser": "/code/ArtHOI-4D-Reconstruction/third_party/diffueraser",
    "ArtHOI-4D-Reconstruction": "/code/ArtHOI-4D-Reconstruction",
    "Particulate": "/code/particulate",
    "EgoForce": "/code/EgoForce",
    "CoTracker": "/code/ArtHOI-4D-Reconstruction/third_party/co-tracker",
    "FoundationPose": "/code/ArtHOI-4D-Reconstruction/third_party/foundationpose",
    "SpatialMP4": "/code/SpatialMP4",
}
CHECKPOINTS = {
    "SAM3D Objects": "checkpoints/hf/ss_generator.ckpt",
    "Particulate": "PartField/model/model_objaverse.ckpt",
    "EgoForce": "_DATA/model_weights.pth",
    "CoTracker": "checkpoints/scaled_offline.pth",
    "FoundationPose": "weights/2024-01-11-20-02-45/model_best.pth",
}


def result(ok: bool, label: str, detail: str = "") -> bool:
    print(f"[{'OK' if ok else 'MISSING'}] {label}" + (f": {detail}" if detail else ""))
    return ok


def git_head(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    completed = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], text=True, capture_output=True)
    return completed.stdout.strip() if completed.returncode == 0 else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-imports", action="store_true", help="Fail when common imports are absent in the current environment.")
    parser.add_argument("--spatialmp4-python", type=Path, default=Path(os.environ.get("SPATIALMP4_PYTHON", "/code/SpatialMP4/build_spatialmp4_patched/python")))
    args = parser.parse_args()
    failures = 0

    code_list = PROJECT_ROOT / "docs/REQUIRED_CODE_NATIVE36_LEFT.txt"
    for line in code_list.read_text().splitlines():
        item = line.strip()
        if not item or item.startswith("#") or item.endswith("/"):
            continue
        failures += not result((PROJECT_ROOT / item).exists(), f"code {item}")
    failures += not result((PROJECT_ROOT / "vlm_sam2_recon").is_dir(), "code vlm_sam2_recon/")

    for tool in REQUIRED_TOOLS:
        failures += not result(shutil.which(tool) is not None, f"tool {tool}")

    lock = json.loads((PROJECT_ROOT / "environment/third_party.lock.json").read_text())
    for repo in lock["repositories"]:
        root = Path(os.environ.get(repo["name"].upper().replace("-", "_").replace(" ", "_") + "_ROOT", DEFAULT_ROOTS[repo["name"]]))
        exists = root.is_dir()
        failures += not result(exists, f"repo {repo['name']}", str(root))
        if exists:
            head = git_head(root)
            failures += not result(head == repo["commit"], f"commit {repo['name']}", head or "not a git checkout")
            checkpoint = CHECKPOINTS.get(repo["name"])
            if checkpoint:
                failures += not result((root / checkpoint).is_file(), f"checkpoint {repo['name']}", checkpoint)

    sam2_checkpoint = Path(os.environ.get(
        "SAM2_CHECKPOINT",
        "/code/ArtHOI-4D-Reconstruction/third_party/sam2/checkpoints/sam2.1_hiera_large.pt",
    ))
    failures += not result(sam2_checkpoint.is_file(), "checkpoint SAM2", str(sam2_checkpoint))

    spatial_path = args.spatialmp4_python.resolve()
    spatial_modules = list(spatial_path.glob("spatialmp4*.so")) if spatial_path.is_dir() else []
    failures += not result(bool(spatial_modules), "SpatialMP4 Python extension", str(spatial_path))

    import_failures = 0
    for module in COMMON_IMPORTS:
        import_failures += not result(importlib.util.find_spec(module) is not None, f"import {module}")
    if args.strict_imports:
        failures += import_failures
    elif import_failures:
        print("[INFO] Import failures are expected when running outside the stage-specific Conda environment.")

    print(f"\nSummary: {failures} required check(s) failed; {import_failures} current-environment import(s) unavailable.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
