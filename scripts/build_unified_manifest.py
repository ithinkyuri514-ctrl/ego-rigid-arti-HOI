#!/usr/bin/env python3
"""Bootstrap the unified project manifest from current script outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vlm_sam2_recon.adapters.legacy_outputs import build_manifest_from_legacy_outputs
from vlm_sam2_recon.manifest_io import validate_or_raise, write_project_manifest
from vlm_sam2_recon.paths import DEFAULT_MANIFEST_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build outputs/project_manifest.json from legacy outputs.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / DEFAULT_MANIFEST_PATH)
    parser.add_argument("--no-validate", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_manifest_from_legacy_outputs(args.project_root)
    if not args.no_validate:
        validate_or_raise(manifest)

    out_path = write_project_manifest(args.output, manifest)
    print(f"Saved unified manifest: {out_path}")
    print(
        "Summary: "
        f"{len(manifest.targets)} targets, "
        f"{len(manifest.masks)} masks, "
        f"{len(manifest.trellis_jobs)} trellis jobs, "
        f"{len(manifest.meshes)} meshes, "
        f"{len(manifest.articulation_joints)} joints, "
        f"{len(manifest.egoforce_sequences)} egoforce sequences"
    )
    issues = manifest.validate()
    if issues:
        print("Validation issues:")
        for issue in issues:
            print(f"  - {issue}")
        return 1
    print("Validation: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
