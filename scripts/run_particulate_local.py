#!/usr/bin/env python3
"""Run or ingest local Particulate articulation inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vlm_sam2_recon.manifest_io import read_project_manifest, validate_or_raise
from vlm_sam2_recon.stages.particulate_local import (
    DEFAULT_PARTICULATE_ROOT,
    DEFAULT_PYTHON,
    build_config,
    config_to_dict,
    run_particulate_jobs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Particulate on TRELLIS meshes and update the project manifest.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--manifest", type=Path, default=None, help="Default: outputs/project_manifest.json")
    parser.add_argument("--particulate-root", type=Path, default=DEFAULT_PARTICULATE_ROOT)
    parser.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--output-root", type=Path, default=None, help="Default: outputs/particulate")
    parser.add_argument("--input-root", type=Path, default=None, help="Default: outputs/particulate_inputs")
    parser.add_argument("--targets", default=None, help="Comma-separated target ids. Default: articulated targets.")
    parser.add_argument("--up-dir", default="Z", choices=["X", "Y", "Z", "-X", "-Y", "-Z"])
    parser.add_argument("--num-points", type=int, default=102400)
    parser.add_argument("--target-faces", type=int, default=50000)
    parser.add_argument("--min-part-confidence", type=float, default=0.0)
    parser.add_argument("--no-strict", action="store_true", default=True)
    parser.add_argument("--strict", dest="no_strict", action="store_false")
    parser.add_argument("--no-export-urdf", action="store_true")
    parser.add_argument("--no-export-mjcf", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--allow-xet", action="store_true", help="Do not set HF_HUB_DISABLE_XET=1.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--ingest-only", action="store_true", help="Do not run infer.py; ingest existing outputs.")
    parser.add_argument("--no-validate", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = build_config(args)
    print(f"Particulate config: {config_to_dict(config)}", flush=True)
    summary = run_particulate_jobs(config)
    print(f"Saved Particulate summary: {summary['results'][0]['run_record'] if summary['results'] else 'none'}")
    print(f"Processed targets: {len(summary['results'])}")

    if not args.no_validate:
        manifest = read_project_manifest(config.manifest_path)
        validate_or_raise(manifest)
        print("Validation: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
