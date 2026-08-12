#!/usr/bin/env python3
"""Run local TRELLIS.2 mesh generation for prepared manifest jobs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vlm_sam2_recon.stages.trellis2_local import (
    DEFAULT_TRELLIS_ROOT,
    DEFAULT_WEIGHTS_DIR,
    build_config,
    config_to_dict,
    run_trellis2_jobs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate TRELLIS.2 GLB meshes from prepared cutout images.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--manifest", type=Path, default=None, help="Default: outputs/project_manifest.json")
    parser.add_argument("--trellis-root", type=Path, default=DEFAULT_TRELLIS_ROOT)
    parser.add_argument("--weights-dir", type=Path, default=DEFAULT_WEIGHTS_DIR)
    parser.add_argument("--output-root", type=Path, default=None, help="Default: outputs/trellis2_local")
    parser.add_argument("--upload-root", type=Path, default=None, help="Default: inputs/trellis2_meshes")
    parser.add_argument("--targets", default=None, help="Comma-separated target ids. Default: all trellis jobs.")
    parser.add_argument("--resolution", choices=["512", "1024", "1536"], default="1024")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decimation-target", type=int, default=500000)
    parser.add_argument("--texture-size", type=int, default=2048)
    parser.add_argument("--max-num-tokens", type=int, default=49152)
    parser.add_argument("--ss-steps", type=int, default=12)
    parser.add_argument("--shape-steps", type=int, default=12)
    parser.add_argument("--tex-steps", type=int, default=12)
    parser.add_argument("--ss-decoder", default=None)
    parser.add_argument("--dinov3-model", default=None)
    parser.add_argument("--rembg-model", default=None)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--load-rembg", action="store_true", help="Load TRELLIS.2 RMBG instead of using masked alpha only.")
    parser.add_argument("--no-preprocess", action="store_true", help="Skip TRELLIS.2 alpha-aware preprocessing.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--webp", action="store_true", help="Use WEBP textures during GLB export.")
    parser.add_argument("--no-webp", action="store_true", help="Keep compatibility with older commands; WEBP is off by default.")
    parser.add_argument("--print-config", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = build_config(args)
    if args.print_config:
        import json

        print(json.dumps(config_to_dict(config), indent=2, ensure_ascii=False))
    summary = run_trellis2_jobs(config)
    print(
        "TRELLIS.2 local generation complete: "
        f"{len(summary['results'])} mesh(es), summary={summary['results'][0]['output_glb'] if summary['results'] else 'none'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
