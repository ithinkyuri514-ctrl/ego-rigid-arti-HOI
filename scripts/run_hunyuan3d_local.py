#!/usr/bin/env python3
"""Run Hunyuan-3D (腾讯云混元生3D) external-API mesh generation for prepared jobs.

Reuses the same cutout inputs as TRELLIS.2 (manifest.trellis_jobs), calls the
Tencent Cloud Hunyuan-3D gateway, downloads the mesh, and registers it in the
ProjectManifest. The API is already wired for the OpenAI-compatible gateway
(https://tokenhub.tencentmaas.com, Bearer key) in
vlm_sam2_recon/stages/hunyuan3d_client.py — you only need to supply the key.

Before running, either export your credentials:

    export HUNYUAN3D_API_KEY="sk-..."            # required
    export HUNYUAN3D_BASE_URL="https://tokenhub.tencentmaas.com"  # optional (this is the default)
    export HUNYUAN3D_MODEL="hy-3d-3.0"           # optional (or hy-3d-3.1)

or pass --api-key / --base-url / --model on the command line.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vlm_sam2_recon.stages.hunyuan3d_local import (
    build_config,
    config_to_dict,
    run_hunyuan3d_jobs,
)


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Hunyuan3D meshes from prepared cutout images.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Optional dotenv file. If omitted, .env.local and .env under project-root are loaded when present.",
    )
    parser.add_argument("--manifest", type=Path, default=None, help="Default: outputs/project_manifest.json")
    parser.add_argument("--output-root", type=Path, default=None, help="Default: outputs/hunyuan3d")
    parser.add_argument("--upload-root", type=Path, default=None, help="Default: inputs/hunyuan3d_meshes")
    parser.add_argument("--targets", default=None, help="Comma-separated target ids. Default: all mesh jobs.")
    # --- API config (falls back to HUNYUAN3D_* env vars) ---
    parser.add_argument("--base-url", default=None, help="Hunyuan3D API base URL (env HUNYUAN3D_BASE_URL).")
    parser.add_argument("--api-key", default=None, help="Hunyuan3D API key (env HUNYUAN3D_API_KEY).")
    parser.add_argument("--model", default=None, help="Optional model/version string (env HUNYUAN3D_MODEL).")
    parser.add_argument("--poll-interval", type=float, default=None, help="Seconds between status polls.")
    parser.add_argument("--poll-timeout", type=float, default=None, help="Max seconds to wait per job.")
    # --- behaviour ---
    parser.add_argument("--prefer-input", choices=["rgba", "white"], default="rgba")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--keep-mesh-id",
        action="store_true",
        help="Do NOT repoint target.mesh_id to the Hunyuan3D mesh (leave TRELLIS.2 as primary).",
    )
    parser.add_argument("--print-config", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    env_files = [args.env_file] if args.env_file else [args.project_root / ".env.local", args.project_root / ".env"]
    for env_file in env_files:
        if env_file is not None:
            load_env_file(env_file.expanduser().resolve())
    config = build_config(args)
    if args.print_config:
        import json

        print(json.dumps(config_to_dict(config), indent=2, ensure_ascii=False))
    summary = run_hunyuan3d_jobs(config)
    first = summary["results"][0]["upload_mesh"] if summary["results"] else "none"
    print(f"Hunyuan3D generation complete: {len(summary['results'])} mesh(es), first={first}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
