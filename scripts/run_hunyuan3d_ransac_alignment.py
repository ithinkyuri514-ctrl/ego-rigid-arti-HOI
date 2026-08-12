#!/usr/bin/env python3
"""Run Hunyuan3D whole-mesh alignment with RANSAC + ICP + mask-edge scoring."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vlm_sam2_recon.stages.hunyuan3d_local import build_config, run_hunyuan3d_jobs  # noqa: E402
from vlm_sam2_recon.stages.hunyuan3d_ransac_alignment import (  # noqa: E402
    DEFAULT_EXPORT_ROOT,
    HunyuanRansacAlignmentConfig,
    run_hunyuan_ransac_alignment,
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Optional dotenv file. If omitted, .env.local and .env under project-root are loaded when present.",
    )
    parser.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT_ROOT)
    parser.add_argument("--target-id", default="target_laptop")
    parser.add_argument("--mesh-path", type=Path, default=None, help="Existing Hunyuan3D mesh. If omitted, tries outputs/inputs hunyuan3d dirs.")
    parser.add_argument("--generate-hunyuan", action="store_true", help="Run Hunyuan3D API first if --mesh-path is not supplied.")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--align-frame", type=int, default=0)
    parser.add_argument("--view-frame", type=int, default=5)
    parser.add_argument("--convention", choices=["camera_to_rig", "rig_to_camera", "direct_same_camera"], default="camera_to_rig")
    parser.add_argument("--depth-min-m", type=float, default=0.1)
    parser.add_argument("--depth-max-m", type=float, default=3.0)
    parser.add_argument("--ransac-iterations", type=int, default=700)
    parser.add_argument("--ransac-inlier-threshold-m", type=float, default=0.025)
    parser.add_argument("--ransac-trim-fraction", type=float, default=0.60)
    parser.add_argument("--icp-iterations", type=int, default=45)
    parser.add_argument("--icp-trim-fraction", type=float, default=0.65)
    parser.add_argument("--edge-refine", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--edge-shift-max-px", type=float, default=60.0)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--hunyuan-api-key", default=None)
    parser.add_argument("--hunyuan-base-url", default=None)
    parser.add_argument("--hunyuan-model", default=None)
    parser.add_argument("--hunyuan-overwrite", action="store_true")
    return parser.parse_args()


def maybe_generate_hunyuan(args: argparse.Namespace) -> None:
    if args.mesh_path is not None or not args.generate_hunyuan:
        return
    if args.hunyuan_api_key:
        os.environ["HUNYUAN3D_API_KEY"] = args.hunyuan_api_key
    if args.hunyuan_base_url:
        os.environ["HUNYUAN3D_BASE_URL"] = args.hunyuan_base_url
    if args.hunyuan_model:
        os.environ["HUNYUAN3D_MODEL"] = args.hunyuan_model
    fake = argparse.Namespace(
        project_root=args.project_root,
        manifest=args.manifest,
        output_root=None,
        upload_root=None,
        targets=args.target_id,
        base_url=args.hunyuan_base_url,
        api_key=args.hunyuan_api_key,
        model=args.hunyuan_model,
        poll_interval=None,
        poll_timeout=None,
        prefer_input="rgba",
        overwrite=args.hunyuan_overwrite,
        keep_mesh_id=False,
    )
    run_hunyuan3d_jobs(build_config(fake))


def main() -> int:
    args = parse_args()
    env_files = [args.env_file] if args.env_file else [args.project_root / ".env.local", args.project_root / ".env"]
    for env_file in env_files:
        if env_file is not None:
            load_env_file(env_file.expanduser().resolve())
    maybe_generate_hunyuan(args)
    result = run_hunyuan_ransac_alignment(
        HunyuanRansacAlignmentConfig(
            project_root=args.project_root,
            export_root=args.export_root,
            target_id=args.target_id,
            mesh_path=args.mesh_path,
            align_frame=args.align_frame,
            view_frame=args.view_frame,
            convention=args.convention,
            output_dir=args.output_dir,
            depth_min_m=args.depth_min_m,
            depth_max_m=args.depth_max_m,
            ransac_iterations=args.ransac_iterations,
            ransac_inlier_threshold_m=args.ransac_inlier_threshold_m,
            ransac_trim_fraction=args.ransac_trim_fraction,
            icp_iterations=args.icp_iterations,
            icp_trim_fraction=args.icp_trim_fraction,
            edge_refine=args.edge_refine,
            edge_shift_max_px=args.edge_shift_max_px,
            random_seed=args.random_seed,
        )
    )
    print(f"Saved Hunyuan3D RANSAC alignment: {result['output_dir']}/alignment_result.json")
    print(f"Method: {result['alignment']['method']}")
    print(f"Scale: {result['alignment']['scale']:.6f}")
    print(f"Overlay: {result['outputs']['projection_overlay']['path']}")
    if result["outputs"].get("view"):
        print(f"View dir: {result['outputs']['view']['view_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
