#!/usr/bin/env python3
"""Align Hunyuan3D Particulate parts by fitting the base to RGB-D."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vlm_sam2_recon.stages.hunyuan3d_particulate_alignment import (  # noqa: E402
    DEFAULT_EXPORT_ROOT,
    HunyuanParticulateBaseAlignConfig,
    run_hunyuan_particulate_base_alignment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT_ROOT)
    parser.add_argument("--target-id", default="target_laptop")
    parser.add_argument("--align-frame", type=int, default=0)
    parser.add_argument("--view-frame", type=int, default=5)
    parser.add_argument("--convention", choices=["camera_to_rig", "rig_to_camera", "direct_same_camera"], default="camera_to_rig")
    parser.add_argument("--particulate-run-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--base-mask-path", type=Path, default=None)
    parser.add_argument("--whole-mask-path", type=Path, default=None)
    parser.add_argument("--depth-min-m", type=float, default=0.1)
    parser.add_argument("--depth-max-m", type=float, default=3.0)
    parser.add_argument("--observed-samples", type=int, default=14000)
    parser.add_argument("--canonical-samples-per-part", type=int, default=26000)
    parser.add_argument("--candidate-base-labels", default="14", help="Comma-separated labels. Default: Hunyuan laptop base part 14.")
    parser.add_argument("--ransac-iterations", type=int, default=900)
    parser.add_argument("--ransac-inlier-threshold-m", type=float, default=0.025)
    parser.add_argument("--icp-iterations", type=int, default=50)
    parser.add_argument("--icp-trim-fraction", type=float, default=0.68)
    parser.add_argument("--silhouette-scale-min-multiplier", type=float, default=1.0)
    parser.add_argument("--silhouette-scale-max-multiplier", type=float, default=5.5)
    parser.add_argument("--silhouette-scale-steps", type=int, default=51)
    parser.add_argument("--silhouette-shift-max-px", type=float, default=260.0)
    parser.add_argument("--silhouette-shift-steps", type=int, default=7)
    parser.add_argument("--silhouette-min-coverage", type=float, default=0.35)
    parser.add_argument("--silhouette-boundary-weight", type=float, default=0.0)
    parser.add_argument("--silhouette-depth-weight", type=float, default=1.0)
    parser.add_argument("--silhouette-inplane-rotation-max-deg", type=float, default=0.0)
    parser.add_argument("--silhouette-inplane-rotation-steps", type=int, default=1)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    labels = [item.strip() for item in args.candidate_base_labels.split(",") if item.strip()] if args.candidate_base_labels else None
    result = run_hunyuan_particulate_base_alignment(
        HunyuanParticulateBaseAlignConfig(
            project_root=args.project_root,
            export_root=args.export_root,
            target_id=args.target_id,
            align_frame=args.align_frame,
            view_frame=args.view_frame,
            convention=args.convention,
            particulate_run_dir=args.particulate_run_dir,
            output_dir=args.output_dir,
            base_mask_path=args.base_mask_path,
            whole_mask_path=args.whole_mask_path,
            depth_min_m=args.depth_min_m,
            depth_max_m=args.depth_max_m,
            observed_samples=args.observed_samples,
            canonical_samples_per_part=args.canonical_samples_per_part,
            candidate_base_labels=labels,
            ransac_iterations=args.ransac_iterations,
            ransac_inlier_threshold_m=args.ransac_inlier_threshold_m,
            icp_iterations=args.icp_iterations,
            icp_trim_fraction=args.icp_trim_fraction,
            silhouette_scale_min_multiplier=args.silhouette_scale_min_multiplier,
            silhouette_scale_max_multiplier=args.silhouette_scale_max_multiplier,
            silhouette_scale_steps=args.silhouette_scale_steps,
            silhouette_shift_max_px=args.silhouette_shift_max_px,
            silhouette_shift_steps=args.silhouette_shift_steps,
            silhouette_min_coverage=args.silhouette_min_coverage,
            silhouette_boundary_weight=args.silhouette_boundary_weight,
            silhouette_depth_weight=args.silhouette_depth_weight,
            silhouette_inplane_rotation_max_deg=args.silhouette_inplane_rotation_max_deg,
            silhouette_inplane_rotation_steps=args.silhouette_inplane_rotation_steps,
            random_seed=args.random_seed,
        )
    )
    cov = result["alignment"].get("silhouette_coverage") or {}
    print(f"Saved Hunyuan Particulate base alignment: {result['outputs']['result_dir']}/alignment_result.json")
    print(f"Base part: {result['base_part_label']}  Screen part: {result['screen_part_label']}")
    print(f"Method: {result['alignment']['method']}")
    print(f"Scale: {result['alignment']['scale']:.6f}")
    if cov:
        print(f"Base coverage={cov.get('coverage', float('nan')):.3f} IoU={cov.get('iou', float('nan')):.3f} precision={cov.get('precision', float('nan')):.3f}")
    print(f"Overlay: {result['outputs']['base_silhouette_overlay']['path']}")
    if result["outputs"].get("view"):
        print(f"View dir: {result['outputs']['view']['view_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
