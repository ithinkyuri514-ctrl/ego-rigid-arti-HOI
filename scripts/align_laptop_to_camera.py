#!/usr/bin/env python3
"""Align the reconstructed laptop to the RGB right-camera frame using depth."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vlm_sam2_recon.stages.camera_alignment import AlignmentConfig, run_laptop_alignment


DEFAULT_EXPORT_ROOT = Path("/code/3DVideo_2026-07-01-19-24-06-667_spatialmp4_export")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Align target_laptop canonical parts to camera coordinates.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT_ROOT)
    parser.add_argument("--target-id", default="target_laptop")
    parser.add_argument("--align-frame", type=int, default=0)
    parser.add_argument("--view-frame", type=int, default=None)
    parser.add_argument("--convention", choices=["camera_to_rig", "rig_to_camera", "direct_same_camera"], default="camera_to_rig")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--particulate-run-path",
        type=Path,
        default=None,
        help="Path to particulate_run.json. Defaults to the original outputs/particulate run.",
    )
    parser.add_argument("--base-mask-path", type=Path, default=None)
    parser.add_argument("--depth-min-m", type=float, default=0.1)
    parser.add_argument("--depth-max-m", type=float, default=3.0)
    parser.add_argument("--depth-quantile-min", type=float, default=0.03)
    parser.add_argument("--depth-quantile-max", type=float, default=0.85)
    parser.add_argument("--canonical-samples", type=int, default=25000)
    parser.add_argument("--observed-samples", type=int, default=12000)
    parser.add_argument("--icp-trim-fraction", type=float, default=0.65)
    parser.add_argument("--icp-iterations", type=int, default=40)
    parser.add_argument("--silhouette-refine", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--silhouette-quantile-min", type=float, default=0.01)
    parser.add_argument("--silhouette-quantile-max", type=float, default=0.99)
    parser.add_argument("--silhouette-scale-min-multiplier", type=float, default=0.95)
    parser.add_argument("--silhouette-scale-max-multiplier", type=float, default=1.15)
    parser.add_argument("--silhouette-scale-steps", type=int, default=211)
    parser.add_argument("--silhouette-boundary-trim-fraction", type=float, default=0.85)
    parser.add_argument("--silhouette-outside-weight", type=float, default=12.0)
    parser.add_argument("--silhouette-boundary-weight", type=float, default=0.9)
    parser.add_argument("--silhouette-bbox-weight", type=float, default=0.10)
    parser.add_argument("--hinge-refine", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hinge-screen-part-label", default="15")
    parser.add_argument("--hinge-base-part-label", default="14")
    parser.add_argument("--hinge-angle-min-deg", type=float, default=-45.0)
    parser.add_argument("--hinge-angle-max-deg", type=float, default=45.0)
    parser.add_argument("--hinge-angle-steps", type=int, default=181)
    parser.add_argument("--hinge-trim-fraction", type=float, default=0.70)
    parser.add_argument("--hinge-plane-distance-weight", type=float, default=1.0)
    parser.add_argument("--hinge-nn-weight", type=float, default=0.15)
    parser.add_argument("--hinge-normal-weight-m-per-deg", type=float, default=0.004)
    parser.add_argument("--hinge-angle-regularizer-m-per-deg", type=float, default=0.00015)
    parser.add_argument(
        "--final-alignment-mode",
        choices=["base_first", "screen_first", "pca_constrained", "pca_direct", "free_icp", "refined"],
        default="base_first",
        help="base_first aligns part_14 then hinges part_15; screen_first aligns part_15 then hinges part_14. refined is a legacy alias for free_icp.",
    )
    parser.add_argument("--constrained-refine", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--constrained-iterations", type=int, default=20)
    parser.add_argument("--constrained-trim-fraction", type=float, default=0.75)
    parser.add_argument("--constrained-scale-min-multiplier", type=float, default=0.95)
    parser.add_argument("--constrained-scale-max-multiplier", type=float, default=1.05)
    parser.add_argument("--constrained-rotation-max-deg", type=float, default=5.0)
    parser.add_argument("--base-first-base-part-label", default="14")
    parser.add_argument("--base-first-screen-part-label", default="15")
    parser.add_argument("--screen-first-screen-part-label", default="15")
    parser.add_argument("--screen-first-base-part-label", default="14")
    parser.add_argument("--screen-first-axis-twist", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--screen-first-axis-twist-max-deg", type=float, default=60.0)
    parser.add_argument("--screen-projection-refine", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--screen-projection-scale-min-multiplier", type=float, default=0.94)
    parser.add_argument("--screen-projection-scale-max-multiplier", type=float, default=1.08)
    parser.add_argument("--screen-projection-shift-max-px", type=float, default=48.0)
    parser.add_argument("--screen-projection-depth-weight", type=float, default=0.25)
    parser.add_argument("--base-visible-surface-constrain", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--base-visible-surface-grid-px", type=int, default=4)
    parser.add_argument("--base-visible-surface-observed-to-model-weight", type=float, default=0.35)
    parser.add_argument("--base-visible-surface-plane-offset-weight", type=float, default=0.15)
    parser.add_argument("--base-visible-surface-normal-weight-m-per-deg", type=float, default=0.010)
    parser.add_argument("--base-visible-surface-snap-offset", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pca-direct-candidates", type=int, default=8)
    parser.add_argument("--pca-direct-screen-part-label", default="15")
    parser.add_argument("--pca-direct-base-part-label", default="14")
    parser.add_argument("--pca-direct-require-semantic-order", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--part-aware", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--screen-part-label", default="14")
    parser.add_argument("--base-part-label", default="15")
    parser.add_argument("--part-aware-candidates", type=int, default=16)
    parser.add_argument("--part-aware-trim-fraction", type=float, default=0.60)
    parser.add_argument("--part-aware-iterations", type=int, default=35)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_laptop_alignment(
        AlignmentConfig(
            project_root=args.project_root,
            export_root=args.export_root,
            target_id=args.target_id,
            align_frame=args.align_frame,
            view_frame=args.view_frame,
            convention=args.convention,
            output_root=args.output_root,
            particulate_run_path=args.particulate_run_path,
            base_mask_path=args.base_mask_path,
            depth_min_m=args.depth_min_m,
            depth_max_m=args.depth_max_m,
            depth_quantile_min=args.depth_quantile_min,
            depth_quantile_max=args.depth_quantile_max,
            canonical_samples=args.canonical_samples,
            observed_samples=args.observed_samples,
            icp_trim_fraction=args.icp_trim_fraction,
            icp_iterations=args.icp_iterations,
            silhouette_refine=args.silhouette_refine,
            silhouette_quantile_min=args.silhouette_quantile_min,
            silhouette_quantile_max=args.silhouette_quantile_max,
            silhouette_scale_min_multiplier=args.silhouette_scale_min_multiplier,
            silhouette_scale_max_multiplier=args.silhouette_scale_max_multiplier,
            silhouette_scale_steps=args.silhouette_scale_steps,
            silhouette_boundary_trim_fraction=args.silhouette_boundary_trim_fraction,
            silhouette_outside_weight=args.silhouette_outside_weight,
            silhouette_boundary_weight=args.silhouette_boundary_weight,
            silhouette_bbox_weight=args.silhouette_bbox_weight,
            hinge_refine=args.hinge_refine,
            hinge_screen_part_label=args.hinge_screen_part_label,
            hinge_base_part_label=args.hinge_base_part_label,
            hinge_angle_min_deg=args.hinge_angle_min_deg,
            hinge_angle_max_deg=args.hinge_angle_max_deg,
            hinge_angle_steps=args.hinge_angle_steps,
            hinge_trim_fraction=args.hinge_trim_fraction,
            hinge_plane_distance_weight=args.hinge_plane_distance_weight,
            hinge_nn_weight=args.hinge_nn_weight,
            hinge_normal_weight_m_per_deg=args.hinge_normal_weight_m_per_deg,
            hinge_angle_regularizer_m_per_deg=args.hinge_angle_regularizer_m_per_deg,
            final_alignment_mode=args.final_alignment_mode,
            base_first_base_part_label=args.base_first_base_part_label,
            base_first_screen_part_label=args.base_first_screen_part_label,
            screen_first_screen_part_label=args.screen_first_screen_part_label,
            screen_first_base_part_label=args.screen_first_base_part_label,
            screen_first_axis_twist=args.screen_first_axis_twist,
            screen_first_axis_twist_max_deg=args.screen_first_axis_twist_max_deg,
            screen_projection_refine=args.screen_projection_refine,
            screen_projection_scale_min_multiplier=args.screen_projection_scale_min_multiplier,
            screen_projection_scale_max_multiplier=args.screen_projection_scale_max_multiplier,
            screen_projection_shift_max_px=args.screen_projection_shift_max_px,
            screen_projection_depth_weight=args.screen_projection_depth_weight,
            base_visible_surface_constrain=args.base_visible_surface_constrain,
            base_visible_surface_grid_px=args.base_visible_surface_grid_px,
            base_visible_surface_observed_to_model_weight=args.base_visible_surface_observed_to_model_weight,
            base_visible_surface_plane_offset_weight=args.base_visible_surface_plane_offset_weight,
            base_visible_surface_normal_weight_m_per_deg=args.base_visible_surface_normal_weight_m_per_deg,
            base_visible_surface_snap_offset=args.base_visible_surface_snap_offset,
            constrained_refine=args.constrained_refine,
            constrained_iterations=args.constrained_iterations,
            constrained_trim_fraction=args.constrained_trim_fraction,
            constrained_scale_min_multiplier=args.constrained_scale_min_multiplier,
            constrained_scale_max_multiplier=args.constrained_scale_max_multiplier,
            constrained_rotation_max_deg=args.constrained_rotation_max_deg,
            pca_direct_candidates=args.pca_direct_candidates,
            pca_direct_screen_part_label=args.pca_direct_screen_part_label,
            pca_direct_base_part_label=args.pca_direct_base_part_label,
            pca_direct_require_semantic_order=args.pca_direct_require_semantic_order,
            part_aware=args.part_aware,
            screen_part_label=args.screen_part_label,
            base_part_label=args.base_part_label,
            part_aware_candidates=args.part_aware_candidates,
            part_aware_trim_fraction=args.part_aware_trim_fraction,
            part_aware_iterations=args.part_aware_iterations,
            random_seed=args.random_seed,
        )
    )
    print(f"Saved alignment: {result['outputs']['result_dir']}/alignment_result.json")
    print(f"Scale: {result['alignment']['scale']:.6f}")
    print(f"Method: {result['alignment']['method']}")
    print(f"Final metric: {result['alignment']['final_metric_m']:.6f} m")
    print(f"View frame: {result['view_frame']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
