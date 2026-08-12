#!/usr/bin/env python3
"""Align the canonical Hunyuan mesh to metric frame-0 right-camera coordinates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vlm_sam2_recon.rigid_pipeline.common import (  # noqa: E402
    read_json,
    update_stage_state,
    write_json,
)
from vlm_sam2_recon.rigid_pipeline.geometry import backproject_depth  # noqa: E402
from vlm_sam2_recon.rigid_pipeline.mesh_alignment import (  # noqa: E402
    fixed_scale_icp,
    fixed_scale_point_to_plane_icp,
    pca_similarity_candidates,
    projection_diagnostics,
    refine_similarity_icp,
    symmetric_chamfer,
)


def parse_args() -> argparse.Namespace:
    workspace = PROJECT_ROOT / "run_rigid_20260715_215524"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=workspace)
    parser.add_argument("--mesh", type=Path, default=None)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--mask", type=Path, default=None)
    parser.add_argument("--depth", type=Path, default=None)
    parser.add_argument("--rgb", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--mesh-samples", type=int, default=40000)
    parser.add_argument("--observed-samples", type=int, default=24000)
    parser.add_argument("--pca-top-k", type=int, default=8)
    parser.add_argument("--sim3-iterations", type=int, default=25)
    parser.add_argument("--icp-iterations-per-level", type=int, default=15)
    parser.add_argument("--min-silhouette-iou", type=float, default=0.35)
    parser.add_argument(
        "--max-depth-rmse-m",
        type=float,
        default=0.08,
        help="Maximum 90%% trimmed depth RMSE; raw RMSE remains in diagnostics.",
    )
    parser.add_argument("--max-depth-median-abs-m", type=float, default=0.04)
    parser.add_argument("--max-chamfer-m", type=float, default=0.05)
    parser.add_argument("--max-point-plane-rmse-m", type=float, default=0.03)
    parser.add_argument("--min-rgb-edge-strength-ratio", type=float, default=0.45)
    parser.add_argument("--allow-qc-failure", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--depth-min-m", type=float, default=0.1)
    parser.add_argument("--depth-max-m", type=float, default=5.0)
    parser.add_argument("--object-depth-low-quantile", type=float, default=0.01)
    parser.add_argument("--object-depth-high-quantile", type=float, default=0.95)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--check", action="store_true", help="Validate inputs without aligning the mesh.")
    return parser.parse_args()


def resolve_mesh(workspace: Path, explicit: Path | None) -> Path:
    if explicit:
        path = explicit.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    candidates = []
    root = workspace / "outputs/05_hunyuan_mesh/whole"
    for pattern in ("*.glb", "*.obj", "*.ply", "*.stl", "*.fbx"):
        candidates.extend(sorted(root.glob(pattern)))
    if not candidates:
        raise FileNotFoundError(f"No Hunyuan mesh found in {root}; pass --mesh explicitly")
    return candidates[0]


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, process=False, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        geometries = [item for item in loaded.geometry.values() if isinstance(item, trimesh.Trimesh)]
        if not geometries:
            raise ValueError(f"Mesh scene is empty: {path}")
        loaded = trimesh.util.concatenate(geometries)
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        raise ValueError(f"Not a triangular mesh: {path}")
    return loaded


def clean_observed_points(points: np.ndarray, sample_count: int, seed: int) -> np.ndarray:
    if len(points) < 100:
        raise ValueError(f"Too few observed object points: {len(points)}")
    center = np.median(points, axis=0)
    distance = np.linalg.norm(points - center, axis=1)
    cutoff = float(np.quantile(distance, 0.985))
    points = points[distance <= cutoff]
    if len(points) > sample_count:
        rng = np.random.default_rng(seed)
        points = points[rng.choice(len(points), sample_count, replace=False)]
    return points


def save_overlay(
    rgb_path: Path,
    observed_mask: np.ndarray,
    rendered_mask: np.ndarray,
    output_path: Path,
) -> None:
    image = Image.open(rgb_path).convert("RGB")
    canvas = np.asarray(image).copy()
    observed_edge = observed_mask ^ (cv2.erode(observed_mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    rendered_edge = rendered_mask ^ (cv2.erode(rendered_mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    canvas[observed_edge] = np.asarray([40, 240, 80], dtype=np.uint8)
    canvas[rendered_edge] = np.asarray([255, 70, 40], dtype=np.uint8)
    Image.fromarray(canvas).save(output_path)


def rgb_silhouette_edge_diagnostics(
    rgb_path: Path,
    observed_mask: np.ndarray,
    rendered_mask: np.ndarray,
) -> dict:
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)
    gradient /= max(float(np.quantile(gradient, 0.98)), 1e-6)
    gradient = np.clip(gradient, 0.0, 1.0)
    kernel = np.ones((3, 3), np.uint8)
    observed_edge = observed_mask ^ (cv2.erode(observed_mask.astype(np.uint8), kernel) > 0)
    rendered_edge = rendered_mask ^ (cv2.erode(rendered_mask.astype(np.uint8), kernel) > 0)
    observed_strength = float(gradient[observed_edge].mean()) if np.any(observed_edge) else 0.0
    rendered_strength = float(gradient[rendered_edge].mean()) if np.any(rendered_edge) else 0.0
    return {
        "observed_mask_boundary_rgb_gradient": observed_strength,
        "rendered_boundary_rgb_gradient": rendered_strength,
        "rendered_to_observed_edge_strength_ratio": float(rendered_strength / max(observed_strength, 1e-6)),
        "observed_boundary_pixels": int(observed_edge.sum()),
        "rendered_boundary_pixels": int(rendered_edge.sum()),
    }


def main() -> int:
    args = parse_args()
    workspace = args.workspace.resolve()
    frame = args.frame_index
    if frame != 0:
        raise ValueError("Stage 07 defines C0 from frame 0; non-zero alignment frames require an explicit Ct-to-C0 conversion")
    mesh_path = resolve_mesh(workspace, args.mesh)
    mask_path = (args.mask or workspace / f"outputs/04_object_masks/combined/{frame:06d}.png").resolve()
    depth_path = (args.depth or workspace / f"outputs/06_dense_depth/metric_depth_npy/{frame:06d}.npy").resolve()
    rgb_path = (args.rgb or workspace / f"outputs/00_rgb_frames/right_rgb_png/{frame:06d}.png").resolve()
    camera_path = workspace / "outputs/00_rgb_frames/camera.json"
    output_dir = (args.output_dir or workspace / f"outputs/07_alignment/frame_{frame:06d}").resolve()
    for path in (mesh_path, mask_path, depth_path, rgb_path, camera_path):
        if not path.exists():
            raise FileNotFoundError(path)
    preflight = {
        "stage": "07_frame0_mesh_alignment",
        "mesh": str(mesh_path),
        "mask": str(mask_path),
        "depth": str(depth_path),
        "rgb": str(rgb_path),
        "camera": str(camera_path),
        "destination_frame": "frame0_right_camera_opencv_rdf" if frame == 0 else f"right_camera_frame_{frame:06d}",
    }
    if args.check:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        print("Stage 07 input check passed; mesh alignment was not run.")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    camera = read_json(camera_path)
    intrinsics = camera["rgb_intrinsics_right"]
    mask = np.asarray(Image.open(mask_path).convert("L")) > 127
    depth = np.load(depth_path).astype(np.float32)
    if depth.shape != mask.shape:
        raise ValueError(f"Depth/mask shape mismatch: {depth.shape} vs {mask.shape}")
    valid = (
        mask
        & np.isfinite(depth)
        & (depth >= args.depth_min_m)
        & (depth <= args.depth_max_m)
    )
    object_depths = depth[valid]
    if object_depths.size < 100:
        raise ValueError(f"Too few valid masked depth pixels: {object_depths.size}")
    low, high = np.quantile(
        object_depths,
        [args.object_depth_low_quantile, args.object_depth_high_quantile],
    )
    filtered_depth = np.where(valid & (depth >= low) & (depth <= high), depth, 0.0)
    observed_points, _ = backproject_depth(filtered_depth, intrinsics)
    observed_points = clean_observed_points(observed_points, args.observed_samples, args.random_seed)

    mesh = load_mesh(mesh_path)
    sample_count = min(args.mesh_samples, max(4000, len(mesh.faces) * 3))
    np.random.seed(args.random_seed)
    canonical_points, _ = trimesh.sample.sample_surface(mesh, sample_count)
    candidates = pca_similarity_candidates(canonical_points, observed_points, trim_fraction=0.68)
    refined = []
    for pca_score, initial in candidates[: args.pca_top_k]:
        result, history = refine_similarity_icp(
            canonical_points,
            observed_points,
            initial,
            iterations=args.sim3_iterations,
            trim_fraction=0.65,
        )
        score = symmetric_chamfer(result.transform(canonical_points), observed_points, 0.68)
        refined.append((score, result, pca_score, history))
    refined.sort(key=lambda item: item[0])
    sim3_score, sim3, pca_score, sim3_history = refined[0]
    point_to_point, icp_history = fixed_scale_icp(
        canonical_points,
        observed_points,
        sim3,
        iterations_per_level=args.icp_iterations_per_level,
        trim_fraction=0.7,
    )
    aligned, point_plane_history = fixed_scale_point_to_plane_icp(
        canonical_points,
        observed_points,
        point_to_point,
        iterations_per_level=max(6, args.icp_iterations_per_level),
    )
    aligned_points = aligned.transform(canonical_points)
    final_chamfer = symmetric_chamfer(aligned_points, observed_points, 0.68)
    diagnostics, rendered_mask, rendered_depth = projection_diagnostics(
        aligned_points,
        depth,
        mask,
        intrinsics,
    )
    rgb_edge_diagnostics = rgb_silhouette_edge_diagnostics(rgb_path, mask, rendered_mask)
    final_point_plane_rmse = (
        float(point_plane_history[-1]["point_to_plane_rmse_m"])
        if point_plane_history
        else None
    )
    quality_checks = {
        "silhouette_iou": {
            "value": diagnostics["silhouette_iou_point_splat"],
            "minimum": args.min_silhouette_iou,
            "passed": diagnostics["silhouette_iou_point_splat"] >= args.min_silhouette_iou,
        },
        "depth_trimmed_rmse_m": {
            "value": diagnostics["depth_trimmed_rmse_m"],
            "maximum": args.max_depth_rmse_m,
            "passed": diagnostics["depth_trimmed_rmse_m"] is not None and diagnostics["depth_trimmed_rmse_m"] <= args.max_depth_rmse_m,
        },
        "depth_median_abs_m": {
            "value": diagnostics["depth_median_abs_m"],
            "maximum": args.max_depth_median_abs_m,
            "passed": diagnostics["depth_median_abs_m"] is not None and diagnostics["depth_median_abs_m"] <= args.max_depth_median_abs_m,
        },
        "trimmed_chamfer_m": {
            "value": final_chamfer,
            "maximum": args.max_chamfer_m,
            "passed": final_chamfer <= args.max_chamfer_m,
        },
        "point_to_plane_rmse_m": {
            "value": final_point_plane_rmse,
            "maximum": args.max_point_plane_rmse_m,
            "passed": final_point_plane_rmse is not None and final_point_plane_rmse <= args.max_point_plane_rmse_m,
        },
        "rgb_silhouette_edge_strength_ratio": {
            "value": rgb_edge_diagnostics["rendered_to_observed_edge_strength_ratio"],
            "minimum": args.min_rgb_edge_strength_ratio,
            "passed": rgb_edge_diagnostics["rendered_to_observed_edge_strength_ratio"] >= args.min_rgb_edge_strength_ratio,
        },
    }
    quality_passed = all(item["passed"] for item in quality_checks.values())

    aligned_mesh = mesh.copy()
    aligned_mesh.apply_transform(aligned.matrix)
    aligned_obj = output_dir / "hunyuan_mesh_aligned_C0.obj"
    aligned_glb = output_dir / "hunyuan_mesh_aligned_C0.glb"
    aligned_mesh.export(aligned_obj)
    aligned_mesh.export(aligned_glb)
    trimesh.points.PointCloud(observed_points).export(output_dir / "observed_object_pointcloud_C0.ply")
    trimesh.points.PointCloud(aligned_points).export(output_dir / "aligned_mesh_samples_C0.ply")
    Image.fromarray(rendered_mask.astype(np.uint8) * 255).save(output_dir / "rendered_mask.png")
    np.save(output_dir / "rendered_depth.npy", rendered_depth)
    save_overlay(rgb_path, mask, rendered_mask, output_dir / "alignment_overlay.png")

    transform_record = {
        "name": "T_C0_from_O",
        "source_frame": "hunyuan_object_canonical",
        "destination_frame": "frame0_right_camera_opencv_rdf" if frame == 0 else f"right_camera_frame_{frame:06d}",
        "matrix_convention": "column_vector_left_multiply",
        "units": "meters",
        "scale": aligned.scale,
        "rotation": aligned.rotation,
        "translation_m": aligned.translation,
        "matrix_sim3": aligned.matrix,
    }
    write_json(output_dir / "T_C0_from_O.json", transform_record)
    np.save(output_dir / "T_C0_from_O.npy", aligned.matrix)
    report = {
        **preflight,
        "status": "completed" if quality_passed else "needs_revision",
        "observed": {
            "valid_mask_depth_pixels": int(valid.sum()),
            "point_count_after_filter": len(observed_points),
            "depth_quantile_range_m": [float(low), float(high)],
        },
        "alignment": {
            "pca_candidate_count": len(candidates),
            "pca_top_k_refined": min(args.pca_top_k, len(candidates)),
            "selected_pca_score_m": pca_score,
            "sim3_chamfer_m": sim3_score,
            "fixed_scale_final_chamfer_m": final_chamfer,
            "scale_locked_for_se3_icp": aligned.scale,
            "sim3_history": sim3_history,
            "fixed_scale_icp_history": icp_history,
            "fixed_scale_point_to_plane_history": point_plane_history,
            "projection_diagnostics": diagnostics,
            "rgb_edge_diagnostics": rgb_edge_diagnostics,
            "quality_checks": quality_checks,
            "quality_passed": quality_passed,
        },
        "transform": transform_record,
        "outputs": {
            "aligned_obj": str(aligned_obj),
            "aligned_glb": str(aligned_glb),
            "transform_json": str(output_dir / "T_C0_from_O.json"),
            "observed_pointcloud": str(output_dir / "observed_object_pointcloud_C0.ply"),
            "overlay": str(output_dir / "alignment_overlay.png"),
        },
    }
    write_json(output_dir / "alignment_report.json", report)
    update_stage_state(
        workspace / "pipeline_state.json",
        "07_frame0_mesh_alignment",
        "completed" if quality_passed else "needs_revision",
        inputs=[str(mesh_path), str(mask_path), str(depth_path), str(camera_path)],
        outputs=[str(output_dir)],
        notes=(
            f"Sim3 plus fixed-scale point-to-point/point-to-plane ICP passed QC; chamfer={final_chamfer:.6f} m."
            if quality_passed
            else f"Alignment written for inspection but failed one or more geometry gates; chamfer={final_chamfer:.6f} m."
        ),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value))
    return 0 if quality_passed or args.allow_qc_failure else 2


if __name__ == "__main__":
    raise SystemExit(main())
