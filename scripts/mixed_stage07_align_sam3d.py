#!/usr/bin/env python3
"""Refine SAM3D pose/scale against global frame-0 metric RGB-D and silhouette."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vlm_sam2_recon.rigid_pipeline.common import update_stage_state, write_json  # noqa: E402
from vlm_sam2_recon.rigid_pipeline.geometry import backproject_depth  # noqa: E402
from vlm_sam2_recon.rigid_pipeline.mesh_alignment import (  # noqa: E402
    Similarity,
    fixed_scale_icp,
    fixed_scale_point_to_plane_icp,
    refine_similarity_icp,
    symmetric_chamfer,
)


PYTORCH3D_TO_OPENCV = np.diag([-1.0, -1.0, 1.0, 1.0]).astype(np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--mesh-samples", type=int, default=40000)
    parser.add_argument("--observed-samples", type=int, default=24000)
    parser.add_argument("--sim3-iterations", type=int, default=20)
    parser.add_argument("--icp-iterations", type=int, default=12)
    parser.add_argument("--depth-min-m", type=float, default=0.15)
    parser.add_argument("--depth-max-m", type=float, default=4.0)
    parser.add_argument("--allow-qc-failure", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--object-id",
        action="append",
        default=None,
        help="Only align the named object; repeat for multiple objects. Existing summary entries are preserved.",
    )
    return parser.parse_args()


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        geometries = [geometry for geometry in mesh.geometry.values() if isinstance(geometry, trimesh.Trimesh)]
        mesh = trimesh.util.concatenate(geometries)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError(f"Invalid mesh: {path}")
    return mesh


def clean_observed(points: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    center = np.median(points, axis=0)
    distance = np.linalg.norm(points - center, axis=1)
    points = points[distance <= np.quantile(distance, 0.985)]
    if len(points) > maximum:
        rng = np.random.default_rng(seed)
        points = points[rng.choice(len(points), maximum, replace=False)]
    return points


def projection_metrics(
    points: np.ndarray,
    observed_depth: np.ndarray,
    observed_mask: np.ndarray,
    intrinsics: dict[str, float],
) -> tuple[dict, np.ndarray, np.ndarray]:
    height, width = observed_mask.shape
    points = np.asarray(points, dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1) & (points[:, 2] > 1e-5)]
    z = points[:, 2]
    u = np.rint(float(intrinsics["fx"]) * points[:, 0] / z + float(intrinsics["cx"])).astype(np.int32)
    v = np.rint(float(intrinsics["fy"]) * points[:, 1] / z + float(intrinsics["cy"])).astype(np.int32)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v, z = u[inside], v[inside], z[inside]
    rendered_depth = np.zeros((height, width), dtype=np.float32)
    if len(z):
        flat = v.astype(np.int64) * width + u
        order = np.argsort(z)
        sorted_flat = flat[order]
        _, first = np.unique(sorted_flat, return_index=True)
        chosen = order[first]
        rendered_depth.reshape(-1)[flat[chosen]] = z[chosen]
    raw_mask = rendered_depth > 0
    rendered_mask = cv2.dilate(raw_mask.astype(np.uint8), np.ones((7, 7), np.uint8), iterations=1) > 0
    rendered_mask = cv2.morphologyEx(
        rendered_mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=1
    ) > 0
    intersection = int(np.count_nonzero(rendered_mask & observed_mask))
    union = int(np.count_nonzero(rendered_mask | observed_mask))
    rendered_pixels = int(rendered_mask.sum())
    overlap_depth = raw_mask & observed_mask & np.isfinite(observed_depth) & (observed_depth > 0)
    residual = np.abs(rendered_depth[overlap_depth] - observed_depth[overlap_depth])
    depth_median = float(np.median(residual)) if residual.size else None
    iou = float(intersection / max(union, 1))
    outside = float(np.count_nonzero(rendered_mask & ~observed_mask) / max(rendered_pixels, 1))
    # IoU already penalizes rendered pixels outside the observed mask. Counting
    # outside_ratio again made thin objects trade a visibly worse silhouette for
    # millimetric depth improvements.
    objective = (1.0 - iou) + (depth_median if depth_median is not None else 0.25)
    return {
        "silhouette_iou": iou,
        "outside_ratio": outside,
        "intersection_pixels": intersection,
        "union_pixels": union,
        "rendered_pixels": rendered_pixels,
        "depth_overlap_pixels": int(overlap_depth.sum()),
        "depth_median_abs_m": depth_median,
        "objective": float(objective),
    }, rendered_mask, rendered_depth


def centered_initial(source: np.ndarray, target: np.ndarray) -> Similarity:
    source_center = np.median(source, axis=0)
    target_center = np.median(target, axis=0)
    source_radius = float(np.quantile(np.linalg.norm(source - source_center, axis=1), 0.85))
    target_radius = float(np.quantile(np.linalg.norm(target - target_center, axis=1), 0.85))
    scale = float(np.clip(target_radius / max(source_radius, 1e-8), 0.45, 2.2))
    return Similarity(scale, np.eye(3), target_center - scale * source_center)


def sam3d_to_calibrated_camera(
    pose: dict,
    intrinsics: dict[str, float],
    width: int,
    height: int,
) -> np.ndarray:
    """Correct SAM3D's inferred pinhole rays to the calibrated RGB camera rays."""
    normalized = np.asarray(pose["intrinsics_normalized"], dtype=np.float64)
    sam_fx = float(normalized[0, 0]) * width
    sam_fy = float(normalized[1, 1]) * height
    sam_cx = float(normalized[0, 2]) * width
    sam_cy = float(normalized[1, 2]) * height
    actual_fx = float(intrinsics["fx"])
    actual_fy = float(intrinsics["fy"])
    actual_cx = float(intrinsics["cx"])
    actual_cy = float(intrinsics["cy"])
    correction = np.eye(4, dtype=np.float64)
    correction[0, 0] = sam_fx / actual_fx
    correction[0, 2] = (sam_cx - actual_cx) / actual_fx
    correction[1, 1] = sam_fy / actual_fy
    correction[1, 2] = (sam_cy - actual_cy) / actual_fy
    return correction


def iou_refine(
    source: np.ndarray,
    initial: Similarity,
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsics: dict[str, float],
) -> tuple[Similarity, dict, np.ndarray, np.ndarray, list[dict]]:
    current = initial
    history = []
    for level, (scales, pixel_offsets, depth_offsets) in enumerate(
        [
            ((0.88, 0.94, 1.0, 1.06, 1.12), (-32, -16, 0, 16, 32), (-0.05, 0.0, 0.05)),
            ((0.97, 1.0, 1.03), (-8, 0, 8), (-0.015, 0.0, 0.015)),
        ]
    ):
        center_points = current.transform(source)
        median_z = float(np.median(center_points[:, 2]))
        candidates = []
        for scale_multiplier in scales:
            for du in pixel_offsets:
                for dv in pixel_offsets:
                    for dz in depth_offsets:
                        translation_delta = np.asarray(
                            [
                                du * median_z / float(intrinsics["fx"]),
                                dv * median_z / float(intrinsics["fy"]),
                                dz,
                            ],
                            dtype=np.float64,
                        )
                        candidate = Similarity(
                            current.scale * scale_multiplier,
                            current.rotation,
                            current.translation + translation_delta,
                        )
                        metrics, rendered_mask, rendered_depth = projection_metrics(
                            candidate.transform(source), depth, mask, intrinsics
                        )
                        candidates.append((metrics["objective"], candidate, metrics, rendered_mask, rendered_depth))
        candidates.sort(key=lambda item: item[0])
        _, current, metrics, rendered_mask, rendered_depth = candidates[0]
        history.append({"level": level, "candidate_count": len(candidates), **metrics})
    return current, metrics, rendered_mask, rendered_depth, history


def save_overlay(rgb_path: Path, observed: np.ndarray, rendered: np.ndarray, output: Path) -> None:
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
    canvas = rgb.astype(np.float32)
    observed_only = observed & ~rendered
    rendered_only = rendered & ~observed
    both = observed & rendered
    canvas[observed_only] = 0.55 * canvas[observed_only] + 0.45 * np.asarray([40, 220, 90])
    canvas[rendered_only] = 0.55 * canvas[rendered_only] + 0.45 * np.asarray([245, 80, 45])
    canvas[both] = 0.55 * canvas[both] + 0.45 * np.asarray([255, 210, 40])
    Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8)).save(output)


def main() -> int:
    args = parse_args()
    workspace = args.workspace.resolve()
    rgb_path = workspace / "outputs/00_rgb_frames/right_rgb_png/000000.png"
    depth_path = workspace / "outputs/06_dense_depth/metric_depth_npy/000000.npy"
    camera_path = workspace / "outputs/00_rgb_frames/camera.json"
    sam3d_summary_path = workspace / "outputs/03_sam3d_frame0/sam3d_frame0_summary.json"
    mask_summary_path = workspace / "outputs/02_sam2_frame0_masks/sam2_frame0_summary.json"
    for path in (rgb_path, depth_path, camera_path, sam3d_summary_path, mask_summary_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    camera = json.loads(camera_path.read_text(encoding="utf-8"))
    intrinsics = camera.get("rgb_intrinsics_selected", camera["rgb_intrinsics_right"])
    selected_eye = str(camera.get("selected_eye", "right")).lower()
    coordinate_frame = f"frame0_{selected_eye}_camera_opencv_rdf"
    depth = np.load(depth_path).astype(np.float32)
    sam3d_summary = json.loads(sam3d_summary_path.read_text(encoding="utf-8"))
    masks = {
        item["object_id"]: item
        for item in json.loads(mask_summary_path.read_text(encoding="utf-8"))["objects"]
    }
    reports = []
    for target_index, target in enumerate(sam3d_summary["objects"]):
        object_id = target["object_id"]
        if args.object_id is not None and object_id not in args.object_id:
            continue
        mask_path = Path(masks[object_id]["mask"])
        mask = np.asarray(Image.open(mask_path).convert("L")) > 127
        valid = mask & np.isfinite(depth) & (depth >= args.depth_min_m) & (depth <= args.depth_max_m)
        values = depth[valid]
        if values.size < 100:
            raise ValueError(f"{object_id}: too few masked depth pixels: {values.size}")
        low, high = np.quantile(values, [0.01, 0.92])
        filtered = np.where(valid & (depth >= low) & (depth <= high), depth, 0.0)
        observed, _ = backproject_depth(filtered, intrinsics)
        observed = clean_observed(observed, args.observed_samples, args.random_seed + target_index)

        pose_path = Path(target["pose"])
        pose = json.loads(pose_path.read_text(encoding="utf-8"))
        posed_mesh = load_mesh(Path(target["mesh_posed_sam3d_camera"]))
        posed_mesh.apply_transform(PYTORCH3D_TO_OPENCV)
        intrinsics_correction = sam3d_to_calibrated_camera(
            pose, intrinsics, depth.shape[1], depth.shape[0]
        )
        calibrated_mesh = posed_mesh.copy()
        calibrated_mesh.apply_transform(intrinsics_correction)
        np.random.seed(args.random_seed + target_index)
        sample_count = min(args.mesh_samples, max(8000, len(calibrated_mesh.faces) * 2))
        source, _ = trimesh.sample.sample_surface(calibrated_mesh, sample_count)
        _, _, source_depth = projection_metrics(source, depth, mask, intrinsics)
        ray_overlap = (
            (source_depth > 0)
            & mask
            & np.isfinite(depth)
            & (depth >= args.depth_min_m)
            & (depth <= args.depth_max_m)
        )
        if np.count_nonzero(ray_overlap) < 100:
            raise ValueError(f"{object_id}: too few rendered depth pixels for metric ray scale")
        ray_scale_samples = depth[ray_overlap] / source_depth[ray_overlap]
        ray_scale_samples = ray_scale_samples[np.isfinite(ray_scale_samples)]
        low_scale, high_scale = np.quantile(ray_scale_samples, [0.1, 0.9])
        metric_ray_scale = float(
            np.median(ray_scale_samples[(ray_scale_samples >= low_scale) & (ray_scale_samples <= high_scale)])
        )
        initial = Similarity(metric_ray_scale, np.eye(3), np.zeros(3))
        centered = centered_initial(source, observed)
        sim3, sim3_history = refine_similarity_icp(
            source, observed, centered, iterations=args.sim3_iterations, trim_fraction=0.58
        )
        p2p, p2p_history = fixed_scale_icp(
            source, observed, sim3, iterations_per_level=args.icp_iterations, trim_fraction=0.62
        )
        try:
            p2plane, p2plane_history = fixed_scale_point_to_plane_icp(
                source,
                observed,
                p2p,
                iterations_per_level=max(6, args.icp_iterations // 2),
                trim_fraction=0.68,
            )
        except Exception as exc:
            p2plane, p2plane_history = p2p, [{"error": repr(exc)}]

        candidates = []
        for name, candidate in (("sam3d_intrinsics_metric", initial), ("center_scale", centered), ("sim3_icp", sim3), ("point_icp", p2p), ("plane_icp", p2plane)):
            metrics, rendered, rendered_depth = projection_metrics(candidate.transform(source), depth, mask, intrinsics)
            chamfer = symmetric_chamfer(candidate.transform(source), observed, 0.62)
            metrics["trimmed_chamfer_m"] = float(chamfer)
            metrics["selection_score"] = float(metrics["objective"] + 3.0 * chamfer)
            candidates.append((metrics["selection_score"], name, candidate, metrics, rendered, rendered_depth))
        candidates.sort(key=lambda item: item[0])
        _, selected_name, selected, selected_metrics, _, _ = candidates[0]
        refined, final_metrics, final_mask, final_depth, iou_history = iou_refine(
            source, selected, depth, mask, intrinsics
        )
        final_chamfer = symmetric_chamfer(refined.transform(source), observed, 0.62)
        final_metrics["trimmed_chamfer_m"] = float(final_chamfer)

        raw_source, _ = trimesh.sample.sample_surface(posed_mesh, sample_count)
        initial_metrics, initial_mask, _ = projection_metrics(raw_source, depth, mask, intrinsics)
        calibrated_initial_metrics, _, _ = projection_metrics(
            initial.transform(source), depth, mask, intrinsics
        )
        output_dir = workspace / f"outputs/07_alignment/{object_id}/frame_000000"
        output_dir.mkdir(parents=True, exist_ok=True)
        initial_mesh = posed_mesh.copy()
        initial_mesh.export(output_dir / "sam3d_initial_C0.glb")
        aligned_mesh = calibrated_mesh.copy()
        aligned_mesh.apply_transform(refined.matrix)
        aligned_mesh.export(output_dir / "sam3d_aligned_C0.glb")
        aligned_mesh.export(output_dir / "sam3d_aligned_C0.ply")
        trimesh.points.PointCloud(observed).export(output_dir / "observed_object_pointcloud_C0.ply")
        save_overlay(rgb_path, mask, initial_mask, output_dir / "sam3d_initial_overlay.png")
        save_overlay(rgb_path, mask, final_mask, output_dir / "alignment_overlay.png")
        Image.fromarray(final_mask.astype(np.uint8) * 255).save(output_dir / "rendered_mask.png")
        np.save(output_dir / "rendered_depth.npy", final_depth)

        sam3d_mesh_to_camera = np.asarray(pose["mesh_zup_to_camera_matrix_column_vector"], dtype=np.float64)
        canonical_to_c0 = (
            refined.matrix
            @ intrinsics_correction
            @ PYTORCH3D_TO_OPENCV
            @ sam3d_mesh_to_camera
        )
        transform = {
            "source_frame": "sam3d_canonical_z_up",
            "destination_frame": coordinate_frame,
            "matrix_convention": "column_vector_left_multiply",
            "T_C0_from_sam3d_canonical": canonical_to_c0,
            "T_C0_from_sam3d_posed_pytorch3d": (
                refined.matrix @ intrinsics_correction @ PYTORCH3D_TO_OPENCV
            ),
            "sam3d_to_calibrated_camera_matrix": intrinsics_correction,
            "intrinsics_correction_policy": "pinhole ray remap from SAM3D inferred K to calibrated selected-eye K",
            "sam3d_initial_scale_xyz": pose.get("scale_xyz"),
            "sam3d_initial_translation": pose.get("translation"),
            "refinement_scale_multiplier": refined.scale,
            "refinement_rotation": refined.rotation,
            "refinement_translation_m": refined.translation,
        }
        report = {
            "object_id": object_id,
            "object_class": target["object_class"],
            "frame_index": 0,
            "coordinate_frame": coordinate_frame,
            "inputs": {"rgb": str(rgb_path), "mask": str(mask_path), "depth": str(depth_path), "sam3d_pose": str(pose_path)},
            "observed": {
                "valid_mask_depth_pixels": int(valid.sum()),
                "point_count": int(len(observed)),
                "depth_quantile_range_m": [float(low), float(high)],
            },
            "sam3d_initial_metrics": initial_metrics,
            "calibrated_metric_initial_metrics": calibrated_initial_metrics,
            "icp_candidates": [
                {"name": name, **metrics}
                for _, name, _, metrics, _, _ in candidates
            ],
            "selected_pre_iou_candidate": selected_name,
            "sim3_history": sim3_history,
            "fixed_scale_icp_history": p2p_history,
            "point_to_plane_history": p2plane_history,
            "iou_refinement_history": iou_history,
            "final_metrics": final_metrics,
            "transform": transform,
            "outputs": {
                "initial_mesh": str(output_dir / "sam3d_initial_C0.glb"),
                "aligned_mesh": str(output_dir / "sam3d_aligned_C0.glb"),
                "observed_pointcloud": str(output_dir / "observed_object_pointcloud_C0.ply"),
                "initial_overlay": str(output_dir / "sam3d_initial_overlay.png"),
                "alignment_overlay": str(output_dir / "alignment_overlay.png"),
            },
        }
        report_path = output_dir / "alignment_report.json"
        write_json(report_path, report)
        reports.append({"object_id": object_id, "report": str(report_path), **report["outputs"], "final_metrics": final_metrics})
        print(
            f"{object_id}: IoU {initial_metrics['silhouette_iou']:.3f} -> {final_metrics['silhouette_iou']:.3f}, "
            f"chamfer={final_chamfer:.4f}m",
            flush=True,
        )
    summary_path = workspace / "outputs/07_alignment/alignment_summary.json"
    if args.object_id is not None and summary_path.is_file():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        updated_ids = {item["object_id"] for item in reports}
        reports = [
            item for item in previous.get("objects", []) if item.get("object_id") not in updated_ids
        ] + reports
    summary = {
        "stage": "07_frame0_multi_object_alignment",
        "status": "completed",
        "coordinate_frame": coordinate_frame,
        "objects": reports,
    }
    write_json(summary_path, summary)
    update_stage_state(
        workspace / "pipeline_state.json",
        "07_frame0_multi_object_alignment",
        "completed",
        inputs=[str(sam3d_summary_path), str(mask_summary_path), str(depth_path), str(camera_path)],
        outputs=[str(summary_path)],
        notes=f"Refined SAM3D pose/scale for {len(reports)} objects with frame-0 metric RGB-D ICP and silhouette IoU.",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
