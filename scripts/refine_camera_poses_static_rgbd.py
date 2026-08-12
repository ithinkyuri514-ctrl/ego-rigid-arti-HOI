#!/usr/bin/env python3
"""Refine a camera trajectory from masked static RGB-D geometry."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d
from PIL import Image
from scipy.ndimage import binary_dilation
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE = PROJECT_ROOT / "run_mixed_20260728_203734_depth40"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_number_list(text: str, cast: type) -> tuple[Any, ...]:
    values = tuple(cast(value.strip()) for value in text.split(",") if value.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected a comma-separated non-empty list")
    return values


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def rotation_degrees(matrix: np.ndarray) -> float:
    return float(np.degrees(Rotation.from_matrix(matrix[:3, :3]).magnitude()))


def transform_step(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    delta = candidate @ np.linalg.inv(reference)
    return float(np.linalg.norm(delta[:3, 3])), rotation_degrees(delta)


def transform_residual(measured: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    error = np.linalg.inv(measured) @ predicted
    return np.concatenate(
        [error[:3, 3], Rotation.from_matrix(error[:3, :3]).as_rotvec()]
    )


def correction_matrix(vector: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_rotvec(vector[3:]).as_matrix()
    matrix[:3, 3] = vector[:3]
    return matrix


def default_mask_dirs(workspace: Path) -> list[Path]:
    return [
        workspace / "outputs/02_hand_masks/combined",
        workspace / "outputs/04_object_masks/bottle/combined",
        workspace / "outputs/04_object_masks/microwave/combined",
    ]


def load_static_cloud(
    depth_path: Path,
    mask_paths: list[Path],
    intrinsics: dict[str, float],
    depth_min: float,
    depth_max: float,
    mask_dilation: int,
) -> tuple[o3d.geometry.PointCloud, dict[str, Any]]:
    depth = np.load(depth_path).astype(np.float32, copy=False)
    invalid = np.zeros(depth.shape, dtype=bool)
    for mask_path in mask_paths:
        if not mask_path.is_file():
            raise FileNotFoundError(mask_path)
        mask = np.asarray(Image.open(mask_path).convert("L")) > 127
        if mask.shape != depth.shape:
            raise ValueError(f"Mask/depth shape mismatch: {mask_path} {mask.shape}/{depth.shape}")
        invalid |= mask
    if mask_dilation > 0:
        invalid = binary_dilation(invalid, iterations=mask_dilation)

    valid = (
        np.isfinite(depth)
        & (depth >= depth_min)
        & (depth <= depth_max)
        & ~invalid
    )
    v, u = np.nonzero(valid)
    z = depth[v, u].astype(np.float64)
    x = (u - float(intrinsics["cx"])) * z / float(intrinsics["fx"])
    y = (v - float(intrinsics["cy"])) * z / float(intrinsics["fy"])
    points = np.column_stack([x, y, z])
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    return cloud, {
        "depth": str(depth_path),
        "depth_pixels": int(depth.size),
        "static_valid_pixels": int(valid.sum()),
        "static_valid_ratio": float(valid.mean()),
    }


def cloud_pyramid(
    cloud: o3d.geometry.PointCloud, voxel_sizes: tuple[float, ...]
) -> list[o3d.geometry.PointCloud]:
    levels: list[o3d.geometry.PointCloud] = []
    for voxel_size in voxel_sizes:
        level = cloud.voxel_down_sample(voxel_size)
        level.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=max(2.5 * voxel_size, 0.025), max_nn=40
            )
        )
        levels.append(level)
    return levels


def register_pair(
    source_levels: list[o3d.geometry.PointCloud],
    target_levels: list[o3d.geometry.PointCloud],
    initial: np.ndarray,
    max_distances: tuple[float, ...],
    iterations: tuple[int, ...],
) -> tuple[np.ndarray, dict[str, Any]]:
    if not (len(source_levels) == len(target_levels) == len(max_distances) == len(iterations)):
        raise ValueError("ICP pyramid parameter lengths do not match")
    transform = initial.copy()
    history: list[dict[str, Any]] = []
    for level_index, (source, target, max_distance, iteration_count) in enumerate(
        zip(source_levels, target_levels, max_distances, iterations)
    ):
        if level_index == 0:
            estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        else:
            loss = o3d.pipelines.registration.TukeyLoss(k=max_distance)
            estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
        result = o3d.pipelines.registration.registration_icp(
            source,
            target,
            max_distance,
            transform,
            estimation,
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-7,
                relative_rmse=1e-7,
                max_iteration=iteration_count,
            ),
        )
        transform = np.asarray(result.transformation, dtype=np.float64)
        history.append(
            {
                "level": level_index,
                "max_correspondence_distance_m": max_distance,
                "fitness": float(result.fitness),
                "inlier_rmse_m": float(result.inlier_rmse),
            }
        )

    final_source, final_target = source_levels[-1], target_levels[-1]
    pre = o3d.pipelines.registration.evaluate_registration(
        final_source, final_target, max_distances[-1], initial
    )
    post = o3d.pipelines.registration.evaluate_registration(
        final_source, final_target, max_distances[-1], transform
    )
    update_translation, update_rotation = transform_step(initial, transform)
    return transform, {
        "pre_fitness": float(pre.fitness),
        "pre_rmse_m": float(pre.inlier_rmse),
        "post_fitness": float(post.fitness),
        "post_rmse_m": float(post.inlier_rmse),
        "update_translation_m": update_translation,
        "update_rotation_deg": update_rotation,
        "levels": history,
    }


def optimize_trajectory(
    original: np.ndarray,
    edges: list[dict[str, Any]],
    prior_translation_sigma: float,
    prior_rotation_sigma_rad: float,
    smooth_translation_sigma: float,
    smooth_rotation_sigma_rad: float,
    measurement_translation_sigma: float,
    measurement_rotation_sigma_rad: float,
    max_nfev: int,
) -> tuple[np.ndarray, np.ndarray, Any]:
    frame_count = len(original)

    def unpack(vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        corrected = original.copy()
        corrections = np.repeat(np.eye(4, dtype=np.float64)[None], frame_count, axis=0)
        for frame in range(1, frame_count):
            correction = correction_matrix(vector[(frame - 1) * 6 : frame * 6])
            corrections[frame] = correction
            corrected[frame] = correction @ original[frame]
        return corrected, corrections

    def residuals(vector: np.ndarray) -> np.ndarray:
        corrected, corrections = unpack(vector)
        values: list[np.ndarray] = []
        for edge in edges:
            source = int(edge["source"])
            target = int(edge["target"])
            predicted = np.linalg.inv(corrected[target]) @ corrected[source]
            residual = transform_residual(edge["measurement"], predicted)
            confidence = float(edge["confidence"])
            values.append(
                confidence
                * np.concatenate(
                    [
                        residual[:3] / measurement_translation_sigma,
                        residual[3:] / measurement_rotation_sigma_rad,
                    ]
                )
            )

        for frame in range(1, frame_count):
            correction = corrections[frame]
            values.append(correction[:3, 3] / prior_translation_sigma)
            values.append(
                Rotation.from_matrix(correction[:3, :3]).as_rotvec()
                / prior_rotation_sigma_rad
            )

        for frame in range(frame_count - 1):
            delta = np.linalg.inv(corrections[frame]) @ corrections[frame + 1]
            values.append(delta[:3, 3] / smooth_translation_sigma)
            values.append(
                Rotation.from_matrix(delta[:3, :3]).as_rotvec()
                / smooth_rotation_sigma_rad
            )
        return np.concatenate(values)

    initial = np.zeros((frame_count - 1) * 6, dtype=np.float64)
    result = least_squares(
        residuals,
        initial,
        method="trf",
        loss="huber",
        f_scale=1.0,
        x_scale="jac",
        max_nfev=max_nfev,
        verbose=1,
    )
    refined, corrections = unpack(result.x)
    return refined, corrections, result


def summarize(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p90": float(np.percentile(array, 90.0)),
        "max": float(np.max(array)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--poses-path", type=Path, default=None)
    parser.add_argument("--depth-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--frame-count", type=int, default=40)
    parser.add_argument("--exclude-mask-dir", type=Path, action="append", default=None)
    parser.add_argument("--mask-dilation", type=int, default=9)
    parser.add_argument("--depth-min", type=float, default=0.15)
    parser.add_argument("--depth-max", type=float, default=3.0)
    parser.add_argument("--edge-strides", default="1,2,3")
    parser.add_argument("--voxel-sizes", default="0.030,0.018,0.012")
    parser.add_argument("--max-distances", default="0.090,0.050,0.030")
    parser.add_argument("--iterations", default="30,20,15")
    parser.add_argument("--min-edge-fitness", type=float, default=0.45)
    parser.add_argument("--max-edge-rmse", type=float, default=0.025)
    parser.add_argument("--max-update-translation", type=float, default=0.12)
    parser.add_argument("--max-update-rotation-deg", type=float, default=12.0)
    parser.add_argument("--prior-translation-sigma", type=float, default=0.05)
    parser.add_argument("--prior-rotation-sigma-deg", type=float, default=5.0)
    parser.add_argument("--smooth-translation-sigma", type=float, default=0.020)
    parser.add_argument("--smooth-rotation-sigma-deg", type=float, default=2.0)
    parser.add_argument("--measurement-translation-sigma", type=float, default=0.010)
    parser.add_argument("--measurement-rotation-sigma-deg", type=float, default=1.0)
    parser.add_argument("--max-nfev", type=int, default=80)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = args.workspace.resolve()
    poses_path = (args.poses_path or workspace / "outputs/00_rgb_frames/poses.npz").resolve()
    depth_dir = (
        args.depth_dir or workspace / "outputs/06_dense_depth/raw_projected_npy"
    ).resolve()
    output_dir = (args.output_dir or workspace / "outputs/00_pose_refinement").resolve()
    output_path = output_dir / "poses_refined.npz"
    manifest_path = output_dir / "manifest.json"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {output_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    pose_data = np.load(poses_path)
    original = np.asarray(pose_data["T_C0_from_Ct"], dtype=np.float64)
    frame_count = int(args.frame_count)
    if frame_count <= 1 or len(original) < frame_count:
        raise ValueError(f"Invalid frame count {frame_count} for pose array {original.shape}")
    original = original[:frame_count]
    if not np.allclose(original[0], np.eye(4), atol=1e-8):
        raise ValueError("Expected frame-0 camera pose to be identity")

    camera = read_json(workspace / "outputs/00_rgb_frames/camera.json")
    intrinsics = camera["rgb_intrinsics_right"]
    depth_paths = sorted(depth_dir.glob("*.npy"))[:frame_count]
    if len(depth_paths) != frame_count:
        raise ValueError(f"Expected {frame_count} depth maps in {depth_dir}, found {len(depth_paths)}")
    mask_dirs = (
        [path.resolve() for path in args.exclude_mask_dir]
        if args.exclude_mask_dir
        else default_mask_dirs(workspace)
    )
    for mask_dir in mask_dirs:
        if len(list(mask_dir.glob("*.png"))) < frame_count:
            raise ValueError(f"Mask directory does not cover {frame_count} frames: {mask_dir}")

    edge_strides = tuple(sorted(set(parse_number_list(args.edge_strides, int))))
    voxel_sizes = parse_number_list(args.voxel_sizes, float)
    max_distances = parse_number_list(args.max_distances, float)
    iterations = parse_number_list(args.iterations, int)
    if not (len(voxel_sizes) == len(max_distances) == len(iterations)):
        raise ValueError("--voxel-sizes, --max-distances, and --iterations must match")
    if any(stride <= 0 for stride in edge_strides):
        raise ValueError("Edge strides must be positive")

    start = time.perf_counter()
    pyramids: list[list[o3d.geometry.PointCloud]] = []
    frame_records: list[dict[str, Any]] = []
    print("Loading masked static native-depth clouds...", flush=True)
    for frame, depth_path in enumerate(depth_paths):
        mask_paths = [mask_dir / f"{frame:06d}.png" for mask_dir in mask_dirs]
        cloud, record = load_static_cloud(
            depth_path,
            mask_paths,
            intrinsics,
            float(args.depth_min),
            float(args.depth_max),
            int(args.mask_dilation),
        )
        levels = cloud_pyramid(cloud, voxel_sizes)
        pyramids.append(levels)
        record.update(
            {
                "frame_index": frame,
                "pyramid_point_counts": [len(level.points) for level in levels],
            }
        )
        frame_records.append(record)

    print(f"Registering static-background edges at strides {edge_strides}...", flush=True)
    accepted_edges: list[dict[str, Any]] = []
    edge_records: list[dict[str, Any]] = []
    for stride in edge_strides:
        for source in range(frame_count - stride):
            target = source + stride
            initial = np.linalg.inv(original[target]) @ original[source]
            measurement, diagnostics = register_pair(
                pyramids[source],
                pyramids[target],
                initial,
                max_distances,
                iterations,
            )
            accepted = (
                diagnostics["post_fitness"] >= float(args.min_edge_fitness)
                and diagnostics["post_rmse_m"] <= float(args.max_edge_rmse)
                and diagnostics["update_translation_m"]
                <= float(args.max_update_translation)
                and diagnostics["update_rotation_deg"]
                <= float(args.max_update_rotation_deg)
            )
            confidence = float(
                np.clip(
                    np.sqrt(diagnostics["post_fitness"])
                    * np.sqrt(0.010 / max(diagnostics["post_rmse_m"], 0.004)),
                    0.5,
                    1.5,
                )
            )
            record = {
                "source": source,
                "target": target,
                "stride": stride,
                "accepted": bool(accepted),
                "confidence": confidence,
                "initial_T_Ctarget_from_Csource": initial.tolist(),
                "measured_T_Ctarget_from_Csource": measurement.tolist(),
                **diagnostics,
            }
            edge_records.append(record)
            if accepted:
                accepted_edges.append(
                    {
                        "source": source,
                        "target": target,
                        "stride": stride,
                        "confidence": confidence,
                        "measurement": measurement,
                    }
                )
            print(
                f"edge {source:02d}->{target:02d} s={stride} "
                f"fit {diagnostics['pre_fitness']:.3f}->{diagnostics['post_fitness']:.3f} "
                f"rmse {diagnostics['post_rmse_m'] * 100:.2f}cm "
                f"step {diagnostics['update_translation_m'] * 100:.2f}cm/"
                f"{diagnostics['update_rotation_deg']:.2f}deg "
                f"{'accepted' if accepted else 'rejected'}",
                flush=True,
            )

    adjacent_accepted = sum(
        edge["stride"] == 1 for edge in accepted_edges
    )
    if adjacent_accepted < frame_count - 2:
        raise RuntimeError(
            f"Too few accepted adjacent edges: {adjacent_accepted}/{frame_count - 1}"
        )

    print(f"Optimizing {frame_count} poses from {len(accepted_edges)} accepted edges...", flush=True)
    refined, corrections, optimization = optimize_trajectory(
        original,
        accepted_edges,
        float(args.prior_translation_sigma),
        np.radians(float(args.prior_rotation_sigma_deg)),
        float(args.smooth_translation_sigma),
        np.radians(float(args.smooth_rotation_sigma_deg)),
        float(args.measurement_translation_sigma),
        np.radians(float(args.measurement_rotation_sigma_deg)),
        int(args.max_nfev),
    )
    refined[0] = np.eye(4)

    validation_records: list[dict[str, Any]] = []
    for edge in accepted_edges:
        source, target = int(edge["source"]), int(edge["target"])
        measured = edge["measurement"]
        original_relative = np.linalg.inv(original[target]) @ original[source]
        refined_relative = np.linalg.inv(refined[target]) @ refined[source]
        original_error = transform_residual(measured, original_relative)
        refined_error = transform_residual(measured, refined_relative)
        record = {
            "source": source,
            "target": target,
            "stride": int(edge["stride"]),
            "original_translation_error_m": float(np.linalg.norm(original_error[:3])),
            "original_rotation_error_deg": float(np.degrees(np.linalg.norm(original_error[3:]))),
            "refined_translation_error_m": float(np.linalg.norm(refined_error[:3])),
            "refined_rotation_error_deg": float(np.degrees(np.linalg.norm(refined_error[3:]))),
        }
        if edge["stride"] == 1:
            source_cloud = pyramids[source][-1]
            target_cloud = pyramids[target][-1]
            original_eval = o3d.pipelines.registration.evaluate_registration(
                source_cloud, target_cloud, max_distances[-1], original_relative
            )
            refined_eval = o3d.pipelines.registration.evaluate_registration(
                source_cloud, target_cloud, max_distances[-1], refined_relative
            )
            record.update(
                {
                    "original_fitness": float(original_eval.fitness),
                    "original_rmse_m": float(original_eval.inlier_rmse),
                    "refined_fitness": float(refined_eval.fitness),
                    "refined_rmse_m": float(refined_eval.inlier_rmse),
                }
            )
        validation_records.append(record)

    correction_translations = [
        float(np.linalg.norm(correction[:3, 3])) for correction in corrections
    ]
    correction_rotations = [rotation_degrees(correction) for correction in corrections]
    adjacent_validation = [record for record in validation_records if record["stride"] == 1]
    summary = {
        "accepted_edge_count": len(accepted_edges),
        "rejected_edge_count": len(edge_records) - len(accepted_edges),
        "adjacent_accepted_edge_count": adjacent_accepted,
        "pose_correction_translation_m": summarize(correction_translations),
        "pose_correction_rotation_deg": summarize(correction_rotations),
        "adjacent_original_translation_error_m": summarize(
            [record["original_translation_error_m"] for record in adjacent_validation]
        ),
        "adjacent_refined_translation_error_m": summarize(
            [record["refined_translation_error_m"] for record in adjacent_validation]
        ),
        "adjacent_original_rotation_error_deg": summarize(
            [record["original_rotation_error_deg"] for record in adjacent_validation]
        ),
        "adjacent_refined_rotation_error_deg": summarize(
            [record["refined_rotation_error_deg"] for record in adjacent_validation]
        ),
        "adjacent_original_fitness": summarize(
            [record["original_fitness"] for record in adjacent_validation]
        ),
        "adjacent_refined_fitness": summarize(
            [record["refined_fitness"] for record in adjacent_validation]
        ),
        "adjacent_original_rmse_m": summarize(
            [record["original_rmse_m"] for record in adjacent_validation]
        ),
        "adjacent_refined_rmse_m": summarize(
            [record["refined_rmse_m"] for record in adjacent_validation]
        ),
    }

    arrays: dict[str, np.ndarray] = {
        "T_C0_from_Ct": refined,
        "T_C0_from_Ct_refined": refined,
        "T_C0_from_Ct_original": original,
        "T_C0_refined_from_C0_original": corrections,
    }
    for key in pose_data.files:
        if key not in arrays:
            value = np.asarray(pose_data[key])
            arrays[key] = value[:frame_count] if value.ndim and len(value) >= frame_count else value
    if "T_W_from_C" in pose_data.files:
        arrays["T_W_from_C_refined"] = np.einsum(
            "ij,tjk->tik", np.asarray(pose_data["T_W_from_C"])[0], refined
        )
    np.savez_compressed(output_path, **arrays)

    manifest = {
        "stage": "00_static_rgbd_camera_pose_refinement",
        "status": "completed",
        "workspace": str(workspace),
        "input_poses": str(poses_path),
        "input_depth": str(depth_dir),
        "output_poses": str(output_path),
        "pose_key": "T_C0_from_Ct",
        "frame_count": frame_count,
        "coordinate_frame": "frame0_right_camera_opencv_rdf",
        "pose_semantics": "p_C0 = T_C0_from_Ct @ p_Ct",
        "excluded_mask_directories": [str(path) for path in mask_dirs],
        "parameters": {
            key: value
            for key, value in vars(args).items()
            if isinstance(value, (str, int, float, bool, type(None)))
        },
        "voxel_sizes_m": list(voxel_sizes),
        "max_correspondence_distances_m": list(max_distances),
        "edge_strides": list(edge_strides),
        "optimization": {
            "success": bool(optimization.success),
            "status": int(optimization.status),
            "message": str(optimization.message),
            "cost": float(optimization.cost),
            "optimality": float(optimization.optimality),
            "nfev": int(optimization.nfev),
        },
        "summary": summary,
        "frames": frame_records,
        "edges": edge_records,
        "validation": validation_records,
        "elapsed_seconds": time.perf_counter() - start,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Refined poses: {output_path}", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
