#!/usr/bin/env python3
"""Refine per-frame FoundationPose SE(3) deltas against masked RGB-D clouds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
from PIL import Image
from scipy.ndimage import binary_dilation
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import trimesh


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORLD_FRAME = "frame0_right_camera_opencv_rdf"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    def convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        return value

    path.write_text(json.dumps(convert(payload), ensure_ascii=False, indent=2) + "\n")


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [item for item in loaded.geometry.values() if isinstance(item, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"Mesh scene is empty: {path}")
        loaded = meshes[0].copy() if len(meshes) == 1 else trimesh.util.concatenate(meshes)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type {type(loaded)!r}: {path}")
    return loaded


def parse_numbers(value: str, cast: type) -> tuple[Any, ...]:
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def pose_step(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    relative = candidate @ np.linalg.inv(reference)
    translation = float(np.linalg.norm(relative[:3, 3]))
    rotation = float(np.rad2deg(Rotation.from_matrix(relative[:3, :3]).magnitude()))
    return translation, rotation


def backproject_masked_c0(
    depth: np.ndarray,
    object_mask: np.ndarray,
    hand_mask: np.ndarray,
    intrinsics: dict[str, float],
    transform_c0_from_ct: np.ndarray,
    hand_dilation: int,
) -> np.ndarray:
    if hand_dilation > 0:
        hand_mask = binary_dilation(hand_mask, iterations=hand_dilation)
    valid = (
        object_mask
        & ~hand_mask
        & np.isfinite(depth)
        & (depth >= 0.15)
        & (depth <= 3.0)
    )
    v, u = np.nonzero(valid)
    z = depth[v, u].astype(np.float64)
    x = (u - float(intrinsics["cx"])) * z / float(intrinsics["fx"])
    y = (v - float(intrinsics["cy"])) * z / float(intrinsics["fy"])
    points_ct = np.column_stack([x, y, z])
    return transform_points(points_ct, transform_c0_from_ct)


def point_cloud(points: np.ndarray, voxel_size: float, normals: bool) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    if voxel_size > 0.0:
        cloud = cloud.voxel_down_sample(voxel_size)
    if normals and len(cloud.points):
        cloud.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=max(3.0 * voxel_size, 0.015), max_nn=50
            )
        )
    return cloud


def refine_pose(
    source_points: np.ndarray,
    target_points: np.ndarray,
    initial: np.ndarray,
    voxel_sizes: tuple[float, ...],
    max_distances: tuple[float, ...],
    iterations: tuple[int, ...],
) -> tuple[np.ndarray, dict[str, Any]]:
    transform = initial.copy()
    levels = []
    for level, (voxel, distance, iteration_count) in enumerate(
        zip(voxel_sizes, max_distances, iterations)
    ):
        source = point_cloud(source_points, voxel, normals=False)
        target = point_cloud(target_points, voxel, normals=level > 0)
        if len(source.points) < 20 or len(target.points) < 20:
            raise ValueError(
                f"Too few ICP points at level {level}: {len(source.points)}/{len(target.points)}"
            )
        if level == 0:
            estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        else:
            loss = o3d.pipelines.registration.TukeyLoss(k=distance)
            estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
        result = o3d.pipelines.registration.registration_icp(
            source,
            target,
            distance,
            transform,
            estimation,
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-7,
                relative_rmse=1e-7,
                max_iteration=iteration_count,
            ),
        )
        transform = np.asarray(result.transformation, dtype=np.float64)
        levels.append(
            {
                "level": level,
                "voxel_size_m": voxel,
                "max_correspondence_distance_m": distance,
                "source_points": len(source.points),
                "target_points": len(target.points),
                "fitness": float(result.fitness),
                "inlier_rmse_m": float(result.inlier_rmse),
            }
        )
    return transform, {"levels": levels, **levels[-1]}


def refine_pose_bounded(
    source_points: np.ndarray,
    target_points: np.ndarray,
    initial: np.ndarray,
    local_mesh_centroid: np.ndarray,
    max_distances: tuple[float, ...],
    iterations: tuple[int, ...],
    max_center_translation_m: float,
    max_rotation_deg: float,
    translation_only: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    target_cloud = point_cloud(target_points, 0.004, normals=True)
    targets = np.asarray(target_cloud.points, dtype=np.float64)
    normals = np.asarray(target_cloud.normals, dtype=np.float64)
    if len(targets) < 20:
        raise ValueError(f"Too few target points: {len(targets)}")
    tree = cKDTree(targets)
    source_initial = transform_points(source_points, initial)
    center = transform_points(local_mesh_centroid[None], initial)[0]
    parameters = np.zeros(3 if translation_only else 6, dtype=np.float64)
    levels = []
    rotation_bound = np.deg2rad(max_rotation_deg)

    def apply_correction(points: np.ndarray, vector: np.ndarray) -> np.ndarray:
        rotation = (
            np.eye(3, dtype=np.float64)
            if translation_only
            else Rotation.from_rotvec(vector[3:]).as_matrix()
        )
        return (points - center) @ rotation.T + center + vector[:3]

    for level, (distance, iteration_count) in enumerate(zip(max_distances, iterations)):
        predicted = apply_correction(source_initial, parameters)
        nearest_distance, nearest_index = tree.query(predicted, workers=-1)
        correspondence = np.isfinite(nearest_distance) & (nearest_distance <= distance)
        indices = np.flatnonzero(correspondence)
        if len(indices) < 30:
            raise ValueError(f"Only {len(indices)} bounded ICP correspondences at level {level}")
        if len(indices) > 12000:
            indices = indices[np.linspace(0, len(indices) - 1, 12000, dtype=np.int64)]
        matched_targets = targets[nearest_index[indices]]
        matched_normals = normals[nearest_index[indices]]

        def residual(vector: np.ndarray) -> np.ndarray:
            moved = apply_correction(source_initial[indices], vector)
            difference = moved - matched_targets
            point_plane = np.einsum("ij,ij->i", difference, matched_normals) / 0.008
            point_point = (0.15 * difference / 0.020).reshape(-1)
            prior_weight = 0.15 * np.sqrt(len(indices))
            translation_prior = prior_weight * vector[:3] / 0.020
            rotation_prior = (
                np.empty(0, dtype=np.float64)
                if translation_only
                else prior_weight * vector[3:] / np.deg2rad(5.0)
            )
            return np.concatenate(
                [point_plane, point_point, translation_prior, rotation_prior]
            )

        translation_component_bound = max_center_translation_m / np.sqrt(3.0)
        rotation_component_bound = rotation_bound / np.sqrt(3.0)
        lower_values = [-translation_component_bound] * 3
        if not translation_only:
            lower_values += [-rotation_component_bound] * 3
        lower = np.asarray(lower_values, dtype=np.float64)
        upper = -lower
        fit = least_squares(
            residual,
            np.clip(parameters, lower + 1e-8, upper - 1e-8),
            bounds=(lower, upper),
            loss="huber",
            f_scale=1.0,
            max_nfev=iteration_count,
        )
        parameters = fit.x
        moved = apply_correction(source_initial[indices], parameters)
        difference = moved - matched_targets
        levels.append(
            {
                "level": level,
                "max_correspondence_distance_m": distance,
                "correspondence_count": len(indices),
                "fitness": float(len(indices) / len(source_points)),
                "inlier_rmse_m": float(np.sqrt(np.mean(np.sum(difference**2, axis=1)))),
                "cost": float(fit.cost),
                "nfev": int(fit.nfev),
            }
        )

    correction_rotation = (
        np.eye(3, dtype=np.float64)
        if translation_only
        else Rotation.from_rotvec(parameters[3:]).as_matrix()
    )
    correction = np.eye(4, dtype=np.float64)
    correction[:3, :3] = correction_rotation
    correction[:3, 3] = center + parameters[:3] - correction_rotation @ center
    candidate = correction @ initial
    return candidate, {
        "method": "bounded_center_parameterized_point_to_plane_icp",
        "center_translation_m": float(np.linalg.norm(parameters[:3])),
        "rotation_deg": (
            0.0 if translation_only else float(np.rad2deg(np.linalg.norm(parameters[3:])))
        ),
        "translation_only": translation_only,
        "parameter_vector": parameters,
        "levels": levels,
        **levels[-1],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--object-id", default="bottle")
    parser.add_argument("--foundationpose-dir", type=Path, required=True)
    parser.add_argument("--aligned-mesh", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--hand-mask-dir", type=Path, required=True)
    parser.add_argument("--depth-dir", type=Path, required=True)
    parser.add_argument("--poses-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mesh-samples", type=int, default=40000)
    parser.add_argument("--hand-mask-dilation", type=int, default=2)
    parser.add_argument("--voxel-sizes", default="0.012,0.007,0.004")
    parser.add_argument("--max-distances", default="0.060,0.035,0.020")
    parser.add_argument("--iterations", default="40,30,20")
    parser.add_argument("--min-fitness", type=float, default=0.08)
    parser.add_argument("--max-rmse-m", type=float, default=0.025)
    parser.add_argument("--max-correction-translation-m", type=float, default=0.04)
    parser.add_argument("--max-correction-rotation-deg", type=float, default=10.0)
    parser.add_argument(
        "--translation-only",
        action="store_true",
        help="Remove FoundationPose rotation and optimize only mesh-center translation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = args.workspace.resolve()
    fp_dir = args.foundationpose_dir.resolve()
    mesh_path = args.aligned_mesh.resolve()
    mask_dir = args.mask_dir.resolve()
    hand_mask_dir = args.hand_mask_dir.resolve()
    depth_dir = args.depth_dir.resolve()
    poses_path = args.poses_path.resolve()
    output_dir = args.output_dir.resolve()

    fp_delta = np.load(fp_dir / "Delta_C0_object_motion.npy").astype(np.float64)
    fp_absolute = np.load(fp_dir / "T_C0_from_aligned_mesh.npy").astype(np.float64)
    frame_indices = np.load(fp_dir / "frame_indices.npy").astype(np.int32)
    with np.load(poses_path) as pose_data:
        camera_poses = pose_data["T_C0_from_Ct"].astype(np.float64)
    camera = read_json(workspace / "outputs/00_rgb_frames/camera.json")
    intrinsics = camera["rgb_intrinsics_right"]
    mask_paths = sorted(mask_dir.glob("*.png"))
    hand_mask_paths = sorted(hand_mask_dir.glob("*.png"))
    depth_paths = sorted(depth_dir.glob("*.npy"))
    frame_count = len(frame_indices)
    if fp_delta.shape != (frame_count, 4, 4) or fp_absolute.shape != (frame_count, 4, 4):
        raise ValueError(f"FoundationPose array mismatch: {fp_delta.shape}/{fp_absolute.shape}")
    if any(len(paths) <= int(frame_indices[-1]) for paths in (mask_paths, hand_mask_paths, depth_paths)):
        raise ValueError("RGB-D mask sequences do not cover all FoundationPose frames")
    if len(camera_poses) <= int(frame_indices[-1]):
        raise ValueError("Camera pose sequence does not cover all FoundationPose frames")

    voxel_sizes = parse_numbers(args.voxel_sizes, float)
    max_distances = parse_numbers(args.max_distances, float)
    iterations = parse_numbers(args.iterations, int)
    if not (len(voxel_sizes) == len(max_distances) == len(iterations)):
        raise ValueError("ICP pyramid argument lengths must match")
    preflight = {
        "stage": "08_foundationpose_masked_rgbd_icp_refinement",
        "object_id": args.object_id,
        "frame_count": frame_count,
        "coordinate_frame": WORLD_FRAME,
        "foundationpose_dir": fp_dir,
        "aligned_mesh": mesh_path,
        "mask_dir": mask_dir,
        "hand_mask_dir": hand_mask_dir,
        "depth_dir": depth_dir,
        "poses_path": poses_path,
        "output_dir": output_dir,
        "icp": {
            "voxel_sizes_m": voxel_sizes,
            "max_correspondence_distances_m": max_distances,
            "iterations": iterations,
            "max_correction_translation_m": args.max_correction_translation_m,
            "max_correction_rotation_deg": args.max_correction_rotation_deg,
            "translation_only": args.translation_only,
        },
    }
    if args.check:
        print(json.dumps(preflight, ensure_ascii=False, indent=2, default=str))
        return

    mesh = load_mesh(mesh_path)
    rng = np.random.default_rng(args.seed)
    source_points, _ = trimesh.sample.sample_surface(mesh, args.mesh_samples, seed=rng)
    refined = fp_delta.copy()
    if args.translation_only:
        mesh_centroid = np.asarray(mesh.centroid, dtype=np.float64)
        for index, delta in enumerate(fp_delta):
            center = transform_points(mesh_centroid[None], delta)[0]
            refined[index] = np.eye(4, dtype=np.float64)
            refined[index, :3, 3] = center - mesh_centroid
    accepted = np.zeros(frame_count, dtype=bool)
    accepted[0] = True
    records = []
    for output_index, frame_value in enumerate(frame_indices):
        frame = int(frame_value)
        depth = np.load(depth_paths[frame]).astype(np.float32)
        object_mask = np.asarray(Image.open(mask_paths[frame]).convert("L")) > 127
        hand_mask = np.asarray(Image.open(hand_mask_paths[frame]).convert("L")) > 127
        target_points = backproject_masked_c0(
            depth,
            object_mask,
            hand_mask,
            intrinsics,
            camera_poses[frame],
            args.hand_mask_dilation,
        )
        if output_index == 0:
            record = {
                "frame": frame,
                "status": "reference_identity",
                "accepted": True,
                "target_points": len(target_points),
                "correction_translation_m": 0.0,
                "correction_rotation_deg": 0.0,
            }
        else:
            try:
                initial_delta = refined[output_index] if args.translation_only else fp_delta[output_index]
                candidate, diagnostics = refine_pose_bounded(
                    source_points,
                    target_points,
                    initial_delta,
                    np.asarray(mesh.centroid, dtype=np.float64),
                    max_distances,
                    iterations,
                    args.max_correction_translation_m,
                    args.max_correction_rotation_deg,
                    args.translation_only,
                )
                correction_translation = diagnostics["center_translation_m"]
                correction_rotation = diagnostics["rotation_deg"]
                accepted_frame = (
                    diagnostics["fitness"] >= args.min_fitness
                    and diagnostics["inlier_rmse_m"] <= args.max_rmse_m
                    and correction_translation <= args.max_correction_translation_m
                    and correction_rotation <= args.max_correction_rotation_deg
                )
                if accepted_frame:
                    refined[output_index] = candidate
                    accepted[output_index] = True
                record = {
                    "frame": frame,
                    "status": "accepted" if accepted_frame else "rejected_kept_foundationpose",
                    "accepted": accepted_frame,
                    "target_points": len(target_points),
                    "correction_translation_m": correction_translation,
                    "correction_rotation_deg": correction_rotation,
                    "icp": diagnostics,
                }
            except Exception as exc:
                record = {
                    "frame": frame,
                    "status": "failed_kept_foundationpose",
                    "accepted": False,
                    "target_points": len(target_points),
                    "error": f"{type(exc).__name__}: {exc}",
                }
        records.append(record)
        print(
            f"frame {frame:02d}: {record['status']} target={len(target_points)}",
            flush=True,
        )

    refined[0] = np.eye(4, dtype=np.float64)
    refined_absolute = np.einsum("tij,jk->tik", refined, fp_absolute[0])
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "Delta_C0_object_motion.npy", refined)
    np.save(output_dir / "T_C0_from_aligned_mesh.npy", refined_absolute)
    np.save(output_dir / "frame_indices.npy", frame_indices)
    np.save(output_dir / "success.npy", np.ones(frame_count, dtype=bool))
    np.save(output_dir / "icp_accepted.npy", accepted)
    manifest = {
        **preflight,
        "status": "completed",
        "pose_policy": (
            "FoundationPose mesh-center initialization plus independent translation-only masked RGB-D ICP"
            if args.translation_only
            else "FoundationPose SE(3) initialization plus independent masked RGB-D ICP"
        ),
        "frame0_policy": "identity delta retained from Stage07 aligned mesh",
        "fallback_policy": "keep FoundationPose delta when ICP fails quality or correction gates",
        "accepted_icp_frames": int(accepted.sum()),
        "fallback_frames": int((~accepted).sum()),
        "frames": records,
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps({"output": str(output_dir), "accepted": int(accepted.sum())}, indent=2))


if __name__ == "__main__":
    main()
