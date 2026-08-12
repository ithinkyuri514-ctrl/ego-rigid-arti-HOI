"""Refine a laptop screen and hinge from first-frame RGB-D observations."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from vlm_sam2_recon.stages.camera_alignment import (
    acute_angle_deg,
    mask_bbox_quantiles,
    plane_from_points,
    project_right_camera_points,
    sample_mesh_points,
    save_mesh_projection_overlay,
    write_json,
)
from vlm_sam2_recon.stages.screen_hinge_tracking import load_mesh


@dataclass
class ScreenMetricCalibrationConfig:
    alignment_dir: Path
    export_root: Path
    output_dir: Path
    base_label: str = "14"
    screen_label: str = "15"
    bbox_quantile_min: float = 0.01
    bbox_quantile_max: float = 0.99
    scale_min: float = 0.85
    scale_max: float = 1.15
    shift_max_m: float = 0.05
    sample_count: int = 50000
    random_seed: int = 20260713


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float64) @ matrix[:3, :3].T + matrix[:3, 3]


def axis_rotation_matrix(origin: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    rotation = Rotation.from_rotvec(np.asarray(axis, dtype=np.float64) * float(angle_rad)).as_matrix()
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = origin - rotation @ origin
    return matrix


def local_scale_shift_matrix(
    pivot: np.ndarray,
    axis: np.ndarray,
    radial: np.ndarray,
    axis_scale: float,
    radial_scale: float,
    axis_shift_m: float,
    radial_shift_m: float,
) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    radial = np.asarray(radial, dtype=np.float64)
    linear = (
        np.eye(3, dtype=np.float64)
        + (float(axis_scale) - 1.0) * np.outer(axis, axis)
        + (float(radial_scale) - 1.0) * np.outer(radial, radial)
    )
    translation = (
        np.asarray(pivot, dtype=np.float64)
        - linear @ np.asarray(pivot, dtype=np.float64)
        + float(axis_shift_m) * axis
        + float(radial_shift_m) * radial
    )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = linear
    matrix[:3, 3] = translation
    return matrix


def observed_plane_intersection(
    base_points: np.ndarray,
    screen_points: np.ndarray,
    reference_origin: np.ndarray,
    reference_axis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    base_center, base_normal, _ = plane_from_points(base_points)
    screen_center, screen_normal, _ = plane_from_points(screen_points)
    axis = np.cross(base_normal, screen_normal)
    axis /= np.linalg.norm(axis) + 1e-12
    reference_axis = reference_axis / (np.linalg.norm(reference_axis) + 1e-12)
    if float(axis @ reference_axis) < 0.0:
        axis = -axis
    constraints = np.vstack([base_normal, screen_normal, axis])
    values = np.asarray(
        [
            float(base_normal @ base_center),
            float(screen_normal @ screen_center),
            float(axis @ reference_origin),
        ],
        dtype=np.float64,
    )
    origin = np.linalg.solve(constraints, values)
    delta = origin - reference_origin
    perpendicular = delta - reference_axis * float(delta @ reference_axis)
    diagnostics = {
        "base_plane_center": base_center,
        "base_plane_normal": base_normal,
        "screen_plane_center": screen_center,
        "screen_plane_normal": screen_normal,
        "axis_direction_delta_deg": acute_angle_deg(axis, reference_axis),
        "axis_origin_shift_m": delta,
        "axis_origin_perpendicular_shift_m": float(np.linalg.norm(perpendicular)),
    }
    return origin, axis, diagnostics


def run_screen_metric_calibration(config: ScreenMetricCalibrationConfig) -> dict[str, Any]:
    config.alignment_dir = Path(config.alignment_dir).expanduser().resolve()
    config.export_root = Path(config.export_root).expanduser().resolve()
    config.output_dir = Path(config.output_dir).expanduser().resolve()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    alignment = read_json(config.alignment_dir / "alignment_result.json")
    meta = read_json(config.export_root / "manifest.json")
    base_mesh = load_mesh(config.alignment_dir / f"part_{config.base_label}_camera.obj")
    screen_mesh = load_mesh(config.alignment_dir / f"part_{config.screen_label}_camera.obj")
    base_points = np.asarray(
        trimesh.load(config.alignment_dir / "observed_base_pointcloud.ply", process=False).vertices,
        dtype=np.float64,
    )
    screen_points = np.asarray(
        trimesh.load(config.alignment_dir / "observed_screen_pointcloud.ply", process=False).vertices,
        dtype=np.float64,
    )
    screen_mask_path = (
        config.alignment_dir
        / "part_masks"
        / "target_laptop_frame_0_screen_projection.mask.npy"
    )
    screen_mask = np.load(screen_mask_path).astype(bool)
    joint_data = read_json(config.alignment_dir / "joint_camera.json")
    if not joint_data.get("joints"):
        raise ValueError(f"No articulated joint in {config.alignment_dir / 'joint_camera.json'}")
    joint = dict(joint_data["joints"][0])
    old_origin = np.asarray(joint["origin_xyz"], dtype=np.float64)
    old_axis = np.asarray(joint["axis_xyz"], dtype=np.float64)
    old_axis /= np.linalg.norm(old_axis) + 1e-12

    new_origin, new_axis, plane_diagnostics = observed_plane_intersection(
        base_points,
        screen_points,
        old_origin,
        old_axis,
    )

    samples = sample_mesh_points(
        screen_mesh,
        min(int(config.sample_count), max(5000, len(screen_mesh.faces) * 2)),
        int(config.random_seed),
    )
    _, mesh_normal, _ = plane_from_points(samples)
    _, observed_normal, _ = plane_from_points(screen_points)
    mesh_radial_normal = mesh_normal - new_axis * float(mesh_normal @ new_axis)
    observed_radial_normal = observed_normal - new_axis * float(observed_normal @ new_axis)
    mesh_radial_normal /= np.linalg.norm(mesh_radial_normal) + 1e-12
    observed_radial_normal /= np.linalg.norm(observed_radial_normal) + 1e-12
    normal_angle = float(
        np.arctan2(
            new_axis @ np.cross(mesh_radial_normal, observed_radial_normal),
            mesh_radial_normal @ observed_radial_normal,
        )
    )
    rigid_rotation = axis_rotation_matrix(new_origin, new_axis, normal_angle)
    rotated_samples = transform_points(samples, rigid_rotation)

    radial = np.cross(observed_normal, new_axis)
    radial /= np.linalg.norm(radial) + 1e-12
    basis = np.column_stack([new_axis, radial, observed_normal])
    center_delta_local = np.median(screen_points @ basis, axis=0) - np.median(rotated_samples @ basis, axis=0)
    center_translation = basis @ center_delta_local
    rigid_translation = np.eye(4, dtype=np.float64)
    rigid_translation[:3, 3] = center_translation
    rigid_matrix = rigid_translation @ rigid_rotation
    rigid_samples = transform_points(samples, rigid_matrix)

    target_bbox = mask_bbox_quantiles(
        screen_mask,
        float(config.bbox_quantile_min),
        float(config.bbox_quantile_max),
    )
    target_size = np.maximum(target_bbox[[2, 3]] - target_bbox[[0, 1]], 1.0)
    pivot = np.median(rigid_samples, axis=0)

    def bbox_residual(parameters: np.ndarray) -> np.ndarray:
        matrix = local_scale_shift_matrix(
            pivot,
            new_axis,
            radial,
            parameters[0],
            parameters[1],
            parameters[2],
            parameters[3],
        )
        candidate = transform_points(rigid_samples, matrix)
        u, v, z = project_right_camera_points(meta, candidate)
        valid = z > 1e-6
        if int(valid.sum()) < 64:
            return np.full(4, 100.0, dtype=np.float64)
        projected_bbox = np.asarray(
            [
                np.quantile(u[valid], config.bbox_quantile_min),
                np.quantile(v[valid], config.bbox_quantile_min),
                np.quantile(u[valid], config.bbox_quantile_max),
                np.quantile(v[valid], config.bbox_quantile_max),
            ],
            dtype=np.float64,
        )
        return (projected_bbox - target_bbox) / np.asarray(
            [target_size[0], target_size[1], target_size[0], target_size[1]],
            dtype=np.float64,
        )

    solution = least_squares(
        bbox_residual,
        np.asarray([1.0, 1.0, 0.0, 0.0], dtype=np.float64),
        bounds=(
            np.asarray([config.scale_min, config.scale_min, -config.shift_max_m, -config.shift_max_m]),
            np.asarray([config.scale_max, config.scale_max, config.shift_max_m, config.shift_max_m]),
        ),
        max_nfev=120,
    )
    local_matrix = local_scale_shift_matrix(
        pivot,
        new_axis,
        radial,
        solution.x[0],
        solution.x[1],
        solution.x[2],
        solution.x[3],
    )
    total_matrix = local_matrix @ rigid_matrix
    refined_screen = screen_mesh.copy()
    refined_screen.apply_transform(total_matrix)

    refined_samples = transform_points(samples, total_matrix)
    u, v, z = project_right_camera_points(meta, refined_samples)
    valid = z > 1e-6
    final_bbox = np.asarray(
        [
            np.quantile(u[valid], config.bbox_quantile_min),
            np.quantile(v[valid], config.bbox_quantile_min),
            np.quantile(u[valid], config.bbox_quantile_max),
            np.quantile(v[valid], config.bbox_quantile_max),
        ]
    )

    base_mesh.export(config.output_dir / f"part_{config.base_label}_camera.obj")
    refined_screen.export(config.output_dir / f"part_{config.screen_label}_camera.obj")
    joint["origin_xyz"] = new_origin.tolist()
    joint["axis_xyz"] = new_axis.tolist()
    write_json(config.output_dir / "joint_camera.json", {"joints": [joint]})

    for relative in (
        "observed_base_pointcloud.ply",
        "observed_screen_pointcloud.ply",
    ):
        source = config.alignment_dir / relative
        if source.exists():
            shutil.copy2(source, config.output_dir / relative)
    source_masks = config.alignment_dir / "part_masks"
    if source_masks.exists():
        shutil.copytree(source_masks, config.output_dir / "part_masks", dirs_exist_ok=True)

    refinement = {
        "method": "rgbd_plane_hinge_and_screen_bbox_refine",
        "base_unchanged": True,
        "old_joint_origin_xyz": old_origin,
        "old_joint_axis_xyz": old_axis,
        "new_joint_origin_xyz": new_origin,
        "new_joint_axis_xyz": new_axis,
        "plane_diagnostics": plane_diagnostics,
        "normal_rotation_deg": float(np.rad2deg(normal_angle)),
        "center_translation_m": center_translation,
        "axis_scale": float(solution.x[0]),
        "radial_scale": float(solution.x[1]),
        "axis_shift_m": float(solution.x[2]),
        "radial_shift_m": float(solution.x[3]),
        "screen_affine_matrix": total_matrix,
        "target_bbox_px": target_bbox,
        "final_bbox_px": final_bbox,
        "bbox_residual_normalized": bbox_residual(solution.x),
    }
    alignment.setdefault("alignment", {})["screen_metric_refine"] = refinement
    alignment.setdefault("outputs", {})["result_dir"] = str(config.output_dir)
    alignment["outputs"]["joint_camera"] = str(config.output_dir / "joint_camera.json")
    write_json(config.output_dir / "alignment_result.json", alignment)

    rgb_path = config.export_root / "rgb_right_png" / "000000.png"
    save_mesh_projection_overlay(
        rgb_path,
        config.output_dir / "aligned_mesh_projection_overlay.png",
        meta,
        [
            {"label": config.base_label, "mesh": base_mesh},
            {"label": config.screen_label, "mesh": refined_screen},
        ],
        seed=int(config.random_seed),
    )
    write_json(config.output_dir / "screen_metric_refine.json", refinement)
    return refinement
