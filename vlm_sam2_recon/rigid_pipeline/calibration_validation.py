"""Validate SpatialMP4 camera-extrinsic direction with RGB/depth edge agreement."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .geometry import PoseSample, backproject_depth, interpolate_pose, pose_matrix, transform_points


def normalized_sensor_extrinsics(raw: np.ndarray, interpretation: str) -> np.ndarray:
    """Return ``T_H_from_sensor`` for either possible raw-matrix convention."""
    matrix = np.asarray(raw, dtype=np.float64)
    if interpretation == "sensor_to_head":
        return matrix
    if interpretation == "head_to_sensor":
        return np.linalg.inv(matrix)
    raise ValueError(f"Unknown extrinsics interpretation: {interpretation}")


def _depth_edges(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    edges = np.zeros_like(valid, dtype=bool)
    for axis in (0, 1):
        left = np.take(depth, range(depth.shape[axis] - 1), axis=axis)
        right = np.take(depth, range(1, depth.shape[axis]), axis=axis)
        left_valid = np.take(valid, range(valid.shape[axis] - 1), axis=axis)
        right_valid = np.take(valid, range(1, valid.shape[axis]), axis=axis)
        threshold = np.maximum(0.025, 0.025 * np.minimum(left, right))
        jump = left_valid & right_valid & (np.abs(left - right) > threshold)
        target = [slice(None), slice(None)]
        target[axis] = slice(0, -1)
        edges[tuple(target)] |= jump
        target[axis] = slice(1, None)
        edges[tuple(target)] |= jump
    return edges


def _rgb_gradient(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)
    scale = float(np.quantile(gradient, 0.98))
    return np.clip(gradient / max(scale, 1e-6), 0.0, 1.0)


def evaluate_extrinsics_interpretation(
    *,
    interpretation: str,
    raw_t_h_from_c: np.ndarray,
    raw_t_h_from_d: np.ndarray,
    rgb_intrinsics: dict[str, float],
    depth_intrinsics: dict[str, float],
    rgb_rows: list[dict],
    rgb_dir: Path,
    depth_rows: list[dict[str, str]],
    spatial_root: Path,
    pose_samples: list[PoseSample],
    depth_min_m: float = 0.1,
    depth_max_m: float = 5.0,
) -> dict:
    t_h_from_c = normalized_sensor_extrinsics(raw_t_h_from_c, interpretation)
    t_h_from_d = normalized_sensor_extrinsics(raw_t_h_from_d, interpretation)
    rgb_times = np.asarray([float(row["rgb_timestamp_s"]) for row in rgb_rows], dtype=np.float64)
    width = int(round(max(float(rgb_intrinsics["cx"]) * 2.0 + 1.0, 1.0)))
    height = int(round(max(float(rgb_intrinsics["cy"]) * 2.0 + 1.0, 1.0)))
    frame_metrics = []

    for depth_index, row in enumerate(depth_rows):
        depth_time = float(row["depth_timestamp_s"])
        rgb_index = int(np.argmin(np.abs(rgb_times - depth_time)))
        rgb_path = rgb_dir / f"{rgb_index:06d}.png"
        depth = np.load(spatial_root / row["depth_meters_npy"]).astype(np.float32)
        valid = np.isfinite(depth) & (depth >= depth_min_m) & (depth <= depth_max_m)
        if int(valid.sum()) < 64 or not rgb_path.is_file():
            continue
        filtered = np.where(valid, depth, 0.0)
        points_d, pixels_d = backproject_depth(filtered, depth_intrinsics)
        pose_d = interpolate_pose(pose_samples, depth_time)[0]
        pose_c = interpolate_pose(pose_samples, float(rgb_times[rgb_index]))[0]
        t_w_from_d = pose_matrix(pose_d) @ t_h_from_d
        t_w_from_c = pose_matrix(pose_c) @ t_h_from_c
        points_c = transform_points(points_d, np.linalg.inv(t_w_from_c) @ t_w_from_d)
        z = points_c[:, 2]
        u = np.rint(float(rgb_intrinsics["fx"]) * points_c[:, 0] / z + float(rgb_intrinsics["cx"])).astype(np.int64)
        v = np.rint(float(rgb_intrinsics["fy"]) * points_c[:, 1] / z + float(rgb_intrinsics["cy"])).astype(np.int64)
        inside = np.isfinite(points_c).all(axis=1) & (z > 1e-6) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        source_edges = _depth_edges(depth, valid)
        edge_flags = source_edges[pixels_d[:, 1].astype(np.int64), pixels_d[:, 0].astype(np.int64)]
        edge_inside = inside & edge_flags
        gradient = _rgb_gradient(rgb_path)
        edge_gradient = gradient[v[edge_inside], u[edge_inside]] if np.any(edge_inside) else np.asarray([])
        all_gradient = gradient[v[inside], u[inside]] if np.any(inside) else np.asarray([])
        frame_metrics.append(
            {
                "depth_index": depth_index,
                "rgb_frame_index": rgb_index,
                "timestamp_delta_s": float(rgb_times[rgb_index] - depth_time),
                "projected_inside_ratio": float(inside.mean()),
                "projected_edge_count": int(edge_inside.sum()),
                "rgb_gradient_at_depth_edges": float(edge_gradient.mean()) if edge_gradient.size else 0.0,
                "rgb_gradient_at_all_depth_points": float(all_gradient.mean()) if all_gradient.size else 0.0,
            }
        )

    if not frame_metrics:
        raise RuntimeError("No valid RGB/depth pairs were available for extrinsics validation")
    inside = float(np.median([item["projected_inside_ratio"] for item in frame_metrics]))
    edge_gradient = float(np.median([item["rgb_gradient_at_depth_edges"] for item in frame_metrics]))
    baseline_gradient = float(np.median([item["rgb_gradient_at_all_depth_points"] for item in frame_metrics]))
    edge_gain = edge_gradient - baseline_gradient
    return {
        "interpretation": interpretation,
        "pair_count": len(frame_metrics),
        "median_projected_inside_ratio": inside,
        "median_rgb_gradient_at_depth_edges": edge_gradient,
        "median_rgb_gradient_at_all_depth_points": baseline_gradient,
        "median_edge_gradient_gain": edge_gain,
        "score": inside + 0.35 * edge_gain,
        "frames": frame_metrics,
    }


def validate_extrinsics_direction(**kwargs) -> dict:
    candidates = [
        evaluate_extrinsics_interpretation(interpretation=name, **kwargs)
        for name in ("sensor_to_head", "head_to_sensor")
    ]
    candidates.sort(key=lambda item: item["score"], reverse=True)
    margin = float(candidates[0]["score"] - candidates[1]["score"])
    selected = candidates[0]["interpretation"]
    # SpatialMP4's Quest example declares sensor_to_head. Keep that convention
    # when the image/depth evidence is too close to distinguish automatically.
    if margin < 0.015:
        selected = "sensor_to_head"
    return {
        "method": "project both raw-extrinsic conventions and compare in-frame coverage plus RGB/depth edge agreement",
        "selected_interpretation": selected,
        "score_margin": margin,
        "decision": "measured" if margin >= 0.015 else "ambiguous_use_spatialmp4_documented_sensor_to_head",
        "candidates": candidates,
    }
