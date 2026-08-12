"""Coordinate conversion and projection checks for EgoForce outputs."""

from __future__ import annotations

import numpy as np


GEOMETRY_KEYS = (
    "hand_vertices",
    "arm_vertices",
    "hand_joints",
    "arm_joints",
    "transl",
    "mano_transl",
)


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply an SE(3) transform to an array whose last dimension is XYZ."""
    points = np.asarray(points)
    transform = np.asarray(transform, dtype=np.float64)
    if points.shape[-1] != 3:
        raise ValueError(f"Expected points with last dimension 3, got {points.shape}")
    if transform.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 transform, got {transform.shape}")
    return (points @ transform[:3, :3].T + transform[:3, 3]).astype(np.float32)


def make_c0_payload(raw: dict[str, np.ndarray], transform: np.ndarray) -> dict[str, np.ndarray]:
    """Create a geometry-only C0 payload without relabeling camera-local pose parameters."""
    payload: dict[str, np.ndarray] = {
        "coordinate_frame": np.asarray("frame0_right_camera_opencv_rdf"),
        "source_coordinate_frame": np.asarray("current_right_camera_opencv_rdf"),
        "T_C0_from_Ct": np.asarray(transform, dtype=np.float64),
    }
    for key in (
        "visible_hand",
        "left_hand_faces",
        "right_hand_faces",
        "arm_faces",
    ):
        if key in raw:
            payload[key] = np.asarray(raw[key])
    for key in GEOMETRY_KEYS:
        if key in raw:
            payload[key] = transform_points(raw[key], transform)
    for key in (
        "egoforce_hand_keypoints_2d",
        "egoforce_hand_keypoint_confidence",
        "egoforce_arm_keypoints_2d",
        "egoforce_arm_keypoint_confidence",
    ):
        if key in raw:
            payload[f"{key}_Ct_image"] = np.asarray(raw[key])
    return payload


def project_points(points: np.ndarray, intrinsics: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    z = points[..., 2]
    valid = np.isfinite(points).all(axis=-1) & (z > 1e-8)
    uv = np.full((*points.shape[:-1], 2), np.nan, dtype=np.float64)
    uv[..., 0][valid] = float(intrinsics["fx"]) * points[..., 0][valid] / z[valid] + float(
        intrinsics["cx"]
    )
    uv[..., 1][valid] = float(intrinsics["fy"]) * points[..., 1][valid] / z[valid] + float(
        intrinsics["cy"]
    )
    return uv, valid


def projection_metrics(
    points: np.ndarray,
    intrinsics: dict[str, float],
    width: int,
    height: int,
) -> dict[str, float | int]:
    uv, positive = project_points(points, intrinsics)
    finite = np.isfinite(np.asarray(points)).all(axis=-1)
    inside = (
        positive
        & (uv[..., 0] >= 0)
        & (uv[..., 0] < width)
        & (uv[..., 1] >= 0)
        & (uv[..., 1] < height)
    )
    count = int(finite.size)
    return {
        "point_count": count,
        "finite_ratio": float(finite.mean()) if count else 0.0,
        "positive_depth_ratio": float(positive.mean()) if count else 0.0,
        "inside_image_ratio": float(inside.mean()) if count else 0.0,
    }


def joint_reprojection_error(
    joints_ct: np.ndarray,
    target_uv: np.ndarray,
    confidence: np.ndarray | None,
    intrinsics: dict[str, float],
) -> dict[str, float | int | None]:
    projected, valid = project_points(joints_ct, intrinsics)
    target_uv = np.asarray(target_uv, dtype=np.float64)
    count = min(len(projected), len(target_uv))
    projected, target_uv, valid = projected[:count], target_uv[:count], valid[:count]
    valid &= np.isfinite(target_uv).all(axis=-1)
    if confidence is not None:
        confidence = np.asarray(confidence).reshape(-1)[:count]
        # EgoForce stores learned regression weights here, not [0, 1] detector scores.
        valid &= np.isfinite(confidence) & (confidence > 0.0)
    errors = np.linalg.norm(projected[valid] - target_uv[valid], axis=-1)
    return {
        "matched_joint_count": int(len(errors)),
        "median_error_px": float(np.median(errors)) if len(errors) else None,
        "p90_error_px": float(np.percentile(errors, 90)) if len(errors) else None,
    }
