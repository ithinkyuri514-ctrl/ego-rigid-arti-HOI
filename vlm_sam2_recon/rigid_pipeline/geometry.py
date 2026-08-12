"""Timestamp and camera geometry helpers for SpatialMP4 rigid reconstruction."""

from __future__ import annotations

import bisect
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PoseSample:
    timestamp_s: float
    translation: np.ndarray
    quaternion_wxyz: np.ndarray


def quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    q = q / max(np.linalg.norm(q), 1e-12)
    w, x, y, z = q
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def slerp_wxyz(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    left = np.asarray(q0, dtype=np.float64)
    right = np.asarray(q1, dtype=np.float64)
    left /= max(np.linalg.norm(left), 1e-12)
    right /= max(np.linalg.norm(right), 1e-12)
    dot = float(np.dot(left, right))
    if dot < 0.0:
        right = -right
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        result = left + float(alpha) * (right - left)
        return result / max(np.linalg.norm(result), 1e-12)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    result = (
        np.sin((1.0 - alpha) * theta) / sin_theta * left
        + np.sin(alpha * theta) / sin_theta * right
    )
    return result / max(np.linalg.norm(result), 1e-12)


def pose_matrix(sample: PoseSample) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quaternion_wxyz_to_matrix(sample.quaternion_wxyz)
    matrix[:3, 3] = np.asarray(sample.translation, dtype=np.float64)
    return matrix


def interpolate_pose(samples: list[PoseSample], timestamp_s: float) -> tuple[PoseSample, int, int, float]:
    if not samples:
        raise ValueError("Pose sample list is empty")
    timestamps = [item.timestamp_s for item in samples]
    pos = bisect.bisect_left(timestamps, timestamp_s)
    if pos <= 0:
        return samples[0], 0, 0, 0.0
    if pos >= len(samples):
        last = len(samples) - 1
        return samples[last], last, last, 0.0
    left_index = pos - 1
    right_index = pos
    left = samples[left_index]
    right = samples[right_index]
    delta = right.timestamp_s - left.timestamp_s
    alpha = 0.0 if delta <= 0 else float((timestamp_s - left.timestamp_s) / delta)
    result = PoseSample(
        timestamp_s=float(timestamp_s),
        translation=(1.0 - alpha) * left.translation + alpha * right.translation,
        quaternion_wxyz=slerp_wxyz(left.quaternion_wxyz, right.quaternion_wxyz, alpha),
    )
    return result, left_index, right_index, alpha


def intrinsics_matrix(intrinsics: dict[str, float]) -> np.ndarray:
    return np.asarray(
        [
            [intrinsics["fx"], 0.0, intrinsics["cx"]],
            [0.0, intrinsics["fy"], intrinsics["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def backproject_depth(depth: np.ndarray, intrinsics: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(depth, dtype=np.float32)
    rows, cols = np.indices(depth.shape, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    z = depth[valid]
    x = (cols[valid] - float(intrinsics["cx"])) / float(intrinsics["fx"]) * z
    y = (rows[valid] - float(intrinsics["cy"])) / float(intrinsics["fy"]) * z
    points = np.stack([x, y, z], axis=1).astype(np.float64)
    pixels = np.stack([cols[valid], rows[valid]], axis=1).astype(np.float64)
    return points, pixels


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    matrix = np.asarray(transform, dtype=np.float64)
    return pts @ matrix[:3, :3].T + matrix[:3, 3]


def project_points_zbuffer(
    points: np.ndarray,
    intrinsics: dict[str, float],
    width: int,
    height: int,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    z = points[:, 2]
    valid = np.isfinite(points).all(axis=1) & (z > 1e-6)
    points = points[valid]
    z = points[:, 2]
    u = np.rint(float(intrinsics["fx"]) * points[:, 0] / z + float(intrinsics["cx"])).astype(np.int64)
    v = np.rint(float(intrinsics["fy"]) * points[:, 1] / z + float(intrinsics["cy"])).astype(np.int64)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v, z = u[inside], v[inside], z[inside]
    flat = v * width + u
    order = np.argsort(z)
    flat_sorted = flat[order]
    _, first = np.unique(flat_sorted, return_index=True)
    chosen = order[first]
    output = np.zeros(height * width, dtype=np.float32)
    output[flat[chosen]] = z[chosen].astype(np.float32)
    return output.reshape(height, width)
