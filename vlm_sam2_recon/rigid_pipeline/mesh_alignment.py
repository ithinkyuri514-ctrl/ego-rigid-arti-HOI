"""Metric Sim3 initialization and fixed-scale ICP for a canonical rigid mesh."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


@dataclass
class Similarity:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    @property
    def matrix(self) -> np.ndarray:
        result = np.eye(4, dtype=np.float64)
        result[:3, :3] = self.scale * self.rotation
        result[:3, 3] = self.translation
        return result

    def transform(self, points: np.ndarray) -> np.ndarray:
        return self.scale * (np.asarray(points) @ self.rotation.T) + self.translation


def trimmed_mean(values: np.ndarray, fraction: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    keep = max(1, min(len(values), int(round(len(values) * fraction))))
    return float(np.mean(np.partition(values, keep - 1)[:keep]))


def pca_basis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    center = np.median(points, axis=0)
    centered = points - center
    covariance = centered.T @ centered / max(len(points), 1)
    values, vectors = np.linalg.eigh(covariance)
    basis = vectors[:, np.argsort(values)[::-1]]
    if np.linalg.det(basis) < 0:
        basis[:, -1] *= -1
    return center, basis


def signed_permutation_rotations() -> list[np.ndarray]:
    rotations = []
    for permutation in itertools.permutations(range(3)):
        base = np.eye(3)[:, permutation]
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            matrix = base @ np.diag(signs)
            if np.linalg.det(matrix) > 0.5:
                rotations.append(matrix)
    return rotations


def symmetric_chamfer(source: np.ndarray, target: np.ndarray, trim_fraction: float) -> float:
    source_tree = cKDTree(source)
    target_tree = cKDTree(target)
    source_to_target = target_tree.query(source, k=1, workers=-1)[0]
    target_to_source = source_tree.query(target, k=1, workers=-1)[0]
    return 0.5 * (
        trimmed_mean(source_to_target, trim_fraction)
        + trimmed_mean(target_to_source, trim_fraction)
    )


def pca_similarity_candidates(
    source: np.ndarray,
    target: np.ndarray,
    trim_fraction: float = 0.7,
) -> list[tuple[float, Similarity]]:
    source_center, source_basis = pca_basis(source)
    target_center, target_basis = pca_basis(target)
    source_radius = float(np.quantile(np.linalg.norm(source - source_center, axis=1), 0.9))
    target_radius = float(np.quantile(np.linalg.norm(target - target_center, axis=1), 0.9))
    scale = target_radius / max(source_radius, 1e-12)
    candidates = []
    for permutation in signed_permutation_rotations():
        rotation = target_basis @ permutation @ source_basis.T
        if np.linalg.det(rotation) < 0.5:
            continue
        translation = target_center - scale * (rotation @ source_center)
        transform = Similarity(scale, rotation, translation)
        score = symmetric_chamfer(transform.transform(source), target, trim_fraction)
        candidates.append((score, transform))
    candidates.sort(key=lambda item: item[0])
    return candidates


def umeyama(source: np.ndarray, target: np.ndarray, allow_scale: bool) -> Similarity:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_centered = source - source_center
    target_centered = target - target_center
    covariance = target_centered.T @ source_centered / len(source)
    left, singular, right_t = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(left @ right_t) < 0:
        correction[-1, -1] = -1
    rotation = left @ correction @ right_t
    if allow_scale:
        variance = float(np.mean(np.sum(source_centered**2, axis=1)))
        scale = float(np.sum(singular * np.diag(correction)) / max(variance, 1e-12))
    else:
        scale = 1.0
    translation = target_center - scale * (rotation @ source_center)
    return Similarity(scale, rotation, translation)


def refine_similarity_icp(
    source: np.ndarray,
    target: np.ndarray,
    initial: Similarity,
    *,
    iterations: int = 25,
    trim_fraction: float = 0.65,
) -> tuple[Similarity, list[dict]]:
    current = initial
    tree = cKDTree(target)
    history = []
    for iteration in range(iterations):
        transformed = current.transform(source)
        distances, indices = tree.query(transformed, k=1, workers=-1)
        keep_count = max(12, int(len(distances) * trim_fraction))
        keep = np.argpartition(distances, keep_count - 1)[:keep_count]
        estimate = umeyama(source[keep], target[indices[keep]], allow_scale=True)
        if estimate.scale <= 0 or not np.isfinite(estimate.scale):
            break
        change = abs(estimate.scale - current.scale) / max(current.scale, 1e-12)
        current = estimate
        score = float(np.sqrt(np.mean(distances[keep] ** 2)))
        history.append({"iteration": iteration, "rmse_m": score, "scale": current.scale})
        if change < 1e-5 and iteration > 3:
            break
    return current, history


def kabsch(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (target - target_center).T @ (source - source_center)
    left, _, right_t = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(left @ right_t) < 0:
        correction[-1, -1] = -1
    rotation = left @ correction @ right_t
    translation = target_center - rotation @ source_center
    return rotation, translation


def fixed_scale_icp(
    source: np.ndarray,
    target: np.ndarray,
    initial: Similarity,
    *,
    max_distances_m: tuple[float, ...] = (0.08, 0.05, 0.03, 0.018),
    iterations_per_level: int = 15,
    trim_fraction: float = 0.7,
) -> tuple[Similarity, list[dict]]:
    scale = float(initial.scale)
    rotation = np.asarray(initial.rotation, dtype=np.float64)
    translation = np.asarray(initial.translation, dtype=np.float64)
    tree = cKDTree(target)
    history = []
    for level, max_distance in enumerate(max_distances_m):
        for iteration in range(iterations_per_level):
            transformed = scale * (source @ rotation.T) + translation
            distances, indices = tree.query(transformed, k=1, workers=-1)
            valid = np.flatnonzero(distances <= max_distance)
            if len(valid) < 20:
                valid = np.argsort(distances)[: max(20, int(len(distances) * 0.35))]
            keep_count = max(12, min(len(valid), int(len(valid) * trim_fraction)))
            keep = valid[np.argpartition(distances[valid], keep_count - 1)[:keep_count]]
            delta_rotation, delta_translation = kabsch(transformed[keep], target[indices[keep]])
            rotation = delta_rotation @ rotation
            translation = delta_rotation @ translation + delta_translation
            rmse = float(np.sqrt(np.mean(distances[keep] ** 2)))
            angle_change = float(np.linalg.norm(delta_rotation - np.eye(3)))
            translation_change = float(np.linalg.norm(delta_translation))
            history.append(
                {
                    "level": level,
                    "iteration": iteration,
                    "max_distance_m": max_distance,
                    "correspondences": len(keep),
                    "rmse_m": rmse,
                }
            )
            if angle_change < 1e-5 and translation_change < 1e-6:
                break
    return Similarity(scale, rotation, translation), history


def estimate_point_normals(points: np.ndarray, neighbors: int = 24) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if len(points) < max(neighbors, 6):
        raise ValueError(f"Too few points for normal estimation: {len(points)}")
    tree = cKDTree(points)
    _, indices = tree.query(points, k=min(neighbors, len(points)), workers=-1)
    neighborhoods = points[indices]
    centered = neighborhoods - neighborhoods.mean(axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", centered, centered) / max(neighborhoods.shape[1], 1)
    _, vectors = np.linalg.eigh(covariance)
    normals = vectors[:, :, 0]
    # Orient toward the frame-0 camera. Point-to-plane residuals are sign
    # invariant, but consistent orientation makes diagnostics easier to read.
    flip = np.einsum("ij,ij->i", normals, points) > 0
    normals[flip] *= -1.0
    return normals


def fixed_scale_point_to_plane_icp(
    source: np.ndarray,
    target: np.ndarray,
    initial: Similarity,
    *,
    target_normals: np.ndarray | None = None,
    max_distances_m: tuple[float, ...] = (0.04, 0.025, 0.015),
    iterations_per_level: int = 12,
    trim_fraction: float = 0.75,
) -> tuple[Similarity, list[dict]]:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    normals = estimate_point_normals(target) if target_normals is None else np.asarray(target_normals, dtype=np.float64)
    scale = float(initial.scale)
    rotation = np.asarray(initial.rotation, dtype=np.float64)
    translation = np.asarray(initial.translation, dtype=np.float64)
    tree = cKDTree(target)
    history = []
    for level, max_distance in enumerate(max_distances_m):
        for iteration in range(iterations_per_level):
            transformed = scale * (source @ rotation.T) + translation
            distances, indices = tree.query(transformed, k=1, workers=-1)
            valid = np.flatnonzero(np.isfinite(distances) & (distances <= max_distance))
            if len(valid) < 30:
                break
            keep_count = max(24, min(len(valid), int(round(len(valid) * trim_fraction))))
            keep = valid[np.argpartition(distances[valid], keep_count - 1)[:keep_count]]
            points = transformed[keep]
            matches = target[indices[keep]]
            match_normals = normals[indices[keep]]
            design = np.column_stack([np.cross(points, match_normals), match_normals])
            rhs = np.einsum("ij,ij->i", match_normals, matches - points)
            delta, *_ = np.linalg.lstsq(design, rhs, rcond=None)
            rotation_step = delta[:3]
            translation_step = delta[3:]
            rotation_norm = float(np.linalg.norm(rotation_step))
            translation_norm = float(np.linalg.norm(translation_step))
            limiter = max(rotation_norm / 0.12, translation_norm / 0.025, 1.0)
            rotation_step /= limiter
            translation_step /= limiter
            delta_rotation = Rotation.from_rotvec(rotation_step).as_matrix()
            rotation = delta_rotation @ rotation
            translation = delta_rotation @ translation + translation_step
            point_plane_rmse = float(np.sqrt(np.mean(rhs**2)))
            history.append(
                {
                    "level": level,
                    "iteration": iteration,
                    "max_distance_m": max_distance,
                    "correspondences": len(keep),
                    "point_to_point_rmse_m": float(np.sqrt(np.mean(distances[keep] ** 2))),
                    "point_to_plane_rmse_m": point_plane_rmse,
                    "rotation_step_rad": float(np.linalg.norm(rotation_step)),
                    "translation_step_m": float(np.linalg.norm(translation_step)),
                }
            )
            if np.linalg.norm(rotation_step) < 1e-5 and np.linalg.norm(translation_step) < 1e-6:
                break
    return Similarity(scale, rotation, translation), history


def projection_diagnostics(
    aligned_points: np.ndarray,
    observed_depth: np.ndarray,
    object_mask: np.ndarray,
    intrinsics: dict[str, float],
) -> tuple[dict, np.ndarray, np.ndarray]:
    height, width = observed_depth.shape
    points = np.asarray(aligned_points, dtype=np.float64)
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 1e-6)
    points = points[valid]
    z = points[:, 2]
    u = np.rint(float(intrinsics["fx"]) * points[:, 0] / z + float(intrinsics["cx"])).astype(np.int64)
    v = np.rint(float(intrinsics["fy"]) * points[:, 1] / z + float(intrinsics["cy"])).astype(np.int64)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v, z = u[inside], v[inside], z[inside]
    rendered_depth = np.zeros((height, width), dtype=np.float32)
    flat = v * width + u
    order = np.argsort(z)
    flat_sorted = flat[order]
    _, first = np.unique(flat_sorted, return_index=True)
    chosen = order[first]
    rendered_depth.reshape(-1)[flat[chosen]] = z[chosen]
    rendered_mask = rendered_depth > 0
    rendered_mask = cv2.dilate(rendered_mask.astype(np.uint8), np.ones((7, 7), np.uint8), iterations=1) > 0
    intersection = int(np.count_nonzero(rendered_mask & object_mask))
    union = int(np.count_nonzero(rendered_mask | object_mask))
    # Use only pixels with an actual z-buffer sample for depth residuals.
    # The dilated rendered mask is appropriate for silhouette IoU but its
    # added pixels have zero rendered depth and would create a false ~Z error.
    overlap = (rendered_depth > 0) & object_mask & (observed_depth > 0)
    residual = rendered_depth[overlap] - observed_depth[overlap]
    abs_residual = np.abs(residual)
    trim_fraction = 0.9
    trim_count = max(1, int(np.floor(abs_residual.size * trim_fraction))) if abs_residual.size else 0
    trimmed_residual = (
        residual[np.argpartition(abs_residual, trim_count - 1)[:trim_count]]
        if trim_count
        else residual
    )
    metrics = {
        "silhouette_iou_point_splat": float(intersection / max(union, 1)),
        "rendered_mask_pixels": int(rendered_mask.sum()),
        "observed_mask_pixels": int(object_mask.sum()),
        "depth_overlap_pixels": int(overlap.sum()),
        "depth_rmse_m": float(np.sqrt(np.mean(residual**2))) if residual.size else None,
        "depth_trim_fraction": trim_fraction,
        "depth_trimmed_pixels": int(trim_count),
        "depth_trimmed_rmse_m": (
            float(np.sqrt(np.mean(trimmed_residual**2))) if trimmed_residual.size else None
        ),
        "depth_median_abs_m": float(np.median(abs_residual)) if residual.size else None,
    }
    return metrics, rendered_mask, rendered_depth
