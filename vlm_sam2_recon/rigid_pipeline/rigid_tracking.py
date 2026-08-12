"""Robust RGB-D point-track filtering and rigid SE(3) estimation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial.transform import Rotation


REJECTION_NAMES = {
    0: "valid",
    1: "tracker_invisible",
    2: "outside_image",
    3: "outside_object_mask",
    4: "missing_local_depth",
    5: "unstable_local_depth",
    6: "temporal_depth_spike",
    7: "temporal_3d_spike",
    8: "globally_nonrigid_track",
    9: "ransac_outlier",
}


@dataclass
class PoseEstimate:
    matrix: np.ndarray
    inliers: np.ndarray
    residuals_m: np.ndarray
    rmse_m: float


def kabsch_se3(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(f"Expected matching Nx3 arrays, got {source.shape} and {target.shape}")
    if len(source) < 3:
        raise ValueError("At least three correspondences are required")
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (target - target_center).T @ (source - source_center)
    left, _, right_t = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(left @ right_t) < 0:
        correction[-1, -1] = -1
    rotation = left @ correction @ right_t
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = target_center - rotation @ source_center
    return transform


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    transform = np.asarray(transform, dtype=np.float64)
    return points @ transform[:3, :3].T + transform[:3, 3]


def ransac_se3(
    source: np.ndarray,
    target: np.ndarray,
    *,
    threshold_m: float,
    iterations: int,
    min_inliers: int,
    seed: int,
) -> PoseEstimate | None:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if len(source) < max(3, min_inliers):
        return None
    rng = np.random.default_rng(seed)
    best_inliers = None
    best_score = (-1, np.inf)
    for _ in range(iterations):
        sample = rng.choice(len(source), 3, replace=False)
        centered = source[sample] - source[sample].mean(axis=0)
        if np.linalg.matrix_rank(centered, tol=1e-5) < 2:
            continue
        try:
            transform = kabsch_se3(source[sample], target[sample])
        except np.linalg.LinAlgError:
            continue
        residuals = np.linalg.norm(transform_points(source, transform) - target, axis=1)
        inliers = residuals <= threshold_m
        count = int(inliers.sum())
        median = float(np.median(residuals[inliers])) if count else np.inf
        score = (count, -median)
        if score > best_score:
            best_score = score
            best_inliers = inliers
    if best_inliers is None or int(best_inliers.sum()) < min_inliers:
        return None
    inliers = best_inliers.copy()
    for _ in range(3):
        transform = kabsch_se3(source[inliers], target[inliers])
        residuals = np.linalg.norm(transform_points(source, transform) - target, axis=1)
        updated = residuals <= threshold_m
        if int(updated.sum()) < min_inliers or np.array_equal(updated, inliers):
            break
        inliers = updated
    transform = kabsch_se3(source[inliers], target[inliers])
    residuals = np.linalg.norm(transform_points(source, transform) - target, axis=1)
    return PoseEstimate(
        matrix=transform,
        inliers=inliers,
        residuals_m=residuals,
        rmse_m=float(np.sqrt(np.mean(residuals[inliers] ** 2))),
    )


def sample_metric_depth_tracks(
    tracks_xy: np.ndarray,
    tracker_valid: np.ndarray,
    masks: np.ndarray,
    depths_m: np.ndarray,
    intrinsics: dict[str, float],
    *,
    radius_px: int = 3,
    depth_min_m: float = 0.1,
    depth_max_m: float = 3.0,
    max_local_mad_m: float = 0.025,
    mask_margin_px: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tracks_xy = np.asarray(tracks_xy, dtype=np.float64)
    tracker_valid = np.asarray(tracker_valid, dtype=bool)
    masks = np.asarray(masks, dtype=bool)
    depths_m = np.asarray(depths_m, dtype=np.float32)
    frames, points = tracks_xy.shape[:2]
    height, width = masks.shape[1:]
    xyz = np.full((frames, points, 3), np.nan, dtype=np.float64)
    sampled_depth = np.full((frames, points), np.nan, dtype=np.float64)
    local_mad = np.full((frames, points), np.nan, dtype=np.float64)
    rejection = np.full((frames, points), 1, dtype=np.uint8)
    mask_distances = np.stack([distance_transform_edt(mask) for mask in masks])
    fx, fy = float(intrinsics["fx"]), float(intrinsics["fy"])
    cx, cy = float(intrinsics["cx"]), float(intrinsics["cy"])
    for frame in range(frames):
        for point in range(points):
            if not tracker_valid[frame, point]:
                continue
            u, v = tracks_xy[frame, point]
            x, y = int(round(u)), int(round(v))
            if x < 0 or x >= width or y < 0 or y >= height:
                rejection[frame, point] = 2
                continue
            if not masks[frame, y, x] or mask_distances[frame, y, x] < mask_margin_px:
                rejection[frame, point] = 3
                continue
            x0, x1 = max(0, x - radius_px), min(width, x + radius_px + 1)
            y0, y1 = max(0, y - radius_px), min(height, y + radius_px + 1)
            patch = depths_m[frame, y0:y1, x0:x1]
            patch_mask = masks[frame, y0:y1, x0:x1]
            values = patch[
                patch_mask
                & np.isfinite(patch)
                & (patch >= depth_min_m)
                & (patch <= depth_max_m)
            ]
            if values.size < 5:
                rejection[frame, point] = 4
                continue
            z = float(np.median(values))
            mad = float(np.median(np.abs(values - z)))
            sampled_depth[frame, point] = z
            local_mad[frame, point] = mad
            if mad > max_local_mad_m:
                rejection[frame, point] = 5
                continue
            xyz[frame, point] = ((u - cx) * z / fx, (v - cy) * z / fy, z)
            rejection[frame, point] = 0
    return xyz, sampled_depth, local_mad, rejection


def reject_temporal_spikes(
    xyz_camera: np.ndarray,
    depths_m: np.ndarray,
    rejection: np.ndarray,
    query_times: np.ndarray,
    *,
    depth_spike_m: float = 0.055,
    point_spike_m: float = 0.075,
) -> np.ndarray:
    output = np.asarray(rejection, dtype=np.uint8).copy()
    frames, points = depths_m.shape
    for point in range(points):
        start = int(query_times[point])
        for frame in range(max(start + 1, 1), frames - 1):
            if output[frame, point] != 0:
                continue
            prev_valid = output[frame - 1, point] == 0
            next_valid = output[frame + 1, point] == 0
            if not (prev_valid and next_valid):
                continue
            expected_depth = 0.5 * (depths_m[frame - 1, point] + depths_m[frame + 1, point])
            if (
                abs(depths_m[frame, point] - expected_depth) > depth_spike_m
                and abs(depths_m[frame, point] - depths_m[frame - 1, point]) > depth_spike_m
                and abs(depths_m[frame, point] - depths_m[frame + 1, point]) > depth_spike_m
            ):
                output[frame, point] = 6
                continue
            expected_xyz = 0.5 * (xyz_camera[frame - 1, point] + xyz_camera[frame + 1, point])
            if (
                np.linalg.norm(xyz_camera[frame, point] - expected_xyz) > point_spike_m
                and np.linalg.norm(xyz_camera[frame, point] - xyz_camera[frame - 1, point]) > point_spike_m
                and np.linalg.norm(xyz_camera[frame, point] - xyz_camera[frame + 1, point]) > point_spike_m
            ):
                output[frame, point] = 7
    return output


def pose_step_metrics(previous: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    relative = np.asarray(current) @ np.linalg.inv(np.asarray(previous))
    rotation_deg = float(np.rad2deg(Rotation.from_matrix(relative[:3, :3]).magnitude()))
    translation_m = float(np.linalg.norm(relative[:3, 3]))
    return rotation_deg, translation_m


def estimate_pose_sequence(
    points_c0: np.ndarray,
    rejection: np.ndarray,
    query_times: np.ndarray,
    *,
    globally_rejected: np.ndarray | None = None,
    ransac_threshold_m: float = 0.025,
    ransac_iterations: int = 500,
    min_inliers: int = 8,
    max_step_rotation_deg: float = 45.0,
    max_step_translation_m: float = 0.18,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    points_c0 = np.asarray(points_c0, dtype=np.float64)
    frames, point_count = points_c0.shape[:2]
    query_times = np.asarray(query_times, dtype=np.int64)
    globally_rejected = (
        np.zeros(point_count, dtype=bool)
        if globally_rejected is None
        else np.asarray(globally_rejected, dtype=bool)
    )
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], frames, axis=0)
    local_points = np.full((point_count, 3), np.nan, dtype=np.float64)
    assigned = np.zeros(point_count, dtype=bool)
    ransac_inlier_mask = np.zeros((frames, point_count), dtype=bool)
    diagnostics: list[dict] = []
    for frame in range(frames):
        if frame == 0:
            status = "completed"
            failure_reason = None
            candidate_indices = np.empty(0, dtype=np.int64)
            estimate = None
        else:
            candidate_indices = np.flatnonzero(
                assigned
                & ~globally_rejected
                & (rejection[frame] == 0)
                & np.isfinite(points_c0[frame]).all(axis=1)
            )
            estimate = ransac_se3(
                local_points[candidate_indices],
                points_c0[frame, candidate_indices],
                threshold_m=ransac_threshold_m,
                iterations=ransac_iterations,
                min_inliers=min_inliers,
                seed=seed + frame,
            )
            status = "completed"
            failure_reason = None
            if estimate is None:
                poses[frame] = poses[frame - 1]
                status = "failed"
                failure_reason = "insufficient_ransac_inliers"
            else:
                rotation_step, translation_step = pose_step_metrics(poses[frame - 1], estimate.matrix)
                if rotation_step > max_step_rotation_deg or translation_step > max_step_translation_m:
                    poses[frame] = poses[frame - 1]
                    status = "failed"
                    failure_reason = "implausible_pose_step"
                else:
                    poses[frame] = estimate.matrix
                    ransac_inlier_mask[frame, candidate_indices[estimate.inliers]] = True
        new_indices = np.flatnonzero(
            (query_times == frame)
            & ~globally_rejected
            & (rejection[frame] == 0)
            & np.isfinite(points_c0[frame]).all(axis=1)
        )
        if status == "completed" and new_indices.size:
            local_points[new_indices] = transform_points(points_c0[frame, new_indices], np.linalg.inv(poses[frame]))
            assigned[new_indices] = True
        rotation_step, translation_step = (
            (0.0, 0.0) if frame == 0 else pose_step_metrics(poses[frame - 1], poses[frame])
        )
        diagnostics.append(
            {
                "frame_index": frame,
                "status": status,
                "failure_reason": failure_reason,
                "candidate_track_count": int(len(candidate_indices)),
                "ransac_inlier_count": int(ransac_inlier_mask[frame].sum()),
                "ransac_rmse_m": estimate.rmse_m if estimate is not None and status == "completed" else None,
                "rotation_step_deg": rotation_step,
                "translation_step_m": translation_step,
                "new_anchor_tracks": int(len(new_indices)),
            }
        )
    return poses, local_points, ransac_inlier_mask, diagnostics


def identify_globally_bad_tracks(
    points_c0: np.ndarray,
    rejection: np.ndarray,
    poses: np.ndarray,
    local_points: np.ndarray,
    query_times: np.ndarray,
    *,
    max_median_residual_m: float = 0.025,
    max_q90_residual_m: float = 0.06,
    min_samples: int = 5,
    min_valid_ratio: float = 0.12,
) -> tuple[np.ndarray, list[dict]]:
    point_count = points_c0.shape[1]
    bad = np.zeros(point_count, dtype=bool)
    records = []
    for point in range(point_count):
        start = int(query_times[point])
        active_frames = np.arange(start, len(poses))
        valid = (rejection[active_frames, point] == 0) & np.isfinite(points_c0[active_frames, point]).all(axis=1)
        frame_ids = active_frames[valid]
        residuals = np.asarray([], dtype=np.float64)
        if np.isfinite(local_points[point]).all() and frame_ids.size:
            predicted = np.stack([transform_points(local_points[point][None], poses[t])[0] for t in frame_ids])
            residuals = np.linalg.norm(predicted - points_c0[frame_ids, point], axis=1)
        valid_ratio = float(frame_ids.size / max(len(active_frames), 1))
        median = float(np.median(residuals)) if residuals.size else None
        q90 = float(np.quantile(residuals, 0.9)) if residuals.size else None
        reasons = []
        if valid_ratio < min_valid_ratio:
            reasons.append("low_valid_ratio")
        if residuals.size >= min_samples and median is not None and median > max_median_residual_m:
            reasons.append("high_median_rigid_residual")
        if residuals.size >= min_samples and q90 is not None and q90 > max_q90_residual_m:
            reasons.append("high_q90_rigid_residual")
        if not np.isfinite(local_points[point]).all():
            reasons.append("reference_point_unassigned")
        bad[point] = bool(reasons)
        records.append(
            {
                "track_index": point,
                "query_frame": start,
                "active_frames": int(len(active_frames)),
                "valid_samples": int(frame_ids.size),
                "valid_ratio": valid_ratio,
                "rigid_residual_median_m": median,
                "rigid_residual_q90_m": q90,
                "rejected": bool(bad[point]),
                "reasons": reasons,
            }
        )
    return bad, records


def estimate_pairwise_pose_sequence(
    points_c0: np.ndarray,
    rejection: np.ndarray,
    *,
    globally_rejected: np.ndarray | None = None,
    ransac_threshold_m: float = 0.025,
    ransac_iterations: int = 500,
    min_inliers: int = 8,
    max_step_rotation_deg: float = 45.0,
    max_step_translation_m: float = 0.18,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Accumulate pairwise rigid transforms after head-pose compensation."""
    points_c0 = np.asarray(points_c0, dtype=np.float64)
    frames, point_count = points_c0.shape[:2]
    globally_rejected = (
        np.zeros(point_count, dtype=bool)
        if globally_rejected is None
        else np.asarray(globally_rejected, dtype=bool)
    )
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], frames, axis=0)
    inlier_mask = np.zeros((frames, point_count), dtype=bool)
    diagnostics = [
        {
            "frame_index": 0,
            "status": "completed",
            "failure_reason": None,
            "candidate_track_count": 0,
            "ransac_inlier_count": 0,
            "ransac_rmse_m": None,
            "rotation_step_deg": 0.0,
            "translation_step_m": 0.0,
            "transform_translation_norm_m": 0.0,
        }
    ]
    for frame in range(1, frames):
        candidates = np.flatnonzero(
            ~globally_rejected
            & (rejection[frame - 1] == 0)
            & (rejection[frame] == 0)
            & np.isfinite(points_c0[frame - 1]).all(axis=1)
            & np.isfinite(points_c0[frame]).all(axis=1)
        )
        estimate = ransac_se3(
            points_c0[frame - 1, candidates],
            points_c0[frame, candidates],
            threshold_m=ransac_threshold_m,
            iterations=ransac_iterations,
            min_inliers=min_inliers,
            seed=seed + frame,
        )
        status = "completed"
        failure_reason = None
        rotation_step = translation_step = 0.0
        if estimate is None:
            poses[frame] = poses[frame - 1]
            status = "failed"
            failure_reason = "insufficient_pairwise_ransac_inliers"
        else:
            rotation_step = float(np.rad2deg(Rotation.from_matrix(estimate.matrix[:3, :3]).magnitude()))
            transform_translation_norm = float(np.linalg.norm(estimate.matrix[:3, 3]))
            # The translation column of a relative SE(3) transform is measured at
            # the world origin.  It can be large when a distant object rotates in
            # place, even though the object itself barely translates.  Gate the
            # motion at the RANSAC inlier centroid instead.
            inlier_source = points_c0[frame - 1, candidates[estimate.inliers]]
            source_center = inlier_source.mean(axis=0)
            target_center = transform_points(source_center[None], estimate.matrix)[0]
            translation_step = float(np.linalg.norm(target_center - source_center))
            if rotation_step > max_step_rotation_deg or translation_step > max_step_translation_m:
                poses[frame] = poses[frame - 1]
                status = "failed"
                failure_reason = "implausible_pairwise_pose_step"
            else:
                poses[frame] = estimate.matrix @ poses[frame - 1]
                inlier_mask[frame, candidates[estimate.inliers]] = True
        diagnostics.append(
            {
                "frame_index": frame,
                "status": status,
                "failure_reason": failure_reason,
                "candidate_track_count": int(len(candidates)),
                "ransac_inlier_count": int(inlier_mask[frame].sum()),
                "ransac_rmse_m": estimate.rmse_m if estimate is not None and status == "completed" else None,
                "rotation_step_deg": rotation_step,
                "translation_step_m": translation_step,
                "transform_translation_norm_m": (
                    transform_translation_norm if estimate is not None else None
                ),
            }
        )
    return poses, inlier_mask, diagnostics


def identify_bad_pairwise_tracks(
    points_c0: np.ndarray,
    rejection: np.ndarray,
    poses: np.ndarray,
    frame_diagnostics: list[dict],
    query_times: np.ndarray,
    *,
    max_median_residual_m: float = 0.025,
    max_q90_residual_m: float = 0.06,
    min_samples: int = 5,
    min_valid_ratio: float = 0.12,
) -> tuple[np.ndarray, list[dict]]:
    point_count = points_c0.shape[1]
    residuals_by_track: list[list[float]] = [[] for _ in range(point_count)]
    valid_pairs = np.zeros(point_count, dtype=np.int64)
    for frame in range(1, len(poses)):
        if frame_diagnostics[frame]["status"] != "completed":
            continue
        relative = poses[frame] @ np.linalg.inv(poses[frame - 1])
        valid = (
            (rejection[frame - 1] == 0)
            & (rejection[frame] == 0)
            & np.isfinite(points_c0[frame - 1]).all(axis=1)
            & np.isfinite(points_c0[frame]).all(axis=1)
        )
        indices = np.flatnonzero(valid)
        if not len(indices):
            continue
        predicted = transform_points(points_c0[frame - 1, indices], relative)
        residuals = np.linalg.norm(predicted - points_c0[frame, indices], axis=1)
        for index, residual in zip(indices, residuals):
            residuals_by_track[index].append(float(residual))
            valid_pairs[index] += 1
    bad = np.zeros(point_count, dtype=bool)
    records = []
    for point in range(point_count):
        active_pairs = max(len(poses) - 1 - int(query_times[point]), 0)
        residuals = np.asarray(residuals_by_track[point], dtype=np.float64)
        # A query on the final frames has too few future pairs to establish a
        # global validity ratio.  Keep it neutral and let per-frame rejection
        # handle it instead of labelling the track globally non-rigid.
        valid_ratio = float(valid_pairs[point] / active_pairs) if active_pairs else 1.0
        median = float(np.median(residuals)) if residuals.size else None
        q90 = float(np.quantile(residuals, 0.9)) if residuals.size else None
        reasons = []
        if active_pairs >= min_samples and valid_ratio < min_valid_ratio:
            reasons.append("low_valid_pair_ratio")
        if residuals.size >= min_samples and median is not None and median > max_median_residual_m:
            reasons.append("high_pairwise_median_residual")
        if residuals.size >= min_samples and q90 is not None and q90 > max_q90_residual_m:
            reasons.append("high_pairwise_q90_residual")
        bad[point] = bool(reasons)
        records.append(
            {
                "track_index": point,
                "query_frame": int(query_times[point]),
                "active_pairs": active_pairs,
                "valid_pairs": int(valid_pairs[point]),
                "valid_pair_ratio": valid_ratio,
                "pairwise_residual_median_m": median,
                "pairwise_residual_q90_m": q90,
                "rejected": bool(bad[point]),
                "reasons": reasons,
            }
        )
    return bad, records
