"""Metric calibration and sparse-depth-guided fusion for Video Depth Anything."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt


@dataclass
class AnchorFit:
    frame_index: int
    depth_index: int
    timestamp_s: float
    representation: str
    scale: float
    shift: float
    valid_count: int
    inlier_count: int
    rmse_m: float
    median_abs_error_m: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resize_prediction(prediction: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    prediction = np.asarray(prediction, dtype=np.float32)
    if prediction.shape == shape:
        return prediction
    return cv2.resize(prediction, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)


def convert_representation(prediction: np.ndarray, representation: str) -> np.ndarray:
    prediction = np.asarray(prediction, dtype=np.float64)
    if representation == "direct":
        return prediction
    if representation == "inverse":
        positive = prediction[prediction > 1e-8]
        epsilon = max(float(np.quantile(positive, 0.001)) * 0.05, 1e-8) if positive.size else 1e-8
        return 1.0 / np.maximum(prediction, epsilon)
    raise ValueError(f"Unknown depth representation: {representation}")


def robust_affine_fit(x: np.ndarray, y: np.ndarray, max_iterations: int = 6) -> tuple[float, float, np.ndarray]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y) & (y > 0)
    x, y = x[valid], y[valid]
    if len(x) < 64:
        raise ValueError(f"Not enough valid depth calibration pixels: {len(x)}")
    x_lo, x_hi = np.quantile(x, [0.01, 0.99])
    y_lo, y_hi = np.quantile(y, [0.01, 0.99])
    keep = (x >= x_lo) & (x <= x_hi) & (y >= y_lo) & (y <= y_hi)
    design = np.column_stack([x, np.ones_like(x)])
    for _ in range(max_iterations):
        if int(keep.sum()) < 32:
            break
        scale, shift = np.linalg.lstsq(design[keep], y[keep], rcond=None)[0]
        residual = y - (scale * x + shift)
        center = float(np.median(residual[keep]))
        mad = float(np.median(np.abs(residual[keep] - center)))
        threshold = max(3.5 * 1.4826 * mad, 0.015)
        next_keep = keep & (np.abs(residual - center) <= threshold)
        if np.array_equal(next_keep, keep):
            break
        keep = next_keep
    scale, shift = np.linalg.lstsq(design[keep], y[keep], rcond=None)[0]
    return float(scale), float(shift), keep


def fit_anchor(
    prediction: np.ndarray,
    true_depth: np.ndarray,
    *,
    frame_index: int,
    depth_index: int,
    timestamp_s: float,
    representation: str,
    depth_min_m: float,
    depth_max_m: float,
) -> AnchorFit:
    pred = convert_representation(prediction, representation)
    true = np.asarray(true_depth, dtype=np.float64)
    valid = (
        np.isfinite(pred)
        & np.isfinite(true)
        & (true >= depth_min_m)
        & (true <= depth_max_m)
        & (pred > 0)
    )
    scale, shift, inliers = robust_affine_fit(pred[valid], true[valid])
    estimated = scale * pred[valid] + shift
    residual = estimated[inliers] - true[valid][inliers]
    return AnchorFit(
        frame_index=frame_index,
        depth_index=depth_index,
        timestamp_s=timestamp_s,
        representation=representation,
        scale=scale,
        shift=shift,
        valid_count=int(valid.sum()),
        inlier_count=int(inliers.sum()),
        rmse_m=float(np.sqrt(np.mean(residual**2))),
        median_abs_error_m=float(np.median(np.abs(residual))),
    )


def evaluate_metric_depth(
    estimated_depth: np.ndarray,
    true_depth: np.ndarray,
    *,
    depth_min_m: float,
    depth_max_m: float,
) -> dict[str, float | int | None]:
    estimated = np.asarray(estimated_depth, dtype=np.float64)
    true = np.asarray(true_depth, dtype=np.float64)
    true_valid = np.isfinite(true) & (true >= depth_min_m) & (true <= depth_max_m)
    valid = (
        np.isfinite(estimated)
        & (estimated >= depth_min_m)
        & (estimated <= depth_max_m)
        & true_valid
    )
    residual = estimated[valid] - true[valid]
    if residual.size == 0:
        return {"true_valid_count": int(true_valid.sum()), "valid_count": 0, "valid_ratio": 0.0, "rmse_m": None, "median_abs_error_m": None, "p90_abs_error_m": None}
    return {
        "true_valid_count": int(true_valid.sum()),
        "valid_count": int(residual.size),
        "valid_ratio": float(residual.size / max(int(true_valid.sum()), 1)),
        "rmse_m": float(np.sqrt(np.mean(residual**2))),
        "median_abs_error_m": float(np.median(np.abs(residual))),
        "p90_abs_error_m": float(np.quantile(np.abs(residual), 0.9)),
    }


def select_representation(fits: dict[str, list[AnchorFit]]) -> str:
    scores = {}
    for representation, items in fits.items():
        valid = [item for item in items if np.isfinite(item.rmse_m) and item.scale > 0]
        scores[representation] = float(np.median([item.rmse_m for item in valid])) if valid else float("inf")
    selected = min(scores, key=scores.get)
    if not np.isfinite(scores[selected]):
        raise RuntimeError(f"No valid VDA calibration representation: {scores}")
    return selected


def interpolate_calibration(
    frame_timestamps: np.ndarray,
    fits: list[AnchorFit],
) -> tuple[np.ndarray, np.ndarray]:
    if not fits:
        raise ValueError("No anchor fits")
    grouped: dict[int, list[AnchorFit]] = {}
    for fit in fits:
        grouped.setdefault(fit.frame_index, []).append(fit)
    frame_indices = np.asarray(sorted(grouped), dtype=np.int64)
    anchor_times = np.asarray([frame_timestamps[index] for index in frame_indices], dtype=np.float64)
    scales = np.asarray([np.median([fit.scale for fit in grouped[index]]) for index in frame_indices])
    shifts = np.asarray([np.median([fit.shift for fit in grouped[index]]) for index in frame_indices])
    scale_per_frame = np.interp(frame_timestamps, anchor_times, scales)
    shift_per_frame = np.interp(frame_timestamps, anchor_times, shifts)
    return scale_per_frame, shift_per_frame


def edge_aware_anchor_fusion(
    base_metric: np.ndarray,
    true_depth: np.ndarray,
    rgb: np.ndarray,
    *,
    spatial_sigma_px: float = 18.0,
    color_sigma: float = 36.0,
    depth_min_m: float = 0.1,
    depth_max_m: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    base = np.asarray(base_metric, dtype=np.float32)
    true = np.asarray(true_depth, dtype=np.float32)
    color = np.asarray(rgb, dtype=np.float32)
    valid = (
        np.isfinite(true)
        & (true >= depth_min_m)
        & (true <= depth_max_m)
        & np.isfinite(base)
        & (base > 0)
    )
    if not np.any(valid):
        return base.copy(), np.zeros_like(base, dtype=np.float32)
    residual = np.zeros_like(base, dtype=np.float32)
    residual[valid] = true[valid] - base[valid]
    distance, nearest = distance_transform_edt(~valid, return_indices=True)
    nearest_residual = residual[nearest[0], nearest[1]]
    nearest_color = color[nearest[0], nearest[1]]
    color_distance = np.linalg.norm(color - nearest_color, axis=2)
    confidence = np.exp(-distance / max(spatial_sigma_px, 1e-6)) * np.exp(
        -color_distance / max(color_sigma, 1e-6)
    )
    fused = base + confidence.astype(np.float32) * nearest_residual
    fused[valid] = true[valid]
    fused[(fused < depth_min_m) | (fused > depth_max_m) | ~np.isfinite(fused)] = 0.0
    confidence[valid] = 1.0
    return fused.astype(np.float32), confidence.astype(np.float32)


def colorize_depth(depth: np.ndarray, depth_min_m: float, depth_max_m: float) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    normalized = np.zeros_like(depth, dtype=np.float32)
    normalized[valid] = np.clip(
        (depth[valid] - depth_min_m) / max(depth_max_m - depth_min_m, 1e-6),
        0.0,
        1.0,
    )
    colored = cv2.applyColorMap(np.rint(normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    colored[~valid] = 0
    return colored
