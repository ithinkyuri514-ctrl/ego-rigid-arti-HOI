#!/usr/bin/env python3
"""Stable CoTracker3 + RGB-D + one-DoF hinge tracking for laptop screens.

This module intentionally lives next to the existing camera/alignment stages and
keeps ``scripts/run_screen_cotracker_dynamic.py`` as a baseline.  The core change
is that every tracked image point is registered into the reference screen frame
using RGB-D and the current hinge pose, then each video frame optimizes only one
scalar hinge angle.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy.optimize import minimize_scalar
from scipy.spatial import cKDTree

from vlm_sam2_recon.stages.camera_alignment import (
    apply_se3_to_mesh,
    camera_to_camera_matrix,
    depth_points_in_right_camera,
    frame_name,
    frame_row,
    project_right_camera_points,
    rotate_mesh_about_axis,
    rotate_points_about_axis,
    transform_joint_se3,
    transform_points,
    write_json,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ALIGNMENT_DIR = PROJECT_ROOT / "outputs/object_alignment_screen_first_base_visible_snap/target_laptop/frame_000000"
DEFAULT_EXPORT_ROOT = Path("/code/3DVideo_2026-07-01-19-24-06-667_spatialmp4_export")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/screen_hinge_rgbd_stable/target_laptop_frames_000000_000019"
DEFAULT_COTRACKER_ROOT = Path("/code/ArtHOI-4D-Reconstruction/third_party/co-tracker")
DEFAULT_COTRACKER_CHECKPOINT = DEFAULT_COTRACKER_ROOT / "checkpoints/scaled_offline.pth"
BASE_PART_LABEL = "14"
SCREEN_PART_LABEL = "15"


@dataclass
class ScreenHingeTrackingConfig:
    alignment_dir: Path = DEFAULT_ALIGNMENT_DIR
    export_root: Path = DEFAULT_EXPORT_ROOT
    output_dir: Path = DEFAULT_OUTPUT_DIR
    start_frame: int = 0
    end_frame: int = 19
    rgb_dir_name: str = "rgb_right_png"
    tracker_rgb_dir: Path | None = None
    tracker_start_frame: int | None = None
    tracker_end_frame: int | None = None
    tracker_stride: int = 1
    tracker_fps: float = 0.0
    eval_fps: float = 0.0
    depth_dir_name: str = "depth_meters_npy"
    depth_unit: str = "m"
    depth_convention: str | None = None
    depth_sample_mode: str = "auto"  # auto, projected, aligned_rgb
    depth_min_m: float = 0.1
    depth_max_m: float = 3.0
    timestamp_tolerance_s: float = 0.035

    cotracker_root: Path = DEFAULT_COTRACKER_ROOT
    cotracker_checkpoint: Path = DEFAULT_COTRACKER_CHECKPOINT
    device: str = "cuda"
    tracker_max_side: int = 768
    cotracker_conf_threshold: float = 0.85
    cotracker_iters: int = 6

    init_max_points: int = 80
    init_anchor_count: int = 3
    init_anchor_min_pixel_distance: float = 120.0
    init_anchor_min_axis_radius_m: float = 0.0
    init_anchor_min_axis_radius_quantile: float = 0.0
    init_anchor_axis_distance_weight: float = 0.8
    init_anchor_feature_weight: float = 1.0
    init_anchor_top_weight: float = 0.6
    init_erode_px: int = 8
    init_min_distance_px: float = 16.0
    init_quality_level: float = 0.01
    reseed_points: int = 3
    min_reseed_registered_points: int = 1
    reseed_quality_level: float = 0.006
    reseed_min_distance_px: float = 14.0
    min_valid_points: int = 3
    low_valid_points: int = 1
    max_reseed_events: int = 3
    min_reseed_interval: int = 2

    depth_sample_radius_px: int = 3
    projected_depth_radius_px: float = 8.0
    roi_dilation_px: int = 18
    query_plane_dist_thresh_m: float = 0.05
    plane_dist_thresh_m: float = 0.06
    reproj_prefilter_thresh_px: float = 90.0
    reproj_inlier_thresh_px: float = 45.0
    max_track_jump_px: float = 130.0
    depth_residual_thresh_m: float = 0.16
    depth_only_min_points: int = 120

    angle_method: str = "three_point_direct"  # loss_1d, three_point_direct
    angle_sign: float = 1.0
    three_point_count: int = 3
    three_point_min_used_points: int = 2
    three_point_candidate_count: int = 8
    three_point_min_axis_radius_m: float = 0.035
    three_point_reproj_prefilter_thresh_px: float = 260.0
    three_point_max_mad_deg: float = 10.0
    three_point_max_residual_deg: float = 24.0
    three_point_incremental: bool = True
    three_point_max_depth_jump_m: float = 0.08
    three_point_depth_ratio_max: float = 1.25
    three_point_monotonic_slack_deg: float = 8.0
    three_point_max_delta_deg: float = 45.0
    three_point_allow_depth_only: bool = False
    angle_min_deg: float = -20.0
    angle_max_deg: float = 140.0
    angle_search_radius_deg: float = 14.0
    reappear_search_radius_deg: float = 32.0
    coarse_steps: int = 73
    max_angle_delta_deg: float = 45.0
    lambda_reproj: float = 1.0
    lambda_depth: float = 1.0
    lambda_plane: float = 0.6
    lambda_temporal: float = 0.02
    lambda_acc: float = 0.01
    reproj_scale_px: float = 18.0
    depth_scale_m: float = 0.045
    plane_scale_m: float = 0.035
    robust_delta: float = 1.5
    mad_sigma: float = 3.5


@dataclass
class DepthSample:
    point_frame: np.ndarray
    uv: np.ndarray
    depth_m: float
    count: int
    mode: str


@dataclass
class FrameEstimate:
    frame: int
    local_index: int
    theta_rad: float
    theta_pred_rad: float
    status: str
    valid_points: int
    confidence: float
    loss: float
    valid_point_ids: list[int]
    should_reseed: bool
    diagnostics: dict[str, Any]


@dataclass
class SequenceEstimate:
    angles_rad: np.ndarray
    frames: list[FrameEstimate]
    observations: list[dict[str, Any]]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_project_path(path_text: str | Path, base_dir: Path | None = None) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    candidates: list[Path] = []
    if base_dir is not None:
        candidates.append((base_dir / path).resolve())
    candidates.append((PROJECT_ROOT / path).resolve())
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        geoms = [geom for geom in mesh.geometry.values() if isinstance(geom, trimesh.Trimesh)]
        if not geoms:
            raise ValueError(f"No mesh geometry found in {path}")
        mesh = trimesh.util.concatenate(geoms)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type from {path}: {type(mesh)!r}")
    return mesh


def load_screen_mask(alignment_dir: Path, result: dict[str, Any]) -> np.ndarray:
    generated = result.get("part_masks", {}).get("generated", {})
    for key in ("screen_projection", "screen"):
        entry = generated.get(key, {})
        if entry.get("mask_npy"):
            path = resolve_project_path(entry["mask_npy"], alignment_dir)
            if path.exists():
                return np.load(path).astype(bool)
    for name in ("target_laptop_frame_0_screen_projection.mask.npy", "target_laptop_frame_0_screen.mask.npy"):
        path = alignment_dir / "part_masks" / name
        if path.exists():
            return np.load(path).astype(bool)
    raise FileNotFoundError(f"Could not resolve screen mask under {alignment_dir}")


def load_rgb_frames(export_root: Path, frames: list[int], rgb_dir_name: str) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for frame in frames:
        path = export_root / rgb_dir_name / f"{frame_name(frame)}.png"
        if not path.exists():
            raise FileNotFoundError(f"Missing RGB frame: {path}")
        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read RGB frame: {path}")
        out.append(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    return out


def load_rgb_frames_from_dir(rgb_dir: Path, frames: list[int]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for frame in frames:
        path = rgb_dir / f"{frame_name(frame)}.png"
        if not path.exists():
            raise FileNotFoundError(f"Missing tracker RGB frame: {path}")
        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read tracker RGB frame: {path}")
        out.append(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    return out


def map_eval_to_tracker_indices(
    eval_frames: list[int],
    tracker_frames: list[int],
    tracker_fps: float,
    eval_fps: float,
) -> np.ndarray:
    if len(eval_frames) > len(tracker_frames):
        raise ValueError("Tracker RGB sequence must cover at least the requested eval span")
    if tracker_fps > 0.0 and eval_fps > 0.0:
        eval0 = float(eval_frames[0])
        tracker0 = float(tracker_frames[0])
        mapped = []
        for frame in eval_frames:
            rel_seconds = (float(frame) - eval0) / float(eval_fps)
            mapped_frame = tracker0 + rel_seconds * float(tracker_fps)
            tracker_idx = int(round(mapped_frame - tracker0))
            mapped.append(int(np.clip(tracker_idx, 0, len(tracker_frames) - 1)))
        return np.asarray(mapped, dtype=np.int64)
    if len(eval_frames) == 1:
        return np.zeros(1, dtype=np.int64)
    mapped_float = np.linspace(0, len(tracker_frames) - 1, len(eval_frames))
    return np.rint(mapped_float).astype(np.int64)


def load_depth_meters(path: Path, depth_unit: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing depth frame: {path}")
    depth = np.load(path).astype(np.float32)
    unit = depth_unit.lower()
    if unit in {"m", "meter", "meters"}:
        return depth
    if unit in {"mm", "millimeter", "millimeters"}:
        return depth / 1000.0
    if unit == "auto":
        finite = depth[np.isfinite(depth) & (depth > 0)]
        if finite.size and float(np.nanmedian(finite)) > 20.0:
            return depth / 1000.0
        return depth
    raise ValueError(f"Unsupported depth unit: {depth_unit}")


def load_frame_rows(export_root: Path) -> dict[int, dict[str, str]]:
    with (export_root / "frames.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {int(row["index"]): row for row in rows}


def check_sequence_inputs(config: ScreenHingeTrackingConfig, frames: list[int], meta: dict[str, Any]) -> list[str]:
    rows = load_frame_rows(config.export_root)
    warnings: list[str] = []
    width = int(meta["rgb_width_per_eye"])
    height = int(meta["rgb_height_per_eye"])
    for frame in frames:
        row = rows.get(frame)
        if row is None:
            raise KeyError(f"Frame {frame} missing from {config.export_root / 'frames.csv'}")
        rgb_path = config.export_root / config.rgb_dir_name / f"{frame_name(frame)}.png"
        depth_path = config.export_root / config.depth_dir_name / f"{frame_name(frame)}.meters.npy"
        if not rgb_path.exists():
            raise FileNotFoundError(f"Missing RGB frame: {rgb_path}")
        if not depth_path.exists():
            raise FileNotFoundError(f"Missing depth frame: {depth_path}")
        try:
            dt = abs(float(row["rgb_timestamp_s"]) - float(row["depth_timestamp_s"]))
            if dt > config.timestamp_tolerance_s:
                warnings.append(f"frame {frame}: RGB/depth timestamp delta {dt:.4f}s exceeds tolerance")
        except KeyError:
            warnings.append("frames.csv lacks rgb/depth timestamp columns; sync check skipped")
        image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (height, width):
            got = None if image is None else image.shape[:2]
            raise ValueError(f"RGB frame {rgb_path} shape {got} does not match manifest {(height, width)}")
    return warnings


def erode_mask(mask: np.ndarray, erode_px: int) -> np.ndarray:
    mask_u8 = (mask.astype(bool).astype(np.uint8) * 255)
    if erode_px <= 0:
        return mask_u8.astype(bool)
    size = 2 * int(erode_px) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.erode(mask_u8, kernel, iterations=1).astype(bool)


def select_good_features(
    rgb: np.ndarray,
    mask: np.ndarray,
    max_points: int,
    min_distance_px: float,
    quality_level: float,
    erode_px: int = 0,
    existing_xy: np.ndarray | None = None,
) -> np.ndarray:
    """Select sparse textured query points with Shi-Tomasi inside a binary mask."""
    if mask.ndim != 2 or not mask.any():
        return np.zeros((0, 2), dtype=np.float32)
    usable_mask = erode_mask(mask.astype(bool), erode_px)
    if usable_mask.sum() < max(8, max_points // 4):
        usable_mask = mask.astype(bool)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask_u8 = (usable_mask.astype(np.uint8) * 255)
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max(max_points * 4, max_points),
        qualityLevel=float(quality_level),
        minDistance=float(min_distance_px),
        mask=mask_u8,
        blockSize=5,
        useHarrisDetector=False,
    )
    if corners is None:
        return np.zeros((0, 2), dtype=np.float32)
    points = corners.reshape(-1, 2).astype(np.float32)
    selected: list[np.ndarray] = []
    existing = np.asarray(existing_xy, dtype=np.float32) if existing_xy is not None and len(existing_xy) else None
    for point in points:
        if existing is not None and np.min(np.linalg.norm(existing - point[None, :], axis=1)) < min_distance_px:
            continue
        if selected:
            selected_arr = np.asarray(selected, dtype=np.float32)
            if np.min(np.linalg.norm(selected_arr - point[None, :], axis=1)) < min_distance_px:
                continue
        selected.append(point)
        if len(selected) >= max_points:
            break
    if not selected:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(selected, dtype=np.float32)


def feature_strengths(rgb: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    if len(points_xy) == 0:
        return np.zeros(0, dtype=np.float64)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    response = cv2.cornerMinEigenVal(gray, blockSize=5, ksize=3)
    h, w = response.shape
    scores = []
    for x, y in points_xy:
        xi = int(np.clip(round(float(x)), 0, w - 1))
        yi = int(np.clip(round(float(y)), 0, h - 1))
        scores.append(float(response[yi, xi]))
    return np.asarray(scores, dtype=np.float64)


def point_axis_distances(points: np.ndarray, origin: np.ndarray, axis: np.ndarray) -> np.ndarray:
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    vec = points - origin[None, :]
    radial = vec - (vec @ axis)[:, None] * axis[None, :]
    return np.linalg.norm(radial, axis=1)


def select_anchor_subset(
    points_xy: np.ndarray,
    points_ref: np.ndarray,
    feature_scores: np.ndarray,
    joint: dict[str, Any],
    count: int,
    min_pixel_distance: float,
    min_axis_radius_m: float,
    min_axis_radius_quantile: float,
    feature_weight: float,
    axis_distance_weight: float,
    top_weight: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    if count <= 0 or len(points_xy) <= count:
        return np.arange(len(points_xy), dtype=np.int64), {"strategy": "all_candidates", "candidate_count": int(len(points_xy))}
    origin = np.asarray(joint["origin_xyz"], dtype=np.float64)
    axis = np.asarray(joint["axis_xyz"], dtype=np.float64)
    axis /= np.linalg.norm(axis) + 1e-12
    axis_dist = point_axis_distances(points_ref, origin, axis)
    finite = np.isfinite(points_ref).all(axis=1) & np.isfinite(feature_scores) & np.isfinite(axis_dist)
    if finite.any():
        q = float(np.clip(min_axis_radius_quantile, 0.0, 0.95))
        quantile_radius = float(np.quantile(axis_dist[finite], q)) if q > 0.0 else 0.0
        radius_floor = max(float(min_axis_radius_m), quantile_radius)
        radius_keep = axis_dist >= radius_floor
        if int((finite & radius_keep).sum()) >= count:
            finite &= radius_keep
    else:
        radius_floor = float(min_axis_radius_m)
    ids = np.flatnonzero(finite)
    if ids.size == 0:
        return np.arange(min(count, len(points_xy)), dtype=np.int64), {"strategy": "fallback_first", "candidate_count": int(len(points_xy))}
    fs = feature_scores[ids]
    ds = axis_dist[ids]
    ys = points_xy[ids, 1]
    fs_norm = (fs - fs.min()) / (max(float(fs.max() - fs.min()), 1e-12))
    ds_norm = (ds - ds.min()) / (max(float(ds.max() - ds.min()), 1e-12))
    top_norm = 1.0 - (ys - ys.min()) / (max(float(ys.max() - ys.min()), 1e-12))
    combined = float(feature_weight) * fs_norm + float(axis_distance_weight) * ds_norm + float(top_weight) * top_norm
    order = ids[np.argsort(-combined)]
    selected: list[int] = []
    for idx in order:
        if selected:
            dist_px = np.linalg.norm(points_xy[np.asarray(selected)] - points_xy[idx][None, :], axis=1)
            if float(np.min(dist_px)) < min_pixel_distance:
                continue
        selected.append(int(idx))
        if len(selected) >= count:
            break
    if len(selected) < count:
        for idx in order:
            if int(idx) not in selected:
                selected.append(int(idx))
            if len(selected) >= count:
                break
    selected_arr = np.asarray(selected[:count], dtype=np.int64)
    diagnostics = {
        "strategy": "feature_plus_far_from_hinge_axis",
        "candidate_count": int(len(points_xy)),
        "selected_indices": selected_arr.tolist(),
        "feature_scores": feature_scores[selected_arr].tolist(),
        "axis_distances_m": axis_dist[selected_arr].tolist(),
        "top_weight": float(top_weight),
        "axis_radius_floor_m": float(radius_floor),
        "selected_xy": points_xy[selected_arr].tolist(),
    }
    return selected_arr, diagnostics


def signed_axis_angle(reference_point: np.ndarray, observed_point: np.ndarray, origin: np.ndarray, axis: np.ndarray) -> float:
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    v0 = reference_point - origin
    v1 = observed_point - origin
    v0 = v0 - axis * float(v0 @ axis)
    v1 = v1 - axis * float(v1 @ axis)
    n0 = np.linalg.norm(v0)
    n1 = np.linalg.norm(v1)
    if n0 < 1e-8 or n1 < 1e-8:
        return float("nan")
    denom = n0 * n1 + 1e-12
    sin_t = float((np.cross(v0, v1) @ axis) / denom)
    cos_t = float((v0 @ v1) / denom)
    return float(np.arctan2(sin_t, np.clip(cos_t, -1.0, 1.0)))


def tracker_resize(frames: list[np.ndarray], max_side: int) -> tuple[np.ndarray, float]:
    h, w = frames[0].shape[:2]
    scale = min(1.0, float(max_side) / float(max(h, w))) if max_side > 0 else 1.0
    if scale < 1.0:
        new_w = max(8, int(round(w * scale)))
        new_h = max(8, int(round(h * scale)))
        resized = [cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA) for frame in frames]
    else:
        resized = frames
    return np.stack(resized, axis=0), scale


def run_cotracker_offline(
    frames_rgb: list[np.ndarray],
    queries_xy: np.ndarray,
    query_frames: np.ndarray,
    cotracker_root: Path,
    checkpoint: Path,
    device: str,
    tracker_max_side: int,
    confidence_threshold: float,
    iters: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if len(queries_xy) == 0:
        raise ValueError("No query points were provided to CoTracker")
    if str(cotracker_root) not in sys.path:
        sys.path.insert(0, str(cotracker_root))
    import torch
    import torch.nn.functional as F
    from cotracker.models.core.model_utils import get_points_on_a_grid  # noqa: WPS433
    from cotracker.predictor import CoTrackerPredictor  # noqa: WPS433

    video_np, scale = tracker_resize(frames_rgb, tracker_max_side)
    video = torch.from_numpy(video_np).permute(0, 3, 1, 2)[None].float().to(device)
    query_scaled = queries_xy.astype(np.float32) * float(scale)
    query_txy = np.column_stack([query_frames.astype(np.float32), query_scaled]).astype(np.float32)
    queries = torch.from_numpy(query_txy)[None].to(device)

    # CoTracker3 offline predictor accepts all sparse query points at once.  It
    # does not provide a stable public API for adding queries mid-forward-pass,
    # so dynamic reseeding is implemented by appending query frames and rerunning
    # this sparse offline pass.  This avoids dense tracking and stays below 16GB.
    model = CoTrackerPredictor(checkpoint=str(checkpoint), offline=True, window_len=max(60, len(frames_rgb))).to(device)
    model.eval()
    bsz, timesteps, channels, height, width = video.shape
    with torch.no_grad():
        video_interp = video.reshape(bsz * timesteps, channels, height, width)
        video_interp = F.interpolate(video_interp, tuple(model.interp_shape), mode="bilinear", align_corners=True)
        video_interp = video_interp.reshape(bsz, timesteps, 3, model.interp_shape[0], model.interp_shape[1])
        queries_model = queries.clone()
        queries_model[:, :, 1:] *= queries_model.new_tensor(
            [
                (model.interp_shape[1] - 1) / (width - 1),
                (model.interp_shape[0] - 1) / (height - 1),
            ]
        )
        query_count = queries_model.shape[1]
        support_grid = get_points_on_a_grid(model.support_grid_size, model.interp_shape, device=video.device)
        support_grid = torch.cat([torch.zeros_like(support_grid[:, :, :1]), support_grid], dim=2)
        support_grid = support_grid.repeat(bsz, 1, 1)
        model_queries = torch.cat([queries_model, support_grid], dim=1)
        tracks, raw_visibility, *_ = model.model.forward(video=video_interp, queries=model_queries, iters=int(iters))
        tracks = tracks[:, :, :query_count]
        raw_visibility = raw_visibility[:, :, :query_count]
        query_t = queries_model[0, :, 0].round().to(torch.int64).clamp(0, timesteps - 1)
        arange = torch.arange(0, len(query_t), device=video.device)
        tracks[0, query_t, arange] = queries_model[0, :, 1:]
        raw_visibility[0, query_t, arange] = 1.0
        tracks *= tracks.new_tensor(
            [
                (width - 1) / (model.interp_shape[1] - 1),
                (height - 1) / (model.interp_shape[0] - 1),
            ]
        )
    tracks_np = tracks[0].detach().cpu().numpy().astype(np.float32)
    confidence_np = raw_visibility[0].detach().cpu().float().numpy().astype(np.float32)
    if scale != 0:
        tracks_np /= float(scale)
    visibility_np = confidence_np >= float(confidence_threshold)
    local_frame_ids = np.arange(tracks_np.shape[0], dtype=np.float32)[:, None]
    before_start = local_frame_ids < query_frames.astype(np.float32)[None, :]
    visibility_np[before_start] = False
    confidence_np[before_start] = 0.0
    info = {
        "backend": "cotracker3_offline_sparse",
        "cotracker_root": str(cotracker_root),
        "checkpoint": str(checkpoint),
        "device": device,
        "input_shape_b_t_c_h_w": [1, len(frames_rgb), 3, int(frames_rgb[0].shape[0]), int(frames_rgb[0].shape[1])],
        "tracker_shape_thw": list(video_np.shape[:3]),
        "tracker_scale": float(scale),
        "query_format": "[t, x, y]",
        "query_count": int(len(queries_xy)),
        "confidence_threshold": float(confidence_threshold),
    }
    return tracks_np, visibility_np, confidence_np, info


class DepthFrameSampler:
    def __init__(
        self,
        meta: dict[str, Any],
        depth_m: np.ndarray,
        convention: str,
        rgb_shape_hw: tuple[int, int],
        mode: str,
        depth_min_m: float,
        depth_max_m: float,
    ) -> None:
        self.meta = meta
        self.depth_m = depth_m
        self.rgb_shape_hw = rgb_shape_hw
        self.depth_min_m = float(depth_min_m)
        self.depth_max_m = float(depth_max_m)
        self.mode = mode
        if self.mode == "auto":
            self.mode = "aligned_rgb" if depth_m.shape == rgb_shape_hw else "projected"
        if self.mode not in {"projected", "aligned_rgb"}:
            raise ValueError(f"Unsupported depth sample mode: {mode}")
        self.points_right: np.ndarray | None = None
        self.uv: np.ndarray | None = None
        self.tree: cKDTree | None = None
        if self.mode == "projected":
            points_right, u, v, inside = depth_points_in_right_camera(
                meta,
                depth_m,
                convention,
                depth_min_m=depth_min_m,
                depth_max_m=depth_max_m,
            )
            idx = np.flatnonzero(inside)
            self.points_right = points_right[idx].astype(np.float64)
            self.uv = np.column_stack([u[idx], v[idx]]).astype(np.float64)
            self.tree = cKDTree(self.uv) if len(self.uv) else None

    def sample(self, xy: np.ndarray, radius_px: float) -> DepthSample | None:
        x = float(xy[0])
        y = float(xy[1])
        h, w = self.rgb_shape_hw
        if not (0.0 <= x < w and 0.0 <= y < h):
            return None
        if self.mode == "aligned_rgb":
            rr = max(1, int(round(radius_px)))
            cx = int(round(x))
            cy = int(round(y))
            x0, x1 = max(0, cx - rr), min(w - 1, cx + rr)
            y0, y1 = max(0, cy - rr), min(h - 1, cy + rr)
            patch = self.depth_m[y0 : y1 + 1, x0 : x1 + 1]
            valid = np.isfinite(patch) & (patch > self.depth_min_m) & (patch < self.depth_max_m)
            if not valid.any():
                return None
            z = float(np.median(patch[valid]))
            kr = self.meta["rgb_intrinsics_right"]
            point = np.asarray([(x - kr["cx"]) * z / kr["fx"], (y - kr["cy"]) * z / kr["fy"], z], dtype=np.float64)
            return DepthSample(point_frame=point, uv=np.asarray([x, y], dtype=np.float64), depth_m=z, count=int(valid.sum()), mode=self.mode)
        if self.tree is None or self.points_right is None or self.uv is None:
            return None
        ids = self.tree.query_ball_point(np.asarray([x, y], dtype=np.float64), r=float(radius_px))
        if not ids:
            dist, idx = self.tree.query(np.asarray([x, y], dtype=np.float64), k=1)
            if not np.isfinite(dist) or dist > radius_px:
                return None
            ids = [int(idx)]
        pts = self.points_right[np.asarray(ids, dtype=np.int64)]
        uv = self.uv[np.asarray(ids, dtype=np.int64)]
        point = np.median(pts, axis=0)
        return DepthSample(point_frame=point, uv=np.median(uv, axis=0), depth_m=float(point[2]), count=len(ids), mode=self.mode)

    def points_in_mask(self, mask: np.ndarray, max_points: int = 2500) -> np.ndarray:
        if self.mode == "projected" and self.points_right is not None and self.uv is not None:
            u = np.clip(np.round(self.uv[:, 0]).astype(np.int64), 0, mask.shape[1] - 1)
            v = np.clip(np.round(self.uv[:, 1]).astype(np.int64), 0, mask.shape[0] - 1)
            keep = mask[v, u]
            pts = self.points_right[keep]
        else:
            yy, xx = np.nonzero(mask)
            if len(xx) == 0:
                return np.zeros((0, 3), dtype=np.float64)
            z = self.depth_m[yy, xx]
            valid = np.isfinite(z) & (z > self.depth_min_m) & (z < self.depth_max_m)
            xx = xx[valid].astype(np.float64)
            yy = yy[valid].astype(np.float64)
            z = z[valid].astype(np.float64)
            kr = self.meta["rgb_intrinsics_right"]
            pts = np.column_stack([(xx - kr["cx"]) * z / kr["fx"], (yy - kr["cy"]) * z / kr["fy"], z])
        if len(pts) > max_points:
            rng = np.random.default_rng(17)
            pts = pts[rng.choice(len(pts), size=max_points, replace=False)]
        return pts.astype(np.float64)


def build_depth_samplers(
    config: ScreenHingeTrackingConfig,
    meta: dict[str, Any],
    frames: list[int],
    rgb_shape_hw: tuple[int, int],
    convention: str,
) -> list[DepthFrameSampler]:
    samplers: list[DepthFrameSampler] = []
    for frame in frames:
        path = config.export_root / config.depth_dir_name / f"{frame_name(frame)}.meters.npy"
        depth_m = load_depth_meters(path, config.depth_unit)
        samplers.append(
            DepthFrameSampler(
                meta,
                depth_m,
                convention,
                rgb_shape_hw,
                config.depth_sample_mode,
                config.depth_min_m,
                config.depth_max_m,
            )
        )
    return samplers


def screen_plane_from_mesh(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    center = vertices.mean(axis=0)
    _, _, vt = np.linalg.svd(vertices - center, full_matrices=False)
    normal = vt[-1]
    normal /= np.linalg.norm(normal) + 1e-12
    return center, normal


def rotate_vectors_about_axis(vectors: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    cos_t = float(np.cos(angle_rad))
    sin_t = float(np.sin(angle_rad))
    return vectors * cos_t + np.cross(axis, vectors) * sin_t + axis * (vectors @ axis)[:, None] * (1.0 - cos_t)


def project_screen_roi(
    meta: dict[str, Any],
    export_root: Path,
    frames: list[int],
    local_idx: int,
    screen_mesh0: trimesh.Trimesh,
    joint: dict[str, Any],
    theta_rad: float,
    image_shape_hw: tuple[int, int],
    dilation_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = image_shape_hw
    origin = np.asarray(joint["origin_xyz"], dtype=np.float64)
    axis = np.asarray(joint["axis_xyz"], dtype=np.float64)
    axis /= np.linalg.norm(axis) + 1e-12
    rotated_vertices = rotate_points_about_axis(np.asarray(screen_mesh0.vertices, dtype=np.float64), origin, axis, theta_rad)
    align_row = frame_row(export_root, frames[0])
    view_row = frame_row(export_root, frames[local_idx])
    t_frame_from_align = camera_to_camera_matrix(meta, align_row, view_row, camera="right")
    vertices_frame = transform_points(rotated_vertices, t_frame_from_align)
    u, v, z = project_right_camera_points(meta, vertices_frame)
    inside = (z > 1e-6) & np.isfinite(u) & np.isfinite(v)
    if int(inside.sum()) < 3:
        return np.zeros((h, w), dtype=bool), np.zeros((0, 2), dtype=np.float32)
    points = np.column_stack([u[inside], v[inside]]).astype(np.float32)
    points[:, 0] = np.clip(points[:, 0], 0, w - 1)
    points[:, 1] = np.clip(points[:, 1], 0, h - 1)
    hull = cv2.convexHull(points.reshape(-1, 1, 2)).reshape(-1, 2).astype(np.float32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)
    if dilation_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(dilation_px) + 1, 2 * int(dilation_px) + 1))
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask.astype(bool), hull


def huber_loss(values: np.ndarray, delta: float) -> np.ndarray:
    values = np.abs(np.asarray(values, dtype=np.float64))
    return np.where(values <= delta, 0.5 * values * values, delta * (values - 0.5 * delta))


def mad_inlier_mask(values: np.ndarray, sigma: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if finite.sum() < 4:
        return finite
    median = float(np.median(values[finite]))
    mad = float(np.median(np.abs(values[finite] - median)))
    scale = max(1.4826 * mad, 1e-6)
    out = finite & (np.abs(values - median) <= sigma * scale)
    if out.sum() < max(3, finite.sum() // 4):
        return finite
    return out


def register_query_points(
    xy: np.ndarray,
    query_frame_idx: int,
    theta_rad: float,
    samplers: list[DepthFrameSampler],
    meta: dict[str, Any],
    export_root: Path,
    frames: list[int],
    joint: dict[str, Any],
    plane_center_ref: np.ndarray,
    plane_normal_ref: np.ndarray,
    config: ScreenHingeTrackingConfig,
) -> tuple[np.ndarray, np.ndarray]:
    origin = np.asarray(joint["origin_xyz"], dtype=np.float64)
    axis = np.asarray(joint["axis_xyz"], dtype=np.float64)
    axis /= np.linalg.norm(axis) + 1e-12
    align_row = frame_row(export_root, frames[0])
    query_row = frame_row(export_root, frames[query_frame_idx])
    t_align_from_query = camera_to_camera_matrix(meta, query_row, align_row, camera="right")
    local_points = np.full((len(xy), 3), np.nan, dtype=np.float64)
    valid = np.zeros(len(xy), dtype=bool)
    for idx, point_xy in enumerate(xy):
        sample = samplers[query_frame_idx].sample(
            point_xy,
            config.projected_depth_radius_px if samplers[query_frame_idx].mode == "projected" else config.depth_sample_radius_px,
        )
        if sample is None:
            continue
        point_align = transform_points(sample.point_frame[None, :], t_align_from_query)[0]
        local = rotate_points_about_axis(point_align[None, :], origin, axis, -theta_rad)[0]
        plane_dist = abs(float((local - plane_center_ref) @ plane_normal_ref))
        if plane_dist > config.query_plane_dist_thresh_m:
            continue
        local_points[idx] = local
        valid[idx] = True
    return local_points, valid


def collect_frame_observations(
    local_idx: int,
    tracks_xy: np.ndarray,
    visibility: np.ndarray,
    confidence: np.ndarray,
    query_frames: np.ndarray,
    local_points: np.ndarray,
    samplers: list[DepthFrameSampler],
    meta: dict[str, Any],
    export_root: Path,
    frames: list[int],
    joint: dict[str, Any],
    screen_mesh0: trimesh.Trimesh,
    theta_pred: float,
    config: ScreenHingeTrackingConfig,
    plane_center_ref: np.ndarray,
    plane_normal_ref: np.ndarray,
    prev_point_align: np.ndarray | None = None,
    prev_point_depth: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    h = int(meta["rgb_height_per_eye"])
    w = int(meta["rgb_width_per_eye"])
    origin = np.asarray(joint["origin_xyz"], dtype=np.float64)
    axis = np.asarray(joint["axis_xyz"], dtype=np.float64)
    axis /= np.linalg.norm(axis) + 1e-12
    use_three_point_direct = config.angle_method == "three_point_direct"
    align_row = frame_row(export_root, frames[0])
    view_row = frame_row(export_root, frames[local_idx])
    t_frame_from_align = camera_to_camera_matrix(meta, align_row, view_row, camera="right")
    t_align_from_frame = camera_to_camera_matrix(meta, view_row, align_row, camera="right")
    roi_mask, _ = project_screen_roi(meta, export_root, frames, local_idx, screen_mesh0, joint, theta_pred, (h, w), config.roi_dilation_px)

    ids: list[int] = []
    xy_obs: list[np.ndarray] = []
    point_frame_obs: list[np.ndarray] = []
    point_align_obs: list[np.ndarray] = []
    conf_obs: list[float] = []
    per_point_theta_obs: list[float] = []
    per_point_delta_obs: list[float] = []
    axis_radius_obs: list[float] = []
    reproj_pred_obs: list[float] = []
    plane_dist_obs: list[float] = []
    depth_residual_obs: list[float] = []
    depth_jump_obs: list[float] = []
    rows: list[dict[str, Any]] = []
    pred_align = rotate_points_about_axis(local_points, origin, axis, theta_pred)
    pred_frame = transform_points(pred_align, t_frame_from_align)
    pred_u, pred_v, pred_z = project_right_camera_points(meta, pred_frame)

    for pid in range(tracks_xy.shape[1]):
        xy = tracks_xy[local_idx, pid]
        visible = bool(visibility[local_idx, pid]) and query_frames[pid] <= local_idx and np.isfinite(local_points[pid]).all()
        in_bounds = bool(0 <= xy[0] < w and 0 <= xy[1] < h)
        jump_ok = True
        if local_idx > int(query_frames[pid]) and local_idx > 0 and visibility[local_idx - 1, pid]:
            jump = float(np.linalg.norm(tracks_xy[local_idx, pid] - tracks_xy[local_idx - 1, pid]))
            jump_ok = jump <= config.max_track_jump_px
        else:
            jump = 0.0
        sample = None
        if visible and in_bounds and jump_ok:
            sample = samplers[local_idx].sample(
                xy,
                config.projected_depth_radius_px if samplers[local_idx].mode == "projected" else config.depth_sample_radius_px,
            )
        depth_ok = sample is not None
        roi_ok = bool(roi_mask[int(np.clip(round(xy[1]), 0, h - 1)), int(np.clip(round(xy[0]), 0, w - 1))]) if in_bounds else False
        reproj_pred = float("inf")
        plane_dist = float("inf")
        depth_residual = float("inf")
        per_point_theta = float("nan")
        per_point_delta = float("nan")
        axis_radius = float("nan")
        depth_jump_m = float("nan")
        depth_ratio = float("nan")
        depth_jump_ok = True
        used = False
        if visible and in_bounds and jump_ok and depth_ok and (roi_ok or use_three_point_direct):
            point_align = transform_points(sample.point_frame[None, :], t_align_from_frame)[0]
            per_point_theta = float(config.angle_sign) * signed_axis_angle(local_points[pid], point_align, origin, axis)
            if (
                prev_point_align is not None
                and pid < len(prev_point_align)
                and np.isfinite(prev_point_align[pid]).all()
            ):
                per_point_delta = float(config.angle_sign) * signed_axis_angle(prev_point_align[pid], point_align, origin, axis)
            if (
                prev_point_depth is not None
                and pid < len(prev_point_depth)
                and np.isfinite(prev_point_depth[pid])
                and prev_point_depth[pid] > 1e-6
            ):
                depth_jump_m = abs(float(sample.depth_m - prev_point_depth[pid]))
                depth_ratio = max(float(sample.depth_m), float(prev_point_depth[pid])) / max(
                    min(float(sample.depth_m), float(prev_point_depth[pid])),
                    1e-6,
                )
                depth_jump_ok = (
                    depth_jump_m <= config.three_point_max_depth_jump_m
                    and depth_ratio <= config.three_point_depth_ratio_max
                )
            theta_for_filter = per_point_theta if use_three_point_direct and np.isfinite(per_point_theta) else theta_pred
            pred_align_point = rotate_points_about_axis(local_points[pid][None, :], origin, axis, theta_for_filter)[0]
            pred_frame_point = transform_points(pred_align_point[None, :], t_frame_from_align)[0]
            pu, pv, pz = project_right_camera_points(meta, pred_frame_point[None, :])
            axis_radius = float(point_axis_distances(local_points[pid][None, :], origin, axis)[0])
            if float(pz[0]) > 1e-6 and np.isfinite(pu[0]) and np.isfinite(pv[0]):
                reproj_pred = float(np.linalg.norm(np.asarray([pu[0], pv[0]]) - xy))
                center_theta = rotate_points_about_axis(plane_center_ref[None, :], origin, axis, theta_for_filter)[0]
                normal_theta = rotate_vectors_about_axis(plane_normal_ref[None, :], axis, theta_for_filter)[0]
                normal_theta /= np.linalg.norm(normal_theta) + 1e-12
                plane_dist = abs(float((point_align - center_theta) @ normal_theta))
                depth_residual = abs(float(pred_frame_point[2] - sample.point_frame[2]))
                reproj_limit = (
                    config.three_point_reproj_prefilter_thresh_px
                    if use_three_point_direct
                    else config.reproj_prefilter_thresh_px
                )
                used = (
                    np.isfinite(per_point_theta)
                    and depth_jump_ok
                    and reproj_pred <= reproj_limit
                    and plane_dist <= config.plane_dist_thresh_m
                    and depth_residual <= config.depth_residual_thresh_m
                )
        rows.append(
            {
                "frame": int(frames[local_idx]),
                "local_frame": int(local_idx),
                "point_id": int(pid),
                "x": float(xy[0]),
                "y": float(xy[1]),
                "visibility": bool(visibility[local_idx, pid]),
                "confidence": float(confidence[local_idx, pid]),
                "in_bounds": in_bounds,
                "depth_ok": depth_ok,
                "roi_ok": roi_ok,
                "jump_px": jump,
                "reproj_pred_px": reproj_pred,
                "plane_dist_m": plane_dist,
                "depth_residual_m": depth_residual,
                "axis_radius_m": axis_radius,
                "depth_jump_m": depth_jump_m,
                "depth_ratio": depth_ratio,
                "depth_jump_ok": depth_jump_ok,
                "per_point_theta_rad": per_point_theta,
                "per_point_theta_deg": float(np.rad2deg(per_point_theta)) if np.isfinite(per_point_theta) else float("nan"),
                "per_point_delta_rad": per_point_delta,
                "per_point_delta_deg": float(np.rad2deg(per_point_delta)) if np.isfinite(per_point_delta) else float("nan"),
                "valid_for_angle": used,
            }
        )
        if used and sample is not None:
            ids.append(pid)
            xy_obs.append(xy.astype(np.float64))
            point_frame_obs.append(sample.point_frame.astype(np.float64))
            point_align_obs.append(transform_points(sample.point_frame[None, :], t_align_from_frame)[0].astype(np.float64))
            conf_obs.append(float(confidence[local_idx, pid]))
            per_point_theta_obs.append(float(per_point_theta))
            per_point_delta_obs.append(float(per_point_delta))
            axis_radius_obs.append(float(axis_radius))
            reproj_pred_obs.append(float(reproj_pred))
            plane_dist_obs.append(float(plane_dist))
            depth_residual_obs.append(float(depth_residual))
            depth_jump_obs.append(float(depth_jump_m))
    obs = {
        "ids": np.asarray(ids, dtype=np.int64),
        "xy": np.asarray(xy_obs, dtype=np.float64).reshape(-1, 2),
        "point_frame": np.asarray(point_frame_obs, dtype=np.float64).reshape(-1, 3),
        "point_align": np.asarray(point_align_obs, dtype=np.float64).reshape(-1, 3),
        "confidence": np.asarray(conf_obs, dtype=np.float64),
        "per_point_theta": np.asarray(per_point_theta_obs, dtype=np.float64),
        "per_point_delta": np.asarray(per_point_delta_obs, dtype=np.float64),
        "axis_radius": np.asarray(axis_radius_obs, dtype=np.float64),
        "reproj_pred": np.asarray(reproj_pred_obs, dtype=np.float64),
        "plane_dist": np.asarray(plane_dist_obs, dtype=np.float64),
        "depth_residual": np.asarray(depth_residual_obs, dtype=np.float64),
        "depth_jump": np.asarray(depth_jump_obs, dtype=np.float64),
    }
    return obs, rows


def score_track_theta(
    theta_rad: float,
    obs: dict[str, np.ndarray],
    local_points: np.ndarray,
    meta: dict[str, Any],
    t_frame_from_align: np.ndarray,
    origin: np.ndarray,
    axis: np.ndarray,
    plane_center_ref: np.ndarray,
    plane_normal_ref: np.ndarray,
    theta_pred: float,
    theta_prev: float,
    theta_prev2: float,
    config: ScreenHingeTrackingConfig,
) -> tuple[float, dict[str, Any]]:
    ids = obs["ids"]
    min_required = max(1, min(3, int(config.low_valid_points)))
    if len(ids) == 0:
        return float("inf"), {"used_points": 0}
    pred_align = rotate_points_about_axis(local_points[ids], origin, axis, theta_rad)
    pred_frame = transform_points(pred_align, t_frame_from_align)
    u, v, z = project_right_camera_points(meta, pred_frame)
    finite = (z > 1e-6) & np.isfinite(u) & np.isfinite(v)
    if finite.sum() < min_required:
        return float("inf"), {"used_points": int(finite.sum())}
    reproj = np.linalg.norm(np.column_stack([u, v]) - obs["xy"], axis=1)
    depth = np.abs(pred_frame[:, 2] - obs["point_frame"][:, 2])
    center_theta = rotate_points_about_axis(plane_center_ref[None, :], origin, axis, theta_rad)[0]
    normal_theta = rotate_vectors_about_axis(plane_normal_ref[None, :], axis, theta_rad)[0]
    normal_theta /= np.linalg.norm(normal_theta) + 1e-12
    plane = np.abs((obs["point_align"] - center_theta) @ normal_theta)
    normed = np.sqrt(
        (reproj / config.reproj_scale_px) ** 2
        + (depth / config.depth_scale_m) ** 2
        + (plane / config.plane_scale_m) ** 2
    )
    inliers = finite & (reproj <= config.reproj_inlier_thresh_px) & mad_inlier_mask(normed, config.mad_sigma)
    if inliers.sum() < min_required:
        inliers = finite & mad_inlier_mask(normed, config.mad_sigma)
    if inliers.sum() < min_required:
        return float("inf"), {"used_points": int(inliers.sum())}
    reproj_loss = float(np.mean(huber_loss(reproj[inliers] / config.reproj_scale_px, config.robust_delta)))
    depth_loss = float(np.mean(huber_loss(depth[inliers] / config.depth_scale_m, config.robust_delta)))
    plane_loss = float(np.mean(huber_loss(plane[inliers] / config.plane_scale_m, config.robust_delta)))
    temporal = float((theta_rad - theta_prev) ** 2)
    accel = float((theta_rad - 2.0 * theta_prev + theta_prev2) ** 2)
    loss = (
        config.lambda_reproj * reproj_loss
        + config.lambda_depth * depth_loss
        + config.lambda_plane * plane_loss
        + config.lambda_temporal * temporal
        + config.lambda_acc * accel
    )
    return loss, {
        "used_points": int(inliers.sum()),
        "candidate_points": int(len(ids)),
        "reproj_median_px": float(np.median(reproj[inliers])),
        "depth_median_m": float(np.median(depth[inliers])),
        "plane_median_m": float(np.median(plane[inliers])),
        "inlier_point_ids": ids[inliers].tolist(),
    }


def wrap_angle_near(theta_rad: np.ndarray, reference_rad: float) -> np.ndarray:
    """Map angles to the equivalent 2*pi branch closest to ``reference_rad``."""
    return reference_rad + np.arctan2(np.sin(theta_rad - reference_rad), np.cos(theta_rad - reference_rad))


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if len(values) == 0:
        return float("nan")
    order = np.argsort(values)
    values = values[order]
    weights = np.maximum(weights[order], 1e-8)
    cutoff = 0.5 * float(np.sum(weights))
    return float(values[np.searchsorted(np.cumsum(weights), cutoff, side="left")])


def estimate_three_point_direct_theta(
    obs: dict[str, np.ndarray],
    theta_pred: float,
    theta_prev: float,
    theta_prev2: float,
    search_radius_deg: float,
    config: ScreenHingeTrackingConfig,
) -> tuple[float, float, dict[str, Any]]:
    """Estimate one hinge angle from a few direct RGB-D point-angle votes.

    Each registered screen point already lives in the reference screen pose.  For
    a frame observation, its RGB-D back-projected 3D position gives an immediate
    signed rotation around the hinge axis.  We choose only a small set of
    high-confidence, far-from-axis, mutually consistent votes so the reported
    result is inspectable point by point instead of hidden inside a free 3D fit.
    """
    ids = obs["ids"]
    if len(ids) == 0:
        return float("nan"), float("inf"), {"method": "three_point_direct", "used_points": 0}
    raw_theta = np.asarray(obs["per_point_theta"], dtype=np.float64)
    raw_delta = np.asarray(obs.get("per_point_delta", np.full_like(raw_theta, np.nan)), dtype=np.float64)
    use_incremental = bool(config.three_point_incremental) and np.isfinite(raw_delta).any()
    if use_incremental:
        max_delta = math.radians(max(1.0, float(config.three_point_max_delta_deg)))
        delta_votes = wrap_angle_near(raw_delta, theta_pred - theta_prev)
        theta_votes = theta_prev + delta_votes
    else:
        delta_votes = np.full_like(raw_theta, np.nan)
        theta_votes = wrap_angle_near(raw_theta, theta_pred)
    angle_min = math.radians(config.angle_min_deg)
    angle_max = math.radians(config.angle_max_deg)
    search_radius = math.radians(max(float(search_radius_deg), 1.0))
    monotonic_slack = math.radians(max(0.0, float(config.three_point_monotonic_slack_deg)))
    finite = (
        np.isfinite(theta_votes)
        & np.isfinite(obs["confidence"])
        & np.isfinite(obs["axis_radius"])
        & np.isfinite(obs["reproj_pred"])
        & np.isfinite(obs["plane_dist"])
        & np.isfinite(obs["depth_residual"])
        & (obs["axis_radius"] >= config.three_point_min_axis_radius_m)
        & (theta_votes >= angle_min)
        & (theta_votes <= angle_max)
        & (np.abs(theta_votes - theta_pred) <= search_radius)
    )
    if use_incremental:
        finite &= np.isfinite(delta_votes) & (np.abs(delta_votes) <= max_delta)
        finite &= theta_votes >= (theta_prev - monotonic_slack)
    if not finite.any():
        return float("nan"), float("inf"), {
            "method": "three_point_direct",
            "used_points": 0,
            "candidate_points": int(len(ids)),
            "reason": "no_finite_consistent_votes",
        }

    radius_scale = max(float(np.nanmedian(obs["axis_radius"][finite])), config.three_point_min_axis_radius_m, 1e-6)
    reproj_penalty = 1.0 / (1.0 + obs["reproj_pred"] / max(config.reproj_scale_px, 1e-6))
    plane_penalty = 1.0 / (1.0 + obs["plane_dist"] / max(config.plane_scale_m, 1e-6))
    depth_penalty = 1.0 / (1.0 + obs["depth_residual"] / max(config.depth_scale_m, 1e-6))
    radius_score = np.clip(obs["axis_radius"] / radius_scale, 0.0, 3.0)
    score = np.maximum(obs["confidence"], 0.05) * radius_score * reproj_penalty * plane_penalty * depth_penalty
    score[~finite] = -np.inf
    order = np.argsort(-score)
    order = order[np.isfinite(score[order]) & (score[order] > 0.0)]
    if len(order) == 0:
        return float("nan"), float("inf"), {
            "method": "three_point_direct",
            "used_points": 0,
            "candidate_points": int(len(ids)),
            "reason": "all_votes_scored_zero",
        }

    max_candidates = max(config.three_point_count, int(config.three_point_candidate_count))
    cand = order[:max_candidates]
    cand_theta = theta_votes[cand]
    cand_weights = score[cand]
    center = weighted_median(cand_theta, cand_weights)
    residual = np.abs(wrap_angle_near(cand_theta, center) - center)
    mad_inliers = mad_inlier_mask(residual, config.mad_sigma)
    max_residual = math.radians(config.three_point_max_residual_deg)
    max_mad = math.radians(config.three_point_max_mad_deg)
    consistent = mad_inliers & (residual <= max_residual)
    if consistent.sum() == 0:
        best = np.asarray([cand[0]], dtype=np.int64)
    else:
        consistent_ids = cand[consistent]
        consistent_order = consistent_ids[np.argsort(-score[consistent_ids])]
        best = consistent_order[: max(1, int(config.three_point_count))]

    selected_theta = theta_votes[best]
    min_used = max(1, min(int(config.three_point_min_used_points), int(config.three_point_count)))
    if len(selected_theta) < min_used:
        return float("nan"), float("inf"), {
            "method": "three_point_direct",
            "mode": "incremental_delta" if use_incremental else "reference_absolute",
            "used_points": int(len(selected_theta)),
            "candidate_points": int(len(ids)),
            "reason": "not_enough_selected_points",
            "candidate_point_ids": ids[cand].astype(int).tolist(),
            "candidate_theta_deg": [float(np.rad2deg(v)) for v in cand_theta],
            "candidate_delta_deg": [float(np.rad2deg(v)) for v in delta_votes[cand]] if use_incremental else [],
        }
    if use_incremental and len(selected_theta) and float(np.median(selected_theta)) < theta_prev - monotonic_slack:
        return float("nan"), float("inf"), {
            "method": "three_point_direct",
            "mode": "incremental_delta",
            "used_points": 0,
            "candidate_points": int(len(ids)),
            "reason": "monotonicity_rejected",
            "candidate_point_ids": ids[cand].astype(int).tolist(),
            "candidate_theta_deg": [float(np.rad2deg(v)) for v in cand_theta],
            "candidate_delta_deg": [float(np.rad2deg(v)) for v in delta_votes[cand]],
        }
    selected_weights = np.maximum(score[best], 1e-8)
    theta = weighted_median(selected_theta, selected_weights)
    theta = float(np.clip(theta, angle_min, angle_max))
    selected_residual = np.abs(wrap_angle_near(selected_theta, theta) - theta)
    dispersion = float(np.median(selected_residual)) if len(selected_residual) else float("inf")
    temporal = float((theta - theta_prev) ** 2)
    accel = float((theta - 2.0 * theta_prev + theta_prev2) ** 2)
    # This loss is diagnostic only; unlike loss_1d, the final angle is the
    # selected points' direct vote, not the minimum of a free objective.
    loss = float((dispersion / max(max_mad, 1e-6)) ** 2 + config.lambda_temporal * temporal + config.lambda_acc * accel)
    diag = {
        "method": "three_point_direct",
        "mode": "incremental_delta" if use_incremental else "reference_absolute",
        "used_points": int(len(best)),
        "candidate_points": int(len(ids)),
        "selected_point_ids": ids[best].astype(int).tolist(),
        "selected_theta_deg": [float(np.rad2deg(v)) for v in selected_theta],
        "selected_delta_deg": [float(np.rad2deg(v)) for v in delta_votes[best]] if use_incremental else [],
        "selected_residual_deg": [float(np.rad2deg(v)) for v in selected_residual],
        "selected_confidence": [float(v) for v in obs["confidence"][best]],
        "selected_axis_radius_m": [float(v) for v in obs["axis_radius"][best]],
        "selected_reproj_pred_px": [float(v) for v in obs["reproj_pred"][best]],
        "selected_plane_dist_m": [float(v) for v in obs["plane_dist"][best]],
        "selected_depth_residual_m": [float(v) for v in obs["depth_residual"][best]],
        "candidate_point_ids": ids[cand].astype(int).tolist(),
        "candidate_theta_deg": [float(np.rad2deg(v)) for v in cand_theta],
        "candidate_delta_deg": [float(np.rad2deg(v)) for v in delta_votes[cand]] if use_incremental else [],
        "candidate_scores": [float(v) for v in cand_weights],
        "theta_pred_deg": float(np.rad2deg(theta_pred)),
        "dispersion_deg": float(np.rad2deg(dispersion)),
    }
    return theta, loss, diag


def score_depth_plane_theta(
    theta_rad: float,
    depth_points_align: np.ndarray,
    origin: np.ndarray,
    axis: np.ndarray,
    plane_center_ref: np.ndarray,
    plane_normal_ref: np.ndarray,
    theta_prev: float,
    theta_prev2: float,
    config: ScreenHingeTrackingConfig,
) -> tuple[float, dict[str, Any]]:
    if len(depth_points_align) < config.depth_only_min_points:
        return float("inf"), {"used_depth_points": int(len(depth_points_align))}
    center_theta = rotate_points_about_axis(plane_center_ref[None, :], origin, axis, theta_rad)[0]
    normal_theta = rotate_vectors_about_axis(plane_normal_ref[None, :], axis, theta_rad)[0]
    normal_theta /= np.linalg.norm(normal_theta) + 1e-12
    distances = np.abs((depth_points_align - center_theta) @ normal_theta)
    inliers = mad_inlier_mask(distances, config.mad_sigma)
    if inliers.sum() < config.depth_only_min_points // 3:
        return float("inf"), {"used_depth_points": int(inliers.sum())}
    plane_loss = float(np.mean(huber_loss(distances[inliers] / config.plane_scale_m, config.robust_delta)))
    temporal = float((theta_rad - theta_prev) ** 2)
    accel = float((theta_rad - 2.0 * theta_prev + theta_prev2) ** 2)
    loss = config.lambda_plane * plane_loss + config.lambda_temporal * temporal + config.lambda_acc * accel
    return loss, {
        "used_depth_points": int(inliers.sum()),
        "depth_plane_median_m": float(np.median(distances[inliers])),
    }


def bounded_1d_optimize(
    score_fn: Any,
    theta_pred: float,
    theta_prev: float,
    config: ScreenHingeTrackingConfig,
    search_radius_deg: float,
) -> tuple[float, float, dict[str, Any]]:
    angle_min = math.radians(config.angle_min_deg)
    angle_max = math.radians(config.angle_max_deg)
    delta = math.radians(min(config.max_angle_delta_deg, search_radius_deg))
    lo = max(angle_min, theta_pred - delta)
    hi = min(angle_max, theta_pred + delta)
    if hi <= lo:
        lo, hi = angle_min, angle_max
    coarse = np.linspace(lo, hi, max(5, int(config.coarse_steps)))
    best_theta = float(theta_pred)
    best_loss = float("inf")
    best_diag: dict[str, Any] = {}
    for theta in coarse:
        loss, diag = score_fn(float(theta))
        if loss < best_loss:
            best_theta = float(theta)
            best_loss = float(loss)
            best_diag = diag
    fine_lo = max(lo, best_theta - math.radians(2.0))
    fine_hi = min(hi, best_theta + math.radians(2.0))
    try:
        result = minimize_scalar(lambda x: score_fn(float(x))[0], bounds=(fine_lo, fine_hi), method="bounded", options={"xatol": 1e-4})
        if result.success and float(result.fun) < best_loss:
            best_theta = float(result.x)
            best_loss = float(result.fun)
            _, best_diag = score_fn(best_theta)
    except Exception as exc:  # pragma: no cover - defensive logging path
        best_diag["minimize_scalar_error"] = str(exc)
    return best_theta, best_loss, best_diag


def estimate_sequence(
    config: ScreenHingeTrackingConfig,
    meta: dict[str, Any],
    frames: list[int],
    tracks_xy: np.ndarray,
    visibility: np.ndarray,
    confidence: np.ndarray,
    query_frames: np.ndarray,
    local_points: np.ndarray,
    samplers: list[DepthFrameSampler],
    screen_mesh0: trimesh.Trimesh,
    joint: dict[str, Any],
    plane_center_ref: np.ndarray,
    plane_normal_ref: np.ndarray,
) -> SequenceEstimate:
    origin = np.asarray(joint["origin_xyz"], dtype=np.float64)
    axis = np.asarray(joint["axis_xyz"], dtype=np.float64)
    axis /= np.linalg.norm(axis) + 1e-12
    align_row = frame_row(config.export_root, frames[0])
    use_three_point_direct = config.angle_method == "three_point_direct"
    if config.angle_method not in {"loss_1d", "three_point_direct"}:
        raise ValueError(f"Unsupported angle_method={config.angle_method!r}; expected loss_1d or three_point_direct")
    measured_valid_threshold = max(1, int(config.three_point_count)) if use_three_point_direct else int(config.min_valid_points)
    low_valid_threshold = 1 if use_three_point_direct else int(config.low_valid_points)
    angles = np.zeros(len(frames), dtype=np.float64)
    estimates: list[FrameEstimate] = []
    all_observations: list[dict[str, Any]] = []
    prev_point_align = np.full_like(local_points, np.nan, dtype=np.float64)
    prev_point_depth = np.full(len(local_points), np.nan, dtype=np.float64)
    last_status = "A_measured"
    for local_idx, frame in enumerate(frames):
        if local_idx == 0:
            init_ids = np.flatnonzero((query_frames <= 0) & np.isfinite(local_points).all(axis=1)).astype(int).tolist()
            estimates.append(
                FrameEstimate(frame, local_idx, 0.0, 0.0, "A_reference", len(init_ids), 1.0, 0.0, init_ids, False, {})
            )
            continue
        theta_prev = float(angles[local_idx - 1])
        theta_prev2 = float(angles[local_idx - 2]) if local_idx >= 2 else theta_prev
        theta_pred = theta_prev + (theta_prev - theta_prev2)
        search_radius = config.reappear_search_radius_deg if last_status.startswith(("C_", "D_")) else config.angle_search_radius_deg
        obs, obs_rows = collect_frame_observations(
            local_idx,
            tracks_xy,
            visibility,
            confidence,
            query_frames,
            local_points,
            samplers,
            meta,
            config.export_root,
            frames,
            joint,
            screen_mesh0,
            theta_pred,
            config,
            plane_center_ref,
            plane_normal_ref,
            prev_point_align,
            prev_point_depth,
        )
        all_observations.extend(obs_rows)
        if len(obs["ids"]):
            prev_point_align[obs["ids"]] = obs["point_align"]
            prev_point_depth[obs["ids"]] = obs["point_frame"][:, 2]
        valid_count = int(len(obs["ids"]))
        view_row = frame_row(config.export_root, frame)
        t_frame_from_align = camera_to_camera_matrix(meta, align_row, view_row, camera="right")
        if valid_count >= measured_valid_threshold:
            status = "A_measured"
        elif valid_count >= low_valid_threshold:
            status = "B_low_track"
        else:
            status = "C_depth_only"
        if status in {"A_measured", "B_low_track"}:
            if use_three_point_direct:
                theta, loss, diag = estimate_three_point_direct_theta(obs, theta_pred, theta_prev, theta_prev2, search_radius, config)
                if np.isfinite(theta):
                    used_ids = [int(v) for v in diag.get("selected_point_ids", obs["ids"].tolist())]
                    confidence_score = min(1.0, max(0.05, len(used_ids) / float(max(1, config.three_point_count))))
                else:
                    status = "H_hold_track_rejected"
                    theta = theta_prev
                    loss = float("inf")
                    used_ids = []
                    confidence_score = 0.05
            else:
                score_fn = lambda theta: score_track_theta(
                    theta,
                    obs,
                    local_points,
                    meta,
                    t_frame_from_align,
                    origin,
                    axis,
                    plane_center_ref,
                    plane_normal_ref,
                    theta_pred,
                    theta_prev,
                    theta_prev2,
                    config,
                )
                theta, loss, diag = bounded_1d_optimize(score_fn, theta_pred, theta_prev, config, search_radius)
                used_ids = [int(v) for v in diag.get("inlier_point_ids", obs["ids"].tolist())]
                confidence_score = min(1.0, max(0.05, len(used_ids) / float(max(1, config.min_valid_points))))
        if status == "C_depth_only":
            if use_three_point_direct and not config.three_point_allow_depth_only:
                status = "H_hold_no_track"
                theta = theta_prev
                loss = float("inf")
                diag = {"reason": "depth_only_disabled_for_three_point_direct"}
                used_ids = []
                confidence_score = 0.05
            else:
                roi_mask, _ = project_screen_roi(
                    meta,
                    config.export_root,
                    frames,
                    local_idx,
                    screen_mesh0,
                    joint,
                    theta_pred,
                    (int(meta["rgb_height_per_eye"]), int(meta["rgb_width_per_eye"])),
                    config.roi_dilation_px,
                )
                depth_points_frame = samplers[local_idx].points_in_mask(roi_mask, max_points=3000)
                if len(depth_points_frame) >= config.depth_only_min_points:
                    t_align_from_frame = camera_to_camera_matrix(meta, view_row, align_row, camera="right")
                    depth_points_align = transform_points(depth_points_frame, t_align_from_frame)
                    score_fn = lambda theta: score_depth_plane_theta(
                        theta,
                        depth_points_align,
                        origin,
                        axis,
                        plane_center_ref,
                        plane_normal_ref,
                        theta_prev,
                        theta_prev2,
                        config,
                    )
                    theta, loss, diag = bounded_1d_optimize(score_fn, theta_pred, theta_prev, config, config.reappear_search_radius_deg)
                    used_ids = []
                    confidence_score = 0.25
                    diag["depth_roi_points"] = int(len(depth_points_frame))
                else:
                    status = "H_hold_no_track_or_depth"
                    theta = theta_prev
                    loss = float("inf")
                    diag = {"reason": "not_enough_track_or_depth", "depth_roi_points": int(len(depth_points_frame))}
                    used_ids = []
                    confidence_score = 0.05
        if last_status.startswith(("D_", "H_")) and status in {"A_measured", "B_low_track"}:
            status = "E_reacquired"
        angles[local_idx] = theta
        should_reseed = (valid_count < measured_valid_threshold and not status.startswith(("D_", "H_"))) or status.startswith("H_hold_")
        estimates.append(
            FrameEstimate(
                frame=frame,
                local_index=local_idx,
                theta_rad=float(theta),
                theta_pred_rad=float(theta_pred),
                status=status,
                valid_points=valid_count,
                confidence=float(confidence_score),
                loss=float(loss),
                valid_point_ids=used_ids,
                should_reseed=bool(should_reseed),
                diagnostics=diag,
            )
        )
        last_status = status
    return SequenceEstimate(angles_rad=angles, frames=estimates, observations=all_observations)


def save_query_overlay(rgb: np.ndarray, mask: np.ndarray, points: np.ndarray, path: Path) -> None:
    image = Image.fromarray(rgb).convert("RGBA")
    tint = Image.new("RGBA", image.size, (30, 220, 80, 0))
    tint.putalpha(Image.fromarray(mask.astype(np.uint8) * 70, mode="L"))
    image = Image.alpha_composite(image, tint)
    draw = ImageDraw.Draw(image)
    for idx, (x, y) in enumerate(points):
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=(255, 230, 0, 255), width=3)
        if idx % 4 == 0:
            draw.text((x + 6, y - 6), str(idx), fill=(255, 230, 0, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path)


def export_dynamic_meshes(
    config: ScreenHingeTrackingConfig,
    meta: dict[str, Any],
    frames: list[int],
    angles_rad: np.ndarray,
    joint: dict[str, Any],
) -> list[dict[str, Any]]:
    base_mesh0 = load_mesh(config.alignment_dir / f"part_{BASE_PART_LABEL}_camera.obj")
    screen_mesh0 = load_mesh(config.alignment_dir / f"part_{SCREEN_PART_LABEL}_camera.obj")
    origin = np.asarray(joint["origin_xyz"], dtype=np.float64)
    axis = np.asarray(joint["axis_xyz"], dtype=np.float64)
    axis /= np.linalg.norm(axis) + 1e-12
    align_row = frame_row(config.export_root, frames[0])
    entries: list[dict[str, Any]] = []
    for local_idx, frame in enumerate(frames):
        frame_dir = config.output_dir / f"frame_{frame_name(frame)}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        view_row = frame_row(config.export_root, frame)
        t_frame_from_align = camera_to_camera_matrix(meta, align_row, view_row, camera="right")
        base_frame = apply_se3_to_mesh(base_mesh0, t_frame_from_align)
        screen_rot = rotate_mesh_about_axis(screen_mesh0, origin, axis, float(angles_rad[local_idx]))
        screen_frame = apply_se3_to_mesh(screen_rot, t_frame_from_align)
        joint_frame = transform_joint_se3(joint, t_frame_from_align)
        base_path = frame_dir / f"part_{BASE_PART_LABEL}_dynamic.obj"
        screen_path = frame_dir / f"part_{SCREEN_PART_LABEL}_dynamic.obj"
        joint_path = frame_dir / "joint_dynamic.json"
        base_frame.export(base_path)
        screen_frame.export(screen_path)
        write_json(joint_path, {"joints": [joint_frame]})
        entries.append(
            {
                "frame": int(frame),
                "angle_rad": float(angles_rad[local_idx]),
                "angle_deg": float(np.rad2deg(angles_rad[local_idx])),
                "base_mesh": str(base_path),
                "screen_mesh": str(screen_path),
                "joint_json": str(joint_path),
            }
        )
    return entries


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_overlay_video(
    config: ScreenHingeTrackingConfig,
    meta: dict[str, Any],
    frames: list[int],
    frames_rgb: list[np.ndarray],
    screen_mesh0: trimesh.Trimesh,
    joint: dict[str, Any],
    estimates: SequenceEstimate,
    tracks_xy: np.ndarray,
    visibility: np.ndarray,
    query_frames: np.ndarray,
    fps: float = 6.0,
) -> Path:
    out_path = config.output_dir / "screen_hinge_overlay.mp4"
    h, w = frames_rgb[0].shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    valid_by_frame = {item.local_index: set(item.valid_point_ids) for item in estimates.frames}
    for local_idx, rgb in enumerate(frames_rgb):
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        est = estimates.frames[local_idx]
        _, hull = project_screen_roi(meta, config.export_root, frames, local_idx, screen_mesh0, joint, est.theta_rad, (h, w), 0)
        if len(hull) >= 3:
            cv2.polylines(bgr, [hull.astype(np.int32)], isClosed=True, color=(255, 160, 40), thickness=3)
        valid_ids = valid_by_frame.get(local_idx, set())
        for pid in range(tracks_xy.shape[1]):
            if query_frames[pid] > local_idx or not visibility[local_idx, pid]:
                continue
            x, y = tracks_xy[local_idx, pid]
            color = (60, 230, 60) if pid in valid_ids else (160, 160, 160)
            cv2.circle(bgr, (int(round(x)), int(round(y))), 3, color, -1)
        text = f"f={frames[local_idx]} theta={math.degrees(est.theta_rad):+.1f} {est.status} valid={est.valid_points}"
        cv2.putText(bgr, text, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (10, 10, 10), 4, cv2.LINE_AA)
        cv2.putText(bgr, text, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(bgr)
    writer.release()
    return out_path


def find_reseed_candidate(sequence: SequenceEstimate, used_frames: set[int], config: ScreenHingeTrackingConfig) -> int | None:
    for item in sequence.frames[1:]:
        if item.local_index in used_frames:
            continue
        if item.local_index < min(used_frames, default=-config.min_reseed_interval) + config.min_reseed_interval:
            continue
        if config.angle_method == "three_point_direct" and item.status not in {"A_measured", "B_low_track", "E_reacquired", "H_hold_track_rejected", "H_hold_no_track"}:
            continue
        if item.should_reseed:
            return item.local_index
    return None


def append_reseed_queries(
    config: ScreenHingeTrackingConfig,
    meta: dict[str, Any],
    frames: list[int],
    frames_rgb: list[np.ndarray],
    samplers: list[DepthFrameSampler],
    screen_mesh0: trimesh.Trimesh,
    joint: dict[str, Any],
    plane_center_ref: np.ndarray,
    plane_normal_ref: np.ndarray,
    sequence: SequenceEstimate,
    tracks_xy: np.ndarray,
    visibility: np.ndarray,
    queries_xy: np.ndarray,
    query_frames: np.ndarray,
    local_points: np.ndarray,
    query_labels: np.ndarray,
    reseed_frame: int,
    tracker_frame_for_reseed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    theta = float(sequence.angles_rad[reseed_frame])
    roi_mask, _ = project_screen_roi(
        meta,
        config.export_root,
        frames,
        reseed_frame,
        screen_mesh0,
        joint,
        theta,
        frames_rgb[reseed_frame].shape[:2],
        config.roi_dilation_px,
    )
    visible_existing = visibility[reseed_frame] & (query_frames <= tracker_frame_for_reseed)
    existing_xy = tracks_xy[reseed_frame, visible_existing] if visible_existing.any() else None
    candidates = select_good_features(
        frames_rgb[reseed_frame],
        roi_mask,
        max_points=config.reseed_points * 3,
        min_distance_px=config.reseed_min_distance_px,
        quality_level=config.reseed_quality_level,
        erode_px=0,
        existing_xy=existing_xy,
    )
    if len(candidates) == 0:
        return queries_xy, query_frames, local_points, query_labels, {"status": "no_features", "selected_points": 0}
    cand_local, cand_valid = register_query_points(
        candidates,
        reseed_frame,
        theta,
        samplers,
        meta,
        config.export_root,
        frames,
        joint,
        plane_center_ref,
        plane_normal_ref,
        config,
    )
    valid_idx = np.flatnonzero(cand_valid)
    if valid_idx.size:
        feature_scores = feature_strengths(frames_rgb[reseed_frame], candidates)
        keep_rel, reseed_selection = select_anchor_subset(
            candidates[valid_idx],
            cand_local[valid_idx],
            feature_scores[valid_idx],
            joint,
            config.reseed_points,
            config.reseed_min_distance_px,
            config.init_anchor_min_axis_radius_m,
            config.init_anchor_min_axis_radius_quantile,
            config.init_anchor_feature_weight,
            config.init_anchor_axis_distance_weight,
            config.init_anchor_top_weight,
        )
        keep_idx = valid_idx[keep_rel]
    else:
        reseed_selection = {"strategy": "no_registered_candidates", "candidate_count": int(len(candidates))}
        keep_idx = valid_idx
    min_registered = min(config.reseed_points, max(config.min_reseed_registered_points, config.low_valid_points))
    if keep_idx.size < min_registered:
        return queries_xy, query_frames, local_points, query_labels, {
            "status": "not_enough_registered",
            "selected_points": int(keep_idx.size),
            "min_required": int(min_registered),
            "selection": reseed_selection,
        }
    new_xy = candidates[keep_idx].astype(np.float32)
    new_local = cand_local[keep_idx].astype(np.float64)
    new_frames = np.full(len(new_xy), tracker_frame_for_reseed, dtype=np.int64)
    new_labels = np.full(len(new_xy), f"reseed_{frames[reseed_frame]}", dtype="<U32")
    return (
        np.vstack([queries_xy, new_xy]),
        np.concatenate([query_frames, new_frames]),
        np.vstack([local_points, new_local]),
        np.concatenate([query_labels, new_labels]),
        {
            "status": "added",
            "selected_points": int(len(new_xy)),
            "frame": int(frames[reseed_frame]),
            "local_frame": int(reseed_frame),
            "theta_deg": float(np.rad2deg(theta)),
            "selection": reseed_selection,
        },
    )


def run_screen_hinge_tracking(config: ScreenHingeTrackingConfig) -> dict[str, Any]:
    config.alignment_dir = config.alignment_dir.resolve()
    config.export_root = config.export_root.resolve()
    config.output_dir = config.output_dir.resolve()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    frames = list(range(config.start_frame, config.end_frame + 1))
    if not frames:
        raise ValueError("No frames requested")
    result = read_json(config.alignment_dir / "alignment_result.json")
    meta = read_json(config.export_root / "manifest.json")
    convention = config.depth_convention or result.get("convention_used", "camera_to_rig")
    sync_warnings = check_sequence_inputs(config, frames, meta)
    joints = read_json(config.alignment_dir / "joint_camera.json").get("joints", [])
    if not joints:
        raise ValueError(f"No joint found in {config.alignment_dir / 'joint_camera.json'}")
    joint = joints[0]
    if "origin_xyz" not in joint or "axis_xyz" not in joint:
        raise ValueError("joint_camera.json must provide origin_xyz and axis_xyz; no camera-axis default is assumed")
    frames_rgb = load_rgb_frames(config.export_root, frames, config.rgb_dir_name)
    if config.tracker_rgb_dir is not None:
        tracker_rgb_dir = config.tracker_rgb_dir.resolve()
        if not tracker_rgb_dir.exists():
            raise FileNotFoundError(f"tracker_rgb_dir does not exist: {tracker_rgb_dir}")
        tracker_start = int(config.tracker_start_frame) if config.tracker_start_frame is not None else 0
        if config.tracker_end_frame is not None:
            tracker_end = int(config.tracker_end_frame)
        else:
            tracker_files = sorted(tracker_rgb_dir.glob("*.png"))
            if not tracker_files:
                raise FileNotFoundError(f"No PNG tracker RGB frames found in {tracker_rgb_dir}")
            tracker_end = int(tracker_files[-1].stem)
        tracker_frames = list(range(tracker_start, tracker_end + 1, max(1, int(config.tracker_stride))))
        tracker_frames_rgb = load_rgb_frames_from_dir(tracker_rgb_dir, tracker_frames)
    else:
        tracker_rgb_dir = config.export_root / config.rgb_dir_name
        tracker_frames = frames
        tracker_frames_rgb = frames_rgb
    eval_to_tracker = map_eval_to_tracker_indices(
        frames,
        tracker_frames,
        float(config.tracker_fps),
        float(config.eval_fps),
    )
    if tracker_frames_rgb[0].shape[:2] != frames_rgb[0].shape[:2]:
        raise ValueError(
            f"Tracker RGB shape {tracker_frames_rgb[0].shape[:2]} must match eval RGB/depth shape {frames_rgb[0].shape[:2]}"
        )
    samplers = build_depth_samplers(config, meta, frames, frames_rgb[0].shape[:2], convention)
    screen_mask = load_screen_mask(config.alignment_dir, result)
    if screen_mask.shape != frames_rgb[0].shape[:2]:
        raise ValueError(f"Screen mask shape {screen_mask.shape} does not match RGB shape {frames_rgb[0].shape[:2]}")
    screen_mesh0 = load_mesh(config.alignment_dir / f"part_{SCREEN_PART_LABEL}_camera.obj")
    plane_center_ref, plane_normal_ref = screen_plane_from_mesh(screen_mesh0)

    init_xy = select_good_features(
        frames_rgb[0],
        screen_mask,
        max_points=config.init_max_points,
        min_distance_px=config.init_min_distance_px,
        quality_level=config.init_quality_level,
        erode_px=config.init_erode_px,
    )
    if len(init_xy) < 3:
        raise RuntimeError("Could not initialize enough textured screen points from the first mask")
    init_local, init_valid = register_query_points(
        init_xy,
        0,
        0.0,
        samplers,
        meta,
        config.export_root,
        frames,
        joint,
        plane_center_ref,
        plane_normal_ref,
        config,
    )
    init_xy = init_xy[init_valid]
    local_points = init_local[init_valid]
    if len(init_xy) < 3:
        raise RuntimeError("Initial points were selected, but too few registered to the screen plane using depth")
    anchor_selection: dict[str, Any] = {"enabled": False, "candidate_count": int(len(init_xy))}
    if config.init_anchor_count > 0:
        feature_scores = feature_strengths(frames_rgb[0], init_xy)
        keep_idx, anchor_selection = select_anchor_subset(
            init_xy,
            local_points,
            feature_scores,
            joint,
            config.init_anchor_count,
            config.init_anchor_min_pixel_distance,
            config.init_anchor_min_axis_radius_m,
            config.init_anchor_min_axis_radius_quantile,
            config.init_anchor_feature_weight,
            config.init_anchor_axis_distance_weight,
            config.init_anchor_top_weight,
        )
        anchor_selection["enabled"] = True
        init_xy = init_xy[keep_idx]
        local_points = local_points[keep_idx]
    queries_xy = init_xy.astype(np.float32)
    query_frames = np.full(len(queries_xy), int(eval_to_tracker[0]), dtype=np.int64)
    label = "anchor" if config.init_anchor_count > 0 else "init"
    query_labels = np.full(len(queries_xy), label, dtype="<U32")
    save_query_overlay(frames_rgb[0], screen_mask, queries_xy, config.output_dir / "initial_good_features.png")

    reseed_events: list[dict[str, Any]] = []
    used_reseed_frames: set[int] = set()
    tracker_info: dict[str, Any] = {}
    final_sequence: SequenceEstimate | None = None
    final_tracks = final_visibility = final_confidence = None
    for pass_idx in range(config.max_reseed_events + 1):
        tracks_xy, visibility, confidence, tracker_info = run_cotracker_offline(
            tracker_frames_rgb,
            queries_xy,
            query_frames,
            config.cotracker_root.resolve(),
            config.cotracker_checkpoint.resolve(),
            config.device,
            config.tracker_max_side,
            config.cotracker_conf_threshold,
            config.cotracker_iters,
        )
        eval_tracks_xy = tracks_xy[eval_to_tracker]
        eval_visibility = visibility[eval_to_tracker]
        eval_confidence = confidence[eval_to_tracker]
        sequence = estimate_sequence(
            config,
            meta,
            frames,
            eval_tracks_xy,
            eval_visibility,
            eval_confidence,
            query_frames,
            local_points,
            samplers,
            screen_mesh0,
            joint,
            plane_center_ref,
            plane_normal_ref,
        )
        final_sequence = sequence
        final_tracks, final_visibility, final_confidence = eval_tracks_xy, eval_visibility, eval_confidence
        if pass_idx >= config.max_reseed_events:
            break
        reseed_frame = find_reseed_candidate(sequence, used_reseed_frames, config)
        if reseed_frame is None:
            break
        used_reseed_frames.add(reseed_frame)
        queries_xy, query_frames, local_points, query_labels, event = append_reseed_queries(
            config,
            meta,
            frames,
            frames_rgb,
            samplers,
            screen_mesh0,
            joint,
            plane_center_ref,
            plane_normal_ref,
            sequence,
            eval_tracks_xy,
            eval_visibility,
            queries_xy,
            query_frames,
            local_points,
            query_labels,
            reseed_frame,
            int(eval_to_tracker[reseed_frame]),
        )
        event["pass"] = int(pass_idx + 1)
        reseed_events.append(event)
        if event.get("status") != "added":
            break

    assert final_sequence is not None and final_tracks is not None and final_visibility is not None and final_confidence is not None
    np.save(config.output_dir / "tracks_2d_xy.npy", final_tracks)
    np.save(config.output_dir / "tracks_visibility.npy", final_visibility)
    np.save(config.output_dir / "tracks_confidence.npy", final_confidence)
    np.save(config.output_dir / "query_frames.npy", query_frames)
    np.save(config.output_dir / "query_points_xy.npy", queries_xy)
    np.save(config.output_dir / "query_local_points_align.npy", local_points)
    np.save(config.output_dir / "screen_angles_rad.npy", final_sequence.angles_rad.astype(np.float32))

    query_rows = [
        {
            "point_id": int(idx),
            "query_frame": int(tracker_frames[int(query_frames[idx])]) if int(query_frames[idx]) < len(tracker_frames) else int(query_frames[idx]),
            "query_tracker_local_frame": int(query_frames[idx]),
            "query_x": float(queries_xy[idx, 0]),
            "query_y": float(queries_xy[idx, 1]),
            "x_screen_ref": float(local_points[idx, 0]),
            "y_screen_ref": float(local_points[idx, 1]),
            "z_screen_ref": float(local_points[idx, 2]),
            "label": str(query_labels[idx]),
            "valid": bool(np.isfinite(local_points[idx]).all()),
        }
        for idx in range(len(queries_xy))
    ]
    write_csv(
        config.output_dir / "query_points.csv",
        query_rows,
        ["point_id", "query_frame", "query_tracker_local_frame", "query_x", "query_y", "x_screen_ref", "y_screen_ref", "z_screen_ref", "label", "valid"],
    )
    angle_rows = [
        {
            "frame": item.frame,
            "local_frame": item.local_index,
            "theta_rad": item.theta_rad,
            "theta_deg": float(np.rad2deg(item.theta_rad)),
            "theta_pred_deg": float(np.rad2deg(item.theta_pred_rad)),
            "status": item.status,
            "valid_points": item.valid_points,
            "confidence": item.confidence,
            "loss": item.loss,
            "angle_method": str(item.diagnostics.get("method", config.angle_method)),
            "selected_point_ids": json.dumps(item.diagnostics.get("selected_point_ids", item.valid_point_ids), ensure_ascii=True),
            "selected_theta_deg": json.dumps(item.diagnostics.get("selected_theta_deg", []), ensure_ascii=True),
            "selected_delta_deg": json.dumps(item.diagnostics.get("selected_delta_deg", []), ensure_ascii=True),
            "dispersion_deg": item.diagnostics.get("dispersion_deg", ""),
            "should_reseed": item.should_reseed,
        }
        for item in final_sequence.frames
    ]
    write_csv(
        config.output_dir / "screen_angles.csv",
        angle_rows,
        [
            "frame",
            "local_frame",
            "theta_rad",
            "theta_deg",
            "theta_pred_deg",
            "status",
            "valid_points",
            "confidence",
            "loss",
            "angle_method",
            "selected_point_ids",
            "selected_theta_deg",
            "selected_delta_deg",
            "dispersion_deg",
            "should_reseed",
        ],
    )
    obs_fields = [
        "frame", "local_frame", "point_id", "x", "y", "visibility", "confidence", "in_bounds", "depth_ok", "roi_ok", "jump_px", "reproj_pred_px", "plane_dist_m", "depth_residual_m", "axis_radius_m", "depth_jump_m", "depth_ratio", "depth_jump_ok", "per_point_theta_rad", "per_point_theta_deg", "per_point_delta_rad", "per_point_delta_deg", "valid_for_angle",
    ]
    write_csv(config.output_dir / "observations.csv", final_sequence.observations, obs_fields)
    frame_entries = export_dynamic_meshes(config, meta, frames, final_sequence.angles_rad, joint)
    overlay_video = save_overlay_video(config, meta, frames, frames_rgb, screen_mesh0, joint, final_sequence, final_tracks, final_visibility, query_frames)
    manifest = {
        "type": "screen_hinge_rgbd_stable",
        "alignment_dir": str(config.alignment_dir),
        "export_root": str(config.export_root),
        "output_dir": str(config.output_dir),
        "frames": frame_entries,
        "frame_indices": frames,
        "tracker_rgb_dir": str(tracker_rgb_dir),
        "tracker_frame_indices": [int(v) for v in tracker_frames],
        "eval_to_tracker_indices": [int(v) for v in eval_to_tracker.tolist()],
        "eval_to_tracker_frame_indices": [int(tracker_frames[int(v)]) for v in eval_to_tracker.tolist()],
        "screen_part_label": SCREEN_PART_LABEL,
        "base_part_label": BASE_PART_LABEL,
        "joint_align_camera": joint,
        "query_format": "[t, x, y] with t as local frame index in the requested range",
        "query_points_csv": str(config.output_dir / "query_points.csv"),
        "observations_csv": str(config.output_dir / "observations.csv"),
        "angles_csv": str(config.output_dir / "screen_angles.csv"),
        "overlay_video": str(overlay_video),
        "initial_query_overlay": str(config.output_dir / "initial_good_features.png"),
        "reseed_events": reseed_events,
        "anchor_selection": anchor_selection,
        "tracker": tracker_info,
        "depth": {
            "unit": config.depth_unit,
            "sample_mode": config.depth_sample_mode,
            "convention": convention,
            "depth_min_m": config.depth_min_m,
            "depth_max_m": config.depth_max_m,
            "depth_sample_radius_px": config.depth_sample_radius_px,
            "projected_depth_radius_px": config.projected_depth_radius_px,
        },
        "config": config.__dict__,
        "sync_warnings": sync_warnings,
        "angle_estimation": [item.__dict__ for item in final_sequence.frames],
    }
    write_json(config.output_dir / "stable_manifest.json", manifest)
    write_json(config.output_dir / "dynamic_manifest.json", manifest)
    return manifest
