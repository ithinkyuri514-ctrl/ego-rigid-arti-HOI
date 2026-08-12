#!/usr/bin/env python3
"""Track laptop screen points and export per-frame dynamic laptop meshes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vlm_sam2_recon.stages.camera_alignment import (  # noqa: E402
    apply_se3_to_mesh,
    camera_to_camera_matrix,
    depth_points_in_right_camera,
    frame_name,
    frame_row,
    project_right_camera_points,
    rotate_mesh_about_axis,
    rotate_points_about_axis,
    transform_points,
    transform_joint_se3,
    write_json,
)


DEFAULT_ALIGNMENT_DIR = PROJECT_ROOT / "outputs/object_alignment_screen_first_base_visible_snap/target_laptop/frame_000000"
DEFAULT_EXPORT_ROOT = Path("/code/3DVideo_2026-07-01-19-24-06-667_spatialmp4_export")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/screen_motion/target_laptop_frames_000000_000019"
DEFAULT_COTRACKER_ROOT = Path("/code/ArtHOI-4D-Reconstruction/third_party/co-tracker")
DEFAULT_COTRACKER_CHECKPOINT = DEFAULT_COTRACKER_ROOT / "checkpoints/scaled_offline.pth"

BASE_PART_LABEL = "14"
SCREEN_PART_LABEL = "15"
QUERY_COLOR = (255, 230, 0)
TRACK_COLORS = np.asarray(
    [
        (255, 230, 0),
        (255, 120, 20),
        (60, 230, 255),
        (120, 255, 80),
        (255, 80, 190),
        (180, 140, 255),
        (255, 255, 255),
        (40, 80, 255),
        (255, 40, 40),
        (40, 255, 130),
        (200, 200, 40),
        (40, 220, 220),
    ],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use CoTracker3 + RGB-D to estimate laptop screen hinge motion and export dynamic meshes."
    )
    parser.add_argument("--alignment-dir", type=Path, default=DEFAULT_ALIGNMENT_DIR)
    parser.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=19)
    parser.add_argument("--cotracker-root", type=Path, default=DEFAULT_COTRACKER_ROOT)
    parser.add_argument("--cotracker-checkpoint", type=Path, default=DEFAULT_COTRACKER_CHECKPOINT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tracker-max-side", type=int, default=768)
    parser.add_argument("--query-mode", choices=["lid_edge", "top_weighted", "dual"], default="dual")
    parser.add_argument("--num-query-points", type=int, default=40)
    parser.add_argument("--surface-query-points", type=int, default=56)
    parser.add_argument("--edge-query-points", type=int, default=36)
    parser.add_argument("--query-margin-px", type=float, default=3.0)
    parser.add_argument("--top-band-ratio", type=float, default=0.45)
    parser.add_argument("--top-band-weight", type=float, default=3.5)
    parser.add_argument("--edge-top-ratio", type=float, default=0.36)
    parser.add_argument("--edge-rows", type=int, default=4)
    parser.add_argument("--cotracker-conf-threshold", type=float, default=0.90)
    parser.add_argument("--enable-reseed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reseed-trigger-points", type=int, default=12)
    parser.add_argument("--reseed-target-points", type=int, default=28)
    parser.add_argument("--reseed-min-existing-points", type=int, default=4)
    parser.add_argument("--reseed-max-events", type=int, default=2)
    parser.add_argument("--reseed-roi-pad-px", type=float, default=70.0)
    parser.add_argument("--reseed-min-distance-px", type=float, default=16.0)
    parser.add_argument("--reseed-use-projected-screen-mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reseed-projected-mask-dilation-px", type=int, default=12)
    parser.add_argument("--depth-neighbor-radius-px", type=float, default=8.0)
    parser.add_argument("--min-angle-points", type=int, default=6)
    parser.add_argument("--angle-method", choices=["hinge_1d", "kabsch_axis", "per_point"], default="hinge_1d")
    parser.add_argument("--angle-trim-fraction", type=float, default=0.65)
    parser.add_argument("--hinge-angle-min-deg", type=float, default=-30.0)
    parser.add_argument("--hinge-angle-max-deg", type=float, default=130.0)
    parser.add_argument("--hinge-angle-coarse-steps", type=int, default=1601)
    parser.add_argument("--hinge-angle-fine-window-deg", type=float, default=3.0)
    parser.add_argument("--hinge-angle-fine-steps", type=int, default=301)
    parser.add_argument("--hinge-depth-weight-px-per-m", type=float, default=120.0)
    parser.add_argument("--hinge-continuity-weight-px-per-deg", type=float, default=0.15)
    parser.add_argument("--monotonic-hinge", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--monotonic-backoff-deg", type=float, default=1.5)
    parser.add_argument("--smooth-angle-spikes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--angle-spike-threshold-deg", type=float, default=12.0)
    parser.add_argument("--max-abs-angle-deg", type=float, default=120.0)
    parser.add_argument("--depth-convention", default=None, choices=[None, "camera_to_rig", "rig_to_camera", "direct_same_camera"])
    parser.add_argument("--depth-min-m", type=float, default=0.1)
    parser.add_argument("--depth-max-m", type=float, default=3.0)
    parser.add_argument("--rgb-dir-name", default="rgb_right_png")
    parser.add_argument("--depth-dir-name", default="depth_meters_npy")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_project_path(path_text: str | Path, base_dir: Path | None = None) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    candidates = []
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
    fallback = alignment_dir / "part_masks" / "target_laptop_frame_0_screen_projection.mask.npy"
    if fallback.exists():
        return np.load(fallback).astype(bool)
    fallback = alignment_dir / "part_masks" / "target_laptop_frame_0_screen.mask.npy"
    if fallback.exists():
        return np.load(fallback).astype(bool)
    raise FileNotFoundError(f"Could not resolve screen mask under {alignment_dir}")


def choose_lid_edge_query_points(
    valid_mask: np.ndarray,
    num_points: int,
    edge_top_ratio: float,
    edge_rows: int,
) -> np.ndarray:
    ys, xs = np.nonzero(valid_mask)
    if len(xs) == 0:
        raise ValueError("No valid mask pixels for lid-edge sampling.")
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    width = max(1, x1 - x0 + 1)
    height = max(1, y1 - y0 + 1)
    edge_top_ratio = float(np.clip(edge_top_ratio, 0.04, 0.85))
    rows = max(1, int(edge_rows))
    cols = max(2, int(np.ceil(num_points / float(rows))))
    top_limit = y0 + edge_top_ratio * height
    x_edges = np.linspace(x0, x1 + 1, cols + 1)
    row_fracs = np.linspace(0.02, edge_top_ratio, rows)

    selected: list[tuple[float, float]] = []
    used: set[tuple[int, int]] = set()

    def add_nearest(candidate: np.ndarray, target_xy: np.ndarray) -> None:
        if len(selected) >= num_points or not candidate.any():
            return
        cxs = xs[candidate]
        cys = ys[candidate]
        tree = cKDTree(np.column_stack([cxs, cys]).astype(np.float64))
        _, idx = tree.query(target_xy.astype(np.float64), k=1)
        point = (int(cxs[idx]), int(cys[idx]))
        if point in used:
            return
        used.add(point)
        selected.append((float(point[0]), float(point[1])))

    top_band = ys <= top_limit
    for row_frac in row_fracs:
        for col_idx in range(cols):
            in_bin = (xs >= x_edges[col_idx]) & (xs < x_edges[col_idx + 1]) & top_band
            if not in_bin.any():
                in_bin = (xs >= x_edges[col_idx]) & (xs < x_edges[col_idx + 1])
            if not in_bin.any():
                continue
            local_top = float(ys[in_bin].min())
            target_x = 0.5 * (x_edges[col_idx] + x_edges[col_idx + 1])
            target_y = min(float(top_limit), local_top + row_frac * height)
            add_nearest(in_bin, np.asarray([target_x, target_y], dtype=np.float64))
            if len(selected) >= num_points:
                return np.asarray(selected, dtype=np.float32)

    if len(selected) < num_points:
        top_candidates = top_band
        grid_cols = max(2, int(np.ceil(np.sqrt(num_points))))
        grid_rows = max(1, int(np.ceil((num_points - len(selected)) / float(grid_cols))))
        for fy in np.linspace(0.02, edge_top_ratio, grid_rows):
            for fx in np.linspace(0.04, 0.96, grid_cols):
                add_nearest(
                    top_candidates,
                    np.asarray([x0 + fx * width, y0 + fy * height], dtype=np.float64),
                )
                if len(selected) >= num_points:
                    return np.asarray(selected, dtype=np.float32)

    if len(selected) < num_points:
        rng = np.random.default_rng(29)
        top_idx = np.flatnonzero(top_band)
        fill_idx = top_idx if top_idx.size else np.arange(len(xs))
        for idx in rng.permutation(fill_idx):
            point = (int(xs[idx]), int(ys[idx]))
            if point in used:
                continue
            used.add(point)
            selected.append((float(point[0]), float(point[1])))
            if len(selected) >= num_points:
                break

    if len(selected) < 3:
        raise ValueError(f"Only selected {len(selected)} lid-edge query points.")
    return np.asarray(selected, dtype=np.float32)


def choose_query_points(
    mask: np.ndarray,
    num_points: int,
    margin_px: float,
    top_band_ratio: float,
    top_band_weight: float,
    query_mode: str,
    edge_top_ratio: float,
    edge_rows: int,
) -> np.ndarray:
    if mask.ndim != 2 or not mask.any():
        raise ValueError("Screen mask is empty.")
    dist = distance_transform_edt(mask)
    valid = dist >= float(margin_px)
    if valid.sum() < max(16, num_points):
        valid = mask.astype(bool)

    if query_mode == "lid_edge":
        return choose_lid_edge_query_points(valid, num_points, edge_top_ratio, edge_rows)

    ys, xs = np.nonzero(valid)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    width = max(1, x1 - x0 + 1)
    height = max(1, y1 - y0 + 1)
    aspect = width / float(height)
    rows = max(4, int(round(np.sqrt(max(1, num_points) / max(aspect, 0.25)))))
    rows = min(rows, max(1, num_points))
    row_centers = (np.arange(rows, dtype=np.float64) + 0.5) / float(rows)
    top_band_ratio = float(np.clip(top_band_ratio, 0.0, 1.0))
    weights = np.ones(rows, dtype=np.float64)
    weights[row_centers <= top_band_ratio] *= max(1.0, float(top_band_weight))
    raw_counts = weights / weights.sum() * float(num_points)
    row_counts = np.maximum(1, np.floor(raw_counts).astype(np.int64))
    while int(row_counts.sum()) < num_points:
        deficits = raw_counts - row_counts
        row_counts[int(np.argmax(deficits))] += 1
    while int(row_counts.sum()) > num_points:
        removable = np.where(row_counts > 1)[0]
        if len(removable) == 0:
            break
        surplus = row_counts - raw_counts
        row_counts[int(removable[np.argmax(surplus[removable])])] -= 1

    selected: list[tuple[float, float]] = []
    used: set[tuple[int, int]] = set()
    y_edges = np.linspace(y0, y1 + 1, rows + 1)
    for row_idx, cols in enumerate(row_counts):
        if cols <= 0:
            continue
        band = (ys >= y_edges[row_idx]) & (ys < y_edges[row_idx + 1])
        if not band.any():
            y_center = 0.5 * (y_edges[row_idx] + y_edges[row_idx + 1])
            band = np.abs(ys.astype(np.float64) - y_center) <= max(2.0, height / rows)
        band_xs = xs[band]
        band_ys = ys[band]
        if len(band_xs) == 0:
            continue
        band_tree = cKDTree(np.column_stack([band_xs, band_ys]).astype(np.float64))
        y_target = float(np.median(band_ys))
        for col_idx in range(int(cols)):
            x_target = float(np.quantile(band_xs, (col_idx + 0.5) / float(cols)))
            _, idx = band_tree.query(np.asarray([x_target, y_target], dtype=np.float64), k=1)
            point = (int(band_xs[idx]), int(band_ys[idx]))
            if point in used:
                continue
            used.add(point)
            selected.append((float(point[0]), float(point[1])))
            if len(selected) >= num_points:
                return np.asarray(selected, dtype=np.float32)

    tree = cKDTree(np.column_stack([xs, ys]).astype(np.float64))
    grid_cols = max(4, int(np.ceil(np.sqrt(num_points))))
    x_fracs = np.linspace(0.08, 0.92, grid_cols)
    y_fracs = np.linspace(0.04, 0.96, max(4, rows))
    targets = np.asarray(
        [[x0 + fx * (x1 - x0), y0 + fy * (y1 - y0)] for fy in y_fracs for fx in x_fracs],
        dtype=np.float64,
    )
    for target in targets:
        _, idx = tree.query(target, k=1)
        point = (int(xs[idx]), int(ys[idx]))
        if point in used:
            continue
        used.add(point)
        selected.append((float(point[0]), float(point[1])))
        if len(selected) >= num_points:
            break

    if len(selected) < num_points:
        rng = np.random.default_rng(13)
        order = rng.permutation(len(xs))
        for idx in order:
            point = (int(xs[idx]), int(ys[idx]))
            if point in used:
                continue
            used.add(point)
            selected.append((float(point[0]), float(point[1])))
            if len(selected) >= num_points:
                break

    if len(selected) < 3:
        raise ValueError(f"Only selected {len(selected)} screen query points.")
    return np.asarray(selected, dtype=np.float32)


def query_overlay_color(label: str) -> tuple[int, int, int]:
    if label == "surface":
        return (255, 230, 0)
    if label == "lid_edge":
        return (255, 80, 40)
    if label == "lid_edge_reseed":
        return (60, 230, 255)
    return QUERY_COLOR


def draw_query_overlay(
    rgb_path: Path,
    mask: np.ndarray,
    queries_xy: np.ndarray,
    output_path: Path,
    query_labels: np.ndarray | None = None,
) -> None:
    image = Image.open(rgb_path).convert("RGBA")
    alpha = Image.fromarray((mask.astype(np.uint8) * 70), mode="L")
    tint = Image.new("RGBA", image.size, (40, 255, 80, 0))
    tint.putalpha(alpha)
    image = Image.alpha_composite(image, tint)
    draw = ImageDraw.Draw(image)
    if query_labels is None:
        query_labels = np.full(len(queries_xy), "query", dtype="<U16")
    for idx, ((x, y), label) in enumerate(zip(queries_xy, query_labels)):
        color = query_overlay_color(str(label))
        r = 5 if len(queries_xy) > 36 else 8
        draw.ellipse((x - r, y - r, x + r, y + r), outline=color + (255,), width=3)
        if len(queries_xy) <= 48 or idx % 4 == 0:
            draw.text((x + 7, y - 7), str(idx), fill=color + (255,))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_path)


def load_rgb_frames(export_root: Path, frames: list[int], rgb_dir_name: str) -> list[np.ndarray]:
    out = []
    for frame in frames:
        path = export_root / rgb_dir_name / f"{frame_name(frame)}.png"
        if not path.exists():
            raise FileNotFoundError(f"Missing RGB frame: {path}")
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read RGB frame: {path}")
        out.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return out


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


def run_cotracker(
    frames_rgb: list[np.ndarray],
    queries_xy: np.ndarray,
    query_times: np.ndarray,
    cotracker_root: Path,
    checkpoint: Path,
    device: str,
    tracker_max_side: int,
    confidence_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if str(cotracker_root) not in sys.path:
        sys.path.insert(0, str(cotracker_root))
    from cotracker.models.core.model_utils import get_points_on_a_grid  # noqa: WPS433
    from cotracker.predictor import CoTrackerPredictor  # noqa: WPS433

    query_times = np.asarray(query_times, dtype=np.float32)
    if len(query_times) != len(queries_xy):
        raise ValueError(f"query_times length {len(query_times)} != queries length {len(queries_xy)}")

    video_np, scale = tracker_resize(frames_rgb, tracker_max_side)
    video = torch.from_numpy(video_np).permute(0, 3, 1, 2)[None].float().to(device)
    query_scaled = queries_xy.astype(np.float32) * float(scale)
    query_txy = np.column_stack([query_times, query_scaled]).astype(np.float32)
    queries = torch.from_numpy(query_txy)[None].to(device)

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

        tracks, raw_visibility, *_ = model.model.forward(video=video_interp, queries=model_queries, iters=6)
        tracks = tracks[:, :, :query_count]
        raw_visibility = raw_visibility[:, :, :query_count]

        for batch_idx in range(bsz):
            query_t = queries_model[batch_idx, :, 0].round().to(torch.int64).clamp(0, timesteps - 1)
            arange = torch.arange(0, len(query_t), device=video.device)
            tracks[batch_idx, query_t, arange] = queries_model[batch_idx, :, 1:]
            raw_visibility[batch_idx, query_t, arange] = 1.0

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
    frame_ids = np.arange(tracks_np.shape[0], dtype=np.float32)[:, None]
    before_start = frame_ids < query_times[None, :]
    visibility_np[before_start] = False
    confidence_np[before_start] = 0.0
    info = {
        "cotracker_root": str(cotracker_root),
        "checkpoint": str(checkpoint),
        "device": str(device),
        "input_original_shape_hw": list(frames_rgb[0].shape[:2]),
        "tracker_shape_thw": list(video_np.shape[:3]),
        "tracker_scale": float(scale),
        "confidence_threshold": float(confidence_threshold),
        "confidence_source": "raw_cotracker_visibility",
        "query_format": "(relative_frame_index, x, y)",
        "query_time_min_max": [float(query_times.min()), float(query_times.max())] if len(query_times) else [0.0, 0.0],
    }
    return tracks_np, visibility_np, confidence_np, info

def backproject_tracks(
    meta: dict[str, Any],
    export_root: Path,
    frames: list[int],
    tracks_xy: np.ndarray,
    visibility: np.ndarray,
    convention: str,
    depth_min_m: float,
    depth_max_m: float,
    radius_px: float,
    depth_dir_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    frame_points = np.full((len(frames), tracks_xy.shape[1], 3), np.nan, dtype=np.float32)
    align_points = np.full_like(frame_points, np.nan)
    neighbor_px = np.full((len(frames), tracks_xy.shape[1]), np.inf, dtype=np.float32)
    stats: list[dict[str, Any]] = []

    align_row = frame_row(export_root, frames[0])
    for local_idx, frame in enumerate(frames):
        depth_path = export_root / depth_dir_name / f"{frame_name(frame)}.meters.npy"
        if not depth_path.exists():
            raise FileNotFoundError(f"Missing depth frame: {depth_path}")
        depth_m = np.load(depth_path)
        points_right, u, v, inside = depth_points_in_right_camera(
            meta,
            depth_m,
            convention,
            depth_min_m=depth_min_m,
            depth_max_m=depth_max_m,
        )
        valid_depth_idx = np.flatnonzero(inside)
        if valid_depth_idx.size == 0:
            stats.append({"frame": int(frame), "valid_track_depth": 0, "projected_depth_points": 0})
            continue
        tree = cKDTree(np.column_stack([u[valid_depth_idx], v[valid_depth_idx]]).astype(np.float64))
        frame_row_i = frame_row(export_root, frame)
        t_align_from_frame = camera_to_camera_matrix(meta, frame_row_i, align_row, camera="right")

        valid_count = 0
        for point_idx, xy in enumerate(tracks_xy[local_idx]):
            if not visibility[local_idx, point_idx]:
                continue
            dist_px, depth_tree_idx = tree.query(xy.astype(np.float64), k=1)
            if not np.isfinite(dist_px) or dist_px > radius_px:
                continue
            source_idx = valid_depth_idx[int(depth_tree_idx)]
            point_frame = points_right[source_idx]
            frame_points[local_idx, point_idx] = point_frame.astype(np.float32)
            align_points[local_idx, point_idx] = transform_points(point_frame[None, :], t_align_from_frame)[0].astype(
                np.float32
            )
            neighbor_px[local_idx, point_idx] = float(dist_px)
            valid_count += 1
        stats.append(
            {
                "frame": int(frame),
                "valid_track_depth": int(valid_count),
                "visible_tracks": int(visibility[local_idx].sum()),
                "projected_depth_points": int(valid_depth_idx.size),
                "median_neighbor_px": float(np.nanmedian(neighbor_px[local_idx][np.isfinite(neighbor_px[local_idx])]))
                if valid_count
                else None,
            }
        )
    return frame_points, align_points, neighbor_px, stats


def signed_axis_angles(
    reference_points: np.ndarray,
    moving_points: np.ndarray,
    origin: np.ndarray,
    axis: np.ndarray,
) -> np.ndarray:
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    v0 = reference_points - origin
    v1 = moving_points - origin
    v0 = v0 - (v0 @ axis)[:, None] * axis[None, :]
    v1 = v1 - (v1 @ axis)[:, None] * axis[None, :]
    r0 = np.linalg.norm(v0, axis=1)
    r1 = np.linalg.norm(v1, axis=1)
    valid = (r0 > 1e-4) & (r1 > 1e-4)
    out = np.full(len(reference_points), np.nan, dtype=np.float64)
    cross = np.cross(v0[valid], v1[valid])
    sin_t = cross @ axis
    cos_t = np.sum(v0[valid] * v1[valid], axis=1) / (r0[valid] * r1[valid] + 1e-12)
    out[valid] = np.arctan2(sin_t, np.clip(cos_t, -1.0, 1.0))
    return out


def kabsch_rotation(reference_points: np.ndarray, moving_points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ref_center = reference_points.mean(axis=0)
    mov_center = moving_points.mean(axis=0)
    ref_centered = reference_points - ref_center
    mov_centered = moving_points - mov_center
    cov = ref_centered.T @ mov_centered
    u, singular, vt = np.linalg.svd(cov)
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0.0:
        vt[-1, :] *= -1.0
        rot = vt.T @ u.T
    trans = mov_center - rot @ ref_center
    return rot, trans, singular


def axis_twist_angle(rot: np.ndarray, axis: np.ndarray) -> float:
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    basis = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(basis @ axis)) > 0.9:
        basis = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    basis = basis - axis * float(basis @ axis)
    basis /= np.linalg.norm(basis) + 1e-12
    other = np.cross(axis, basis)
    other /= np.linalg.norm(other) + 1e-12

    rot_basis = rot @ basis
    rot_other = rot @ other
    cos_t = float(rot_basis @ basis + rot_other @ other)
    sin_t = float(rot_basis @ other - rot_other @ basis)
    return float(np.arctan2(sin_t, cos_t))


def robust_kabsch_axis_angle(
    reference_points: np.ndarray,
    moving_points: np.ndarray,
    axis: np.ndarray,
    min_points: int,
    trim_fraction: float,
) -> dict[str, Any]:
    rot, trans, singular = kabsch_rotation(reference_points, moving_points)
    fitted = reference_points @ rot.T + trans
    residual = np.linalg.norm(fitted - moving_points, axis=1)
    keep_count = max(int(np.ceil(len(reference_points) * trim_fraction)), min_points)
    keep_count = min(max(keep_count, min_points), len(reference_points))
    keep_idx = np.argsort(residual)[:keep_count]
    if keep_count < len(reference_points) and keep_count >= min_points:
        rot, trans, singular = kabsch_rotation(reference_points[keep_idx], moving_points[keep_idx])
        fitted = reference_points[keep_idx] @ rot.T + trans
        residual_used = np.linalg.norm(fitted - moving_points[keep_idx], axis=1)
    else:
        residual_used = residual
    angle = axis_twist_angle(rot, axis)
    return {
        "angle_rad": float(angle),
        "used_points": int(keep_count),
        "kabsch_singular_values": singular.tolist(),
        "residual_median_m": float(np.median(residual_used)),
        "residual_q90_m": float(np.quantile(residual_used, 0.90)),
    }


def trimmed_mean(values: np.ndarray, fraction: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("inf")
    keep = max(1, int(np.ceil(values.size * float(np.clip(fraction, 0.0, 1.0)))))
    return float(np.mean(np.partition(values, keep - 1)[:keep]))


def hinge_1d_score(
    theta_rad: float,
    reference_angle_rad: float,
    ref_points_align: np.ndarray,
    observed_points_align: np.ndarray,
    observed_tracks_xy: np.ndarray,
    meta: dict[str, Any],
    align_to_current: np.ndarray,
    origin: np.ndarray,
    axis: np.ndarray,
    trim_fraction: float,
    depth_weight_px_per_m: float,
    continuity_weight_px_per_deg: float,
    previous_angle_rad: float,
) -> tuple[float, dict[str, float]]:
    pred_align = rotate_points_about_axis(ref_points_align, origin, axis, theta_rad - reference_angle_rad)
    pred_cam = transform_points(pred_align, align_to_current)
    u, v, z = project_right_camera_points(meta, pred_cam)
    valid = (z > 1e-6) & np.isfinite(u) & np.isfinite(v)
    if valid.sum() < 4:
        return float("inf"), {"projected_count": int(valid.sum())}
    reproj = np.linalg.norm(np.stack([u[valid], v[valid]], axis=1) - observed_tracks_xy[valid], axis=1)
    reproj_score = trimmed_mean(reproj, trim_fraction)
    depth_score_m = trimmed_mean(np.linalg.norm(pred_align[valid] - observed_points_align[valid], axis=1), trim_fraction)
    continuity_score = abs(float(np.rad2deg(theta_rad - previous_angle_rad))) * float(continuity_weight_px_per_deg)
    score = reproj_score + float(depth_weight_px_per_m) * depth_score_m + continuity_score
    return score, {
        "projected_count": int(valid.sum()),
        "reprojection_trimmed_px": float(reproj_score),
        "depth_trimmed_m": float(depth_score_m),
        "continuity_score": float(continuity_score),
    }


def optimize_hinge_angle_1d(
    ref_points_align: np.ndarray,
    observed_points_align: np.ndarray,
    observed_tracks_xy: np.ndarray,
    meta: dict[str, Any],
    align_to_current: np.ndarray,
    origin: np.ndarray,
    axis: np.ndarray,
    previous_angle_rad: float,
    reference_angle_rad: float,
    min_angle_rad: float,
    max_angle_rad: float,
    coarse_steps: int,
    fine_window_rad: float,
    fine_steps: int,
    trim_fraction: float,
    depth_weight_px_per_m: float,
    continuity_weight_px_per_deg: float,
) -> dict[str, Any]:
    def eval_theta(theta: float) -> tuple[float, dict[str, float]]:
        return hinge_1d_score(
            theta,
            reference_angle_rad,
            ref_points_align,
            observed_points_align,
            observed_tracks_xy,
            meta,
            align_to_current,
            origin,
            axis,
            trim_fraction,
            depth_weight_px_per_m,
            continuity_weight_px_per_deg,
            previous_angle_rad,
        )

    coarse_steps = max(3, int(coarse_steps))
    fine_steps = max(3, int(fine_steps))
    coarse = np.linspace(min_angle_rad, max_angle_rad, coarse_steps)
    best_theta = float(previous_angle_rad)
    best_score = float("inf")
    best_metrics: dict[str, float] = {}
    for theta in coarse:
        score, metrics = eval_theta(float(theta))
        if score < best_score:
            best_score = float(score)
            best_theta = float(theta)
            best_metrics = metrics

    lo = max(min_angle_rad, best_theta - fine_window_rad)
    hi = min(max_angle_rad, best_theta + fine_window_rad)
    fine = np.linspace(lo, hi, fine_steps)
    for theta in fine:
        score, metrics = eval_theta(float(theta))
        if score < best_score:
            best_score = float(score)
            best_theta = float(theta)
            best_metrics = metrics

    return {
        "angle_rad": best_theta,
        "angle_deg": float(np.rad2deg(best_theta)),
        "score": best_score,
        **best_metrics,
    }


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cutoff = 0.5 * float(sorted_weights.sum())
    return float(sorted_values[np.searchsorted(np.cumsum(sorted_weights), cutoff, side="left")])


def estimate_angles(
    frames: list[int],
    align_points: np.ndarray,
    tracks_xy: np.ndarray,
    visibility: np.ndarray,
    query_start_frames: np.ndarray,
    query_base_angles: np.ndarray,
    joint: dict[str, Any],
    meta: dict[str, Any],
    export_root: Path,
    min_points: int,
    max_abs_angle_deg: float,
    method: str,
    trim_fraction: float,
    hinge_angle_min_deg: float,
    hinge_angle_max_deg: float,
    hinge_angle_coarse_steps: int,
    hinge_angle_fine_window_deg: float,
    hinge_angle_fine_steps: int,
    hinge_depth_weight_px_per_m: float,
    hinge_continuity_weight_px_per_deg: float,
    query_labels: np.ndarray | None = None,
    monotonic_hinge: bool = True,
    monotonic_backoff_deg: float = 1.5,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    origin = np.asarray(joint["origin_xyz"], dtype=np.float64)
    axis = np.asarray(joint["axis_xyz"], dtype=np.float64)
    axis /= np.linalg.norm(axis) + 1e-12

    query_start_frames = np.asarray(query_start_frames, dtype=np.int64)
    query_base_angles = np.asarray(query_base_angles, dtype=np.float64)
    if len(query_start_frames) != align_points.shape[1]:
        raise ValueError("query_start_frames must match number of tracked points")
    if len(query_base_angles) != align_points.shape[1]:
        raise ValueError("query_base_angles must match number of tracked points")
    if query_labels is None:
        query_labels_arr = np.full(align_points.shape[1], "tracked", dtype="<U24")
    else:
        query_labels_arr = np.asarray(query_labels).astype(str)
        if len(query_labels_arr) != align_points.shape[1]:
            raise ValueError("query_labels must match number of tracked points")
    query_group_labels = np.asarray(
        ["lid_edge" if label.startswith("lid_edge") else label for label in query_labels_arr],
        dtype="<U24",
    )

    max_abs = np.deg2rad(float(max_abs_angle_deg))
    min_angle_rad = max(np.deg2rad(float(hinge_angle_min_deg)), -max_abs)
    max_angle_rad = min(np.deg2rad(float(hinge_angle_max_deg)), max_abs)
    backoff_rad = max(0.0, float(np.deg2rad(monotonic_backoff_deg)))
    frame_rows = [frame_row(export_root, frame) for frame in frames]
    align_row = frame_rows[0]
    align_to_current = [camera_to_camera_matrix(meta, align_row, row, camera="right") for row in frame_rows]
    angles = np.zeros(len(frames), dtype=np.float64)
    diagnostics: list[dict[str, Any]] = []
    previous_angle = 0.0
    group_keys = sorted(
        {(int(start), str(label)) for start, label in zip(query_start_frames, query_group_labels)},
        key=lambda item: (item[0], item[1]),
    )

    for local_idx, frame in enumerate(frames):
        if local_idx == 0:
            diagnostics.append(
                {
                    "frame": int(frame),
                    "angle_rad": 0.0,
                    "angle_deg": 0.0,
                    "valid_points": int(valid_track_mask(align_points, visibility, local_idx).sum()),
                    "status": "estimated",
                    "method": "query_grouped_" + method,
                    "candidate_groups": [],
                }
            )
            continue

        candidates: list[float] = []
        weights: list[float] = []
        candidate_groups: list[dict[str, Any]] = []
        all_valid_indices: list[int] = []

        known_here = query_base_angles[query_start_frames == local_idx]
        known_here = known_here[np.isfinite(known_here)]
        if local_idx > 0 and known_here.size:
            known_angle = float(np.median(known_here))
            known_lower_bound = max(min_angle_rad, previous_angle - backoff_rad) if monotonic_hinge else min_angle_rad
            known_angle = float(np.clip(known_angle, known_lower_bound, max_angle_rad))
            candidates.append(known_angle)
            weights.append(float(max(known_here.size, 1)))
            candidate_groups.append(
                {
                    "start_frame": int(frame),
                    "start_local_index": int(local_idx),
                    "candidate_angle_rad": known_angle,
                    "candidate_angle_deg": float(np.rad2deg(known_angle)),
                    "valid_points": int(known_here.size),
                    "status": "known_query_start_angle",
                }
            )

        for start_frame, group_label in group_keys:
            if start_frame >= local_idx:
                continue
            if start_frame < 0 or start_frame >= len(frames):
                continue
            group_mask = (query_start_frames == start_frame) & (query_group_labels == group_label)
            ref = align_points[start_frame].astype(np.float64)
            cur = align_points[local_idx].astype(np.float64)
            valid = (
                group_mask
                & visibility[start_frame]
                & visibility[local_idx]
                & np.isfinite(ref).all(axis=1)
                & np.isfinite(cur).all(axis=1)
            )
            valid_idx = np.flatnonzero(valid)
            if valid_idx.size < max(1, min_points):
                if valid_idx.size:
                    all_valid_indices.extend(valid_idx.tolist())
                continue

            ref_valid_points = ref[valid_idx]
            cur_valid_points = cur[valid_idx]
            per_point = signed_axis_angles(ref_valid_points, cur_valid_points, origin, axis)
            per_point = per_point[np.isfinite(per_point)]
            if method == "hinge_1d":
                base_angles = query_base_angles[group_mask]
                base_angles = base_angles[np.isfinite(base_angles)]
                group_base_angle = float(np.median(base_angles)) if base_angles.size else float(angles[start_frame])
                estimate = optimize_hinge_angle_1d(
                    ref_valid_points,
                    cur_valid_points,
                    tracks_xy[local_idx, valid_idx].astype(np.float64),
                    meta,
                    align_to_current[local_idx],
                    origin,
                    axis,
                    previous_angle_rad=previous_angle,
                    reference_angle_rad=group_base_angle,
                    min_angle_rad=min_angle_rad,
                    max_angle_rad=max_angle_rad,
                    coarse_steps=hinge_angle_coarse_steps,
                    fine_window_rad=np.deg2rad(float(hinge_angle_fine_window_deg)),
                    fine_steps=hinge_angle_fine_steps,
                    trim_fraction=float(np.clip(trim_fraction, 0.0, 1.0)),
                    depth_weight_px_per_m=hinge_depth_weight_px_per_m,
                    continuity_weight_px_per_deg=hinge_continuity_weight_px_per_deg,
                )
                candidate = float(estimate["angle_rad"])
                delta = candidate - group_base_angle
                used_points = int(estimate.get("projected_count", valid_idx.size))
            elif method == "kabsch_axis":
                estimate = robust_kabsch_axis_angle(
                    ref_valid_points,
                    cur_valid_points,
                    axis,
                    min_points=max(1, min_points),
                    trim_fraction=float(np.clip(trim_fraction, 0.0, 1.0)),
                )
                delta = float(estimate["angle_rad"])
                used_points = int(estimate["used_points"])
            else:
                if per_point.size == 0:
                    continue
                median = float(np.median(per_point))
                mad = float(np.median(np.abs(per_point - median)))
                keep = np.abs(per_point - median) <= max(np.deg2rad(10.0), 2.5 * mad)
                robust = per_point[keep] if keep.any() else per_point
                delta = float(np.median(robust))
                used_points = int(robust.size)
                estimate = {"used_points": used_points}

            if method != "hinge_1d":
                group_base_angles = query_base_angles[group_mask]
                group_base_angles = group_base_angles[np.isfinite(group_base_angles)]
                base_angle = float(np.median(group_base_angles)) if group_base_angles.size else float(angles[start_frame])
                candidate = float(base_angle + delta)
            candidate = candidate + 2.0 * np.pi * round((previous_angle - candidate) / (2.0 * np.pi))
            lower_bound = max(min_angle_rad, previous_angle - backoff_rad) if monotonic_hinge else min_angle_rad
            candidate = float(np.clip(candidate, lower_bound, max_angle_rad))
            candidate_weight = max(float(used_points), 1.0)
            if method == "hinge_1d":
                candidate_score = float(estimate.get("score", np.inf))
                candidate_weight = max(candidate_weight / max(candidate_score, 1.0), 1e-3)
            candidates.append(candidate)
            weights.append(candidate_weight)
            all_valid_indices.extend(valid_idx.tolist())
            candidate_groups.append(
                {
                    "start_frame": int(frames[start_frame]),
                    "start_local_index": int(start_frame),
                    "track_label": str(group_label),
                    "candidate_angle_rad": candidate,
                    "candidate_angle_deg": float(np.rad2deg(candidate)),
                    "delta_angle_deg": float(np.rad2deg(delta)),
                    "valid_points": int(valid_idx.size),
                    "candidate_weight": float(candidate_weight),
                    "valid_point_indices": valid_idx.tolist(),
                    "per_point_angle_deg": np.rad2deg(per_point).tolist(),
                    **estimate,
                }
            )

        if not candidates:
            angles[local_idx] = previous_angle
            diagnostics.append(
                {
                    "frame": int(frame),
                    "angle_rad": float(previous_angle),
                    "angle_deg": float(np.rad2deg(previous_angle)),
                    "valid_points": int(len(set(all_valid_indices))),
                    "valid_point_indices": sorted(set(all_valid_indices)),
                    "status": "carried_previous_not_enough_points",
                    "candidate_groups": candidate_groups,
                }
            )
            continue

        angle = weighted_median(np.asarray(candidates, dtype=np.float64), np.asarray(weights, dtype=np.float64))
        lower_bound = max(min_angle_rad, previous_angle - backoff_rad) if monotonic_hinge else min_angle_rad
        angle = float(np.clip(angle, lower_bound, max_angle_rad))
        angles[local_idx] = angle
        previous_angle = angle
        diagnostics.append(
            {
                "frame": int(frame),
                "angle_rad": angle,
                "angle_deg": float(np.rad2deg(angle)),
                "valid_points": int(len(set(all_valid_indices))),
                "valid_point_indices": sorted(set(all_valid_indices)),
                "status": "estimated",
                "method": "query_grouped_" + method,
                "candidate_groups": candidate_groups,
            }
        )

    return angles, diagnostics

def smooth_single_frame_angle_spikes(
    angles_rad: np.ndarray,
    diagnostics: list[dict[str, Any]],
    threshold_deg: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if len(angles_rad) < 3:
        return angles_rad, diagnostics
    out = angles_rad.copy()
    threshold = np.deg2rad(float(threshold_deg))
    for idx in range(1, len(out) - 1):
        prev_angle = out[idx - 1]
        next_angle = out[idx + 1]
        current = out[idx]
        if abs(current - prev_angle) <= threshold:
            continue
        if abs(current - next_angle) <= threshold:
            continue
        if abs(next_angle - prev_angle) > threshold:
            continue
        replacement = 0.5 * (prev_angle + next_angle)
        diagnostics[idx]["raw_angle_rad_before_smoothing"] = float(current)
        diagnostics[idx]["raw_angle_deg_before_smoothing"] = float(np.rad2deg(current))
        diagnostics[idx]["angle_rad"] = float(replacement)
        diagnostics[idx]["angle_deg"] = float(np.rad2deg(replacement))
        diagnostics[idx]["temporal_smoothing"] = "single_frame_spike_interpolated"
        out[idx] = replacement
    return out, diagnostics


def save_track_overlay(
    rgb_path: Path,
    tracks_xy: np.ndarray,
    visibility: np.ndarray,
    output_path: Path,
) -> None:
    image = Image.open(rgb_path).convert("RGBA")
    draw = ImageDraw.Draw(image)
    for idx, (xy, visible) in enumerate(zip(tracks_xy, visibility)):
        if not visible:
            continue
        color = tuple(int(c) for c in TRACK_COLORS[idx % len(TRACK_COLORS)])
        x, y = float(xy[0]), float(xy[1])
        r = 5
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color + (230,), outline=(0, 0, 0, 220), width=1)
        draw.text((x + 7, y - 7), str(idx), fill=color + (255,))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_path)


def export_dynamic_meshes(
    meta: dict[str, Any],
    export_root: Path,
    alignment_dir: Path,
    output_dir: Path,
    frames: list[int],
    angles_rad: np.ndarray,
    joint: dict[str, Any],
    tracks_xy: np.ndarray,
    visibility: np.ndarray,
    frame_points: np.ndarray,
    rgb_dir_name: str,
) -> list[dict[str, Any]]:
    base_mesh0 = load_mesh(alignment_dir / f"part_{BASE_PART_LABEL}_camera.obj")
    screen_mesh0 = load_mesh(alignment_dir / f"part_{SCREEN_PART_LABEL}_camera.obj")
    origin = np.asarray(joint["origin_xyz"], dtype=np.float64)
    axis = np.asarray(joint["axis_xyz"], dtype=np.float64)
    axis /= np.linalg.norm(axis) + 1e-12
    align_row = frame_row(export_root, frames[0])

    frame_entries: list[dict[str, Any]] = []
    for local_idx, frame in enumerate(frames):
        frame_dir = output_dir / f"frame_{frame_name(frame)}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        view_row = frame_row(export_root, frame)
        t_frame_from_align = camera_to_camera_matrix(meta, align_row, view_row, camera="right")

        screen_rot_align = rotate_mesh_about_axis(screen_mesh0, origin, axis, float(angles_rad[local_idx]))
        base_frame = apply_se3_to_mesh(base_mesh0, t_frame_from_align)
        screen_frame = apply_se3_to_mesh(screen_rot_align, t_frame_from_align)
        joint_frame = transform_joint_se3(joint, t_frame_from_align)

        base_path = frame_dir / f"part_{BASE_PART_LABEL}_dynamic.obj"
        screen_path = frame_dir / f"part_{SCREEN_PART_LABEL}_dynamic.obj"
        base_frame.export(base_path)
        screen_frame.export(screen_path)
        joint_path = write_json(frame_dir / "joint_dynamic.json", {"joints": [joint_frame]})

        valid_track = np.isfinite(frame_points[local_idx]).all(axis=1) & visibility[local_idx]
        track_points_path = None
        if valid_track.any():
            colors = TRACK_COLORS[np.arange(valid_track.sum()) % len(TRACK_COLORS)]
            track_points_path = frame_dir / "tracked_screen_points_camera.ply"
            trimesh.PointCloud(frame_points[local_idx, valid_track], colors=colors).export(track_points_path)

        overlay_path = frame_dir / "cotracker_tracks_overlay.png"
        save_track_overlay(
            export_root / rgb_dir_name / f"{frame_name(frame)}.png",
            tracks_xy[local_idx],
            visibility[local_idx],
            overlay_path,
        )

        frame_entries.append(
            {
                "frame": int(frame),
                "angle_rad": float(angles_rad[local_idx]),
                "angle_deg": float(np.rad2deg(angles_rad[local_idx])),
                "base_mesh": str(base_path),
                "screen_mesh": str(screen_path),
                "joint_json": str(joint_path),
                "track_points_ply": str(track_points_path) if track_points_path else None,
                "track_overlay": str(overlay_path),
                "valid_track_points": int(valid_track.sum()),
            }
        )
    return frame_entries


def save_mesh_projection_overlay(
    meta: dict[str, Any],
    rgb_path: Path,
    mesh_paths: list[Path],
    output_path: Path,
    colors: list[tuple[int, int, int, int]],
    samples_per_mesh: int = 12000,
) -> None:
    image = Image.open(rgb_path).convert("RGBA")
    draw = ImageDraw.Draw(image)
    rng = np.random.default_rng(19)
    for mesh_path, color in zip(mesh_paths, colors):
        mesh = load_mesh(mesh_path)
        count = min(samples_per_mesh, max(1000, len(mesh.faces)))
        points, _ = trimesh.sample.sample_surface(mesh, count, seed=int(rng.integers(0, 2**31 - 1)))
        u, v, z = project_right_camera_points(meta, points)
        inside = (z > 1e-6) & (u >= 0) & (u < image.width) & (v >= 0) & (v < image.height)
        for x, y in zip(u[inside], v[inside]):
            draw.ellipse((x - 1.0, y - 1.0, x + 1.0, y + 1.0), fill=color)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_path)


def valid_track_mask(frame_points: np.ndarray, visibility: np.ndarray, frame_idx: int) -> np.ndarray:
    return visibility[frame_idx] & np.isfinite(frame_points[frame_idx]).all(axis=1)


def project_screen_mesh_mask(
    meta: dict[str, Any],
    screen_mesh0: trimesh.Trimesh,
    joint: dict[str, Any],
    export_root: Path,
    frames: list[int],
    local_frame_idx: int,
    angle_rad: float,
    image_shape_hw: tuple[int, int],
    dilation_px: int,
) -> np.ndarray:
    height, width = image_shape_hw
    origin = np.asarray(joint["origin_xyz"], dtype=np.float64)
    axis = np.asarray(joint["axis_xyz"], dtype=np.float64)
    axis /= np.linalg.norm(axis) + 1e-12
    vertices_align = rotate_points_about_axis(screen_mesh0.vertices.astype(np.float64), origin, axis, angle_rad)
    align_row = frame_row(export_root, frames[0])
    view_row = frame_row(export_root, frames[local_frame_idx])
    t_frame_from_align = camera_to_camera_matrix(meta, align_row, view_row, camera="right")
    vertices_frame = transform_points(vertices_align, t_frame_from_align)
    u, v, z = project_right_camera_points(meta, vertices_frame)
    inside = (z > 1e-6) & np.isfinite(u) & np.isfinite(v)
    if int(inside.sum()) < 3:
        return np.zeros((height, width), dtype=bool)
    points = np.column_stack([u[inside], v[inside]]).astype(np.float32)
    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)
    hull = cv2.convexHull(points.reshape(-1, 1, 2)).astype(np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    dilation_px = max(0, int(dilation_px))
    if dilation_px:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilation_px + 1, 2 * dilation_px + 1))
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask.astype(bool)


def select_reseed_points(
    frame_rgb: np.ndarray,
    tracks_xy: np.ndarray,
    valid_mask: np.ndarray,
    target_points: int,
    min_existing_points: int,
    roi_pad_px: float,
    min_distance_px: float,
    candidate_mask: np.ndarray | None = None,
) -> np.ndarray:
    existing = tracks_xy[valid_mask]
    if len(existing) < min_existing_points:
        return np.zeros((0, 2), dtype=np.float32)

    height, width = frame_rgb.shape[:2]
    if candidate_mask is not None and candidate_mask.shape != (height, width):
        raise ValueError("candidate_mask must match the frame image shape")
    if candidate_mask is not None and candidate_mask.any():
        mys, mxs = np.nonzero(candidate_mask)
        x0 = max(0, int(np.floor(mxs.min() - roi_pad_px * 0.25)))
        x1 = min(width - 1, int(np.ceil(mxs.max() + roi_pad_px * 0.25)))
        y0 = max(0, int(np.floor(mys.min() - roi_pad_px * 0.25)))
        y1 = min(height - 1, int(np.ceil(mys.max() + roi_pad_px * 0.25)))
    else:
        x0 = max(0, int(np.floor(existing[:, 0].min() - roi_pad_px)))
        x1 = min(width - 1, int(np.ceil(existing[:, 0].max() + roi_pad_px)))
        y0 = max(0, int(np.floor(existing[:, 1].min() - roi_pad_px * 0.7)))
        y1 = min(height - 1, int(np.ceil(existing[:, 1].max() + roi_pad_px * 0.7)))
    if x1 <= x0 or y1 <= y0:
        return np.zeros((0, 2), dtype=np.float32)

    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    roi = gray[y0 : y1 + 1, x0 : x1 + 1]
    roi_candidate_mask = None
    if candidate_mask is not None and candidate_mask.any():
        roi_candidate_mask = candidate_mask[y0 : y1 + 1, x0 : x1 + 1]
    corners = cv2.goodFeaturesToTrack(
        roi,
        maxCorners=max(target_points * 6, target_points),
        qualityLevel=0.006,
        minDistance=max(4.0, float(min_distance_px) * 0.65),
        blockSize=5,
        useHarrisDetector=False,
    )

    candidates: list[tuple[float, float, float]] = []
    if corners is not None:
        gx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx * gx + gy * gy)
        for corner in corners.reshape(-1, 2):
            x = float(corner[0] + x0)
            y = float(corner[1] + y0)
            lx = int(np.clip(round(corner[0]), 0, roi.shape[1] - 1))
            ly = int(np.clip(round(corner[1]), 0, roi.shape[0] - 1))
            if roi_candidate_mask is not None and not bool(roi_candidate_mask[ly, lx]):
                continue
            candidates.append((x, y, float(grad[ly, lx])))

    if len(candidates) < target_points:
        edges = cv2.Canny(roi, 60, 140)
        if roi_candidate_mask is not None:
            edges = np.where(roi_candidate_mask, edges, 0).astype(np.uint8)
        ey, ex = np.nonzero(edges)
        if len(ex):
            order = np.linspace(0, len(ex) - 1, min(len(ex), target_points * 8)).round().astype(np.int64)
            for idx in order:
                x = float(ex[idx] + x0)
                y = float(ey[idx] + y0)
                candidates.append((x, y, 1.0))

    if not candidates:
        return np.zeros((0, 2), dtype=np.float32)

    candidates.sort(key=lambda item: item[2], reverse=True)
    selected: list[tuple[float, float]] = []
    existing_tree = cKDTree(existing.astype(np.float64)) if len(existing) else None
    for x, y, _score in candidates:
        point = np.asarray([x, y], dtype=np.float64)
        if existing_tree is not None:
            dist, _ = existing_tree.query(point, k=1)
            if dist < min_distance_px * 0.5:
                continue
        if selected:
            selected_arr = np.asarray(selected, dtype=np.float64)
            if np.min(np.linalg.norm(selected_arr - point[None, :], axis=1)) < min_distance_px:
                continue
        selected.append((float(x), float(y)))
        if len(selected) >= target_points:
            break

    return np.asarray(selected, dtype=np.float32)


def find_reseed_frame(
    frame_points: np.ndarray,
    visibility: np.ndarray,
    trigger_points: int,
    min_existing_points: int,
    used_frames: set[int],
    point_mask: np.ndarray | None = None,
) -> int | None:
    if point_mask is None:
        point_mask = np.ones(frame_points.shape[1], dtype=bool)
    point_mask = np.asarray(point_mask, dtype=bool)
    if len(point_mask) != frame_points.shape[1]:
        raise ValueError("point_mask must match the tracked point count")
    counts = np.asarray(
        [int((valid_track_mask(frame_points, visibility, idx) & point_mask).sum()) for idx in range(len(frame_points))]
    )
    for idx in range(1, len(counts) - 1):
        if idx in used_frames:
            continue
        if counts[idx] < trigger_points and counts[idx] >= min_existing_points and counts[idx - 1] >= trigger_points:
            return int(idx)
    return None


def main() -> None:
    args = parse_args()
    alignment_dir = args.alignment_dir.resolve()
    export_root = args.export_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.end_frame < args.start_frame:
        raise ValueError("--end-frame must be >= --start-frame")
    frames = list(range(args.start_frame, args.end_frame + 1))
    result = read_json(alignment_dir / "alignment_result.json")
    meta = read_json(export_root / "manifest.json")
    convention = args.depth_convention or result.get("convention_used", "camera_to_rig")
    joints = read_json(alignment_dir / "joint_camera.json").get("joints", [])
    if not joints:
        raise ValueError(f"No joint found in {alignment_dir / 'joint_camera.json'}")
    joint = joints[0]

    screen_mask = load_screen_mask(alignment_dir, result)
    query_components: list[dict[str, Any]] = []
    if args.query_mode == "dual":
        surface_queries = choose_query_points(
            screen_mask,
            args.surface_query_points,
            args.query_margin_px,
            args.top_band_ratio,
            args.top_band_weight,
            "top_weighted",
            args.edge_top_ratio,
            args.edge_rows,
        )
        edge_queries = choose_query_points(
            screen_mask,
            args.edge_query_points,
            args.query_margin_px,
            args.top_band_ratio,
            args.top_band_weight,
            "lid_edge",
            args.edge_top_ratio,
            args.edge_rows,
        )
        queries_xy = np.vstack([surface_queries, edge_queries]).astype(np.float32)
        query_labels = np.asarray(
            ["surface"] * len(surface_queries) + ["lid_edge"] * len(edge_queries),
            dtype="<U24",
        )
        query_components.extend(
            [
                {"label": "surface", "mode": "top_weighted", "count": int(len(surface_queries))},
                {"label": "lid_edge", "mode": "lid_edge", "count": int(len(edge_queries))},
            ]
        )
    else:
        queries_xy = choose_query_points(
            screen_mask,
            args.num_query_points,
            args.query_margin_px,
            args.top_band_ratio,
            args.top_band_weight,
            args.query_mode,
            args.edge_top_ratio,
            args.edge_rows,
        ).astype(np.float32)
        label = "lid_edge" if args.query_mode == "lid_edge" else "surface"
        query_labels = np.full(len(queries_xy), label, dtype="<U24")
        query_components.append({"label": label, "mode": args.query_mode, "count": int(len(queries_xy))})

    query_overlay = output_dir / "query_points_frame0.png"
    draw_query_overlay(
        export_root / args.rgb_dir_name / f"{frame_name(frames[0])}.png",
        screen_mask,
        queries_xy,
        query_overlay,
        query_labels,
    )

    component_text = ", ".join(f"{item['label']}={item['count']}" for item in query_components)
    print(f"Selected {len(queries_xy)} initial screen query points ({component_text}).")
    frames_rgb = load_rgb_frames(export_root, frames, args.rgb_dir_name)
    screen_mesh0_for_reseed = load_mesh(alignment_dir / f"part_{SCREEN_PART_LABEL}_camera.obj")
    query_start_frames = np.zeros(len(queries_xy), dtype=np.int64)
    query_base_angles = np.zeros(len(queries_xy), dtype=np.float64)
    reseed_events: list[dict[str, Any]] = []
    used_reseed_frames: set[int] = set()

    tracks_xy = visibility = track_confidence = frame_points = align_points = depth_neighbor_px = None
    depth_stats: list[dict[str, Any]] = []
    tracker_info: dict[str, Any] = {}
    for pass_idx in range(max(1, int(args.reseed_max_events) + 1)):
        print(
            f"Running CoTracker3 pass {pass_idx + 1} on frames {frames[0]}-{frames[-1]} "
            f"with {len(queries_xy)} queries..."
        )
        tracks_xy, visibility, track_confidence, tracker_info = run_cotracker(
            frames_rgb,
            queries_xy,
            query_start_frames,
            args.cotracker_root.resolve(),
            args.cotracker_checkpoint.resolve(),
            args.device,
            args.tracker_max_side,
            args.cotracker_conf_threshold,
        )

        print("Back-projecting tracked points with RGB-D...")
        frame_points, align_points, depth_neighbor_px, depth_stats = backproject_tracks(
            meta,
            export_root,
            frames,
            tracks_xy,
            visibility,
            convention,
            args.depth_min_m,
            args.depth_max_m,
            args.depth_neighbor_radius_px,
            args.depth_dir_name,
        )

        if not args.enable_reseed or pass_idx >= int(args.reseed_max_events):
            break
        pass_angles, _pass_diagnostics = estimate_angles(
            frames,
            align_points,
            tracks_xy,
            visibility,
            query_start_frames,
            query_base_angles,
            joint,
            meta,
            export_root,
            args.min_angle_points,
            args.max_abs_angle_deg,
            args.angle_method,
            args.angle_trim_fraction,
            args.hinge_angle_min_deg,
            args.hinge_angle_max_deg,
            args.hinge_angle_coarse_steps,
            args.hinge_angle_fine_window_deg,
            args.hinge_angle_fine_steps,
            args.hinge_depth_weight_px_per_m,
            args.hinge_continuity_weight_px_per_deg,
            query_labels,
            args.monotonic_hinge,
            args.monotonic_backoff_deg,
        )
        reseed_point_mask = np.char.startswith(query_labels.astype(str), "lid_edge")
        reseed_frame = find_reseed_frame(
            frame_points,
            visibility,
            args.reseed_trigger_points,
            args.reseed_min_existing_points,
            used_reseed_frames,
            reseed_point_mask,
        )
        if reseed_frame is None:
            break
        valid_mask = valid_track_mask(frame_points, visibility, reseed_frame) & reseed_point_mask
        new_queries = select_reseed_points(
            frames_rgb[reseed_frame],
            tracks_xy[reseed_frame],
            valid_mask,
            args.reseed_target_points,
            args.reseed_min_existing_points,
            args.reseed_roi_pad_px,
            args.reseed_min_distance_px,
            project_screen_mesh_mask(
                meta,
                screen_mesh0_for_reseed,
                joint,
                export_root,
                frames,
                reseed_frame,
                float(pass_angles[reseed_frame]),
                frames_rgb[reseed_frame].shape[:2],
                args.reseed_projected_mask_dilation_px,
            )
            if args.reseed_use_projected_screen_mask
            else None,
        )
        used_reseed_frames.add(reseed_frame)
        if len(new_queries) < args.reseed_min_existing_points:
            reseed_events.append(
                {
                    "pass": int(pass_idx + 1),
                    "frame": int(frames[reseed_frame]),
                    "local_frame": int(reseed_frame),
                    "selected_points": int(len(new_queries)),
                    "status": "skipped_not_enough_new_points",
                }
            )
            break
        reseed_base_angle = float(pass_angles[reseed_frame])
        queries_xy = np.vstack([queries_xy, new_queries.astype(np.float32)])
        query_labels = np.concatenate(
            [query_labels, np.full(len(new_queries), "lid_edge_reseed", dtype="<U24")]
        )
        query_start_frames = np.concatenate(
            [query_start_frames, np.full(len(new_queries), reseed_frame, dtype=np.int64)]
        )
        query_base_angles = np.concatenate(
            [query_base_angles, np.full(len(new_queries), reseed_base_angle, dtype=np.float64)]
        )
        reseed_events.append(
            {
                "pass": int(pass_idx + 1),
                "frame": int(frames[reseed_frame]),
                "local_frame": int(reseed_frame),
                "selected_points": int(len(new_queries)),
                "existing_valid_points": int(valid_mask.sum()),
                "base_angle_rad": reseed_base_angle,
                "base_angle_deg": float(np.rad2deg(reseed_base_angle)),
                "status": "added",
            }
        )

    assert tracks_xy is not None and visibility is not None and track_confidence is not None
    assert frame_points is not None and align_points is not None and depth_neighbor_px is not None
    np.save(output_dir / "tracks_2d_xy.npy", tracks_xy)
    np.save(output_dir / "tracks_visibility.npy", visibility)
    np.save(output_dir / "tracks_confidence.npy", track_confidence)
    np.save(output_dir / "query_track_labels.npy", query_labels)
    np.save(output_dir / "query_start_frames.npy", query_start_frames)
    np.save(output_dir / "query_base_angles_rad.npy", query_base_angles.astype(np.float32))
    np.save(output_dir / "tracked_points_frame_camera.npy", frame_points)
    np.save(output_dir / "tracked_points_align_camera.npy", align_points)
    np.save(output_dir / "tracked_depth_neighbor_px.npy", depth_neighbor_px)

    angles_rad, angle_diagnostics = estimate_angles(
        frames,
        align_points,
        tracks_xy,
        visibility,
        query_start_frames,
        query_base_angles,
        joint,
        meta,
        export_root,
        args.min_angle_points,
        args.max_abs_angle_deg,
        args.angle_method,
        args.angle_trim_fraction,
        args.hinge_angle_min_deg,
        args.hinge_angle_max_deg,
        args.hinge_angle_coarse_steps,
        args.hinge_angle_fine_window_deg,
        args.hinge_angle_fine_steps,
        args.hinge_depth_weight_px_per_m,
        args.hinge_continuity_weight_px_per_deg,
        query_labels,
        args.monotonic_hinge,
        args.monotonic_backoff_deg,
    )
    if args.smooth_angle_spikes:
        angles_rad, angle_diagnostics = smooth_single_frame_angle_spikes(
            angles_rad,
            angle_diagnostics,
            args.angle_spike_threshold_deg,
        )
    np.save(output_dir / "screen_angles_rad.npy", angles_rad.astype(np.float32))

    print("Exporting dynamic laptop meshes...")
    frame_entries = export_dynamic_meshes(
        meta,
        export_root,
        alignment_dir,
        output_dir,
        frames,
        angles_rad,
        joint,
        tracks_xy,
        visibility,
        frame_points,
        args.rgb_dir_name,
    )

    # A compact visual sanity check on the first and last frame.
    for entry in (frame_entries[0], frame_entries[-1]):
        frame = int(entry["frame"])
        save_mesh_projection_overlay(
            meta,
            export_root / args.rgb_dir_name / f"{frame_name(frame)}.png",
            [Path(entry["base_mesh"]), Path(entry["screen_mesh"])],
            output_dir / f"mesh_projection_frame_{frame_name(frame)}.png",
            [(220, 50, 65, 190), (75, 145, 210, 190)],
        )

    manifest = {
        "type": "screen_hinge_motion",
        "alignment_dir": str(alignment_dir),
        "export_root": str(export_root),
        "output_dir": str(output_dir),
        "frames": frame_entries,
        "frame_indices": frames,
        "base_part_label": BASE_PART_LABEL,
        "screen_part_label": SCREEN_PART_LABEL,
        "joint_name": joint.get("name"),
        "joint_align_camera": joint,
        "query_points_xy": queries_xy,
        "query_track_labels": query_labels,
        "query_start_frames": query_start_frames,
        "query_base_angles_rad": query_base_angles,
        "reseed_events": reseed_events,
        "query_sampling": {
            "mode": args.query_mode,
            "num_query_points": int(args.num_query_points),
            "surface_query_points": int(args.surface_query_points),
            "edge_query_points": int(args.edge_query_points),
            "components": query_components,
            "query_margin_px": float(args.query_margin_px),
            "top_band_ratio": float(args.top_band_ratio),
            "top_band_weight": float(args.top_band_weight),
            "edge_top_ratio": float(args.edge_top_ratio),
            "edge_rows": int(args.edge_rows),
            "reseed_enabled": bool(args.enable_reseed),
            "reseed_trigger_points": int(args.reseed_trigger_points),
            "reseed_target_points": int(args.reseed_target_points),
            "reseed_max_events": int(args.reseed_max_events),
            "reseed_use_projected_screen_mask": bool(args.reseed_use_projected_screen_mask),
            "reseed_projected_mask_dilation_px": int(args.reseed_projected_mask_dilation_px),
            "strategy": "dual_surface_plus_lid_edge" if args.query_mode == "dual" else ("lid_edge_weighted" if args.query_mode == "lid_edge" else "vertical_bands_with_top_weight"),
        },
        "query_overlay": str(query_overlay),
        "tracker": tracker_info,
        "depth": {
            "convention": convention,
            "depth_min_m": float(args.depth_min_m),
            "depth_max_m": float(args.depth_max_m),
            "neighbor_radius_px": float(args.depth_neighbor_radius_px),
            "per_frame_stats": depth_stats,
        },
        "angle_estimation": {
            "method": args.angle_method,
            "min_angle_points": int(args.min_angle_points),
            "trim_fraction": float(args.angle_trim_fraction),
            "smooth_single_frame_spikes": bool(args.smooth_angle_spikes),
            "spike_threshold_deg": float(args.angle_spike_threshold_deg),
            "max_abs_angle_deg": float(args.max_abs_angle_deg),
            "hinge_angle_min_deg": float(args.hinge_angle_min_deg),
            "hinge_angle_max_deg": float(args.hinge_angle_max_deg),
            "hinge_depth_weight_px_per_m": float(args.hinge_depth_weight_px_per_m),
            "hinge_continuity_weight_px_per_deg": float(args.hinge_continuity_weight_px_per_deg),
            "monotonic_hinge": bool(args.monotonic_hinge),
            "monotonic_backoff_deg": float(args.monotonic_backoff_deg),
            "per_frame": angle_diagnostics,
        },
    }
    write_json(output_dir / "dynamic_manifest.json", manifest)

    print(f"Saved dynamic manifest: {output_dir / 'dynamic_manifest.json'}")
    print("Estimated screen angles:")
    for entry in frame_entries:
        print(
            f"  frame {entry['frame']:06d}: {entry['angle_deg']:+.2f} deg, "
            f"valid tracks={entry['valid_track_points']}"
        )


if __name__ == "__main__":
    main()
