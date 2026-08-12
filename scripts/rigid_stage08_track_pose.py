#!/usr/bin/env python3
"""Track rigid-object points with CoTracker3 and estimate per-frame C0 poses."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vlm_sam2_recon.rigid_pipeline.common import (  # noqa: E402
    read_json,
    update_stage_state,
    write_json,
)
from vlm_sam2_recon.rigid_pipeline.geometry import (  # noqa: E402
    backproject_depth,
    transform_points,
)
from vlm_sam2_recon.rigid_pipeline.mesh_alignment import (  # noqa: E402
    Similarity,
    fixed_scale_icp,
    projection_diagnostics,
)
from vlm_sam2_recon.rigid_pipeline.rigid_tracking import (  # noqa: E402
    REJECTION_NAMES,
    estimate_pairwise_pose_sequence,
    identify_bad_pairwise_tracks,
    pose_step_metrics,
    reject_temporal_spikes,
    sample_metric_depth_tracks,
)


DEFAULT_COTRACKER_ROOT = Path("/code/ArtHOI-4D-Reconstruction/third_party/co-tracker")
DEFAULT_COTRACKER_CHECKPOINT = DEFAULT_COTRACKER_ROOT / "checkpoints/scaled_offline.pth"
TRACK_COLORS = np.asarray(
    [(40, 220, 80), (255, 190, 25), (40, 180, 255), (235, 70, 80), (185, 100, 245)],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=PROJECT_ROOT / "run_rigid_20260715_215524")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--rgb-dir",
        type=Path,
        default=None,
        help="Tracking RGB frames. Mixed runs should pass DiffuEraser hand-removed frames.",
    )
    parser.add_argument(
        "--poses-path",
        type=Path,
        default=None,
        help="Camera trajectory NPZ. Defaults to outputs/00_rgb_frames/poses.npz.",
    )
    parser.add_argument(
        "--depth-dir",
        type=Path,
        default=None,
        help="Metric depth NPY sequence. Defaults to outputs/06_dense_depth/metric_depth_npy.",
    )
    parser.add_argument("--mask-dir", type=Path, default=None)
    parser.add_argument(
        "--exclude-depth-mask-dir",
        type=Path,
        default=None,
        help="Optional masks (for example hands) whose pixels are invalidated in metric depth only.",
    )
    parser.add_argument(
        "--exclude-depth-mask-dilation-px",
        type=int,
        default=2,
        help="Dilate excluded pixels before invalidating depth (covers projected-depth splats).",
    )
    parser.add_argument("--transform0", type=Path, default=None)
    parser.add_argument("--aligned-mesh", type=Path, default=None)
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="Inclusive final frame for rigid tracking/optimization.",
    )
    parser.add_argument("--cotracker-root", type=Path, default=DEFAULT_COTRACKER_ROOT)
    parser.add_argument("--cotracker-checkpoint", type=Path, default=DEFAULT_COTRACKER_CHECKPOINT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--anchor-frames", default="0,12,24,36,48")
    parser.add_argument(
        "--allow-nonzero-first-anchor",
        action="store_true",
        help="Do not inject frame 0 when tracking an isolated later interaction segment.",
    )
    parser.add_argument("--queries-per-anchor", type=int, default=40)
    parser.add_argument(
        "--query-points-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON list of explicit CoTracker queries as "
            "{'frame': int, 'xy': [x, y]}; overrides automatic mask sampling."
        ),
    )
    parser.add_argument("--query-mask-margin-px", type=float, default=6.0)
    parser.add_argument("--tracker-max-side", type=int, default=768)
    parser.add_argument("--tracker-confidence", type=float, default=0.75)
    parser.add_argument(
        "--overlay-fps",
        type=float,
        default=None,
        help="FPS for tracks_overlay.mp4. Defaults to the median RGB timestamp interval.",
    )
    parser.add_argument(
        "--reuse-tracks",
        action="store_true",
        help="Reuse CoTracker arrays already present in --output-dir and rerun only RGB-D pose estimation.",
    )
    parser.add_argument(
        "--backward-tracking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run CoTracker's reverse-video pass for frames before each query time.",
    )
    parser.add_argument("--depth-radius-px", type=int, default=3)
    parser.add_argument("--max-local-depth-mad-m", type=float, default=0.025)
    parser.add_argument("--depth-spike-m", type=float, default=0.055)
    parser.add_argument("--point-spike-m", type=float, default=0.075)
    parser.add_argument("--ransac-threshold-m", type=float, default=0.025)
    parser.add_argument("--ransac-iterations", type=int, default=600)
    parser.add_argument("--min-ransac-inliers", type=int, default=8)
    parser.add_argument(
        "--pairwise-max-rotation-rate-deg-s",
        type=float,
        default=675.0,
        help="Pairwise rigid-motion gate in deg/s (equivalent to 45 deg/frame at 15 fps).",
    )
    parser.add_argument(
        "--pairwise-max-center-speed-m-s",
        type=float,
        default=2.7,
        help="Pairwise object-center gate in m/s (equivalent to 0.18 m/frame at 15 fps).",
    )
    parser.add_argument(
        "--pnp-max-rotation-rate-deg-s",
        type=float,
        default=225.0,
        help="PnP motion gate in deg/s (equivalent to 15 deg/frame at 15 fps).",
    )
    parser.add_argument(
        "--pnp-max-center-speed-m-s",
        type=float,
        default=1.05,
        help="PnP object-center gate in m/s (equivalent to 0.07 m/frame at 15 fps).",
    )
    parser.add_argument(
        "--enable-pnp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use frame-0 3D-to-2D PnP after pairwise RGB-D estimation.",
    )
    parser.add_argument("--max-global-median-residual-m", type=float, default=0.025)
    parser.add_argument("--max-global-q90-residual-m", type=float, default=0.06)
    parser.add_argument("--enable-icp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--icp-samples", type=int, default=6000)
    parser.add_argument("--icp-max-update-rotation-deg", type=float, default=8.0)
    parser.add_argument("--icp-max-update-translation-m", type=float, default=0.025)
    parser.add_argument("--max-failed-frame-ratio", type=float, default=0.10)
    parser.add_argument("--min-median-silhouette-iou", type=float, default=0.65)
    parser.add_argument("--min-q10-silhouette-iou", type=float, default=0.55)
    parser.add_argument("--min-single-frame-silhouette-iou", type=float, default=0.45)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--skip-stage-state-update",
        action="store_true",
        help="Write tracking artifacts without changing pipeline_state.json (used by articulate-part wrappers).",
    )
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def parse_anchor_frames(text: str, frame_count: int, require_frame0: bool = True) -> list[int]:
    frames = sorted({int(value.strip()) for value in text.split(",") if value.strip()})
    if not frames:
        raise ValueError("At least one anchor frame is required")
    if require_frame0 and frames[0] != 0:
        frames.insert(0, 0)
    if any(frame < 0 or frame >= frame_count for frame in frames):
        raise ValueError(f"Anchor frames outside [0, {frame_count - 1}]: {frames}")
    return frames


def load_frames(paths: list[Path]) -> list[np.ndarray]:
    return [np.asarray(Image.open(path).convert("RGB")) for path in paths]


def discover_image_sequence(directory: Path) -> list[Path]:
    for pattern in ("*.png", "*.jpg", "*.jpeg"):
        paths = sorted(directory.glob(pattern))
        if paths:
            return paths
    return []


def choose_query_points(rgb: np.ndarray, mask: np.ndarray, count: int, margin_px: float) -> np.ndarray:
    distance = distance_transform_edt(mask)
    valid = distance >= margin_px
    if valid.sum() < count:
        valid = mask & (distance >= 2.0)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)
    ys, xs = np.where(valid)
    if len(xs) < 3:
        raise ValueError("Too few valid pixels for CoTracker queries")
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    aspect = max(0.3, (x1 - x0 + 1) / max(y1 - y0 + 1, 1))
    cols = max(4, int(round(np.sqrt(count * aspect))))
    rows = max(4, int(np.ceil(count / cols)))
    selected: list[tuple[float, float]] = []
    used: set[tuple[int, int]] = set()
    for row in range(rows):
        ya = int(round(y0 + row / rows * (y1 - y0 + 1)))
        yb = int(round(y0 + (row + 1) / rows * (y1 - y0 + 1)))
        for col in range(cols):
            xa = int(round(x0 + col / cols * (x1 - x0 + 1)))
            xb = int(round(x0 + (col + 1) / cols * (x1 - x0 + 1)))
            region = valid[ya:yb, xa:xb]
            ry, rx = np.where(region)
            if not len(rx):
                continue
            scores = gradient[ya + ry, xa + rx] + 5.0 * distance[ya + ry, xa + rx]
            index = int(np.argmax(scores))
            point = (int(xa + rx[index]), int(ya + ry[index]))
            if point not in used:
                used.add(point)
                selected.append((float(point[0]), float(point[1])))
            if len(selected) >= count:
                return np.asarray(selected, dtype=np.float32)
    remaining = np.flatnonzero(valid)
    order = remaining[np.argsort((gradient + 5.0 * distance).reshape(-1)[remaining])[::-1]]
    for flat in order:
        y, x = np.unravel_index(flat, valid.shape)
        if all((x - px) ** 2 + (y - py) ** 2 >= 36 for px, py in selected):
            selected.append((float(x), float(y)))
        if len(selected) >= count:
            break
    return np.asarray(selected, dtype=np.float32)


def tracker_resize(frames: list[np.ndarray], max_side: int) -> tuple[np.ndarray, float]:
    height, width = frames[0].shape[:2]
    scale = min(1.0, float(max_side) / max(height, width))
    if scale < 1.0:
        size = (int(round(width * scale)), int(round(height * scale)))
        frames = [cv2.resize(frame, size, interpolation=cv2.INTER_AREA) for frame in frames]
    return np.stack(frames), scale


def run_cotracker(
    frames: list[np.ndarray],
    queries_xy: np.ndarray,
    query_times: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    root = args.cotracker_root.resolve()
    checkpoint = args.cotracker_checkpoint.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from cotracker.models.core.model_utils import get_points_on_a_grid
    from cotracker.predictor import CoTrackerPredictor

    video_np, scale = tracker_resize(frames, args.tracker_max_side)
    video = torch.from_numpy(video_np).permute(0, 3, 1, 2)[None].float().to(args.device)
    query_txy = np.column_stack([query_times, queries_xy * scale]).astype(np.float32)
    queries = torch.from_numpy(query_txy)[None].to(args.device)
    model = CoTrackerPredictor(
        checkpoint=str(checkpoint),
        offline=True,
        window_len=60,
    ).to(args.device)
    model.eval()
    batch, timesteps, channels, height, width = video.shape
    with torch.no_grad():
        video_interp = F.interpolate(
            video.reshape(batch * timesteps, channels, height, width),
            tuple(model.interp_shape),
            mode="bilinear",
            align_corners=True,
        ).reshape(batch, timesteps, 3, *model.interp_shape)
        queries_model = queries.clone()
        queries_model[:, :, 1:] *= queries_model.new_tensor(
            [(model.interp_shape[1] - 1) / (width - 1), (model.interp_shape[0] - 1) / (height - 1)]
        )
        query_count = queries_model.shape[1]
        support = get_points_on_a_grid(model.support_grid_size, model.interp_shape, device=video.device)
        support = torch.cat([torch.zeros_like(support[:, :, :1]), support], dim=2).repeat(batch, 1, 1)
        all_queries = torch.cat([queries_model, support], dim=1)
        tracks, confidence, *_ = model.model.forward(
            video=video_interp, queries=all_queries, iters=6
        )
        if args.backward_tracking:
            tracks, confidence = model._compute_backward_tracks(
                video_interp, all_queries, tracks, confidence
            )
        tracks, confidence = tracks[:, :, :query_count], confidence[:, :, :query_count]
        qtime = queries_model[0, :, 0].round().long().clamp(0, timesteps - 1)
        qindex = torch.arange(query_count, device=video.device)
        tracks[0, qtime, qindex] = queries_model[0, :, 1:]
        confidence[0, qtime, qindex] = 1.0
        tracks *= tracks.new_tensor([(width - 1) / (model.interp_shape[1] - 1), (height - 1) / (model.interp_shape[0] - 1)])
    tracks_np = tracks[0].cpu().numpy().astype(np.float32) / scale
    confidence_np = confidence[0].float().cpu().numpy().astype(np.float32)
    visible = confidence_np >= args.tracker_confidence
    if not args.backward_tracking:
        visible[np.arange(len(frames))[:, None] < query_times[None, :]] = False
    torch.cuda.empty_cache()
    return tracks_np, visible, confidence_np, {
        "checkpoint": str(checkpoint),
        "device": args.device,
        "tracker_scale": float(scale),
        "tracker_input_shape": list(video_np.shape),
        "confidence_threshold": args.tracker_confidence,
        "backward_tracking": bool(args.backward_tracking),
    }


def draw_query_overlays(
    rgb_paths: list[Path], masks: np.ndarray, anchors: list[int], queries: np.ndarray, query_times: np.ndarray, output: Path
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for anchor_index, anchor in enumerate(anchors):
        image = Image.open(rgb_paths[anchor]).convert("RGBA")
        tint = Image.new("RGBA", image.size, (40, 230, 80, 0))
        tint.putalpha(Image.fromarray(masks[anchor].astype(np.uint8) * 60))
        image = Image.alpha_composite(image, tint)
        draw = ImageDraw.Draw(image)
        indices = np.flatnonzero(query_times == anchor)
        color = tuple(int(x) for x in TRACK_COLORS[anchor_index % len(TRACK_COLORS)]) + (255,)
        for index in indices:
            x, y = queries[index]
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline=color, width=3)
            draw.text((x + 7, y - 7), str(index), fill=color)
        image.convert("RGB").save(output / f"anchor_{anchor:06d}.jpg", quality=92)


def observed_cloud_c0(
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsics: dict,
    transform_c0_from_ct: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    valid = mask & np.isfinite(depth) & (depth >= 0.1) & (depth <= 3.0)
    values = depth[valid]
    if values.size < 100:
        return np.empty((0, 3), dtype=np.float64)
    low, high = np.quantile(values, [0.01, 0.95])
    filtered = np.where(valid & (depth >= low) & (depth <= high), depth, 0.0)
    points, _ = backproject_depth(filtered, intrinsics)
    points = transform_points(points, transform_c0_from_ct)
    if len(points) > count:
        points = points[rng.choice(len(points), count, replace=False)]
    return points


def one_way_trimmed_rmse(source: np.ndarray, target: np.ndarray, fraction: float = 0.65) -> float:
    distance = cKDTree(target).query(source, k=1, workers=-1)[0]
    keep = max(1, int(len(distance) * fraction))
    values = np.partition(distance, keep - 1)[:keep]
    return float(np.sqrt(np.mean(values**2)))


def refine_poses_icp(
    poses: np.ndarray,
    frame_diagnostics: list[dict],
    source_points_c0: np.ndarray,
    depths: np.ndarray,
    masks: np.ndarray,
    intrinsics: dict,
    transforms_c0_from_ct: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[dict]]:
    refined = poses.copy()
    rng = np.random.default_rng(args.random_seed + 100)
    for frame in range(len(poses)):
        record = frame_diagnostics[frame]
        if frame == 0 or record["status"] != "completed" or not args.enable_icp:
            record["icp_accepted"] = False
            record["icp_reason"] = "frame0_or_tracking_failed_or_disabled"
            continue
        target = observed_cloud_c0(
            depths[frame], masks[frame], intrinsics, transforms_c0_from_ct[frame], args.icp_samples, rng
        )
        if len(target) < 300:
            record["icp_accepted"] = False
            record["icp_reason"] = "too_few_observed_points"
            continue
        pre = one_way_trimmed_rmse(transform_points(source_points_c0, poses[frame]), target)
        candidate, history = fixed_scale_icp(
            source_points_c0,
            target,
            Similarity(1.0, poses[frame, :3, :3], poses[frame, :3, 3]),
            max_distances_m=(0.035, 0.022, 0.014),
            iterations_per_level=5,
            trim_fraction=0.62,
        )
        post = one_way_trimmed_rmse(candidate.transform(source_points_c0), target)
        update_rotation, update_translation = pose_step_metrics(poses[frame], candidate.matrix)
        c0_to_ct = np.linalg.inv(transforms_c0_from_ct[frame])
        pre_ct = transform_points(transform_points(source_points_c0, poses[frame]), c0_to_ct)
        post_ct = transform_points(candidate.transform(source_points_c0), c0_to_ct)
        pre_projection, _, _ = projection_diagnostics(
            pre_ct, depths[frame], masks[frame], intrinsics
        )
        post_projection, _, _ = projection_diagnostics(
            post_ct, depths[frame], masks[frame], intrinsics
        )
        accepted = (
            post <= pre * 1.01
            and update_rotation <= args.icp_max_update_rotation_deg
            and update_translation <= args.icp_max_update_translation_m
            and post_projection["silhouette_iou_point_splat"]
            >= pre_projection["silhouette_iou_point_splat"] - 0.005
        )
        if accepted:
            refined[frame] = candidate.matrix
        record.update(
            {
                "icp_accepted": bool(accepted),
                "icp_reason": "accepted" if accepted else "update_gate_or_no_improvement",
                "icp_pre_rmse_m": pre,
                "icp_post_rmse_m": post,
                "icp_update_rotation_deg": update_rotation,
                "icp_update_translation_m": update_translation,
                "icp_iterations": len(history),
                "icp_pre_silhouette_iou": pre_projection["silhouette_iou_point_splat"],
                "icp_post_silhouette_iou": post_projection["silhouette_iou_point_splat"],
            }
        )
    return refined, frame_diagnostics


def refine_poses_pnp(
    pairwise_poses: np.ndarray,
    local_points: np.ndarray,
    tracks_xy: np.ndarray,
    rejection: np.ndarray,
    globally_bad: np.ndarray,
    query_times: np.ndarray,
    intrinsics: dict,
    transforms_c0_from_ct: np.ndarray,
    timestamps_s: np.ndarray,
    object_center_c0: np.ndarray,
    diagnostics: list[dict],
    max_rotation_rate_deg_s: float,
    max_center_speed_m_s: float,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Stabilize rotation using frame-0 metric 3D points and current 2D tracks."""
    camera_matrix = np.asarray(
        [
            [intrinsics["fx"], 0.0, intrinsics["cx"]],
            [0.0, intrinsics["fy"], intrinsics["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    poses = pairwise_poses.copy()
    poses[0] = np.eye(4)
    pnp_inliers = np.zeros(rejection.shape, dtype=bool)
    reference_tracks = (query_times == 0) & ~globally_bad & np.isfinite(local_points).all(axis=1)
    pnp_inliers[0, reference_tracks & (rejection[0] == 0)] = True
    diagnostics[0].update(
        {
            "pnp_status": "frame0_identity",
            "pnp_candidate_count": int((reference_tracks & (rejection[0] == 0)).sum()),
            "pnp_inlier_count": int(pnp_inliers[0].sum()),
            "pnp_reprojection_rmse_px": 0.0,
        }
    )
    for frame in range(1, len(poses)):
        frame_dt_s = float(timestamps_s[frame] - timestamps_s[frame - 1])
        if not np.isfinite(frame_dt_s) or frame_dt_s <= 0.0:
            raise ValueError(f"Invalid RGB timestamp interval at frame {frame}: {frame_dt_s}")
        max_rotation_step_deg = float(max_rotation_rate_deg_s * frame_dt_s)
        max_center_step_m = float(max_center_speed_m_s * frame_dt_s)
        indices = np.flatnonzero(reference_tracks & (rejection[frame] == 0))
        status = "failed"
        reason = "too_few_frame0_reference_tracks"
        candidate = None
        inlier_indices = np.empty(0, dtype=np.int64)
        reprojection_rmse = None
        if len(indices) >= 8:
            initial_ct_from_metric = np.linalg.inv(transforms_c0_from_ct[frame]) @ poses[frame - 1]
            rvec, _ = cv2.Rodrigues(initial_ct_from_metric[:3, :3])
            tvec = initial_ct_from_metric[:3, 3].reshape(3, 1).copy()
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                local_points[indices].astype(np.float64),
                tracks_xy[frame, indices].astype(np.float64),
                camera_matrix,
                None,
                rvec,
                tvec,
                True,
                iterationsCount=300,
                reprojectionError=5.0,
                confidence=0.999,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if success and inliers is not None and len(inliers) >= 8:
                local_inliers = inliers[:, 0]
                try:
                    rvec, tvec = cv2.solvePnPRefineLM(
                        local_points[indices[local_inliers]].astype(np.float64),
                        tracks_xy[frame, indices[local_inliers]].astype(np.float64),
                        camera_matrix,
                        None,
                        rvec,
                        tvec,
                    )
                except cv2.error:
                    pass
                rotation, _ = cv2.Rodrigues(rvec)
                ct_from_metric = np.eye(4, dtype=np.float64)
                ct_from_metric[:3, :3] = rotation
                ct_from_metric[:3, 3] = tvec[:, 0]
                candidate = transforms_c0_from_ct[frame] @ ct_from_metric
                projected, _ = cv2.projectPoints(
                    local_points[indices[local_inliers]].astype(np.float64),
                    rvec,
                    tvec,
                    camera_matrix,
                    None,
                )
                residual = projected[:, 0] - tracks_xy[frame, indices[local_inliers]]
                reprojection_rmse = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
                relative = candidate @ np.linalg.inv(poses[frame - 1])
                rotation_step = float(np.rad2deg(Rotation.from_matrix(relative[:3, :3]).magnitude()))
                previous_center = transform_points(object_center_c0[None], poses[frame - 1])[0]
                candidate_center = transform_points(object_center_c0[None], candidate)[0]
                center_step = float(np.linalg.norm(candidate_center - previous_center))
                diagnostics[frame].update(
                    {
                        "pnp_frame_dt_s": frame_dt_s,
                        "pnp_candidate_rotation_step_deg": rotation_step,
                        "pnp_candidate_center_step_m": center_step,
                        "pnp_max_rotation_step_deg": max_rotation_step_deg,
                        "pnp_max_center_step_m": max_center_step_m,
                    }
                )
                if (
                    rotation_step <= max_rotation_step_deg
                    and center_step <= max_center_step_m
                    and reprojection_rmse <= 5.0
                ):
                    poses[frame] = candidate
                    inlier_indices = indices[local_inliers]
                    pnp_inliers[frame, inlier_indices] = True
                    status = "completed"
                    reason = None
                else:
                    reason = "pnp_motion_or_reprojection_gate"
        if status != "completed":
            # Preserve a valid 3D-3D estimate rather than silently copying the previous pose.
            poses[frame] = pairwise_poses[frame]
            status = "pairwise_fallback"
        diagnostics[frame].update(
            {
                "pnp_status": status,
                "pnp_failure_reason": reason,
                "pnp_candidate_count": int(len(indices)),
                "pnp_inlier_count": int(len(inlier_indices)),
                "pnp_reprojection_rmse_px": reprojection_rmse,
            }
        )
    return poses, pnp_inliers, diagnostics


def projection_metrics(
    delta_poses: np.ndarray,
    source_points_c0: np.ndarray,
    depths: np.ndarray,
    masks: np.ndarray,
    intrinsics: dict,
    transforms_c0_from_ct: np.ndarray,
    diagnostics: list[dict],
) -> None:
    for frame in range(len(delta_poses)):
        points_c0 = transform_points(source_points_c0, delta_poses[frame])
        points_ct = transform_points(points_c0, np.linalg.inv(transforms_c0_from_ct[frame]))
        metrics, _, _ = projection_diagnostics(points_ct, depths[frame], masks[frame], intrinsics)
        diagnostics[frame]["silhouette_iou"] = metrics["silhouette_iou_point_splat"]
        diagnostics[frame]["depth_trimmed_rmse_m"] = metrics["depth_trimmed_rmse_m"]
        diagnostics[frame]["depth_median_abs_m"] = metrics["depth_median_abs_m"]


def save_track_video(
    frames: list[np.ndarray],
    masks: np.ndarray,
    tracks: np.ndarray,
    valid: np.ndarray,
    query_times: np.ndarray,
    output: Path,
    fps: float,
) -> None:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for frame_index, rgb in enumerate(frames):
        canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        contours, _ = cv2.findContours(masks[frame_index].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, (40, 230, 80), 2)
        for point, (x, y) in enumerate(tracks[frame_index]):
            if frame_index < query_times[point]:
                continue
            color_rgb = TRACK_COLORS[np.searchsorted(np.unique(query_times), query_times[point]) % len(TRACK_COLORS)]
            color = tuple(int(v) for v in color_rgb[::-1]) if valid[frame_index, point] else (40, 40, 230)
            cv2.circle(canvas, (int(round(x)), int(round(y))), 3, color, -1, cv2.LINE_AA)
        writer.write(canvas)
    writer.release()


def main() -> int:
    args = parse_args()
    workspace = args.workspace.resolve()
    output = (args.output_dir or workspace / "outputs/08_tracking").resolve()
    rgb_dir = (args.rgb_dir or workspace / "outputs/00_rgb_frames/right_rgb_png").absolute()
    mask_dir = (args.mask_dir or workspace / "outputs/04_object_masks/combined").resolve()
    depth_dir = (
        args.depth_dir.resolve()
        if args.depth_dir is not None
        else workspace / "outputs/06_dense_depth/metric_depth_npy"
    )
    camera_path = workspace / "outputs/00_rgb_frames/camera.json"
    poses_path = (
        args.poses_path.resolve()
        if args.poses_path is not None
        else workspace / "outputs/00_rgb_frames/poses.npz"
    )
    transform0_path = (
        args.transform0 or workspace / "outputs/07_alignment/frame_000000/T_C0_from_O.npy"
    ).resolve()
    aligned_mesh_path = (
        args.aligned_mesh
        or workspace / "outputs/07_alignment/frame_000000/hunyuan_mesh_aligned_C0.glb"
    ).resolve()
    rgb_paths = discover_image_sequence(rgb_dir)
    mask_paths = sorted(mask_dir.glob("*.png"))
    depth_paths = sorted(depth_dir.glob("*.npy"))
    if not (len(rgb_paths) == len(mask_paths) == len(depth_paths) and rgb_paths):
        raise ValueError(f"RGB/mask/depth count mismatch: {len(rgb_paths)}/{len(mask_paths)}/{len(depth_paths)}")
    full_frame_count = len(rgb_paths)
    exclude_depth_mask_paths: list[Path] = []
    exclude_depth_mask_dir: Path | None = None
    if args.exclude_depth_mask_dir is not None:
        exclude_depth_mask_dir = args.exclude_depth_mask_dir.resolve()
        exclude_depth_mask_paths = sorted(exclude_depth_mask_dir.glob("*.png"))
        if len(exclude_depth_mask_paths) != full_frame_count:
            raise ValueError(
                "Exclude-depth mask count mismatch: "
                f"{len(exclude_depth_mask_paths)} vs RGB {full_frame_count}"
            )
    if args.end_frame is not None:
        if not 0 <= args.end_frame < full_frame_count:
            raise ValueError(f"--end-frame must be in [0, {full_frame_count - 1}]")
        stop = args.end_frame + 1
        rgb_paths = rgb_paths[:stop]
        mask_paths = mask_paths[:stop]
        depth_paths = depth_paths[:stop]
        exclude_depth_mask_paths = exclude_depth_mask_paths[:stop]
    for path in (camera_path, poses_path, transform0_path, aligned_mesh_path, args.cotracker_checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)
    anchors = parse_anchor_frames(
        args.anchor_frames, len(rgb_paths), require_frame0=not args.allow_nonzero_first_anchor
    )
    preflight = {
        "stage": "08_rigid_pose_tracking_frame0",
        "frame_count": len(rgb_paths),
        "full_timeline_frame_count": full_frame_count,
        "optimization_end_frame_inclusive": len(rgb_paths) - 1,
        "anchor_frames": anchors,
        "queries_per_anchor": args.queries_per_anchor,
        "reuse_tracks": bool(args.reuse_tracks),
        "motion_gates": {
            "pairwise_max_rotation_rate_deg_s": args.pairwise_max_rotation_rate_deg_s,
            "pairwise_max_center_speed_m_s": args.pairwise_max_center_speed_m_s,
            "pnp_max_rotation_rate_deg_s": args.pnp_max_rotation_rate_deg_s,
            "pnp_max_center_speed_m_s": args.pnp_max_center_speed_m_s,
            "policy": "rate limits are multiplied by the actual RGB timestamp interval",
        },
        "pnp_enabled": bool(args.enable_pnp),
        "tracking_rgb_dir": str(rgb_dir),
        "camera_poses": str(poses_path),
        "depth_dir": str(depth_dir),
        "exclude_depth_mask_dir": (
            str(exclude_depth_mask_dir) if exclude_depth_mask_dir is not None else None
        ),
        "exclude_depth_mask_dilation_px": args.exclude_depth_mask_dilation_px,
        "tracking_rgb_policy": (
            "diffueraser_hand_removed"
            if "03_diffueraser" in rgb_dir.parts or "inpainted" in rgb_dir.name
            else "caller_supplied_or_raw"
        ),
        "cotracker_checkpoint": str(args.cotracker_checkpoint.resolve()),
        "world_frame": "frame0_right_camera_opencv_rdf",
    }
    if args.check:
        print(json.dumps(preflight, indent=2))
        return 0

    output.mkdir(parents=True, exist_ok=True)
    camera = read_json(camera_path)
    intrinsics = camera["rgb_intrinsics_right"]
    frames = load_frames(rgb_paths)
    masks = np.stack([np.asarray(Image.open(path).convert("L")) > 127 for path in mask_paths])
    depths = np.stack([np.load(path).astype(np.float32) for path in depth_paths])
    excluded_depth_pixels = np.zeros(len(depths), dtype=np.int64)
    if exclude_depth_mask_paths:
        exclude_depth_masks = np.stack(
            [np.asarray(Image.open(path).convert("L")) > 127 for path in exclude_depth_mask_paths]
        )
        dilation_px = int(args.exclude_depth_mask_dilation_px)
        if dilation_px < 0:
            raise ValueError("--exclude-depth-mask-dilation-px must be >= 0")
        if dilation_px:
            kernel = np.ones((2 * dilation_px + 1, 2 * dilation_px + 1), dtype=np.uint8)
            exclude_depth_masks = np.stack(
                [cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) > 0 for mask in exclude_depth_masks]
            )
        excluded_depth_pixels = np.count_nonzero(
            (depths > 0) & exclude_depth_masks,
            axis=(1, 2),
        )
        depths[exclude_depth_masks] = 0.0
    with np.load(poses_path) as pose_data:
        required_pose_keys = {"T_C0_from_Ct", "rgb_timestamps_s"}
        missing_pose_keys = required_pose_keys.difference(pose_data.files)
        if missing_pose_keys:
            raise KeyError(f"Missing pose keys in {poses_path}: {sorted(missing_pose_keys)}")
        transforms_c0_from_ct = pose_data["T_C0_from_Ct"].astype(np.float64)[: len(rgb_paths)]
        timestamps = pose_data["rgb_timestamps_s"].astype(np.float64)[: len(rgb_paths)]
    if transforms_c0_from_ct.shape != (len(rgb_paths), 4, 4) or timestamps.shape != (len(rgb_paths),):
        raise ValueError(
            f"Pose timeline does not cover tracking frames: transforms={transforms_c0_from_ct.shape}, "
            f"timestamps={timestamps.shape}, frames={len(rgb_paths)}"
        )
    timestamp_deltas = np.diff(timestamps)
    if len(timestamp_deltas) and (
        not np.isfinite(timestamp_deltas).all() or np.any(timestamp_deltas <= 0.0)
    ):
        raise ValueError("RGB timestamps must be finite and strictly increasing")
    nominal_frame_dt_s = float(np.median(timestamp_deltas)) if len(timestamp_deltas) else 1.0 / 15.0
    pairwise_max_rotation_step_deg = float(
        args.pairwise_max_rotation_rate_deg_s * nominal_frame_dt_s
    )
    pairwise_max_center_step_m = float(
        args.pairwise_max_center_speed_m_s * nominal_frame_dt_s
    )
    transform0 = np.load(transform0_path).astype(np.float64)

    if args.reuse_tracks:
        queries = np.load(output / "query_points_xy.npy").astype(np.float32)
        query_times = np.load(output / "query_times.npy").astype(np.int64)
        tracks = np.load(output / "tracks_2d.npy").astype(np.float32)
        confidence = np.load(output / "track_confidence.npy").astype(np.float32)
        expected_tracks_shape = (len(frames), len(queries), 2)
        if tracks.shape != expected_tracks_shape or confidence.shape != expected_tracks_shape[:2]:
            raise ValueError(
                "Stored CoTracker arrays do not match this run: "
                f"tracks={tracks.shape}, confidence={confidence.shape}, expected={expected_tracks_shape}"
            )
        if query_times.shape != (len(queries),):
            raise ValueError(f"Stored query_times has shape {query_times.shape}, expected {(len(queries),)}")
        tracker_visible = confidence >= args.tracker_confidence
        tracker_visible[np.arange(len(frames))[:, None] < query_times[None, :]] = False
        tracker_info = {
            "source": "reused_output_arrays",
            "confidence_threshold": args.tracker_confidence,
        }
    else:
        if args.query_points_json is not None:
            query_records = json.loads(args.query_points_json.read_text(encoding="utf-8"))
            if not isinstance(query_records, list) or not query_records:
                raise ValueError("--query-points-json must contain a non-empty JSON list")
            queries = np.asarray([record["xy"] for record in query_records], dtype=np.float32)
            query_times = np.asarray([record["frame"] for record in query_records], dtype=np.int64)
            if queries.ndim != 2 or queries.shape[1] != 2 or not np.isfinite(queries).all():
                raise ValueError("Explicit query xy values must form a finite Nx2 array")
            if np.any(query_times < 0) or np.any(query_times >= len(frames)):
                raise ValueError("Explicit query frames fall outside the RGB sequence")
            for query, query_time in zip(queries, query_times):
                x, y = np.round(query).astype(np.int64)
                if not (0 <= y < masks.shape[1] and 0 <= x < masks.shape[2]):
                    raise ValueError(f"Explicit query {query.tolist()} falls outside the image")
                if not masks[query_time, y, x]:
                    raise ValueError(
                        f"Explicit query {query.tolist()} is outside frame {query_time} mask"
                    )
        else:
            queries_by_anchor = [
                choose_query_points(frames[a], masks[a], args.queries_per_anchor, args.query_mask_margin_px)
                for a in anchors
            ]
            queries = np.concatenate(queries_by_anchor, axis=0)
            query_times = np.concatenate(
                [np.full(len(points), anchor, dtype=np.int64) for anchor, points in zip(anchors, queries_by_anchor)]
            )
        print(f"Running CoTracker3 on {len(frames)} frames with {len(queries)} queries...", flush=True)
        tracks, tracker_visible, confidence, tracker_info = run_cotracker(frames, queries, query_times, args)
    draw_query_overlays(rgb_paths, masks, anchors, queries, query_times, output / "query_overlays")

    xyz_ct, sampled_depth, local_mad, rejection = sample_metric_depth_tracks(
        tracks,
        tracker_visible,
        masks,
        depths,
        intrinsics,
        radius_px=args.depth_radius_px,
        max_local_mad_m=args.max_local_depth_mad_m,
    )
    rejection = reject_temporal_spikes(
        xyz_ct,
        sampled_depth,
        rejection,
        query_times,
        depth_spike_m=args.depth_spike_m,
        point_spike_m=args.point_spike_m,
    )
    xyz_c0 = np.full_like(xyz_ct, np.nan)
    for frame in range(len(frames)):
        valid = (rejection[frame] == 0) & np.isfinite(xyz_ct[frame]).all(axis=1)
        xyz_c0[frame, valid] = transform_points(xyz_ct[frame, valid], transforms_c0_from_ct[frame])

    pass1_poses, _, pass1_diagnostics = estimate_pairwise_pose_sequence(
        xyz_c0,
        rejection,
        ransac_threshold_m=args.ransac_threshold_m,
        ransac_iterations=args.ransac_iterations,
        min_inliers=args.min_ransac_inliers,
        max_step_rotation_deg=pairwise_max_rotation_step_deg,
        max_step_translation_m=pairwise_max_center_step_m,
        seed=args.random_seed,
    )
    globally_bad, track_records = identify_bad_pairwise_tracks(
        xyz_c0,
        rejection,
        pass1_poses,
        pass1_diagnostics,
        query_times,
        max_median_residual_m=args.max_global_median_residual_m,
        max_q90_residual_m=args.max_global_q90_residual_m,
    )
    final_rejection = rejection.copy()
    global_rejection_mask = (final_rejection == 0) & globally_bad[None, :]
    final_rejection[global_rejection_mask] = 8
    delta_poses, inlier_mask, diagnostics = estimate_pairwise_pose_sequence(
        xyz_c0,
        final_rejection,
        globally_rejected=globally_bad,
        ransac_threshold_m=args.ransac_threshold_m,
        ransac_iterations=args.ransac_iterations,
        min_inliers=args.min_ransac_inliers,
        max_step_rotation_deg=pairwise_max_rotation_step_deg,
        max_step_translation_m=pairwise_max_center_step_m,
        seed=args.random_seed + 1000,
    )
    local_points = np.full((len(queries), 3), np.nan, dtype=np.float64)
    for point, anchor in enumerate(query_times):
        if final_rejection[anchor, point] == 0 and np.isfinite(xyz_c0[anchor, point]).all():
            local_points[point] = transform_points(xyz_c0[anchor, point][None], np.linalg.inv(delta_poses[anchor]))[0]
    mesh = trimesh.load(aligned_mesh_path, process=False, force="mesh")
    object_center_c0 = np.asarray(mesh.centroid, dtype=np.float64)
    if args.enable_pnp:
        delta_poses, pnp_inlier_mask, diagnostics = refine_poses_pnp(
            delta_poses,
            local_points,
            tracks,
            final_rejection,
            globally_bad,
            query_times,
            intrinsics,
            transforms_c0_from_ct,
            timestamps,
            object_center_c0,
            diagnostics,
            args.pnp_max_rotation_rate_deg_s,
            args.pnp_max_center_speed_m_s,
        )
    else:
        pnp_inlier_mask = np.zeros(final_rejection.shape, dtype=bool)
        for record in diagnostics:
            record.update(
                {
                    "pnp_status": "disabled_depth_first_pairwise",
                    "pnp_failure_reason": None,
                    "pnp_candidate_count": 0,
                    "pnp_inlier_count": 0,
                    "pnp_reprojection_rmse_px": None,
                }
            )
    candidate_mask = (final_rejection == 0) & np.isfinite(local_points).all(axis=1)[None, :]
    final_rejection[candidate_mask & ~inlier_mask & (np.arange(len(frames))[:, None] > 0)] = 9

    np.random.seed(args.random_seed)
    source_points_c0, _ = trimesh.sample.sample_surface(mesh, args.icp_samples)
    delta_poses, diagnostics = refine_poses_icp(
        delta_poses,
        diagnostics,
        source_points_c0,
        depths,
        masks,
        intrinsics,
        transforms_c0_from_ct,
        args,
    )
    projection_metrics(
        delta_poses,
        source_points_c0,
        depths,
        masks,
        intrinsics,
        transforms_c0_from_ct,
        diagnostics,
    )
    object_poses = np.einsum("tij,jk->tik", delta_poses, transform0)
    for frame, record in enumerate(diagnostics):
        base_valid = int(np.count_nonzero(rejection[frame] == 0))
        final_valid = int(np.count_nonzero(final_rejection[frame] == 0))
        final_rotation_step = 0.0
        final_center_step = 0.0
        if frame > 0:
            relative = delta_poses[frame] @ np.linalg.inv(delta_poses[frame - 1])
            final_rotation_step = float(
                np.rad2deg(Rotation.from_matrix(relative[:3, :3]).magnitude())
            )
            previous_center = transform_points(object_center_c0[None], delta_poses[frame - 1])[0]
            current_center = transform_points(object_center_c0[None], delta_poses[frame])[0]
            final_center_step = float(np.linalg.norm(current_center - previous_center))
        record.update(
            {
                "rgb_timestamp_s": float(timestamps[frame]),
                "tracker_visible_count": int(tracker_visible[frame].sum()),
                "depth_valid_count_before_global_filter": base_valid,
                "valid_track_count": final_valid,
                "depth_valid_ratio": float(base_valid / max(int(tracker_visible[frame].sum()), 1)),
                "excluded_depth_pixels": int(excluded_depth_pixels[frame]),
                "global_rejected_track_count": int(globally_bad.sum()),
                "final_rotation_step_deg": final_rotation_step,
                "final_object_center_step_m": final_center_step,
                "T_C0_from_O": object_poses[frame].tolist(),
                "Delta_C0_object_motion": delta_poses[frame].tolist(),
            }
        )

    np.save(output / "query_points_xy.npy", queries)
    np.save(output / "query_times.npy", query_times)
    np.save(output / "tracks_2d.npy", tracks)
    np.save(output / "track_confidence.npy", confidence)
    np.save(output / "track_depth_m.npy", sampled_depth)
    np.save(output / "track_local_depth_mad_m.npy", local_mad)
    np.save(output / "tracks_3d_Ct_raw.npy", xyz_ct)
    np.save(output / "tracks_3d_C0_raw.npy", xyz_c0)
    np.save(output / "track_rejection_codes.npy", final_rejection)
    np.save(output / "track_valid.npy", final_rejection == 0)
    np.save(output / "globally_rejected_tracks.npy", globally_bad)
    np.save(output / "track_object_local_points_C0metric.npy", local_points)
    np.save(output / "Delta_C0_object_motion.npy", delta_poses)
    np.save(output / "T_C0_from_O.npy", object_poses)
    write_json(
        output / "coordinate_frames.json",
        {
            "world_frame": "frame0_right_camera_opencv_rdf",
            "matrix_convention": "column_vector_left_multiply",
            "policy": "All downstream and visualized point clouds use C0. Ct arrays are diagnostic-only raw inputs.",
            "artifacts": {
                "tracks_3d_Ct_raw.npy": {
                    "coordinate_frame": "per_frame_right_camera_Ct",
                    "diagnostic_only": True,
                    "downstream_default": False,
                },
                "tracks_3d_C0_raw.npy": {
                    "coordinate_frame": "frame0_right_camera_opencv_rdf",
                    "transform": "p_C0(t) = T_C0_from_Ct(t) @ p_Ct(t)",
                    "downstream_default": True,
                },
                "track_object_local_points_C0metric.npy": {
                    "coordinate_frame": "metric_object_reference_at_frame0",
                    "downstream_default": True,
                },
                "dynamic_rgbd_pointcloud": {
                    "coordinate_frame": "frame0_right_camera_opencv_rdf",
                    "source": "per-frame metric depth backprojected in Ct then transformed by T_C0_from_Ct",
                    "downstream_default": True,
                },
                "dynamic_object_depth_pointcloud": {
                    "coordinate_frame": "frame0_right_camera_opencv_rdf",
                    "source": "mask-filtered metric depth backprojected in Ct then transformed by T_C0_from_Ct",
                    "downstream_default": True,
                },
                "icp_observed_pointcloud": {
                    "coordinate_frame": "frame0_right_camera_opencv_rdf",
                    "downstream_default": True,
                },
            },
        },
    )
    write_json(output / "global_track_diagnostics.json", {"tracks": track_records})
    with (output / "frame_diagnostics.jsonl").open("w", encoding="utf-8") as stream:
        for record in diagnostics:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    write_json(
        output / "object_poses_frame0.json",
        {
            **preflight,
            "matrix_convention": "column_vector_left_multiply",
            "pose_note": "Delta matrices are SE(3); T_C0_from_O preserves the fixed Stage07 Sim3 scale.",
            "tracker": tracker_info,
            "rejection_codes": REJECTION_NAMES,
            "global_rejected_tracks": int(globally_bad.sum()),
            "frames": diagnostics,
        },
    )
    if args.overlay_fps is not None:
        overlay_fps = float(args.overlay_fps)
    elif len(timestamps) > 1:
        positive_dt = np.diff(timestamps)
        positive_dt = positive_dt[np.isfinite(positive_dt) & (positive_dt > 0)]
        overlay_fps = float(1.0 / np.median(positive_dt)) if len(positive_dt) else 15.0
    else:
        overlay_fps = 15.0
    if not np.isfinite(overlay_fps) or overlay_fps <= 0:
        raise ValueError(f"Invalid overlay FPS: {overlay_fps}")
    save_track_video(
        frames,
        masks,
        tracks,
        final_rejection == 0,
        query_times,
        output / "tracks_overlay.mp4",
        overlay_fps,
    )

    failed = sum(record["status"] != "completed" for record in diagnostics)
    pnp_fallbacks = sum(record.get("pnp_status") == "pairwise_fallback" for record in diagnostics)
    pnp_disabled_frames = sum(
        record.get("pnp_status") == "disabled_depth_first_pairwise" for record in diagnostics
    )
    silhouette_ious = np.asarray([record["silhouette_iou"] for record in diagnostics], dtype=np.float64)
    median_silhouette_iou = float(np.median(silhouette_ious))
    q10_silhouette_iou = float(np.quantile(silhouette_ious, 0.10))
    minimum_silhouette_iou = float(silhouette_ious.min())
    # PnP is an optional frame-0 stabilization layer.  A validated pairwise
    # RGB-D pose remains the intended fallback after frame-0 features disappear.
    # Use robust silhouette statistics because short hand occlusions and image
    # boundary clipping make a strict all-frame minimum misleading.
    passed = (
        failed / len(diagnostics) <= args.max_failed_frame_ratio
        and median_silhouette_iou >= args.min_median_silhouette_iou
        and q10_silhouette_iou >= args.min_q10_silhouette_iou
        and minimum_silhouette_iou >= args.min_single_frame_silhouette_iou
    )
    if not args.skip_stage_state_update:
        update_stage_state(
            workspace / "pipeline_state.json",
            "08_rigid_pose_tracking_frame0",
            "completed" if passed else "needs_revision",
            inputs=[str(rgb_dir), str(mask_dir), str(depth_dir), str(poses_path), str(transform0_path)],
            outputs=[str(output)],
            notes=(
                f"CoTracker3 RGB-D rigid tracking completed; failed_frames={failed}, pnp_fallbacks={pnp_fallbacks}, globally_rejected_tracks={int(globally_bad.sum())}."
                if passed
                else f"Rigid tracking needs revision; failed_frames={failed}/{len(diagnostics)}."
            ),
        )
    summary = {
        "status": "completed" if passed else "needs_revision",
        "frame_count": len(frames),
        "query_count": len(queries),
        "global_rejected_tracks": int(globally_bad.sum()),
        "failed_frames": failed,
        "pnp_fallbacks": pnp_fallbacks,
        "pnp_disabled_frames": pnp_disabled_frames,
        "median_valid_tracks": float(np.median([record["valid_track_count"] for record in diagnostics])),
        "median_ransac_inliers": float(np.median([record["ransac_inlier_count"] for record in diagnostics[1:]])),
        "median_silhouette_iou": median_silhouette_iou,
        "q10_silhouette_iou": q10_silhouette_iou,
        "minimum_silhouette_iou": minimum_silhouette_iou,
        "overlay_fps": overlay_fps,
        "output_dir": str(output),
    }
    print(json.dumps(summary, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
