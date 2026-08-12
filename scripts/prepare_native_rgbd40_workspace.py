#!/usr/bin/env python3
"""Prepare a frame-0-compatible workspace from native SpatialMP4 RGB-D pairs.

The source export contains RGB frames sampled at the native depth cadence.  This
script deliberately does not decode the 50 fps RGB stream, resample to 15 fps,
or run/use Video Depth Anything.  Each selected row contributes its native
right-eye RGB image, native metric depth image, RGB/depth timestamps, and poses.

Two projected-depth products are written:

* ``raw_projected_npy``: one-pixel z-buffer projection, kept for calibration
  auditing and reproducibility.
* ``metric_depth_npy``: a small nearest-z splat of the same native points,
  intended for tracking at 1624x1232 where a raw 320x240 projection is sparse.

The temporal camera poses use the RGB image-axis correction already validated
by Stage 00.  Depth-to-RGB projection intentionally uses the uncorrected sensor
extrinsics, matching the existing pipeline's calibration convention.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vlm_sam2_recon.rigid_pipeline.common import read_json, write_csv, write_json  # noqa: E402
from vlm_sam2_recon.rigid_pipeline.geometry import (  # noqa: E402
    PoseSample,
    backproject_depth,
    pose_matrix,
    project_points_zbuffer,
    transform_points,
)


DEFAULT_EXPORT = Path("/tmp/3DVideo_2026-07-28-20-37-34-217_spatialmp4_export")
DEFAULT_SOURCE_WORKSPACE = PROJECT_ROOT / "run_mixed_20260728_203734"
DEFAULT_WORKSPACE = PROJECT_ROOT / "run_mixed_20260728_203734_depth40"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--spatial-export", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--source-workspace", type=Path, default=DEFAULT_SOURCE_WORKSPACE)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-count", type=int, default=40)
    parser.add_argument("--depth-min-m", type=float, default=0.1)
    parser.add_argument("--depth-max-m", type=float, default=5.0)
    parser.add_argument(
        "--tracking-splat-radius",
        type=int,
        default=2,
        help="Pixel radius for nearest-z native-depth splatting at right-RGB resolution.",
    )
    parser.add_argument(
        "--timestamp-tolerance-s",
        type=float,
        default=1e-6,
        help="Maximum delta for reusing old masks/inpainted RGB; RGB pixels must also match exactly.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--depth-cache-dir",
        type=Path,
        default=None,
        help="Backing directory for large NPY files; defaults to /tmp/vlm_sam2_recon_cache/<run>/native_depth.",
    )
    parser.add_argument(
        "--nearest-artifact-max-delta-s",
        type=float,
        default=0.025,
        help="Maximum timestamp delta for temporal-nearest reuse into canonical Stage03/04 inputs.",
    )
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def row_pose(row: dict[str, str], prefix: str) -> np.ndarray:
    timestamp_key = f"{prefix}_timestamp"
    if timestamp_key not in row:
        timestamp_key = f"{prefix}_timestamp_s"
    sample = PoseSample(
        timestamp_s=float(row[timestamp_key]),
        translation=np.asarray([float(row[f"{prefix}_{axis}"]) for axis in ("x", "y", "z")]),
        quaternion_wxyz=np.asarray(
            [float(row[f"{prefix}_q{axis}"]) for axis in ("w", "x", "y", "z")]
        ),
    )
    return pose_matrix(sample)


def decoded_pixels_equal(left: Path, right: Path) -> bool:
    with Image.open(left) as left_image, Image.open(right) as right_image:
        left_rgb = np.asarray(left_image.convert("RGB"))
        right_rgb = np.asarray(right_image.convert("RGB"))
    return left_rgb.shape == right_rgb.shape and np.array_equal(left_rgb, right_rgb)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remove_owned_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def prepare_output_roots(workspace: Path, overwrite: bool) -> None:
    if workspace in {Path("/"), PROJECT_ROOT, PROJECT_ROOT.parent}:
        raise ValueError(f"Refusing unsafe workspace path: {workspace}")
    owned = [
        workspace / "outputs/00_rgb_frames",
        workspace / "outputs/00_native_reuse",
        workspace / "outputs/01_vlm",
        workspace / "outputs/03_sam3d_frame0",
        workspace / "outputs/06_dense_depth",
        workspace / "outputs/07_alignment",
        workspace / "configs",
        workspace / "pipeline_state.json",
    ]
    existing = [path for path in owned if path.exists() or path.is_symlink()]
    if existing and not overwrite:
        raise FileExistsError(
            "Native RGB-D workspace outputs already exist; pass --overwrite: "
            + ", ".join(str(path) for path in existing)
        )
    if overwrite:
        for path in existing:
            remove_owned_path(path)
    (workspace / "outputs").mkdir(parents=True, exist_ok=True)
    (workspace / "configs").mkdir(parents=True, exist_ok=True)


def native_fps(timestamps: np.ndarray) -> tuple[float, float]:
    if len(timestamps) < 2:
        return 0.0, 0.0
    deltas = np.diff(timestamps)
    if np.any(deltas <= 0):
        raise ValueError("Native RGB timestamps are not strictly increasing")
    median_fps = 1.0 / float(np.median(deltas))
    span_fps = float(len(timestamps) - 1) / float(timestamps[-1] - timestamps[0])
    return median_fps, span_fps


def nearest_z_splat(depth: np.ndarray, radius: int) -> np.ndarray:
    """Splat sparse projected depth; overlapping surfaces keep the nearest z."""
    depth = np.asarray(depth, dtype=np.float32)
    if radius <= 0:
        return depth.copy()
    height, width = depth.shape
    ys, xs = np.nonzero(np.isfinite(depth) & (depth > 0))
    zs = depth[ys, xs]
    output = np.full(height * width, np.inf, dtype=np.float32)
    for dy in range(-radius, radius + 1):
        yy = ys + dy
        valid_y = (yy >= 0) & (yy < height)
        if not np.any(valid_y):
            continue
        for dx in range(-radius, radius + 1):
            xx = xs + dx
            valid = valid_y & (xx >= 0) & (xx < width)
            if not np.any(valid):
                continue
            flat = yy[valid] * width + xx[valid]
            np.minimum.at(output, flat, zs[valid])
    output[~np.isfinite(output)] = 0.0
    return output.reshape(height, width)


def project_native_depth(
    row: dict[str, str],
    spatial_export: Path,
    camera: dict[str, Any],
    depth_min_m: float,
    depth_max_m: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    depth_path = spatial_export / row["depth_meters_npy"]
    depth = np.load(depth_path).astype(np.float32)
    expected_shape = (int(camera["depth_height"]), int(camera["depth_width"]))
    if depth.shape != expected_shape:
        raise ValueError(f"Unexpected native depth shape {depth.shape}; expected {expected_shape}: {depth_path}")
    valid = np.isfinite(depth) & (depth >= depth_min_m) & (depth <= depth_max_m)
    points_d, _ = backproject_depth(np.where(valid, depth, 0.0), camera["depth_intrinsics"])

    # Spatial calibration maps the physical depth/RGB sensors through the head
    # frame.  Do not apply the RGB image-axis correction here: that correction is
    # only for temporal camera-pose composition, not cross-sensor calibration.
    t_w_from_d = row_pose(row, "depth_pose") @ np.asarray(camera["T_H_from_D"], dtype=np.float64)
    t_w_from_c = row_pose(row, "rgb_pose") @ np.asarray(
        camera["T_H_from_C_right"], dtype=np.float64
    )
    t_c_from_d = np.linalg.inv(t_w_from_c) @ t_w_from_d
    points_c = transform_points(points_d, t_c_from_d)
    projected = project_points_zbuffer(
        points_c,
        camera["rgb_intrinsics_right"],
        int(camera["rgb_width_per_eye"]),
        int(camera["rgb_height_per_eye"]),
    )
    report = {
        "source_depth": str(depth_path),
        "valid_source_depth_pixels": int(valid.sum()),
        "projected_valid_pixels": int(np.count_nonzero(projected)),
        "T_Ct_rgb_from_Dt_depth": t_c_from_d,
    }
    return depth, projected, report


def link_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    destination.symlink_to(source.resolve())


def link_directory(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        remove_owned_path(destination)
    destination.symlink_to(source.resolve(), target_is_directory=True)


def remap_index(old_index: int, old_times: np.ndarray, native_times: np.ndarray) -> int:
    if not 0 <= old_index < len(old_times):
        raise IndexError(f"Old frame index outside timeline: {old_index}")
    return int(np.argmin(np.abs(native_times - old_times[old_index])))


def remap_vlm_result(
    source_path: Path,
    destination: Path,
    old_times: np.ndarray,
    native_times: np.ndarray,
    workspace: Path,
) -> dict[str, Any]:
    payload = copy.deepcopy(read_json(source_path))
    metadata = payload.setdefault("metadata", {})
    old_frame_count = metadata.get("frame_count")
    old_sampled = metadata.get("sampled_global_frames", [])
    metadata.update(
        {
            "frame_count": int(len(native_times)),
            "sampled_global_frames": list(
                dict.fromkeys(remap_index(int(index), old_times, native_times) for index in old_sampled)
            ),
            "modeling_global_frame": 0,
            "source_result": str(source_path),
            "source_frame_count": old_frame_count,
            "frame_index_remap": "old frame timestamp -> nearest native RGB-D timestamp",
            "native_rgbd_only": True,
        }
    )
    result = payload.get("vlm_result", payload)
    for event in result.get("events", []):
        event["source_15fps_frames"] = {
            "start_frame": int(event["start_frame"]),
            "end_frame": int(event["end_frame"]),
            "evidence_frames": [int(index) for index in event.get("evidence_frames", [])],
        }
        event["start_frame"] = remap_index(int(event["start_frame"]), old_times, native_times)
        event["end_frame"] = remap_index(int(event["end_frame"]), old_times, native_times)
        event["evidence_frames"] = list(
            dict.fromkeys(
                remap_index(int(index), old_times, native_times)
                for index in event.get("evidence_frames", [])
            )
        )
        completion = event.get("articulation_completion")
        if isinstance(completion, dict):
            frame_keys = (
                "first_motion_frame",
                "terminal_reached_frame",
                "confirmation_frame",
            )
            source_completion = {
                key: completion.get(key)
                for key in frame_keys
                if completion.get(key) is not None
            }
            source_completion["completion_evidence_frames"] = [
                int(index) for index in completion.get("completion_evidence_frames", [])
            ]
            completion["source_15fps_frames"] = source_completion
            for key in frame_keys:
                value = completion.get(key)
                if value is not None:
                    completion[key] = remap_index(int(value), old_times, native_times)
            completion["completion_evidence_frames"] = list(
                dict.fromkeys(
                    remap_index(int(index), old_times, native_times)
                    for index in completion.get("completion_evidence_frames", [])
                )
            )
    for obj in result.get("objects", []):
        source_evidence = [int(index) for index in obj.get("manipulation_evidence_frames", [])]
        obj["source_15fps_manipulation_evidence_frames"] = source_evidence
        obj["manipulation_evidence_frames"] = list(
            dict.fromkeys(remap_index(index, old_times, native_times) for index in source_evidence)
        )
        frame0 = obj.get("global_frame0")
        if isinstance(frame0, dict):
            frame0["frame_index"] = 0
            frame0["timestamp_s"] = float(native_times[0])
            frame0["frame_path"] = str(
                workspace / "outputs/00_rgb_frames/right_rgb_png/000000.png"
            )
    write_json(destination, payload)
    return payload


def write_configs(
    workspace: Path,
    source_workspace: Path,
    spatial_export: Path,
    effective_fps: float,
) -> None:
    run_id = workspace.name.removeprefix("run_")
    mixed_source = read_json(source_workspace / "configs/mixed_recon_config.json")
    mixed = copy.deepcopy(mixed_source)
    mixed.update(
        {
            "run_id": run_id,
            "workspace_dir": str(workspace),
            "spatial_export_root": str(spatial_export),
            "camera_id": "right",
            "tracker_fps": effective_fps,
            "input_policy": "native SpatialMP4 RGB-D rows 0..39; no 15fps RGB and no VDA depth",
        }
    )
    write_json(workspace / "configs/mixed_recon_config.json", mixed)

    rigid_source = read_json(source_workspace / "configs/rigid_recon_config.json")
    rigid = copy.deepcopy(rigid_source)
    rigid.update(
        {
            "run_id": run_id,
            "workspace_dir": str(workspace),
            "spatial_export_root": str(spatial_export),
            "camera_id": "right",
            "tracker_fps": effective_fps,
            "true_depth_fps": effective_fps,
            "depth_policy": "native metric depth only; nearest-z radius-2 splat for tracking; VDA prohibited",
        }
    )
    rigid.setdefault("inputs", {}).update(
        {
            "true_depth_dir": str(spatial_export / "depth_meters_npy"),
            "true_depth_timeline": str(spatial_export / "frames.csv"),
            "head_pose_csv": str(spatial_export / "pose/head_pose.csv"),
            "head_pose_jsonl": str(spatial_export / "pose/head_pose.jsonl"),
        }
    )
    rigid.setdefault("policies", {}).update(
        {
            "dense_depth_source": "native_metric_only_no_vda",
            "tracking_depth": "nearest_z_splat_radius_2_of_native_projection",
        }
    )
    write_json(workspace / "configs/rigid_recon_config.json", rigid)


def write_pipeline_state(
    workspace: Path,
    source_workspace: Path,
    spatial_export: Path,
    frame_count: int,
) -> None:
    state = copy.deepcopy(read_json(source_workspace / "pipeline_state.json"))
    state.update(
        {
            "run_id": workspace.name.removeprefix("run_"),
            "workspace_dir": str(workspace),
            "spatial_export_root": str(spatial_export),
            "camera_id": "right",
            "native_rgbd_policy": {
                "source_indices": [0, frame_count - 1],
                "frame_count": frame_count,
                "rgb": "native right-eye export PNG",
                "depth": "native metric depth projected to right RGB; no VDA",
                "tracking_splat": "radius 2, nearest z",
            },
        }
    )
    completed = {
        "00_rgb_extract": "Prepared 40 native right-eye RGB-D pairs; no 15fps resampling and no VDA.",
        "01_vlm_mixed_interactions": "Reused semantics and remapped old event indices by timestamp onto the native RGB-D timeline.",
        "02_hand_masks": "Tracking compatibility masks were remapped from the nearest old timestamp (max 25 ms); provenance is explicit.",
        "03_diffueraser_hand_removal": "Tracking compatibility inpainted RGB was remapped from the nearest old timestamp (max 25 ms); native RGB/depth/pose remain authoritative.",
        "04_sam2_object_masks": "Bottle and microwave masks were remapped from the nearest old timestamp (max 25 ms) for direct Stage08 use.",
        "05_sam3d_frame0_reconstruction": "Reused source run because native frame 0 is pixel-identical.",
        "06_dense_depth_metric_calibration": "Projected native metric depth for every frame; tracking depth is nearest-z radius-2 splat only.",
        "07_frame0_multi_object_alignment": "Reused source frame-0 alignment in the identical right-camera C0 frame.",
    }
    for stage in state.get("stages", []):
        name = str(stage.get("stage"))
        if name in completed:
            stage["status"] = "completed"
            stage["notes"] = completed[name]
        else:
            stage["status"] = "pending"
            stage["notes"] = "Must run on the native 40-frame timeline; old 15fps artifacts are not accepted wholesale."
        stage["inputs"] = []
        stage["outputs"] = []
    write_json(workspace / "pipeline_state.json", state)


def main() -> int:
    args = parse_args()
    workspace = args.workspace.resolve()
    spatial_export = args.spatial_export.resolve()
    source_workspace = args.source_workspace.resolve()
    if args.frame_start < 0 or args.frame_count <= 0:
        raise ValueError("--frame-start must be >=0 and --frame-count must be positive")
    if args.tracking_splat_radius < 0:
        raise ValueError("--tracking-splat-radius must be >=0")
    required = [
        spatial_export / "frames.csv",
        spatial_export / "manifest.json",
        source_workspace / "outputs/00_rgb_frames/camera.json",
        source_workspace / "outputs/00_rgb_frames/timeline.csv",
        source_workspace / "outputs/01_vlm/mixed_interactions.json",
        source_workspace / "outputs/03_sam3d_frame0",
        source_workspace / "outputs/07_alignment",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    prepare_output_roots(workspace, args.overwrite)

    all_rows = sorted(read_rows(spatial_export / "frames.csv"), key=lambda row: int(row["index"]))
    selected = all_rows[args.frame_start : args.frame_start + args.frame_count]
    if len(selected) != args.frame_count:
        raise ValueError(f"Requested {args.frame_count} rows, found {len(selected)}")
    expected_indices = list(range(args.frame_start, args.frame_start + args.frame_count))
    actual_indices = [int(row["index"]) for row in selected]
    if actual_indices != expected_indices:
        raise ValueError(f"Native export indices are not contiguous: {actual_indices}")

    camera = read_json(source_workspace / "outputs/00_rgb_frames/camera.json")
    if camera.get("selected_eye") != "right":
        raise ValueError("Source calibration is not the right eye")
    native_times = np.asarray([float(row["rgb_timestamp"]) for row in selected], dtype=np.float64)
    effective_fps, span_fps = native_fps(native_times)

    stage00 = workspace / "outputs/00_rgb_frames"
    png_dir = stage00 / "right_rgb_png"
    jpeg_dir = stage00 / "sam2_jpeg"
    depth_output_root = workspace / "outputs/06_dense_depth"
    depth_output_root.mkdir(parents=True, exist_ok=True)
    depth_cache_root = (
        args.depth_cache_dir.resolve()
        if args.depth_cache_dir is not None
        else Path("/tmp/vlm_sam2_recon_cache") / workspace.name / "native_depth"
    )
    if args.overwrite and depth_cache_root.exists():
        if depth_cache_root in {Path("/"), Path("/tmp"), Path("/tmp/vlm_sam2_recon_cache")}:
            raise ValueError(f"Refusing unsafe depth cache path: {depth_cache_root}")
        shutil.rmtree(depth_cache_root)
    raw_cache_dir = depth_cache_root / "raw_projected_npy"
    compatibility_raw_cache_dir = depth_cache_root / "true_depth_projected"
    metric_cache_dir = depth_cache_root / "metric_depth_npy"
    for path in (png_dir, jpeg_dir, raw_cache_dir, compatibility_raw_cache_dir, metric_cache_dir):
        path.mkdir(parents=True, exist_ok=True)
    raw_dir = depth_output_root / "raw_projected_npy"
    compatibility_raw_dir = depth_output_root / "true_depth_projected"
    metric_dir = depth_output_root / "metric_depth_npy"
    link_directory(raw_cache_dir, raw_dir)
    link_directory(compatibility_raw_cache_dir, compatibility_raw_dir)
    link_directory(metric_cache_dir, metric_dir)

    t_h_from_c_pose = np.asarray(camera["T_H_from_C_right_pose"], dtype=np.float64)
    t_w_from_h_all: list[np.ndarray] = []
    t_w_from_c_all: list[np.ndarray] = []
    t_c0_from_ct_all: list[np.ndarray] = []
    projection_records: list[dict[str, Any]] = []
    timeline_rows: list[dict[str, Any]] = []
    t_w_from_c0: np.ndarray | None = None

    for frame_index, row in enumerate(selected):
        source_index = int(row["index"])
        source_rgb = spatial_export / row["right_rgb_png"]
        if not source_rgb.is_file():
            raise FileNotFoundError(source_rgb)
        output_rgb = png_dir / f"{frame_index:06d}.png"
        shutil.copy2(source_rgb, output_rgb)
        with Image.open(source_rgb) as image:
            rgb = image.convert("RGB")
            expected_size = (int(camera["rgb_width_per_eye"]), int(camera["rgb_height_per_eye"]))
            if rgb.size != expected_size:
                raise ValueError(f"Unexpected native right RGB size {rgb.size}; expected {expected_size}")
            rgb.save(
                jpeg_dir / f"{frame_index:06d}.jpg",
                quality=args.jpeg_quality,
                subsampling=0,
            )

        _, raw_projected, depth_report = project_native_depth(
            row,
            spatial_export,
            camera,
            args.depth_min_m,
            args.depth_max_m,
        )
        tracking_depth = nearest_z_splat(raw_projected, args.tracking_splat_radius)
        raw_path = raw_dir / f"{frame_index:06d}.npy"
        compatibility_raw_path = (
            compatibility_raw_dir
            / f"depth_{source_index:06d}_to_rgb_{frame_index:06d}.npy"
        )
        metric_path = metric_dir / f"{frame_index:06d}.npy"
        np.save(raw_path, raw_projected)
        link_file(raw_path, compatibility_raw_path)
        np.save(metric_path, tracking_depth)

        t_w_from_h = row_pose(row, "rgb_pose")
        t_w_from_c = t_w_from_h @ t_h_from_c_pose
        if t_w_from_c0 is None:
            t_w_from_c0 = t_w_from_c
        t_c0_from_ct = np.linalg.inv(t_w_from_c0) @ t_w_from_c
        t_w_from_h_all.append(t_w_from_h)
        t_w_from_c_all.append(t_w_from_c)
        t_c0_from_ct_all.append(t_c0_from_ct)

        rgb_timestamp = float(row["rgb_timestamp"])
        depth_timestamp = float(row["depth_timestamp"])
        projection_record = {
            "frame_index": frame_index,
            "source_export_index": source_index,
            "rgb_timestamp_s": rgb_timestamp,
            "depth_timestamp_s": depth_timestamp,
            "depth_minus_rgb_s": depth_timestamp - rgb_timestamp,
            **depth_report,
            "raw_projected_depth": str(raw_path),
            "tracking_metric_depth": str(metric_path),
            "tracking_splat_radius_px": args.tracking_splat_radius,
            "tracking_valid_pixels": int(np.count_nonzero(tracking_depth)),
        }
        projection_records.append(projection_record)
        timeline_rows.append(
            {
                "frame_index": frame_index,
                "rgb_timestamp_s": f"{rgb_timestamp:.12f}",
                "source_rgb_index": source_index,
                "source_export_index": source_index,
                "right_rgb_png": f"right_rgb_png/{frame_index:06d}.png",
                "sam2_jpeg": f"sam2_jpeg/{frame_index:06d}.jpg",
                "true_depth_nearest_index": source_index,
                "true_depth_timestamp_s": f"{depth_timestamp:.12f}",
                "true_depth_delta_s": f"{depth_timestamp - rgb_timestamp:.12f}",
                "true_depth_meters_npy": str(spatial_export / row["depth_meters_npy"]),
                "raw_projected_depth_npy": str(raw_path),
                "metric_depth_npy": str(metric_path),
                "rgb_pose_timestamp_s": row["rgb_pose_timestamp"],
                "depth_pose_timestamp_s": row["depth_pose_timestamp"],
            }
        )
        print(
            f"[{frame_index + 1:02d}/{len(selected)}] native={source_index:02d} "
            f"rgb={rgb_timestamp:.6f} depth={depth_timestamp:.6f} "
            f"raw={depth_report['projected_valid_pixels']} splat={projection_record['tracking_valid_pixels']}",
            flush=True,
        )

    if t_w_from_c0 is None:
        raise RuntimeError("No frames were prepared")
    if not np.allclose(t_c0_from_ct_all[0], np.eye(4), atol=1e-10):
        raise RuntimeError("First temporal pose is not identity in C0")

    write_csv(stage00 / "timeline.csv", timeline_rows, list(timeline_rows[0]))
    np.savez_compressed(
        stage00 / "poses.npz",
        T_W_from_H=np.asarray(t_w_from_h_all, dtype=np.float64),
        T_W_from_C=np.asarray(t_w_from_c_all, dtype=np.float64),
        T_C0_from_Ct=np.asarray(t_c0_from_ct_all, dtype=np.float64),
        rgb_timestamps_s=native_times,
        depth_timestamps_s=np.asarray(
            [float(row["depth_timestamp"]) for row in selected], dtype=np.float64
        ),
        source_export_indices=np.asarray(actual_indices, dtype=np.int64),
    )
    camera.update(
        {
            "selected_eye": "right",
            "native_rgbd_frame_count": len(selected),
            "native_rgbd_source_indices": [actual_indices[0], actual_indices[-1]],
            "temporal_pose_frame": "frame0_right_camera_opencv_rdf",
            "depth_projection_policy": (
                "T_Crgb_from_Ddepth = inv(T_W_H(rgb_ts)*T_H_Cright) * "
                "(T_W_H(depth_ts)*T_H_D); no RGB pose-axis correction in cross-sensor projection"
            ),
        }
    )
    write_json(stage00 / "camera.json", camera)

    native_video = stage00 / "right_rgb_native40.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            f"{effective_fps:.12f}",
            "-i",
            str(png_dir / "%06d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "16",
            str(native_video),
        ],
        check=True,
    )
    # Several legacy stage scripts still use this filename.  The target is the
    # native-cadence video; no 15 fps RGB has been generated or copied.
    link_file(native_video, stage00 / "right_rgb_15fps.mp4")

    old_timeline = read_rows(source_workspace / "outputs/00_rgb_frames/timeline.csv")
    old_times = np.asarray([float(row["rgb_timestamp_s"]) for row in old_timeline], dtype=np.float64)
    reuse_root = workspace / "outputs/00_native_reuse"
    artifact_sources = {
        "hand_combined": source_workspace / "outputs/02_hand_masks/combined",
        "hand_object": source_workspace / "outputs/02_hand_masks/objects/hand",
        "sam2_bottle": source_workspace / "outputs/02_sam2_frame0_masks/propagated/objects/bottle",
        "sam2_microwave": source_workspace / "outputs/02_sam2_frame0_masks/propagated/objects/microwave",
        "diffueraser_rgb": source_workspace / "outputs/03_diffueraser/inpainted_frames_png",
        "stage04_bottle": source_workspace / "outputs/04_object_masks/bottle/objects/bottle",
        "stage04_microwave": source_workspace / "outputs/04_object_masks/microwave/objects/microwave",
    }
    reuse_records: list[dict[str, Any]] = []
    for native_index, native_time in enumerate(native_times):
        old_index = int(np.argmin(np.abs(old_times - native_time)))
        delta = float(old_times[old_index] - native_time)
        native_rgb = png_dir / f"{native_index:06d}.png"
        old_rgb = source_workspace / f"outputs/00_rgb_frames/right_rgb_png/{old_index:06d}.png"
        timestamp_equivalent = abs(delta) <= args.timestamp_tolerance_s
        pixels_equal = timestamp_equivalent and old_rgb.is_file() and decoded_pixels_equal(native_rgb, old_rgb)
        linked: dict[str, str] = {}
        if pixels_equal:
            for name, source_dir in artifact_sources.items():
                source = source_dir / f"{old_index:06d}.png"
                if source.is_file():
                    destination = reuse_root / name / f"{native_index:06d}.png"
                    link_file(source, destination)
                    linked[name] = str(destination)
        reuse_records.append(
            {
                "native_frame_index": native_index,
                "native_timestamp_s": float(native_time),
                "old_nearest_frame_index": old_index,
                "old_nearest_timestamp_s": float(old_times[old_index]),
                "old_minus_native_s": delta,
                "timestamp_equivalent": timestamp_equivalent,
                "decoded_right_rgb_pixels_equal": pixels_equal,
                "linked_reference_artifacts": linked,
            }
        )

    for prompt in (
        source_workspace / "outputs/02_hand_masks/hand_prompts.json",
        source_workspace / "outputs/04_object_masks/bottle/object_prompts.json",
        source_workspace / "outputs/04_object_masks/microwave/object_prompts.json",
    ):
        if prompt.is_file():
            link_file(prompt, reuse_root / "prompts" / prompt.parent.name / prompt.name)
    write_json(
        reuse_root / "reuse_manifest.json",
        {
            "policy": (
                "Reference-only reuse requires timestamp equivalence and decoded RGB pixel identity. "
                "Unmatched frames are intentionally absent; SAM2 and DiffuEraser must run on the native sequence."
            ),
            "timestamp_tolerance_s": args.timestamp_tolerance_s,
            "records": reuse_records,
        },
    )

    # Tracking-ready compatibility inputs.  Unlike the strict audit above,
    # these use the nearest old 15 fps artifact when it is within 25 ms.  The
    # native RGB, native metric depth, and native poses remain authoritative;
    # only hand-removed appearance and masks are temporally borrowed.
    canonical_sources = {
        "diffueraser_rgb": source_workspace / "outputs/03_diffueraser/inpainted_frames_png",
        "hand": source_workspace / "outputs/02_hand_masks/objects/hand",
        "bottle": source_workspace / "outputs/04_object_masks/bottle/objects/bottle",
        "microwave": source_workspace / "outputs/04_object_masks/microwave/objects/microwave",
    }
    canonical_destinations = {
        "diffueraser_rgb": [workspace / "outputs/03_diffueraser/inpainted_frames_png"],
        "hand": [
            workspace / "outputs/02_hand_masks/objects/hand",
            workspace / "outputs/02_hand_masks/combined",
        ],
        "bottle": [
            workspace / "outputs/04_object_masks/bottle/objects/bottle",
            workspace / "outputs/04_object_masks/bottle/combined",
            workspace / "outputs/02_sam2_frame0_masks/propagated/objects/bottle",
        ],
        "microwave": [
            workspace / "outputs/04_object_masks/microwave/objects/microwave",
            workspace / "outputs/04_object_masks/microwave/combined",
            workspace / "outputs/02_sam2_frame0_masks/propagated/objects/microwave",
        ],
    }
    temporal_nearest_records: list[dict[str, Any]] = []
    for native_index, native_time in enumerate(native_times):
        old_index = int(np.argmin(np.abs(old_times - native_time)))
        delta = float(old_times[old_index] - native_time)
        if abs(delta) > args.nearest_artifact_max_delta_s:
            raise RuntimeError(
                f"No old artifact within {args.nearest_artifact_max_delta_s}s for native frame "
                f"{native_index}: nearest delta={delta:+.6f}s"
            )
        linked: dict[str, list[str]] = {}
        for name, source_dir in canonical_sources.items():
            source = source_dir / f"{old_index:06d}.png"
            if not source.is_file():
                if name in {"diffueraser_rgb", "bottle"}:
                    raise FileNotFoundError(source)
                continue
            linked[name] = []
            for destination_dir in canonical_destinations[name]:
                destination = destination_dir / f"{native_index:06d}.png"
                link_file(source, destination)
                linked[name].append(str(destination))
        temporal_nearest_records.append(
            {
                "native_frame_index": native_index,
                "native_timestamp_s": float(native_time),
                "old_frame_index": old_index,
                "old_timestamp_s": float(old_times[old_index]),
                "old_minus_native_s": delta,
                "absolute_delta_s": abs(delta),
                "reuse_class": "exact_timestamp_pixel_equal"
                if reuse_records[native_index]["decoded_right_rgb_pixels_equal"]
                else "temporal_nearest",
                "linked": linked,
            }
        )

    inpainted_dir = workspace / "outputs/03_diffueraser/inpainted_frames_png"
    inpainted_video = workspace / "outputs/03_diffueraser/inpainted_right_rgb_native40.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            f"{effective_fps:.12f}",
            "-i",
            str(inpainted_dir / "%06d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "16",
            str(inpainted_video),
        ],
        check=True,
    )
    link_file(
        inpainted_video,
        workspace / "outputs/03_diffueraser/inpainted_right_rgb_15fps.mp4",
    )
    canonical_reuse_manifest = {
        "status": "completed",
        "policy": "temporal_nearest_old_15fps_artifact_for_native_tracking_compatibility",
        "max_absolute_delta_s": args.nearest_artifact_max_delta_s,
        "authoritative_geometry": "native metric depth plus native pose; no VDA",
        "warning": (
            "Inpainted RGB and masks are nearest-timestamp compatibility artifacts. "
            "They are not claimed to be native-frame SAM2/DiffuEraser inference."
        ),
        "records": temporal_nearest_records,
    }
    write_json(workspace / "outputs/temporal_nearest_reuse_manifest.json", canonical_reuse_manifest)
    write_json(
        workspace / "outputs/03_diffueraser/diffueraser_manifest.json",
        {
            **canonical_reuse_manifest,
            "stage": "03_diffueraser_hand_removal",
            "output_frames": str(inpainted_dir),
            "output_video": str(inpainted_video),
            "inference_reused": True,
        },
    )
    for object_id in ("bottle", "microwave"):
        write_json(
            workspace / f"outputs/04_object_masks/{object_id}/temporal_nearest_manifest.json",
            {
                **canonical_reuse_manifest,
                "stage": "04_sam2_object_masks",
                "object_id": object_id,
                "mask_directory": str(
                    workspace / f"outputs/04_object_masks/{object_id}/objects/{object_id}"
                ),
            },
        )

    vlm_output = workspace / "outputs/01_vlm/mixed_interactions.json"
    remapped_vlm = remap_vlm_result(
        source_workspace / "outputs/01_vlm/mixed_interactions.json",
        vlm_output,
        old_times,
        native_times,
        workspace,
    )
    write_json(
        workspace / "outputs/01_vlm/vlm_reuse_manifest.json",
        {
            "source": str(source_workspace / "outputs/01_vlm/mixed_interactions.json"),
            "destination": str(vlm_output),
            "policy": "Reuse semantic interpretation; remap every referenced frame through timestamps.",
            "events": remapped_vlm.get("vlm_result", remapped_vlm).get("events", []),
        },
    )

    frame0_native = png_dir / "000000.png"
    frame0_old = source_workspace / "outputs/00_rgb_frames/right_rgb_png/000000.png"
    frame0_rgb_equal = decoded_pixels_equal(frame0_native, frame0_old)
    if not frame0_rgb_equal:
        raise RuntimeError("Native frame 0 is not pixel-identical to the source workspace; cannot reuse C0 meshes")
    old_raw_frame0 = source_workspace / "outputs/06_dense_depth/true_depth_projected/depth_000000_to_rgb_000000.npy"
    new_raw_frame0 = raw_dir / "000000.npy"
    frame0_depth_check: dict[str, Any] = {"old_raw_exists": old_raw_frame0.is_file()}
    if old_raw_frame0.is_file():
        old_raw = np.load(old_raw_frame0)
        new_raw = np.load(new_raw_frame0)
        common_valid = (old_raw > 0) & (new_raw > 0)
        frame0_depth_check.update(
            {
                "shape_equal": old_raw.shape == new_raw.shape,
                "arrays_equal": np.array_equal(old_raw, new_raw),
                "max_abs_m": float(np.max(np.abs(old_raw - new_raw))),
                "common_valid_pixels": int(common_valid.sum()),
            }
        )
    link_directory(
        source_workspace / "outputs/03_sam3d_frame0",
        workspace / "outputs/03_sam3d_frame0",
    )
    link_directory(
        source_workspace / "outputs/07_alignment",
        workspace / "outputs/07_alignment",
    )

    write_configs(workspace, source_workspace, spatial_export, effective_fps)
    write_pipeline_state(workspace, source_workspace, spatial_export, len(selected))

    projection_manifest = {
        "stage": "06_native_metric_depth_projection",
        "status": "completed",
        "source": str(spatial_export),
        "source_indices": [actual_indices[0], actual_indices[-1]],
        "frame_count": len(selected),
        "depth_units": "meters_float32",
        "depth_range_m": [args.depth_min_m, args.depth_max_m],
        "vda_used": False,
        "raw_projection": {
            "directory": str(raw_dir),
            "policy": "single-pixel nearest-z z-buffer of native 320x240 metric depth",
        },
        "tracking_metric_depth": {
            "directory": str(metric_dir),
            "policy": "nearest-z square splat of raw native projection",
            "radius_px": args.tracking_splat_radius,
            "diameter_px": 2 * args.tracking_splat_radius + 1,
            "overlap_policy": "nearest z wins",
        },
        "records": projection_records,
    }
    write_json(workspace / "outputs/06_dense_depth/native_depth_manifest.json", projection_manifest)

    mapping_rows = []
    for timeline, reuse in zip(timeline_rows, reuse_records, strict=True):
        mapping_rows.append(
            {
                **timeline,
                "old_nearest_frame_index": reuse["old_nearest_frame_index"],
                "old_nearest_timestamp_s": f"{reuse['old_nearest_timestamp_s']:.12f}",
                "old_minus_native_s": f"{reuse['old_minus_native_s']:.12f}",
                "old_timestamp_equivalent": reuse["timestamp_equivalent"],
                "old_rgb_pixels_equal": reuse["decoded_right_rgb_pixels_equal"],
            }
        )
    write_csv(stage00 / "native40_mapping.csv", mapping_rows, list(mapping_rows[0]))

    manifest = {
        "stage": "00_native_rgbd40_prepare",
        "status": "completed",
        "workspace": str(workspace),
        "spatial_export": str(spatial_export),
        "selected_eye": "right",
        "selection": {
            "source_indices": [actual_indices[0], actual_indices[-1]],
            "frame_count": len(selected),
            "policy": "take native exported RGB-D rows directly; no RGB resampling",
        },
        "timestamps": {
            "first_rgb_s": float(native_times[0]),
            "last_rgb_s": float(native_times[-1]),
            "median_interval_fps": effective_fps,
            "full_span_fps": span_fps,
        },
        "rgb": {
            "source": "spatial export rgb_right_png",
            "directory": str(png_dir),
            "video": str(native_video),
            "legacy_filename_symlink": str(stage00 / "right_rgb_15fps.mp4"),
            "legacy_filename_note": "symlink target is native cadence; no 15fps frames are used",
        },
        "depth": {
            "source": "spatial export depth_meters_npy",
            "vda_used": False,
            "raw_projected_directory": str(raw_dir),
            "tracking_metric_directory": str(metric_dir),
            "tracking_splat_radius_px": args.tracking_splat_radius,
            "overlap_policy": "nearest z",
        },
        "poses": {
            "path": str(stage00 / "poses.npz"),
            "temporal_transform": "T_C0_from_Ct",
            "coordinate_frame": "frame0_right_camera_opencv_rdf",
        },
        "frame0_reuse": {
            "source_workspace": str(source_workspace),
            "decoded_rgb_pixels_equal": frame0_rgb_equal,
            "native_rgb_sha256": sha256(frame0_native),
            "source_rgb_sha256": sha256(frame0_old),
            "raw_depth_projection_check": frame0_depth_check,
            "sam3d": str(workspace / "outputs/03_sam3d_frame0"),
            "alignment": str(workspace / "outputs/07_alignment"),
        },
        "timestamp_equivalent_old_artifacts": {
            "matched_frame_count": int(sum(item["decoded_right_rgb_pixels_equal"] for item in reuse_records)),
            "reference_root": str(reuse_root),
            "policy": "reference only; rerun propagation/inpainting for the complete native sequence",
        },
        "tracking_compatibility_artifacts": {
            "policy": "temporal_nearest from old 15fps artifacts",
            "max_absolute_delta_s": args.nearest_artifact_max_delta_s,
            "frame_count": len(temporal_nearest_records),
            "manifest": str(workspace / "outputs/temporal_nearest_reuse_manifest.json"),
            "inpainted_rgb": str(inpainted_dir),
            "bottle_masks": str(workspace / "outputs/04_object_masks/bottle/objects/bottle"),
            "provenance_warning": "appearance/masks are approximate; RGB-D geometry and poses are native",
        },
        "outputs": {
            "timeline": str(stage00 / "timeline.csv"),
            "mapping": str(stage00 / "native40_mapping.csv"),
            "camera": str(stage00 / "camera.json"),
            "poses": str(stage00 / "poses.npz"),
            "depth_manifest": str(workspace / "outputs/06_dense_depth/native_depth_manifest.json"),
            "vlm": str(vlm_output),
        },
    }
    write_json(stage00 / "stage00_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
