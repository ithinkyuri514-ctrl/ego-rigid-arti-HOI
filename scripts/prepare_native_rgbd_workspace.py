#!/usr/bin/env python3
"""Prepare an exact depth-matched SpatialMP4 RGB-D timeline for the pipeline."""

from __future__ import annotations

import argparse
import csv
import json
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

from vlm_sam2_recon.rigid_pipeline.common import update_stage_state, write_csv, write_json  # noqa: E402
from vlm_sam2_recon.rigid_pipeline.geometry import (  # noqa: E402
    PoseSample,
    backproject_depth,
    pose_matrix,
    project_points_zbuffer,
    transform_points,
)
from vlm_sam2_recon.stages.camera_alignment import rgb_pose_axis_correction  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--spatial-export", type=Path, required=True)
    parser.add_argument("--eye", choices=["left", "right"], default="right")
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-count", type=int, default=None)
    parser.add_argument("--depth-min-m", type=float, default=0.1)
    parser.add_argument("--depth-max-m", type=float, default=5.0)
    parser.add_argument("--tracking-splat-radius", type=int, default=2)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--depth-cache-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def row_value(row: dict[str, str], *names: str) -> str:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    raise KeyError(f"None of {names} found in SpatialMP4 row")


def row_pose(row: dict[str, str], prefix: str) -> np.ndarray:
    sample = PoseSample(
        timestamp_s=float(row_value(row, f"{prefix}_timestamp_s", f"{prefix}_timestamp")),
        translation=np.asarray([float(row[f"{prefix}_{axis}"]) for axis in "xyz"]),
        quaternion_wxyz=np.asarray([float(row[f"{prefix}_q{axis}"]) for axis in "wxyz"]),
    )
    return pose_matrix(sample)


def nearest_z_splat(depth: np.ndarray, radius: int) -> np.ndarray:
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
        for dx in range(-radius, radius + 1):
            xx = xs + dx
            valid = valid_y & (xx >= 0) & (xx < width)
            if np.any(valid):
                np.minimum.at(output, yy[valid] * width + xx[valid], zs[valid])
    output[~np.isfinite(output)] = 0.0
    return output.reshape(height, width)


def remove_owned(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def link_directory(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        remove_owned(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source.resolve(), target_is_directory=True)


def camera_from_manifest(manifest: dict[str, Any], eye: str) -> dict[str, Any]:
    intrinsics = dict(manifest[f"rgb_intrinsics_{eye}"])
    t_h_from_c = np.asarray(manifest[f"rgb_extrinsics_{eye}"], dtype=np.float64)
    t_h_from_d = np.asarray(manifest["depth_extrinsics"], dtype=np.float64)
    pose_correction = rgb_pose_axis_correction({"rgb_pose_image_rotation_deg": -90.0})
    return {
        "camera_frame": "opencv_rdf",
        "selected_eye": eye,
        "rgb_width_stereo": int(manifest["rgb_width_per_eye"]) * 2,
        "rgb_width_per_eye": int(manifest["rgb_width_per_eye"]),
        "rgb_height_per_eye": int(manifest["rgb_height_per_eye"]),
        "depth_width": int(manifest["depth_width"]),
        "depth_height": int(manifest["depth_height"]),
        "rgb_intrinsics_selected": intrinsics,
        # Backward-compatible aliases consumed by the existing stages.
        "rgb_intrinsics_right": intrinsics,
        "depth_intrinsics": dict(manifest["depth_intrinsics"]),
        "raw_rgb_extrinsics_selected": t_h_from_c,
        "raw_rgb_extrinsics_right": t_h_from_c,
        "raw_depth_extrinsics": t_h_from_d,
        "extrinsics_interpretation": "sensor_to_head",
        "T_H_from_C_selected": t_h_from_c,
        "T_H_from_C_right": t_h_from_c,
        "T_H_from_D": t_h_from_d,
        "rgb_pose_image_rotation_deg": -90.0,
        "T_H_from_C_right_pose": t_h_from_c @ pose_correction,
        "compatibility_note": "Fields ending in right contain the selected-eye calibration.",
    }


def project_depth(
    row: dict[str, str], spatial_export: Path, camera: dict[str, Any], depth_min: float, depth_max: float
) -> tuple[np.ndarray, dict[str, Any]]:
    depth_path = spatial_export / row_value(row, "depth_meters_npy", "depth_interpolated_meters_npy")
    depth = np.load(depth_path).astype(np.float32)
    valid = np.isfinite(depth) & (depth >= depth_min) & (depth <= depth_max)
    points_d, _ = backproject_depth(np.where(valid, depth, 0.0), camera["depth_intrinsics"])
    t_w_from_d = row_pose(row, "depth_pose") @ np.asarray(camera["T_H_from_D"])
    t_w_from_c = row_pose(row, "rgb_pose") @ np.asarray(camera["T_H_from_C_selected"])
    t_c_from_d = np.linalg.inv(t_w_from_c) @ t_w_from_d
    projected = project_points_zbuffer(
        transform_points(points_d, t_c_from_d),
        camera["rgb_intrinsics_selected"],
        int(camera["rgb_width_per_eye"]),
        int(camera["rgb_height_per_eye"]),
    )
    return projected, {
        "source_depth": str(depth_path),
        "valid_source_depth_pixels": int(valid.sum()),
        "projected_valid_pixels": int(np.count_nonzero(projected)),
        "T_Ct_rgb_from_Dt_depth": t_c_from_d,
    }


def main() -> int:
    args = parse_args()
    workspace = args.workspace.resolve()
    spatial_export = args.spatial_export.resolve()
    if workspace in {Path("/"), PROJECT_ROOT, PROJECT_ROOT.parent}:
        raise ValueError(f"Refusing unsafe workspace: {workspace}")
    rows = sorted(read_rows(spatial_export / "frames.csv"), key=lambda row: int(row["index"]))
    end = None if args.frame_count is None else args.frame_start + args.frame_count
    rows = rows[args.frame_start:end]
    if not rows:
        raise ValueError("No depth-matched rows selected")
    manifest = json.loads((spatial_export / "manifest.json").read_text(encoding="utf-8"))
    camera = camera_from_manifest(manifest, args.eye)

    stage00 = workspace / "outputs/00_rgb_frames"
    stage06 = workspace / "outputs/06_dense_depth"
    owned = [stage00, stage06]
    existing = [path for path in owned if path.exists() or path.is_symlink()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Outputs exist; pass --overwrite: {existing}")
    if args.overwrite:
        for path in existing:
            remove_owned(path)
    png_dir = stage00 / "right_rgb_png"
    jpeg_dir = stage00 / "sam2_jpeg"
    png_dir.mkdir(parents=True, exist_ok=True)
    jpeg_dir.mkdir(parents=True, exist_ok=True)

    cache = (
        args.depth_cache_dir.resolve()
        if args.depth_cache_dir
        else Path("/tmp/vlm_sam2_recon_cache") / workspace.name / "native_depth"
    )
    if args.overwrite and cache.exists():
        remove_owned(cache)
    raw_dir = cache / "raw_projected_npy"
    metric_dir = cache / "metric_depth_npy"
    raw_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)
    link_directory(raw_dir, stage06 / "raw_projected_npy")
    link_directory(raw_dir, stage06 / "true_depth_projected")
    link_directory(metric_dir, stage06 / "metric_depth_npy")

    t_h_from_c_pose = np.asarray(camera["T_H_from_C_right_pose"], dtype=np.float64)
    t_w_from_c0 = None
    t_w_from_h_all, t_w_from_c_all, t_c0_from_ct_all = [], [], []
    timeline, depth_records, timestamps = [], [], []
    eye_key = f"{args.eye}_rgb_png"
    for frame_index, row in enumerate(rows):
        source_index = int(row["index"])
        source_rgb = spatial_export / row_value(row, eye_key)
        destination = png_dir / f"{frame_index:06d}.png"
        shutil.copy2(source_rgb, destination)
        with Image.open(source_rgb) as image:
            image.convert("RGB").save(jpeg_dir / f"{frame_index:06d}.jpg", quality=args.jpeg_quality, subsampling=0)

        projected, report = project_depth(row, spatial_export, camera, args.depth_min_m, args.depth_max_m)
        tracking_depth = nearest_z_splat(projected, args.tracking_splat_radius)
        raw_path = raw_dir / f"{frame_index:06d}.npy"
        metric_path = metric_dir / f"{frame_index:06d}.npy"
        np.save(raw_path, projected.astype(np.float32))
        np.save(metric_path, tracking_depth.astype(np.float32))

        timestamp = float(row_value(row, "rgb_timestamp_s", "rgb_timestamp"))
        depth_timestamp = float(row_value(row, "depth_timestamp_s", "depth_timestamp"))
        t_w_from_h = row_pose(row, "rgb_pose")
        t_w_from_c = t_w_from_h @ t_h_from_c_pose
        if t_w_from_c0 is None:
            t_w_from_c0 = t_w_from_c
        t_c0_from_ct = np.linalg.inv(t_w_from_c0) @ t_w_from_c
        t_w_from_h_all.append(t_w_from_h)
        t_w_from_c_all.append(t_w_from_c)
        t_c0_from_ct_all.append(t_c0_from_ct)
        timestamps.append(timestamp)
        timeline.append({
            "frame_index": frame_index,
            "rgb_timestamp_s": f"{timestamp:.12f}",
            "source_rgb_index": source_index,
            "source_export_index": source_index,
            "right_rgb_png": f"right_rgb_png/{frame_index:06d}.png",
            "sam2_jpeg": f"sam2_jpeg/{frame_index:06d}.jpg",
            "true_depth_nearest_index": source_index,
            "true_depth_timestamp_s": f"{depth_timestamp:.12f}",
            "true_depth_delta_s": f"{depth_timestamp - timestamp:.12f}",
            "true_depth_meters_npy": report["source_depth"],
            "raw_projected_depth_npy": str(raw_path),
            "metric_depth_npy": str(metric_path),
        })
        depth_records.append({
            "frame_index": frame_index,
            "source_export_index": source_index,
            "rgb_timestamp_s": timestamp,
            "depth_timestamp_s": depth_timestamp,
            "depth_minus_rgb_s": depth_timestamp - timestamp,
            **report,
            "raw_projected_depth": str(raw_path),
            "tracking_metric_depth": str(metric_path),
            "tracking_splat_radius_px": args.tracking_splat_radius,
            "tracking_valid_pixels": int(np.count_nonzero(tracking_depth)),
        })
        print(f"[{frame_index + 1:02d}/{len(rows)}] source={source_index:02d} projected={report['projected_valid_pixels']} splat={np.count_nonzero(tracking_depth)}", flush=True)

    times = np.asarray(timestamps, dtype=np.float64)
    fps = 1.0 / float(np.median(np.diff(times))) if len(times) > 1 else float(manifest["depth_fps"])
    write_csv(stage00 / "timeline.csv", timeline, list(timeline[0]))
    np.savez_compressed(
        stage00 / "poses.npz",
        T_W_from_H=np.asarray(t_w_from_h_all),
        T_W_from_C=np.asarray(t_w_from_c_all),
        T_C0_from_Ct=np.asarray(t_c0_from_ct_all),
        rgb_timestamps_s=times,
        source_export_indices=np.asarray([int(row["index"]) for row in rows]),
    )
    camera.update({
        "native_rgbd_frame_count": len(rows),
        "temporal_pose_frame": f"frame0_{args.eye}_camera_opencv_rdf",
        "depth_projection_policy": "per-row depth pose to selected-eye RGB pose; nearest-z z-buffer",
    })
    write_json(stage00 / "camera.json", camera)
    write_json(stage06 / "native_depth_projection_manifest.json", {
        "stage": "06_native_metric_depth",
        "status": "completed",
        "selected_eye": args.eye,
        "frame_count": len(rows),
        "records": depth_records,
    })
    video = stage00 / "right_rgb_native_rgbd.mp4"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", f"{fps:.12f}",
        "-i", str(png_dir / "%06d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", str(video),
    ], check=True)
    (stage00 / "right_rgb_15fps.mp4").symlink_to(video.name)
    stage_manifest = {
        "stage": "00_rgb_extract",
        "status": "completed",
        "selected_eye": args.eye,
        "input_policy": "exact native depth-matched SpatialMP4 export rows; no temporal resampling",
        "spatial_export": str(spatial_export),
        "effective_fps": fps,
        "selected_frame_count": len(rows),
        "outputs": {"right_rgb_png": str(png_dir), "video": str(video), "poses": str(stage00 / "poses.npz")},
    }
    write_json(stage00 / "stage00_manifest.json", stage_manifest)
    update_stage_state(
        workspace / "pipeline_state.json", "00_rgb_extract", "completed",
        inputs=[str(spatial_export / "frames.csv")], outputs=[str(stage00), str(stage06)],
        notes=f"Prepared {len(rows)} exact depth-matched {args.eye}-eye RGB-D frames at native cadence ({fps:.3f} fps); no 15 fps resampling and no VDA.",
    )
    print(json.dumps(stage_manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
