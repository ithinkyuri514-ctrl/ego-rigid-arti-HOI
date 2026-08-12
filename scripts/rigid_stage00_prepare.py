#!/usr/bin/env python3
"""Export a timestamped single-eye timeline from SpatialMP4 for reconstruction."""

from __future__ import annotations

import argparse
import bisect
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vlm_sam2_recon.rigid_pipeline.common import (  # noqa: E402
    read_csv,
    update_stage_state,
    write_csv,
    write_json,
)
from vlm_sam2_recon.rigid_pipeline.calibration_validation import (  # noqa: E402
    normalized_sensor_extrinsics,
    validate_extrinsics_direction,
)
from vlm_sam2_recon.rigid_pipeline.geometry import (  # noqa: E402
    PoseSample,
    interpolate_pose,
    pose_matrix,
)
from vlm_sam2_recon.stages.camera_alignment import rgb_pose_axis_correction  # noqa: E402


# Container timestamp conversions can differ from the exported CSV by a few
# microseconds. This is still far below half a source-frame period.
TIMESTAMP_TOLERANCE_S = 1e-4


def import_spatialmp4():
    try:
        import spatialmp4

        return spatialmp4
    except ImportError:
        build_dir = Path("/code/SpatialMP4/build_spatialmp4_patched/python")
        if build_dir.is_dir():
            sys.path.insert(0, str(build_dir))
            import spatialmp4

            return spatialmp4
        raise RuntimeError(
            "Cannot import spatialmp4. The extension must be built against an OpenCV version available at runtime. "
            "Rebuild /code/SpatialMP4/build_spatialmp4_patched and verify it with `ldd .../spatialmp4*.so`."
        )


def parse_args() -> argparse.Namespace:
    default_workspace = PROJECT_ROOT / "run_rigid_20260715_215524"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=default_workspace)
    parser.add_argument("--video", type=Path, default=Path("/code/3DVideo_2026-07-15-21-55-24-056.mp4"))
    parser.add_argument(
        "--spatial-export",
        type=Path,
        default=Path("/code/3DVideo_2026-07-15-21-55-24-056_spatialmp4_depth_pose_export"),
    )
    parser.add_argument("--target-fps", type=float, default=15.0)
    parser.add_argument(
        "--start-timestamp",
        type=float,
        default=None,
        help="Inclusive RGB timestamp for global frame 0. The sampling grid is anchored here.",
    )
    parser.add_argument(
        "--end-timestamp",
        type=float,
        default=None,
        help="Inclusive final RGB timestamp. Defaults to the last pose-covered RGB frame.",
    )
    parser.add_argument("--eye", choices=["left", "right"], default="right")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--allow-pose-extrapolation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def calibration_dict(value) -> dict[str, float]:
    return {key: float(getattr(value, key)) for key in ("fx", "fy", "cx", "cy")}


def extrinsics_matrix(value) -> np.ndarray:
    if hasattr(value, "as_se3"):
        return np.asarray(value.as_se3(), dtype=np.float64)
    return np.asarray(value.extrinsics, dtype=np.float64)


def load_pose_samples(path: Path) -> list[PoseSample]:
    rows = read_csv(path)
    samples = []
    for row in rows:
        timestamp = row.get("timestamp_s", row.get("timestamp"))
        if timestamp in (None, ""):
            raise KeyError(f"Pose row in {path} has neither timestamp_s nor timestamp")
        samples.append(
            PoseSample(
                timestamp_s=float(timestamp),
                translation=np.asarray([row["x"], row["y"], row["z"]], dtype=np.float64),
                quaternion_wxyz=np.asarray(
                    [row["qw"], row["qx"], row["qy"], row["qz"]],
                    dtype=np.float64,
                ),
            )
        )
    samples.sort(key=lambda item: item.timestamp_s)
    return samples


def choose_source_indices(
    timestamps: list[float],
    target_fps: float,
    *,
    minimum_timestamp: float | None = None,
    maximum_timestamp: float | None = None,
) -> list[int]:
    if not timestamps:
        return []
    if target_fps <= 0:
        return list(range(len(timestamps)))
    period = 1.0 / target_fps
    valid_indices = [
        index
        for index, timestamp in enumerate(timestamps)
        if (minimum_timestamp is None or timestamp >= minimum_timestamp - TIMESTAMP_TOLERANCE_S)
        and (maximum_timestamp is None or timestamp <= maximum_timestamp + TIMESTAMP_TOLERANCE_S)
    ]
    if not valid_indices:
        return []
    first_valid, last_valid = valid_indices[0], valid_indices[-1]
    grid_start = float(minimum_timestamp) if minimum_timestamp is not None else timestamps[first_valid]
    if abs(grid_start - timestamps[first_valid]) <= TIMESTAMP_TOLERANCE_S:
        grid_start = timestamps[first_valid]
    else:
        grid_start = max(grid_start, timestamps[first_valid])
    targets = np.arange(grid_start, timestamps[last_valid] + 0.5 * period, period)
    selected: list[int] = []
    for target in targets:
        pos = bisect.bisect_left(timestamps, float(target))
        candidates = []
        if pos < len(timestamps) and pos <= last_valid:
            candidates.append(pos)
        if pos > first_valid and timestamps[pos - 1] >= grid_start - TIMESTAMP_TOLERANCE_S:
            candidates.append(pos - 1)
        index = min(candidates, key=lambda item: abs(timestamps[item] - target))
        if not selected or index != selected[-1]:
            selected.append(index)
    return selected


def nearest_depth_row(rows: list[dict[str, str]], timestamp_s: float) -> tuple[int, dict[str, str], float]:
    timestamps = [float(row["depth_timestamp_s"]) for row in rows]
    pos = bisect.bisect_left(timestamps, timestamp_s)
    candidates = [max(0, min(len(rows) - 1, pos))]
    if pos > 0:
        candidates.append(pos - 1)
    index = min(candidates, key=lambda item: abs(timestamps[item] - timestamp_s))
    return index, rows[index], timestamps[index] - timestamp_s


def normalize_depth_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Support both historical and current SpatialMP4 export CSV headers."""
    normalized = []
    for row in rows:
        item = dict(row)
        if "depth_timestamp_s" not in item and "depth_timestamp" in item:
            item["depth_timestamp_s"] = item["depth_timestamp"]
        if "depth_meters_npy" not in item and "depth_interpolated_meters_npy" in item:
            item["depth_meters_npy"] = item["depth_interpolated_meters_npy"]
        normalized.append(item)
    return normalized


def archive_legacy_stage00(output_dir: Path, workspace: Path) -> Path | None:
    legacy = sorted(output_dir.glob("*.png"))
    legacy.extend(path for path in (output_dir / "rgb_timeline.csv", output_dir / "rgb_extract_manifest.json") if path.exists())
    if not legacy:
        return None
    archive = workspace / "scratch" / "legacy_stage00_stereo"
    suffix = 1
    while archive.exists():
        archive = workspace / "scratch" / f"legacy_stage00_stereo_{suffix:02d}"
        suffix += 1
    archive.mkdir(parents=True)
    for path in legacy:
        shutil.move(str(path), archive / path.name)
    return archive


def prepare_output(output_dir: Path, workspace: Path, overwrite: bool) -> tuple[Path, Path, Path | None]:
    png_dir = output_dir / "right_rgb_png"
    jpeg_dir = output_dir / "sam2_jpeg"
    required = [output_dir / "timeline.csv", output_dir / "camera.json"]
    if any(path.exists() for path in required) and not overwrite:
        raise FileExistsError(f"Stage 00 output exists; pass --overwrite: {output_dir}")
    if overwrite:
        legacy_archive = archive_legacy_stage00(output_dir, workspace)
        for path in (png_dir, jpeg_dir):
            if path.exists():
                shutil.rmtree(path)
        for name in ("timeline.csv", "camera.json", "poses.npz", "stage00_manifest.json", "right_rgb_15fps.mp4"):
            path = output_dir / name
            if path.exists():
                path.unlink()
    else:
        legacy_archive = None
    png_dir.mkdir(parents=True, exist_ok=True)
    jpeg_dir.mkdir(parents=True, exist_ok=True)
    return png_dir, jpeg_dir, legacy_archive


def main() -> int:
    args = parse_args()
    workspace = args.workspace.resolve()
    video = args.video.resolve()
    spatial_export = args.spatial_export.resolve()
    output_dir = workspace / "outputs" / "00_rgb_frames"
    png_dir, jpeg_dir, legacy_archive = prepare_output(output_dir, workspace, args.overwrite)
    state_path = workspace / "pipeline_state.json"
    update_stage_state(state_path, "00_rgb_extract", "running")

    sm = import_spatialmp4()
    reader = sm.Reader(str(video))
    get_intrinsics = reader.get_rgb_intrinsics_left if args.eye == "left" else reader.get_rgb_intrinsics_right
    get_extrinsics = reader.get_rgb_extrinsics_left if args.eye == "left" else reader.get_rgb_extrinsics_right
    rgb_attr = "left_rgb" if args.eye == "left" else "right_rgb"
    raw_t_h_from_c = extrinsics_matrix(get_extrinsics())
    raw_t_h_from_d = extrinsics_matrix(reader.get_depth_extrinsics())
    calibration = {
        "camera_frame": "opencv_rdf",
        "raw_extrinsics_source": "SpatialMP4 reader",
        "rgb_width_stereo": int(reader.get_rgb_width() * 2),
        "rgb_width_per_eye": int(reader.get_rgb_width()),
        "rgb_height_per_eye": int(reader.get_rgb_height()),
        "depth_width": int(reader.get_depth_width()),
        "depth_height": int(reader.get_depth_height()),
        "selected_eye": args.eye,
        "rgb_intrinsics_selected": calibration_dict(get_intrinsics()),
        "rgb_intrinsics_right": calibration_dict(get_intrinsics()),
        "depth_intrinsics": calibration_dict(reader.get_depth_intrinsics()),
        "raw_rgb_extrinsics_selected": raw_t_h_from_c,
        "raw_rgb_extrinsics_right": raw_t_h_from_c,
        "raw_depth_extrinsics": raw_t_h_from_d,
    }

    reader.set_read_mode(sm.ReadMode.RGB_ONLY)
    source_timestamps: list[float] = []
    while reader.has_next():
        frame = reader.load_rgb()
        source_timestamps.append(float(frame.timestamp))
    if not source_timestamps or np.any(np.diff(source_timestamps) <= 0):
        raise RuntimeError("SpatialMP4 RGB timestamps are empty or non-monotonic")

    pose_samples = load_pose_samples(spatial_export / "pose" / "head_pose.csv")
    depth_timeline = spatial_export / "depth_frames.csv"
    if not depth_timeline.is_file():
        depth_timeline = spatial_export / "frames.csv"
    if not depth_timeline.is_file():
        raise FileNotFoundError(
            f"Spatial export has neither depth_frames.csv nor frames.csv: {spatial_export}"
        )
    depth_rows = normalize_depth_rows(read_csv(depth_timeline))
    minimum_timestamp = args.start_timestamp
    maximum_timestamp = args.end_timestamp
    if not args.allow_pose_extrapolation:
        minimum_timestamp = max(
            pose_samples[0].timestamp_s,
            minimum_timestamp if minimum_timestamp is not None else pose_samples[0].timestamp_s,
        )
        maximum_timestamp = min(
            pose_samples[-1].timestamp_s,
            maximum_timestamp if maximum_timestamp is not None else pose_samples[-1].timestamp_s,
        )
    if minimum_timestamp is not None and maximum_timestamp is not None and minimum_timestamp > maximum_timestamp:
        raise ValueError(
            f"Invalid timestamp bounds after pose coverage: {minimum_timestamp} > {maximum_timestamp}"
        )
    source_indices = choose_source_indices(
        source_timestamps,
        args.target_fps,
        minimum_timestamp=minimum_timestamp,
        maximum_timestamp=maximum_timestamp,
    )
    selected_lookup = {source_index: frame_index for frame_index, source_index in enumerate(source_indices)}
    reader.reset()
    reader.set_read_mode(sm.ReadMode.RGB_ONLY)
    timeline_rows = []
    source_index = 0
    while reader.has_next():
        frame = reader.load_rgb()
        if source_index not in selected_lookup:
            source_index += 1
            continue
        frame_index = selected_lookup[source_index]
        timestamp_s = float(frame.timestamp)
        selected_bgr = np.asarray(getattr(frame, rgb_attr))
        if selected_bgr.shape != (calibration["rgb_height_per_eye"], calibration["rgb_width_per_eye"], 3):
            raise ValueError(f"Unexpected {args.eye}-eye frame shape: {selected_bgr.shape}")
        selected_rgb = np.ascontiguousarray(selected_bgr[..., ::-1])
        png_rel = Path("right_rgb_png") / f"{frame_index:06d}.png"
        jpeg_rel = Path("sam2_jpeg") / f"{frame_index:06d}.jpg"
        image = Image.fromarray(selected_rgb)
        image.save(output_dir / png_rel, compress_level=3)
        image.save(output_dir / jpeg_rel, quality=args.jpeg_quality, subsampling=0)
        depth_index, depth_row, depth_delta = nearest_depth_row(depth_rows, timestamp_s)
        timeline_rows.append(
            {
                "frame_index": frame_index,
                "rgb_timestamp_s": f"{timestamp_s:.12f}",
                "source_rgb_index": source_index,
                "right_rgb_png": str(png_rel),
                "sam2_jpeg": str(jpeg_rel),
                "true_depth_nearest_index": depth_index,
                "true_depth_timestamp_s": depth_row["depth_timestamp_s"],
                "true_depth_delta_s": f"{depth_delta:.12f}",
                "true_depth_meters_npy": str(spatial_export / depth_row["depth_meters_npy"]),
            }
        )
        source_index += 1
    timeline_rows.sort(key=lambda row: int(row["frame_index"]))
    if len(timeline_rows) != len(source_indices):
        raise RuntimeError(f"Decoded selected RGB mismatch: expected={len(source_indices)} actual={len(timeline_rows)}")

    extrinsics_validation = validate_extrinsics_direction(
        raw_t_h_from_c=raw_t_h_from_c,
        raw_t_h_from_d=raw_t_h_from_d,
        rgb_intrinsics=calibration["rgb_intrinsics_selected"],
        depth_intrinsics=calibration["depth_intrinsics"],
        rgb_rows=timeline_rows,
        rgb_dir=png_dir,
        depth_rows=depth_rows,
        spatial_root=spatial_export,
        pose_samples=pose_samples,
    )
    interpretation = extrinsics_validation["selected_interpretation"]
    t_h_from_c = normalized_sensor_extrinsics(raw_t_h_from_c, interpretation)
    t_h_from_d = normalized_sensor_extrinsics(raw_t_h_from_d, interpretation)
    # Keep the spatial RGB/depth extrinsics unchanged. SpatialMP4's exported RGB image
    # axes require an additional -90 degree rotation only when composing temporal head
    # poses. Applying that correction to depth-to-RGB calibration would corrupt geometry.
    pose_axis_correction = rgb_pose_axis_correction({"rgb_pose_image_rotation_deg": -90.0})
    t_h_from_c_pose = t_h_from_c @ pose_axis_correction
    calibration.update(
        {
            "extrinsics_interpretation": interpretation,
            "T_H_from_C_selected": t_h_from_c,
            "T_H_from_C_right": t_h_from_c,
            "T_H_from_D": t_h_from_d,
            "rgb_pose_image_rotation_deg": -90.0,
            "T_H_from_C_right_pose": t_h_from_c_pose,
            "extrinsics_validation": extrinsics_validation,
        }
    )
    write_json(output_dir / "camera.json", calibration)

    t_w_from_c0 = None
    t_w_from_h_all = []
    t_c0_from_ct_all = []
    for row in timeline_rows:
        timestamp_s = float(row["rgb_timestamp_s"])
        pose, pose_left, pose_right, pose_alpha = interpolate_pose(pose_samples, timestamp_s)
        t_w_from_h = pose_matrix(pose)
        t_w_from_c = t_w_from_h @ t_h_from_c_pose
        if t_w_from_c0 is None:
            t_w_from_c0 = t_w_from_c
        t_c0_from_ct = np.linalg.inv(t_w_from_c0) @ t_w_from_c
        t_w_from_h_all.append(t_w_from_h)
        t_c0_from_ct_all.append(t_c0_from_ct)

        row["head_pose_left_index"] = pose_left
        row["head_pose_right_index"] = pose_right
        row["head_pose_alpha"] = f"{pose_alpha:.9f}"

    if not timeline_rows:
        update_stage_state(state_path, "00_rgb_extract", "failed", notes="No RGB frames decoded.")
        raise RuntimeError("No RGB frames decoded from SpatialMP4")

    write_csv(output_dir / "timeline.csv", timeline_rows, list(timeline_rows[0]))
    np.savez_compressed(
        output_dir / "poses.npz",
        T_W_from_H=np.asarray(t_w_from_h_all, dtype=np.float64),
        T_C0_from_Ct=np.asarray(t_c0_from_ct_all, dtype=np.float64),
        rgb_timestamps_s=np.asarray([float(row["rgb_timestamp_s"]) for row in timeline_rows]),
    )

    output_video = output_dir / "right_rgb_15fps.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            str(args.target_fps),
            "-i",
            str(png_dir / "%06d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "16",
            str(output_video),
        ],
        check=True,
    )
    manifest = {
        "stage": "00_rgb_extract",
        "status": "completed",
        "video": str(video),
        "selected_eye": args.eye,
        "spatial_export": str(spatial_export),
        "depth_timeline": str(depth_timeline),
        "target_fps": args.target_fps,
        "source_frame_count": len(source_timestamps),
        "selected_frame_count": len(timeline_rows),
        "timestamp_start_s": float(timeline_rows[0]["rgb_timestamp_s"]),
        "timestamp_end_s": float(timeline_rows[-1]["rgb_timestamp_s"]),
        "requested_timestamp_start_s": args.start_timestamp,
        "requested_timestamp_end_s": args.end_timestamp,
        "pose_extrapolation_allowed": args.allow_pose_extrapolation,
        "legacy_stereo_archive": str(legacy_archive) if legacy_archive else None,
        "extrinsics_validation": extrinsics_validation,
        "outputs": {
            "right_rgb_png": str(png_dir),
            "sam2_jpeg": str(jpeg_dir),
            "right_rgb_video": str(output_video),
            "timeline": str(output_dir / "timeline.csv"),
            "camera": str(output_dir / "camera.json"),
            "poses": str(output_dir / "poses.npz"),
        },
    }
    write_json(output_dir / "stage00_manifest.json", manifest)
    update_stage_state(
        state_path,
        "00_rgb_extract",
        "completed",
        inputs=[str(video), str(spatial_export)],
        outputs=[str(output_dir)],
        notes=f"Exported {len(timeline_rows)} timestamped 1624x1232 {args.eye}-eye frames (stored under backward-compatible right_rgb paths).",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
