#!/usr/bin/env python3
"""Estimate missing-frame depth with VDA and guide it with timestamped metric depth."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vlm_sam2_recon.rigid_pipeline.common import (  # noqa: E402
    read_csv,
    read_json,
    update_stage_state,
    write_csv,
    write_json,
)
from vlm_sam2_recon.rigid_pipeline.depth_fusion import (  # noqa: E402
    colorize_depth,
    convert_representation,
    edge_aware_anchor_fusion,
    evaluate_metric_depth,
    fit_anchor,
    interpolate_calibration,
    resize_prediction,
    select_representation,
)
from vlm_sam2_recon.rigid_pipeline.geometry import (  # noqa: E402
    PoseSample,
    backproject_depth,
    interpolate_pose,
    pose_matrix,
    project_points_zbuffer,
    transform_points,
)


def parse_args() -> argparse.Namespace:
    workspace = PROJECT_ROOT / "run_rigid_20260715_215524"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=workspace)
    parser.add_argument("--vda-npz", type=Path, default=None, help="Existing VDA NPZ; skips inference when present.")
    parser.add_argument(
        "--vda-root",
        type=Path,
        default=Path("/code/ArtHOI-4D-Reconstruction/third_party/Video-Depth-Anything"),
    )
    parser.add_argument("--vda-python", type=Path, default=Path("/opt/conda/envs/arthoi/bin/python"))
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--max-res", type=int, default=1280)
    parser.add_argument("--depth-min-m", type=float, default=0.1)
    parser.add_argument("--depth-max-m", type=float, default=5.0)
    parser.add_argument("--spatial-sigma-px", type=float, default=18.0)
    parser.add_argument("--color-sigma", type=float, default=36.0)
    parser.add_argument("--holdout-stride", type=int, default=5)
    parser.add_argument("--holdout-offset", type=int, default=2)
    parser.add_argument("--max-heldout-rmse-m", type=float, default=0.20)
    parser.add_argument("--max-heldout-median-abs-m", type=float, default=0.10)
    parser.add_argument("--min-heldout-valid-ratio", type=float, default=0.80)
    parser.add_argument("--allow-qc-failure", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--check", action="store_true", help="Validate prerequisites and print VDA command only.")
    return parser.parse_args()


def load_pose_samples(path: Path) -> list[PoseSample]:
    samples = []
    for row in read_csv(path):
        timestamp = row.get("timestamp_s", row.get("timestamp"))
        if timestamp in (None, ""):
            raise KeyError(f"Pose row in {path} has neither timestamp_s nor timestamp")
        samples.append(
            PoseSample(
                timestamp_s=float(timestamp),
                translation=np.asarray([row["x"], row["y"], row["z"]], dtype=np.float64),
                quaternion_wxyz=np.asarray([row["qw"], row["qx"], row["qy"], row["qz"]], dtype=np.float64),
            )
        )
    return samples


def vda_command(args: argparse.Namespace, input_video: Path, raw_dir: Path) -> list[str]:
    return [
        str(args.vda_python.resolve()),
        str(args.vda_root.resolve() / "run.py"),
        "--input_video",
        str(input_video),
        "--output_dir",
        str(raw_dir),
        "--encoder",
        "vitl",
        "--input_size",
        str(args.input_size),
        "--max_res",
        str(args.max_res),
        "--target_fps",
        "-1",
        "--save_npz",
    ]


def prepare_dirs(output_dir: Path, overwrite: bool) -> dict[str, Path]:
    paths = {
        "raw": output_dir / "vda_raw",
        "projected": output_dir / "true_depth_projected",
        "base": output_dir / "metric_base_npy",
        "metric": output_dir / "metric_depth_npy",
        "confidence": output_dir / "anchor_confidence_npy",
        "vis": output_dir / "metric_depth_vis",
    }
    if overwrite:
        for path in paths.values():
            if path.exists():
                shutil.rmtree(path)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def project_true_depth_anchors(
    workspace: Path,
    camera: dict,
    rgb_rows: list[dict[str, str]],
    output_dir: Path,
    depth_min_m: float,
    depth_max_m: float,
) -> list[dict]:
    config = read_json(workspace / "configs/rigid_recon_config.json")
    spatial_root_value = config["inputs"].get("spatial_export_root", config.get("spatial_export_root"))
    if not spatial_root_value:
        raise KeyError("Missing spatial_export_root in rigid_recon_config.json")
    spatial_root = Path(spatial_root_value)
    depth_rows = read_csv(Path(config["inputs"]["true_depth_timeline"]))
    pose_samples = load_pose_samples(Path(config["inputs"]["head_pose_csv"]))
    rgb_pose_data = np.load(workspace / "outputs/00_rgb_frames/poses.npz")
    t_w_from_h_rgb = np.asarray(rgb_pose_data["T_W_from_H"], dtype=np.float64)
    rgb_timestamps = np.asarray([float(row["rgb_timestamp_s"]) for row in rgb_rows])
    t_h_from_c = np.asarray(camera["T_H_from_C_right"], dtype=np.float64)
    t_h_from_d = np.asarray(camera["T_H_from_D"], dtype=np.float64)
    width = int(camera["rgb_width_per_eye"])
    height = int(camera["rgb_height_per_eye"])
    records = []
    for depth_index, row in enumerate(depth_rows):
        depth_timestamp_value = row.get("depth_timestamp_s", row.get("depth_timestamp"))
        if depth_timestamp_value in (None, ""):
            raise KeyError(
                "Depth timeline row has neither depth_timestamp_s nor depth_timestamp"
            )
        depth_path_value = row.get("depth_meters_npy", row.get("depth_interpolated_meters_npy"))
        if depth_path_value in (None, ""):
            raise KeyError(
                "Depth timeline row has neither depth_meters_npy nor depth_interpolated_meters_npy"
            )
        depth_timestamp = float(depth_timestamp_value)
        rgb_index = int(np.argmin(np.abs(rgb_timestamps - depth_timestamp)))
        depth_source = spatial_root / depth_path_value
        depth = np.load(depth_source).astype(np.float32)
        valid = np.isfinite(depth) & (depth >= depth_min_m) & (depth <= depth_max_m)
        filtered = np.where(valid, depth, 0.0)
        points_d, _ = backproject_depth(filtered, camera["depth_intrinsics"])
        pose_d, pose_left, pose_right, pose_alpha = interpolate_pose(pose_samples, depth_timestamp)
        t_w_from_d = pose_matrix(pose_d) @ t_h_from_d
        t_w_from_c = t_w_from_h_rgb[rgb_index] @ t_h_from_c
        t_c_from_d = np.linalg.inv(t_w_from_c) @ t_w_from_d
        points_c = transform_points(points_d, t_c_from_d)
        projected = project_points_zbuffer(
            points_c,
            camera["rgb_intrinsics_right"],
            width,
            height,
        )
        output_path = output_dir / f"depth_{depth_index:06d}_to_rgb_{rgb_index:06d}.npy"
        np.save(output_path, projected)
        records.append(
            {
                "depth_index": depth_index,
                "depth_timestamp_s": depth_timestamp,
                "rgb_frame_index": rgb_index,
                "rgb_timestamp_s": float(rgb_timestamps[rgb_index]),
                "timestamp_delta_s": float(rgb_timestamps[rgb_index] - depth_timestamp),
                "source_depth": str(depth_source),
                "projected_depth": str(output_path),
                "projected_valid_pixels": int(np.count_nonzero(projected)),
                "head_pose_left_index": pose_left,
                "head_pose_right_index": pose_right,
                "head_pose_alpha": pose_alpha,
            }
        )
    return records


def main() -> int:
    args = parse_args()
    workspace = args.workspace.resolve()
    stage00 = workspace / "outputs/00_rgb_frames"
    input_video = stage00 / "right_rgb_15fps.mp4"
    timeline_path = stage00 / "timeline.csv"
    camera_path = stage00 / "camera.json"
    for path in (input_video, timeline_path, camera_path, args.vda_root.resolve() / "run.py", args.vda_python.resolve()):
        if not path.exists():
            raise FileNotFoundError(path)
    output_dir = workspace / "outputs/06_dense_depth"
    paths = prepare_dirs(output_dir, args.overwrite)
    command = vda_command(args, input_video, paths["raw"])
    if args.check:
        print(json.dumps({"stage": "06_dense_depth_metric_calibration", "vda_command": command}, indent=2))
        print("Stage 06 prerequisite check passed; VDA was not run.")
        return 0

    vda_npz = args.vda_npz.resolve() if args.vda_npz else paths["raw"] / f"{input_video.stem}_depths.npz"
    if not vda_npz.exists():
        subprocess.run(command, cwd=str(args.vda_root.resolve()), check=True)
    if not vda_npz.is_file():
        raise FileNotFoundError(vda_npz)
    predictions = np.asarray(np.load(vda_npz)["depths"], dtype=np.float32)
    rgb_rows = read_csv(timeline_path)
    rgb_frames = sorted((stage00 / "right_rgb_png").glob("*.png"))
    if len(predictions) != len(rgb_rows) or len(rgb_frames) != len(rgb_rows):
        raise ValueError(
            f"Frame count mismatch: VDA={len(predictions)} timeline={len(rgb_rows)} RGB={len(rgb_frames)}"
        )
    camera = read_json(camera_path)
    height = int(camera["rgb_height_per_eye"])
    width = int(camera["rgb_width_per_eye"])
    anchors = project_true_depth_anchors(
        workspace,
        camera,
        rgb_rows,
        paths["projected"],
        args.depth_min_m,
        args.depth_max_m,
    )

    fits_by_representation = {"direct": [], "inverse": []}
    fit_errors = []
    resized_cache: dict[int, np.ndarray] = {}
    for anchor in anchors:
        frame_index = int(anchor["rgb_frame_index"])
        prediction = resized_cache.setdefault(
            frame_index,
            resize_prediction(predictions[frame_index], (height, width)),
        )
        true_projected = np.load(anchor["projected_depth"])
        for representation in fits_by_representation:
            try:
                fit = fit_anchor(
                    prediction,
                    true_projected,
                    frame_index=frame_index,
                    depth_index=int(anchor["depth_index"]),
                    timestamp_s=float(anchor["depth_timestamp_s"]),
                    representation=representation,
                    depth_min_m=args.depth_min_m,
                    depth_max_m=args.depth_max_m,
                )
                fits_by_representation[representation].append(fit)
            except Exception as exc:
                fit_errors.append(
                    {
                        "depth_index": anchor["depth_index"],
                        "frame_index": frame_index,
                        "representation": representation,
                        "error": str(exc),
                    }
                )
    frame_timestamps = np.asarray([float(row["rgb_timestamp_s"]) for row in rgb_rows])
    holdout_depth_indices = {
        int(anchor["depth_index"])
        for anchor in anchors
        if args.holdout_stride > 1
        and int(anchor["depth_index"]) % args.holdout_stride == args.holdout_offset % args.holdout_stride
    }
    training_fits_by_representation = {
        key: [fit for fit in values if fit.depth_index not in holdout_depth_indices]
        for key, values in fits_by_representation.items()
    }
    representation = select_representation(training_fits_by_representation)
    training_fits = [
        fit for fit in training_fits_by_representation[representation]
        if fit.scale > 0
    ]
    training_scale, training_shift = interpolate_calibration(frame_timestamps, training_fits)
    heldout_records = []
    for anchor in anchors:
        depth_index = int(anchor["depth_index"])
        if depth_index not in holdout_depth_indices:
            continue
        frame_index = int(anchor["rgb_frame_index"])
        prediction = resized_cache.setdefault(
            frame_index,
            resize_prediction(predictions[frame_index], (height, width)),
        )
        represented = convert_representation(prediction, representation)
        estimated = training_scale[frame_index] * represented + training_shift[frame_index]
        metrics = evaluate_metric_depth(
            estimated,
            np.load(anchor["projected_depth"]),
            depth_min_m=args.depth_min_m,
            depth_max_m=args.depth_max_m,
        )
        heldout_records.append(
            {
                "depth_index": depth_index,
                "rgb_frame_index": frame_index,
                "timestamp_delta_s": anchor["timestamp_delta_s"],
                **metrics,
            }
        )
    valid_heldout = [record for record in heldout_records if record["rmse_m"] is not None]
    heldout_summary = {
        "depth_indices": sorted(holdout_depth_indices),
        "count": len(heldout_records),
        "valid_count": len(valid_heldout),
        "median_rmse_m": float(np.median([record["rmse_m"] for record in valid_heldout])) if valid_heldout else None,
        "median_abs_error_m": float(np.median([record["median_abs_error_m"] for record in valid_heldout])) if valid_heldout else None,
        "median_valid_ratio": float(np.median([record["valid_ratio"] for record in valid_heldout])) if valid_heldout else 0.0,
        "records": heldout_records,
    }
    heldout_summary["passed"] = bool(valid_heldout) and (
        heldout_summary["median_rmse_m"] <= args.max_heldout_rmse_m
        and heldout_summary["median_abs_error_m"] <= args.max_heldout_median_abs_m
        and heldout_summary["median_valid_ratio"] >= args.min_heldout_valid_ratio
    )
    heldout_summary["threshold_max_median_rmse_m"] = args.max_heldout_rmse_m
    heldout_summary["threshold_max_median_abs_error_m"] = args.max_heldout_median_abs_m
    heldout_summary["threshold_min_median_valid_ratio"] = args.min_heldout_valid_ratio

    # Held-out anchors are used only for validation above. Refit with every
    # anchor afterwards to maximize production-depth accuracy.
    selected_fits = [fit for fit in fits_by_representation[representation] if fit.scale > 0]
    scale_per_frame, shift_per_frame = interpolate_calibration(frame_timestamps, selected_fits)
    anchors_by_frame: dict[int, list[dict]] = {}
    for anchor in anchors:
        anchors_by_frame.setdefault(int(anchor["rgb_frame_index"]), []).append(anchor)

    output_rows = []
    for frame_index, rgb_path in enumerate(rgb_frames):
        prediction = resized_cache.get(frame_index)
        if prediction is None:
            prediction = resize_prediction(predictions[frame_index], (height, width))
        represented = convert_representation(prediction, representation)
        base = (scale_per_frame[frame_index] * represented + shift_per_frame[frame_index]).astype(np.float32)
        base[(base < args.depth_min_m) | (base > args.depth_max_m) | ~np.isfinite(base)] = 0.0
        rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
        fused = base.copy()
        confidence = np.zeros((height, width), dtype=np.float32)
        anchor_indices = []
        for anchor in sorted(anchors_by_frame.get(frame_index, []), key=lambda item: abs(item["timestamp_delta_s"]), reverse=True):
            true_projected = np.load(anchor["projected_depth"])
            fused, local_confidence = edge_aware_anchor_fusion(
                fused,
                true_projected,
                rgb,
                spatial_sigma_px=args.spatial_sigma_px,
                color_sigma=args.color_sigma,
                depth_min_m=args.depth_min_m,
                depth_max_m=args.depth_max_m,
            )
            confidence = np.maximum(confidence, local_confidence)
            anchor_indices.append(int(anchor["depth_index"]))
        base_path = paths["base"] / f"{frame_index:06d}.npy"
        metric_path = paths["metric"] / f"{frame_index:06d}.npy"
        confidence_path = paths["confidence"] / f"{frame_index:06d}.npy"
        vis_path = paths["vis"] / f"{frame_index:06d}.png"
        np.save(base_path, base)
        np.save(metric_path, fused)
        np.save(confidence_path, confidence)
        Image.fromarray(colorize_depth(fused, args.depth_min_m, args.depth_max_m)).save(vis_path)
        output_rows.append(
            {
                "frame_index": frame_index,
                "rgb_timestamp_s": f"{frame_timestamps[frame_index]:.12f}",
                "metric_depth_npy": str(metric_path),
                "metric_base_npy": str(base_path),
                "anchor_confidence_npy": str(confidence_path),
                "depth_vis_png": str(vis_path),
                "vda_representation": representation,
                "scale": f"{scale_per_frame[frame_index]:.12g}",
                "shift": f"{shift_per_frame[frame_index]:.12g}",
                "true_depth_anchor_indices": ",".join(map(str, anchor_indices)),
                "has_true_depth_guidance": bool(anchor_indices),
            }
        )

    write_csv(output_dir / "depth_timeline.csv", output_rows, list(output_rows[0]))
    report = {
        "stage": "06_dense_depth_metric_calibration",
        "status": "completed" if heldout_summary["passed"] else "needs_revision",
        "input_video": str(input_video),
        "vda_npz": str(vda_npz),
        "frame_count": len(output_rows),
        "true_depth_anchor_count": len(anchors),
        "selected_representation": representation,
        "representation_selection_policy": "training anchors only; held-out anchors excluded until final production refit",
        "representation_scores_rmse_median": {
            key: float(np.median([fit.rmse_m for fit in values])) if values else None
            for key, values in fits_by_representation.items()
        },
        "anchor_fits": {
            key: [fit.to_dict() for fit in values]
            for key, values in fits_by_representation.items()
        },
        "fit_errors": fit_errors,
        "heldout_validation": heldout_summary,
        "fusion": {
            "anchor_pixels": "hard replacement with projected original metric depth",
            "neighborhood": "RGB-edge-aware nearest-anchor residual propagation",
            "missing_frames": "timestamp-interpolated VDA scale/shift from original depth anchors",
            "spatial_sigma_px": args.spatial_sigma_px,
            "color_sigma": args.color_sigma,
        },
        "outputs": {
            "metric_depth_dir": str(paths["metric"]),
            "depth_timeline": str(output_dir / "depth_timeline.csv"),
            "projected_true_depth_dir": str(paths["projected"]),
        },
    }
    write_json(output_dir / "calibration_report.json", report)
    update_stage_state(
        workspace / "pipeline_state.json",
        "06_dense_depth_metric_calibration",
        "completed" if heldout_summary["passed"] else "needs_revision",
        inputs=[str(input_video), str(vda_npz), str(workspace / "configs/rigid_recon_config.json")],
        outputs=[str(output_dir)],
        notes=(
            f"Generated {len(output_rows)} metric depth frames; held-out anchor validation passed."
            if heldout_summary["passed"]
            else f"Generated depth but held-out validation failed: median RMSE={heldout_summary['median_rmse_m']} m."
        ),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if heldout_summary["passed"] or args.allow_qc_failure else 2


if __name__ == "__main__":
    raise SystemExit(main())
