#!/usr/bin/env python3
"""Project the synchronized SpatialMP4 metric depth of global frame 0 into the right RGB camera."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vlm_sam2_recon.rigid_pipeline.common import jsonable, update_stage_state, write_json  # noqa: E402
from vlm_sam2_recon.rigid_pipeline.geometry import backproject_depth, project_points_zbuffer, transform_points  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--depth-min-m", type=float, default=0.15)
    parser.add_argument("--depth-max-m", type=float, default=4.0)
    return parser.parse_args()


def pose_from_row(row: dict[str, str], prefix: str) -> np.ndarray:
    from vlm_sam2_recon.rigid_pipeline.geometry import pose_matrix, PoseSample

    return pose_matrix(PoseSample(float(row[f"{prefix}_timestamp_s"]), np.asarray([float(row[f"{prefix}_{a}"]) for a in ("x", "y", "z")]), np.asarray([float(row[f"{prefix}_q{a}"]) for a in ("w", "x", "y", "z")])) )


def main() -> int:
    args = parse_args()
    workspace = args.workspace.resolve()
    root = Path(json.loads((workspace / "configs/mixed_recon_config.json").read_text())["spatial_export_root"])
    camera = json.loads((workspace / "outputs/00_rgb_frames/camera.json").read_text())
    rows = list(csv.DictReader((root / "frames.csv").open(newline="", encoding="utf-8")))
    row = rows[0]
    depth = np.load(root / row["depth_meters_npy"]).astype(np.float32)
    valid = np.isfinite(depth) & (depth >= args.depth_min_m) & (depth <= args.depth_max_m)
    points_d, _ = backproject_depth(np.where(valid, depth, 0.0), camera["depth_intrinsics"])
    t_w_from_d = pose_from_row(row, "depth_pose") @ np.asarray(camera["T_H_from_D"], dtype=np.float64)
    t_w_from_c = pose_from_row(row, "rgb_pose") @ np.asarray(camera["T_H_from_C_right"], dtype=np.float64)
    t_c_from_d = np.linalg.inv(t_w_from_c) @ t_w_from_d
    points_c = transform_points(points_d, t_c_from_d)
    projected = project_points_zbuffer(points_c, camera["rgb_intrinsics_right"], camera["rgb_width_per_eye"], camera["rgb_height_per_eye"])
    out = workspace / "outputs/06_dense_depth"
    (out / "metric_depth_npy").mkdir(parents=True, exist_ok=True)
    (out / "true_depth_projected").mkdir(parents=True, exist_ok=True)
    np.save(out / "metric_depth_npy/000000.npy", projected)
    np.save(out / "true_depth_projected/depth_000000_to_rgb_000000.npy", projected)
    report = {
        "stage": "06_dense_depth_metric_calibration",
        "status": "completed",
        "method": "synchronized SpatialMP4 metric depth projected from depth camera to right RGB frame 0",
        "frame_index": 0,
        "source_depth": str(root / row["depth_meters_npy"]),
        "valid_source_depth_pixels": int(valid.sum()),
        "projected_valid_rgb_pixels": int(np.count_nonzero(projected)),
        "T_C_from_D": t_c_from_d,
        "outputs": {"metric_depth": str(out / "metric_depth_npy/000000.npy")},
    }
    write_json(out / "calibration_report.json", report)
    update_stage_state(workspace / "pipeline_state.json", "06_dense_depth_metric_calibration", "completed", inputs=[str(root / row["depth_meters_npy"]), str(workspace / "outputs/00_rgb_frames/camera.json")], outputs=[str(out)], notes=f"Projected {report['projected_valid_rgb_pixels']} metric depth pixels into global right-eye frame 0; used directly for RGB-D point-cloud alignment.")
    print(json.dumps(jsonable(report), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
