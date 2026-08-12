#!/usr/bin/env python3
"""Prepare the mixed pipeline's frame-0-anchored 15 fps right-eye timeline."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--spatial-export", type=Path, required=True)
    parser.add_argument("--eye", choices=["left", "right"], default="right")
    parser.add_argument("--target-fps", type=float, default=15.0)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def exported_rgb_bounds(frames_csv: Path) -> tuple[float, float]:
    with frames_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No rows in {frames_csv}")
    key = "rgb_timestamp_s" if "rgb_timestamp_s" in rows[0] else "rgb_timestamp"
    timestamps = [float(row[key]) for row in rows]
    if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError(f"Non-monotonic RGB timestamps in {frames_csv}")
    return timestamps[0], timestamps[-1]


def main() -> int:
    args = parse_args()
    spatial_export = args.spatial_export.resolve()
    start_timestamp, end_timestamp = exported_rgb_bounds(spatial_export / "frames.csv")
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/rigid_stage00_prepare.py"),
        "--workspace",
        str(args.workspace.resolve()),
        "--video",
        str(args.video.resolve()),
        "--spatial-export",
        str(spatial_export),
        "--eye",
        args.eye,
        "--target-fps",
        str(args.target_fps),
        "--start-timestamp",
        repr(start_timestamp),
        "--end-timestamp",
        repr(end_timestamp),
    ]
    if args.overwrite:
        command.append("--overwrite")
    subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
