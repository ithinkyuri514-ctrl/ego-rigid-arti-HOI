#!/usr/bin/env python3
"""Place raw EgoForce hands and a static laptop in frame-0 coordinates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vlm_sam2_recon.stages.pose_compensated_scene import (  # noqa: E402
    PoseCompensatedSceneConfig,
    run_pose_compensated_scene,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment-dir", type=Path, required=True)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--rgb-dir", type=Path, required=True)
    parser.add_argument("--hand-dir", type=Path, required=True)
    parser.add_argument("--pose-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=135)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--hand-side", choices=["left", "right"], default="right")
    return parser.parse_args()


def main() -> None:
    manifest = run_pose_compensated_scene(PoseCompensatedSceneConfig(**vars(parse_args())))
    print(f"Wrote pose-only sequence: {manifest['output_dir']}")
    print(f"Frames: {manifest['frame_indices'][0]}-{manifest['frame_indices'][-1]}")
    print(f"Detected {manifest['hand_side']} hand frames: {manifest['detected_hand_frames']}")
    print(f"Manifest: {manifest['output_dir']}/dynamic_manifest.json")


if __name__ == "__main__":
    main()

