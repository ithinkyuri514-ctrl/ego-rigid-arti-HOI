#!/usr/bin/env python3
"""Run stable CoTracker3 + RGB-D + one-DoF hinge screen tracking."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vlm_sam2_recon.stages.screen_hinge_tracking import (  # noqa: E402
    DEFAULT_ALIGNMENT_DIR,
    DEFAULT_COTRACKER_CHECKPOINT,
    DEFAULT_COTRACKER_ROOT,
    DEFAULT_EXPORT_ROOT,
    DEFAULT_OUTPUT_DIR,
    ScreenHingeTrackingConfig,
    run_screen_hinge_tracking,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment-dir", type=Path, default=DEFAULT_ALIGNMENT_DIR)
    parser.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=19)
    parser.add_argument("--rgb-dir-name", default="rgb_right_png")
    parser.add_argument("--tracker-rgb-dir", type=Path, default=None)
    parser.add_argument("--tracker-start-frame", type=int, default=None)
    parser.add_argument("--tracker-end-frame", type=int, default=None)
    parser.add_argument("--tracker-stride", type=int, default=1)
    parser.add_argument("--tracker-fps", type=float, default=0.0)
    parser.add_argument("--eval-fps", type=float, default=0.0)
    parser.add_argument("--depth-dir-name", default="depth_meters_npy")
    parser.add_argument("--depth-unit", choices=["m", "meters", "mm", "auto"], default="m")
    parser.add_argument("--depth-convention", choices=["camera_to_rig", "rig_to_camera", "direct_same_camera"], default=None)
    parser.add_argument("--depth-sample-mode", choices=["auto", "projected", "aligned_rgb"], default="auto")
    parser.add_argument("--depth-min-m", type=float, default=0.1)
    parser.add_argument("--depth-max-m", type=float, default=3.0)
    parser.add_argument("--timestamp-tolerance-s", type=float, default=0.035)

    parser.add_argument("--cotracker-root", type=Path, default=DEFAULT_COTRACKER_ROOT)
    parser.add_argument("--cotracker-checkpoint", type=Path, default=DEFAULT_COTRACKER_CHECKPOINT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tracker-max-side", type=int, default=768)
    parser.add_argument("--cotracker-conf-threshold", type=float, default=0.85)
    parser.add_argument("--cotracker-iters", type=int, default=6)

    parser.add_argument("--init-max-points", type=int, default=80)
    parser.add_argument("--init-anchor-count", type=int, default=3)
    parser.add_argument("--init-anchor-min-pixel-distance", type=float, default=120.0)
    parser.add_argument("--init-anchor-min-axis-radius-m", type=float, default=0.0)
    parser.add_argument("--init-anchor-min-axis-radius-quantile", type=float, default=0.0)
    parser.add_argument("--init-anchor-axis-distance-weight", type=float, default=0.8)
    parser.add_argument("--init-anchor-feature-weight", type=float, default=1.0)
    parser.add_argument("--init-anchor-top-weight", type=float, default=0.6)
    parser.add_argument("--init-erode-px", type=int, default=8)
    parser.add_argument("--init-min-distance-px", type=float, default=16.0)
    parser.add_argument("--init-quality-level", type=float, default=0.01)
    parser.add_argument("--reseed-points", type=int, default=3)
    parser.add_argument("--min-reseed-registered-points", type=int, default=1)
    parser.add_argument("--reseed-quality-level", type=float, default=0.006)
    parser.add_argument("--reseed-min-distance-px", type=float, default=14.0)
    parser.add_argument("--min-valid-points", type=int, default=3)
    parser.add_argument("--low-valid-points", type=int, default=1)
    parser.add_argument("--max-reseed-events", type=int, default=3)
    parser.add_argument("--min-reseed-interval", type=int, default=2)

    parser.add_argument("--depth-sample-radius-px", type=int, default=3)
    parser.add_argument("--projected-depth-radius-px", type=float, default=8.0)
    parser.add_argument("--roi-dilation-px", type=int, default=18)
    parser.add_argument("--query-plane-dist-thresh-m", type=float, default=0.05)
    parser.add_argument("--plane-dist-thresh-m", type=float, default=0.06)
    parser.add_argument("--reproj-prefilter-thresh-px", type=float, default=90.0)
    parser.add_argument("--reproj-inlier-thresh-px", type=float, default=45.0)
    parser.add_argument("--max-track-jump-px", type=float, default=130.0)
    parser.add_argument("--depth-residual-thresh-m", type=float, default=0.16)
    parser.add_argument("--depth-only-min-points", type=int, default=120)

    parser.add_argument("--angle-method", choices=["loss_1d", "three_point_direct"], default="three_point_direct")
    parser.add_argument("--angle-sign", type=float, default=1.0)
    parser.add_argument("--three-point-count", type=int, default=3)
    parser.add_argument("--three-point-min-used-points", type=int, default=2)
    parser.add_argument("--three-point-candidate-count", type=int, default=8)
    parser.add_argument("--three-point-min-axis-radius-m", type=float, default=0.035)
    parser.add_argument("--three-point-reproj-prefilter-thresh-px", type=float, default=260.0)
    parser.add_argument("--three-point-max-mad-deg", type=float, default=10.0)
    parser.add_argument("--three-point-max-residual-deg", type=float, default=24.0)
    parser.add_argument("--three-point-incremental", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--three-point-max-depth-jump-m", type=float, default=0.08)
    parser.add_argument("--three-point-depth-ratio-max", type=float, default=1.25)
    parser.add_argument("--three-point-monotonic-slack-deg", type=float, default=8.0)
    parser.add_argument("--three-point-max-delta-deg", type=float, default=45.0)
    parser.add_argument("--three-point-allow-depth-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--angle-min-deg", type=float, default=-20.0)
    parser.add_argument("--angle-max-deg", type=float, default=140.0)
    parser.add_argument("--angle-search-radius-deg", type=float, default=14.0)
    parser.add_argument("--reappear-search-radius-deg", type=float, default=32.0)
    parser.add_argument("--coarse-steps", type=int, default=73)
    parser.add_argument("--max-angle-delta-deg", type=float, default=45.0)
    parser.add_argument("--lambda-reproj", type=float, default=1.0)
    parser.add_argument("--lambda-depth", type=float, default=1.0)
    parser.add_argument("--lambda-plane", type=float, default=0.6)
    parser.add_argument("--lambda-temporal", type=float, default=0.02)
    parser.add_argument("--lambda-acc", type=float, default=0.01)
    parser.add_argument("--reproj-scale-px", type=float, default=18.0)
    parser.add_argument("--depth-scale-m", type=float, default=0.045)
    parser.add_argument("--plane-scale-m", type=float, default=0.035)
    parser.add_argument("--robust-delta", type=float, default=1.5)
    parser.add_argument("--mad-sigma", type=float, default=3.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ScreenHingeTrackingConfig(**vars(args))
    manifest = run_screen_hinge_tracking(config)
    print(f"Saved stable manifest: {manifest['output_dir']}/stable_manifest.json")
    print(f"Saved angle CSV: {manifest['angles_csv']}")
    print(f"Saved overlay video: {manifest['overlay_video']}")
    for event in manifest.get("reseed_events", []):
        print(f"Reseed pass {event.get('pass')}: {event}")


if __name__ == "__main__":
    main()
