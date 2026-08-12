#!/usr/bin/env python3
"""Refine the first-frame laptop screen and hinge using RGB-D observations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vlm_sam2_recon.stages.screen_metric_calibration import (  # noqa: E402
    ScreenMetricCalibrationConfig,
    run_screen_metric_calibration,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment-dir", type=Path, required=True)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scale-min", type=float, default=0.85)
    parser.add_argument("--scale-max", type=float, default=1.15)
    parser.add_argument("--shift-max-m", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    result = run_screen_metric_calibration(ScreenMetricCalibrationConfig(**vars(parse_args())))
    print(f"Axis shift: {1000.0 * result['plane_diagnostics']['axis_origin_perpendicular_shift_m']:.2f} mm")
    print(f"Axis direction delta: {result['plane_diagnostics']['axis_direction_delta_deg']:.3f} deg")
    print(f"Screen scales: axis={result['axis_scale']:.5f}, radial={result['radial_scale']:.5f}")
    print(f"Target bbox: {result['target_bbox_px']}")
    print(f"Final bbox: {result['final_bbox_px']}")


if __name__ == "__main__":
    main()
