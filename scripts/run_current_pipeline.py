#!/usr/bin/env python3
"""Run the accepted current laptop reconstruction pipeline.

This wrapper intentionally points to the contact-driven articulated laptop
pipeline. CoTracker RGB-D hinge tracking is kept as a legacy/baseline path, not
the default pipeline entry.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vlm_sam2_recon.stages.contact_driven_screen import (  # noqa: E402
    DEFAULT_ALIGNMENT_DIR,
    DEFAULT_EXPORT_ROOT,
    DEFAULT_HAND_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RGB_DIR,
    ContactDrivenScreenConfig,
    run_contact_driven_screen,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment-dir", type=Path, default=DEFAULT_ALIGNMENT_DIR)
    parser.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT_ROOT)
    parser.add_argument("--rgb-dir", type=Path, default=DEFAULT_RGB_DIR)
    parser.add_argument("--hand-dir", type=Path, default=DEFAULT_HAND_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=57)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--pose-fps", type=float, default=5.0)
    parser.add_argument("--hand-side", choices=["left", "right"], default="left")
    parser.add_argument("--contact-force-frame", type=int, default=None)
    parser.add_argument(
        "--vlm-contact-json",
        type=Path,
        default=None,
        help="Optional Qwen/VLM JSON containing the first semantic hand-laptop contact frame.",
    )
    parser.add_argument("--vlm-contact-target-id", default="target_laptop")
    parser.add_argument(
        "--contact-fingers",
        default=None,
        help="Optional comma-separated semantic fingertip candidates; otherwise read from the VLM contact JSON.",
    )
    parser.add_argument("--enforce-monotonic-after-contact", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-solver-nfev", type=int, default=120)
    parser.add_argument("--hand-refine-mode", choices=["translation", "global_rigid"], default="translation")
    parser.add_argument("--max-hand-rotation-deg", type=float, default=28.0)
    parser.add_argument("--hand-rot-prior-scale-deg", type=float, default=18.0)
    parser.add_argument("--hand-rot-smooth-scale-deg", type=float, default=12.0)
    parser.add_argument("--weight-hand-rot-prior", type=float, default=0.4)
    parser.add_argument("--weight-hand-rot-smooth", type=float, default=0.8)
    return parser.parse_args()


def main() -> None:
    values = vars(parse_args())
    raw_fingers = values.pop("contact_fingers")
    if raw_fingers:
        values["contact_fingers"] = tuple(item.strip() for item in raw_fingers.split(",") if item.strip())
    config = ContactDrivenScreenConfig(**values)
    manifest = run_contact_driven_screen(config)
    print(f"Saved current pipeline manifest: {manifest['output_dir']}/contact_driven_manifest.json")
    print(f"Saved angle/contact CSV: {manifest['optimization_csv']}")
    print(f"Saved dynamic manifest: {manifest['dynamic_manifest']}")
    print(f"Detected contact frame: {manifest['contact']['frame']} ({manifest['contact']['fingertip_name']})")


if __name__ == "__main__":
    main()
