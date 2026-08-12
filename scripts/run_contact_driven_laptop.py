#!/usr/bin/env python3
"""Run contact-driven laptop screen motion from 15fps EgoForce hand meshes."""

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
    parser = argparse.ArgumentParser(description="Contact-driven articulated laptop screen + hand correction.")
    parser.add_argument("--alignment-dir", type=Path, default=DEFAULT_ALIGNMENT_DIR)
    parser.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT_ROOT)
    parser.add_argument("--rgb-dir", type=Path, default=DEFAULT_RGB_DIR)
    parser.add_argument("--hand-dir", type=Path, default=DEFAULT_HAND_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=57)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--pose-fps", type=float, default=5.0)
    parser.add_argument(
        "--pose-csv",
        type=Path,
        default=None,
        help="Optional per-tracker-frame head pose CSV. When provided, pose row index i is used for RGB/hand frame i.",
    )
    parser.add_argument("--hand-side", choices=["left", "right"], default="left")
    parser.add_argument("--contact-force-frame", type=int, default=None)
    parser.add_argument(
        "--vlm-contact-json",
        type=Path,
        default=None,
        help="Optional Qwen/VLM JSON containing the first semantic hand-laptop contact frame.",
    )
    parser.add_argument("--vlm-contact-target-id", default="target_laptop")
    parser.add_argument("--vlm-contact-mode", choices=["force", "window"], default="force")
    parser.add_argument("--vlm-contact-window-before", type=int, default=0)
    parser.add_argument("--vlm-contact-window-after", type=int, default=24)
    parser.add_argument(
        "--contact-fingers",
        default=None,
        help="Optional comma-separated semantic fingertip candidates, e.g. thumb,index. VLM JSON is used when omitted.",
    )
    parser.add_argument("--contact-distance-threshold-m", type=float, default=None)
    parser.add_argument("--contact-distance-consecutive-frames", type=int, default=3)
    parser.add_argument("--contact-distance-min-hits", type=int, default=2)
    parser.add_argument("--contact-search-start", type=int, default=0)
    parser.add_argument("--contact-search-end", type=int, default=None)
    parser.add_argument("--screen-angle-min-deg", type=float, default=-120.0)
    parser.add_argument("--screen-angle-max-deg", type=float, default=140.0)
    parser.add_argument("--max-hand-translation-m", type=float, default=0.18)
    parser.add_argument("--contact-scale-m", type=float, default=0.015)
    parser.add_argument("--radius-scale-m", type=float, default=0.025)
    parser.add_argument("--axis-scale-m", type=float, default=0.025)
    parser.add_argument("--hand-prior-scale-m", type=float, default=0.075)
    parser.add_argument("--hand-refine-mode", choices=["translation", "global_rigid"], default="translation")
    parser.add_argument("--max-hand-rotation-deg", type=float, default=28.0)
    parser.add_argument("--hand-rot-prior-scale-deg", type=float, default=18.0)
    parser.add_argument("--hand-rot-smooth-scale-deg", type=float, default=12.0)
    parser.add_argument("--theta-smooth-scale-deg", type=float, default=16.0)
    parser.add_argument("--theta-acc-scale-deg", type=float, default=24.0)
    parser.add_argument("--weight-contact", type=float, default=8.0)
    parser.add_argument("--weight-radius", type=float, default=3.5)
    parser.add_argument("--weight-axis", type=float, default=3.5)
    parser.add_argument("--weight-hand-prior", type=float, default=1.0)
    parser.add_argument("--weight-hand-smooth", type=float, default=1.5)
    parser.add_argument("--weight-hand-rot-prior", type=float, default=0.4)
    parser.add_argument("--weight-hand-rot-smooth", type=float, default=0.8)
    parser.add_argument("--weight-theta-smooth", type=float, default=0.6)
    parser.add_argument("--weight-theta-acc", type=float, default=0.35)
    parser.add_argument("--weight-penetration", type=float, default=1.0)
    parser.add_argument("--monotonic-slack-deg", type=float, default=8.0)
    parser.add_argument("--enforce-monotonic-after-contact", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contact_fingers = None
    if args.contact_fingers:
        contact_fingers = tuple(item.strip() for item in args.contact_fingers.split(",") if item.strip())
    config = ContactDrivenScreenConfig(
        alignment_dir=args.alignment_dir,
        export_root=args.export_root,
        rgb_dir=args.rgb_dir,
        hand_dir=args.hand_dir,
        output_dir=args.output_dir,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        fps=args.fps,
        pose_fps=args.pose_fps,
        pose_csv=args.pose_csv,
        hand_side=args.hand_side,
        contact_force_frame=args.contact_force_frame,
        vlm_contact_json=args.vlm_contact_json,
        vlm_contact_target_id=args.vlm_contact_target_id,
        vlm_contact_mode=args.vlm_contact_mode,
        vlm_contact_window_before=args.vlm_contact_window_before,
        vlm_contact_window_after=args.vlm_contact_window_after,
        contact_fingers=contact_fingers,
        contact_distance_threshold_m=args.contact_distance_threshold_m,
        contact_distance_consecutive_frames=args.contact_distance_consecutive_frames,
        contact_distance_min_hits=args.contact_distance_min_hits,
        contact_search_start=args.contact_search_start,
        contact_search_end=args.contact_search_end,
        screen_angle_min_deg=args.screen_angle_min_deg,
        screen_angle_max_deg=args.screen_angle_max_deg,
        max_hand_translation_m=args.max_hand_translation_m,
        contact_scale_m=args.contact_scale_m,
        radius_scale_m=args.radius_scale_m,
        axis_scale_m=args.axis_scale_m,
        hand_prior_scale_m=args.hand_prior_scale_m,
        hand_refine_mode=args.hand_refine_mode,
        max_hand_rotation_deg=args.max_hand_rotation_deg,
        hand_rot_prior_scale_deg=args.hand_rot_prior_scale_deg,
        hand_rot_smooth_scale_deg=args.hand_rot_smooth_scale_deg,
        theta_smooth_scale_deg=args.theta_smooth_scale_deg,
        theta_acc_scale_deg=args.theta_acc_scale_deg,
        weight_contact=args.weight_contact,
        weight_radius=args.weight_radius,
        weight_axis=args.weight_axis,
        weight_hand_prior=args.weight_hand_prior,
        weight_hand_smooth=args.weight_hand_smooth,
        weight_hand_rot_prior=args.weight_hand_rot_prior,
        weight_hand_rot_smooth=args.weight_hand_rot_smooth,
        weight_theta_smooth=args.weight_theta_smooth,
        weight_theta_acc=args.weight_theta_acc,
        weight_penetration=args.weight_penetration,
        monotonic_slack_deg=args.monotonic_slack_deg,
        enforce_monotonic_after_contact=args.enforce_monotonic_after_contact,
    )
    manifest = run_contact_driven_screen(config)
    contact = manifest["contact"]
    print(f"Wrote contact-driven sequence: {manifest['output_dir']}")
    print(
        "Contact frame "
        f"{contact['frame']:06d} ({contact['fingertip_name']}, distance={contact['distance_m']:.4f} m)"
    )
    print(f"Manifest: {Path(manifest['output_dir']) / 'dynamic_manifest.json'}")


if __name__ == "__main__":
    main()
