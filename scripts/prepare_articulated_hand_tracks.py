#!/usr/bin/env python3
"""Prepare disjoint SAM2 hand-track clips with global timeline mappings."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


DEFAULT_TRACKS = [
    {
        "track_id": "left_hand_1",
        "global_start": 0,
        "global_end": None,
        "event_ids": [],
        "event_roles": ["derive_from_final_mask_overlap"],
        "note": "First visible episode of the left hand; valid interval is derived after SAM2/manual QC.",
    },
    {
        "track_id": "left_hand_2",
        "global_start": 0,
        "global_end": None,
        "event_ids": [],
        "event_roles": ["derive_from_final_mask_overlap"],
        "note": "Same left hand after re-entry; re-entry and valid interval are derived from the actual mask/QC.",
    },
    {
        "track_id": "right_hand",
        "global_start": 0,
        "global_end": None,
        "event_ids": [],
        "event_roles": ["derive_from_final_mask_overlap"],
        "note": "Independent right-hand identity track; empty masks are valid while the hand is out of view.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workspace = args.workspace.resolve()
    stage00 = workspace / "outputs/00_rgb_frames"
    timeline_path = stage00 / "timeline.csv"
    png_root = stage00 / "right_rgb_png"
    jpeg_root = stage00 / "sam2_jpeg"
    output_root = workspace / "outputs/02_hand_masks"
    clips_root = workspace / "scratch/hand_track_clips"
    if not timeline_path.is_file():
        raise FileNotFoundError(timeline_path)
    with timeline_path.open(newline="", encoding="utf-8") as stream:
        timeline = list(csv.DictReader(stream))
    if args.overwrite:
        shutil.rmtree(output_root, ignore_errors=True)
        shutil.rmtree(clips_root, ignore_errors=True)
    output_root.mkdir(parents=True, exist_ok=True)
    clips_root.mkdir(parents=True, exist_ok=True)

    plan = {
        "camera_eye": "left",
        "global_frame_count": len(timeline),
        "global_reference": "frame0_left_camera",
        "policy": {
            "one_sam2_object_per_track": True,
            "left_hand_split_reason": "hand exits the image and later re-enters; do not bridge identity across the gap",
            "diffueraser_mask": "OR all tracks after mapping local masks back to global frames",
            "contact_masks": "keep tracks separate for per-hand event/contact association",
        },
        "tracks": [],
    }
    for spec in DEFAULT_TRACKS:
        start = max(0, int(spec["global_start"]))
        end = len(timeline) - 1 if spec["global_end"] is None else min(len(timeline) - 1, int(spec["global_end"]))
        track_root = clips_root / spec["track_id"]
        png_dir = track_root / "display_png"
        jpeg_dir = track_root / "sam2_jpeg"
        png_dir.mkdir(parents=True, exist_ok=True)
        jpeg_dir.mkdir(parents=True, exist_ok=True)
        mapping = []
        for local_index, global_index in enumerate(range(start, end + 1)):
            png_source = png_root / f"{global_index:06d}.png"
            jpeg_source = jpeg_root / f"{global_index:06d}.jpg"
            if not png_source.is_file() or not jpeg_source.is_file():
                raise FileNotFoundError(png_source if not png_source.is_file() else jpeg_source)
            (png_dir / f"{local_index:06d}.png").symlink_to(png_source)
            (jpeg_dir / f"{local_index:06d}.jpg").symlink_to(jpeg_source)
            row = timeline[global_index]
            mapping.append(
                {
                    "local_index": local_index,
                    "global_rgb_index": global_index,
                    "source_video_frame_index": int(row["source_rgb_index"]),
                    "timestamp_s": float(row["rgb_timestamp_s"]),
                }
            )
        mapping_path = track_root / "frame_mapping.json"
        mapping_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        record = dict(spec)
        record.update(
            {
                "global_start": start,
                "global_end": end,
                "valid_global_start": None,
                "valid_global_end": None,
                "valid_range_source": "post_sam2_mask_and_manual_identity_qc",
                "local_frame_count": len(mapping),
                "sam2_frame_dir": str(jpeg_dir),
                "display_frame_dir": str(png_dir),
                "output_dir": str(output_root / spec["track_id"]),
                "frame_mapping": str(mapping_path),
            }
        )
        plan["tracks"].append(record)
    plan_path = output_root / "hand_track_plan.json"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"plan": str(plan_path), "tracks": plan["tracks"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
