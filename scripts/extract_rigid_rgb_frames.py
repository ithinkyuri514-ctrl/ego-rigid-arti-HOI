#!/usr/bin/env python3
"""Extract fixed-fps RGB frames for a rigid-object reconstruction workspace."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract RGB frames and a timestamp timeline from an mp4.")
    parser.add_argument("--video-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    video_path = args.video_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    timeline_path = output_dir / "rgb_timeline.csv"
    manifest_path = output_dir / "rgb_extract_manifest.json"

    if not video_path.exists():
        raise FileNotFoundError(video_path)
    if timeline_path.exists() and not args.overwrite:
        print(f"RGB frames already extracted: {output_dir}")
        print(f"Timeline: {timeline_path}")
        return 0

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise RuntimeError("ffmpeg and ffprobe are required for reliable SpatialMP4 frame extraction.")

    probe = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "format=duration:stream=avg_frame_rate,nb_frames",
            "-of",
            "json",
            str(video_path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    probe_data = json.loads(probe.stdout)
    duration = float(probe_data.get("format", {}).get("duration") or 0.0)
    stream = (probe_data.get("streams") or [{}])[0]
    rate = str(stream.get("avg_frame_rate") or "0/0")
    num, den = [float(x) for x in rate.split("/")] if "/" in rate else (0.0, 1.0)
    source_fps = num / den if den else 0.0
    source_frame_count = int(stream.get("nb_frames") or 0)

    for old_png in output_dir.glob("*.png"):
        old_png.unlink()
    output_pattern = output_dir / "%06d.png"
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-map",
            "0:v:0",
            "-vf",
            f"fps={float(args.fps)}",
            "-start_number",
            "0",
            str(output_pattern),
        ],
        check=True,
    )

    extracted = sorted(output_dir.glob("*.png"))
    rows: list[dict[str, object]] = []
    for frame_id, out_path in enumerate(extracted):
        timestamp_s = frame_id / float(args.fps)
        rows.append(
            {
                "frame": frame_id,
                "timestamp_s": f"{timestamp_s:.9f}",
                "image": out_path.name,
                "source_video": str(video_path),
                "source_fps": f"{source_fps:.9f}",
            }
        )

    with timeline_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "timestamp_s", "image", "source_video", "source_fps"])
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "video_path": str(video_path),
        "output_dir": str(output_dir),
        "timeline_csv": str(timeline_path),
        "target_fps": float(args.fps),
        "source_fps": source_fps,
        "source_frame_count": source_frame_count,
        "duration_sec": duration,
        "extracted_frame_count": len(rows),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Extracted {len(rows)} RGB frames to {output_dir}")
    print(f"Timeline: {timeline_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
