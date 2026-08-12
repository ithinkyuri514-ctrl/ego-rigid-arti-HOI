#!/usr/bin/env python3
"""Prepare a right-camera clip and hand-mask video for DiffuEraser.

This is intentionally a small utility for the rigid-object pipeline.  It does
not try to solve full hand segmentation; it creates a conservative hand mask
around the manipulation region so DiffuEraser can remove hand pixels before
object SAM2/Hunyuan prompting.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def parse_csv_ints(value: str) -> tuple[int, ...]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    return tuple(int(item) for item in items)


def parse_seed_points(value: str) -> list[tuple[int, int]]:
    if not value:
        return []
    points: list[tuple[int, int]] = []
    for chunk in value.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        x, y = parse_csv_ints(chunk)
        points.append((x, y))
    return points


def parse_boxes(value: str) -> list[tuple[int, int, int, int]]:
    if not value:
        return []
    boxes: list[tuple[int, int, int, int]] = []
    for chunk in value.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parsed = parse_csv_ints(chunk)
        if len(parsed) != 4:
            raise ValueError(f"Box must be x0,y0,x1,y1, got {chunk!r}")
        boxes.append((parsed[0], parsed[1], parsed[2], parsed[3]))
    return boxes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-dir", required=True, help="Directory containing stereo RGB PNG frames.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--frame-pattern", default="{frame:06d}.png")
    parser.add_argument(
        "--camera",
        choices=["right", "left", "full"],
        default="right",
        help="Which stereo crop to export. The rigid pipeline uses right camera.",
    )
    parser.add_argument(
        "--hand-roi",
        default="360,560,850,1232",
        help="x0,y0,x1,y1 ROI in cropped camera coordinates where hand removal is allowed.",
    )
    parser.add_argument(
        "--seed-points",
        default="520,1060;610,980;735,1040",
        help="Semicolon separated x,y points near hand pixels. Components near these points are kept.",
    )
    parser.add_argument(
        "--protect-boxes",
        default="",
        help="Semicolon separated x0,y0,x1,y1 boxes to erase from the hand mask, usually the target object.",
    )
    parser.add_argument("--max-component-distance", type=float, default=130.0)
    parser.add_argument("--min-component-area", type=int, default=80)
    parser.add_argument("--dilate-iter", type=int, default=9)
    parser.add_argument("--blur-kernel", type=int, default=7)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def crop_camera(image: Image.Image, camera: str) -> Image.Image:
    if camera == "full":
        return image
    width, height = image.size
    half = width // 2
    if camera == "left":
        return image.crop((0, 0, half, height))
    return image.crop((half, 0, width, height))


def component_distance_to_seeds(labels: np.ndarray, label_id: int, seeds: list[tuple[int, int]]) -> float:
    ys, xs = np.where(labels == label_id)
    if len(xs) == 0:
        return float("inf")
    if not seeds:
        return 0.0
    pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    best = float("inf")
    for sx, sy in seeds:
        dist = np.sqrt(((pts - np.array([[sx, sy]], dtype=np.float32)) ** 2).sum(axis=1)).min()
        best = min(best, float(dist))
    return best


def skin_hand_mask(rgb: np.ndarray, roi: tuple[int, int, int, int], seeds: list[tuple[int, int]], args: argparse.Namespace) -> np.ndarray:
    height, width = rgb.shape[:2]
    x0, y0, x1, y1 = roi
    x0, x1 = sorted((max(0, x0), min(width, x1)))
    y0, y1 = sorted((max(0, y0), min(height, y1)))

    crop = rgb[y0:y1, x0:x1]
    ycrcb = cv2.cvtColor(crop, cv2.COLOR_RGB2YCrCb)
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    y_channel, cr, cb = cv2.split(ycrcb)
    hue, sat, val = cv2.split(hsv)

    # Broad but ROI-limited skin detector. The connected-component pass below
    # keeps only regions close to the user-provided hand seed points.
    skin = (
        (y_channel > 35)
        & (cr > 132)
        & (cr < 182)
        & (cb > 72)
        & (cb < 145)
        & (sat > 18)
        & (val > 45)
        & (hue < 38)
    )
    skin = skin.astype(np.uint8) * 255
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)

    local_seeds = [(sx - x0, sy - y0) for sx, sy in seeds if x0 <= sx < x1 and y0 <= sy < y1]
    count, labels, stats, _ = cv2.connectedComponentsWithStats((skin > 0).astype(np.uint8), connectivity=8)
    kept = np.zeros_like(skin, dtype=np.uint8)
    for label_id in range(1, count):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < args.min_component_area:
            continue
        distance = component_distance_to_seeds(labels, label_id, local_seeds)
        if distance <= args.max_component_distance:
            kept[labels == label_id] = 255

    if args.dilate_iter > 0:
        kept = cv2.dilate(kept, np.ones((5, 5), np.uint8), iterations=args.dilate_iter)
    if args.blur_kernel > 1:
        k = args.blur_kernel if args.blur_kernel % 2 == 1 else args.blur_kernel + 1
        kept = cv2.GaussianBlur(kept, (k, k), 0)
        kept = (kept > 16).astype(np.uint8) * 255

    out = np.zeros((height, width), dtype=np.uint8)
    out[y0:y1, x0:x1] = kept
    for px0, py0, px1, py1 in parse_boxes(args.protect_boxes):
        px0, px1 = sorted((max(0, px0), min(width, px1)))
        py0, py1 = sorted((max(0, py0), min(height, py1)))
        out[py0:py1, px0:px1] = 0
    return out


def write_video_from_frames(frame_dir: Path, pattern: str, output: Path, fps: float) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frame_dir / pattern),
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        str(output),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    frame_dir = Path(args.frame_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    image_dir = output_dir / "right_frames"
    mask_dir = output_dir / "hand_mask_frames"
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {output_dir}")
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    roi = parse_csv_ints(args.hand_roi)
    if len(roi) != 4:
        raise ValueError("--hand-roi must be x0,y0,x1,y1")
    seeds = parse_seed_points(args.seed_points)
    protect_boxes = parse_boxes(args.protect_boxes)

    frame_records = []
    for out_idx, frame_id in enumerate(range(args.start_frame, args.end_frame + 1)):
        src = frame_dir / args.frame_pattern.format(frame=frame_id)
        if not src.is_file():
            raise FileNotFoundError(src)
        image = crop_camera(Image.open(src).convert("RGB"), args.camera)
        rgb = np.asarray(image)
        mask = skin_hand_mask(rgb, tuple(roi), seeds, args)

        image_path = image_dir / f"{out_idx:06d}.png"
        mask_path = mask_dir / f"{out_idx:06d}.png"
        image.save(image_path)
        Image.fromarray(mask).save(mask_path)
        frame_records.append(
            {
                "clip_frame": out_idx,
                "source_frame": frame_id,
                "image": str(image_path),
                "hand_mask": str(mask_path),
                "mask_pixels": int((mask > 0).sum()),
            }
        )

    video_path = output_dir / "right_input.mp4"
    mask_video_path = output_dir / "hand_mask.mp4"
    write_video_from_frames(image_dir, "%06d.png", video_path, args.fps)
    write_video_from_frames(mask_dir, "%06d.png", mask_video_path, args.fps)

    manifest = {
        "frame_dir": str(frame_dir),
        "camera": args.camera,
        "fps": args.fps,
        "start_frame": args.start_frame,
        "end_frame": args.end_frame,
        "hand_roi": list(roi),
        "seed_points": [list(item) for item in seeds],
        "protect_boxes": [list(item) for item in protect_boxes],
        "right_frame_dir": str(image_dir),
        "hand_mask_frame_dir": str(mask_dir),
        "input_video": str(video_path),
        "input_mask": str(mask_video_path),
        "frames": frame_records,
    }
    manifest_path = output_dir / "diffueraser_clip_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote DiffuEraser clip: {video_path}")
    print(f"Wrote hand mask video: {mask_video_path}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
