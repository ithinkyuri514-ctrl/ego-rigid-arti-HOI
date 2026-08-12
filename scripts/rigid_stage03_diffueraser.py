#!/usr/bin/env python3
"""Validate Stage 02 artifacts and run DiffuEraser on the right-eye video."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vlm_sam2_recon.rigid_pipeline.common import (  # noqa: E402
    run_checked,
    update_stage_state,
    validate_matching_videos,
    write_json,
)


def parse_args() -> argparse.Namespace:
    workspace = PROJECT_ROOT / "run_rigid_20260715_215524"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=workspace)
    parser.add_argument("--input-video", type=Path, default=None)
    parser.add_argument("--input-mask", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--diffueraser-root",
        type=Path,
        default=Path("/code/ArtHOI-4D-Reconstruction/third_party/diffueraser"),
    )
    parser.add_argument("--python", type=Path, default=Path("/opt/conda/envs/diffueraser/bin/python"))
    parser.add_argument("--mask-dilation-iter", type=int, default=8)
    parser.add_argument("--max-img-size", type=int, default=960)
    parser.add_argument("--max-outside-mask-mae", type=float, default=8.0)
    parser.add_argument("--max-outside-mask-p95", type=float, default=24.0)
    parser.add_argument("--allow-qc-failure", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--reuse-existing-output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip DiffuEraser inference when the expected output video already exists.",
    )
    parser.add_argument(
        "--extract-png-frames",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also export a lossless PNG sequence. Stage 04 normally decodes the video directly.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def inpainting_qc(original_path: Path, mask_path: Path, output_path: Path, dilation: int) -> dict:
    captures = [cv2.VideoCapture(str(path)) for path in (original_path, mask_path, output_path)]
    if not all(capture.isOpened() for capture in captures):
        raise RuntimeError("Could not open one or more videos for DiffuEraser QC")
    frame_metrics = []
    kernel = np.ones((3, 3), dtype=np.uint8)
    while True:
        reads = [capture.read() for capture in captures]
        if not all(ok for ok, _ in reads):
            break
        original, mask_frame, output = [frame for _, frame in reads]
        mask = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY) > 127
        protected = cv2.dilate(mask.astype(np.uint8), kernel, iterations=max(1, dilation)) > 0
        outside = ~protected
        difference = np.mean(np.abs(output.astype(np.float32) - original.astype(np.float32)), axis=2)
        values = difference[outside]
        frame_metrics.append(
            {
                "frame_index": len(frame_metrics),
                "outside_mask_mae": float(values.mean()) if values.size else 0.0,
                "outside_mask_p95": float(np.quantile(values, 0.95)) if values.size else 0.0,
                "outside_mask_changed_fraction_gt20": float(np.mean(values > 20.0)) if values.size else 0.0,
            }
        )
    for capture in captures:
        capture.release()
    if not frame_metrics:
        raise RuntimeError("DiffuEraser QC decoded no frames")
    mae = float(np.median([item["outside_mask_mae"] for item in frame_metrics]))
    p95 = float(np.median([item["outside_mask_p95"] for item in frame_metrics]))
    return {
        "frame_count": len(frame_metrics),
        "median_outside_mask_mae": mae,
        "median_outside_mask_p95": p95,
        "frames": frame_metrics,
    }


def video_size(path: Path) -> tuple[int, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    return width, height


def restore_native_canvas(
    original_path: Path,
    mask_path: Path,
    output_path: Path,
    dilation: int,
) -> Path | None:
    """Composite a resized DiffuEraser result into the original-resolution RGB video."""
    target_width, target_height = video_size(original_path)
    output_width, output_height = video_size(output_path)
    if (output_width, output_height) == (target_width, target_height):
        return None

    raw_output = output_path.with_name(
        f"{output_path.stem}_raw_{output_width}x{output_height}{output_path.suffix}"
    )
    if raw_output.exists():
        raise FileExistsError(f"Refusing to overwrite preserved DiffuEraser output: {raw_output}")
    temporary_output = output_path.with_name(f"{output_path.stem}_native_tmp{output_path.suffix}")
    output_path.replace(raw_output)
    dilation_chain = ",".join(["dilation"] * max(1, dilation))
    filter_complex = (
        f"[1:v]scale={target_width}:{target_height}:flags=lanczos[inpaint];"
        f"[2:v]scale={target_width}:{target_height}:flags=neighbor,format=gray,"
        f"lut=y='if(gte(val,128),255,0)',{dilation_chain},gblur=sigma=2[mask];"
        "[0:v][inpaint][mask]maskedmerge[out]"
    )
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(original_path),
                "-i",
                str(raw_output),
                "-i",
                str(mask_path),
                "-filter_complex",
                filter_complex,
                "-map",
                "[out]",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-shortest",
                str(temporary_output),
            ],
            check=True,
        )
        temporary_output.replace(output_path)
    except Exception:
        temporary_output.unlink(missing_ok=True)
        raw_output.replace(output_path)
        raise
    return raw_output


def main() -> int:
    args = parse_args()
    workspace = args.workspace.resolve()
    input_video = (args.input_video or workspace / "outputs/00_rgb_frames/right_rgb_15fps.mp4").resolve()
    input_mask = (args.input_mask or workspace / "outputs/02_hand_masks/hand_mask.mp4").resolve()
    output_dir = (args.output_dir or workspace / "outputs/03_diffueraser").resolve()
    root = args.diffueraser_root.resolve()
    for path in (input_video, input_mask, root / "run_diffueraser.py", args.python.resolve()):
        if not path.exists():
            raise FileNotFoundError(path)
    validation = validate_matching_videos(input_video, input_mask, fps_tolerance=0.02)
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_output = output_dir / f"inpainted_{input_video.stem}.mp4"
    command = [
        str(args.python.resolve()),
        str(root / "run_diffueraser.py"),
        "--input_video",
        str(input_video),
        "--input_mask",
        str(input_mask),
        "--save_path",
        str(output_dir),
        "--mask_dilation_iter",
        str(args.mask_dilation_iter),
        "--max_img_size",
        str(args.max_img_size),
    ]
    manifest = {
        "stage": "03_diffueraser_hand_removal",
        "status": "dry_run" if args.dry_run else "running",
        "input_video": str(input_video),
        "input_mask": str(input_mask),
        "validation": validation,
        "expected_output": str(expected_output),
        "command": command,
    }
    write_json(output_dir / "diffueraser_manifest.json", manifest)
    reused_output = args.reuse_existing_output and expected_output.is_file()
    if reused_output:
        print(f"Reusing existing DiffuEraser output: {expected_output}", flush=True)
    else:
        run_checked(command, cwd=root, dry_run=args.dry_run)
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    if not expected_output.is_file():
        raise FileNotFoundError(f"DiffuEraser finished without expected output: {expected_output}")
    raw_output = restore_native_canvas(
        input_video,
        input_mask,
        expected_output,
        args.mask_dilation_iter,
    )
    output_validation = validate_matching_videos(input_video, expected_output, fps_tolerance=0.02)
    qc = inpainting_qc(input_video, input_mask, expected_output, max(1, args.mask_dilation_iter // 2))
    qc["threshold_max_outside_mask_mae"] = args.max_outside_mask_mae
    qc["threshold_max_outside_mask_p95"] = args.max_outside_mask_p95
    qc["passed"] = (
        qc["median_outside_mask_mae"] <= args.max_outside_mask_mae
        and qc["median_outside_mask_p95"] <= args.max_outside_mask_p95
    )
    frames_dir = None
    if args.extract_png_frames:
        frames_dir = output_dir / "frames_png"
        frames_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(expected_output),
                "-start_number", "0", str(frames_dir / "%06d.png"),
            ],
            check=True,
        )
    manifest.update(
        {
            "status": "completed" if qc["passed"] else "needs_revision",
            "output_validation": output_validation,
            "quality_control": qc,
            "output_frames": str(frames_dir) if frames_dir else None,
            "inference_reused": reused_output,
            "raw_diffueraser_output": str(raw_output) if raw_output else None,
        }
    )
    write_json(output_dir / "diffueraser_manifest.json", manifest)
    update_stage_state(
        workspace / "pipeline_state.json",
        "03_diffueraser_hand_removal",
        "completed" if qc["passed"] else "needs_revision",
        inputs=[str(input_video), str(input_mask)],
        outputs=[
            str(expected_output),
            *([str(frames_dir)] if frames_dir else []),
            str(output_dir / "diffueraser_manifest.json"),
        ],
        notes=(
            "DiffuEraser passed video-contract and outside-mask drift checks."
            if qc["passed"]
            else "DiffuEraser output needs revision because pixels outside the hand mask changed too much."
        ),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if qc["passed"] or args.allow_qc_failure else 2


if __name__ == "__main__":
    raise SystemExit(main())
