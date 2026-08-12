#!/usr/bin/env python3
"""Propagate a Qwen-selected object box with SAM2 on the hand-removed video."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vlm_sam2_recon.rigid_pipeline.common import (  # noqa: E402
    update_stage_state,
    video_metadata,
    write_json,
)
from vlm_sam2_recon.rigid_pipeline.sam2_video import (  # noqa: E402
    DEFAULT_SAM2_CHECKPOINT,
    DEFAULT_SAM2_CONFIG,
    DEFAULT_SAM2_ROOT,
    add_points_or_box,
    build_video_predictor,
    list_display_frames,
    load_vlm_target,
    mask_sequence_qc,
    normalized_box_to_pixels,
    propagate_bidirectional,
    save_propagation_outputs,
)


def parse_args() -> argparse.Namespace:
    workspace = PROJECT_ROOT / "run_rigid_20260715_215524"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=workspace)
    parser.add_argument("--vlm-json", type=Path, default=None)
    parser.add_argument("--inpainted-video", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--sam2-root", type=Path, default=DEFAULT_SAM2_ROOT)
    parser.add_argument("--sam2-config", default=DEFAULT_SAM2_CONFIG)
    parser.add_argument("--sam2-checkpoint", type=Path, default=DEFAULT_SAM2_CHECKPOINT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--use-alternate-boxes", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--overwrite-frames", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-area-ratio", type=float, default=3.5)
    parser.add_argument("--allow-qc-failure", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fill-mask-holes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--interactive", action="store_true", help="Open a browser UI for human point prompts.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7864)
    parser.add_argument("--offload-video-to-cpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--offload-state-to-cpu", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--check", action="store_true", help="Validate VLM/video contracts without loading SAM2.")
    return parser.parse_args()


def extract_frames(video: Path, jpeg_dir: Path, png_dir: Path | None, overwrite: bool) -> None:
    if overwrite:
        for directory in (jpeg_dir, png_dir):
            if directory is None:
                continue
            if directory.exists():
                shutil.rmtree(directory)
    jpeg_dir.mkdir(parents=True, exist_ok=True)
    if png_dir is not None:
        png_dir.mkdir(parents=True, exist_ok=True)
    if list(jpeg_dir.glob("*.jpg")) and (png_dir is None or list(png_dir.glob("*.png"))):
        return
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video),
            "-start_number",
            "0",
            "-q:v",
            "2",
            str(jpeg_dir / "%06d.jpg"),
        ],
        check=True,
    )
    if png_dir is not None:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(video),
                "-start_number",
                "0",
                str(png_dir / "%06d.png"),
            ],
            check=True,
        )


def resolve_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    workspace = args.workspace.resolve()
    vlm_json = (args.vlm_json or workspace / "outputs/01_vlm/event_and_keyframes.json").resolve()
    inpainted_video = (
        args.inpainted_video
        or workspace / "outputs/03_diffueraser/inpainted_right_rgb_15fps.mp4"
    ).resolve()
    output_dir = (args.output_dir or workspace / "outputs/04_object_masks").resolve()
    for path in (vlm_json, inpainted_video):
        if not path.is_file():
            raise FileNotFoundError(path)
    return vlm_json, inpainted_video, output_dir


def conditioning_boxes(target: dict, width: int, height: int, use_alternates: bool) -> list[dict]:
    selected = target.get("selected_keyframe") or {}
    items = [selected]
    if use_alternates:
        items.extend(target.get("alternate_keyframes") or [])
    prompts = []
    for item in items:
        frame_index = item.get("frame_index")
        box = item.get("bbox_2d_norm_1000")
        if not isinstance(frame_index, int) or not isinstance(box, list):
            continue
        prompts.append(
            {
                "frame_index": frame_index,
                "bbox_2d_norm_1000": box,
                "box_xyxy_pixels": normalized_box_to_pixels(box, width, height),
            }
        )
    if not prompts:
        raise ValueError("Qwen output contains no valid selected keyframe bbox")
    return prompts


def solid_object_mask(mask: np.ndarray) -> np.ndarray:
    """Convert a transparent-object response into one solid silhouette."""
    binary = np.asarray(mask, dtype=np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        binary = (labels == largest).astype(np.uint8)
    points = cv2.findNonZero(binary)
    if points is not None and len(points) >= 3:
        hull = cv2.convexHull(points)
        solid = np.zeros_like(binary)
        cv2.fillConvexPoly(solid, hull, 1)
        binary = solid
    return np.asarray(binary_fill_holes(binary > 0), dtype=bool)


def main() -> int:
    args = parse_args()
    workspace = args.workspace.resolve()
    vlm_json, inpainted_video, output_dir = resolve_inputs(args)
    _, target = load_vlm_target(vlm_json)
    metadata = video_metadata(inpainted_video)
    prompts = conditioning_boxes(target, int(metadata["width"]), int(metadata["height"]), args.use_alternate_boxes)
    stage00_frames = list_display_frames(workspace / "outputs/00_rgb_frames/right_rgb_png")
    if int(metadata["frame_count"]) != len(stage00_frames):
        raise ValueError(
            f"Inpainted/Stage00 frame mismatch: {metadata['frame_count']} vs {len(stage00_frames)}"
        )
    for item in prompts:
        if not 0 <= item["frame_index"] < len(stage00_frames):
            raise IndexError(f"Qwen frame index outside video: {item['frame_index']}")
    preflight = {
        "stage": "04_sam2_object_masks",
        "policy": "Qwen box prompts propagated only on the DiffuEraser hand-removed video",
        "vlm_json": str(vlm_json),
        "inpainted_video": str(inpainted_video),
        "target_id": target.get("object_id", "target_rigid_object"),
        "video": metadata,
        "prompts": prompts,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "object_mask_preflight.json", preflight)
    if args.check:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        print("Stage 04 contract check passed; SAM2 was not loaded.")
        return 0

    jpeg_dir = output_dir / "inpainted_sam2_jpeg"
    png_dir = output_dir / "inpainted_rgb_png"
    extract_frames(inpainted_video, jpeg_dir, png_dir, args.overwrite_frames)
    display_frames = list_display_frames(png_dir)
    sam2_frames = list_display_frames(jpeg_dir)
    if len(display_frames) != len(stage00_frames) or len(sam2_frames) != len(stage00_frames):
        raise ValueError("Extracted inpainted frame count does not match Stage 00")

    if args.interactive:
        from rigid_stage04_object_mask_server import run_server

        return run_server(
            args,
            workspace=workspace,
            output_dir=output_dir,
            target=target,
            sam2_frames=sam2_frames,
            display_frames=display_frames,
        )

    predictor = build_video_predictor(
        args.sam2_root,
        args.sam2_config,
        args.sam2_checkpoint,
        args.device,
    )
    state = predictor.init_state(str(jpeg_dir), offload_video_to_cpu=True, offload_state_to_cpu=False)
    object_id = str(target.get("object_id") or "target_rigid_object")
    conditioning = []
    for prompt in prompts:
        add_points_or_box(
            predictor,
            state,
            frame_index=prompt["frame_index"],
            object_id=object_id,
            box_xyxy=prompt["box_xyxy_pixels"],
        )
        conditioning.append(prompt["frame_index"])
    results = propagate_bidirectional(predictor, state, conditioning)
    if args.fill_mask_holes:
        for masks in results.values():
            for result_object_id, mask in list(masks.items()):
                masks[result_object_id] = solid_object_mask(mask)
    summary = save_propagation_outputs(
        results,
        display_frames,
        output_dir,
        fps=args.fps,
        video_stem="object_mask",
    )
    qc = mask_sequence_qc(summary, max_area_ratio=args.max_area_ratio)
    write_json(output_dir / "object_mask_qc.json", qc)

    prompt_dir = output_dir / "mesh_prompt_frame0"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(display_frames[0], prompt_dir / "rgb_no_hand.png")
    shutil.copy2(output_dir / "combined/000000.png", prompt_dir / "object_mask.png")
    write_json(
        prompt_dir / "prompt_manifest.json",
        {
            "frame_index": 0,
            "rgb": str(prompt_dir / "rgb_no_hand.png"),
            "mask": str(prompt_dir / "object_mask.png"),
            "source": "SAM2 propagation on DiffuEraser video from Qwen-selected bbox",
        },
    )
    update_stage_state(
        workspace / "pipeline_state.json",
        "04_sam2_object_masks",
        "completed" if qc["passed"] else "needs_revision",
        inputs=[str(vlm_json), str(inpainted_video)],
        outputs=[str(output_dir)],
        notes=(
            f"Qwen-box SAM2 propagation passed QC for {summary['frame_count']} frames."
            if qc["passed"]
            else f"Object-mask propagation needs revision: empty={qc['empty_frames']} jumps={qc['abrupt_area_change_frames']}"
        ),
    )
    summary["qc"] = qc
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if qc["passed"] or args.allow_qc_failure else 2


if __name__ == "__main__":
    raise SystemExit(main())
