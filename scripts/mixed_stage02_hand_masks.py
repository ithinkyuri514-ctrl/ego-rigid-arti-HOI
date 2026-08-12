#!/usr/bin/env python3
"""Create full-timeline SAM2 hand masks seeded by EgoForce mesh projections."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vlm_sam2_recon.rigid_pipeline.sam2_video import (  # noqa: E402
    DEFAULT_SAM2_CHECKPOINT,
    DEFAULT_SAM2_CONFIG,
    DEFAULT_SAM2_ROOT,
    add_mask,
    build_video_predictor,
    list_display_frames,
    mask_sequence_qc,
    propagate_bidirectional,
    save_propagation_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--anchor-frames", default="0,10,20,30")
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="Inclusive propagation end frame; defaults to the last sampled RGB frame.",
    )
    parser.add_argument("--sam2-root", type=Path, default=DEFAULT_SAM2_ROOT)
    parser.add_argument("--sam2-config", default=DEFAULT_SAM2_CONFIG)
    parser.add_argument("--sam2-checkpoint", type=Path, default=DEFAULT_SAM2_CHECKPOINT)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def projected_hand_mask(raw_path: Path, intrinsics: dict, shape: tuple[int, int]) -> np.ndarray:
    with np.load(raw_path) as raw:
        vertices = np.asarray(raw["hand_vertices"][1], dtype=np.float64)
        faces = np.asarray(raw["right_hand_faces"], dtype=np.int64)
        visible = bool(np.asarray(raw["visible_hand"]).reshape(2)[1])
    if not visible:
        raise ValueError(f"No right hand in {raw_path}")
    z = vertices[:, 2]
    uv = np.column_stack(
        [
            float(intrinsics["fx"]) * vertices[:, 0] / z + float(intrinsics["cx"]),
            float(intrinsics["fy"]) * vertices[:, 1] / z + float(intrinsics["cy"]),
        ]
    )
    height, width = shape
    mask = np.zeros(shape, dtype=np.uint8)
    for face in faces:
        if np.any(z[face] <= 0) or not np.isfinite(uv[face]).all():
            continue
        polygon = np.rint(uv[face]).astype(np.int32)
        if polygon[:, 0].max() < 0 or polygon[:, 1].max() < 0:
            continue
        if polygon[:, 0].min() >= width or polygon[:, 1].min() >= height:
            continue
        cv2.fillConvexPoly(mask, polygon, 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
    mask = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=1)
    if int(mask.sum()) < 500:
        raise ValueError(f"Projected hand mask too small in {raw_path}: {int(mask.sum())}")
    return mask.astype(bool)


def main() -> int:
    args = parse_args()
    workspace = args.workspace.resolve()
    rgb_frames = list_display_frames(workspace / "outputs/00_rgb_frames/right_rgb_png")
    sam2_frames = list_display_frames(workspace / "outputs/00_rgb_frames/sam2_jpeg")
    if len(rgb_frames) != len(sam2_frames):
        raise ValueError("RGB/SAM2 frame count mismatch")
    end_frame = len(rgb_frames) - 1 if args.end_frame is None else int(args.end_frame)
    if not 0 <= end_frame < len(rgb_frames):
        raise ValueError("Invalid --end-frame")
    anchors = sorted({int(value) for value in args.anchor_frames.split(",")})
    if any(frame < 0 or frame > end_frame for frame in anchors):
        raise ValueError("Hand-mask anchors must lie inside the propagation interval")
    camera = json.loads((workspace / "outputs/00_rgb_frames/camera.json").read_text(encoding="utf-8"))
    height, width = np.asarray(Image.open(rgb_frames[0])).shape[:2]
    seeds = {}
    for frame in anchors:
        raw_path = workspace / f"outputs/09_egoforce/raw_Ct/{frame:06d}_egoforce_meshes.npz"
        seeds[frame] = projected_hand_mask(raw_path, camera["rgb_intrinsics_right"], (height, width))

    predictor = build_video_predictor(
        args.sam2_root,
        args.sam2_config,
        args.sam2_checkpoint,
        args.device,
    )
    state = predictor.init_state(str(workspace / "outputs/00_rgb_frames/sam2_jpeg"), offload_video_to_cpu=True)
    for frame, mask in seeds.items():
        add_mask(predictor, state, frame_index=frame, object_id="right_hand", mask=mask)
    results = propagate_bidirectional(predictor, state, anchors)
    results = {frame: masks for frame, masks in results.items() if frame <= end_frame}
    for frame, mask in seeds.items():
        results[frame]["right_hand"] = mask

    output = workspace / "outputs/02_hand_masks"
    if output.exists():
        shutil.rmtree(output)
    summary = save_propagation_outputs(
        results,
        rgb_frames[: end_frame + 1],
        output,
        fps=15.0,
        video_stem="hand_mask",
    )
    qc = mask_sequence_qc(summary, max_area_ratio=4.5)
    manifest = {
        **summary,
        "stage": "02_sam2_right_hand_masks",
        "status": "completed" if qc["passed"] else "needs_revision",
        "source": "egoforce_projection_seeded_sam2",
        "propagation_end_frame_inclusive": end_frame,
        "timeline_frame_count": len(rgb_frames),
        "conditioning_frames": anchors,
        "qc": qc,
    }
    (output / "propagation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"frame_count": summary["frame_count"], "propagation_end_frame_inclusive": end_frame, "conditioning_frames": anchors, "qc": qc}, indent=2))
    return 0 if qc["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
