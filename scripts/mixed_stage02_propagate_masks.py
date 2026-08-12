#!/usr/bin/env python3
"""Propagate confirmed mixed-pipeline frame-0 object masks over the full timeline."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vlm_sam2_recon.rigid_pipeline.common import update_stage_state, write_json  # noqa: E402
from vlm_sam2_recon.rigid_pipeline.sam2_video import (  # noqa: E402
    DEFAULT_SAM2_CHECKPOINT,
    DEFAULT_SAM2_CONFIG,
    DEFAULT_SAM2_ROOT,
    add_mask,
    build_video_predictor,
    list_display_frames,
    mask_sequence_qc,
    overlay_mask,
    propagate_bidirectional,
    save_propagation_outputs,
)


OBJECT_OVERLAY_COLORS = {
    "cup": (30, 144, 255),
    "microwave": (225, 55, 50),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--mask-summary", type=Path, default=None)
    parser.add_argument("--sam2-frame-dir", type=Path, default=None)
    parser.add_argument("--display-frame-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--sam2-root", type=Path, default=DEFAULT_SAM2_ROOT)
    parser.add_argument("--sam2-config", default=DEFAULT_SAM2_CONFIG)
    parser.add_argument("--sam2-checkpoint", type=Path, default=DEFAULT_SAM2_CHECKPOINT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--max-area-ratio", type=float, default=4.5)
    parser.add_argument("--allow-terminal-empty", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-qc-failure", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--offload-video-to-cpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--offload-state-to-cpu", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--check", action="store_true", help="Validate inputs without loading SAM2.")
    return parser.parse_args()


def per_object_qc(summary: dict, max_area_ratio: float, allow_terminal_empty: bool) -> dict:
    objects = {}
    for object_id in summary["object_ids"]:
        areas = np.asarray(
            [frame["objects"][object_id]["area_pixels"] for frame in summary["frames"]],
            dtype=np.int64,
        )
        empty = np.flatnonzero(areas <= 0)
        terminal_empty = []
        if empty.size:
            first_empty = int(empty[0])
            if np.all(areas[first_empty:] <= 0):
                terminal_empty = list(range(first_empty, len(areas)))
        allowed_empty = bool(allow_terminal_empty and empty.tolist() == terminal_empty)
        object_summary = {
            "frames": [
                {"combined_area_pixels": frame["objects"][object_id]["area_pixels"]}
                for frame in summary["frames"]
            ]
        }
        qc = mask_sequence_qc(
            object_summary,
            max_area_ratio=max_area_ratio,
            allow_empty=allowed_empty,
        )
        nonempty = np.flatnonzero(areas > 0)
        qc.update(
            {
                "visibility_start_frame": int(nonempty[0]) if nonempty.size else None,
                "visibility_end_frame": int(nonempty[-1]) if nonempty.size else None,
                "terminal_empty_frames": terminal_empty,
                "terminal_empty_allowed": bool(allow_terminal_empty),
                "nonterminal_empty_frames": sorted(set(empty.tolist()) - set(terminal_empty)),
            }
        )
        objects[object_id] = qc
    return {
        "passed": all(item["passed"] for item in objects.values()),
        "objects": objects,
    }


def save_per_object_overlays(summary: dict, display_frames: list[Path], output_dir: Path) -> dict[str, str]:
    overlay_root = output_dir / "object_overlays"
    if overlay_root.exists():
        shutil.rmtree(overlay_root)
    overlay_dirs = {}
    for object_index, object_id in enumerate(summary["object_ids"]):
        directory = overlay_root / object_id
        directory.mkdir(parents=True, exist_ok=True)
        color = OBJECT_OVERLAY_COLORS.get(
            object_id,
            ((40 + 83 * object_index) % 255, (120 + 47 * object_index) % 255, 220),
        )
        overlay_dirs[object_id] = str(directory)
        for frame in summary["frames"]:
            frame_index = int(frame["frame_index"])
            mask_path = Path(frame["objects"][object_id]["mask"])
            mask = np.asarray(Image.open(mask_path).convert("L")) > 127
            overlay_path = directory / f"{frame_index:06d}.jpg"
            overlay_mask(Image.open(display_frames[frame_index]), mask, color=color).save(
                overlay_path,
                quality=92,
            )
            frame["objects"][object_id]["overlay"] = str(overlay_path)
    return overlay_dirs


def load_fps(workspace: Path, explicit: float | None) -> float:
    if explicit is not None:
        if explicit <= 0:
            raise ValueError("--fps must be positive")
        return float(explicit)
    manifest_path = workspace / "outputs/00_rgb_frames/stage00_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fps_value = manifest.get("target_fps", manifest.get("effective_fps"))
    if fps_value is None:
        raise ValueError(f"Missing target_fps/effective_fps in {manifest_path}")
    fps = float(fps_value)
    if fps <= 0:
        raise ValueError(f"Invalid FPS in {manifest_path}: {fps}")
    return fps


def load_seed_masks(
    mask_summary_path: Path,
    expected_shape: tuple[int, int],
) -> tuple[dict[str, np.ndarray], list[dict]]:
    summary = json.loads(mask_summary_path.read_text(encoding="utf-8"))
    objects = summary.get("objects") or []
    if not objects:
        raise ValueError(f"No frame-0 objects in {mask_summary_path}")
    masks = {}
    records = []
    for item in objects:
        object_id = str(item["object_id"])
        if int(item.get("frame_index", 0)) != 0:
            raise ValueError(f"{object_id}: mixed propagation requires a global frame-0 seed")
        mask_path = Path(item["mask"]).resolve()
        if not mask_path.is_file():
            raise FileNotFoundError(mask_path)
        mask = np.asarray(Image.open(mask_path).convert("L")) > 127
        if mask.shape != expected_shape:
            raise ValueError(f"{object_id}: mask shape {mask.shape} != frame shape {expected_shape}")
        if not mask.any():
            raise ValueError(f"{object_id}: frame-0 seed mask is empty")
        if object_id in masks:
            raise ValueError(f"Duplicate object_id in mask summary: {object_id}")
        masks[object_id] = mask
        records.append(
            {
                "object_id": object_id,
                "source_mask": str(mask_path),
                "source": item.get("mask_source", "frame0_summary"),
                "area_pixels": int(mask.sum()),
            }
        )
    return masks, records


def main() -> int:
    args = parse_args()
    workspace = args.workspace.resolve()
    mask_summary_path = (
        args.mask_summary or workspace / "outputs/02_sam2_frame0_masks/sam2_frame0_summary.json"
    ).resolve()
    sam2_frame_dir = (args.sam2_frame_dir or workspace / "outputs/00_rgb_frames/sam2_jpeg").resolve()
    display_frame_dir = (
        args.display_frame_dir or workspace / "outputs/00_rgb_frames/right_rgb_png"
    ).resolve()
    output_dir = (
        args.output_dir or workspace / "outputs/02_sam2_frame0_masks/propagated"
    ).resolve()
    for path in (mask_summary_path, sam2_frame_dir, display_frame_dir):
        if not path.exists():
            raise FileNotFoundError(path)

    display_frames = list_display_frames(display_frame_dir)
    sam2_frames = list_display_frames(sam2_frame_dir)
    if len(display_frames) != len(sam2_frames):
        raise ValueError(f"SAM2/display frame mismatch: {len(sam2_frames)} vs {len(display_frames)}")
    expected_indices = list(range(len(display_frames)))
    if [int(path.stem) for path in display_frames] != expected_indices:
        raise ValueError("Display frame names must be contiguous zero-based integers")
    if [int(path.stem) for path in sam2_frames] != expected_indices:
        raise ValueError("SAM2 frame names must be contiguous zero-based integers")
    with Image.open(display_frames[0]) as image:
        expected_shape = (image.height, image.width)
    seed_masks, seed_records = load_seed_masks(mask_summary_path, expected_shape)
    fps = load_fps(workspace, args.fps)
    preflight = {
        "stage": "02_sam2_full_timeline_propagation",
        "policy": "Confirmed global frame-0 masks initialize forward SAM2 propagation on the full right-eye timeline.",
        "frame_count": len(display_frames),
        "fps": fps,
        "sam2_frame_dir": str(sam2_frame_dir),
        "display_frame_dir": str(display_frame_dir),
        "mask_summary": str(mask_summary_path),
        "output_dir": str(output_dir),
        "seed_frame_index": 0,
        "seed_masks": seed_records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "propagation_preflight.json", preflight)
    if args.check:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 0

    predictor = build_video_predictor(
        args.sam2_root,
        args.sam2_config,
        args.sam2_checkpoint,
        args.device,
    )
    state = predictor.init_state(
        str(sam2_frame_dir),
        offload_video_to_cpu=args.offload_video_to_cpu,
        offload_state_to_cpu=args.offload_state_to_cpu,
    )
    for object_id, mask in seed_masks.items():
        add_mask(
            predictor,
            state,
            frame_index=0,
            object_id=object_id,
            mask=mask,
        )
    results = propagate_bidirectional(predictor, state, [0])

    # Preserve the exact human-confirmed masks in exported frame 0. SAM2 still uses its
    # internally resized conditioning masks for propagation to subsequent frames.
    results[0].update(seed_masks)
    summary = save_propagation_outputs(
        results,
        display_frames,
        output_dir,
        fps=fps,
        video_stem="mixed_object_masks",
    )
    object_overlay_dirs = save_per_object_overlays(summary, display_frames, output_dir)
    qc = per_object_qc(summary, args.max_area_ratio, args.allow_terminal_empty)
    summary.update(
        {
            "stage": "02_sam2_full_timeline_propagation",
            "status": "completed" if qc["passed"] else "needs_revision",
            "direction": "forward_from_global_frame0",
            "seed_frame_index": 0,
            "seed_masks": seed_records,
            "object_overlay_dirs": object_overlay_dirs,
            "qc": qc,
        }
    )
    write_json(output_dir / "propagation_manifest.json", summary)
    write_json(output_dir / "propagation_qc.json", qc)
    update_stage_state(
        workspace / "pipeline_state.json",
        "02_sam2_frame0_masks",
        "completed" if qc["passed"] else "needs_revision",
        inputs=[str(mask_summary_path), str(sam2_frame_dir), str(display_frame_dir)],
        outputs=[str(workspace / "outputs/02_sam2_frame0_masks"), str(output_dir / "propagation_manifest.json")],
        notes=(
            f"Confirmed frame-0 masks and propagated {len(seed_masks)} objects over all "
            f"{len(display_frames)} right-eye frames; per-object QC passed."
            if qc["passed"]
            else f"Full-timeline object-mask propagation needs revision: {qc['objects']}"
        ),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if qc["passed"] or args.allow_qc_failure else 2


if __name__ == "__main__":
    raise SystemExit(main())
