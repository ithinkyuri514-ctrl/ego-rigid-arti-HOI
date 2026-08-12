#!/usr/bin/env python3
"""Interactive SAM2 propagation for one articulated part on hand-removed RGB."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vlm_sam2_recon.rigid_pipeline.common import video_metadata, write_json  # noqa: E402
from vlm_sam2_recon.rigid_pipeline.sam2_video import (  # noqa: E402
    DEFAULT_SAM2_CHECKPOINT,
    DEFAULT_SAM2_CONFIG,
    DEFAULT_SAM2_ROOT,
    list_display_frames,
)
from rigid_stage04_object_masks import extract_frames  # noqa: E402
from rigid_stage04_object_mask_server import run_server  # noqa: E402


DEFAULT_WORKSPACE = PROJECT_ROOT / "run_mixed_20260728_203734"
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--object-id", default="microwave", help="Parent articulated object id.")
    parser.add_argument("--part-id", required=True, help="Particulate child/link id, for example door or link_0.")
    parser.add_argument("--part-name", default=None, help="Optional human-readable part name.")
    parser.add_argument(
        "--parent-mask-dir",
        type=Path,
        default=None,
        help="Whole-object propagated masks used as a hard per-frame intersection constraint.",
    )
    parser.add_argument("--inpainted-video", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Exact part output directory; defaults to outputs/04_object_masks/<object>/parts/<part>.",
    )
    parser.add_argument(
        "--frame-cache-dir",
        type=Path,
        default=None,
        help="Shared decoded-JPEG cache; defaults under /tmp/vlm_sam2_recon_cache.",
    )
    parser.add_argument("--sam2-root", type=Path, default=DEFAULT_SAM2_ROOT)
    parser.add_argument("--sam2-config", default=DEFAULT_SAM2_CONFIG)
    parser.add_argument("--sam2-checkpoint", type=Path, default=DEFAULT_SAM2_CHECKPOINT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--overwrite-frames", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-area-ratio", type=float, default=6.0)
    parser.add_argument(
        "--save-overlays",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save propagated-mask overlays so every result frame remains inspectable in the UI.",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7865)
    parser.add_argument("--offload-video-to-cpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--offload-state-to-cpu", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--check", action="store_true", help="Validate the right-eye video contract without loading SAM2.")
    parser.set_defaults(enable_hunyuan=False)
    return parser.parse_args()


def validate_id(label: str, value: str) -> str:
    if not SAFE_ID.fullmatch(value):
        raise ValueError(f"{label} must match {SAFE_ID.pattern!r}, got {value!r}")
    return value


def load_stage00(workspace: Path) -> tuple[dict, float]:
    path = workspace / "outputs/00_rgb_frames/stage00_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    eye = str(payload.get("selected_eye", "")).lower()
    if eye not in {"left", "right"}:
        raise ValueError(f"Unsupported selected_eye={eye!r} in {path}")
    fps = float(payload.get("target_fps", payload.get("effective_fps", 15.0)))
    return payload, fps


def main() -> int:
    args = parse_args()
    workspace = args.workspace.resolve()
    object_id = validate_id("--object-id", args.object_id)
    part_id = validate_id("--part-id", args.part_id)
    stage00, stage00_fps = load_stage00(workspace)
    selected_eye = str(stage00["selected_eye"]).lower()
    fps = float(args.fps if args.fps is not None else stage00_fps)
    inpainted_video = (
        args.inpainted_video
        or workspace / "outputs/03_diffueraser/inpainted_right_rgb_15fps.mp4"
    ).resolve()
    raw_rgb_dir = workspace / "outputs/00_rgb_frames/right_rgb_png"
    parent_mask_dir = (
        args.parent_mask_dir
        or workspace / "outputs/04_object_masks" / object_id / "objects" / object_id
    ).resolve()
    for path in (inpainted_video, raw_rgb_dir, parent_mask_dir):
        if not path.exists():
            raise FileNotFoundError(path)

    output_dir = (
        args.output_dir
        or workspace / "outputs/04_object_masks" / object_id / "parts" / part_id
    ).resolve()
    metadata = video_metadata(inpainted_video)
    raw_frames = list_display_frames(raw_rgb_dir)
    if int(metadata["frame_count"]) != len(raw_frames):
        raise ValueError(
            f"Inpainted/Stage00 frame mismatch: {metadata['frame_count']} vs {len(raw_frames)}"
        )
    if abs(float(metadata["fps"]) - fps) > 0.05:
        raise ValueError(f"Inpainted video is {metadata['fps']} FPS, expected {fps} FPS")
    parent_masks = sorted(parent_mask_dir.glob("*.png"))
    if len(parent_masks) != len(raw_frames):
        raise ValueError(f"Parent-mask/Stage00 frame mismatch: {len(parent_masks)} vs {len(raw_frames)}")

    cache_dir = (
        args.frame_cache_dir.resolve()
        if args.frame_cache_dir
        else Path("/tmp/vlm_sam2_recon_cache") / workspace.name / "stage03_inpainted_sam2_jpeg"
    )
    target = {
        "object_id": part_id,
        "parent_object_id": object_id,
        "part_id": part_id,
        "name_en": args.part_name or part_id,
        "name_zh": args.part_name or part_id,
        "object_class": "articulated_dynamic_part",
        "artifact_kind": "articulated_part_mask",
        "camera": selected_eye,
        "image_coordinate_frame": f"{selected_eye}_rgb_image_pixels",
        "source_policy": f"Interactive SAM2 on DiffuEraser hand-removed {selected_eye}-eye RGB.",
        "constraint_mask_dir": str(parent_mask_dir),
        "_skip_pipeline_state_updates": True,
    }
    preflight = {
        "stage": "04_sam2_articulated_part_mask",
        "status": "ready_for_interactive_prompts",
        "parent_object_id": object_id,
        "part_id": part_id,
        "selected_eye": selected_eye,
        "world_frame": f"frame0_{selected_eye}_camera_opencv_rdf",
        "image_coordinate_frame": f"{selected_eye}_rgb_image_pixels",
        "tracking_rgb_policy": "diffueraser_hand_removed",
        "parent_mask_constraint": {
            "operation": "per_frame_intersection",
            "mask_dir": str(parent_mask_dir),
            "frame_count": len(parent_masks),
        },
        "inpainted_video": str(inpainted_video),
        "sam2_frame_dir": str(cache_dir),
        "frame_count": len(raw_frames),
        "fps": fps,
        "width": int(metadata["width"]),
        "height": int(metadata["height"]),
        "timestamp_start_s": stage00.get("timestamp_start_s"),
        "output_dir": str(output_dir),
        "propagated_mask_dir": str(output_dir / "objects" / part_id),
        "propagation_manifest": str(output_dir / "propagation_manifest.json"),
    }
    if args.check:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 0

    extract_frames(inpainted_video, cache_dir, None, args.overwrite_frames)
    sam2_frames = list_display_frames(cache_dir)
    if len(sam2_frames) != len(raw_frames):
        raise ValueError(f"Decoded/inpainted frame mismatch: {len(sam2_frames)} vs {len(raw_frames)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "part_mask_preflight.json", preflight)
    args.fps = fps
    return run_server(
        args,
        workspace=workspace,
        output_dir=output_dir,
        target=target,
        sam2_frames=sam2_frames,
        display_frames=sam2_frames,
    )


if __name__ == "__main__":
    raise SystemExit(main())
