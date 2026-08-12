#!/usr/bin/env python3
"""Interactive mixed-pipeline object masks on the DiffuEraser hand-removed video."""

from __future__ import annotations

import argparse
import json
import shutil
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--vlm-json", type=Path, default=None)
    parser.add_argument("--object-id", default=None, help="Manipulated object id from mixed_interactions.json.")
    parser.add_argument("--inpainted-video", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--frame-cache-dir",
        type=Path,
        default=None,
        help="Local cache for decoded SAM2 JPEGs; defaults under /tmp to avoid slow workspace small-file I/O.",
    )
    parser.add_argument(
        "--output-cache-dir",
        type=Path,
        default=None,
        help="Local backing directory for new per-object mask outputs; the workspace keeps a stable symlink.",
    )
    parser.add_argument("--sam2-root", type=Path, default=DEFAULT_SAM2_ROOT)
    parser.add_argument("--sam2-config", default=DEFAULT_SAM2_CONFIG)
    parser.add_argument("--sam2-checkpoint", type=Path, default=DEFAULT_SAM2_CHECKPOINT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--overwrite-frames", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-area-ratio", type=float, default=4.5)
    parser.add_argument(
        "--save-overlays",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write full-resolution visualization frames after propagation.",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7864)
    parser.add_argument("--offload-video-to-cpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--offload-state-to-cpu", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--check", action="store_true", help="Validate paths and selected object; do not load SAM2.")
    parser.set_defaults(enable_hunyuan=False)
    return parser.parse_args()


def load_fps(workspace: Path, explicit: float | None) -> float:
    if explicit is not None:
        return float(explicit)
    manifest_path = workspace / "outputs/00_rgb_frames/stage00_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return float(manifest.get("target_fps", manifest.get("effective_fps", 15.0)))


def load_mixed_target(vlm_json: Path, object_id: str | None) -> tuple[dict, dict]:
    payload = json.loads(vlm_json.read_text(encoding="utf-8"))
    result = payload.get("vlm_result") or payload
    objects = result.get("objects") or []
    if not objects:
        raise ValueError(f"No manipulated objects in {vlm_json}")
    if object_id is None:
        rigid = [item for item in objects if item.get("object_class") == "rigid"]
        target = rigid[0] if rigid else objects[0]
    else:
        matches = [item for item in objects if item.get("object_id") == object_id]
        if len(matches) != 1:
            raise KeyError(f"Expected one object_id={object_id!r}, found {len(matches)}")
        target = matches[0]
    return payload, target


def mirror_latest_result_for_legacy_scripts(workspace: Path, object_output_dir: Path, target: dict) -> None:
    """Write a small compatibility index for scripts that still look under outputs/02."""
    prompt_manifest_path = object_output_dir / "mesh_prompt_frame0/prompt_manifest.json"
    propagation_manifest_path = object_output_dir / "propagation_manifest.json"
    if not prompt_manifest_path.is_file() or not propagation_manifest_path.is_file():
        return
    prompt = json.loads(prompt_manifest_path.read_text(encoding="utf-8"))
    compat_root = workspace / "outputs/02_sam2_frame0_masks"
    summary_path = compat_root / "sam2_frame0_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        objects = [item for item in summary.get("objects", []) if item.get("object_id") != target["object_id"]]
    else:
        summary = {
            "stage": "04_sam2_object_masks_compat_frame0_summary",
            "status": "completed",
            "camera": "right",
            "coordinate_frame": "global_frame0_right_image",
            "frame_index": 0,
            "source": "Compatibility mirror of interactive Stage 04 object masks.",
            "objects": [],
        }
        objects = []
    objects.append(
        {
            "object_id": target["object_id"],
            "name_en": target.get("name_en"),
            "name_zh": target.get("name_zh"),
            "object_class": target.get("object_class"),
            "frame_index": 0,
            "rgb": prompt["rgb"],
            "mask": prompt["mask"],
            "mask_source": "interactive_stage04_on_diffueraser",
        }
    )
    summary["objects"] = sorted(objects, key=lambda item: item["object_id"])
    write_json(summary_path, summary)

    source_mask_dir = object_output_dir / "objects" / target["object_id"]
    compat_mask_dir = compat_root / "propagated/objects" / target["object_id"]
    if source_mask_dir.is_dir():
        compat_mask_dir.parent.mkdir(parents=True, exist_ok=True)
        if compat_mask_dir.exists():
            shutil.rmtree(compat_mask_dir)
        shutil.copytree(source_mask_dir, compat_mask_dir)


def main() -> int:
    args = parse_args()
    workspace = args.workspace.resolve()
    vlm_json = (args.vlm_json or workspace / "outputs/01_vlm/mixed_interactions.json").resolve()
    inpainted_video = (
        args.inpainted_video
        or workspace / "outputs/03_diffueraser/inpainted_right_rgb_15fps.mp4"
    ).resolve()
    for path in (vlm_json, inpainted_video, workspace / "outputs/00_rgb_frames/right_rgb_png"):
        if not path.exists():
            raise FileNotFoundError(path)
    _, target = load_mixed_target(vlm_json, args.object_id)
    object_id = str(target["object_id"])
    target["_mirror_mixed_legacy_masks"] = True
    output_root = (args.output_dir or workspace / "outputs/04_object_masks").resolve()
    output_dir = output_root / object_id
    if not output_dir.exists() and not output_dir.is_symlink():
        output_cache_root = (
            args.output_cache_dir.resolve()
            if args.output_cache_dir
            else Path("/tmp/vlm_sam2_recon_cache") / workspace.name / object_id / "artifact_output"
        )
        output_cache_root.mkdir(parents=True, exist_ok=True)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        output_dir.symlink_to(output_cache_root, target_is_directory=True)
    metadata = video_metadata(inpainted_video)
    stage00_frames = list_display_frames(workspace / "outputs/00_rgb_frames/right_rgb_png")
    if int(metadata["frame_count"]) != len(stage00_frames):
        raise ValueError(f"Inpainted/Stage00 frame mismatch: {metadata['frame_count']} vs {len(stage00_frames)}")
    cache_root = (
        args.frame_cache_dir.resolve()
        if args.frame_cache_dir
        else Path("/tmp/vlm_sam2_recon_cache") / workspace.name / object_id
    )
    jpeg_dir = cache_root / "inpainted_sam2_jpeg"
    extract_frames(inpainted_video, jpeg_dir, None, args.overwrite_frames)
    display_frames = list_display_frames(jpeg_dir)
    sam2_frames = list_display_frames(jpeg_dir)
    fps = load_fps(workspace, args.fps)
    preflight = {
        "stage": "04_sam2_object_masks",
        "policy": "Human positive/negative point prompts on the DiffuEraser hand-removed 15fps video.",
        "vlm_json": str(vlm_json),
        "inpainted_video": str(inpainted_video),
        "object_id": object_id,
        "object_class": target.get("object_class"),
        "frame_count": len(display_frames),
        "fps": fps,
        "output_dir": str(output_dir),
    }
    write_json(output_dir / "object_mask_preflight.json", preflight)
    if args.check:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 0

    args.fps = fps
    result = run_server(
        args,
        workspace=workspace,
        output_dir=output_dir,
        target=target,
        sam2_frames=sam2_frames,
        display_frames=display_frames,
    )
    mirror_latest_result_for_legacy_scripts(workspace, output_dir, target)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
