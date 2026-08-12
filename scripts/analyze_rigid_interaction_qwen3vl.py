#!/usr/bin/env python3
"""Use Qwen3-VL to choose a rigid interaction target and reconstruction keyframe."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vlm_sam2_recon.rigid_pipeline.common import update_stage_state  # noqa: E402


DEFAULT_MODEL_PATH = "/code/models/Qwen3-VL-8B-Instruct"

PROMPT = """You are selecting a rigid object reconstruction target from an egocentric hand-object interaction video.

Return strict JSON only. Do not use markdown.

Task:
1. Describe the hand-object interaction event.
2. Identify the main object being manipulated by the hand.
3. Decide whether this object is rigid. Choose a rigid object, not a hand, table, background, bag, box, or articulated laptop.
4. Select one best keyframe for object mesh reconstruction: the object should be visible, sharp, low-occlusion, and have a clear silhouette.
5. Select 2-4 alternate keyframes useful for mask correction/tracking.
6. Provide a normalized bbox in [0,1000] for the object in each selected keyframe.
7. Mention if hand removal / inpainting is needed before segmentation.

Use the frame indices exactly as labeled in the input. If uncertain, lower confidence instead of hallucinating.

JSON schema:
{
  "scene_summary": "short description",
  "interaction_event": {
    "event_type": "pick|place|move|rotate|open|close|touch|unknown",
    "description": "what the hand does to the object",
    "hand_side": "left|right|both|unknown",
    "evidence_frames": [0],
    "confidence": 0.0
  },
  "target_object": {
    "object_id": "target_rigid_object",
    "name_zh": "Chinese name",
    "name_en": "English canonical name",
    "category": "phone|cup|bottle|box|tool|toy|remote|other|unknown",
    "object_class_for_reconstruction": "rigid|articulated|unknown",
    "rigidity_reason": "why it should be treated as one rigid body",
    "selected_keyframe": {
      "frame_index": 0,
      "time_sec": 0.0,
      "bbox_2d_norm_1000": [0, 0, 1000, 1000],
      "occlusion_level": "none|low|medium|high|unknown",
      "view_quality": "excellent|good|fair|poor|unknown",
      "selection_reason": "why this is the best mesh prompt frame"
    },
    "alternate_keyframes": [
      {
        "frame_index": 0,
        "time_sec": 0.0,
        "bbox_2d_norm_1000": [0, 0, 1000, 1000],
        "reason": "why useful"
      }
    ],
    "needs_hand_removal_for_mesh_prompt": true,
    "needs_hand_removal_for_mask_tracking": true,
    "texture_or_geometry_notes": "object appearance notes",
    "confidence": 0.0
  },
  "ignored_objects": [
    {"name_en": "table", "reason": "background"}
  ],
  "uncertainty_notes": ["short notes"]
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rigid-object VLM target and keyframe selector.")
    parser.add_argument("--workspace", type=Path, default=PROJECT_ROOT / "run_rigid_20260715_215524")
    parser.add_argument("--frame-dir", type=Path, default=None)
    parser.add_argument("--timeline", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=Path(DEFAULT_MODEL_PATH))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--raw-fps", type=float, default=15.0)
    parser.add_argument("--keyframes", default=None, help="Comma-separated frame indices. Default samples uniformly.")
    parser.add_argument("--max-keyframes", type=int, default=18)
    parser.add_argument("--frame-max-tokens", type=int, default=768)
    parser.add_argument("--frame-min-tokens", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=1800)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--no-flash-attn", action="store_true")
    return parser.parse_args()


def extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


def parse_keyframes(raw: str | None, frame_count: int, max_keyframes: int) -> list[int]:
    if raw:
        values = [int(item.strip()) for item in raw.split(",") if item.strip()]
        return sorted(set(max(0, min(frame_count - 1, item)) for item in values))
    if frame_count <= max_keyframes:
        return list(range(frame_count))
    step = (frame_count - 1) / float(max_keyframes - 1)
    return sorted(set(round(i * step) for i in range(max_keyframes)))


def load_frame_timestamps(timeline: Path | None, frame_count: int, fallback_fps: float) -> list[float]:
    fallback = [index / fallback_fps for index in range(frame_count)]
    if timeline is None:
        return fallback
    with timeline.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != frame_count:
        raise ValueError(f"Timeline/frame count mismatch: timeline={len(rows)} frames={frame_count}")
    timestamps = []
    for index, row in enumerate(rows):
        timestamp = row.get("rgb_timestamp_s") or row.get("timestamp_s")
        if timestamp is None:
            raise KeyError(f"Timeline row {index} has no rgb_timestamp_s/timestamp_s")
        timestamps.append(float(timestamp))
    return timestamps


def validate_result(parsed: dict, frame_count: int) -> None:
    target = parsed.get("target_object")
    if not isinstance(target, dict):
        raise ValueError("Qwen result has no target_object")
    if target.get("object_class_for_reconstruction") != "rigid":
        raise ValueError(f"Qwen target is not a rigid object: {target.get('object_class_for_reconstruction')}")
    selected = target.get("selected_keyframe")
    if not isinstance(selected, dict):
        raise ValueError("Qwen target has no selected_keyframe")
    items = [selected] + list(target.get("alternate_keyframes") or [])
    for item in items:
        frame_index = item.get("frame_index")
        box = item.get("bbox_2d_norm_1000")
        if not isinstance(frame_index, int) or not 0 <= frame_index < frame_count:
            raise ValueError(f"Invalid Qwen keyframe index: {frame_index}")
        if not isinstance(box, list) or len(box) != 4 or not all(isinstance(value, (int, float)) for value in box):
            raise ValueError(f"Invalid Qwen bbox: {box}")
        x0, y0, x1, y1 = [float(value) for value in box]
        if not (0 <= x0 < x1 <= 1000 and 0 <= y0 < y1 <= 1000):
            raise ValueError(f"Qwen bbox is outside normalized bounds: {box}")


def main() -> int:
    args = parse_args()
    workspace = args.workspace.resolve()
    frame_dir = (args.frame_dir or workspace / "outputs/00_rgb_frames/right_rgb_png").resolve()
    if args.timeline is None:
        args.timeline = workspace / "outputs/00_rgb_frames/timeline.csv"
    model_path = args.model_path.resolve()
    output = (args.output or workspace / "outputs/01_vlm/event_and_keyframes.json").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frames = sorted(frame_dir.glob("*.png"))
    if not frames:
        raise FileNotFoundError(f"No PNG frames found in {frame_dir}")
    keyframes = parse_keyframes(args.keyframes, len(frames), args.max_keyframes)
    frame_timestamps = load_frame_timestamps(args.timeline.resolve() if args.timeline else None, len(frames), args.raw_fps)

    content = [
        {
            "type": "text",
            "text": (
                PROMPT
                + f"\nInput is {len(keyframes)} labeled keyframes from {len(frames)} total frames at {args.raw_fps} fps. "
                + "Use only labeled frame indices for keyframe and evidence fields."
            ),
        }
    ]
    for idx in keyframes:
        content.extend(
            [
                {"type": "text", "text": f"Frame {idx} (captured t={frame_timestamps[idx]:.6f}s):"},
                {
                    "type": "image",
                    "image": f"file://{frames[idx]}",
                    "min_pixels": args.frame_min_tokens * 32 * 32,
                    "max_pixels": args.frame_max_tokens * 32 * 32,
                },
            ]
        )
    messages = [{"role": "user", "content": content}]

    print(f"Loading model: {model_path}")
    load_kwargs = {"dtype": torch.bfloat16, "device_map": "auto"}
    if not args.no_flash_attn:
        load_kwargs["attn_implementation"] = "flash_attention_2"
    model = AutoModelForImageTextToText.from_pretrained(str(model_path), **load_kwargs)
    model.eval()
    processor = AutoProcessor.from_pretrained(str(model_path))

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages, image_patch_size=16)
    inputs = processor(text=text, images=images, videos=videos, return_tensors="pt", do_resize=False).to(model.device)

    generation_kwargs = {"max_new_tokens": args.max_new_tokens, "do_sample": args.temperature > 0}
    if args.temperature > 0:
        generation_kwargs["temperature"] = args.temperature

    print(f"Running Qwen3-VL rigid analysis on keyframes: {keyframes}")
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **generation_kwargs)
    generated_trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated_ids)]
    response = processor.batch_decode(generated_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    raw_path = output.with_suffix(".raw.txt")
    raw_path.write_text(response + "\n", encoding="utf-8")

    parsed = None
    parse_error = None
    try:
        parsed = extract_json(response)
        target = parsed.get("target_object") if isinstance(parsed, dict) else None
        if isinstance(target, dict):
            target.setdefault("object_id", "target_rigid_object")
            selected = target.get("selected_keyframe") or {}
            frame_index = selected.get("frame_index")
            if isinstance(frame_index, int) and 0 <= frame_index < len(frames):
                selected["frame_file"] = frames[frame_index].name
                selected["frame_path"] = str(frames[frame_index].resolve())
                selected["time_sec"] = frame_timestamps[frame_index]
            target["selected_keyframe"] = selected
            for alt in target.get("alternate_keyframes", []) or []:
                alt_frame = alt.get("frame_index")
                if isinstance(alt_frame, int) and 0 <= alt_frame < len(frames):
                    alt["frame_file"] = frames[alt_frame].name
                    alt["frame_path"] = str(frames[alt_frame].resolve())
                    alt["time_sec"] = frame_timestamps[alt_frame]
            parsed["target_object"] = target
            parsed["target_objects"] = [target]
        validate_result(parsed, len(frames))
    except Exception as exc:
        parse_error = repr(exc)

    result = {
        "metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_path": str(model_path),
            "frame_dir": str(frame_dir),
            "timeline": str(args.timeline.resolve()) if args.timeline else None,
            "frame_count": len(frames),
            "raw_fps": args.raw_fps,
            "timestamp_policy": "captured RGB timestamps from timeline" if args.timeline else "fallback frame_index/raw_fps",
            "keyframe_indices": keyframes,
            "keyframe_files": [frames[idx].name for idx in keyframes],
            "raw_response_path": str(raw_path),
            "flash_attention_2": not args.no_flash_attn,
        },
        "vlm_result": parsed,
    }
    if parse_error:
        result["parse_error"] = parse_error
        result["raw_model_output"] = response
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    update_stage_state(
        workspace / "pipeline_state.json",
        "01_vlm_event_and_keyframes",
        "failed" if parse_error else "completed",
        inputs=[str(frame_dir), str(args.timeline.resolve())],
        outputs=[str(output), str(raw_path)],
        notes=f"Qwen output validation failed: {parse_error}" if parse_error else "Qwen rigid target, keyframes, and bboxes passed schema validation.",
    )
    print(f"Wrote JSON: {output}")
    print(f"Wrote raw response: {raw_path}")
    return 2 if parse_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
