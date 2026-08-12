#!/usr/bin/env python3
"""Use Qwen3-VL to propose global event slices for an articulated object run."""

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

DEFAULT_MODEL_PATH = "/code/models/Qwen3-VL-8B-Instruct"

PROMPT = """You are analyzing an egocentric RGB video containing an articulated object.
Return strict JSON only, without markdown.

Identify one articulated object and split its interaction into separate event episodes.
The current target is likely a microwave with a fixed body and a hinged door, but do not
assume the object name or hard-code event frame numbers. The same object instance may
appear in multiple events (for example opening and closing); preserve that identity.

Use the labeled frame indices exactly. Event boundaries are coarse candidates and must
include enough context before contact and after release. The final reconstruction will
use global frame indices and timestamps, not event-local frame zero.

JSON schema:
{
  "scene_summary": "short description",
  "objects": [
    {
      "object_id": "articulated_object_0",
      "name_en": "canonical name",
      "name_zh": "中文名",
      "object_class": "articulated",
      "root_part": "fixed/root part",
      "moving_parts": ["moving part"],
      "joint_hypotheses": [{"parent": "root", "child": "part", "joint_type": "revolute", "axis_hint": "description"}],
      "confidence": 0.0
    }
  ],
  "events": [
    {
      "event_id": "event_000",
      "object_id": "articulated_object_0",
      "active_part": "moving part",
      "interaction_type": "articulated",
      "action": "open|close|slide|rotate|unknown",
      "description": "what the hand does",
      "hand_side": "left|right|both|unknown",
      "start_frame_candidate": 0,
      "end_frame_candidate": 0,
      "evidence_frames": [0],
      "pre_state": "description",
      "post_state": "description",
      "contact_fingers_candidate": ["thumb", "index"],
      "confidence": 0.0
    }
  ],
  "mesh_keyframes": [
    {"frame_index": 0, "time_sec": 0.0, "reason": "low occlusion and clear whole object"}
  ],
  "uncertainty_notes": ["short notes"]
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--frame-dir", type=Path, default=None)
    parser.add_argument("--timeline", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=Path(DEFAULT_MODEL_PATH))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-keyframes", type=int, default=18)
    parser.add_argument("--keyframes", default=None)
    parser.add_argument("--frame-max-tokens", type=int, default=768)
    parser.add_argument("--frame-min-tokens", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=2400)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--no-flash-attn", action="store_true")
    return parser.parse_args()


def extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("VLM response contains no JSON object")
    return json.loads(text[start : end + 1])


def timeline_records(timeline: Path, count: int) -> list[dict]:
    with timeline.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != count:
        raise ValueError(f"timeline/frame mismatch: {len(rows)} vs {count}")
    records = []
    for global_index, row in enumerate(rows):
        records.append(
            {
                "global_rgb_index": global_index,
                "timestamp_s": float(row.get("rgb_timestamp_s") or row["timestamp_s"]),
                "source_video_frame_index": int(row.get("source_rgb_index", global_index)),
            }
        )
    return records


def selected_indices(raw: str | None, count: int, maximum: int) -> list[int]:
    if raw:
        return sorted(set(max(0, min(count - 1, int(v))) for v in raw.split(",") if v.strip()))
    if count <= maximum:
        return list(range(count))
    return sorted(set(round(i * (count - 1) / (maximum - 1)) for i in range(maximum)))


def attach_global_frame_metadata(parsed: dict, frames: list[Path], records: list[dict]) -> None:
    for obj in parsed.get("objects", []) or []:
        for joint in obj.get("joint_hypotheses", []) or []:
            joint["status"] = "unverified_vlm_semantic_hint"
            joint["must_validate_with"] = ["Particulate", "SAM2 part masks", "RGB-D geometry", "C0 projection"]
    for event in parsed.get("events", []) or []:
        for key in ("start_frame_candidate", "end_frame_candidate"):
            value = event.get(key)
            if isinstance(value, int) and 0 <= value < len(frames):
                event[f"{key}_timestamp_s"] = records[value]["timestamp_s"]
                event[f"{key}_source_video_frame_index"] = records[value]["source_video_frame_index"]
                event[f"{key}_source_frame_file"] = frames[value].name
        event["frame_index_space"] = "global_rgb_timeline"
    for item in parsed.get("mesh_keyframes", []) or []:
        value = item.get("frame_index")
        if isinstance(value, int) and 0 <= value < len(frames):
            item["time_sec"] = records[value]["timestamp_s"]
            item["source_video_frame_index"] = records[value]["source_video_frame_index"]
            item["frame_file"] = frames[value].name
            item["frame_index_space"] = "global_rgb_timeline"


def validate(parsed: dict, count: int) -> None:
    objects = parsed.get("objects")
    events = parsed.get("events")
    if not isinstance(objects, list) or not objects:
        raise ValueError("VLM returned no objects")
    if not isinstance(events, list) or not events:
        raise ValueError("VLM returned no events")
    articulated = [obj for obj in objects if obj.get("object_class") == "articulated"]
    if not articulated:
        raise ValueError("VLM returned no articulated object")
    object_ids = {obj.get("object_id") for obj in objects}
    for event in events:
        if event.get("object_id") not in object_ids:
            raise ValueError(f"Event references unknown object: {event.get('object_id')}")
        start = event.get("start_frame_candidate")
        end = event.get("end_frame_candidate")
        if not isinstance(start, int) or not isinstance(end, int) or not (0 <= start <= end < count):
            raise ValueError(f"Invalid global event range: {start}, {end}")


def main() -> int:
    args = parse_args()
    workspace = args.workspace.resolve()
    frame_dir = (args.frame_dir or workspace / "outputs/00_rgb_frames/right_rgb_png").resolve()
    timeline = (args.timeline or workspace / "outputs/00_rgb_frames/timeline.csv").resolve()
    output = (args.output or workspace / "outputs/01_vlm/event_slices.json").resolve()
    frames = sorted(frame_dir.glob("*.png"))
    if not frames:
        raise FileNotFoundError(frame_dir)
    records = timeline_records(timeline, len(frames))
    indices = selected_indices(args.keyframes, len(frames), args.max_keyframes)
    content = [{"type": "text", "text": PROMPT + f"\nThere are {len(frames)} global RGB frames. Use only these labels."}]
    for idx in indices:
        content += [
            {"type": "text", "text": f"Global frame {idx}, timestamp {records[idx]['timestamp_s']:.6f}s:"},
            {"type": "image", "image": f"file://{frames[idx]}", "min_pixels": args.frame_min_tokens * 32 * 32, "max_pixels": args.frame_max_tokens * 32 * 32},
        ]
    messages = [{"role": "user", "content": content}]
    print(f"Loading Qwen3-VL: {args.model_path}")
    load_kwargs = {"dtype": torch.bfloat16, "device_map": "auto"}
    if not args.no_flash_attn:
        load_kwargs["attn_implementation"] = "flash_attention_2"
    model = AutoModelForImageTextToText.from_pretrained(str(args.model_path.resolve()), **load_kwargs)
    model.eval()
    processor = AutoProcessor.from_pretrained(str(args.model_path.resolve()))
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages, image_patch_size=16)
    inputs = processor(text=text, images=images, videos=videos, return_tensors="pt", do_resize=False).to(model.device)
    generation = {"max_new_tokens": args.max_new_tokens, "do_sample": args.temperature > 0}
    if args.temperature > 0:
        generation["temperature"] = args.temperature
    with torch.inference_mode():
        generated = model.generate(**inputs, **generation)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
    response = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".raw.txt").write_text(response + "\n", encoding="utf-8")
    parsed = extract_json(response)
    validate(parsed, len(frames))
    attach_global_frame_metadata(parsed, frames, records)
    parsed["metadata"] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(args.model_path.resolve()),
        "frame_dir": str(frame_dir),
        "timeline": str(timeline),
        "global_frame_index_policy": "all indices refer to the full RGB timeline",
        "sampled_prompt_indices": indices,
    }
    output.write_text(json.dumps(parsed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "events": len(parsed["events"]), "sampled_indices": indices}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
