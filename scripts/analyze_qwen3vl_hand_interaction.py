#!/usr/bin/env python3
import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vlm_sam2_recon.stages.vlm_contact_semantics import (  # noqa: E402
    aggregate_contact_windows,
    build_rgbd_composite,
    colorize_depth,
    find_coarse_contact,
    load_export_rows,
    make_windows,
    match_depth_frame,
    parse_contact_response,
    project_depth_to_right_image,
    read_json,
)


DEFAULT_FRAME_DIR = "/code/3DVideo_2026-07-01-19-24-06-667_spatialmp4_export/rgb_right_png"
DEFAULT_MODEL_PATH = "/code/models/Qwen3-VL-8B-Instruct"


PROMPT = """You are a vision-language reconstruction target selector for egocentric 3D/4D reconstruction.

Task:
Analyze the provided ordered RGB frame sequence from the current camera view.
First list the visible objects in the video. Then select exactly two target objects for reconstruction:
1) smartphone / phone
2) laptop

Ignore boxes, cartons, packaging, bags, desk dividers, office background, and other context objects as reconstruction targets for now.
For each target object, choose exactly one best reconstruction keyframe with the least occlusion and clearest object geometry.
Also judge whether each target is a rigid object or an articulated object.

Important rules:
- Return strict JSON only. No markdown, no commentary.
- Do not hallucinate. If evidence is weak, use null or "unknown" and lower confidence.
- Focus on selecting reconstruction targets and keyframes, not on describing the whole room.
- The only reconstruction targets are phone and laptop. Do not include boxes as targets.
- You may list other visible objects in visible_objects, but they must have "target_for_reconstruction": false.
- Use frame indices from the input sequence, where the first frame is 0.
- Frame index and time are different: for a 5 fps sequence, frame 10 is about 2 seconds, frame 20 is about 4 seconds, and frame 30 is about 6 seconds.
- For keyframes, evidence_frames, and frame_range, use actual frame indices, not seconds.
- For first_contact_frame, use the first actual frame where the hand physically touches/presses/grabs the target object.
- Do not mark hovering, pointing, or being merely near the object as contact.
- For laptop contact, distinguish screen contact from base/keyboard contact when visible.
- For physical contact, identify only fingers with clear visual evidence. Allowed names are thumb, index, middle, ring, and pinky.
- primary_contact_finger must be one member of contact_fingers, or null when no finger is reliable.
- If labeled keyframes are provided, select keyframes from those labels unless there is a strong reason not to.
- Prefer a frame where the target object is visible, large enough, sharp, and minimally occluded by the hand or other objects.
- Do not choose a frame where the hand covers a large part of the target if a less occluded frame exists.
- Use normalized 2D boxes in [x1, y1, x2, y2] with coordinates from 0 to 1000 when visible; use null if uncertain.
- Rigid means the object shape has no moving joints during normal use, e.g. phone.
- Articulated means the object has movable parts connected by joints/hinges, e.g. laptop screen and base connected by a hinge.

JSON schema:
{
  "scene_summary": "short description of the desk scene and target objects",
  "visible_objects": [
    {
      "object_id": "vis_1",
      "name_zh": "Chinese object name",
      "name_en": "English canonical object name",
      "category": "device|container|furniture|cable|headphones|cup|bag|box|surface|other|unknown",
      "target_for_reconstruction": false,
      "reason_target_or_ignored": "short reason",
      "visible_frame_indices": [0],
      "confidence": 0.0
    }
  ],
  "target_objects": [
    {
      "object_id": "target_laptop",
      "name_zh": "笔记本电脑",
      "name_en": "laptop",
      "category": "device",
      "object_class_for_reconstruction": "rigid|articulated|unknown",
      "rigidity_reason": "explain hinge/joint or lack of moving parts",
      "selected_keyframe": {
        "frame_index": 0,
        "time_sec": 0.0,
        "bbox_2d_norm_1000": [x1, y1, x2, y2] | null,
        "occlusion_level": "none|low|medium|high|unknown",
        "view_quality": "excellent|good|fair|poor|unknown",
        "selection_reason": "why this frame is best for reconstruction"
      },
      "alternate_keyframes": [
        {
          "frame_index": 0,
          "time_sec": 0.0,
          "reason": "backup frame if useful"
        }
      ],
      "observed_state": "open|closed|partially_closed|placed_on_surface|held|unknown",
      "interaction_summary": "brief hand-object or state-change summary",
      "hand_object_interaction": {
        "first_contact_frame": {
          "frame_index": 0 | null,
          "time_sec": 0.0 | null,
          "hand_side": "left|right|both|unknown",
          "contacted_part": "screen|base|keyboard|trackpad|lid_edge|whole_object|unknown",
          "contact_type": "touch|press|push|grasp|release|none|unknown",
          "contact_fingers": ["thumb|index|middle|ring|pinky"],
          "primary_contact_finger": "thumb|index|middle|ring|pinky" | null,
          "evidence_frames": [0],
          "reason": "visual evidence for the first physical contact, or why contact is uncertain",
          "confidence": 0.0
        },
        "contact_summary": "short temporal summary of hand-object contact"
      },
      "relations_to_other_targets": [
        {
          "relation": "on|under|near|touching|supports|supported_by|unknown",
          "object_id": "target_phone|target_laptop",
          "evidence_frames": [0],
          "confidence": 0.0
        }
      ],
      "reconstruction_notes": {
        "recommended_mask": "target_laptop|target_phone",
        "parts_to_reconstruct": ["whole object"],
        "articulation_parts": ["base", "screen", "hinge"],
        "texture_or_geometry_notes": "short notes useful for reconstruction"
      },
      "confidence": 0.0
    }
  ],
  "selected_reconstruction_keyframes": [
    {
      "target_object_id": "target_laptop",
      "frame_index": 0,
      "time_sec": 0.0,
      "purpose": "reconstruct laptop",
      "why_low_occlusion": "short explanation",
      "confidence": 0.0
    },
    {
      "target_object_id": "target_phone",
      "frame_index": 0,
      "time_sec": 0.0,
      "purpose": "reconstruct phone",
      "why_low_occlusion": "short explanation",
      "confidence": 0.0
    }
  ],
  "ignored_objects": [
    {
      "name_zh": "纸箱",
      "name_en": "box",
      "reason": "context only; user requested not to reconstruct boxes"
    }
  ],
  "global_relations": [
    {
      "subject_id": "target_phone|target_laptop|hand|context",
      "relation": "on|near|touches|occludes|opens|closes|places|unknown",
      "object_id": "target_phone|target_laptop|context",
      "evidence_frames": [0],
      "confidence": 0.0
    }
  ],
  "hand_object_events": [
    {
      "event_type": "first_contact",
      "target_object_id": "target_laptop",
      "frame_index": 0 | null,
      "time_sec": 0.0 | null,
      "hand_side": "left|right|both|unknown",
      "contacted_part": "screen|base|keyboard|trackpad|lid_edge|whole_object|unknown",
      "contact_fingers": ["thumb|index|middle|ring|pinky"],
      "primary_contact_finger": "thumb|index|middle|ring|pinky" | null,
      "evidence_frames": [0],
      "confidence": 0.0,
      "reason": "why this is the first visible physical contact"
    }
  ],
  "reconstruction_plan": {
    "targets_in_order": ["target_laptop", "target_phone"],
    "exclude_from_reconstruction": ["boxes", "cartons", "packaging", "bags", "background"],
    "expected_object_motion": {
      "target_laptop": "articulated hinge motion if visible",
      "target_phone": "rigid body motion if visible"
    },
    "notes_for_downstream_pipeline": "short notes"
  },
  "uncertainty_notes": ["short uncertainty notes"]
}

The output must contain exactly two target_objects: one laptop and one phone.
"""


CONTACT_PROMPT = """You are analyzing first-person RGB-D frames of a hand manipulating an articulated laptop.

The image has synchronized columns. Each column contains one RGB frame on top and a depth image projected into the right RGB camera below. Blue depth is nearer and red depth is farther. Black depth pixels are missing measurements. A printed dt value is the nearest-depth timestamp offset; treat depth as weak evidence when that offset is large.

For every labeled frame, independently decide whether the left and right hand physically contact the target laptop. Hovering, pointing, a visible gap, touching a background object, or merely approaching the laptop is not contact. Use temporal consistency across the three neighboring frames, but do not copy one answer to every frame.

When contact is true:
- identify the contacted laptop part: screen, lid_edge, base, keyboard, trackpad, or unknown;
- list only visibly supported contact fingers from thumb, index, middle, ring, pinky;
- exclude ambiguous or occluded fingers;
- assign confidence in [0,1].

Return strict JSON only, without markdown or hidden reasoning:
{
  "contacts": [
    {
      "frame": 0,
      "r_contact": false,
      "l_contact": false,
      "r_fingers": [],
      "l_fingers": [],
      "contacted_part": "unknown",
      "confidence": 0.0
    }
  ]
}
Return exactly one entry for each requested frame id.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-dir", default=DEFAULT_FRAME_DIR)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", default=None)
    parser.add_argument("--raw-fps", type=float, default=5.0)
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--video-token-budget", type=int, default=12288)
    parser.add_argument("--frame-max-tokens", type=int, default=768)
    parser.add_argument("--frame-min-tokens", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=2600)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--no-flash-attn", action="store_true")
    parser.add_argument(
        "--keyframes",
        default=None,
        help="Comma-separated frame indices. If set, runs labeled multi-image inference instead of video inference.",
    )
    parser.add_argument(
        "--contact-pass",
        action="store_true",
        help="Run overlapping three-frame RGB-D contact/finger analysis instead of scene target selection.",
    )
    parser.add_argument(
        "--contact-source-json",
        type=Path,
        default=None,
        help="Existing coarse scene-analysis JSON used to center the fine contact search.",
    )
    parser.add_argument("--contact-target-id", default="target_laptop")
    parser.add_argument("--contact-hand-side", choices=["auto", "left", "right"], default="auto")
    parser.add_argument("--contact-start-frame", type=int, default=None)
    parser.add_argument("--contact-end-frame", type=int, default=None)
    parser.add_argument("--contact-search-before", type=int, default=8)
    parser.add_argument("--contact-search-after", type=int, default=18)
    parser.add_argument("--contact-window-size", type=int, default=3)
    parser.add_argument("--contact-window-stride", type=int, default=1)
    parser.add_argument("--contact-panel-width", type=int, default=448)
    parser.add_argument("--contact-max-windows", type=int, default=0, help="0 runs all windows; positive values are for debugging.")
    parser.add_argument("--export-root", type=Path, default=None, help="SpatialMP4 export containing manifest.json, frames.csv, and depth.")
    parser.add_argument("--depth-convention", choices=["camera_to_rig", "rig_to_camera", "direct_same_camera"], default="camera_to_rig")
    parser.add_argument("--depth-min-m", type=float, default=0.1)
    parser.add_argument("--depth-max-m", type=float, default=3.0)
    parser.add_argument("--depth-splat-radius-px", type=int, default=2)
    parser.add_argument("--rgb-time-offset-s", type=float, default=0.0)
    parser.add_argument(
        "--resume-contact-windows",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse valid per-window raw model responses when resuming an interrupted run.",
    )
    return parser.parse_args()


def extract_json(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise json.JSONDecodeError("No JSON object found", text, 0)


def clamp_frame(idx: int, frame_count: int) -> int:
    return max(0, min(frame_count - 1, idx))


def frame_range_from_time(time_range, fps: float, frame_count: int):
    if (
        not isinstance(time_range, list)
        or len(time_range) != 2
        or not all(isinstance(v, (int, float)) for v in time_range)
    ):
        return None
    start = clamp_frame(round(float(time_range[0]) * fps), frame_count)
    end = clamp_frame(round(float(time_range[1]) * fps), frame_count)
    if end < start:
        start, end = end, start
    return [start, end]


def representative_frames(frame_range, frame_count: int):
    if not frame_range:
        return []
    start, end = frame_range
    mid = clamp_frame(round((start + end) / 2), frame_count)
    return sorted({clamp_frame(start, frame_count), mid, clamp_frame(end, frame_count)})


def add_temporal_grounding(parsed, frames, fps: float):
    if not isinstance(parsed, dict):
        return None
    frame_count = len(frames)
    notes = []

    def enrich(item):
        if not isinstance(item, dict):
            return
        converted = frame_range_from_time(item.get("time_range_sec"), fps, frame_count)
        if converted is None:
            return
        reps = representative_frames(converted, frame_count)
        item["frame_range_estimated_from_time_sec"] = converted
        item["evidence_frame_files_estimated"] = [frames[i].name for i in reps]
        evidence = item.get("evidence_frames")
        if isinstance(evidence, list) and evidence and max(evidence) <= int((frame_count - 1) / fps):
            notes.append(
                "Some model-provided evidence_frames may be seconds rather than frame indices; "
                "prefer frame_range_estimated_from_time_sec for tracking."
            )

    def enrich_frame_ref(item):
        if not isinstance(item, dict):
            return
        if isinstance(item.get("frame_index"), int):
            idx = clamp_frame(item["frame_index"], frame_count)
            item["frame_index"] = idx
            item["frame_file"] = frames[idx].name
            item["frame_path"] = str(frames[idx])
        evidence = item.get("evidence_frames")
        if isinstance(evidence, list):
            fixed = []
            for value in evidence:
                if isinstance(value, int):
                    fixed.append(clamp_frame(value, frame_count))
            if fixed:
                item["evidence_frames"] = sorted(dict.fromkeys(fixed))
                item["evidence_frame_files"] = [frames[idx].name for idx in item["evidence_frames"]]

    for item in parsed.get("interacted_objects", []) or []:
        enrich(item)
    for item in parsed.get("temporal_events", []) or []:
        enrich(item)
        enrich_frame_ref(item)
    for item in parsed.get("hand_object_events", []) or []:
        enrich(item)
        enrich_frame_ref(item)

    for item in parsed.get("target_objects", []) or []:
        if not isinstance(item, dict):
            continue
        selected = item.get("selected_keyframe")
        if isinstance(selected, dict) and isinstance(selected.get("frame_index"), int):
            idx = clamp_frame(selected["frame_index"], frame_count)
            selected["frame_index"] = idx
            selected["frame_file"] = frames[idx].name
            selected["frame_path"] = str(frames[idx])
        alternates = item.get("alternate_keyframes")
        if isinstance(alternates, list):
            for alt in alternates:
                if isinstance(alt, dict) and isinstance(alt.get("frame_index"), int):
                    idx = clamp_frame(alt["frame_index"], frame_count)
                    alt["frame_index"] = idx
                    alt["frame_file"] = frames[idx].name
                    alt["frame_path"] = str(frames[idx])
        interaction = item.get("hand_object_interaction")
        if isinstance(interaction, dict):
            for key in ("first_contact_frame", "first_touch_frame", "contact_start_frame"):
                enrich_frame_ref(interaction.get(key))

    for item in parsed.get("selected_reconstruction_keyframes", []) or []:
        if isinstance(item, dict) and isinstance(item.get("frame_index"), int):
            idx = clamp_frame(item["frame_index"], frame_count)
            item["frame_index"] = idx
            item["frame_file"] = frames[idx].name
            item["frame_path"] = str(frames[idx])

    if notes:
        parsed.setdefault("postprocess_notes", [])
        for note in sorted(set(notes)):
            if note not in parsed["postprocess_notes"]:
                parsed["postprocess_notes"].append(note)
    return {
        "fps": fps,
        "frame_count": frame_count,
        "frame_index_rule": "frame_index = round(time_sec * fps), clamped to available frames",
    }


def parse_keyframes(value: str | None, frame_count: int):
    if not value:
        return None
    indices = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        idx = int(chunk)
        if idx < 0 or idx >= frame_count:
            raise ValueError(f"Keyframe index out of range: {idx}; valid range is 0..{frame_count - 1}")
        indices.append(idx)
    if not indices:
        raise ValueError("--keyframes did not contain any valid frame indices")
    return sorted(dict.fromkeys(indices))


def load_qwen(model_path: Path, no_flash_attn: bool):
    print(f"Loading model: {model_path}")
    load_kwargs = {"dtype": torch.bfloat16, "device_map": "auto"}
    if not no_flash_attn:
        load_kwargs["attn_implementation"] = "flash_attention_2"
    model = AutoModelForImageTextToText.from_pretrained(str(model_path), **load_kwargs)
    model.eval()
    return model, AutoProcessor.from_pretrained(str(model_path))


def generate_single_image_response(model, processor, image_path: Path, prompt: str, args: argparse.Namespace) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"file://{image_path}",
                    "min_pixels": args.frame_min_tokens * 32 * 32,
                    "max_pixels": args.frame_max_tokens * 32 * 32,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages, image_patch_size=16)
    inputs = processor(text=text, images=images, videos=videos, return_tensors="pt", do_resize=False)
    inputs = inputs.to(model.device)
    generation_kwargs = {
        "max_new_tokens": min(args.max_new_tokens, 1200),
        "do_sample": args.temperature > 0,
    }
    if args.temperature > 0:
        generation_kwargs["temperature"] = args.temperature
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **generation_kwargs)
    generated_trimmed = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    del inputs, generated_ids, generated_trimmed
    return response


def run_contact_pass(args: argparse.Namespace, frames: list[Path], model_path: Path) -> None:
    if args.contact_source_json is None:
        raise ValueError("--contact-pass requires --contact-source-json with the coarse first-contact result")
    if args.export_root is None:
        raise ValueError("--contact-pass requires --export-root for calibrated depth projection")
    source_path = args.contact_source_json.expanduser().resolve()
    export_root = args.export_root.expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Contact source JSON not found: {source_path}")
    if not export_root.is_dir():
        raise FileNotFoundError(f"SpatialMP4 export not found: {export_root}")

    source = read_json(source_path)
    coarse = find_coarse_contact(source, args.contact_target_id)
    coarse_frame = int(coarse["frame_index"])
    start = args.contact_start_frame
    end = args.contact_end_frame
    if start is None:
        start = coarse_frame - max(0, args.contact_search_before)
    if end is None:
        end = coarse_frame + max(0, args.contact_search_after)
    start = max(0, int(start))
    end = min(len(frames) - 1, int(end))
    if end < start:
        raise ValueError(f"Empty fine-contact search range: {start}..{end}")
    windows = make_windows(start, end, args.contact_window_size, args.contact_window_stride)
    if args.contact_max_windows > 0:
        windows = windows[: args.contact_max_windows]

    preferred_hand = None if args.contact_hand_side == "auto" else args.contact_hand_side
    if preferred_hand is None and coarse.get("hand_side") in ("left", "right"):
        preferred_hand = coarse["hand_side"]
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else frames[0].parent.parent / "qwen3vl_contact_fingers_rgbd.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    window_dir = output.parent / f"{output.stem}_windows"
    window_dir.mkdir(parents=True, exist_ok=True)

    meta = read_json(export_root / "manifest.json")
    rows = load_export_rows(export_root)
    depth_cache: dict[int, object] = {}
    model = processor = None
    all_contacts: list[list[dict]] = []
    window_records = []
    print(f"Fine contact range {start}..{end}: {len(windows)} overlapping windows")
    for window_index, frame_ids in enumerate(windows):
        stem = f"window_{frame_ids[0]:06d}_{frame_ids[-1]:06d}"
        composite_path = window_dir / f"{stem}.jpg"
        raw_path = window_dir / f"{stem}.raw.txt"
        parsed_contacts = None
        if args.resume_contact_windows and raw_path.exists():
            try:
                parsed_contacts = parse_contact_response(extract_json(raw_path.read_text(encoding="utf-8")), frame_ids)
                if {item["frame"] for item in parsed_contacts} != set(frame_ids):
                    parsed_contacts = None
            except (ValueError, json.JSONDecodeError):
                parsed_contacts = None

        matches = [
            match_depth_frame(frame, args.raw_fps, export_root, rows, args.rgb_time_offset_s)
            for frame in frame_ids
        ]
        if parsed_contacts is None:
            depth_images = []
            for match in matches:
                if match.frame_index not in depth_cache:
                    depth = np.load(match.depth_path).astype(np.float32)
                    aligned = project_depth_to_right_image(
                        meta,
                        depth,
                        convention=args.depth_convention,
                        depth_min_m=args.depth_min_m,
                        depth_max_m=args.depth_max_m,
                        splat_radius_px=args.depth_splat_radius_px,
                    )
                    depth_cache[match.frame_index] = colorize_depth(aligned, args.depth_min_m, args.depth_max_m)
                depth_images.append(depth_cache[match.frame_index])
            composite = build_rgbd_composite(
                [frames[frame] for frame in frame_ids],
                depth_images,
                matches,
                panel_width=args.contact_panel_width,
            )
            composite.save(composite_path, quality=92)
            if model is None or processor is None:
                model, processor = load_qwen(model_path, args.no_flash_attn)
            requested = ", ".join(str(frame) for frame in frame_ids)
            prompt = (
                CONTACT_PROMPT
                + f"\nTarget object id: {args.contact_target_id}. Requested frame ids: [{requested}]."
                + (f" The coarse scene pass identified the {preferred_hand} hand." if preferred_hand else "")
            )
            print(f"[{window_index + 1}/{len(windows)}] Qwen contact inference: {frame_ids}")
            response = generate_single_image_response(model, processor, composite_path, prompt, args)
            raw_path.write_text(response + "\n", encoding="utf-8")
            parsed_contacts = parse_contact_response(extract_json(response), frame_ids)
            if {item["frame"] for item in parsed_contacts} != set(frame_ids):
                raise ValueError(f"Model response for {frame_ids} omitted or changed requested frame ids: {response}")
        else:
            print(f"[{window_index + 1}/{len(windows)}] Reusing contact response: {raw_path.name}")

        all_contacts.append(parsed_contacts)
        window_records.append(
            {
                "window_index": window_index,
                "frame_indices": frame_ids,
                "composite_path": str(composite_path),
                "raw_response_path": str(raw_path),
                "depth_matches": [
                    {
                        "rgb_frame": frame,
                        "depth_frame": match.frame_index,
                        "target_timestamp_s": match.target_timestamp_s,
                        "depth_timestamp_s": match.depth_timestamp_s,
                        "delta_s": match.delta_s,
                        "depth_path": str(match.depth_path),
                    }
                    for frame, match in zip(frame_ids, matches)
                ],
                "contacts": parsed_contacts,
            }
        )

    contact_analysis = aggregate_contact_windows(
        all_contacts,
        target_id=args.contact_target_id,
        fps=args.raw_fps,
        preferred_hand=preferred_hand,
    )
    max_abs_depth_delta = max(
        abs(match["delta_s"])
        for window in window_records
        for match in window["depth_matches"]
    )
    result = {
        "metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "analysis_mode": "overlapping_rgbd_contact_fingers",
            "model_path": str(model_path),
            "frame_dir": str(frames[0].parent),
            "frame_count": len(frames),
            "raw_fps": args.raw_fps,
            "export_root": str(export_root),
            "source_coarse_json": str(source_path),
            "coarse_contact": coarse,
            "search_frame_range": [start, end],
            "window_size": args.contact_window_size,
            "window_stride": args.contact_window_stride,
            "depth_convention": args.depth_convention,
            "rgb_time_offset_s": args.rgb_time_offset_s,
            "max_abs_depth_timestamp_delta_s": max_abs_depth_delta,
            "depth_is_auxiliary": True,
            "window_dir": str(window_dir),
        },
        "vlm_result": {"contact_analysis": contact_analysis},
        "windows": window_records,
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    first = contact_analysis["first_contact_frame"]
    print(f"Wrote fine contact JSON: {output}")
    print(
        "First strict-majority contact: "
        f"frame={first['frame_index']} hand={first['hand_side']} "
        f"fingers={first['contact_fingers']} confidence={first['confidence']:.3f}"
    )


def main() -> None:
    args = parse_args()
    frame_dir = Path(args.frame_dir).resolve()
    model_path = Path(args.model_path).resolve()
    if not frame_dir.is_dir():
        raise FileNotFoundError(f"Frame directory not found: {frame_dir}")
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    frames = sorted(frame_dir.glob("*.png"))
    if not frames:
        raise FileNotFoundError(f"No PNG frames found in {frame_dir}")
    if args.contact_pass:
        run_contact_pass(args, frames, model_path)
        return
    keyframe_indices = parse_keyframes(args.keyframes, len(frames))

    if args.output:
        output = Path(args.output).resolve()
    elif keyframe_indices is not None:
        output = frame_dir.parent / "qwen3vl8b_reconstruction_targets_right_keyframes.json"
    else:
        output = frame_dir.parent / "qwen3vl8b_reconstruction_targets_right.json"
    raw_output = output.with_suffix(".raw.txt")

    if keyframe_indices is None:
        video_item = {
            "type": "video",
            "video": [f"file://{frame}" for frame in frames],
            "raw_fps": args.raw_fps,
            "sample_fps": args.sample_fps,
            "min_pixels": args.frame_min_tokens * 32 * 32,
            "max_pixels": args.frame_max_tokens * 32 * 32,
            "total_pixels": args.video_token_budget * 32 * 32 * 2,
        }
        content = [
            video_item,
            {
                "type": "text",
                "text": (
                    PROMPT
                    + f"\nInput has {len(frames)} extracted RGB frames at {args.raw_fps} fps. "
                    + "Frame filenames are ordered numerically from the directory."
                ),
            },
        ]
    else:
        labels = ", ".join(f"{idx}=t{idx / args.raw_fps:.1f}s" for idx in keyframe_indices)
        content = [
            {
                "type": "text",
                "text": (
                    PROMPT
                    + f"\nInput is a set of labeled keyframes sampled from {len(frames)} total frames at {args.raw_fps} fps. "
                    + f"Only use evidence frame indices from these labels when possible: {labels}. "
                    + "Infer coarse intervals between labeled frames if an action clearly spans them."
                ),
            }
        ]
        for idx in keyframe_indices:
            content.extend(
                [
                    {"type": "text", "text": f"Frame {idx} (t={idx / args.raw_fps:.1f}s):"},
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
    load_kwargs = {
        "dtype": torch.bfloat16,
        "device_map": "auto",
    }
    if not args.no_flash_attn:
        load_kwargs["attn_implementation"] = "flash_attention_2"
    model = AutoModelForImageTextToText.from_pretrained(str(model_path), **load_kwargs)
    model.eval()
    processor = AutoProcessor.from_pretrained(str(model_path))

    if keyframe_indices is None:
        print(f"Preparing {len(frames)} frames from: {frame_dir}")
    else:
        print(f"Preparing labeled keyframes {keyframe_indices} from: {frame_dir}")
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if keyframe_indices is None:
        images, videos, video_kwargs = process_vision_info(
            messages,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        if videos is not None:
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)
        else:
            video_metadatas = None
        inputs = processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=video_metadatas,
            return_tensors="pt",
            do_resize=False,
            **video_kwargs,
        )
    else:
        images, videos = process_vision_info(messages, image_patch_size=16)
        inputs = processor(
            text=text,
            images=images,
            videos=videos,
            return_tensors="pt",
            do_resize=False,
        )
    inputs = inputs.to(model.device)

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
    }
    if args.temperature > 0:
        generation_kwargs["temperature"] = args.temperature

    print("Running Qwen3-VL inference...")
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **generation_kwargs)
    generated_trimmed = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    raw_output.write_text(response + "\n", encoding="utf-8")

    parsed = None
    parse_error = None
    try:
        parsed = extract_json(response)
    except json.JSONDecodeError as exc:
        parse_error = str(exc)
    temporal_grounding = add_temporal_grounding(parsed, frames, args.raw_fps) if parsed is not None else None

    result = {
        "metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_path": str(model_path),
            "frame_dir": str(frame_dir),
            "frame_count": len(frames),
            "raw_fps": args.raw_fps,
            "sample_fps": args.sample_fps,
            "duration_sec": len(frames) / args.raw_fps,
            "frame_files": [frame.name for frame in frames],
            "keyframe_indices": keyframe_indices,
            "keyframe_files": [frames[idx].name for idx in keyframe_indices] if keyframe_indices else None,
            "video_token_budget": args.video_token_budget,
            "frame_min_tokens": args.frame_min_tokens,
            "frame_max_tokens": args.frame_max_tokens,
            "flash_attention_2": not args.no_flash_attn,
            "raw_response_path": str(raw_output),
        },
        "vlm_result": parsed,
    }
    if temporal_grounding is not None:
        result["metadata"]["postprocess_temporal_grounding"] = temporal_grounding
    if parse_error is not None:
        result["parse_error"] = parse_error
        result["raw_model_output"] = response

    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote JSON: {output}")
    print(f"Wrote raw response: {raw_output}")


if __name__ == "__main__":
    main()
