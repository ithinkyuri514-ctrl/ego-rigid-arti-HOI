#!/usr/bin/env python3
"""Use local Qwen3-VL to select one unambiguous stable bottle-grasp frame."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINGERS = ("thumb", "index", "middle", "ring", "pinky")


PROMPT = """You are proposing per-finger hand-object contact states for metric 3D optimization.

The image is a temporally ordered grid of candidate first-person frames. Each cell has annotated RGB on top and calibrated metric depth below. Hand-mask boundaries are blue and projected fingertip labels are T/I/M/R/P.

The ONLY target object is the small upright cylindrical BOTTLE enclosed by the GREEN object-mask contour. The green contour is the authoritative target identifier, including while the bottle is near or inside another object and while part of its contour is occluded. Analyze contact only between the hand/fingers and this green-contour bottle. The microwave, microwave door and interior, control panel, countertop, and every other scene object are background context: ignore them completely and never describe any of them as the contact target. Do not switch targets when the bottle approaches the microwave.

The word bottle is provided only to disambiguate the green mask; it is not evidence that contact occurs. Do not assume that a nearby hand is touching the bottle, do not infer contact from the expected action or object category, and do not assume front-side or back-side contact. A fully occluded finger may be UNKNOWN rather than contact.

Infer zero or more continuous FIRM-CONTACT segments. For each segment, independently assign a contact probability in [0,1] to thumb, index, middle, ring, and pinky. Use temporal evidence: disappearance of a visible gap, persistent proximity, occlusion ordering, and hand/object co-motion. Do not infer contact from semantics alone. Also choose one anchor frame with the clearest geometry from the strongest segment. If there is no reliable firm contact, use an empty contact_segments list and anchor_frame -1.

For the anchor frame, provide a weak side hypothesis for each finger with probabilities for front, back, and uncertain. This is only a visual proposal; downstream metric depth will make the final side decision.

Return one strict JSON object with exactly these keys:
- anchor_frame: integer;
- contact_segments: list of objects with start_frame, end_frame, and finger_probabilities;
- side_hypotheses: object keyed by thumb/index/middle/ring/pinky, each containing front/back/uncertain probabilities;
- confidence: number;
- reason: specific temporal, boundary, occlusion, or depth evidence.

The reason must explicitly identify the green-contour bottle and cite geometric or temporal evidence such as fingertip-to-contour gap, overlap/occlusion ordering, calibrated depth, or hand/bottle co-motion. Do not mention or reason about background scene objects. Do not output markdown. Be conservative and use low probabilities or uncertain when evidence is hidden or ambiguous.
"""


def font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu//DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def colorize_depth(depth: np.ndarray) -> Image.Image:
    valid = np.isfinite(depth) & (depth > 0.1) & (depth < 3.0)
    image = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if valid.any():
        lo, hi = np.quantile(depth[valid], [0.02, 0.98])
        t = np.clip((depth - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
        image[..., 0] = np.clip(255.0 * (2.0 * t - 0.25), 0, 255).astype(np.uint8)
        image[..., 1] = np.clip(255.0 * (1.0 - np.abs(2.0 * t - 1.0)), 0, 255).astype(np.uint8)
        image[..., 2] = np.clip(255.0 * (1.25 - 2.0 * t), 0, 255).astype(np.uint8)
        image[~valid] = 0
    return Image.fromarray(image)


def resolve_bottle_mask_dir(workspace: Path) -> Path:
    candidates = (
        workspace / "outputs/04_object_masks/combined",
        workspace / "outputs/04_object_masks/bottle/objects/bottle",
        workspace / "outputs/02_sam2_frame0_masks/propagated/objects/bottle",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "Bottle mask directory is missing; checked: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def fingertips_for_frame(workspace: Path, frame: int) -> np.ndarray | None:
    manifest_path = workspace / "outputs/09_egoforce/dynamic_manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = manifest["frames"][frame]
    raw_path = record.get("raw_Ct_npz")
    if not raw_path:
        return None
    selected_side_index = int(record.get("selected_raw_side_index", 1))
    if selected_side_index not in (0, 1):
        return None
    with np.load(raw_path) as raw:
        keypoints = raw["egoforce_hand_keypoints_2d"][selected_side_index]
    return np.asarray(keypoints[[4, 8, 12, 16, 20]], dtype=np.float32)


def annotated_rgb(workspace: Path, object_mask_dir: Path, frame: int) -> Image.Image:
    image = np.asarray(
        Image.open(workspace / f"outputs/00_rgb_frames/right_rgb_png/{frame:06d}.png").convert("RGB")
    ).copy()
    hand = np.asarray(
        Image.open(workspace / f"outputs/02_hand_masks/combined/{frame:06d}.png").convert("L")
    ) > 127
    obj = np.asarray(
        Image.open(object_mask_dir / f"{frame:06d}.png").convert("L")
    ) > 127
    hand_dilated = np.asarray(
        Image.fromarray(hand.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(7))
    ) > 0
    object_dilated = np.asarray(
        Image.fromarray(obj.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(7))
    ) > 0
    hand_edge = hand_dilated ^ hand
    object_edge = object_dilated ^ obj
    image[hand_edge] = (40, 135, 255)
    image[object_edge] = (40, 235, 90)
    result = Image.fromarray(image)
    draw = ImageDraw.Draw(result)
    tips = fingertips_for_frame(workspace, frame)
    if tips is not None:
        for label, (x, y) in zip(("T", "I", "M", "R", "P"), tips):
            if not np.isfinite([x, y]).all():
                continue
            radius = 18
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(0, 0, 0), outline=(255, 255, 255), width=3)
            draw.text((x - 8, y - 14), label, font=font(24), fill=(255, 255, 255))
    return result


def make_grid(workspace: Path, frames: list[int], output: Path) -> None:
    width, half_height = 320, 192
    columns = 4
    rows = int(np.ceil(len(frames) / columns))
    cells = []
    label_font = font(24)
    object_mask_dir = resolve_bottle_mask_dir(workspace)
    for frame in frames:
        rgb = annotated_rgb(workspace, object_mask_dir, frame).resize(
            (width, half_height), Image.Resampling.LANCZOS
        )
        depth = np.load(
            workspace / f"outputs/06_dense_depth/metric_depth_npy/{frame:06d}.npy"
        )
        depth_image = colorize_depth(depth).resize((width, half_height), Image.Resampling.BILINEAR)
        cell = Image.new("RGB", (width, half_height * 2), (0, 0, 0))
        cell.paste(rgb, (0, 0))
        cell.paste(depth_image, (0, half_height))
        draw = ImageDraw.Draw(cell)
        draw.rectangle((6, 6, 142, 42), fill=(0, 0, 0))
        draw.text((12, 7), f"frame {frame}", font=label_font, fill=(255, 255, 255))
        cells.append(cell)
    grid = Image.new("RGB", (width * columns, half_height * 2 * rows), (0, 0, 0))
    for index, cell in enumerate(cells):
        grid.paste(cell, ((index % columns) * width, (index // columns) * half_height * 2))
    grid.save(output, quality=94)


def extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError(f"Qwen returned no JSON: {text}")
    return json.loads(match.group(0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=PROJECT_ROOT / "run_rigid_20260715_215524",
    )
    parser.add_argument("--model", type=Path, default=Path("/code/models/Qwen3-VL-8B-Instruct"))
    parser.add_argument(
        "--candidates",
        default="2,5,8,11,14,17,20,23,26,29,32,35,38,42,46,50",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = args.workspace.resolve()
    output = workspace / "outputs/11_contact_optimization"
    output.mkdir(parents=True, exist_ok=True)
    frames = [int(value) for value in args.candidates.split(",")]
    grid_path = output / "vlm_contact_anchor_candidates.jpg"
    make_grid(workspace, frames, grid_path)
    model = AutoModelForImageTextToText.from_pretrained(
        str(args.model),
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    ).eval()
    processor = AutoProcessor.from_pretrained(str(args.model))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{grid_path}", "min_pixels": 1024 * 32 * 32, "max_pixels": 2304 * 32 * 32},
                {"type": "text", "text": PROMPT + f"\nAllowed anchor frames: {frames}."},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages, image_patch_size=16)
    inputs = processor(text=text, images=images, videos=videos, return_tensors="pt", do_resize=False).to(model.device)
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=500, do_sample=False)
    trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated)]
    response = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
    (output / "vlm_contact_anchor.raw.txt").write_text(response + "\n", encoding="utf-8")
    result = extract_json(response)
    anchor = int(result["anchor_frame"])
    segments = result.get("contact_segments", [])
    if not segments or anchor < 0:
        raise ValueError(f"Qwen found no reliable contact segment: {result}")
    containing = [
        segment
        for segment in segments
        if int(segment["start_frame"]) <= anchor <= int(segment["end_frame"])
    ]
    if anchor not in frames or not containing:
        raise ValueError(f"Invalid anchor result: {result}")
    confidence = float(result.get("confidence", 0.0))
    reason = str(result.get("reason", "")).strip()
    if confidence < 0.8 or len(reason) < 40 or "short evidence" in reason.lower():
        raise ValueError(f"Qwen did not provide an absolutely reliable anchor: {result}")
    normalized_reason = reason.lower().replace("-", " ").replace("_", " ")
    target_is_explicit = (
        "green" in normalized_reason
        and "bottle" in normalized_reason
        and any(
            marker in normalized_reason
            for marker in ("contour", "outline", "boundary", "mask")
        )
    )
    forbidden_scene_terms = (
        "microwave",
        "oven",
        "appliance",
        "door",
        "interior",
        "control panel",
        "countertop",
        "background",
    )
    geometry_evidence_terms = (
        "fingertip",
        "finger",
        "gap",
        "proximity",
        "overlap",
        "occlusion",
        "depth",
        "co motion",
        "separation",
    )
    if (
        not target_is_explicit
        or any(term in normalized_reason for term in forbidden_scene_terms)
        or not any(term in normalized_reason for term in geometry_evidence_terms)
    ):
        raise ValueError(
            "Qwen did not ground contact exclusively on the green-contour bottle: "
            f"{result}"
        )
    strongest = containing[0]
    probabilities = {
        finger: float(strongest.get("finger_probabilities", {}).get(finger, 0.0))
        for finger in FINGERS
    }
    result["contact_interval"] = [
        int(strongest["start_frame"]),
        int(strongest["end_frame"]),
    ]
    result["finger_contact_probabilities"] = probabilities
    result["certain_contact_fingers"] = [
        finger for finger, probability in probabilities.items() if probability >= 0.65
    ]
    result["model_path"] = str(args.model)
    result["candidate_frames"] = frames
    result["candidate_grid"] = str(grid_path)
    (output / "vlm_contact_anchor.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
