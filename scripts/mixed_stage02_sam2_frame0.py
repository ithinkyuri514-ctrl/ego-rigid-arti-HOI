#!/usr/bin/env python3
"""Create global frame-0 SAM2 masks before required full-timeline propagation.

Run ``mixed_stage02_propagate_masks.py`` after frame-0 mask QC or manual correction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vlm_sam2_recon.rigid_pipeline.common import update_stage_state, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--sam2-root", type=Path, default=Path("/code/sam2"))
    parser.add_argument(
        "--sam2-checkpoint",
        type=Path,
        default=Path("/code/ArtHOI-4D-Reconstruction/third_party/sam2/checkpoints/sam2.1_hiera_large.pt"),
    )
    parser.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--multimask-output", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def box_pixels(box: list[float], width: int, height: int) -> np.ndarray:
    x0, y0, x1, y1 = [float(value) for value in box]
    return np.asarray(
        [x0 * width / 1000.0, y0 * height / 1000.0, x1 * width / 1000.0, y1 * height / 1000.0],
        dtype=np.float32,
    )


def clean_mask(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask, dtype=np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return binary > 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.flatnonzero(areas >= max(64, int(areas.max() * 0.01))) + 1
    return np.isin(labels, keep)


def overlay(rgb: np.ndarray, mask: np.ndarray, box: np.ndarray) -> Image.Image:
    canvas = rgb.astype(np.float32).copy()
    color = np.asarray([30, 144, 255], dtype=np.float32)
    canvas[mask] = 0.58 * canvas[mask] + 0.42 * color
    result = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(result)
    draw.rectangle(box.tolist(), outline=(255, 210, 0), width=5)
    return result


def main() -> int:
    args = parse_args()
    workspace = args.workspace.resolve()
    vlm_path = workspace / "outputs/01_vlm/mixed_interactions.json"
    frame_path = workspace / "outputs/00_rgb_frames/right_rgb_png/000000.png"
    if not vlm_path.is_file() or not frame_path.is_file():
        raise FileNotFoundError(f"Missing VLM/frame0 inputs: {vlm_path}, {frame_path}")
    data = json.loads(vlm_path.read_text(encoding="utf-8"))
    targets = data.get("vlm_result", {}).get("objects") or []
    if not targets:
        raise ValueError("VLM output has no manipulated objects")
    sys.path.insert(0, str(args.sam2_root.resolve()))
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model = build_sam2(args.sam2_config, str(args.sam2_checkpoint.resolve()), device=args.device)
    predictor = SAM2ImagePredictor(model)
    image = Image.open(frame_path).convert("RGB")
    rgb = np.asarray(image)
    height, width = rgb.shape[:2]
    predictor.set_image(rgb)
    output_root = workspace / "outputs/02_sam2_frame0_masks"
    records = []
    for target in targets:
        object_id = str(target["object_id"])
        normalized = target["global_frame0"]["bbox_2d_norm_1000"]
        box = box_pixels(normalized, width, height)
        with torch.inference_mode(), torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")
        ):
            masks, scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box,
                multimask_output=args.multimask_output,
                return_logits=False,
                normalize_coords=True,
            )
        index = int(np.argmax(scores))
        mask = clean_mask(np.asarray(masks[index], dtype=bool))
        object_dir = output_root / object_id
        combined = object_dir / "combined"
        overlays = object_dir / "overlays"
        combined.mkdir(parents=True, exist_ok=True)
        overlays.mkdir(parents=True, exist_ok=True)
        mask_path = combined / "000000.png"
        overlay_path = overlays / "000000.jpg"
        Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
        overlay(rgb, mask, box).save(overlay_path, quality=94)
        records.append(
            {
                "object_id": object_id,
                "name_en": target.get("name_en"),
                "name_zh": target.get("name_zh"),
                "object_class": target.get("object_class"),
                "frame_index": 0,
                "rgb": str(frame_path),
                "bbox_2d_norm_1000": normalized,
                "box_xyxy_pixels": box.tolist(),
                "sam2_score": float(scores[index]),
                "mask_pixels": int(mask.sum()),
                "mask_area_ratio": float(mask.mean()),
                "mask": str(mask_path),
                "overlay": str(overlay_path),
            }
        )
        print(f"{object_id}: score={scores[index]:.4f}, pixels={mask.sum()}", flush=True)
    summary = {
        "stage": "02_sam2_frame0_masks",
        "status": "completed",
        "camera": "right",
        "coordinate_frame": "global_frame0_right_image",
        "frame_index": 0,
        "source_vlm": str(vlm_path),
        "objects": records,
    }
    summary_path = output_root / "sam2_frame0_summary.json"
    write_json(summary_path, summary)
    update_stage_state(
        workspace / "pipeline_state.json",
        "02_sam2_frame0_masks",
        "completed",
        inputs=[str(vlm_path), str(frame_path)],
        outputs=[str(output_root)],
        notes=f"Generated automatic box-prompt SAM2 masks for {len(records)} manipulated objects on global right-eye frame 0.",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
