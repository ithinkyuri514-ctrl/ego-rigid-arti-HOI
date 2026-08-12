"""SAM2 video prompting, bidirectional propagation, and artifact export."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .common import write_json


DEFAULT_SAM2_ROOT = Path("/code/sam2")
DEFAULT_SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEFAULT_SAM2_CHECKPOINT = Path(
    "/code/ArtHOI-4D-Reconstruction/third_party/sam2/checkpoints/sam2.1_hiera_large.pt"
)


def build_video_predictor(
    sam2_root: Path,
    config: str,
    checkpoint: Path,
    device: str,
    *,
    vos_optimized: bool = False,
):
    root = sam2_root.resolve()
    checkpoint = checkpoint.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"SAM2 root not found: {root}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from sam2.build_sam import build_sam2_video_predictor

    return build_sam2_video_predictor(
        config,
        str(checkpoint),
        device=device,
        vos_optimized=vos_optimized,
    )


def add_points_or_box(
    predictor,
    inference_state,
    *,
    frame_index: int,
    object_id: str,
    positive_points: list[list[float]] | None = None,
    negative_points: list[list[float]] | None = None,
    box_xyxy: list[float] | None = None,
) -> dict[str, np.ndarray]:
    positive = positive_points or []
    negative = negative_points or []
    if not positive and not negative and box_xyxy is None:
        raise ValueError("At least one positive/negative point or box is required")
    coords = np.asarray(positive + negative, dtype=np.float32).reshape(-1, 2)
    labels = np.asarray([1] * len(positive) + [0] * len(negative), dtype=np.int32)
    _, object_ids, logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=int(frame_index),
        obj_id=str(object_id),
        points=coords if len(coords) else None,
        labels=labels if len(labels) else None,
        box=np.asarray(box_xyxy, dtype=np.float32) if box_xyxy is not None else None,
        clear_old_points=True,
        normalize_coords=True,
    )
    return logits_to_masks(object_ids, logits)


def add_mask(
    predictor,
    inference_state,
    *,
    frame_index: int,
    object_id: str,
    mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Condition the video predictor with a full-resolution binary mask."""
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {binary.shape}")
    if not binary.any():
        raise ValueError(f"Conditioning mask for {object_id!r} is empty")
    _, object_ids, logits = predictor.add_new_mask(
        inference_state=inference_state,
        frame_idx=int(frame_index),
        obj_id=str(object_id),
        mask=binary,
    )
    return logits_to_masks(object_ids, logits)


def logits_to_masks(object_ids, logits) -> dict[str, np.ndarray]:
    values = logits.detach().float().cpu().numpy()
    masks: dict[str, np.ndarray] = {}
    for index, object_id in enumerate(object_ids):
        mask = values[index]
        if mask.ndim == 3:
            mask = mask[0]
        masks[str(object_id)] = np.asarray(mask > 0.0, dtype=bool)
    return masks


def propagate_bidirectional(
    predictor,
    inference_state,
    conditioning_frames: list[int],
) -> dict[int, dict[str, np.ndarray]]:
    if not conditioning_frames:
        raise ValueError("No conditioning frames were provided")
    num_frames = int(inference_state["num_frames"])
    first = min(conditioning_frames)
    last = max(conditioning_frames)
    results: dict[int, dict[str, np.ndarray]] = {}

    for frame_index, object_ids, logits in predictor.propagate_in_video(
        inference_state,
        start_frame_idx=first,
        max_frame_num_to_track=num_frames,
        reverse=False,
    ):
        results[int(frame_index)] = logits_to_masks(object_ids, logits)

    if last > 0:
        for frame_index, object_ids, logits in predictor.propagate_in_video(
            inference_state,
            start_frame_idx=last,
            max_frame_num_to_track=num_frames,
            reverse=True,
        ):
            results[int(frame_index)] = logits_to_masks(object_ids, logits)

    missing = sorted(set(range(num_frames)) - set(results))
    if missing:
        raise RuntimeError(f"SAM2 propagation did not return all frames: {missing}")
    return results


def list_display_frames(frame_dir: Path) -> list[Path]:
    extensions = {".png", ".jpg", ".jpeg"}
    frames = [path for path in frame_dir.iterdir() if path.suffix.lower() in extensions]
    frames.sort(key=lambda path: int(path.stem))
    if not frames:
        raise FileNotFoundError(f"No display frames found: {frame_dir}")
    return frames


def overlay_mask(image: Image.Image, mask: np.ndarray, color=(225, 55, 50), alpha=0.46) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != rgb.shape[:2]:
        raise ValueError(f"Mask/image mismatch: mask={mask.shape} image={rgb.shape[:2]}")
    tint = np.asarray(color, dtype=np.float32)
    rgb[mask] = ((1.0 - alpha) * rgb[mask] + alpha * tint).astype(np.uint8)
    return Image.fromarray(rgb)


def save_propagation_outputs(
    results: dict[int, dict[str, np.ndarray]],
    display_frames: list[Path],
    output_dir: Path,
    *,
    fps: float,
    video_stem: str,
    save_overlays: bool = True,
) -> dict[str, Any]:
    if len(results) != len(display_frames):
        raise ValueError(f"Result/frame mismatch: masks={len(results)} frames={len(display_frames)}")
    if output_dir.exists():
        for child in (output_dir / "objects", output_dir / "combined", output_dir / "overlays"):
            if child.exists():
                shutil.rmtree(child)
    object_ids = sorted({object_id for frame in results.values() for object_id in frame})
    combined_dir = output_dir / "combined"
    overlay_dir = output_dir / "overlays"
    combined_dir.mkdir(parents=True, exist_ok=True)
    if save_overlays:
        overlay_dir.mkdir(parents=True, exist_ok=True)
    object_dirs = {}
    for object_id in object_ids:
        path = output_dir / "objects" / object_id
        path.mkdir(parents=True, exist_ok=True)
        object_dirs[object_id] = path

    frame_records = []
    for frame_index in range(len(display_frames)):
        masks = results[frame_index]
        shape = next(iter(masks.values())).shape
        combined = np.zeros(shape, dtype=bool)
        per_object = {}
        for object_id in object_ids:
            mask = np.asarray(masks.get(object_id, np.zeros(shape, dtype=bool)), dtype=bool)
            combined |= mask
            mask_path = object_dirs[object_id] / f"{frame_index:06d}.png"
            Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
            per_object[object_id] = {
                "mask": str(mask_path),
                "area_pixels": int(mask.sum()),
            }
        combined_path = combined_dir / f"{frame_index:06d}.png"
        Image.fromarray(combined.astype(np.uint8) * 255).save(combined_path)
        overlay_path = None
        if save_overlays:
            overlay_path = overlay_dir / f"{frame_index:06d}.jpg"
            overlay_mask(Image.open(display_frames[frame_index]), combined).save(
                overlay_path,
                quality=92,
            )
        frame_records.append(
            {
                "frame_index": frame_index,
                "source_frame": str(display_frames[frame_index]),
                "combined_mask": str(combined_path),
                "combined_area_pixels": int(combined.sum()),
                "overlay": str(overlay_path) if overlay_path else None,
                "objects": per_object,
            }
        )

    mask_video = output_dir / f"{video_stem}.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(combined_dir / "%06d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "0",
            str(mask_video),
        ],
        check=True,
    )
    summary = {
        "frame_count": len(display_frames),
        "fps": fps,
        "object_ids": object_ids,
        "mask_video": str(mask_video),
        "combined_mask_dir": str(combined_dir),
        "object_mask_dirs": {key: str(value) for key, value in object_dirs.items()},
        "overlay_dir": str(overlay_dir) if save_overlays else None,
        "frames": frame_records,
    }
    write_json(output_dir / "propagation_manifest.json", summary)
    return summary


def mask_sequence_qc(
    summary: dict[str, Any],
    *,
    max_area_ratio: float = 3.5,
    allow_empty: bool = False,
) -> dict[str, Any]:
    areas = np.asarray([int(item["combined_area_pixels"]) for item in summary["frames"]], dtype=np.float64)
    empty = np.flatnonzero(areas <= 0).tolist()
    adjacent_nonempty = (areas[1:] > 0) & (areas[:-1] > 0)
    ratios = np.ones(max(len(areas) - 1, 0), dtype=np.float64)
    ratios[adjacent_nonempty] = areas[1:][adjacent_nonempty] / areas[:-1][adjacent_nonempty]
    symmetric_ratios = np.maximum(ratios, 1.0 / np.maximum(ratios, 1e-12))
    jumps = (np.flatnonzero(adjacent_nonempty & (symmetric_ratios > max_area_ratio)) + 1).tolist()
    passed = (allow_empty or not empty) and not jumps
    return {
        "passed": passed,
        "empty_frames": empty,
        "abrupt_area_change_frames": jumps,
        "max_symmetric_area_ratio": float(symmetric_ratios.max()) if symmetric_ratios.size else 1.0,
        "area_pixels_min": int(areas.min()) if areas.size else 0,
        "area_pixels_median": float(np.median(areas)) if areas.size else 0.0,
        "area_pixels_max": int(areas.max()) if areas.size else 0,
        "threshold_max_symmetric_area_ratio": max_area_ratio,
        "empty_frames_allowed": allow_empty,
    }


def normalized_box_to_pixels(box: list[float], width: int, height: int) -> list[float]:
    if len(box) != 4:
        raise ValueError(f"Expected bbox [x0,y0,x1,y1], got {box}")
    x0, y0, x1, y1 = [float(item) for item in box]
    result = [x0 / 1000.0 * width, y0 / 1000.0 * height, x1 / 1000.0 * width, y1 / 1000.0 * height]
    result[0], result[2] = sorted((max(0.0, result[0]), min(width - 1.0, result[2])))
    result[1], result[3] = sorted((max(0.0, result[1]), min(height - 1.0, result[3])))
    return result


def load_vlm_target(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    result = data.get("vlm_result") or {}
    target = result.get("target_object")
    if target is None:
        targets = result.get("target_objects") or []
        target = targets[0] if targets else None
    if not isinstance(target, dict):
        raise ValueError(f"No target_object in VLM output: {path}")
    return data, target
