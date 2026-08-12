"""RGB-D contact-window preparation and robust VLM vote aggregation.

The model-specific inference remains in ``scripts/analyze_qwen3vl_hand_interaction.py``.
This module intentionally contains only deterministic data processing so the
contact schema and depth synchronization can be tested without loading Qwen.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


VALID_FINGERS = ("thumb", "index", "middle", "ring", "pinky")
FINGER_ALIASES = {
    "thumb_tip": "thumb",
    "index_tip": "index",
    "middle_tip": "middle",
    "ring_tip": "ring",
    "pinky_tip": "pinky",
    "little": "pinky",
    "little_finger": "pinky",
}


@dataclass(frozen=True)
class DepthFrameMatch:
    frame_index: int
    target_timestamp_s: float
    rgb_timestamp_s: float
    depth_timestamp_s: float
    depth_path: Path

    @property
    def delta_s(self) -> float:
        return self.depth_timestamp_s - self.target_timestamp_s


def normalize_finger_name(value: Any) -> str | None:
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    text = FINGER_ALIASES.get(text, text)
    return text if text in VALID_FINGERS else None


def normalize_fingers(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [item for item in values.replace(",", " ").split() if item]
    if not isinstance(values, (list, tuple, set)):
        return []
    normalized = {finger for item in values if (finger := normalize_finger_name(item)) is not None}
    return [finger for finger in VALID_FINGERS if finger in normalized]


def _target_matches(item: dict[str, Any], target_id: str) -> bool:
    values = [
        item.get("target_object_id"),
        item.get("target_id"),
        item.get("object_id"),
        item.get("name_en"),
    ]
    values = [str(value).lower() for value in values if value not in (None, "")]
    if not values:
        return True
    wanted = target_id.lower()
    return any(value == wanted or wanted in value or ("laptop" in wanted and "laptop" in value) for value in values)


def _frame_index(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and np.isfinite(value):
        return int(round(float(value)))
    if isinstance(value, str) and Path(value).stem.isdigit():
        return int(Path(value).stem)
    if isinstance(value, dict):
        for key in ("frame_index", "frame", "first_contact_frame", "contact_frame"):
            if (frame := _frame_index(value.get(key))) is not None:
                return frame
    return None


def find_coarse_contact(data: dict[str, Any], target_id: str) -> dict[str, Any]:
    """Find the existing coarse first-contact event in legacy or new VLM JSON."""
    result = data.get("vlm_result", data)
    if not isinstance(result, dict):
        raise ValueError("VLM source JSON has no object result")

    candidates: list[dict[str, Any]] = []
    for target in result.get("target_objects", []) or []:
        if not isinstance(target, dict) or not _target_matches(target, target_id):
            continue
        interaction = target.get("hand_object_interaction")
        if isinstance(interaction, dict):
            for key in ("first_contact_frame", "first_contact", "contact_start_frame"):
                value = interaction.get(key)
                if isinstance(value, dict):
                    candidate = dict(value)
                    candidate.setdefault("target_object_id", target.get("object_id", target_id))
                    candidates.append(candidate)
    for key in ("contact_analysis", "first_contact", "hand_object_interaction"):
        value = result.get(key)
        if isinstance(value, dict):
            nested = value.get("first_contact_frame")
            candidates.append(dict(nested) if isinstance(nested, dict) else dict(value))
    for key in ("hand_object_events", "first_contact_events"):
        for value in result.get(key, []) or []:
            if isinstance(value, dict):
                candidates.append(value)

    for candidate in candidates:
        if not _target_matches(candidate, target_id):
            continue
        frame = _frame_index(candidate)
        if frame is None:
            continue
        return {
            **candidate,
            "frame_index": frame,
            "contact_fingers": normalize_fingers(
                candidate.get("contact_fingers", candidate.get("fingers"))
            ),
        }
    raise ValueError(f"No coarse first-contact event found for {target_id!r}")


def make_windows(start_frame: int, end_frame: int, window_size: int = 3, stride: int = 1) -> list[list[int]]:
    if start_frame < 0 or end_frame < start_frame:
        raise ValueError(f"Invalid frame range {start_frame}..{end_frame}")
    if window_size < 1 or stride < 1:
        raise ValueError("window_size and stride must be positive")
    frame_count = end_frame - start_frame + 1
    if frame_count <= window_size:
        return [list(range(start_frame, end_frame + 1))]
    starts = list(range(start_frame, end_frame - window_size + 2, stride))
    final_start = end_frame - window_size + 1
    if starts[-1] != final_start:
        starts.append(final_start)
    return [list(range(start, start + window_size)) for start in starts]


def load_export_rows(export_root: Path) -> list[dict[str, str]]:
    path = export_root / "frames.csv"
    if not path.exists():
        raise FileNotFoundError(f"Depth synchronization table not found: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Depth synchronization table is empty: {path}")
    return rows


def match_depth_frame(
    rgb_frame: int,
    rgb_fps: float,
    export_root: Path,
    rows: list[dict[str, str]],
    time_offset_s: float = 0.0,
) -> DepthFrameMatch:
    if rgb_fps <= 0.0:
        raise ValueError("rgb_fps must be positive")
    target = time_offset_s + float(rgb_frame) / rgb_fps
    row = min(rows, key=lambda item: abs(float(item["depth_timestamp_s"]) - target))
    depth_path = export_root / row["depth_meters_npy"]
    if not depth_path.exists():
        raise FileNotFoundError(f"Depth frame listed by frames.csv is missing: {depth_path}")
    return DepthFrameMatch(
        frame_index=int(row["index"]),
        target_timestamp_s=target,
        rgb_timestamp_s=float(row["rgb_timestamp_s"]),
        depth_timestamp_s=float(row["depth_timestamp_s"]),
        depth_path=depth_path,
    )


def project_depth_to_right_image(
    meta: dict[str, Any],
    depth_m: np.ndarray,
    convention: str = "camera_to_rig",
    depth_min_m: float = 0.1,
    depth_max_m: float = 3.0,
    splat_radius_px: int = 2,
) -> np.ndarray:
    """Project a depth-camera image into the right RGB image with a z-buffer."""
    depth = np.asarray(depth_m, dtype=np.float32)
    kd = meta["depth_intrinsics"]
    kr = meta["rgb_intrinsics_right"]
    yy, xx = np.indices(depth.shape)
    z_depth = depth.reshape(-1).astype(np.float64)
    x_depth = xx.reshape(-1).astype(np.float64)
    y_depth = yy.reshape(-1).astype(np.float64)
    valid = np.isfinite(z_depth) & (z_depth > depth_min_m) & (z_depth < depth_max_m)
    z_depth = z_depth[valid]
    x_depth = x_depth[valid]
    y_depth = y_depth[valid]
    points_depth = np.stack(
        [
            (x_depth - kd["cx"]) / kd["fx"] * z_depth,
            (y_depth - kd["cy"]) / kd["fy"] * z_depth,
            z_depth,
            np.ones_like(z_depth),
        ],
        axis=1,
    )
    t_depth = np.asarray(meta["depth_extrinsics"], dtype=np.float64)
    t_right = np.asarray(meta["rgb_extrinsics_right"], dtype=np.float64)
    if convention == "camera_to_rig":
        t_right_depth = np.linalg.inv(t_right) @ t_depth
    elif convention == "rig_to_camera":
        t_right_depth = t_right @ np.linalg.inv(t_depth)
    elif convention == "direct_same_camera":
        t_right_depth = np.eye(4, dtype=np.float64)
    else:
        raise ValueError(f"Unknown depth/RGB convention: {convention}")
    points = (t_right_depth @ points_depth.T).T[:, :3]
    u = kr["fx"] * points[:, 0] / points[:, 2] + kr["cx"]
    v = kr["fy"] * points[:, 1] / points[:, 2] + kr["cy"]
    height = int(meta["rgb_height_per_eye"])
    width = int(meta["rgb_width_per_eye"])
    inside = (points[:, 2] > 0.0) & (u >= 0.0) & (u < width) & (v >= 0.0) & (v < height)
    zbuffer = np.full(height * width, np.inf, dtype=np.float32)
    valid_ids = np.flatnonzero(inside & np.isfinite(points[:, 2]))
    if len(valid_ids) == 0:
        return np.full((height, width), np.nan, dtype=np.float32)
    ui = np.rint(u[valid_ids]).astype(np.int64)
    vi = np.rint(v[valid_ids]).astype(np.int64)
    z = points[valid_ids, 2].astype(np.float32)
    radius = max(0, int(splat_radius_px))
    for dy in range(-radius, radius + 1):
        yy = vi + dy
        for dx in range(-radius, radius + 1):
            xx = ui + dx
            keep = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
            np.minimum.at(zbuffer, yy[keep] * width + xx[keep], z[keep])
    aligned = zbuffer.reshape(height, width)
    aligned[~np.isfinite(aligned)] = np.nan
    return aligned


def colorize_depth(depth_m: np.ndarray, depth_min_m: float = 0.1, depth_max_m: float = 3.0) -> Image.Image:
    """Colorize valid metric depth as blue-near to red-far; invalid pixels are black."""
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > depth_min_m) & (depth < depth_max_m)
    image = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if not valid.any():
        return Image.fromarray(image)
    lo, hi = np.quantile(depth[valid], [0.02, 0.98])
    scale = max(float(hi - lo), 1e-6)
    t = np.clip((depth - lo) / scale, 0.0, 1.0)
    t = np.nan_to_num(t, nan=0.0, posinf=1.0, neginf=0.0)
    # Compact turbo-like map: blue -> cyan -> yellow -> red.
    image[..., 0] = np.clip(255.0 * (2.0 * t - 0.25), 0.0, 255.0).astype(np.uint8)
    image[..., 1] = np.clip(255.0 * (1.0 - np.abs(2.0 * t - 1.0)), 0.0, 255.0).astype(np.uint8)
    image[..., 2] = np.clip(255.0 * (1.25 - 2.0 * t), 0.0, 255.0).astype(np.uint8)
    image[~valid] = 0
    return Image.fromarray(image)


def _font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if path.exists():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _labeled_panel(image: Image.Image, label: str, width: int) -> Image.Image:
    ratio = width / float(image.width)
    panel = image.convert("RGB").resize((width, max(1, int(round(image.height * ratio)))), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(panel)
    font = _font(max(14, width // 28))
    bbox = draw.textbbox((0, 0), label, font=font)
    draw.rectangle((6, 6, bbox[2] + 18, bbox[3] + 16), fill=(0, 0, 0))
    draw.text((12, 10), label, font=font, fill=(255, 255, 255))
    return panel


def build_rgbd_composite(
    frame_paths: list[Path],
    depth_images: list[Image.Image],
    depth_matches: list[DepthFrameMatch],
    panel_width: int = 512,
) -> Image.Image:
    if not (len(frame_paths) == len(depth_images) == len(depth_matches)):
        raise ValueError("RGB, depth, and depth-match lists must have equal length")
    rgb_panels: list[Image.Image] = []
    depth_panels: list[Image.Image] = []
    for frame_path, depth_image, match in zip(frame_paths, depth_images, depth_matches):
        frame = int(frame_path.stem)
        rgb_panels.append(_labeled_panel(Image.open(frame_path), f"RGB frame {frame}", panel_width))
        depth_panels.append(
            _labeled_panel(
                depth_image,
                f"aligned depth for {frame}  dt={match.delta_s * 1000.0:+.0f}ms",
                panel_width,
            )
        )
    separator = 8
    row_width = sum(panel.width for panel in rgb_panels) + separator * (len(rgb_panels) - 1)
    row_height = max(panel.height for panel in rgb_panels)
    canvas = Image.new("RGB", (row_width, row_height * 2 + separator), color=(0, 0, 0))
    for row, panels in enumerate((rgb_panels, depth_panels)):
        x = 0
        y = row * (row_height + separator)
        for panel in panels:
            canvas.paste(panel, (x, y))
            x += panel.width + separator
    return canvas


def parse_contact_response(data: dict[str, Any], expected_frames: Iterable[int]) -> list[dict[str, Any]]:
    expected = set(int(frame) for frame in expected_frames)
    contacts = data.get("contacts", [])
    if not isinstance(contacts, list):
        raise ValueError("Contact response must contain a contacts list")
    parsed: list[dict[str, Any]] = []
    for item in contacts:
        if not isinstance(item, dict):
            continue
        frame = _frame_index(item.get("frame", item.get("frame_index")))
        if frame not in expected:
            continue
        right_contact = bool(item.get("r_contact", item.get("right_contact", False)))
        left_contact = bool(item.get("l_contact", item.get("left_contact", False)))
        confidence = item.get("confidence", 1.0)
        try:
            confidence = float(np.clip(float(confidence), 0.0, 1.0))
        except (TypeError, ValueError):
            confidence = 1.0
        parsed.append(
            {
                "frame": int(frame),
                "right_contact": right_contact,
                "left_contact": left_contact,
                "right_fingers": normalize_fingers(item.get("r_fingers", item.get("right_fingers"))),
                "left_fingers": normalize_fingers(item.get("l_fingers", item.get("left_fingers"))),
                "contacted_part": str(item.get("contacted_part", "unknown")),
                "confidence": confidence,
            }
        )
    return parsed


def _choose_finger_combination(votes: list[dict[str, Any]], hand: str) -> list[str]:
    combinations: Counter[tuple[str, ...]] = Counter()
    last_seen: dict[tuple[str, ...], int] = {}
    for idx, vote in enumerate(votes):
        if not vote[f"{hand}_contact"]:
            continue
        combo = tuple(normalize_fingers(vote[f"{hand}_fingers"]))
        combinations[combo] += 1
        last_seen[combo] = idx
    if not combinations:
        return []
    count = max(combinations.values())
    choices = [combo for combo, value in combinations.items() if value == count]
    min_length = min(len(combo) for combo in choices)
    choices = [combo for combo in choices if len(combo) == min_length]
    chosen = max(choices, key=lambda combo: last_seen[combo])
    return list(chosen)


def aggregate_contact_windows(
    window_contacts: list[list[dict[str, Any]]],
    target_id: str,
    fps: float,
    preferred_hand: str | None = None,
) -> dict[str, Any]:
    """Aggregate overlapping window predictions using ArtHOI-style strict voting."""
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for contacts in window_contacts:
        for contact in contacts:
            by_frame[int(contact["frame"])].append(contact)
    per_frame: list[dict[str, Any]] = []
    for frame in sorted(by_frame):
        votes = by_frame[frame]
        item: dict[str, Any] = {"frame_index": frame, "time_sec": frame / fps, "vote_count": len(votes)}
        for hand in ("left", "right"):
            positive = sum(bool(vote[f"{hand}_contact"]) for vote in votes)
            contact = positive > len(votes) - positive
            item[f"{hand}_contact"] = contact
            item[f"{hand}_positive_votes"] = positive
            item[f"{hand}_contact_vote_ratio"] = positive / len(votes)
            item[f"{hand}_fingers"] = _choose_finger_combination(votes, hand) if contact else []
            positive_confidences = [vote["confidence"] for vote in votes if vote[f"{hand}_contact"]]
            item[f"{hand}_positive_mean_confidence"] = (
                float(np.mean(positive_confidences)) if positive_confidences else 0.0
            )
        active_votes = [vote for vote in votes if vote["left_contact"] or vote["right_contact"]]
        parts = [vote["contacted_part"] for vote in active_votes if vote["contacted_part"] not in ("", "unknown")]
        item["contacted_part"] = Counter(parts).most_common(1)[0][0] if parts else "unknown"
        item["mean_model_confidence"] = float(np.mean([vote["confidence"] for vote in votes]))
        per_frame.append(item)

    if preferred_hand not in ("left", "right"):
        positive_counts = {
            hand: sum(bool(item[f"{hand}_contact"]) for item in per_frame)
            for hand in ("left", "right")
        }
        preferred_hand = max(("left", "right"), key=lambda hand: positive_counts[hand])
    positive_frames = [item for item in per_frame if item[f"{preferred_hand}_contact"]]
    first = positive_frames[0] if positive_frames else None
    fingers = list(first[f"{preferred_hand}_fingers"]) if first else []
    primary = fingers[0] if fingers else None
    first_contact = {
        "frame_index": int(first["frame_index"]) if first else None,
        "time_sec": float(first["time_sec"]) if first else None,
        "hand_side": preferred_hand if first else "unknown",
        "contacted_part": first["contacted_part"] if first else "unknown",
        "contact_type": "touch" if first else "none",
        "contact_fingers": fingers,
        "primary_contact_finger": primary,
        "evidence_frames": [int(item["frame_index"]) for item in positive_frames],
        "confidence": (
            float(
                first[f"{preferred_hand}_contact_vote_ratio"]
                * first[f"{preferred_hand}_positive_mean_confidence"]
            )
            if first
            else 0.0
        ),
        "reason": "strict majority across overlapping RGB-D contact windows" if first else "no strict-majority contact frame",
    }
    return {
        "target_object_id": target_id,
        "event_type": "first_contact",
        "aggregation": "strict contact majority; ArtHOI-style finger-combination mode",
        "first_contact_frame": first_contact,
        "per_frame_contacts": per_frame,
    }


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
