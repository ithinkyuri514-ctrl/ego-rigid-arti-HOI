#!/usr/bin/env python3
"""Register temporally-nearest 15 fps artifacts onto native RGB-D frames.

The native SpatialMP4 RGB-D cadence does not, in general, land on the old
15 fps RGB timestamps.  Blindly reusing the nearest DiffuEraser image and SAM2
mask therefore introduces a small but visible spatial offset.  This utility
keeps the old inference products, but registers them to each exact native RGB
frame with dense backward optical flow::

    target native pixel -> corresponding source 15 fps pixel

Frames whose timestamps and decoded RGB pixels are identical use an identity
mapping.  For the remaining frames, OpenCV DIS estimates target-to-source flow
at a configurable scale and the flow is rescaled to the native resolution.

DiffuEraser RGB is deliberately *not* copied wholesale.  The exact native RGB
is retained outside a dilated, warped hand mask, while warped hand-removed RGB
is feathered only inside that support.  SAM2 masks use nearest-neighbour
sampling and are written as binary PNGs.

Before canonical files are replaced, their old temporal-nearest symlink targets
or regular files are preserved under ``outputs/00_native_reuse``.  The source
workspace is never modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE = PROJECT_ROOT / "run_mixed_20260728_203734_depth40"
DEFAULT_SOURCE_WORKSPACE = PROJECT_ROOT / "run_mixed_20260728_203734"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--source-workspace", type=Path, default=DEFAULT_SOURCE_WORKSPACE)
    parser.add_argument(
        "--mapping-csv",
        type=Path,
        default=None,
        help="Native-to-old mapping; defaults to Stage00 native40_mapping.csv.",
    )
    parser.add_argument(
        "--objects",
        nargs="*",
        default=None,
        help="Object ids to register; defaults to all source Stage04 object directories.",
    )
    parser.add_argument(
        "--flow-scale",
        type=float,
        default=0.5,
        help="Scale used for DIS estimation; vectors are rescaled to native pixels.",
    )
    parser.add_argument(
        "--dis-preset",
        choices=("ultrafast", "fast", "medium"),
        default="medium",
    )
    parser.add_argument("--exact-tolerance-s", type=float, default=1e-6)
    parser.add_argument("--max-source-delta-s", type=float, default=0.025)
    parser.add_argument(
        "--hand-dilate-px",
        type=int,
        default=32,
        help="Native-resolution dilation radius for the local DiffuEraser paste support.",
    )
    parser.add_argument(
        "--feather-px",
        type=float,
        default=12.0,
        help="Inward feather width at the dilated hand-support boundary.",
    )
    parser.add_argument("--mask-threshold", type=int, default=127)
    parser.add_argument("--video-crf", type=int, default=16)
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=None,
        help="Defaults to outputs/00_native_reuse/temporal_nearest_before_dense_flow.",
    )
    parser.add_argument("--skip-video", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False, default=json_value)
            stream.write("\n")
        os.chmod(temporary_name, 0o644)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_png(path: Path, image: np.ndarray) -> None:
    """Write a PNG without ever following an existing canonical symlink."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".png", dir=path.parent)
    os.close(fd)
    try:
        if not cv2.imwrite(temporary_name, image, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
            raise RuntimeError(f"OpenCV failed to write {temporary_name}")
        os.chmod(temporary_name, 0o644)
        # os.replace would replace a symlink rather than its target, but unlink
        # explicitly to make the source-workspace safety invariant obvious.
        if path.is_symlink():
            path.unlink()
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def read_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not decode RGB image: {path}")
    return image


def read_binary_mask(path: Path, threshold: int, shape: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not decode mask: {path}")
    if mask.shape != shape:
        raise ValueError(f"Mask shape {mask.shape} != RGB shape {shape}: {path}")
    return np.where(mask > threshold, 255, 0).astype(np.uint8)


def discover_objects(source_workspace: Path) -> list[str]:
    root = source_workspace / "outputs/04_object_masks"
    if not root.is_dir():
        raise FileNotFoundError(root)
    objects = []
    for child in sorted(root.iterdir()):
        object_id = child.name
        if (child / "objects" / object_id).is_dir():
            objects.append(object_id)
    if not objects:
        raise RuntimeError(f"No Stage04 object-mask directories found below {root}")
    return objects


def preserve_previous(path: Path, backup: Path) -> dict[str, Any]:
    """Preserve the pre-registration canonical artifact exactly once."""
    record: dict[str, Any] = {
        "canonical_path": str(path),
        "backup_path": str(backup),
        "existed": bool(path.exists() or path.is_symlink()),
    }
    if not record["existed"]:
        return record
    backup.parent.mkdir(parents=True, exist_ok=True)
    if backup.exists() or backup.is_symlink():
        record["backup_reused"] = True
        if backup.is_symlink():
            record["backup_source"] = str(backup.resolve(strict=False))
        return record
    if path.is_symlink():
        source = path.resolve(strict=True)
        backup.symlink_to(source)
        record.update(
            {
                "previous_kind": "symlink",
                "previous_link_target": os.readlink(path),
                "backup_source": str(source),
            }
        )
    elif path.is_file():
        shutil.copy2(path, backup)
        record.update({"previous_kind": "regular_file", "backup_source": str(backup)})
    else:
        raise ValueError(f"Expected a file or symlink at canonical artifact path: {path}")
    return record


def preserve_manifest(source: Path, destination: Path) -> str | None:
    if not source.is_file():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copy2(source, destination)
    return str(destination)


def dis_preset(name: str) -> int:
    return {
        "ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
        "fast": cv2.DISOPTICAL_FLOW_PRESET_FAST,
        "medium": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
    }[name]


def estimate_target_to_source_flow(
    target_bgr: np.ndarray,
    source_bgr: np.ndarray,
    estimator: cv2.DISOpticalFlow,
    scale: float,
) -> np.ndarray:
    height, width = target_bgr.shape[:2]
    scaled_width = max(16, int(round(width * scale)))
    scaled_height = max(16, int(round(height * scale)))
    size = (scaled_width, scaled_height)
    target_gray = cv2.cvtColor(
        cv2.resize(target_bgr, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY
    )
    source_gray = cv2.cvtColor(
        cv2.resize(source_bgr, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY
    )
    flow_small = estimator.calc(target_gray, source_gray, None)
    if flow_small is None or flow_small.shape != (scaled_height, scaled_width, 2):
        raise RuntimeError(f"DIS returned an invalid flow array: {None if flow_small is None else flow_small.shape}")
    flow = cv2.resize(flow_small, (width, height), interpolation=cv2.INTER_LINEAR)
    flow[..., 0] /= scaled_width / width
    flow[..., 1] /= scaled_height / height
    if not np.all(np.isfinite(flow)):
        raise RuntimeError("Dense optical flow contains NaN or infinity")
    return flow.astype(np.float32, copy=False)


def remap_image(
    image: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    interpolation: int,
    border_mode: int,
) -> np.ndarray:
    return cv2.remap(
        image,
        map_x,
        map_y,
        interpolation,
        borderMode=border_mode,
        borderValue=0,
    )


def dilated_feather_alpha(mask: np.ndarray, radius: int, feather_px: float) -> tuple[np.ndarray, np.ndarray]:
    support = mask > 0
    if radius > 0 and np.any(support):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        support = cv2.dilate(support.astype(np.uint8), kernel, iterations=1) > 0
    if not np.any(support):
        return support, np.zeros(mask.shape, dtype=np.float32)
    if feather_px <= 0:
        return support, support.astype(np.float32)
    distance = cv2.distanceTransform(support.astype(np.uint8), cv2.DIST_L2, 5)
    alpha = np.clip(distance / float(feather_px), 0.0, 1.0).astype(np.float32)
    alpha[~support] = 0.0
    return support, alpha


def compose_local_inpaint(
    native_bgr: np.ndarray,
    warped_inpaint_bgr: np.ndarray,
    warped_hand_mask: np.ndarray,
    dilate_px: int,
    feather_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    support, alpha = dilated_feather_alpha(warped_hand_mask, dilate_px, feather_px)
    if not np.any(support):
        return native_bgr.copy(), support, alpha
    alpha3 = alpha[..., None]
    composite = np.rint(
        native_bgr.astype(np.float32) * (1.0 - alpha3)
        + warped_inpaint_bgr.astype(np.float32) * alpha3
    ).clip(0, 255).astype(np.uint8)
    # This explicit assignment is also checked in the manifest QC.
    composite[~support] = native_bgr[~support]
    return composite, support, alpha


def mask_stats(source: np.ndarray, warped: np.ndarray) -> dict[str, Any]:
    source_pixels = int(np.count_nonzero(source))
    warped_pixels = int(np.count_nonzero(warped))
    union = int(np.count_nonzero((source > 0) | (warped > 0)))
    intersection = int(np.count_nonzero((source > 0) & (warped > 0)))
    return {
        "source_foreground_pixels": source_pixels,
        "warped_foreground_pixels": warped_pixels,
        "warped_over_source_area": float(warped_pixels / source_pixels) if source_pixels else None,
        "warped_vs_unwarped_iou": float(intersection / union) if union else 1.0,
    }


def finite_float(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def summarize_frames(records: list[dict[str, Any]], object_ids: list[str]) -> dict[str, Any]:
    flow_records = [record for record in records if record["registration_mode"] == "dense_flow"]
    before = np.asarray([record["flow_qc"]["photometric_mae_before"] for record in flow_records])
    after = np.asarray([record["flow_qc"]["photometric_mae_after"] for record in flow_records])
    summary: dict[str, Any] = {
        "frame_count": len(records),
        "exact_identity_frames": sum(record["registration_mode"] == "exact_identity" for record in records),
        "dense_flow_frames": len(flow_records),
        "max_absolute_timestamp_delta_s": max(record["absolute_delta_s"] for record in records),
        "outside_native_changed_pixels_total": sum(
            record["inpaint_qc"]["outside_support_changed_pixels"] for record in records
        ),
    }
    if len(flow_records):
        summary["flow_photometric_qc"] = {
            "median_mae_before": float(np.median(before)),
            "median_mae_after": float(np.median(after)),
            "mean_mae_before": float(np.mean(before)),
            "mean_mae_after": float(np.mean(after)),
            "frames_improved": int(np.count_nonzero(after < before)),
            "median_improvement_fraction": float(np.median(1.0 - after / np.maximum(before, 1e-6))),
            "median_flow_magnitude_px": float(
                np.median([record["flow_qc"]["magnitude_median_px"] for record in flow_records])
            ),
            "median_p95_flow_magnitude_px": float(
                np.median([record["flow_qc"]["magnitude_p95_px"] for record in flow_records])
            ),
        }
    mask_ids = ["hand", *object_ids]
    summary["mask_area_qc"] = {}
    for mask_id in mask_ids:
        ratios = [
            record["mask_qc"][mask_id]["warped_over_source_area"]
            for record in records
            if record["mask_qc"][mask_id]["warped_over_source_area"] is not None
        ]
        summary["mask_area_qc"][mask_id] = {
            "median_warped_over_source_area": float(np.median(ratios)) if ratios else None,
            "min_warped_over_source_area": float(np.min(ratios)) if ratios else None,
            "max_warped_over_source_area": float(np.max(ratios)) if ratios else None,
        }
    return summary


def build_video(frames: Path, output: Path, fps: float, crf: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{output.stem}.", suffix=".mp4", dir=output.parent)
    os.close(fd)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-framerate",
                f"{fps:.12f}",
                "-i",
                str(frames / "%06d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                str(crf),
                temporary_name,
            ],
            check=True,
        )
        os.chmod(temporary_name, 0o644)
        if output.is_symlink():
            output.unlink()
        os.replace(temporary_name, output)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def replace_video_alias(target: Path, alias: Path) -> None:
    alias.parent.mkdir(parents=True, exist_ok=True)
    if alias.exists() or alias.is_symlink():
        alias.unlink()
    alias.symlink_to(target.resolve())


def main() -> None:
    args = parse_args()
    workspace = args.workspace.resolve()
    source_workspace = args.source_workspace.resolve()
    if workspace == source_workspace:
        raise ValueError("Source and destination workspaces must be different")
    if not 0 < args.flow_scale <= 1.0:
        raise ValueError("--flow-scale must be in (0, 1]")
    if args.hand_dilate_px < 0 or args.feather_px < 0:
        raise ValueError("Hand dilation and feather widths must be non-negative")

    mapping_csv = args.mapping_csv or workspace / "outputs/00_rgb_frames/native40_mapping.csv"
    mapping_rows = read_csv(mapping_csv)
    if not mapping_rows:
        raise RuntimeError(f"Mapping CSV is empty: {mapping_csv}")
    source_timeline_path = source_workspace / "outputs/00_rgb_frames/timeline.csv"
    source_timeline = read_csv(source_timeline_path)
    source_times = np.asarray([float(row["rgb_timestamp_s"]) for row in source_timeline])

    object_ids = args.objects if args.objects is not None else discover_objects(source_workspace)
    object_ids = list(dict.fromkeys(object_ids))
    if not object_ids:
        raise ValueError("At least one object id is required")

    target_rgb_dir = workspace / "outputs/00_rgb_frames/right_rgb_png"
    source_rgb_dir = source_workspace / "outputs/00_rgb_frames/right_rgb_png"
    source_inpaint_dir = source_workspace / "outputs/03_diffueraser/inpainted_frames_png"
    source_hand_dir = source_workspace / "outputs/02_hand_masks/objects/hand"
    output_inpaint_dir = workspace / "outputs/03_diffueraser/inpainted_frames_png"
    output_hand_dirs = [
        workspace / "outputs/02_hand_masks/objects/hand",
        workspace / "outputs/02_hand_masks/combined",
    ]
    source_object_dirs = {
        object_id: source_workspace / f"outputs/04_object_masks/{object_id}/objects/{object_id}"
        for object_id in object_ids
    }
    output_object_dirs = {
        object_id: [
            workspace / f"outputs/04_object_masks/{object_id}/objects/{object_id}",
            workspace / f"outputs/04_object_masks/{object_id}/combined",
            workspace / f"outputs/02_sam2_frame0_masks/propagated/objects/{object_id}",
        ]
        for object_id in object_ids
    }

    backup_root = args.backup_dir or (
        workspace / "outputs/00_native_reuse/temporal_nearest_before_dense_flow"
    )
    preserved_manifests = {
        "top_level_temporal_nearest": preserve_manifest(
            workspace / "outputs/temporal_nearest_reuse_manifest.json",
            backup_root / "manifests/temporal_nearest_reuse_manifest.json",
        ),
        "stage03_diffueraser": preserve_manifest(
            workspace / "outputs/03_diffueraser/diffueraser_manifest.json",
            backup_root / "manifests/stage03_diffueraser_manifest.json",
        ),
    }
    for object_id in object_ids:
        preserved_manifests[f"stage04_{object_id}"] = preserve_manifest(
            workspace / f"outputs/04_object_masks/{object_id}/temporal_nearest_manifest.json",
            backup_root / f"manifests/stage04_{object_id}_temporal_nearest_manifest.json",
        )

    estimator = cv2.DISOpticalFlow_create(dis_preset(args.dis_preset))
    estimator.setUseSpatialPropagation(True)
    frame_records: list[dict[str, Any]] = []
    backup_records: list[dict[str, Any]] = []
    native_times: list[float] = []

    for ordinal, row in enumerate(mapping_rows):
        native_index = int(row.get("frame_index", ordinal))
        native_time = float(row["rgb_timestamp_s"])
        native_times.append(native_time)
        if "old_nearest_frame_index" in row and row["old_nearest_frame_index"] != "":
            source_index = int(row["old_nearest_frame_index"])
        else:
            source_index = int(np.argmin(np.abs(source_times - native_time)))
        if not 0 <= source_index < len(source_times):
            raise IndexError(f"Mapped source frame is out of range: {source_index}")
        source_time = float(source_times[source_index])
        delta = source_time - native_time
        if abs(delta) > args.max_source_delta_s:
            raise RuntimeError(
                f"Nearest source delta {delta:+.6f}s exceeds {args.max_source_delta_s}s "
                f"at native frame {native_index}"
            )

        target_rgb_path = target_rgb_dir / f"{native_index:06d}.png"
        source_rgb_path = source_rgb_dir / f"{source_index:06d}.png"
        source_inpaint_path = source_inpaint_dir / f"{source_index:06d}.png"
        source_hand_path = source_hand_dir / f"{source_index:06d}.png"
        target_bgr = read_bgr(target_rgb_path)
        source_bgr = read_bgr(source_rgb_path)
        source_inpaint_bgr = read_bgr(source_inpaint_path)
        if source_bgr.shape != target_bgr.shape or source_inpaint_bgr.shape != target_bgr.shape:
            raise ValueError(
                f"RGB shape mismatch at native/source {native_index}/{source_index}: "
                f"target={target_bgr.shape}, source={source_bgr.shape}, inpaint={source_inpaint_bgr.shape}"
            )
        height, width = target_bgr.shape[:2]
        source_hand = read_binary_mask(source_hand_path, args.mask_threshold, (height, width))
        source_masks = {"hand": source_hand}
        for object_id, directory in source_object_dirs.items():
            source_masks[object_id] = read_binary_mask(
                directory / f"{source_index:06d}.png", args.mask_threshold, (height, width)
            )

        pixels_equal = bool(np.array_equal(target_bgr, source_bgr))
        exact = abs(delta) <= args.exact_tolerance_s and pixels_equal
        if exact:
            registration_mode = "exact_identity"
            warped_source_bgr = source_bgr
            warped_inpaint_bgr = source_inpaint_bgr
            warped_masks = {mask_id: mask.copy() for mask_id, mask in source_masks.items()}
            magnitude = np.zeros((height, width), dtype=np.float32)
            valid = np.ones((height, width), dtype=bool)
        else:
            registration_mode = "dense_flow"
            flow = estimate_target_to_source_flow(
                target_bgr, source_bgr, estimator, args.flow_scale
            )
            base_x, base_y = np.meshgrid(
                np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
            )
            map_x = base_x + flow[..., 0]
            map_y = base_y + flow[..., 1]
            valid = (map_x >= 0.0) & (map_x <= width - 1.0) & (map_y >= 0.0) & (map_y <= height - 1.0)
            warped_source_bgr = remap_image(
                source_bgr, map_x, map_y, cv2.INTER_LINEAR, cv2.BORDER_REFLECT_101
            )
            warped_inpaint_bgr = remap_image(
                source_inpaint_bgr, map_x, map_y, cv2.INTER_LINEAR, cv2.BORDER_REFLECT_101
            )
            warped_masks = {
                mask_id: np.where(
                    remap_image(mask, map_x, map_y, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)
                    > args.mask_threshold,
                    255,
                    0,
                ).astype(np.uint8)
                for mask_id, mask in source_masks.items()
            }
            magnitude = np.linalg.norm(flow, axis=2)

        composite, paste_support, alpha = compose_local_inpaint(
            target_bgr,
            warped_inpaint_bgr,
            warped_masks["hand"],
            args.hand_dilate_px,
            args.feather_px,
        )
        output_inpaint = output_inpaint_dir / f"{native_index:06d}.png"
        backup_records.append(
            preserve_previous(
                output_inpaint, backup_root / f"stage03_diffueraser/{native_index:06d}.png"
            )
        )
        atomic_write_png(output_inpaint, composite)

        for mirror_index, output_dir in enumerate(output_hand_dirs):
            output_path = output_dir / f"{native_index:06d}.png"
            backup_records.append(
                preserve_previous(
                    output_path,
                    backup_root / f"stage02_hand/mirror_{mirror_index}/{native_index:06d}.png",
                )
            )
            atomic_write_png(output_path, warped_masks["hand"])

        for object_id, output_dirs in output_object_dirs.items():
            for mirror_index, output_dir in enumerate(output_dirs):
                output_path = output_dir / f"{native_index:06d}.png"
                backup_records.append(
                    preserve_previous(
                        output_path,
                        backup_root
                        / f"stage04_{object_id}/mirror_{mirror_index}/{native_index:06d}.png",
                    )
                )
                atomic_write_png(output_path, warped_masks[object_id])

        before_mae = float(np.mean(cv2.absdiff(target_bgr, source_bgr)))
        after_mae = float(np.mean(cv2.absdiff(target_bgr, warped_source_bgr)))
        outside_changed = int(
            np.count_nonzero(np.any(composite[~paste_support] != target_bgr[~paste_support], axis=1))
        )
        inside_change = cv2.absdiff(composite, target_bgr)
        frame_records.append(
            {
                "native_frame_index": native_index,
                "native_timestamp_s": native_time,
                "source_frame_index": source_index,
                "source_timestamp_s": source_time,
                "source_minus_native_s": delta,
                "absolute_delta_s": abs(delta),
                "decoded_rgb_pixels_equal": pixels_equal,
                "registration_mode": registration_mode,
                "flow_qc": {
                    "direction": "target_native_to_source_15fps",
                    "photometric_mae_before": before_mae,
                    "photometric_mae_after": after_mae,
                    "photometric_improvement_fraction": finite_float(
                        1.0 - after_mae / max(before_mae, 1e-6)
                    ),
                    "magnitude_mean_px": float(np.mean(magnitude)),
                    "magnitude_median_px": float(np.median(magnitude)),
                    "magnitude_p95_px": float(np.percentile(magnitude, 95)),
                    "magnitude_max_px": float(np.max(magnitude)),
                    "out_of_bounds_sampling_fraction": float(1.0 - np.mean(valid)),
                },
                "mask_qc": {
                    mask_id: mask_stats(source_masks[mask_id], warped_masks[mask_id])
                    for mask_id in source_masks
                },
                "inpaint_qc": {
                    "warped_hand_foreground_fraction": float(
                        np.mean(warped_masks["hand"] > 0)
                    ),
                    "dilated_paste_support_fraction": float(np.mean(paste_support)),
                    "full_alpha_fraction": float(np.mean(alpha >= 1.0)),
                    "outside_support_changed_pixels": outside_changed,
                    "inside_support_mean_absolute_change": float(
                        np.mean(inside_change[paste_support])
                    )
                    if np.any(paste_support)
                    else 0.0,
                },
            }
        )
        print(
            f"[{ordinal + 1:02d}/{len(mapping_rows):02d}] native {native_index:06d} <- "
            f"old {source_index:06d} delta={delta:+.3f}s {registration_mode} "
            f"MAE {before_mae:.3f}->{after_mae:.3f}",
            flush=True,
        )

    timestamps = np.asarray(native_times, dtype=np.float64)
    if len(timestamps) > 1:
        fps = float(1.0 / np.median(np.diff(timestamps)))
    else:
        fps = 1.0
    output_video = workspace / "outputs/03_diffueraser/inpainted_right_rgb_native40.mp4"
    if not args.skip_video:
        build_video(output_inpaint_dir, output_video, fps, args.video_crf)
        replace_video_alias(
            output_video, workspace / "outputs/03_diffueraser/inpainted_right_rgb_15fps.mp4"
        )

    summary = summarize_frames(frame_records, object_ids)
    if summary["outside_native_changed_pixels_total"] != 0:
        raise RuntimeError("Local inpaint violated exact-native background preservation")
    manifest_path = workspace / "outputs/native_artifact_dense_flow_manifest.json"
    manifest: dict[str, Any] = {
        "status": "completed",
        "stage": "native_artifact_dense_flow_registration",
        "workspace": str(workspace),
        "source_workspace": str(source_workspace),
        "source_mapping": str(mapping_csv),
        "source_timeline": str(source_timeline_path),
        "policy": {
            "mapping": "nearest old 15fps timestamp, bounded by max_source_delta_s",
            "exact_frame": "identity iff timestamp-equivalent and decoded RGB pixels are equal",
            "non_exact_frame": "dense backward optical flow target(native)->source(old)",
            "mask_sampling": "nearest-neighbour, binary uint8",
            "inpaint_compositing": (
                "exact native RGB outside dilated warped hand support; distance-transform feathered "
                "warped DiffuEraser RGB inside"
            ),
            "authoritative_geometry": "native metric depth and native camera pose; VDA not used",
        },
        "parameters": {
            "flow_method": "OpenCV DISOpticalFlow",
            "dis_preset": args.dis_preset,
            "flow_scale": args.flow_scale,
            "flow_direction": "target_native_to_source_15fps",
            "exact_tolerance_s": args.exact_tolerance_s,
            "max_source_delta_s": args.max_source_delta_s,
            "hand_dilate_px": args.hand_dilate_px,
            "feather_px": args.feather_px,
            "mask_threshold": args.mask_threshold,
        },
        "objects": object_ids,
        "outputs": {
            "inpainted_frames": str(output_inpaint_dir),
            "inpainted_video": str(output_video) if not args.skip_video else None,
            "hand_masks": [str(path) for path in output_hand_dirs],
            "object_masks": {
                object_id: [str(path) for path in paths]
                for object_id, paths in output_object_dirs.items()
            },
        },
        "pre_registration_backup": {
            "root": str(backup_root),
            "preserved_manifests": preserved_manifests,
            "artifact_records": backup_records,
        },
        "quality_control": summary,
        "frames": frame_records,
    }
    atomic_write_json(manifest_path, manifest)

    stage03_manifest = {
        "status": "completed",
        "stage": "03_diffueraser_hand_removal",
        "inference_reused": True,
        "registration": "dense target-native-to-source-15fps optical flow with exact-frame identity",
        "local_composite": True,
        "native_background_preserved_exactly": True,
        "output_frames": str(output_inpaint_dir),
        "output_video": str(output_video) if not args.skip_video else None,
        "registration_manifest": str(manifest_path),
        "quality_control": summary,
    }
    atomic_write_json(workspace / "outputs/03_diffueraser/diffueraser_manifest.json", stage03_manifest)
    atomic_write_json(
        workspace / "outputs/02_hand_masks/dense_flow_registration_manifest.json",
        {
            "status": "completed",
            "stage": "02_hand_masks",
            "registration_manifest": str(manifest_path),
            "mask_directory": str(output_hand_dirs[0]),
            "frame_count": len(frame_records),
        },
    )
    for object_id in object_ids:
        atomic_write_json(
            workspace
            / f"outputs/04_object_masks/{object_id}/dense_flow_registration_manifest.json",
            {
                "status": "completed",
                "stage": "04_sam2_object_masks",
                "object_id": object_id,
                "registration_manifest": str(manifest_path),
                "mask_directory": str(output_object_dirs[object_id][0]),
                "frame_count": len(frame_records),
                "quality_control": summary["mask_area_qc"][object_id],
            },
        )

    print(json.dumps({"manifest": str(manifest_path), "quality_control": summary}, indent=2))


if __name__ == "__main__":
    main()
