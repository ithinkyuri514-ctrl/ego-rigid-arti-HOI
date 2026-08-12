#!/usr/bin/env python3
"""Refine native right-eye sparse metric depth with LingBot-Depth v0.5."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE = PROJECT_ROOT / "run_mixed_20260728_203734_depth40"
DEFAULT_MODEL = Path("/code/lingbot-depth/ckpt/lingbot-depth-v0.5/model.pt")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def discover_images(directory: Path) -> list[Path]:
    for pattern in ("*.png", "*.jpg", "*.jpeg"):
        paths = sorted(directory.glob(pattern))
        if paths:
            return paths
    return []


def depth_preview(
    depth: np.ndarray,
    valid: np.ndarray,
    minimum: float,
    maximum: float,
) -> np.ndarray:
    normalized = np.clip((depth - minimum) / max(maximum - minimum, 1e-6), 0.0, 1.0)
    colored = cv2.applyColorMap((normalized * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def depth_stats(depth: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    values = depth[valid]
    if not len(values):
        return {"valid_pixels": 0, "valid_ratio": 0.0, "percentiles_m": None}
    percentiles = np.percentile(values, [1.0, 50.0, 99.0])
    return {
        "valid_pixels": int(len(values)),
        "valid_ratio": float(len(values) / depth.size),
        "percentiles_m": {
            "p01": float(percentiles[0]),
            "p50": float(percentiles[1]),
            "p99": float(percentiles[2]),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--rgb-dir", type=Path, default=None)
    parser.add_argument("--raw-depth-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--frame-count", type=int, default=40)
    parser.add_argument("--resolution-level", type=int, choices=range(10), default=9)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--input-min-depth", type=float, default=0.1)
    parser.add_argument("--input-max-depth", type=float, default=5.0)
    parser.add_argument("--preview-min-depth", type=float, default=0.1)
    parser.add_argument("--preview-max-depth", type=float, default=3.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = args.workspace.resolve()
    model_path = args.model.resolve()
    rgb_dir = (args.rgb_dir or workspace / "outputs/00_rgb_frames/right_rgb_png").resolve()
    raw_depth_dir = (
        args.raw_depth_dir or workspace / "outputs/06_dense_depth/raw_projected_npy"
    ).resolve()
    output_dir = (args.output_dir or workspace / "outputs/06_lingbot_depth").resolve()
    depth_output_dir = output_dir / "metric_depth_npy"
    mask_output_dir = output_dir / "valid_mask_png"
    preview_output_dir = output_dir / "preview"

    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    camera_path = workspace / "outputs/00_rgb_frames/camera.json"
    camera = read_json(camera_path)
    if str(camera.get("selected_eye", "right")).lower() != "right":
        raise ValueError(f"Expected right-eye input, got {camera.get('selected_eye')!r}")
    intrinsics = camera["rgb_intrinsics_right"]

    rgb_paths = discover_images(rgb_dir)
    raw_depth_paths = sorted(raw_depth_dir.glob("*.npy"))
    frame_count = int(args.frame_count)
    if frame_count <= 0:
        raise ValueError("--frame-count must be positive")
    if len(rgb_paths) < frame_count or len(raw_depth_paths) < frame_count:
        raise ValueError(
            f"Requested {frame_count} frames, available RGB/raw depth="
            f"{len(rgb_paths)}/{len(raw_depth_paths)}"
        )
    rgb_paths = rgb_paths[:frame_count]
    raw_depth_paths = raw_depth_paths[:frame_count]

    existing = list(depth_output_dir.glob("*.npy")) if depth_output_dir.is_dir() else []
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Output already contains {len(existing)} depth maps: {depth_output_dir}; "
            "pass --overwrite to replace matching frame files"
        )
    for directory in (depth_output_dir, mask_output_dir, preview_output_dir):
        directory.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    from mdm.model.v2 import MDMModel

    torch.set_float32_matmul_precision("high")
    print(f"Loading LingBot-Depth from {model_path} on {device}...", flush=True)
    load_start = time.perf_counter()
    model = MDMModel.from_pretrained(str(model_path)).to(device).eval()
    load_seconds = time.perf_counter() - load_start
    print(f"Model loaded in {load_seconds:.2f}s", flush=True)

    records: list[dict[str, Any]] = []
    sequence_start = time.perf_counter()
    image_shape: tuple[int, int] | None = None
    normalized_k: np.ndarray | None = None
    for frame_index, (rgb_path, raw_depth_path) in enumerate(zip(rgb_paths, raw_depth_paths)):
        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Failed to read RGB image: {rgb_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        raw_depth = np.load(raw_depth_path).astype(np.float32, copy=False)
        height, width = rgb.shape[:2]
        if raw_depth.shape != (height, width):
            raise ValueError(
                f"Frame {frame_index} RGB/depth mismatch: {(height, width)}/{raw_depth.shape}"
            )
        if image_shape is None:
            image_shape = (height, width)
            normalized_k = np.asarray(
                [
                    [float(intrinsics["fx"]) / width, 0.0, float(intrinsics["cx"]) / width],
                    [0.0, float(intrinsics["fy"]) / height, float(intrinsics["cy"]) / height],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
        elif image_shape != (height, width):
            raise ValueError(
                f"RGB shape changed at frame {frame_index}: "
                f"{image_shape} -> {(height, width)}"
            )

        raw_valid = (
            np.isfinite(raw_depth)
            & (raw_depth >= float(args.input_min_depth))
            & (raw_depth <= float(args.input_max_depth))
        )
        clean_depth = np.where(raw_valid, raw_depth, 0.0).astype(np.float32)
        image_tensor = (
            torch.from_numpy(np.ascontiguousarray(rgb))
            .to(device=device, dtype=torch.float32)
            .permute(2, 0, 1)
            .unsqueeze(0)
            / 255.0
        )
        depth_tensor = torch.from_numpy(clean_depth).to(device=device).unsqueeze(0)
        intrinsics_tensor = torch.from_numpy(normalized_k).to(device=device).unsqueeze(0)

        infer_start = time.perf_counter()
        output = model.infer(
            image_tensor,
            depth_in=depth_tensor,
            intrinsics=intrinsics_tensor,
            resolution_level=int(args.resolution_level),
            apply_mask=False,
            use_fp16=not args.no_fp16,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        infer_seconds = time.perf_counter() - infer_start

        predicted = (
            output["depth"]
            .squeeze(0)
            .detach()
            .float()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        if predicted.shape != (height, width):
            raise ValueError(
                f"Frame {frame_index} LingBot output shape {predicted.shape}, expected {(height, width)}"
            )
        learned_mask_tensor = output.get("mask")
        if learned_mask_tensor is None:
            learned_mask = np.ones((height, width), dtype=bool)
        else:
            learned_mask = learned_mask_tensor.squeeze(0).detach().cpu().numpy().astype(bool)
        finite_positive = np.isfinite(predicted) & (predicted > 0.0)

        output_path = depth_output_dir / f"{frame_index:06d}.npy"
        mask_path = mask_output_dir / f"{frame_index:06d}.png"
        preview_path = preview_output_dir / f"{frame_index:06d}.png"
        np.save(output_path, predicted)
        cv2.imwrite(str(mask_path), learned_mask.astype(np.uint8) * 255)
        preview_valid = finite_positive & learned_mask
        cv2.imwrite(
            str(preview_path),
            depth_preview(
                predicted,
                preview_valid,
                float(args.preview_min_depth),
                float(args.preview_max_depth),
            ),
        )

        record = {
            "frame_index": frame_index,
            "rgb": str(rgb_path),
            "raw_metric_depth": str(raw_depth_path),
            "output_metric_depth": str(output_path),
            "learned_valid_mask": str(mask_path),
            "preview": str(preview_path),
            "inference_seconds": infer_seconds,
            "input": depth_stats(clean_depth, raw_valid),
            "prediction": depth_stats(predicted, finite_positive),
            "prediction_learned_valid_ratio": float(learned_mask.mean()),
        }
        records.append(record)
        prediction_stats = record["prediction"]
        print(
            f"[{frame_index + 1:02d}/{frame_count}] {infer_seconds:.2f}s | "
            f"input {record['input']['valid_ratio']:.2%} | "
            f"prediction {prediction_stats['valid_ratio']:.2%} | "
            f"p50 {prediction_stats['percentiles_m']['p50']:.3f}m",
            flush=True,
        )

        del image_tensor, depth_tensor, intrinsics_tensor, output

    elapsed_seconds = time.perf_counter() - sequence_start
    assert image_shape is not None and normalized_k is not None
    manifest = {
        "stage": "06_lingbot_depth_native40",
        "status": "completed",
        "model": str(model_path),
        "model_role": "RGB-D metric depth completion and refinement",
        "rgb_only_model": False,
        "workspace": str(workspace),
        "frame_count": frame_count,
        "selected_eye": "right",
        "camera_frame": "opencv_rdf",
        "depth_definition": "optical-axis Z depth",
        "depth_units": "meters_float32",
        "invalid_input_value": 0.0,
        "prediction_mask_policy": "unmasked model prediction saved; learned mask saved separately",
        "image_shape_hw": list(image_shape),
        "intrinsics_pixels": intrinsics,
        "intrinsics_normalized": normalized_k.tolist(),
        "input_depth_directory": str(raw_depth_dir),
        "output_depth_directory": str(depth_output_dir),
        "resolution_level": int(args.resolution_level),
        "mixed_precision": not args.no_fp16,
        "model_load_seconds": load_seconds,
        "sequence_inference_seconds": elapsed_seconds,
        "records": records,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {frame_count} metric depth maps to {depth_output_dir}", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
