#!/usr/bin/env python3
"""Generate a Hunyuan3D rigid mesh directly from one RGB frame and object mask."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vlm_sam2_recon.stages.hunyuan3d_client import Hunyuan3DClient, Hunyuan3DClientConfig  # noqa: E402
from vlm_sam2_recon.rigid_pipeline.common import update_stage_state  # noqa: E402


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Hunyuan3D mesh from a rigid object RGB/mask pair.")
    parser.add_argument("--workspace", type=Path, default=PROJECT_ROOT / "run_rigid_20260715_215524")
    parser.add_argument("--rgb", type=Path, default=None)
    parser.add_argument("--mask", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--target-id", default="target_rigid_object")
    parser.add_argument("--canvas-size", type=int, default=1024)
    parser.add_argument("--object-scale", type=float, default=0.88)
    parser.add_argument("--pad-ratio", type=float, default=0.08)
    parser.add_argument("--min-pad", type=int, default=24)
    parser.add_argument("--prefer-input", choices=["rgba", "white"], default="rgba")
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--poll-interval", type=float, default=None)
    parser.add_argument("--poll-timeout", type=float, default=None)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--prepare-only", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def load_mask(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        mask = np.load(path)
    else:
        mask = np.asarray(Image.open(path).convert("L")) > 127
    return np.asarray(mask, dtype=bool)


def bbox_from_mask(mask: np.ndarray, pad_ratio: float, min_pad: int) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        raise ValueError("Mask is empty.")
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    pad = max(int(min_pad), int(max(x1 - x0, y1 - y0) * float(pad_ratio)))
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(mask.shape[1], x1 + pad)
    y1 = min(mask.shape[0], y1 + pad)
    return x0, y0, x1, y1


def paste_on_square(
    rgb_crop: Image.Image,
    alpha_crop: Image.Image,
    canvas_size: int,
    object_scale: float,
    background: tuple[int, int, int, int],
) -> Image.Image:
    max_side = max(rgb_crop.size)
    target_side = max(1, int(canvas_size * object_scale))
    scale = min(1.0, target_side / float(max_side))
    new_size = (max(1, int(rgb_crop.width * scale)), max(1, int(rgb_crop.height * scale)))
    rgb_resized = rgb_crop.resize(new_size, Image.Resampling.LANCZOS)
    alpha_resized = alpha_crop.resize(new_size, Image.Resampling.NEAREST)
    canvas = Image.new("RGBA", (canvas_size, canvas_size), background)
    x = (canvas_size - new_size[0]) // 2
    y = (canvas_size - new_size[1]) // 2
    canvas.paste(rgb_resized.convert("RGBA"), (x, y), alpha_resized)
    return canvas


def prepare_inputs(args: argparse.Namespace) -> dict[str, object]:
    rgb_path = args.rgb.resolve()
    mask_path = args.mask.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb = Image.open(rgb_path).convert("RGB")
    mask = load_mask(mask_path)
    if mask.shape != (rgb.height, rgb.width):
        raise ValueError(f"Mask/image size mismatch: mask={mask.shape}, image={(rgb.height, rgb.width)}")

    x0, y0, x1, y1 = bbox_from_mask(mask, args.pad_ratio, args.min_pad)
    rgb_crop = rgb.crop((x0, y0, x1, y1))
    alpha_crop = Image.fromarray(mask[y0:y1, x0:x1].astype(np.uint8) * 255, mode="L")

    source_frame = out_dir / "source_frame.png"
    mask_copy = out_dir / "mask.png"
    crop_rgb = out_dir / "crop_rgb.png"
    cutout_rgba = out_dir / "cutout_rgba.png"
    cutout_white = out_dir / "cutout_white.png"
    cutout_black = out_dir / "cutout_black.png"
    rgb.save(source_frame)
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(mask_copy)
    rgb_crop.save(crop_rgb)
    paste_on_square(rgb_crop, alpha_crop, args.canvas_size, args.object_scale, (0, 0, 0, 0)).save(cutout_rgba)
    paste_on_square(rgb_crop, alpha_crop, args.canvas_size, args.object_scale, (255, 255, 255, 255)).convert("RGB").save(cutout_white)
    paste_on_square(rgb_crop, alpha_crop, args.canvas_size, args.object_scale, (0, 0, 0, 255)).convert("RGB").save(cutout_black)
    metadata = {
        "target_id": args.target_id,
        "rgb": str(rgb_path),
        "mask": str(mask_path),
        "bbox_xyxy": [x0, y0, x1, y1],
        "canvas_size": args.canvas_size,
        "object_scale": args.object_scale,
        "inputs": {
            "source_frame": str(source_frame),
            "mask": str(mask_copy),
            "crop_rgb": str(crop_rgb),
            "cutout_rgba": str(cutout_rgba),
            "cutout_white": str(cutout_white),
            "cutout_black": str(cutout_black),
        },
    }
    (out_dir / "hunyuan_input_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return metadata


def inspect_mesh(path: Path) -> dict[str, object]:
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = [geometry for geometry in loaded.geometry.values() if isinstance(geometry, trimesh.Trimesh)]
    elif isinstance(loaded, trimesh.Trimesh):
        geometries = [loaded]
    else:
        geometries = []
    face_count = sum(len(geometry.faces) for geometry in geometries)
    vertex_count = sum(len(geometry.vertices) for geometry in geometries)
    finite = all(np.isfinite(geometry.vertices).all() for geometry in geometries)
    bounds = np.vstack([geometry.bounds for geometry in geometries]) if geometries else np.empty((0, 3))
    extent = np.ptp(bounds, axis=0).tolist() if len(bounds) else None
    has_texture = any(
        getattr(getattr(geometry, "visual", None), "material", None) is not None
        for geometry in geometries
    )
    return {
        "geometry_count": len(geometries),
        "vertex_count": vertex_count,
        "face_count": face_count,
        "finite_vertices": finite,
        "canonical_extent": extent,
        "watertight": bool(geometries) and all(geometry.is_watertight for geometry in geometries),
        "has_material_or_texture": has_texture,
        "renderable": bool(geometries) and face_count > 0 and vertex_count >= 3 and finite,
    }


def main() -> int:
    args = parse_args()
    workspace = args.workspace.resolve()
    args.rgb = (args.rgb or workspace / "outputs/04_object_masks/mesh_prompt_frame0/rgb_no_hand.png").resolve()
    args.mask = (args.mask or workspace / "outputs/04_object_masks/mesh_prompt_frame0/object_mask.png").resolve()
    args.output_dir = (args.output_dir or workspace / "outputs/05_hunyuan_mesh").resolve()
    env_files = [args.env_file] if args.env_file else [PROJECT_ROOT / ".env.local", PROJECT_ROOT / ".env"]
    for env_file in env_files:
        if env_file is not None:
            load_env_file(env_file.expanduser().resolve())

    out_dir = args.output_dir.resolve()
    mesh_dir = out_dir / "whole"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(mesh_dir.glob(f"{args.target_id}.*"))
    if existing and not args.overwrite:
        print(f"Skip existing mesh: {existing[0]}")
        update_stage_state(
            workspace / "pipeline_state.json",
            "05_hunyuan_mesh",
            "completed",
            inputs=[str(args.rgb), str(args.mask)],
            outputs=[str(existing[0])],
            notes="Reused existing Hunyuan3D mesh.",
        )
        return 0

    metadata = prepare_inputs(args)
    if args.prepare_only:
        print(f"Prepared Hunyuan inputs in {out_dir}")
        update_stage_state(
            workspace / "pipeline_state.json",
            "05_hunyuan_mesh",
            "pending",
            inputs=[str(args.rgb), str(args.mask)],
            outputs=[str(out_dir)],
            notes="Hunyuan3D prompt prepared; API generation has not run yet.",
        )
        return 0

    client_cfg = Hunyuan3DClientConfig.from_env(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        poll_interval_sec=args.poll_interval,
        poll_timeout_sec=args.poll_timeout,
    )
    client_cfg.require_configured()
    client = Hunyuan3DClient(client_cfg)
    input_key = "cutout_rgba" if args.prefer_input == "rgba" else "cutout_white"
    input_path = Path(metadata["inputs"][input_key])
    print(f"[hunyuan3d] generating {args.target_id} from {input_path}")
    result = client.reconstruct(image_path=input_path, dest_path=mesh_dir / args.target_id)
    mesh_qc = inspect_mesh(result.mesh_path)
    run_record = {
        "target_id": args.target_id,
        "input_image": str(input_path),
        "mesh_path": str(result.mesh_path),
        "mesh_format": result.mesh_format,
        "task_id": result.task_id,
        "raw_response": result.raw_response,
        "mesh_quality": mesh_qc,
        "metadata": metadata,
        "api": {
            "base_url": client_cfg.base_url,
            "model": client_cfg.model,
            "enable_pbr": client_cfg.enable_pbr,
            "face_count": client_cfg.face_count,
        },
    }
    (out_dir / "hunyuan3d_run.json").write_text(
        json.dumps(run_record, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Hunyuan3D mesh: {result.mesh_path}")
    update_stage_state(
        workspace / "pipeline_state.json",
        "05_hunyuan_mesh",
        "completed" if mesh_qc["renderable"] else "needs_revision",
        inputs=[str(args.rgb), str(args.mask)],
        outputs=[str(result.mesh_path)],
        notes=(
            f"Hunyuan3D mesh generated with model {client_cfg.model} and passed renderability checks."
            if mesh_qc["renderable"]
            else "Hunyuan3D returned a mesh that failed renderability checks."
        ),
    )
    return 0 if mesh_qc["renderable"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
