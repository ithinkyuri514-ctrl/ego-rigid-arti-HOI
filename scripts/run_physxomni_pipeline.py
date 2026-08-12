#!/usr/bin/env python3
"""Dispatch reconstructed targets to PhysX-Omni or external rigid-mesh input."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_EXPORT_ROOT = Path("/code/3DVideo_2026-07-01-19-24-06-667_spatialmp4_export")
DEFAULT_PROJECT_ROOT = Path("/code/vlm_sam2_recon")
MESH_EXTENSIONS = {".obj", ".glb", ".gltf", ".ply", ".stl", ".off"}


@dataclass
class Target:
    object_id: str
    name_en: str
    object_class: str
    frame_index: int
    frame_path: Path
    raw: dict[str, Any]


def load_targets(path: Path) -> list[Target]:
    data = json.loads(path.read_text(encoding="utf-8"))
    targets = []
    for item in data["vlm_result"]["target_objects"]:
        keyframe = item["selected_keyframe"]
        targets.append(
            Target(
                object_id=item["object_id"],
                name_en=item.get("name_en", item["object_id"]),
                object_class=item["object_class_for_reconstruction"],
                frame_index=int(keyframe["frame_index"]),
                frame_path=Path(keyframe["frame_path"]),
                raw=item,
            )
        )
    return targets


def find_mask(mask_root: Path, target: Target) -> Path:
    target_dir = mask_root / target.object_id
    if not target_dir.exists():
        raise FileNotFoundError(f"Mask directory not found: {target_dir}")

    patterns = [
        f"{target.object_id}_frame_{target.frame_index}_interactive.mask.png",
        f"{target.object_id}_frame_{target.frame_index}.mask.png",
        f"{target.object_id}_frame_{target.frame_index}_interactive.mask.npy",
        f"{target.object_id}_frame_{target.frame_index}.mask.npy",
    ]
    for name in patterns:
        path = target_dir / name
        if path.exists():
            return path

    candidates = sorted(target_dir.glob(f"{target.object_id}_frame_{target.frame_index}*.mask.*"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No SAM2 mask found for {target.object_id} frame {target.frame_index}")


def load_mask(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        mask = np.load(path)
    else:
        mask = np.array(Image.open(path).convert("L"))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask > 0


def bbox_from_mask(mask: np.ndarray, pad_ratio: float, min_pad: int) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError("Mask is empty; cannot build PhysX-Omni condition image.")

    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    w, h = x1 - x0, y1 - y0
    pad = max(min_pad, int(max(w, h) * pad_ratio))
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

    rgb_resized = rgb_crop.resize(new_size, Image.LANCZOS)
    alpha_resized = alpha_crop.resize(new_size, Image.NEAREST)

    canvas = Image.new("RGBA", (canvas_size, canvas_size), background)
    x = (canvas_size - new_size[0]) // 2
    y = (canvas_size - new_size[1]) // 2
    canvas.paste(rgb_resized.convert("RGBA"), (x, y), alpha_resized)
    return canvas


def make_condition_images(
    target: Target,
    mask_path: Path,
    out_dir: Path,
    canvas_size: int,
    pad_ratio: float,
    min_pad: int,
    object_scale: float,
) -> dict[str, str]:
    image = Image.open(target.frame_path).convert("RGB")
    mask = load_mask(mask_path)
    if mask.shape[:2] != (image.height, image.width):
        raise ValueError(
            f"Mask/image size mismatch for {target.object_id}: "
            f"mask={mask.shape[:2]}, image={(image.height, image.width)}"
        )

    x0, y0, x1, y1 = bbox_from_mask(mask, pad_ratio=pad_ratio, min_pad=min_pad)
    rgb_crop = image.crop((x0, y0, x1, y1))
    alpha_crop = Image.fromarray((mask[y0:y1, x0:x1].astype(np.uint8) * 255), mode="L")

    transparent = paste_on_square(
        rgb_crop,
        alpha_crop,
        canvas_size=canvas_size,
        object_scale=object_scale,
        background=(0, 0, 0, 0),
    )
    white = paste_on_square(
        rgb_crop,
        alpha_crop,
        canvas_size=canvas_size,
        object_scale=object_scale,
        background=(255, 255, 255, 255),
    ).convert("RGB")

    out_dir.mkdir(parents=True, exist_ok=True)
    transparent_path = out_dir / f"{target.object_id}_cutout_transparent.png"
    white_path = out_dir / f"{target.object_id}_cutout_white.png"
    metadata_path = out_dir / f"{target.object_id}_condition_metadata.json"
    transparent.save(transparent_path)
    white.save(white_path)

    metadata = {
        "target_id": target.object_id,
        "source_frame": str(target.frame_path),
        "mask_path": str(mask_path),
        "bbox_xyxy": [x0, y0, x1, y1],
        "canvas_size": canvas_size,
        "object_scale": object_scale,
        "condition_image": str(white_path),
        "transparent_cutout": str(transparent_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "condition_image": str(white_path),
        "transparent_cutout": str(transparent_path),
        "metadata": str(metadata_path),
    }


def run_logged(cmd: list[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = " ".join(cmd)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ cd {cwd}\n$ {printable}\n")
        log.flush()
        print(f"$ {printable}")
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        code = proc.wait()
        if code != 0:
            raise subprocess.CalledProcessError(code, cmd)


def copytree_replace(src: Path, dst: Path, force: bool) -> None:
    if dst.exists():
        if not force:
            return
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def run_physx_for_articulated(target: Target, args: argparse.Namespace) -> dict[str, Any]:
    run_name = f"vlm_sam2_{target.object_id}"
    target_root = args.output_root / "articulated" / target.object_id
    input_dir = target_root / "inputs"
    basepath = target_root / "physx_base"
    run_dir = basepath / run_name
    log_path = args.output_root / "logs" / f"{run_name}.log"

    mask_path = find_mask(args.mask_root, target)
    condition = make_condition_images(
        target=target,
        mask_path=mask_path,
        out_dir=input_dir,
        canvas_size=args.canvas_size,
        pad_ratio=args.pad_ratio,
        min_pad=args.min_pad,
        object_scale=args.object_scale,
    )

    image_input_dir = target_root / "physx_input_images"
    image_input_dir.mkdir(parents=True, exist_ok=True)
    physx_input_image = image_input_dir / f"{run_name}.png"
    shutil.copy2(condition["condition_image"], physx_input_image)

    if args.force and run_dir.exists():
        shutil.rmtree(run_dir)

    if not (run_dir / "allind.npy").exists():
        repo_out = args.physx_repo / "ours_demo" / run_name
        if args.force and repo_out.exists():
            shutil.rmtree(repo_out)

        run_logged(
            [
                "conda",
                "run",
                "-n",
                args.conda_env,
                "python",
                "1vlm_demo.py",
                "--imagepath",
                str(image_input_dir),
                "--modelpath",
                str(args.physx_model_path),
            ],
            cwd=args.physx_repo,
            log_path=log_path,
        )
        if not repo_out.exists():
            raise FileNotFoundError(f"PhysX-Omni VLM output not found: {repo_out}")
        basepath.mkdir(parents=True, exist_ok=True)
        copytree_replace(repo_out, run_dir, force=True)

    if not any((run_dir / "objs").glob("*/*.obj")):
        run_logged(
            [
                "conda",
                "run",
                "-n",
                args.conda_env,
                "python",
                "2infer_geo.py",
                "--outputpath",
                str(basepath),
                "--index",
                "0",
                "--range",
                "1",
            ],
            cwd=args.physx_repo,
            log_path=log_path,
        )

    if not (run_dir / "basic.urdf").exists() or not (run_dir / "basic.xml").exists():
        run_logged(
            [
                "conda",
                "run",
                "-n",
                args.conda_env,
                "python",
                "3jsongen_update.py",
                "--basepath",
                str(basepath),
                "--voxel_define",
                "64",
            ],
            cwd=args.physx_repo,
            log_path=log_path,
        )

    summary = summarize_physx_output(target, run_dir, condition)
    summary_path = target_root / "physxomni_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def summarize_physx_output(target: Target, run_dir: Path, condition: dict[str, str]) -> dict[str, Any]:
    part_meshes = sorted(run_dir.glob("objs/*/*.obj"))
    voxel_parts = sorted(run_dir.glob("ind_*.npy"))
    coord_files = sorted(run_dir.glob("coord_*.txt"))
    info_json = run_dir / "basic_info.json"
    info_txt = run_dir / "basic_info.txt"

    return {
        "target_id": target.object_id,
        "name_en": target.name_en,
        "object_class": target.object_class,
        "physx_run_dir": str(run_dir),
        "condition": condition,
        "basic_info_json": str(info_json) if info_json.exists() else None,
        "basic_info_txt": str(info_txt) if info_txt.exists() else None,
        "urdf": str(run_dir / "basic.urdf") if (run_dir / "basic.urdf").exists() else None,
        "mjcf_xml": str(run_dir / "basic.xml") if (run_dir / "basic.xml").exists() else None,
        "part_meshes": [str(p) for p in part_meshes],
        "voxel_parts": [str(p) for p in voxel_parts],
        "coord_files": [str(p) for p in coord_files],
    }


def maybe_load_mesh_stats(path: Path) -> dict[str, Any]:
    stats: dict[str, Any] = {"path": str(path), "suffix": path.suffix}
    try:
        import trimesh

        mesh = trimesh.load(path, force="mesh")
        stats.update(
            {
                "vertices": int(len(getattr(mesh, "vertices", []))),
                "faces": int(len(getattr(mesh, "faces", []))),
                "bounds": np.asarray(mesh.bounds).tolist() if getattr(mesh, "bounds", None) is not None else None,
            }
        )
    except Exception as exc:  # optional validation only
        stats["load_error"] = repr(exc)
    return stats


def handle_rigid_target(target: Target, args: argparse.Namespace) -> dict[str, Any]:
    upload_dir = args.rigid_mesh_root / target.object_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    mesh_files = sorted(p for p in upload_dir.rglob("*") if p.suffix.lower() in MESH_EXTENSIONS)

    summary = {
        "target_id": target.object_id,
        "name_en": target.name_en,
        "object_class": target.object_class,
        "status": "external_mesh_found" if mesh_files else "waiting_for_external_trellis2_mesh",
        "expected_upload_dir": str(upload_dir),
        "accepted_extensions": sorted(MESH_EXTENSIONS),
        "meshes": [maybe_load_mesh_stats(p) for p in mesh_files],
    }
    summary_path = args.output_root / "rigid" / target.object_id / "rigid_mesh_manifest.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run articulated targets through PhysX-Omni and register rigid external meshes."
    )
    parser.add_argument(
        "--targets-json",
        type=Path,
        default=DEFAULT_EXPORT_ROOT / "qwen3vl8b_reconstruction_targets_right_keyframes.json",
    )
    parser.add_argument("--mask-root", type=Path, default=DEFAULT_PROJECT_ROOT / "outputs" / "sam2_masks")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_PROJECT_ROOT / "outputs" / "physxomni")
    parser.add_argument("--rigid-mesh-root", type=Path, default=DEFAULT_PROJECT_ROOT / "inputs" / "rigid_meshes")
    parser.add_argument("--physx-repo", type=Path, default=Path("/code/PhysX-Omni"))
    parser.add_argument("--physx-model-path", type=Path, default=Path("/code/PhysX-Omni/pretrain"))
    parser.add_argument("--conda-env", default="physx-anything")
    parser.add_argument("--canvas-size", type=int, default=1024)
    parser.add_argument("--object-scale", type=float, default=0.88)
    parser.add_argument("--pad-ratio", type=float, default=0.08)
    parser.add_argument("--min-pad", type=int, default=24)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-physx", action="store_true", help="Only create inputs/manifests.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.rigid_mesh_root.mkdir(parents=True, exist_ok=True)

    targets = load_targets(args.targets_json)
    manifest: dict[str, Any] = {
        "targets_json": str(args.targets_json),
        "output_root": str(args.output_root),
        "articulated": [],
        "rigid": [],
    }

    for target in targets:
        if target.object_class == "articulated":
            if args.skip_physx:
                mask_path = find_mask(args.mask_root, target)
                condition = make_condition_images(
                    target=target,
                    mask_path=mask_path,
                    out_dir=args.output_root / "articulated" / target.object_id / "inputs",
                    canvas_size=args.canvas_size,
                    pad_ratio=args.pad_ratio,
                    min_pad=args.min_pad,
                    object_scale=args.object_scale,
                )
                manifest["articulated"].append({"target_id": target.object_id, "condition": condition})
            else:
                manifest["articulated"].append(run_physx_for_articulated(target, args))
        elif target.object_class == "rigid":
            manifest["rigid"].append(handle_rigid_target(target, args))
        else:
            manifest.setdefault("unknown", []).append(target.raw)

    manifest_path = args.output_root / "reconstruction_dispatch_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
