#!/usr/bin/env python3
"""Prepare Trellis2 jobs and ingest externally generated meshes."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_EXPORT_ROOT = Path("/code/3DVideo_2026-07-01-19-24-06-667_spatialmp4_export")
DEFAULT_PROJECT_ROOT = Path("/code/vlm_sam2_recon")
MESH_EXTENSIONS = {".obj", ".glb", ".gltf", ".ply", ".stl", ".off"}
OPTIONAL_METADATA_FILES = {
    "trellis2_metadata.json",
    "parts.json",
    "articulation.json",
    "joints.json",
    "axes.json",
}


@dataclass
class Target:
    object_id: str
    name_en: str
    name_zh: str
    object_class: str
    frame_index: int
    frame_path: Path
    raw: dict[str, Any]


def load_targets(path: Path) -> list[Target]:
    data = json.loads(path.read_text(encoding="utf-8"))
    targets: list[Target] = []
    for item in data["vlm_result"]["target_objects"]:
        keyframe = item["selected_keyframe"]
        targets.append(
            Target(
                object_id=item["object_id"],
                name_en=item.get("name_en", item["object_id"]),
                name_zh=item.get("name_zh", ""),
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

    names = [
        f"{target.object_id}_frame_{target.frame_index}_interactive.mask.png",
        f"{target.object_id}_frame_{target.frame_index}.mask.png",
        f"{target.object_id}_frame_{target.frame_index}_interactive.mask.npy",
        f"{target.object_id}_frame_{target.frame_index}.mask.npy",
    ]
    for name in names:
        path = target_dir / name
        if path.exists():
            return path

    candidates = sorted(target_dir.glob(f"{target.object_id}_frame_{target.frame_index}*.mask.*"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No mask found for {target.object_id} frame {target.frame_index}")


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
        raise ValueError("Mask is empty.")

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


def write_upload_readme(upload_dir: Path, target: Target) -> None:
    readme = upload_dir / "README.md"
    if readme.exists():
        return
    readme.write_text(
        "\n".join(
            [
                f"# Trellis2 Mesh Upload: {target.object_id}",
                "",
                "Put meshes generated on the Trellis2 machine here, then rerun:",
                "",
                "```bash",
                "cd /code/vlm_sam2_recon",
                "python scripts/trellis2_mesh_interface.py --mode ingest",
                "```",
                "",
                "Accepted layout:",
                "",
                "```text",
                f"{upload_dir}/",
                "  whole/",
                f"    {target.object_id}.glb    # or .obj/.ply/.stl/.off",
                "  parts/                    # optional, for part-level meshes",
                "    part_00.glb",
                "    part_01.glb",
                "  trellis2_metadata.json     # optional",
                "  parts.json                 # optional",
                "  articulation.json          # optional axes/joints metadata",
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )


def prepare_target_job(target: Target, args: argparse.Namespace) -> dict[str, Any]:
    mask_path = find_mask(args.mask_root, target)
    image = Image.open(target.frame_path).convert("RGB")
    mask = load_mask(mask_path)
    if mask.shape[:2] != (image.height, image.width):
        raise ValueError(
            f"Mask/image size mismatch for {target.object_id}: "
            f"mask={mask.shape[:2]}, image={(image.height, image.width)}"
        )

    x0, y0, x1, y1 = bbox_from_mask(mask, pad_ratio=args.pad_ratio, min_pad=args.min_pad)
    rgb_crop = image.crop((x0, y0, x1, y1))
    alpha_crop = Image.fromarray((mask[y0:y1, x0:x1].astype(np.uint8) * 255), mode="L")

    job_dir = args.output_root / "jobs" / target.object_id
    job_dir.mkdir(parents=True, exist_ok=True)

    source_copy = job_dir / "source_frame.png"
    mask_copy = job_dir / "mask.png"
    crop_rgb_path = job_dir / "crop_rgb.png"
    cutout_rgba_path = job_dir / "cutout_rgba.png"
    cutout_white_path = job_dir / "cutout_white.png"
    cutout_black_path = job_dir / "cutout_black.png"
    metadata_path = job_dir / "trellis2_input_metadata.json"

    image.save(source_copy)
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(mask_copy)
    rgb_crop.save(crop_rgb_path)
    paste_on_square(
        rgb_crop,
        alpha_crop,
        canvas_size=args.canvas_size,
        object_scale=args.object_scale,
        background=(0, 0, 0, 0),
    ).save(cutout_rgba_path)
    paste_on_square(
        rgb_crop,
        alpha_crop,
        canvas_size=args.canvas_size,
        object_scale=args.object_scale,
        background=(255, 255, 255, 255),
    ).convert("RGB").save(cutout_white_path)
    paste_on_square(
        rgb_crop,
        alpha_crop,
        canvas_size=args.canvas_size,
        object_scale=args.object_scale,
        background=(0, 0, 0, 255),
    ).convert("RGB").save(cutout_black_path)

    upload_dir = args.upload_root / target.object_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    (upload_dir / "whole").mkdir(exist_ok=True)
    (upload_dir / "parts").mkdir(exist_ok=True)
    write_upload_readme(upload_dir, target)

    metadata = {
        "target_id": target.object_id,
        "name_en": target.name_en,
        "name_zh": target.name_zh,
        "object_class_for_reconstruction": target.object_class,
        "source_frame": str(target.frame_path),
        "source_frame_copy": str(source_copy),
        "frame_index": target.frame_index,
        "mask_path": str(mask_path),
        "mask_copy": str(mask_copy),
        "bbox_xyxy": [x0, y0, x1, y1],
        "canvas_size": args.canvas_size,
        "object_scale": args.object_scale,
        "trellis2_inputs": {
            "preferred": str(cutout_rgba_path),
            "cutout_rgba": str(cutout_rgba_path),
            "cutout_white": str(cutout_white_path),
            "cutout_black": str(cutout_black_path),
            "crop_rgb": str(crop_rgb_path),
        },
        "upload_contract": {
            "upload_dir": str(upload_dir),
            "whole_mesh_dir": str(upload_dir / "whole"),
            "part_mesh_dir": str(upload_dir / "parts"),
            "accepted_mesh_extensions": sorted(MESH_EXTENSIONS),
            "optional_metadata_files": sorted(OPTIONAL_METADATA_FILES),
        },
        "vlm_target": target.raw,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "target_id": target.object_id,
        "name_en": target.name_en,
        "object_class_for_reconstruction": target.object_class,
        "job_dir": str(job_dir),
        "metadata": str(metadata_path),
        "preferred_trellis2_input": str(cutout_rgba_path),
        "fallback_trellis2_input": str(cutout_white_path),
        "upload_dir": str(upload_dir),
    }


def maybe_mesh_stats(path: Path) -> dict[str, Any]:
    stats: dict[str, Any] = {"path": str(path), "suffix": path.suffix.lower()}
    try:
        import trimesh  # type: ignore

        mesh = trimesh.load(path, force="mesh", process=False)
        stats.update(
            {
                "vertices": int(len(getattr(mesh, "vertices", []))),
                "faces": int(len(getattr(mesh, "faces", []))),
                "bounds": np.asarray(mesh.bounds).tolist() if getattr(mesh, "bounds", None) is not None else None,
            }
        )
    except Exception as exc:
        stats["mesh_stats_error"] = repr(exc)
    return stats


def mesh_role(path: Path, upload_dir: Path) -> str:
    try:
        rel = path.relative_to(upload_dir)
    except ValueError:
        return "unknown"
    if rel.parts and rel.parts[0] == "whole":
        return "whole"
    if rel.parts and rel.parts[0] == "parts":
        return "part"
    return "whole" if path.stem.lower() in {"mesh", "model", "object", upload_dir.name.lower()} else "unknown"


def load_optional_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"load_error": repr(exc), "path": str(path)}


def ingest_target_meshes(target: Target, args: argparse.Namespace) -> dict[str, Any]:
    upload_dir = args.upload_root / target.object_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    (upload_dir / "whole").mkdir(exist_ok=True)
    (upload_dir / "parts").mkdir(exist_ok=True)
    write_upload_readme(upload_dir, target)

    mesh_files = sorted(
        p for p in upload_dir.rglob("*") if p.is_file() and p.suffix.lower() in MESH_EXTENSIONS
    )
    meshes = []
    for path in mesh_files:
        item = maybe_mesh_stats(path)
        item["role"] = mesh_role(path, upload_dir)
        item["relative_path"] = str(path.relative_to(upload_dir))
        meshes.append(item)

    optional_metadata = {
        name: load_optional_json(upload_dir / name)
        for name in sorted(OPTIONAL_METADATA_FILES)
        if (upload_dir / name).exists()
    }

    summary = {
        "target_id": target.object_id,
        "name_en": target.name_en,
        "object_class_for_reconstruction": target.object_class,
        "status": "mesh_uploaded" if meshes else "waiting_for_trellis2_upload",
        "upload_dir": str(upload_dir),
        "accepted_mesh_extensions": sorted(MESH_EXTENSIONS),
        "meshes": meshes,
        "optional_metadata": optional_metadata,
    }

    out_dir = args.output_root / "uploaded_meshes" / target.object_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "mesh_manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def write_project_readme(args: argparse.Namespace) -> None:
    path = args.output_root / "README.md"
    path.write_text(
        "\n".join(
            [
                "# Trellis2 Mesh Interface",
                "",
                "This folder contains prepared image/mask jobs for running Trellis2 on another machine.",
                "",
                "Run on this machine to prepare inputs:",
                "",
                "```bash",
                "cd /code/vlm_sam2_recon",
                "python scripts/trellis2_mesh_interface.py --mode prepare --make-zip",
                "```",
                "",
                "Copy `trellis2_jobs.zip` or `jobs/` to the Trellis2 machine. For each target, use:",
                "",
                "- `cutout_rgba.png` as preferred image input when Trellis2 supports alpha.",
                "- `cutout_white.png` as fallback image input.",
                "- `trellis2_input_metadata.json` for target id, frame, mask, bbox, and upload contract.",
                "",
                "After generating meshes, upload them back under:",
                "",
                "```text",
                "/code/vlm_sam2_recon/inputs/trellis2_meshes/<target_id>/",
                "  whole/<target_id>.glb",
                "  parts/*.glb                  # optional",
                "  trellis2_metadata.json        # optional",
                "  articulation.json             # optional",
                "```",
                "",
                "Then ingest uploaded meshes:",
                "",
                "```bash",
                "cd /code/vlm_sam2_recon",
                "python scripts/trellis2_mesh_interface.py --mode ingest",
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare and ingest a Trellis2 mesh reconstruction interface.")
    parser.add_argument("--mode", choices=["prepare", "ingest", "all"], default="all")
    parser.add_argument(
        "--targets-json",
        type=Path,
        default=DEFAULT_EXPORT_ROOT / "qwen3vl8b_reconstruction_targets_right_keyframes.json",
    )
    parser.add_argument("--mask-root", type=Path, default=DEFAULT_PROJECT_ROOT / "outputs" / "sam2_masks")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_PROJECT_ROOT / "outputs" / "trellis2_interface")
    parser.add_argument("--upload-root", type=Path, default=DEFAULT_PROJECT_ROOT / "inputs" / "trellis2_meshes")
    parser.add_argument("--canvas-size", type=int, default=1024)
    parser.add_argument("--object-scale", type=float, default=0.88)
    parser.add_argument("--pad-ratio", type=float, default=0.08)
    parser.add_argument("--min-pad", type=int, default=24)
    parser.add_argument("--make-zip", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.upload_root.mkdir(parents=True, exist_ok=True)
    write_project_readme(args)

    targets = load_targets(args.targets_json)
    manifest: dict[str, Any] = {
        "targets_json": str(args.targets_json),
        "output_root": str(args.output_root),
        "upload_root": str(args.upload_root),
        "mesh_backend": "trellis2_external",
        "local_trellis2_env_available": False,
        "jobs": [],
        "uploaded_meshes": [],
    }

    if args.mode in {"prepare", "all"}:
        manifest["jobs"] = [prepare_target_job(target, args) for target in targets]
        jobs_manifest = args.output_root / "trellis2_jobs.json"
        jobs_manifest.write_text(json.dumps(manifest["jobs"], indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved Trellis2 jobs: {jobs_manifest}")

        if args.make_zip:
            archive_base = args.output_root / "trellis2_jobs"
            archive_path = shutil.make_archive(str(archive_base), "zip", root_dir=args.output_root, base_dir="jobs")
            print(f"Saved Trellis2 job archive: {archive_path}")

    if args.mode in {"ingest", "all"}:
        manifest["uploaded_meshes"] = [ingest_target_meshes(target, args) for target in targets]
        uploaded_manifest = args.output_root / "trellis2_uploaded_mesh_manifest.json"
        uploaded_manifest.write_text(
            json.dumps(manifest["uploaded_meshes"], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Saved uploaded mesh manifest: {uploaded_manifest}")

    full_manifest = args.output_root / "trellis2_interface_manifest.json"
    full_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved interface manifest: {full_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
