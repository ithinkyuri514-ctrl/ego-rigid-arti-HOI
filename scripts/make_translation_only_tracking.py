#!/usr/bin/env python3
"""Convert an SE(3) mesh trajectory to centroid-preserving translation only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"Mesh scene is empty: {path}")
        loaded = meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type {type(loaded)!r}: {path}")
    return loaded


def transform_point(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
    return transform[:3, :3] @ point + transform[:3, 3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-dir", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--motion-start-frame",
        type=int,
        default=None,
        help="Keep the object at its frame-0 pose before this inclusive motion frame.",
    )
    parser.add_argument(
        "--motion-end-frame",
        type=int,
        default=None,
        help="Freeze the object at this frame's pose afterward.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = args.tracking_dir.resolve()
    mesh_path = args.mesh.resolve()
    output_dir = args.output_dir.resolve()
    absolute_path = source_dir / "T_C0_from_aligned_mesh.npy"
    if not absolute_path.is_file():
        raise FileNotFoundError(absolute_path)

    absolute = np.load(absolute_path).astype(np.float64)
    if absolute.ndim != 3 or absolute.shape[1:] != (4, 4) or not len(absolute):
        raise ValueError(f"Malformed pose array: {absolute.shape}")
    centroid = np.asarray(load_mesh(mesh_path).centroid, dtype=np.float64)
    centers = np.stack([transform_point(pose, centroid) for pose in absolute])
    start = 0 if args.motion_start_frame is None else int(args.motion_start_frame)
    end = len(absolute) - 1 if args.motion_end_frame is None else int(args.motion_end_frame)
    if not 0 <= start <= end < len(absolute):
        raise ValueError(f"Invalid motion interval {start}..{end} for {len(absolute)} frames")
    if not np.isfinite(centers[[0, *range(start, end + 1)]]).all():
        raise ValueError("Non-finite mesh centers inside the retained motion interval")
    centers[:start] = centers[0]
    centers[end + 1 :] = centers[end]

    delta = np.repeat(np.eye(4, dtype=np.float64)[None], len(absolute), axis=0)
    delta[:, :3, 3] = centers - centers[0]
    absolute_translation_only = np.repeat(absolute[0][None], len(absolute), axis=0)
    absolute_translation_only[:, :3, 3] += centers - centers[0]

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "Delta_C0_object_motion.npy", delta)
    np.save(output_dir / "T_C0_from_aligned_mesh.npy", absolute_translation_only)
    for name in ("frame_indices.npy", "success.npy"):
        source = source_dir / name
        if source.is_file():
            np.save(output_dir / name, np.load(source))
    manifest = {
        "method": "mesh_centroid_translation_only",
        "source_tracking_dir": str(source_dir),
        "source_pose_array": str(absolute_path),
        "mesh": str(mesh_path),
        "mesh_centroid_local": centroid.tolist(),
        "frame_count": len(absolute),
        "rotation_policy": "fixed at frame 0",
        "translation_policy": "preserve per-frame transformed mesh centroid",
        "motion_interval_inclusive": [start, end],
        "outside_interval_policy": "hold frame 0 before motion; hold motion-end pose afterward",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
