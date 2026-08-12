#!/usr/bin/env python3
"""Reset Stage 11 and build only its watertight collision/depth proxy."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt
import trimesh


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(type(mesh))
    return mesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=PROJECT_ROOT / "run_rigid_20260715_215524",
    )
    parser.add_argument("--pitch", type=float, default=0.0025)
    parser.add_argument("--padding-m", type=float, default=0.06)
    parser.add_argument("--source-mesh", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = args.workspace.resolve()
    output = workspace / "outputs/11_contact_optimization"
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    source_path = (
        args.source_mesh
        or workspace / "outputs/07_alignment/frame_000000/hunyuan_mesh_aligned_C0.glb"
    ).resolve()
    mesh = load_mesh(source_path)
    voxel = mesh.voxelized(pitch=args.pitch).fill()
    occupancy = np.asarray(voxel.matrix, dtype=bool)
    padding = max(2, int(np.ceil(args.padding_m / args.pitch)))
    occupancy = np.pad(occupancy, padding, mode="constant", constant_values=False)
    outside = distance_transform_edt(~occupancy)
    inside = distance_transform_edt(occupancy)
    sdf = np.where(occupancy, -(inside - 0.5), outside - 0.5).astype(np.float32) * args.pitch
    origin = np.asarray(voxel.transform[:3, 3], dtype=np.float64) - padding * args.pitch
    np.savez_compressed(
        output / "collision_sdf_C0.npz",
        sdf_xyz=sdf,
        origin_xyz=origin,
        pitch_m=np.asarray(args.pitch),
        axis_order=np.asarray("xyz"),
    )
    proxy = voxel.marching_cubes
    proxy.apply_transform(voxel.transform)
    proxy_path = output / "collision_proxy_C0.obj"
    proxy.export(proxy_path)
    manifest = {
        "schema_version": 2,
        "source_mesh": str(source_path),
        "source_watertight": bool(mesh.is_watertight),
        "collision_proxy": str(proxy_path),
        "collision_proxy_watertight": bool(proxy.is_watertight),
        "voxel_pitch_m": args.pitch,
        "sdf_shape_xyz": list(sdf.shape),
        "coordinate_frame": "frame0_right_camera_opencv_rdf",
        "reset_policy": "Previous nearest-surface Stage 11 outputs were deleted before this v2 run.",
    }
    (output / "collision_proxy_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
