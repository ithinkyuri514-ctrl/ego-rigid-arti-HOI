#!/usr/bin/env python3
"""Viser inspection of global frame-0 RGB-D, aligned meshes, and observed object clouds."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import trimesh
import viser
from PIL import Image


COLORS = [(230, 57, 70), (42, 157, 143), (244, 162, 97), (69, 123, 157), (131, 56, 236)]


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        geometries = [geometry for geometry in mesh.geometry.values() if isinstance(geometry, trimesh.Trimesh)]
        mesh = trimesh.util.concatenate(geometries)
    return mesh


def backproject(depth: np.ndarray, intrinsics: dict[str, float], stride: int = 5) -> np.ndarray:
    z = depth[::stride, ::stride]
    v, u = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
    valid = np.isfinite(z) & (z > 0.15) & (z < 4.0)
    z, u, v = z[valid], u[valid], v[valid]
    x = (u - float(intrinsics["cx"])) * z / float(intrinsics["fx"])
    y = (v - float(intrinsics["cy"])) * z / float(intrinsics["fy"])
    return np.column_stack([x, y, z]).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8098)
    parser.add_argument("--rgbd-stride", type=int, default=5)
    parser.add_argument("--mesh-opacity", type=float, default=0.82)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    camera = json.loads((workspace / "outputs/00_rgb_frames/camera.json").read_text(encoding="utf-8"))
    rgb_path = workspace / "outputs/00_rgb_frames/right_rgb_png/000000.png"
    depth_path = workspace / "outputs/06_dense_depth/metric_depth_npy/000000.npy"
    summary_path = workspace / "outputs/07_alignment/alignment_summary.json"
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    depth = np.load(depth_path).astype(np.float32)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    server = viser.ViserServer(host=args.host, port=args.port)
    scene = server.scene
    scene.set_up_direction("-y")
    gui = server.gui
    show_rgbd = gui.add_checkbox("Show frame-0 RGB-D", initial_value=True)
    show_camera = gui.add_checkbox("Show C0 camera", initial_value=True)
    rgbd_handle = scene.add_point_cloud(
        "/rgbd/frame0",
        points=backproject(depth, camera["rgb_intrinsics_right"], max(1, args.rgbd_stride)),
        colors=(160, 180, 190),
        point_size=0.003,
        point_shape="circle",
    )
    camera_handle = scene.add_frame("/C0_right", axes_length=0.12, axes_radius=0.004)
    height, width = rgb.shape[:2]
    fov_y = 2.0 * np.arctan2(height * 0.5, float(camera["rgb_intrinsics_right"]["fy"]))
    frustum = scene.add_camera_frustum(
        "/C0_right/rgb",
        fov=float(fov_y),
        aspect=float(width / height),
        scale=0.14,
        image=rgb,
        color=(40, 40, 40),
        variant="wireframe",
    )

    @show_rgbd.on_update
    def _(_) -> None:
        rgbd_handle.visible = bool(show_rgbd.value)

    @show_camera.on_update
    def _(_) -> None:
        camera_handle.visible = bool(show_camera.value)
        frustum.visible = bool(show_camera.value)

    for index, item in enumerate(summary["objects"]):
        object_id = item["object_id"]
        color = COLORS[index % len(COLORS)]
        mesh = load_mesh(Path(item["aligned_mesh"]))
        mesh_handle = scene.add_mesh_simple(
            f"/objects/{object_id}/aligned_mesh_C0",
            vertices=np.asarray(mesh.vertices, dtype=np.float32),
            faces=np.asarray(mesh.faces, dtype=np.uint32),
            color=color,
            opacity=args.mesh_opacity,
            side="double",
        )
        control = gui.add_checkbox(f"Show {object_id} mesh", initial_value=True)

        @control.on_update
        def _(_, handle=mesh_handle, checkbox=control) -> None:
            handle.visible = bool(checkbox.value)
        cloud_path = Path(item["observed_pointcloud"])
        cloud = trimesh.load(cloud_path, process=False)
        cloud_handle = scene.add_point_cloud(
            f"/objects/{object_id}/observed_depth",
            points=np.asarray(cloud.vertices, dtype=np.float32),
            colors=color,
            point_size=0.006,
            point_shape="circle",
        )
        cloud_control = gui.add_checkbox(f"Show {object_id} depth", initial_value=True)

        @cloud_control.on_update
        def _(_, handle=cloud_handle, checkbox=cloud_control) -> None:
            handle.visible = bool(checkbox.value)
        scene.add_label(f"/objects/{object_id}/label", text=object_id, position=np.asarray(mesh.bounds).mean(axis=0))

    print(f"Mixed frame-0 Viser: http://localhost:{args.port}", flush=True)
    print(f"Workspace: {workspace}", flush=True)
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
