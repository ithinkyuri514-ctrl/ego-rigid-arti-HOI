#!/usr/bin/env python3
"""Inspect the frame-0 aligned rigid mesh against its metric RGB-D data."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
import trimesh
import viser


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE = PROJECT_ROOT / "run_rigid_20260715_215524"


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = [geometry for geometry in loaded.geometry.values() if isinstance(geometry, trimesh.Trimesh)]
        if not geometries:
            raise ValueError(f"Mesh scene is empty: {path}")
        loaded = geometries[0].copy() if len(geometries) == 1 else trimesh.util.concatenate(geometries)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type from {path}: {type(loaded)!r}")
    return loaded


def backproject_rgbd(
    depth: np.ndarray,
    rgb: np.ndarray,
    intrinsics: dict[str, float],
    min_depth: float,
    max_depth: float,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    v, u = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
    z = depth[::stride, ::stride]
    valid = np.isfinite(z) & (z >= min_depth) & (z <= max_depth)
    z, u, v = z[valid], u[valid], v[valid]
    x = (u.astype(np.float32) - float(intrinsics["cx"])) * z / float(intrinsics["fx"])
    y = (v.astype(np.float32) - float(intrinsics["cy"])) * z / float(intrinsics["fy"])
    points = np.column_stack((x, y, z)).astype(np.float32)
    colors = rgb[v, u, :3].astype(np.uint8)
    return np.ascontiguousarray(points), np.ascontiguousarray(colors)


def load_cloud(path: Path) -> tuple[np.ndarray, np.ndarray]:
    cloud = trimesh.load(path, process=False)
    points = np.asarray(cloud.vertices, dtype=np.float32)
    colors = np.full((len(points), 3), (30, 210, 90), dtype=np.uint8)
    return points, colors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8097)
    parser.add_argument("--rgbd-stride", type=int, default=3)
    parser.add_argument("--depth-min-m", type=float, default=0.1)
    parser.add_argument("--depth-max-m", type=float, default=3.0)
    parser.add_argument("--mesh-opacity", type=float, default=0.82)
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    frame_dir = workspace / "outputs/07_alignment/frame_000000"
    rgb_path = workspace / "outputs/00_rgb_frames/right_rgb_png/000000.png"
    depth_path = workspace / "outputs/06_dense_depth/metric_depth_npy/000000.npy"
    camera_path = workspace / "outputs/00_rgb_frames/camera.json"
    mesh_path = frame_dir / "hunyuan_mesh_aligned_C0.glb"
    observed_path = frame_dir / "observed_object_pointcloud_C0.ply"
    report_path = frame_dir / "alignment_report.json"
    for path in (rgb_path, depth_path, camera_path, mesh_path, observed_path, report_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    camera = json.loads(camera_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    intrinsics = camera["rgb_intrinsics_right"]
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    depth = np.load(depth_path).astype(np.float32)
    points, colors = backproject_rgbd(
        depth,
        rgb,
        intrinsics,
        args.depth_min_m,
        args.depth_max_m,
        max(1, args.rgbd_stride),
    )
    observed_points, observed_colors = load_cloud(observed_path)
    mesh = load_mesh(mesh_path)

    server = viser.ViserServer(host=args.host, port=args.port)
    scene = server.scene
    scene.set_up_direction("-y")
    gui = server.gui
    show_mesh = gui.add_checkbox("Show mesh", initial_value=True)
    show_rgb = gui.add_checkbox("Show RGB", initial_value=True)
    show_rgbd = gui.add_checkbox("Show RGB-D", initial_value=True)
    show_object_points = gui.add_checkbox("Show object depth points", initial_value=True)
    show_camera = gui.add_checkbox("Show C0 axes", initial_value=True)

    camera_axes_handle = scene.add_frame("/C0", axes_length=0.08, axes_radius=0.003)
    rgbd_handle = scene.add_point_cloud(
        "/rgbd/frame_000000",
        points=points,
        colors=colors,
        point_size=0.002,
        point_shape="circle",
    )
    object_points_handle = scene.add_point_cloud(
        "/object/observed_depth_points",
        points=observed_points,
        colors=observed_colors,
        point_size=0.0035,
        point_shape="circle",
    )
    uv = getattr(mesh.visual, "uv", None)
    material = getattr(mesh.visual, "material", None)
    if uv is not None and material is not None:
        mesh_handle = scene.add_mesh_trimesh("/object/aligned_hunyuan_mesh", mesh=mesh)
    else:
        mesh_handle = scene.add_mesh_simple(
            "/object/aligned_hunyuan_mesh",
            vertices=np.asarray(mesh.vertices, dtype=np.float32),
            faces=np.asarray(mesh.faces, dtype=np.uint32),
            color=(245, 145, 35),
            opacity=args.mesh_opacity,
            side="double",
        )
    height, width = rgb.shape[:2]
    fov_y = 2.0 * np.arctan2(height * 0.5, float(intrinsics["fy"]))
    rgb_handle = scene.add_camera_frustum(
        "/C0/rgb",
        fov=float(fov_y),
        aspect=float(width / height),
        scale=0.12,
        image=rgb,
        color=(30, 30, 30),
        variant="wireframe",
    )

    def bind_visibility(control: object, handle: object) -> None:
        @control.on_update
        def _(_) -> None:
            handle.visible = bool(control.value)

    bind_visibility(show_mesh, mesh_handle)
    bind_visibility(show_rgb, rgb_handle)
    bind_visibility(show_rgbd, rgbd_handle)
    bind_visibility(show_object_points, object_points_handle)
    bind_visibility(show_camera, camera_axes_handle)

    diagnostics = report["alignment"]["projection_diagnostics"]
    print(
        json.dumps(
            {
                "coordinate_frame": "frame0_right_camera_opencv_rdf",
                "rgbd_points": len(points),
                "object_depth_points": len(observed_points),
                "mesh_vertices": len(mesh.vertices),
                "silhouette_iou": diagnostics["silhouette_iou_point_splat"],
                "depth_trimmed_rmse_m": diagnostics["depth_trimmed_rmse_m"],
                "url": f"http://localhost:{args.port}",
            },
            indent=2,
        )
    )
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
