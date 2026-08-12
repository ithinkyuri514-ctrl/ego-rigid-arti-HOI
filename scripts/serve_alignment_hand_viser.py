#!/usr/bin/env python3
"""Visualize camera-aligned laptop parts, depth points, and EgoForce hand."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import trimesh
import viser

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vlm_sam2_recon.stages.camera_alignment import depth_points_in_right_camera


DEFAULT_ALIGNMENT_DIR = Path("/code/vlm_sam2_recon/outputs/object_alignment/target_laptop/frame_000000")
DEFAULT_EGOFORCE_DIR = Path("/code/vlm_sam2_recon/outputs/egoforce_rgb_right")
DEFAULT_EXPORT_ROOT = Path("/code/3DVideo_2026-07-01-19-24-06-667_spatialmp4_export")

PART_COLORS = {
    "0": (220, 50, 65),
    "1": (75, 145, 210),
    "14": (220, 50, 65),
    "15": (75, 145, 210),
}
HAND_COLOR = (245, 190, 135)
ARM_COLOR = (150, 120, 95)
POINT_COLOR = (90, 230, 120)
JOINT_COLOR = (255, 40, 40)


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        geoms = [geom for geom in mesh.geometry.values() if isinstance(geom, trimesh.Trimesh)]
        if not geoms:
            raise ValueError(f"No mesh geometry found in {path}")
        mesh = trimesh.util.concatenate(geoms)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type from {path}: {type(mesh)!r}")
    return mesh


def add_mesh(scene: viser.SceneApi, name: str, mesh: trimesh.Trimesh, color: tuple[int, int, int], opacity: float) -> None:
    scene.add_mesh_simple(
        name,
        vertices=np.asarray(mesh.vertices, dtype=np.float32),
        faces=np.asarray(mesh.faces, dtype=np.uint32),
        color=color,
        opacity=opacity,
        wireframe=False,
        side="double",
    )


def load_point_cloud(path: Path) -> tuple[np.ndarray, np.ndarray]:
    cloud = trimesh.load(path, process=False)
    points = np.asarray(getattr(cloud, "vertices", []), dtype=np.float32)
    colors = getattr(getattr(cloud, "visual", None), "vertex_colors", None)
    if colors is None or len(colors) != len(points):
        colors = np.tile(np.asarray(POINT_COLOR, dtype=np.uint8), (len(points), 1))
    else:
        colors = np.asarray(colors)[:, :3].astype(np.uint8)
    return points, colors


def add_joint_axis(scene: viser.SceneApi, joint: dict, axis_length: float) -> None:
    origin = np.asarray(joint["origin_xyz"], dtype=np.float32)
    axis = np.asarray(joint["axis_xyz"], dtype=np.float32)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    p0 = origin - axis * axis_length * 0.5
    p1 = origin + axis * axis_length * 0.5
    scene.add_line_segments(
        f"/joints/{joint['name']}/axis",
        points=np.asarray([[p0, p1]], dtype=np.float32),
        colors=JOINT_COLOR,
        line_width=7,
    )
    scene.add_icosphere(
        f"/joints/{joint['name']}/origin",
        radius=0.012,
        color=JOINT_COLOR,
        position=origin,
        subdivisions=2,
    )
    limit = joint.get("limit", {})
    label = f"{joint['name']} {joint['type']} axis=[{axis[0]:.2f},{axis[1]:.2f},{axis[2]:.2f}]"
    if "lower" in limit and "upper" in limit:
        label += f" limit=({float(limit['lower']):.2f},{float(limit['upper']):.2f})"
    scene.add_label(
        f"/joints/{joint['name']}/label",
        text=label,
        position=origin + np.asarray([0.0, -0.035, 0.0], dtype=np.float32),
        font_size_mode="screen",
        font_screen_scale=0.8,
        anchor="bottom-center",
    )


def frame_name(frame: int) -> str:
    return f"{frame:06d}"


def add_rgb_reference(
    scene: viser.SceneApi,
    export_root: Path,
    view_frame: int,
    frustum_scale: float,
    rotate_180: bool,
) -> dict:
    meta = json.loads((export_root / "manifest.json").read_text(encoding="utf-8"))
    rgb_path = export_root / "rgb_right_png" / f"{frame_name(view_frame)}.png"
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    if rotate_180:
        rgb = np.ascontiguousarray(np.rot90(rgb, 2))
    intr = meta["rgb_intrinsics_right"]
    width = int(meta["rgb_width_per_eye"])
    height = int(meta["rgb_height_per_eye"])
    fy = float(intr["fy"])
    fov_y = float(2.0 * np.arctan2(height * 0.5, fy))
    aspect = float(width / height)

    scene.add_camera_frustum(
        "/input_camera",
        fov=fov_y,
        aspect=aspect,
        scale=float(frustum_scale),
        line_width=2.0,
        color=(30, 30, 30),
        image=rgb,
        format="auto",
        variant="wireframe",
    )
    return {
        "rgb_path": str(rgb_path),
        "frustum_scale": frustum_scale,
        "rotate_180": rotate_180,
        "fov_y_rad": fov_y,
        "aspect": aspect,
    }


def add_current_rgbd_point_cloud(
    scene: viser.SceneApi,
    export_root: Path,
    view_frame: int,
    convention: str,
    depth_min_m: float,
    depth_max_m: float,
    max_points: int,
    point_size: float,
) -> dict:
    meta = json.loads((export_root / "manifest.json").read_text(encoding="utf-8"))
    rgb_path = export_root / "rgb_right_png" / f"{frame_name(view_frame)}.png"
    depth_path = export_root / "depth_meters_npy" / f"{frame_name(view_frame)}.meters.npy"
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    depth_m = np.load(depth_path)
    raw_valid = np.isfinite(depth_m) & (depth_m > depth_min_m) & (depth_m < depth_max_m)

    points_right, u, v, inside = depth_points_in_right_camera(
        meta,
        depth_m,
        convention,
        depth_min_m=depth_min_m,
        depth_max_m=depth_max_m,
    )
    idx = np.flatnonzero(inside)
    if idx.size > max_points:
        rng = np.random.default_rng(17)
        idx = rng.choice(idx, size=max_points, replace=False)

    width = int(meta["rgb_width_per_eye"])
    height = int(meta["rgb_height_per_eye"])
    ui = np.clip(np.round(u[idx]).astype(np.int64), 0, width - 1)
    vi = np.clip(np.round(v[idx]).astype(np.int64), 0, height - 1)
    colors = np.ascontiguousarray(rgb[vi, ui, :3].astype(np.uint8))
    points = np.ascontiguousarray(points_right[idx].astype(np.float32))

    scene.add_point_cloud(
        "/current_rgbd/depth_points",
        points=points,
        colors=colors,
        point_size=point_size,
        point_shape="circle",
    )
    return {
        "rgb_path": str(rgb_path),
        "depth_path": str(depth_path),
        "convention": convention,
        "raw_depth_shape": list(depth_m.shape),
        "raw_valid_ratio": float(raw_valid.mean()),
        "projected_inside_points": int(inside.sum()),
        "displayed_points": int(len(points)),
        "depth_min_m": float(depth_min_m),
        "depth_max_m": float(depth_max_m),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve camera-aligned laptop and EgoForce hand in viser.")
    parser.add_argument("--alignment-dir", type=Path, default=DEFAULT_ALIGNMENT_DIR)
    parser.add_argument("--egoforce-dir", type=Path, default=DEFAULT_EGOFORCE_DIR)
    parser.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT_ROOT)
    parser.add_argument("--view-frame", type=int, default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8096)
    parser.add_argument("--laptop-opacity", type=float, default=0.86)
    parser.add_argument("--hand-opacity", type=float, default=0.95)
    parser.add_argument("--point-size", type=float, default=0.004)
    parser.add_argument("--axis-length", type=float, default=0.42)
    parser.add_argument("--rgb-frustum-scale", type=float, default=0.25)
    parser.add_argument("--rotate-rgb-180", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--show-current-rgbd", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--current-rgbd-max-points", type=int, default=55000)
    parser.add_argument("--current-rgbd-point-size", type=float, default=0.002)
    parser.add_argument(
        "--depth-convention",
        choices=["camera_to_rig", "rig_to_camera", "direct_same_camera"],
        default="camera_to_rig",
    )
    parser.add_argument("--depth-min-m", type=float, default=0.1)
    parser.add_argument("--depth-max-m", type=float, default=3.0)
    parser.add_argument("--no-rgb", action="store_true")
    parser.add_argument("--no-hand", action="store_true", help="Visualize the aligned object/RGB-D without requiring EgoForce hand meshes.")
    parser.add_argument("--show-arm", action="store_true")
    args = parser.parse_args()

    alignment_dir = args.alignment_dir.resolve()
    result = json.loads((alignment_dir / "alignment_result.json").read_text(encoding="utf-8"))
    view_frame = args.view_frame or result.get("view_frame")
    if view_frame is None:
        raise ValueError("No view frame supplied and alignment_result.json has no view_frame.")
    view_dir = alignment_dir / f"view_frame_{frame_name(view_frame)}"
    if not view_dir.exists():
        raise FileNotFoundError(f"Missing view dir: {view_dir}")

    server = viser.ViserServer(host=args.host, port=args.port)
    scene = server.scene
    scene.set_up_direction("-y")
    scene.add_frame("/camera_frame", axes_length=0.08, axes_radius=0.003)
    rgb_info = None
    if not args.no_rgb:
        rgb_info = add_rgb_reference(
            scene,
            args.export_root.resolve(),
            int(view_frame),
            args.rgb_frustum_scale,
            args.rotate_rgb_180,
        )
    current_rgbd_info = None
    if args.show_current_rgbd:
        current_rgbd_info = add_current_rgbd_point_cloud(
            scene,
            args.export_root.resolve(),
            int(view_frame),
            args.depth_convention,
            args.depth_min_m,
            args.depth_max_m,
            args.current_rgbd_max_points,
            args.current_rgbd_point_size,
        )

    all_bounds = []
    for mesh_path in sorted(view_dir.glob("part_*_view_camera.obj")):
        label = mesh_path.name.split("_")[1]
        mesh = load_mesh(mesh_path)
        color = PART_COLORS.get(label, (180, 180, 180))
        add_mesh(scene, f"/laptop/part_{label}", mesh, color=color, opacity=args.laptop_opacity)
        all_bounds.append(np.asarray(mesh.bounds, dtype=np.float32))
        center = np.asarray(mesh.bounds, dtype=np.float32).mean(axis=0)
        scene.add_label(
            f"/labels/laptop_part_{label}",
            text=f"part_{label}",
            position=center,
            font_size_mode="screen",
            font_screen_scale=0.75,
            anchor="center-center",
        )

    pc_path = view_dir / "observed_mask_pointcloud_view_camera.ply"
    if pc_path.exists():
        points, colors = load_point_cloud(pc_path)
        scene.add_point_cloud(
            "/observed_depth/laptop_mask_points",
            points=points,
            colors=colors,
            point_size=args.point_size,
            point_shape="circle",
        )

    part_pc_specs = [
        ("base", view_dir / "observed_base_pointcloud_view_camera.ply", PART_COLORS.get("14", (220, 50, 65))),
        ("screen", view_dir / "observed_screen_pointcloud_view_camera.ply", PART_COLORS.get("15", (75, 145, 210))),
    ]
    for name, path, fallback_color in part_pc_specs:
        if path.exists():
            points, colors = load_point_cloud(path)
            if len(colors) != len(points):
                colors = np.tile(np.asarray(fallback_color, dtype=np.uint8), (len(points), 1))
            scene.add_point_cloud(
                f"/observed_depth/{name}_mask_points",
                points=points,
                colors=colors,
                point_size=args.point_size * 1.15,
                point_shape="circle",
            )

    joints = json.loads((view_dir / "joint_view_camera.json").read_text(encoding="utf-8")).get("joints", [])
    for joint in joints:
        add_joint_axis(scene, joint, args.axis_length)

    hand_path = args.egoforce_dir / f"{frame_name(view_frame)}_left_hand.obj"
    if not args.no_hand:
        if not hand_path.exists():
            raise FileNotFoundError(f"Missing EgoForce hand mesh: {hand_path}")
        hand_mesh = load_mesh(hand_path)
        add_mesh(scene, "/hand/left_hand", hand_mesh, HAND_COLOR, args.hand_opacity)
        all_bounds.append(np.asarray(hand_mesh.bounds, dtype=np.float32))

        if args.show_arm:
            arm_path = args.egoforce_dir / f"{frame_name(view_frame)}_left_arm.obj"
            if arm_path.exists():
                add_mesh(scene, "/hand/left_arm", load_mesh(arm_path), ARM_COLOR, 0.55)

    if all_bounds:
        bounds = np.stack(all_bounds)
        xyz_min = bounds[:, 0, :].min(axis=0)
        xyz_max = bounds[:, 1, :].max(axis=0)
        center = (xyz_min + xyz_max) * 0.5
        dims = xyz_max - xyz_min
        scene.add_box(
            "/scene_bounds",
            dimensions=dims,
            color=(40, 40, 40),
            wireframe=True,
            opacity=0.22,
            position=center,
        )

    url = f"http://localhost:{args.port}" if args.host == "0.0.0.0" else f"http://{args.host}:{args.port}"
    print(f"Alignment + hand viser is running: {url}", flush=True)
    print(f"Alignment dir: {alignment_dir}", flush=True)
    print(f"View frame: {view_frame}", flush=True)
    if rgb_info:
        print(f"RGB frustum: {rgb_info}", flush=True)
    if current_rgbd_info:
        print(f"Current RGB-D point cloud: {current_rgbd_info}", flush=True)
    if args.no_hand:
        print("Hand mesh: disabled (--no-hand)", flush=True)
    else:
        print(f"Hand mesh: {hand_path}", flush=True)
    print("Laptop parts:", flush=True)
    for mesh_path in sorted(view_dir.glob("part_*_view_camera.obj")):
        print(f"  {mesh_path}", flush=True)
    print("Joints:", flush=True)
    for joint in joints:
        print(
            f"  {joint['name']} type={joint['type']} origin={joint['origin_xyz']} axis={joint['axis_xyz']}",
            flush=True,
        )

    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
