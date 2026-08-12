#!/usr/bin/env python3
"""Diagnose right-hand EgoForce outputs against RGB, pose, masks, and depth."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import trimesh
import viser


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE = PROJECT_ROOT / "run_mixed_20260728_131115"
RIGHT_INDEX = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8101)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--initial-frame", type=int, default=15)
    parser.add_argument("--depth-stride", type=int, default=2)
    parser.add_argument("--max-hand-depth-points", type=int, default=25000)
    parser.add_argument("--frustum-scale", type=float, default=0.12)
    return parser.parse_args()


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = [geom for geom in loaded.geometry.values() if isinstance(geom, trimesh.Trimesh)]
        if not geometries:
            raise ValueError(f"Mesh scene is empty: {path}")
        loaded = geometries[0].copy() if len(geometries) == 1 else trimesh.util.concatenate(geometries)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(type(loaded))
    return loaded


def add_mesh(
    scene: viser.SceneApi,
    name: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    color: tuple[int, int, int],
    opacity: float = 1.0,
) -> Any:
    return scene.add_mesh_simple(
        name,
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.uint32),
        color=color,
        opacity=float(opacity),
        side="double",
    )


def remove_handle(handle: Any | None) -> None:
    if handle is not None and hasattr(handle, "remove"):
        handle.remove()


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return (np.asarray(points) @ transform[:3, :3].T + transform[:3, 3]).astype(np.float32)


def wxyz(rotation: np.ndarray) -> np.ndarray:
    xyzw = Rotation.from_matrix(rotation).as_quat()
    return np.asarray([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float32)


def resize_preview(image: np.ndarray, max_width: int = 640) -> np.ndarray:
    if image.shape[1] <= max_width:
        return image
    scale = float(max_width) / float(image.shape[1])
    height = max(1, int(round(image.shape[0] * scale)))
    return np.asarray(Image.fromarray(image).resize((max_width, height), Image.Resampling.LANCZOS))


def projected_uv(points_ct: np.ndarray, intrinsics: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_ct, dtype=np.float64)
    z = points[:, 2]
    valid = np.isfinite(points).all(axis=1) & (z > 1e-8)
    uv = np.full((len(points), 2), np.nan, dtype=np.float64)
    uv[valid, 0] = float(intrinsics["fx"]) * points[valid, 0] / z[valid] + float(intrinsics["cx"])
    uv[valid, 1] = float(intrinsics["fy"]) * points[valid, 1] / z[valid] + float(intrinsics["cy"])
    return uv, valid


def make_projection_overlay(
    rgb: np.ndarray,
    mask: np.ndarray,
    vertices_ct: np.ndarray | None,
    joints_ct: np.ndarray | None,
    intrinsics: dict[str, float],
) -> np.ndarray:
    image = Image.fromarray(rgb).convert("RGBA")
    if mask.any():
        tint = Image.new("RGBA", image.size, (30, 220, 90, 0))
        alpha = Image.fromarray(mask.astype(np.uint8) * 88)
        tint.putalpha(alpha)
        image = Image.alpha_composite(image, tint)
    draw = ImageDraw.Draw(image)
    if vertices_ct is not None:
        uv, valid = projected_uv(vertices_ct, intrinsics)
        height, width = rgb.shape[:2]
        inside = valid & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
        for x, y in uv[inside][::4]:
            draw.ellipse((x - 1.5, y - 1.5, x + 1.5, y + 1.5), fill=(255, 170, 35, 210))
    if joints_ct is not None:
        uv, valid = projected_uv(joints_ct, intrinsics)
        height, width = rgb.shape[:2]
        inside = valid & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
        for x, y in uv[inside]:
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(40, 120, 255, 235), outline=(255, 255, 255, 255))
    return np.asarray(image.convert("RGB"))


def hand_depth_points_c0(
    depth: np.ndarray,
    rgb: np.ndarray,
    mask: np.ndarray,
    intrinsics: dict[str, float],
    transform: np.ndarray,
    stride: int,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    stride = max(1, int(stride))
    sampled_mask = mask[::stride, ::stride]
    sampled_depth = depth[::stride, ::stride].astype(np.float32)
    yy, xx = np.nonzero(sampled_mask & np.isfinite(sampled_depth) & (sampled_depth > 0.1) & (sampled_depth < 2.0))
    point_count = int(len(xx))
    if point_count == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8), 0
    if point_count > max_points:
        rng = np.random.default_rng(20260728 + point_count)
        keep = rng.choice(point_count, size=int(max_points), replace=False)
        yy, xx = yy[keep], xx[keep]
    z = sampled_depth[yy, xx]
    full_x = xx * stride
    full_y = yy * stride
    x = (full_x - float(intrinsics["cx"])) * z / float(intrinsics["fx"])
    y = (full_y - float(intrinsics["cy"])) * z / float(intrinsics["fy"])
    points_ct = np.column_stack([x, y, z]).astype(np.float32)
    colors = rgb[np.clip(full_y, 0, rgb.shape[0] - 1), np.clip(full_x, 0, rgb.shape[1] - 1)].astype(np.uint8)
    return transform_points(points_ct, transform), np.ascontiguousarray(colors), point_count


def main() -> None:
    args = parse_args()
    workspace = args.workspace.resolve()
    rgb_paths = sorted((workspace / "outputs/00_rgb_frames/right_rgb_png").glob("*.png"))
    depth_paths = sorted((workspace / "outputs/06_dense_depth/metric_depth_npy").glob("*.npy"))
    mask_dir = workspace / "outputs/02_hand_masks/objects/right_hand"
    if not mask_dir.is_dir():
        mask_dir = workspace / "outputs/02_hand_masks/combined"
    mask_paths = sorted(mask_dir.glob("*.png"))
    raw_dir = workspace / "outputs/09_egoforce/raw_Ct"
    manifest_path = workspace / "outputs/09_egoforce/dynamic_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = manifest["frames"]
    camera = json.loads((workspace / "outputs/00_rgb_frames/camera.json").read_text(encoding="utf-8"))
    intrinsics = camera["rgb_intrinsics_right"]
    poses = np.load(workspace / "outputs/00_rgb_frames/poses.npz")["T_C0_from_Ct"]
    frame_count = min(len(rgb_paths), len(depth_paths), len(mask_paths), len(frames), len(poses))
    if frame_count == 0:
        raise FileNotFoundError(f"No diagnostic frames found under {workspace}")

    server = viser.ViserServer(host=args.host, port=args.port)
    scene, gui = server.scene, server.gui
    scene.set_up_direction("-y")
    scene.add_frame("/C0", axes_length=0.08, axes_radius=0.003)
    ct_frame = scene.add_frame("/Ct", axes_length=0.055, axes_radius=0.002)

    play = gui.add_checkbox("Play", initial_value=False)
    initial_frame = int(np.clip(args.initial_frame, 0, frame_count - 1))
    frame_control = gui.add_slider("Frame", min=0, max=frame_count - 1, step=1, initial_value=initial_frame)
    fps_control = gui.add_slider("FPS", min=1.0, max=30.0, step=1.0, initial_value=float(args.fps))
    show_rgb = gui.add_checkbox("Show RGB", initial_value=False)
    show_overlay = gui.add_checkbox("Show mask + projection overlay", initial_value=True)
    show_egoforce_detection = gui.add_checkbox("Show EgoForce 2D detections", initial_value=False)
    show_raw_c0 = gui.add_checkbox("Show selected EgoForce hand in C0", initial_value=False)
    show_raw_ct = gui.add_checkbox("Show raw EgoForce hand under Ct frame", initial_value=False)
    show_accepted_c0 = gui.add_checkbox("Show corrected manifest C0 OBJ", initial_value=True)
    show_arm = gui.add_checkbox("Show right arm", initial_value=False)
    show_joints = gui.add_checkbox("Show joints", initial_value=True)
    show_hand_depth = gui.add_checkbox("Show hand depth points", initial_value=True)
    show_full_depth = gui.add_checkbox("Show sparse full RGB-D", initial_value=False)
    status = gui.add_markdown("Loading")

    handles: dict[str, Any] = {}
    current_frame = initial_frame

    def clear_handles() -> None:
        for handle in list(handles.values()):
            remove_handle(handle)
        handles.clear()

    def remember(key: str, handle: Any | None) -> None:
        if handle is not None:
            handles[key] = handle

    def add_frustum(name: str, image: np.ndarray, pose: np.ndarray, visible: bool) -> None:
        height, width = image.shape[:2]
        fov_y = 2.0 * np.arctan2(height * 0.5, float(intrinsics["fy"]))
        remember(
            name,
            scene.add_camera_frustum(
                f"/image/{name}",
                fov=float(fov_y),
                aspect=float(width / height),
                scale=float(args.frustum_scale),
                image=resize_preview(image),
                wxyz=wxyz(pose[:3, :3]),
                position=pose[:3, 3].astype(np.float32),
                visible=visible,
            ),
        )

    def show_frame(frame: int) -> None:
        nonlocal current_frame
        frame = int(np.clip(frame, 0, frame_count - 1))
        current_frame = frame
        clear_handles()

        rgb = np.asarray(Image.open(rgb_paths[frame]).convert("RGB"))
        depth = np.load(depth_paths[frame]).astype(np.float32)
        mask = np.asarray(Image.open(mask_paths[frame]).convert("L")) > 127
        pose = poses[frame]
        ct_frame.wxyz = wxyz(pose[:3, :3])
        ct_frame.position = pose[:3, 3].astype(np.float32)

        raw_path = raw_dir / f"{frame:06d}_egoforce_meshes.npz"
        raw_visible = False
        raw_vertices_ct: np.ndarray | None = None
        raw_faces: np.ndarray | None = None
        raw_joints_ct: np.ndarray | None = None
        raw_arm_vertices_ct: np.ndarray | None = None
        raw_arm_faces: np.ndarray | None = None
        if raw_path.is_file():
            with np.load(raw_path) as raw:
                visible = np.asarray(raw["visible_hand"], dtype=bool).reshape(2)
                selected_index = int(frames[frame].get("selected_raw_side_index", RIGHT_INDEX))
                if selected_index < 0 or selected_index > 1:
                    selected_index = RIGHT_INDEX
                raw_visible = bool(visible[selected_index])
                if raw_visible:
                    raw_vertices_ct = raw["hand_vertices"][selected_index].astype(np.float32)
                    raw_faces_key = "right_hand_faces" if selected_index == RIGHT_INDEX else "left_hand_faces"
                    raw_faces = raw[raw_faces_key].astype(np.uint32)
                    raw_joints_ct = raw["hand_joints"][selected_index].astype(np.float32)
                    raw_arm_vertices_ct = raw["arm_vertices"][selected_index].astype(np.float32)
                    raw_arm_faces = raw["arm_faces"].astype(np.uint32)

        overlay = make_projection_overlay(rgb, mask, raw_vertices_ct, raw_joints_ct, intrinsics)
        add_frustum("rgb", rgb, pose, bool(show_rgb.value))
        add_frustum("mask_projection_overlay", overlay, pose, bool(show_overlay.value))
        detection_path = raw_dir / f"{frame:06d}_detections.jpg"
        if detection_path.is_file():
            detection = np.asarray(Image.open(detection_path).convert("RGB"))
            add_frustum("egoforce_2d_detections", detection, pose, bool(show_egoforce_detection.value))

        accepted = frames[frame].get("right_hand_C0")
        raw_c0_vertices: np.ndarray | None = None
        if raw_visible and raw_vertices_ct is not None and raw_faces is not None:
            raw_c0_vertices = transform_points(raw_vertices_ct, pose)
            if bool(show_raw_c0.value):
                remember(
                    "raw_c0",
                    add_mesh(scene, "/right_hand/raw_egoforce_C0", raw_c0_vertices, raw_faces, (245, 150, 35), 0.7),
                )
            if bool(show_raw_ct.value):
                remember(
                    "raw_ct",
                    add_mesh(scene, "/Ct/right_hand/raw_egoforce_Ct", raw_vertices_ct, raw_faces, (180, 90, 245), 0.45),
                )
            if bool(show_joints.value) and raw_joints_ct is not None:
                joints_c0 = transform_points(raw_joints_ct, pose)
                remember(
                    "joints",
                    scene.add_point_cloud(
                        "/right_hand/raw_joints_C0",
                        points=joints_c0,
                        colors=np.full((len(joints_c0), 3), (40, 120, 255), dtype=np.uint8),
                        point_size=0.009,
                        point_shape="circle",
                    ),
                )
            if bool(show_arm.value) and raw_arm_vertices_ct is not None and raw_arm_faces is not None:
                remember(
                    "raw_arm",
                    add_mesh(
                        scene,
                        "/right_arm/raw_egoforce_C0",
                        transform_points(raw_arm_vertices_ct, pose),
                        raw_arm_faces,
                        (130, 105, 80),
                        0.45,
                    ),
                )

        if bool(show_accepted_c0.value) and accepted:
            accepted_mesh = load_mesh(Path(accepted))
            remember(
                "accepted_c0",
                add_mesh(
                    scene,
                    "/right_hand/sam2_accepted_C0_obj",
                    np.asarray(accepted_mesh.vertices, dtype=np.float32),
                    np.asarray(accepted_mesh.faces, dtype=np.uint32),
                    (45, 205, 120),
                    0.65,
                ),
            )

        hand_points, hand_colors, hand_depth_count = hand_depth_points_c0(
            depth,
            rgb,
            mask,
            intrinsics,
            pose,
            args.depth_stride,
            args.max_hand_depth_points,
        )
        if bool(show_hand_depth.value) and len(hand_points):
            remember(
                "hand_depth",
                scene.add_point_cloud(
                    "/depth/right_hand_mask_points_C0",
                    points=hand_points,
                    colors=hand_colors,
                    point_size=0.003,
                    point_shape="circle",
                ),
            )

        if bool(show_full_depth.value):
            full_points, full_colors, _ = hand_depth_points_c0(
                depth,
                rgb,
                np.ones_like(mask, dtype=bool),
                intrinsics,
                pose,
                max(args.depth_stride * 5, 8),
                45000,
            )
            remember(
                "full_depth",
                scene.add_point_cloud(
                    "/depth/sparse_rgbd_C0",
                    points=full_points,
                    colors=full_colors,
                    point_size=0.0015,
                    point_shape="circle",
                ),
            )

        median_mm: float | None = None
        p90_mm: float | None = None
        if raw_c0_vertices is not None and len(hand_points):
            distances, _ = cKDTree(raw_c0_vertices).query(hand_points, k=1)
            median_mm = float(np.median(distances) * 1000.0)
            p90_mm = float(np.percentile(distances, 90) * 1000.0)

        accepted_text = "yes" if accepted else "no"
        selected_side = frames[frame].get("selected_raw_side", "right")
        visible_text = "yes" if raw_visible else "no"
        dist_text = "n/a" if median_mm is None else f"{median_mm:.1f}/{p90_mm:.1f} mm"
        status.content = (
            f"Frame {frame} | selected raw side {selected_side} | selected visible {visible_text} | "
            f"SAM2 accepted {accepted_text} | "
            f"mask area {int(mask.sum())} px | hand depth pts {hand_depth_count} | "
            f"depth-to-EgoForce median/p90 {dist_text}"
        )

    def refresh() -> None:
        show_frame(current_frame)

    @frame_control.on_update
    def _(_: Any) -> None:
        show_frame(int(frame_control.value))

    for control in (
        show_rgb,
        show_overlay,
        show_egoforce_detection,
        show_raw_c0,
        show_raw_ct,
        show_accepted_c0,
        show_arm,
        show_joints,
        show_hand_depth,
        show_full_depth,
    ):
        @control.on_update
        def _(_: Any) -> None:
            refresh()

    show_frame(initial_frame)
    print(
        json.dumps(
            {
                "frame_count": frame_count,
                "right_hand_only": True,
                "workspace": str(workspace),
                "raw_Ct_dir": str(raw_dir),
                "hand_mask_dir": str(mask_dir),
                "coordinate_frame": "frame0_right_camera_opencv_rdf",
                "default_layers": [
                    "mask + projection overlay",
                    "corrected manifest right hand in C0",
                    "raw right-hand joints",
                    "right-hand mask depth points",
                ],
                "url": f"http://localhost:{args.port}",
            },
            indent=2,
        ),
        flush=True,
    )

    last_tick = time.time()
    while True:
        now = time.time()
        if bool(play.value) and now - last_tick >= 1.0 / max(float(fps_control.value), 1.0):
            frame_control.value = (int(frame_control.value) + 1) % frame_count
            last_tick = now
        time.sleep(0.01)


if __name__ == "__main__":
    main()
