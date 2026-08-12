#!/usr/bin/env python3
"""Visualize Stage 10 articulated motion with per-frame RGB-D track lifting."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
import viser


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE = PROJECT_ROOT / "run_mixed_20260802_142359_native36_left"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument(
        "--tracking-dir",
        type=Path,
        default=None,
        help="Stage 10 part directory containing T_C0_from_part.npy and raw_se3/.",
    )
    parser.add_argument("--dynamic-part", type=Path, default=None)
    parser.add_argument("--static-part", type=Path, default=None)
    parser.add_argument("--joint-json", type=Path, default=None)
    parser.add_argument("--poses-path", type=Path, default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8098)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--initial-frame", type=int, default=6)
    return parser.parse_args()


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        meshes = [geometry for geometry in mesh.geometry.values() if isinstance(geometry, trimesh.Trimesh)]
        mesh = trimesh.util.concatenate(meshes)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh at {path}: {type(mesh)!r}")
    return mesh


def quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
    xyzw = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    return xyzw[[3, 0, 1, 2]]


def transform_per_frame(points_ct: np.ndarray, transforms_c0_from_ct: np.ndarray) -> np.ndarray:
    points_c0 = np.full_like(points_ct, np.nan, dtype=np.float64)
    for frame, transform in enumerate(transforms_c0_from_ct):
        finite = np.isfinite(points_ct[frame]).all(axis=1)
        points_c0[frame, finite] = (
            points_ct[frame, finite] @ transform[:3, :3].T + transform[:3, 3]
        )
    return points_c0


def remove(handle: Any | None) -> None:
    if handle is not None:
        handle.remove()


def add_colored_mesh(
    scene: viser.SceneApi,
    name: str,
    mesh: trimesh.Trimesh,
    color: tuple[int, int, int],
) -> Any:
    return scene.add_mesh_simple(
        name,
        vertices=np.asarray(mesh.vertices, dtype=np.float32),
        faces=np.asarray(mesh.faces, dtype=np.uint32),
        color=color,
        opacity=0.9,
        flat_shading=False,
        side="double",
    )


def main() -> None:
    args = parse_args()
    workspace = args.workspace.resolve()
    tracking = (
        args.tracking_dir.resolve()
        if args.tracking_dir is not None
        else workspace / "outputs/10_articulate_tracking_axis_cotracker_corners/link_14"
    )
    part_root = workspace / "outputs/12_particulate/laptop"
    dynamic_path = (args.dynamic_part or part_root / "parts_C0/part_14.obj").resolve()
    static_path = (args.static_part or part_root / "parts_C0/part_15.obj").resolve()
    joint_path = (args.joint_json or part_root / "joint_axes_C0.json").resolve()
    poses_path = (
        args.poses_path.resolve()
        if args.poses_path is not None
        else workspace / "outputs/00_pose_refinement/poses_refined.npz"
    )

    dynamic_mesh = load_mesh(dynamic_path)
    static_mesh = load_mesh(static_path)
    part_poses = np.load(tracking / "T_C0_from_part.npy").astype(np.float64)
    angles = np.load(tracking / "joint_angles_rad.npy").astype(np.float64)
    tracking_manifest = json.loads(
        (tracking / "articulate_part_tracking_manifest.json").read_text(encoding="utf-8")
    )
    selected_by_start = tracking_manifest["selected_track_indices_by_interaction_start"]
    selected_track = int(next(iter(selected_by_start.values()))[0])
    points_c0 = np.load(tracking / "upper_left_track_3d_C0.npy").astype(np.float64)
    used_depth = np.load(tracking / "upper_left_track_used_depth_m.npy").astype(np.float64)
    depth_source = np.load(tracking / "upper_left_track_depth_source.npy").astype(np.uint8)
    confidence = np.load(tracking / "raw_se3/track_confidence.npy").astype(np.float64)
    transforms_c0_from_ct = np.load(poses_path)["T_C0_from_Ct"].astype(np.float64)
    joints = json.loads(joint_path.read_text(encoding="utf-8"))["joints"]
    joint = joints[0]
    origin = np.asarray(joint["origin_C0"], dtype=np.float64)
    axis = np.asarray(joint["axis_C0"], dtype=np.float64)
    axis /= np.linalg.norm(axis)

    server = viser.ViserServer(host=args.host, port=args.port)
    scene, gui = server.scene, server.gui
    scene.set_up_direction("-y")
    scene.add_frame("/C0", axes_length=0.08, axes_radius=0.003)
    add_colored_mesh(scene, "/laptop/static_link_15", static_mesh, (70, 145, 210))
    dynamic_handle = add_colored_mesh(
        scene, "/laptop/dynamic_link_14", dynamic_mesh, (230, 70, 70)
    )
    camera_handle = scene.add_frame("/current_camera_Ct", axes_length=0.06, axes_radius=0.002)
    axis_points = np.asarray(
        [[origin - 0.22 * axis, origin + 0.22 * axis]], dtype=np.float32
    )
    scene.add_line_segments(
        "/joint/axis", points=axis_points, colors=(40, 240, 120), line_width=5
    )
    scene.add_icosphere(
        "/joint/origin", radius=0.012, color=(40, 240, 120), position=origin
    )

    center = np.vstack([dynamic_mesh.vertices, static_mesh.vertices]).mean(axis=0)

    @server.on_client_connect
    def _frame_camera(client: viser.ClientHandle) -> None:
        client.camera.position = center + np.asarray([0.32, -0.24, -0.48])
        client.camera.look_at = center
        client.camera.up_direction = np.asarray([0.0, -1.0, 0.0])

    play = gui.add_checkbox("Play", initial_value=False)
    frame_slider = gui.add_slider(
        "Frame",
        min=0,
        max=len(part_poses) - 1,
        step=1,
        initial_value=int(np.clip(args.initial_frame, 0, len(part_poses) - 1)),
    )
    fps_slider = gui.add_slider("FPS", min=1.0, max=15.0, step=1.0, initial_value=args.fps)
    show_tracks = gui.add_checkbox("Per-frame depth tracks", initial_value=True)
    show_trails = gui.add_checkbox("Track trails", initial_value=True)
    show_camera = gui.add_checkbox("Camera pose", initial_value=True)
    status = gui.add_markdown("")
    previous = gui.add_button("Previous")
    following = gui.add_button("Next")

    current_frame = int(frame_slider.value)
    point_handle: Any | None = None
    trail_handle: Any | None = None

    def show_frame(frame: int) -> None:
        nonlocal current_frame, point_handle, trail_handle
        current_frame = int(np.clip(frame, 0, len(part_poses) - 1))
        transform = part_poses[current_frame]
        dynamic_handle.position = transform[:3, 3]
        dynamic_handle.wxyz = quaternion_wxyz(transform)
        camera_transform = transforms_c0_from_ct[current_frame]
        camera_handle.position = camera_transform[:3, 3]
        camera_handle.wxyz = quaternion_wxyz(camera_transform)
        camera_handle.visible = bool(show_camera.value)

        remove(point_handle)
        remove(trail_handle)
        point_handle = None
        trail_handle = None
        valid = np.zeros(points_c0.shape[1], dtype=bool)
        valid[selected_track] = (
            np.isfinite(points_c0[current_frame, selected_track]).all()
            and confidence[current_frame, selected_track] >= 0.5
        )
        if bool(show_tracks.value) and valid.any():
            point_handle = scene.add_point_cloud(
                "/tracks/current_depth_lifts",
                points=np.ascontiguousarray(points_c0[current_frame, valid].astype(np.float32)),
                colors=np.tile(np.asarray([[255, 220, 0]], dtype=np.uint8), (int(valid.sum()), 1)),
                point_size=0.014,
                point_shape="circle",
            )
        if bool(show_trails.value):
            segments = []
            for index in range(1, current_frame + 1):
                pair = points_c0[index - 1 : index + 1, selected_track]
                if np.isfinite(pair).all():
                    segments.append(pair)
            if segments:
                trail_handle = scene.add_line_segments(
                    "/tracks/per_frame_depth_trails",
                    points=np.asarray(segments, dtype=np.float32),
                    colors=(255, 170, 20),
                    line_width=3,
                )
        depth = used_depth[current_frame, selected_track]
        depth_text = "n/a" if not np.isfinite(depth) else f"{depth:.3f} m"
        source_text = {
            1: "current frame",
            2: "previous-frame fallback",
        }.get(int(depth_source[current_frame, selected_track]), "unavailable")
        status.content = (
            f"**Frame {current_frame}**  ·  angle **{np.rad2deg(angles[current_frame]):.2f}°**  "
            f"· upper-left track **{selected_track}**  · depth **{depth_text}** ({source_text})  "
            f"· confidence **{confidence[current_frame, selected_track]:.3f}**"
        )

    @frame_slider.on_update
    def _(_: Any) -> None:
        show_frame(int(frame_slider.value))

    @show_tracks.on_update
    def _(_: Any) -> None:
        show_frame(current_frame)

    @show_trails.on_update
    def _(_: Any) -> None:
        show_frame(current_frame)

    @show_camera.on_update
    def _(_: Any) -> None:
        show_frame(current_frame)

    @previous.on_click
    def _(_: Any) -> None:
        frame_slider.value = max(0, current_frame - 1)
        show_frame(int(frame_slider.value))

    @following.on_click
    def _(_: Any) -> None:
        frame_slider.value = min(len(part_poses) - 1, current_frame + 1)
        show_frame(int(frame_slider.value))

    show_frame(current_frame)
    url = f"http://localhost:{args.port}" if args.host == "0.0.0.0" else f"http://{args.host}:{args.port}"
    print(f"Stage 10 articulated viser is running: {url}", flush=True)
    print(f"Tracking directory: {tracking}", flush=True)

    try:
        while True:
            if bool(play.value):
                time.sleep(1.0 / max(float(fps_slider.value), 1e-6))
                frame_slider.value = (current_frame + 1) % len(part_poses)
                show_frame(int(frame_slider.value))
            else:
                time.sleep(0.05)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
