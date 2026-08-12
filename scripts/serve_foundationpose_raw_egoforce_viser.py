#!/usr/bin/env python3
"""Visualize independent FoundationPose estimates with raw EgoForce geometry."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import trimesh
import viser


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE = PROJECT_ROOT / "run_mixed_20260728_203734"


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = [
            geometry
            for geometry in loaded.geometry.values()
            if isinstance(geometry, trimesh.Trimesh)
        ]
        if not geometries:
            raise ValueError(f"Mesh scene is empty: {path}")
        loaded = (
            geometries[0].copy()
            if len(geometries) == 1
            else trimesh.util.concatenate(geometries)
        )
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type {type(loaded)!r}: {path}")
    return loaded


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return np.asarray(points) @ transform[:3, :3].T + transform[:3, 3]


def wxyz(rotation: np.ndarray) -> np.ndarray:
    xyzw = Rotation.from_matrix(rotation).as_quat()
    return np.asarray([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float32)


def resolve_aligned_mesh(workspace: Path, object_id: str) -> Path:
    summary_path = workspace / "outputs/07_alignment/alignment_summary.json"
    summary = read_json(summary_path)
    matches = [
        record
        for record in summary.get("objects", [])
        if record.get("object_id") == object_id
    ]
    if len(matches) != 1:
        raise KeyError(
            f"Expected exactly one Stage07 mesh for {object_id!r}, found {len(matches)}"
        )
    path = Path(matches[0]["aligned_mesh"])
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def resolve_object_mask_dir(
    workspace: Path, object_id: str, expected_count: int
) -> Path:
    candidates = [
        workspace / "outputs/04_object_masks" / object_id / "objects" / object_id,
        workspace
        / "outputs/02_sam2_frame0_masks"
        / "propagated"
        / "objects"
        / object_id,
    ]
    for candidate in candidates:
        if len(list(candidate.glob("*.png"))) == expected_count:
            return candidate
    counts = {str(path): len(list(path.glob("*.png"))) for path in candidates}
    raise FileNotFoundError(
        f"No complete {object_id!r} mask sequence for {expected_count} frames: {counts}"
    )


def add_mesh_handle(
    scene: viser.SceneApi,
    name: str,
    mesh: trimesh.Trimesh,
    color: tuple[int, int, int],
    visible: bool,
) -> Any:
    return scene.add_mesh_simple(
        name,
        vertices=np.asarray(mesh.vertices, dtype=np.float32),
        faces=np.asarray(mesh.faces, dtype=np.uint32),
        color=color,
        side="double",
        visible=visible,
    )


def backproject(
    depth: np.ndarray,
    rgb: np.ndarray,
    intrinsics: dict[str, float],
    transform: np.ndarray,
    stride: int,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    v, u = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
    z = depth[::stride, ::stride]
    valid = np.isfinite(z) & (z >= 0.1) & (z <= 3.0)
    if mask is not None:
        valid &= mask[::stride, ::stride]
    z, u, v = z[valid], u[valid], v[valid]
    x = (u - float(intrinsics["cx"])) * z / float(intrinsics["fx"])
    y = (v - float(intrinsics["cy"])) * z / float(intrinsics["fy"])
    points_ct = np.column_stack([x, y, z])
    points_c0 = transform_points(points_ct, transform).astype(np.float32)
    colors = rgb[v, u].astype(np.uint8)
    return np.ascontiguousarray(points_c0), np.ascontiguousarray(colors)


def resize_preview(rgb: np.ndarray, max_width: int = 480) -> np.ndarray:
    if rgb.shape[1] <= max_width:
        return rgb
    scale = float(max_width) / rgb.shape[1]
    height = max(1, int(round(rgb.shape[0] * scale)))
    return np.asarray(
        Image.fromarray(rgb).resize((max_width, height), Image.Resampling.LANCZOS)
    )


def load_foundationpose_timeline(
    workspace: Path, object_id: str, timeline_frame_count: int
) -> tuple[np.ndarray, int, list[dict[str, Any]], dict[str, Any]]:
    output_dir = workspace / "outputs/08_foundationpose_independent" / object_id
    manifest = read_json(output_dir / "manifest.json")
    poses = np.load(output_dir / "T_C0_from_aligned_mesh.npy").astype(np.float64)
    frame_indices = np.load(output_dir / "frame_indices.npy").astype(np.int64)
    success = np.load(output_dir / "success.npy").astype(bool)
    if poses.shape != (len(frame_indices), 4, 4) or len(success) != len(frame_indices):
        raise ValueError(
            f"Malformed FoundationPose outputs: poses={poses.shape}, "
            f"indices={frame_indices.shape}, success={success.shape}"
        )
    expected_indices = np.arange(len(frame_indices), dtype=np.int64)
    if not np.array_equal(frame_indices, expected_indices):
        raise ValueError(
            "This viewer requires FoundationPose estimates beginning at frame 0 "
            "with no skipped frames"
        )
    if not success.all() or not np.isfinite(poses).all():
        failed = frame_indices[~success].tolist()
        raise ValueError(f"FoundationPose has failed/non-finite frames: {failed}")
    if not 0 < len(poses) <= timeline_frame_count:
        raise ValueError(
            f"Invalid FoundationPose length {len(poses)} for {timeline_frame_count} frames"
        )

    timeline = np.repeat(poses[-1][None], timeline_frame_count, axis=0)
    timeline[: len(poses)] = poses
    diagnostics_path = output_dir / "frame_diagnostics.jsonl"
    diagnostics = [
        json.loads(line)
        for line in diagnostics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(diagnostics) != len(poses):
        raise ValueError(
            f"FoundationPose diagnostics count {len(diagnostics)} != pose count {len(poses)}"
        )
    return timeline, len(poses) - 1, diagnostics, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--object-id", default="bottle")
    parser.add_argument("--articulated-object-id", default="microwave")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8098)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--initial-frame", type=int, default=49)
    parser.add_argument("--rgbd-stride", type=int, default=8)
    parser.add_argument("--object-stride", type=int, default=3)
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    rgb_paths = sorted((workspace / "outputs/00_rgb_frames/right_rgb_png").glob("*.png"))
    depth_paths = sorted(
        (workspace / "outputs/06_dense_depth/metric_depth_npy").glob("*.npy")
    )
    timeline_frame_count = len(rgb_paths)
    if timeline_frame_count == 0:
        raise FileNotFoundError("No Stage00 right RGB frames")
    if len(depth_paths) != timeline_frame_count:
        raise ValueError(
            f"RGB/depth count mismatch: {timeline_frame_count}/{len(depth_paths)}"
        )

    camera = read_json(workspace / "outputs/00_rgb_frames/camera.json")
    intrinsics = camera["rgb_intrinsics_right"]
    pose_data = np.load(workspace / "outputs/00_rgb_frames/poses.npz")
    camera_poses = pose_data["T_C0_from_Ct"].astype(np.float64)
    if camera_poses.shape != (timeline_frame_count, 4, 4):
        raise ValueError(
            f"Camera pose shape {camera_poses.shape} does not match timeline"
        )

    object_poses, foundationpose_end, diagnostics, foundationpose_manifest = (
        load_foundationpose_timeline(workspace, args.object_id, timeline_frame_count)
    )
    object_mesh_path = resolve_aligned_mesh(workspace, args.object_id)
    articulated_mesh_path = resolve_aligned_mesh(
        workspace, args.articulated_object_id
    )
    object_mask_dir = resolve_object_mask_dir(
        workspace, args.object_id, timeline_frame_count
    )
    mask_paths = sorted(object_mask_dir.glob("*.png"))

    hand_manifest_path = workspace / "outputs/09_egoforce/dynamic_manifest.json"
    hand_manifest = read_json(hand_manifest_path)
    hand_frames = hand_manifest["frames"]
    if hand_manifest.get("coordinate_frame") != "frame0_right_camera_opencv_rdf":
        raise ValueError("EgoForce geometry must already be pose-compensated into C0")
    if len(hand_frames) != timeline_frame_count:
        raise ValueError(
            f"RGB/EgoForce count mismatch: {timeline_frame_count}/{len(hand_frames)}"
        )
    missing_hands = [
        index
        for index, record in enumerate(hand_frames)
        if not record.get("right_hand_C0") or not record.get("right_arm_C0")
    ]
    if missing_hands:
        raise ValueError(f"Missing raw right-hand/arm EgoForce geometry: {missing_hands}")

    object_mesh = load_mesh(object_mesh_path)
    articulated_mesh = load_mesh(articulated_mesh_path)
    initial_frame = int(np.clip(args.initial_frame, 0, timeline_frame_count - 1))
    focus_center = transform_points(
        np.asarray(object_mesh.centroid, dtype=np.float64)[None],
        object_poses[initial_frame],
    )[0]

    server = viser.ViserServer(host=args.host, port=args.port)
    scene, gui = server.scene, server.gui
    scene.set_up_direction("-y")

    @server.on_client_connect
    def _frame_camera(client: viser.ClientHandle) -> None:
        client.camera.position = focus_center + np.array([0.20, -0.14, -0.34])
        client.camera.look_at = focus_center
        client.camera.up_direction = np.array([0.0, -1.0, 0.0])

    play = gui.add_checkbox("Play", initial_value=False)
    frame_control = gui.add_slider(
        "Frame",
        min=0,
        max=timeline_frame_count - 1,
        step=1,
        initial_value=initial_frame,
    )
    fps_control = gui.add_slider(
        "FPS", min=1.0, max=30.0, step=1.0, initial_value=args.fps
    )
    show_c0 = gui.add_checkbox("Show C0 axes", initial_value=True)
    show_object_mesh = gui.add_checkbox(
        f"Show {args.object_id} mesh", initial_value=True
    )
    show_articulated_mesh = gui.add_checkbox(
        f"Show static {args.articulated_object_id}", initial_value=True
    )
    show_rgb = gui.add_checkbox("Show RGB frustum", initial_value=False)
    show_rgbd = gui.add_checkbox("Show RGB-D", initial_value=False)
    show_object_depth = gui.add_checkbox("Show object depth", initial_value=True)
    show_right_hand = gui.add_checkbox("Show raw right hand", initial_value=True)
    show_right_arm = gui.add_checkbox("Show raw right arm", initial_value=False)
    status = gui.add_markdown(f"Frame {initial_frame}")

    c0_handle = scene.add_frame(
        "/C0", axes_length=0.08, axes_radius=0.003, visible=bool(show_c0.value)
    )
    object_mesh_handle = add_mesh_handle(
        scene, f"/{args.object_id}/foundationpose_mesh", object_mesh, (245, 145, 35), False
    )
    articulated_mesh_handle = add_mesh_handle(
        scene,
        f"/{args.articulated_object_id}/static_aligned_mesh",
        articulated_mesh,
        (145, 195, 235),
        bool(show_articulated_mesh.value),
    )

    frame_handles: list[dict[str, Any]] = []
    print(f"Preloading {timeline_frame_count} raw-observation frames...", flush=True)
    for frame in range(timeline_frame_count):
        prefix = f"/frames/{frame:06d}"
        handles: dict[str, Any] = {}
        rgb = np.asarray(Image.open(rgb_paths[frame]).convert("RGB"))
        depth = np.load(depth_paths[frame]).astype(np.float32)
        mask = np.asarray(Image.open(mask_paths[frame]).convert("L")) > 127
        if rgb.shape[:2] != depth.shape or mask.shape != depth.shape:
            raise ValueError(
                f"Frame {frame} RGB/depth/mask shape mismatch: "
                f"{rgb.shape}/{depth.shape}/{mask.shape}"
            )
        camera_pose = camera_poses[frame]
        height, width = rgb.shape[:2]
        fov_y = 2.0 * np.arctan2(height * 0.5, float(intrinsics["fy"]))
        handles["rgb"] = scene.add_camera_frustum(
            f"{prefix}/rgb",
            fov=float(fov_y),
            aspect=float(width / height),
            scale=0.12,
            image=resize_preview(rgb),
            color=(30, 30, 30),
            wxyz=wxyz(camera_pose[:3, :3]),
            position=camera_pose[:3, 3].astype(np.float32),
            visible=False,
        )
        rgbd_points, rgbd_colors = backproject(
            depth,
            rgb,
            intrinsics,
            camera_pose,
            max(1, args.rgbd_stride),
        )
        handles["rgbd"] = scene.add_point_cloud(
            f"{prefix}/rgbd",
            points=rgbd_points,
            colors=rgbd_colors,
            point_size=0.003,
            point_shape="circle",
            visible=False,
        )
        object_points, object_colors = backproject(
            depth,
            rgb,
            intrinsics,
            camera_pose,
            max(1, args.object_stride),
            mask,
        )
        if len(object_colors):
            object_colors = np.full_like(object_colors, (35, 225, 90))
        handles["object_depth"] = scene.add_point_cloud(
            f"{prefix}/object_depth",
            points=object_points,
            colors=object_colors,
            point_size=0.0035,
            point_shape="circle",
            visible=False,
        )

        hand_record = hand_frames[frame]
        right_hand = load_mesh(Path(hand_record["right_hand_C0"]))
        right_arm = load_mesh(Path(hand_record["right_arm_C0"]))
        handles["right_hand"] = add_mesh_handle(
            scene,
            f"{prefix}/raw_egoforce/right_hand_C0",
            right_hand,
            (65, 135, 245),
            False,
        )
        handles["right_arm"] = add_mesh_handle(
            scene,
            f"{prefix}/raw_egoforce/right_arm_C0",
            right_arm,
            (45, 95, 200),
            False,
        )
        frame_handles.append(handles)
        if (frame + 1) % 10 == 0 or frame + 1 == timeline_frame_count:
            print(f"Preloaded {frame + 1}/{timeline_frame_count} frames", flush=True)

    current_frame = initial_frame

    def desired_visibility(key: str) -> bool:
        controls = {
            "rgb": show_rgb,
            "rgbd": show_rgbd,
            "object_depth": show_object_depth,
            "right_hand": show_right_hand,
            "right_arm": show_right_arm,
        }
        return bool(controls[key].value)

    def update_status(frame: int) -> None:
        if frame <= foundationpose_end:
            diagnostic = diagnostics[frame]
            score = diagnostic.get("foundationpose_score")
            score_text = "n/a" if score is None else f"{float(score):.3f}"
            pose_text = f"independent FoundationPose, score {score_text}"
        else:
            pose_text = f"held FoundationPose pose from frame {foundationpose_end}"
        selected_side = hand_frames[frame].get("selected_raw_side", "right")
        status.content = (
            f"Frame {frame}/{timeline_frame_count - 1} | {pose_text} | "
            f"raw EgoForce selected side {selected_side} | no temporal/contact optimization"
        )

    def show_frame(frame: int) -> None:
        nonlocal current_frame
        frame = int(np.clip(frame, 0, timeline_frame_count - 1))
        object_pose = object_poses[frame]
        with server.atomic():
            for handle in frame_handles[current_frame].values():
                handle.visible = False
            object_mesh_handle.wxyz = wxyz(object_pose[:3, :3])
            object_mesh_handle.position = object_pose[:3, 3].astype(np.float32)
            object_mesh_handle.visible = bool(show_object_mesh.value)
            articulated_mesh_handle.visible = bool(show_articulated_mesh.value)
            for key, handle in frame_handles[frame].items():
                handle.visible = desired_visibility(key)
        current_frame = frame
        update_status(frame)

    @frame_control.on_update
    def _(_) -> None:
        show_frame(int(frame_control.value))

    def bind(control: Any, key: str) -> None:
        @control.on_update
        def _(_) -> None:
            frame_handles[current_frame][key].visible = desired_visibility(key)

    bind(show_rgb, "rgb")
    bind(show_rgbd, "rgbd")
    bind(show_object_depth, "object_depth")
    bind(show_right_hand, "right_hand")
    bind(show_right_arm, "right_arm")

    @show_c0.on_update
    def _(_) -> None:
        c0_handle.visible = bool(show_c0.value)

    @show_object_mesh.on_update
    def _(_) -> None:
        object_mesh_handle.visible = bool(show_object_mesh.value)

    @show_articulated_mesh.on_update
    def _(_) -> None:
        articulated_mesh_handle.visible = bool(show_articulated_mesh.value)

    show_frame(initial_frame)
    print(
        json.dumps(
            {
                "frame_count": timeline_frame_count,
                "foundationpose_frames": foundationpose_end + 1,
                "foundationpose_policy": foundationpose_manifest["pose_policy"],
                "post_rigid_policy": f"hold frame {foundationpose_end} pose",
                "object_id": args.object_id,
                "object_mesh": str(object_mesh_path),
                "object_mask_dir": str(object_mask_dir),
                "articulated_object_id": args.articulated_object_id,
                "articulated_mesh": str(articulated_mesh_path),
                "hand_source": str(hand_manifest_path),
                "hand_policy": "raw pose-compensated EgoForce C0 geometry only",
                "coordinate_frame": "frame0_right_camera_opencv_rdf",
                "scene_up": "-y",
                "url": f"http://localhost:{args.port}",
            },
            indent=2,
        ),
        flush=True,
    )

    last_tick = time.time()
    while True:
        now = time.time()
        if play.value and now - last_tick >= 1.0 / max(float(fps_control.value), 1.0):
            frame_control.value = (int(frame_control.value) + 1) % timeline_frame_count
            last_tick = now
        time.sleep(0.01)


if __name__ == "__main__":
    main()
