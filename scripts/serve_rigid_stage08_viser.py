#!/usr/bin/env python3
"""Play rigid object and pose-compensated EgoForce hands with RGB-D."""

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
DEFAULT_WORKSPACE = PROJECT_ROOT / "run_rigid_20260715_215524"


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = [item for item in loaded.geometry.values() if isinstance(item, trimesh.Trimesh)]
        if not geometries:
            raise ValueError(f"Mesh scene is empty: {path}")
        loaded = geometries[0].copy() if len(geometries) == 1 else trimesh.util.concatenate(geometries)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(type(loaded))
    return loaded


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return np.asarray(points) @ transform[:3, :3].T + transform[:3, 3]


def wxyz(rotation: np.ndarray) -> np.ndarray:
    xyzw = Rotation.from_matrix(rotation).as_quat()
    return np.asarray([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float32)


def closest_rotation(linear: np.ndarray) -> np.ndarray:
    """Remove uniform Sim3 scale before showing a transform as an axis frame."""
    u, _, vt = np.linalg.svd(np.asarray(linear, dtype=np.float64))
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return rotation


def resolve_object_inputs(workspace: Path, object_id: str) -> tuple[Path, Path]:
    alignment_summary = workspace / "outputs/07_alignment/alignment_summary.json"
    if alignment_summary.is_file():
        summary = json.loads(alignment_summary.read_text(encoding="utf-8"))
        matches = [item for item in summary.get("objects", []) if item.get("object_id") == object_id]
        if len(matches) != 1:
            raise KeyError(f"Expected one aligned object {object_id!r}, found {len(matches)}")
        mesh_path = Path(matches[0]["aligned_mesh"])
        stage04_mask_dir = workspace / "outputs/04_object_masks" / object_id / "objects" / object_id
        mask_dir = stage04_mask_dir if stage04_mask_dir.is_dir() else (
            workspace / "outputs/02_sam2_frame0_masks/propagated/objects" / object_id
        )
        return mesh_path, mask_dir
    return (
        workspace / "outputs/07_alignment/frame_000000/hunyuan_mesh_aligned_C0.glb",
        workspace / "outputs/04_object_masks/combined",
    )


def resolve_aligned_mesh_from_summary(workspace: Path, object_id: str) -> Path | None:
    alignment_summary = workspace / "outputs/07_alignment/alignment_summary.json"
    if not alignment_summary.is_file():
        return None
    summary = json.loads(alignment_summary.read_text(encoding="utf-8"))
    matches = [item for item in summary.get("objects", []) if item.get("object_id") == object_id]
    if not matches:
        return None
    return Path(matches[0]["aligned_mesh"])


def add_mesh_handle(
    scene: viser.SceneApi,
    name: str,
    mesh: trimesh.Trimesh,
    color: tuple[int, int, int],
    visible: bool,
) -> Any:
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.uint32)
    textured_mesh = getattr(mesh.visual, "uv", None) is not None and getattr(
        mesh.visual, "material", None
    ) is not None
    if textured_mesh:
        return scene.add_mesh_trimesh(name, mesh=mesh, visible=visible)
    return scene.add_mesh_simple(
        name,
        vertices=vertices,
        faces=faces,
        color=color,
        side="double",
        visible=visible,
    )


def backproject(
    depth: np.ndarray,
    rgb: np.ndarray,
    intrinsics: dict,
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
    points = transform_points(np.column_stack([x, y, z]), transform).astype(np.float32)
    colors = rgb[v, u].astype(np.uint8)
    return np.ascontiguousarray(points), np.ascontiguousarray(colors)


def resize_preview(rgb: np.ndarray, max_width: int = 640) -> np.ndarray:
    if rgb.shape[1] <= max_width:
        return rgb
    scale = float(max_width) / rgb.shape[1]
    height = max(1, int(round(rgb.shape[0] * scale)))
    return np.asarray(Image.fromarray(rgb).resize((max_width, height), Image.Resampling.LANCZOS))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8097)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--rgbd-stride", type=int, default=4)
    parser.add_argument("--object-stride", type=int, default=2)
    parser.add_argument("--initial-frame", type=int, default=24)
    parser.add_argument("--object-id", default="cup")
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    rgb_paths = sorted((workspace / "outputs/00_rgb_frames/right_rgb_png").glob("*.png"))
    depth_paths = sorted((workspace / "outputs/06_dense_depth/metric_depth_npy").glob("*.npy"))
    mesh_path, mask_dir = resolve_object_inputs(workspace, args.object_id)
    microwave_mesh_path = resolve_aligned_mesh_from_summary(workspace, "microwave")
    mask_paths = sorted(mask_dir.glob("*.png"))
    tracking = workspace / "outputs/08_tracking"
    hand_manifest_path = workspace / "outputs/09_egoforce/dynamic_manifest.json"
    contact_manifest_path = workspace / "outputs/11_contact_optimization/dynamic_manifest.json"
    camera = json.loads((workspace / "outputs/00_rgb_frames/camera.json").read_text(encoding="utf-8"))
    intrinsics = camera["rgb_intrinsics_right"]
    pose_data = np.load(workspace / "outputs/00_rgb_frames/poses.npz")
    camera_poses = pose_data["T_C0_from_Ct"]
    delta_poses = np.load(tracking / "Delta_C0_object_motion.npy")
    object_poses = np.load(tracking / "T_C0_from_O.npy")
    tracks_c0 = np.load(tracking / "tracks_3d_C0_raw.npy")
    track_valid = np.load(tracking / "track_valid.npy")
    frame_diagnostics = [json.loads(line) for line in (tracking / "frame_diagnostics.jsonl").read_text().splitlines()]
    hand_manifest = json.loads(hand_manifest_path.read_text(encoding="utf-8"))
    hand_frames = hand_manifest["frames"]
    contact_manifest = json.loads(contact_manifest_path.read_text(encoding="utf-8"))
    optimized_frames = contact_manifest["frames"]
    contact_targets = json.loads(Path(contact_manifest["contact_target_trajectory"]).read_text(encoding="utf-8"))
    contact_correspondences = json.loads(
        Path(contact_manifest["contact_correspondences"]).read_text(encoding="utf-8")
    )
    rigid_frame_count = len(delta_poses)
    timeline_frame_count = len(rgb_paths)
    exact_lengths = {
        "object_poses": len(object_poses),
        "tracks": len(tracks_c0),
        "track_valid": len(track_valid),
        "tracking_diagnostics": len(frame_diagnostics),
        "contact_frames": len(optimized_frames),
    }
    bad_exact = {name: count for name, count in exact_lengths.items() if count != rigid_frame_count}
    timeline_lengths = {
        "rgb": len(rgb_paths),
        "depth": len(depth_paths),
        "object_masks": len(mask_paths),
        "camera_poses": len(camera_poses),
        "egoforce_frames": len(hand_frames),
    }
    bad_timeline = {name: count for name, count in timeline_lengths.items() if count != timeline_frame_count}
    if bad_exact or bad_timeline:
        raise ValueError(
            f"Rigid viewer frame mismatch: rigid={rigid_frame_count}, "
            f"exact={bad_exact}, timeline={bad_timeline}"
        )
    if hand_manifest["coordinate_frame"] != "frame0_right_camera_opencv_rdf":
        raise ValueError("Viser only accepts pose-compensated C0 hand geometry")
    mesh = load_mesh(mesh_path)
    microwave_mesh = load_mesh(microwave_mesh_path) if microwave_mesh_path is not None else None

    server = viser.ViserServer(host=args.host, port=args.port)
    scene, gui = server.scene, server.gui
    scene.set_up_direction("-y")
    focus_frame = int(np.clip(args.initial_frame, 0, rigid_frame_count - 1))
    focus_center = transform_points(
        np.asarray(mesh.centroid, dtype=np.float64)[None], delta_poses[focus_frame]
    )[0]

    @server.on_client_connect
    def _frame_camera(client: viser.ClientHandle) -> None:
        client.camera.position = focus_center + np.array([0.20, -0.12, -0.32])
        client.camera.look_at = focus_center
        client.camera.up_direction = np.array([0.0, -1.0, 0.0])

    play = gui.add_checkbox("Play", initial_value=False)
    initial_frame = int(np.clip(args.initial_frame, 0, len(rgb_paths) - 1))
    frame_control = gui.add_slider(
        "Frame", min=0, max=len(rgb_paths) - 1, step=1, initial_value=initial_frame
    )
    fps_control = gui.add_slider("FPS", min=1.0, max=30.0, step=1.0, initial_value=args.fps)
    show_c0 = gui.add_checkbox("Show C0 axes", initial_value=True)
    show_object_axes = gui.add_checkbox("Show object axes", initial_value=True)
    show_camera_axes = gui.add_checkbox("Show camera axes", initial_value=False)
    show_mesh = gui.add_checkbox("Show mesh", initial_value=True)
    show_articulated_mesh = gui.add_checkbox("Show articulated mesh", initial_value=True)
    show_rgb = gui.add_checkbox("Show RGB", initial_value=False)
    show_rgbd = gui.add_checkbox("Show RGB-D", initial_value=False)
    show_object = gui.add_checkbox("Show object depth", initial_value=True)
    show_tracks = gui.add_checkbox("Show filtered 3D tracks", initial_value=True)
    show_left_hand = gui.add_checkbox("Show raw left hand", initial_value=False)
    show_right_hand = gui.add_checkbox("Show raw right hand", initial_value=False)
    show_arms = gui.add_checkbox("Show raw arms", initial_value=False)
    show_depth_hand = gui.add_checkbox("Show depth-aligned right hand", initial_value=False)
    show_hand_points = gui.add_checkbox("Show hand depth point cloud", initial_value=True)
    show_optimized_hand = gui.add_checkbox("Show final fused-contact hand", initial_value=True)
    show_optimized_arm = gui.add_checkbox("Show optimized arm", initial_value=True)
    show_contact_points = gui.add_checkbox("Show optimized contact points", initial_value=True)
    show_back_targets = gui.add_checkbox("Show selected-surface targets", initial_value=True)
    show_contact_heatmap = gui.add_checkbox("Show contact/collision heatmap", initial_value=True)
    show_collision_proxy = gui.add_checkbox("Show collision proxy", initial_value=False)
    status = gui.add_markdown(f"Frame {initial_frame}")

    c0_handle = scene.add_frame("/C0", axes_length=0.08, axes_radius=0.003)
    object_axes_handle = scene.add_frame(
        "/object/canonical_axes", axes_length=0.06, axes_radius=0.0025, visible=False
    )
    collision_proxy = load_mesh(Path(contact_manifest["collision_proxy"]))
    mesh_vertices = np.asarray(mesh.vertices, dtype=np.float32)
    mesh_faces = np.asarray(mesh.faces, dtype=np.uint32)
    proxy_vertices = np.asarray(collision_proxy.vertices, dtype=np.float32)
    proxy_faces = np.asarray(collision_proxy.faces, dtype=np.uint32)
    textured_mesh = getattr(mesh.visual, "uv", None) is not None and getattr(
        mesh.visual, "material", None
    ) is not None
    if textured_mesh:
        mesh_handle = scene.add_mesh_trimesh("/object/mesh", mesh=mesh, visible=False)
    else:
        mesh_handle = add_mesh_handle(scene, "/object/mesh", mesh, (245, 145, 35), False)
    articulated_mesh_handle = None
    if microwave_mesh is not None:
        articulated_mesh_handle = add_mesh_handle(
            scene,
            "/articulated/microwave/mesh",
            microwave_mesh,
            (170, 205, 255),
            bool(show_articulated_mesh.value),
        )
    collision_proxy_handle = scene.add_mesh_simple(
        "/object/collision_proxy",
        vertices=proxy_vertices,
        faces=proxy_faces,
        color=(70, 210, 240),
        opacity=0.25,
        side="double",
        visible=False,
    )
    frame_handles: list[dict[str, Any]] = []
    palette = {
        "index": (255, 70, 40),
        "middle": (255, 155, 35),
        "ring": (245, 225, 45),
        "pinky": (210, 80, 255),
    }

    print(f"Preloading {len(rgb_paths)} Viser frames...", flush=True)
    for frame in range(len(rgb_paths)):
        prefix = f"/frames/{frame:06d}"
        handles: dict[str, Any] = {}
        rgb = np.asarray(Image.open(rgb_paths[frame]).convert("RGB"))
        depth = np.load(depth_paths[frame]).astype(np.float32)
        mask = np.asarray(Image.open(mask_paths[frame]).convert("L")) > 127
        camera_pose = camera_poses[frame]

        height, width = rgb.shape[:2]
        fov_y = 2.0 * np.arctan2(height * 0.5, float(intrinsics["fy"]))
        rgb_preview = resize_preview(rgb)
        handles["rgb"] = scene.add_camera_frustum(
            f"{prefix}/rgb",
            fov=float(fov_y),
            aspect=float(width / height),
            scale=0.12,
            image=rgb_preview,
            color=(30, 30, 30),
            wxyz=wxyz(camera_pose[:3, :3]),
            position=camera_pose[:3, 3].astype(np.float32),
            visible=False,
        )
        handles["camera_axes"] = scene.add_frame(
            f"{prefix}/camera_axes",
            axes_length=0.045,
            axes_radius=0.0018,
            wxyz=wxyz(camera_pose[:3, :3]),
            position=camera_pose[:3, 3].astype(np.float32),
            visible=False,
        )
        points, colors = backproject(
            depth, rgb, intrinsics, camera_pose, max(1, args.rgbd_stride)
        )
        handles["rgbd"] = scene.add_point_cloud(
            f"{prefix}/rgbd",
            points=points,
            colors=colors,
            point_size=0.002,
            point_shape="circle",
            visible=False,
        )
        object_points, object_colors = backproject(
            depth, rgb, intrinsics, camera_pose, max(1, args.object_stride), mask
        )
        if len(object_colors):
            object_colors = np.full_like(object_colors, (35, 225, 90))
        handles["object"] = scene.add_point_cloud(
            f"{prefix}/object_depth",
            points=object_points,
            colors=object_colors,
            point_size=0.0035,
            point_shape="circle",
            visible=False,
        )
        if frame < rigid_frame_count:
            valid = track_valid[frame] & np.isfinite(tracks_c0[frame]).all(axis=1)
            track_points = tracks_c0[frame, valid].astype(np.float32)
        else:
            track_points = np.empty((0, 3), dtype=np.float32)
        handles["tracks"] = scene.add_point_cloud(
            f"{prefix}/filtered_tracks",
            points=track_points,
            colors=np.full((len(track_points), 3), (255, 210, 25), dtype=np.uint8),
            point_size=0.006,
            point_shape="circle",
            visible=False,
        )

        hand_record = hand_frames[frame]
        for side, color in (("left", (245, 70, 70)), ("right", (70, 125, 245))):
            if side == "right" and frame >= rigid_frame_count:
                continue
            hand_path = hand_record.get(f"{side}_hand_C0")
            arm_path = hand_record.get(f"{side}_arm_C0")
            if hand_path:
                hand_mesh = load_mesh(Path(hand_path))
                handles[f"{side}_hand"] = scene.add_mesh_simple(
                    f"{prefix}/{side}_hand_C0",
                    vertices=np.asarray(hand_mesh.vertices, dtype=np.float32),
                    faces=np.asarray(hand_mesh.faces, dtype=np.uint32),
                    color=color,
                    side="double",
                    visible=False,
                )
            if arm_path:
                arm_mesh = load_mesh(Path(arm_path))
                handles[f"{side}_arm"] = scene.add_mesh_simple(
                    f"{prefix}/{side}_arm_C0",
                    vertices=np.asarray(arm_mesh.vertices, dtype=np.float32),
                    faces=np.asarray(arm_mesh.faces, dtype=np.uint32),
                    color=tuple(max(0, value - 35) for value in color),
                    side="double",
                    visible=False,
                )

        if frame < rigid_frame_count:
            delta = delta_poses[frame]
            optimized_record = optimized_frames[frame]
            depth_hand_path = optimized_record.get("depth_aligned_hand_C0")
            if depth_hand_path:
                depth_hand_mesh = load_mesh(Path(depth_hand_path))
                handles["depth_hand"] = scene.add_mesh_simple(
                    f"{prefix}/right_hand_depth_aligned_C0",
                    vertices=np.asarray(depth_hand_mesh.vertices, dtype=np.float32),
                    faces=np.asarray(depth_hand_mesh.faces, dtype=np.uint32),
                    color=(80, 170, 255),
                    opacity=0.65,
                    side="double",
                    visible=False,
                )
            hand_points_path = optimized_record.get("hand_pointcloud_C0")
            if hand_points_path:
                with np.load(hand_points_path) as hand_points_data:
                    hand_points = hand_points_data["points_C0"].astype(np.float32)
                handles["hand_points"] = scene.add_point_cloud(
                    f"{prefix}/hand_depth_points_C0",
                    points=hand_points,
                    colors=np.full((len(hand_points), 3), (70, 190, 255), dtype=np.uint8),
                    point_size=0.003,
                    point_shape="circle",
                    visible=False,
                )
            optimized_path = optimized_record.get("optimized_geometry_C0")
            if optimized_path:
                with np.load(optimized_path) as optimized:
                    optimized_vertices = optimized["hand_vertices"].astype(np.float32)
                    optimized_faces = optimized["hand_faces"].astype(np.uint32)
                    signed_distance = optimized[
                        "signed_distance_to_collision_proxy_m"
                    ].astype(np.float32)
                    vertex_colors = np.full(
                        (len(optimized_vertices), 3), (55, 210, 120), dtype=np.uint8
                    )
                    vertex_colors[np.abs(signed_distance) < 0.003] = (255, 215, 35)
                    vertex_colors[signed_distance < 0.0] = (255, 45, 45)
                    colored_mesh = trimesh.Trimesh(
                        vertices=optimized_vertices,
                        faces=optimized_faces,
                        vertex_colors=vertex_colors,
                        process=False,
                    )
                    handles["optimized_hand_heatmap"] = scene.add_mesh_trimesh(
                        f"{prefix}/right_hand_optimized_heatmap_C0",
                        mesh=colored_mesh,
                        visible=False,
                    )
                    handles["optimized_hand_plain"] = scene.add_mesh_simple(
                        f"{prefix}/right_hand_optimized_C0",
                        vertices=optimized_vertices,
                        faces=optimized_faces,
                        color=(55, 210, 120),
                        side="double",
                        visible=False,
                    )
                    handles["optimized_arm"] = scene.add_mesh_simple(
                        f"{prefix}/right_arm_optimized_C0",
                        vertices=optimized["arm_vertices"].astype(np.float32),
                        faces=optimized["arm_faces"].astype(np.uint32),
                        color=(45, 155, 100),
                        side="double",
                        visible=False,
                    )

                frame_targets = contact_targets.get(str(frame))
                if frame_targets:
                    hand_contacts = []
                    back_targets_c0 = []
                    target_colors = []
                    for finger, targets in frame_targets.items():
                        target_object = np.asarray(targets, dtype=np.float32)
                        target_c0 = transform_points(target_object, delta)
                        region = np.asarray(
                            contact_correspondences[finger]["finger_region_vertex_indices"],
                            dtype=np.int64,
                        )
                        pairwise = np.linalg.norm(
                            target_c0[:, None, :] - optimized_vertices[region][None, :, :],
                            axis=-1,
                        )
                        target_index, region_index = np.unravel_index(
                            np.argmin(pairwise), pairwise.shape
                        )
                        hand_contacts.append(optimized_vertices[region[region_index]])
                        back_targets_c0.append(target_c0[target_index])
                        target_colors.append(palette.get(finger, (255, 80, 20)))
                    handles["contact_points"] = scene.add_point_cloud(
                        f"{prefix}/optimized_contact_points",
                        points=np.asarray(hand_contacts, dtype=np.float32),
                        colors=np.asarray(target_colors, dtype=np.uint8),
                        point_size=0.008,
                        point_shape="circle",
                        visible=False,
                    )
                    handles["back_targets"] = scene.add_point_cloud(
                        f"{prefix}/back_surface_targets",
                        points=np.asarray(back_targets_c0, dtype=np.float32),
                        colors=np.asarray(target_colors, dtype=np.uint8),
                        point_size=0.011,
                        point_shape="diamond",
                        visible=False,
                    )
        else:
            raw_right_hand_path = hand_record.get("right_hand_C0")
            raw_right_arm_path = hand_record.get("right_arm_C0")
            if raw_right_hand_path:
                fallback_hand_mesh = load_mesh(Path(raw_right_hand_path))
                handles["fallback_hand"] = scene.add_mesh_simple(
                    f"{prefix}/right_hand_raw_fallback_C0",
                    vertices=np.asarray(fallback_hand_mesh.vertices, dtype=np.float32),
                    faces=np.asarray(fallback_hand_mesh.faces, dtype=np.uint32),
                    color=(70, 125, 245),
                    side="double",
                    visible=False,
                )
            if raw_right_arm_path:
                fallback_arm_mesh = load_mesh(Path(raw_right_arm_path))
                handles["fallback_arm"] = scene.add_mesh_simple(
                    f"{prefix}/right_arm_raw_fallback_C0",
                    vertices=np.asarray(fallback_arm_mesh.vertices, dtype=np.float32),
                    faces=np.asarray(fallback_arm_mesh.faces, dtype=np.uint32),
                    color=(35, 95, 205),
                    side="double",
                    visible=False,
                )
        frame_handles.append(handles)
        if (frame + 1) % 8 == 0 or frame + 1 == len(rgb_paths):
            print(f"Preloaded {frame + 1}/{len(rgb_paths)} frames", flush=True)

    current_frame = initial_frame

    def desired_visibility(key: str) -> bool:
        controls = {
            "mesh": show_mesh,
            "articulated_mesh": show_articulated_mesh,
            "collision_proxy": show_collision_proxy,
            "rgb": show_rgb,
            "rgbd": show_rgbd,
            "object": show_object,
            "tracks": show_tracks,
            "left_hand": show_left_hand,
            "right_hand": show_right_hand,
            "left_arm": show_arms,
            "right_arm": show_arms,
            "depth_hand": show_depth_hand,
            "hand_points": show_hand_points,
            "optimized_arm": show_optimized_arm,
            "contact_points": show_contact_points,
            "back_targets": show_back_targets,
            "camera_axes": show_camera_axes,
        }
        if key == "fallback_hand":
            return bool(show_optimized_hand.value or show_right_hand.value)
        if key == "fallback_arm":
            return bool(show_optimized_arm.value or show_arms.value)
        if key == "optimized_hand_heatmap":
            return bool(show_optimized_hand.value and show_contact_heatmap.value)
        if key == "optimized_hand_plain":
            return bool(show_optimized_hand.value and not show_contact_heatmap.value)
        control = controls.get(key)
        return bool(control.value) if control is not None else False

    def update_status(frame: int) -> None:
        hand_record = hand_frames[frame]
        if frame < rigid_frame_count:
            record = frame_diagnostics[frame]
            optimized_record = optimized_frames[frame]
            status.content = (
                f"Frame {frame} | valid tracks {record['valid_track_count']} | "
                f"PnP inliers {record['pnp_inlier_count']} | IoU {record['silhouette_iou']:.3f} | "
                f"raw hands {', '.join(hand_record['detected_sides']) or 'none'} | "
                f"contact {optimized_record.get('contact', False)} | "
                f"role {optimized_record.get('contact_role', 'none')} | "
                f"anchor {contact_manifest['anchor_frame']}"
            )
        else:
            status.content = (
                f"Frame {frame} | raw hands {', '.join(hand_record['detected_sides']) or 'none'} | "
                f"rigid optimized through {rigid_frame_count - 1} | articulated microwave static"
            )

    def show_frame(frame: int) -> None:
        nonlocal current_frame
        frame = int(np.clip(frame, 0, len(frame_handles) - 1))
        rigid_index = min(frame, rigid_frame_count - 1)
        delta = delta_poses[rigid_index]
        object_pose = object_poses[rigid_index]
        with server.atomic():
            for handle in frame_handles[current_frame].values():
                handle.visible = False
            mesh_handle.wxyz = wxyz(delta[:3, :3])
            mesh_handle.position = delta[:3, 3].astype(np.float32)
            mesh_handle.visible = bool(show_mesh.value)
            if articulated_mesh_handle is not None:
                articulated_mesh_handle.visible = bool(show_articulated_mesh.value)
            collision_proxy_handle.wxyz = wxyz(delta[:3, :3])
            collision_proxy_handle.position = delta[:3, 3].astype(np.float32)
            collision_proxy_handle.visible = bool(show_collision_proxy.value)
            object_axes_handle.wxyz = wxyz(closest_rotation(object_pose[:3, :3]))
            object_axes_handle.position = object_pose[:3, 3].astype(np.float32)
            object_axes_handle.visible = bool(show_object_axes.value)
            for key, handle in frame_handles[frame].items():
                handle.visible = desired_visibility(key)
        current_frame = frame
        update_status(frame)

    @frame_control.on_update
    def _(_) -> None:
        show_frame(int(frame_control.value))

    def bind(control: Any, *keys: str) -> None:
        @control.on_update
        def _(_) -> None:
            with server.atomic():
                for key in keys:
                    handle = frame_handles[current_frame].get(key)
                    if handle is not None:
                        handle.visible = desired_visibility(key)

    bind(show_rgb, "rgb")
    bind(show_rgbd, "rgbd")
    bind(show_object, "object")
    bind(show_tracks, "tracks")
    bind(show_left_hand, "left_hand")
    bind(show_right_hand, "right_hand")
    bind(show_arms, "left_arm", "right_arm")
    bind(show_depth_hand, "depth_hand")
    bind(show_hand_points, "hand_points")
    bind(show_optimized_hand, "optimized_hand_heatmap", "optimized_hand_plain", "fallback_hand")
    bind(show_optimized_arm, "optimized_arm", "fallback_arm")
    bind(show_contact_points, "contact_points")
    bind(show_back_targets, "back_targets")
    bind(show_camera_axes, "camera_axes")
    bind(show_contact_heatmap, "optimized_hand_heatmap", "optimized_hand_plain")
    bind(show_right_hand, "fallback_hand")
    bind(show_arms, "fallback_arm")

    @show_mesh.on_update
    def _(_) -> None:
        mesh_handle.visible = bool(show_mesh.value)

    @show_collision_proxy.on_update
    def _(_) -> None:
        collision_proxy_handle.visible = bool(show_collision_proxy.value)

    @show_c0.on_update
    def _(_) -> None:
        c0_handle.visible = bool(show_c0.value)

    @show_object_axes.on_update
    def _(_) -> None:
        object_axes_handle.visible = bool(show_object_axes.value)

    @show_articulated_mesh.on_update
    def _(_) -> None:
        if articulated_mesh_handle is not None:
            articulated_mesh_handle.visible = bool(show_articulated_mesh.value)

    show_frame(initial_frame)
    print(
        json.dumps(
            {
                "frame_count": len(rgb_paths),
                "full_timeline_frame_count": int(hand_manifest["frame_count"]),
                "rigid_end_frame_inclusive": rigid_frame_count - 1,
                "object_id": args.object_id,
                "aligned_mesh": str(mesh_path),
                "articulated_mesh": str(microwave_mesh_path) if microwave_mesh_path is not None else None,
                "object_mask_dir": str(mask_dir),
                "mesh_vertices": len(mesh.vertices),
                "coordinate_frame": "frame0_right_camera_opencv_rdf",
                "egoforce_detected_frames": hand_manifest["detected_frame_count"],
                "contact_optimized_frames": contact_manifest["optimized_right_hand_frame_count"],
                "contact_optimization_status": contact_manifest["status"],
                "playback_mode": (
                    f"rigid_and_optimized_hand_0_{rigid_frame_count - 1}_then_"
                    f"raw_hand_fallback_{rigid_frame_count}_{timeline_frame_count - 1}"
                    if rigid_frame_count < timeline_frame_count
                    else f"rigid_and_optimized_hand_0_{rigid_frame_count - 1}"
                ),
                "url": f"http://localhost:{args.port}",
            },
            indent=2,
        )
    )
    last_tick = time.time()
    while True:
        now = time.time()
        if play.value and now - last_tick >= 1.0 / max(float(fps_control.value), 1.0):
            next_frame = (int(frame_control.value) + 1) % len(rgb_paths)
            frame_control.value = next_frame
            last_tick = now
        time.sleep(0.01)


if __name__ == "__main__":
    main()
