#!/usr/bin/env python3
"""Lightweight Viser viewer for C0-compensated rigid RGB-D tracking outputs."""

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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE = PROJECT_ROOT / "run_mixed_20260728_203734"
HAND_COLORS = {
    "left": (235, 70, 75),
    "right": (60, 120, 245),
}
COMPARISON_HAND_COLORS = {
    "left": (245, 155, 45),
    "right": (235, 70, 75),
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
        loaded = geometries[0].copy() if len(geometries) == 1 else trimesh.util.concatenate(geometries)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type {type(loaded)!r}: {path}")
    return loaded


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return np.asarray(points) @ transform[:3, :3].T + transform[:3, 3]


def closest_rotation(linear: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(np.asarray(linear, dtype=np.float64))
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return rotation


def wxyz(linear: np.ndarray) -> np.ndarray:
    xyzw = Rotation.from_matrix(closest_rotation(linear)).as_quat()
    return np.asarray([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float32)


def discover_images(directory: Path) -> list[Path]:
    for pattern in ("*.png", "*.jpg", "*.jpeg"):
        paths = sorted(directory.glob(pattern))
        if paths:
            return paths
    return []


def resolve_aligned_mesh(workspace: Path, object_id: str, explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    summary_path = workspace / "outputs/07_alignment/alignment_summary.json"
    if summary_path.is_file():
        matches = [
            record
            for record in read_json(summary_path).get("objects", [])
            if record.get("object_id") == object_id
        ]
        if len(matches) == 1:
            path = Path(matches[0]["aligned_mesh"]).resolve()
            if path.is_file():
                return path
        if len(matches) > 1:
            raise ValueError(f"Multiple Stage07 alignment records for {object_id!r}")
    candidates = [
        workspace / "outputs/07_alignment" / object_id / "frame_000000/sam3d_aligned_C0.glb",
        workspace / "outputs/07_alignment" / object_id / "frame_000000/hunyuan_mesh_aligned_C0.glb",
        workspace / "outputs/07_alignment/frame_000000/hunyuan_mesh_aligned_C0.glb",
    ]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"No Stage07 aligned C0 mesh for {object_id!r}: {candidates}")


def resolve_mask_dir(
    workspace: Path,
    object_id: str,
    frame_count: int,
    explicit: Path | None,
) -> Path:
    candidates = [explicit.resolve()] if explicit is not None else [
        workspace / "outputs/04_object_masks" / object_id / "objects" / object_id,
        workspace / "outputs/04_object_masks" / object_id / "combined",
        workspace / "outputs/04_object_masks/objects" / object_id,
        workspace / "outputs/04_object_masks/combined",
        workspace / "outputs/02_sam2_frame0_masks/propagated/objects" / object_id,
    ]
    counts: dict[str, int] = {}
    for candidate in candidates:
        count = len(list(candidate.glob("*.png"))) if candidate.is_dir() else 0
        counts[str(candidate)] = count
        if count >= frame_count:
            return candidate.resolve()
    raise FileNotFoundError(
        f"No object-mask sequence with at least {frame_count} frames for {object_id!r}: {counts}"
    )


def resolve_fps(workspace: Path, fps_arg: str, pose_data: Any) -> float:
    if fps_arg.lower() != "auto":
        fps = float(fps_arg)
        if not np.isfinite(fps) or fps <= 0.0:
            raise ValueError(f"--fps must be 'auto' or a positive number, got {fps_arg!r}")
        return fps
    manifest_path = workspace / "outputs/00_rgb_frames/stage00_manifest.json"
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        for key in ("target_fps", "effective_fps", "fps"):
            value = manifest.get(key)
            if value is not None and float(value) > 0.0:
                return float(value)
    if "rgb_timestamps_s" in pose_data.files:
        timestamps = np.asarray(pose_data["rgb_timestamps_s"], dtype=np.float64)
        dt = np.diff(timestamps)
        dt = dt[np.isfinite(dt) & (dt > 1e-6)]
        if len(dt):
            return float(1.0 / np.median(dt))
    return 15.0


def load_pose_timeline(tracking_dir: Path, frame_count: int) -> tuple[np.ndarray, int, str]:
    delta_path = tracking_dir / "Delta_C0_object_motion.npy"
    object_pose_path = tracking_dir / "T_C0_from_O.npy"
    if delta_path.is_file():
        tracked = np.load(delta_path).astype(np.float64)
        source = str(delta_path)
    elif object_pose_path.is_file():
        object_poses = np.load(object_pose_path).astype(np.float64)
        if object_poses.ndim != 3 or object_poses.shape[1:] != (4, 4) or not len(object_poses):
            raise ValueError(f"Malformed object pose array: {object_pose_path} {object_poses.shape}")
        tracked = np.einsum("tij,jk->tik", object_poses, np.linalg.inv(object_poses[0]))
        source = f"{object_pose_path} (converted relative to frame 0)"
    else:
        raise FileNotFoundError(f"No rigid pose array in {tracking_dir}")
    if tracked.ndim != 3 or tracked.shape[1:] != (4, 4) or not len(tracked):
        raise ValueError(f"Malformed rigid delta poses: {tracked.shape}")
    if not np.isfinite(tracked).all():
        raise ValueError("Rigid pose array contains non-finite values")
    used_count = min(len(tracked), frame_count)
    timeline = np.repeat(tracked[used_count - 1][None], frame_count, axis=0)
    timeline[:used_count] = tracked[:used_count]
    return timeline, used_count, source


def load_track_timeline(
    tracking_dir: Path, frame_count: int
) -> tuple[np.ndarray, np.ndarray, int]:
    points_path = tracking_dir / "tracks_3d_C0_raw.npy"
    valid_path = tracking_dir / "track_valid.npy"
    if not points_path.is_file() or not valid_path.is_file():
        return (
            np.empty((0, 0, 3), dtype=np.float32),
            np.empty((0, 0), dtype=bool),
            0,
        )
    points = np.load(points_path).astype(np.float32)
    valid = np.load(valid_path).astype(bool)
    if points.ndim != 3 or points.shape[2] != 3 or valid.shape != points.shape[:2]:
        raise ValueError(f"Malformed tracks: points={points.shape}, valid={valid.shape}")
    used_count = min(len(points), frame_count)
    return points[:used_count], valid[:used_count], used_count


def backproject_c0(
    depth: np.ndarray,
    rgb: np.ndarray,
    intrinsics: dict[str, float],
    transform_c0_from_ct: np.ndarray,
    stride: int,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    stride = max(1, int(stride))
    v, u = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
    z = depth[::stride, ::stride]
    valid = np.isfinite(z) & (z >= 0.1) & (z <= 3.0)
    if mask is not None:
        valid &= mask[::stride, ::stride]
    z, u, v = z[valid], u[valid], v[valid]
    x = (u - float(intrinsics["cx"])) * z / float(intrinsics["fx"])
    y = (v - float(intrinsics["cy"])) * z / float(intrinsics["fy"])
    points_ct = np.column_stack([x, y, z])
    points_c0 = transform_points(points_ct, transform_c0_from_ct).astype(np.float32)
    colors = rgb[v, u].astype(np.uint8)
    return np.ascontiguousarray(points_c0), np.ascontiguousarray(colors)


def add_mesh_handle(scene: Any, name: str, mesh: trimesh.Trimesh) -> Any:
    textured = getattr(mesh.visual, "uv", None) is not None and getattr(
        mesh.visual, "material", None
    ) is not None
    if textured:
        return scene.add_mesh_trimesh(name, mesh=mesh, visible=False)
    return scene.add_mesh_simple(
        name,
        vertices=np.asarray(mesh.vertices, dtype=np.float32),
        faces=np.asarray(mesh.faces, dtype=np.uint32),
        color=(245, 145, 35),
        side="double",
        visible=False,
    )


def add_hand_mesh_handle(scene: Any, name: str, mesh: trimesh.Trimesh, side: str) -> Any:
    return scene.add_mesh_simple(
        name,
        vertices=np.asarray(mesh.vertices, dtype=np.float32),
        faces=np.asarray(mesh.faces, dtype=np.uint32),
        color=HAND_COLORS[side],
        side="double",
        visible=False,
    )


def add_comparison_hand_mesh_handle(
    scene: Any, name: str, mesh: trimesh.Trimesh, side: str
) -> Any:
    return scene.add_mesh_simple(
        name,
        vertices=np.asarray(mesh.vertices, dtype=np.float32),
        faces=np.asarray(mesh.faces, dtype=np.uint32),
        color=COMPARISON_HAND_COLORS[side],
        side="double",
        visible=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--object-id", default="bottle")
    parser.add_argument(
        "--articulated-object-id",
        default="microwave",
        help="Optional Stage07 mesh shown statically in C0; pass an empty string to disable.",
    )
    parser.add_argument("--articulated-static-mesh", type=Path, default=None)
    parser.add_argument("--articulated-dynamic-mesh", type=Path, default=None)
    parser.add_argument("--articulated-tracking-dir", type=Path, default=None)
    parser.add_argument("--rgb-dir", type=Path, default=None)
    parser.add_argument("--depth-dir", type=Path, default=None)
    parser.add_argument(
        "--hand-pointcloud-dir",
        type=Path,
        default=None,
        help="Optional per-frame NPZ files containing points_C0 used for hand depth alignment.",
    )
    parser.add_argument("--poses-path", type=Path, default=None)
    parser.add_argument("--pose-key", default="T_C0_from_Ct")
    parser.add_argument("--mask-dir", type=Path, default=None)
    parser.add_argument("--aligned-mesh", type=Path, default=None)
    parser.add_argument("--tracking-dir", type=Path, default=None)
    parser.add_argument(
        "--hand-manifest",
        type=Path,
        default=None,
        help="Pose-compensated EgoForce manifest. Defaults to outputs/09_egoforce/dynamic_manifest.json.",
    )
    parser.add_argument(
        "--comparison-hand-manifest",
        type=Path,
        default=None,
        help="Optional second C0 hand sequence shown for before/after comparison.",
    )
    parser.add_argument("--frame-count", type=int, default=40, help="Viewer timeline length; 0 uses all RGB frames.")
    parser.add_argument("--initial-frame", type=int, default=0)
    parser.add_argument("--rgbd-stride", type=int, default=8)
    parser.add_argument("--object-stride", type=int, default=3)
    parser.add_argument("--fps", default="auto", help="Playback FPS or 'auto' from Stage00/timestamps.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument(
        "--pointcloud-only",
        action="store_true",
        help="Start with only the full C0-compensated RGB-D point cloud visible.",
    )
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = args.workspace.resolve()
    rgb_dir = (args.rgb_dir or workspace / "outputs/00_rgb_frames/right_rgb_png").resolve()
    depth_dir = (args.depth_dir or workspace / "outputs/06_dense_depth/metric_depth_npy").resolve()
    hand_pointcloud_dir = (
        args.hand_pointcloud_dir.resolve()
        if args.hand_pointcloud_dir is not None
        else None
    )
    tracking_dir = (args.tracking_dir or workspace / "outputs/08_tracking").resolve()
    rgb_all = discover_images(rgb_dir)
    depth_all = sorted(depth_dir.glob("*.npy"))
    if not rgb_all:
        raise FileNotFoundError(f"No RGB frames in {rgb_dir}")
    requested_count = len(rgb_all) if args.frame_count == 0 else int(args.frame_count)
    if requested_count <= 0:
        raise ValueError("--frame-count must be non-negative")
    if len(rgb_all) < requested_count or len(depth_all) < requested_count:
        raise ValueError(
            f"Requested {requested_count} frames, available RGB/depth={len(rgb_all)}/{len(depth_all)}"
        )
    frame_count = requested_count
    rgb_paths = rgb_all[:frame_count]
    depth_paths = depth_all[:frame_count]
    mask_dir = resolve_mask_dir(workspace, args.object_id, frame_count, args.mask_dir)
    mask_paths = sorted(mask_dir.glob("*.png"))[:frame_count]
    mesh_path = resolve_aligned_mesh(workspace, args.object_id, args.aligned_mesh)
    articulated_mesh_path: Path | None = None
    if args.articulated_object_id:
        if args.articulated_static_mesh is not None:
            articulated_mesh_path = args.articulated_static_mesh.resolve()
        else:
            try:
                articulated_mesh_path = resolve_aligned_mesh(
                    workspace, args.articulated_object_id, None
                )
            except FileNotFoundError:
                articulated_mesh_path = None
    articulated_dynamic_mesh_path = (
        args.articulated_dynamic_mesh.resolve()
        if args.articulated_dynamic_mesh is not None
        else None
    )
    articulated_tracking_dir = (
        args.articulated_tracking_dir.resolve()
        if args.articulated_tracking_dir is not None
        else None
    )
    articulated_delta: np.ndarray | None = None
    articulated_tracked_count = 0
    articulated_pose_source: str | None = None
    if articulated_dynamic_mesh_path is not None or articulated_tracking_dir is not None:
        if articulated_dynamic_mesh_path is None or articulated_tracking_dir is None:
            raise ValueError(
                "--articulated-dynamic-mesh and --articulated-tracking-dir must be used together"
            )
        articulated_delta, articulated_tracked_count, articulated_pose_source = (
            load_pose_timeline(articulated_tracking_dir, frame_count)
        )

    camera_path = workspace / "outputs/00_rgb_frames/camera.json"
    poses_path = (
        args.poses_path.resolve()
        if args.poses_path is not None
        else workspace / "outputs/00_rgb_frames/poses.npz"
    )
    camera = read_json(camera_path)
    stage00_path = workspace / "outputs/00_rgb_frames/stage00_manifest.json"
    stage00 = read_json(stage00_path) if stage00_path.is_file() else {}
    selected_eye = str(
        stage00.get("selected_eye", camera.get("selected_eye", "right"))
    ).lower()
    if selected_eye not in {"left", "right"}:
        raise ValueError(f"Unsupported selected eye: {selected_eye!r}")
    world_frame = f"frame0_{selected_eye}_camera_opencv_rdf"
    intrinsics = camera.get("rgb_intrinsics_selected")
    if intrinsics is None:
        intrinsics = camera[f"rgb_intrinsics_{selected_eye}"]
    pose_data = np.load(poses_path)
    if args.pose_key not in pose_data.files:
        raise KeyError(f"Pose key {args.pose_key!r} not found in {poses_path}: {pose_data.files}")
    camera_poses = pose_data[args.pose_key].astype(np.float64)
    if camera_poses.shape[0] < frame_count or camera_poses.shape[1:] != (4, 4):
        raise ValueError(f"Camera poses {camera_poses.shape} do not cover {frame_count} frames")
    camera_poses = camera_poses[:frame_count]
    playback_fps = resolve_fps(workspace, str(args.fps), pose_data)
    object_delta, tracked_pose_count, pose_source = load_pose_timeline(tracking_dir, frame_count)
    tracks_c0, track_valid, track_frame_count = load_track_timeline(tracking_dir, frame_count)
    hand_manifest_path = (
        args.hand_manifest.resolve()
        if args.hand_manifest is not None
        else workspace / "outputs/09_egoforce/dynamic_manifest.json"
    )
    hand_manifest: dict[str, Any] | None = None
    hand_frames: list[dict[str, Any]] = [{} for _ in range(frame_count)]
    hand_side_counts = {"left": 0, "right": 0}
    if hand_manifest_path.is_file():
        hand_manifest = read_json(hand_manifest_path)
        if hand_manifest.get("coordinate_frame") != world_frame:
            raise ValueError(
                f"Hand manifest must contain C0 geometry in {world_frame}: {hand_manifest_path}"
            )
        manifest_frames = hand_manifest.get("frames", [])
        if len(manifest_frames) < frame_count:
            raise ValueError(
                f"Hand manifest has {len(manifest_frames)} frames, expected at least {frame_count}"
            )
        hand_frames = manifest_frames[:frame_count]
        manifest_pose_source = hand_manifest.get("pose_source")
        if manifest_pose_source and Path(manifest_pose_source).resolve() != poses_path:
            raise ValueError(
                "Hand geometry and RGB-D use different camera poses: "
                f"{manifest_pose_source} vs {poses_path}"
            )
        hand_side_counts = {
            side: sum(bool(record.get(f"{side}_hand_C0")) for record in hand_frames)
            for side in HAND_COLORS
        }
    elif args.hand_manifest is not None:
        raise FileNotFoundError(hand_manifest_path)

    comparison_hand_frames: list[dict[str, Any]] = [{} for _ in range(frame_count)]
    comparison_hand_counts = {"left": 0, "right": 0}
    comparison_hand_manifest_path = (
        args.comparison_hand_manifest.resolve()
        if args.comparison_hand_manifest is not None
        else None
    )
    if comparison_hand_manifest_path is not None:
        comparison_manifest = read_json(comparison_hand_manifest_path)
        if comparison_manifest.get("coordinate_frame") != world_frame:
            raise ValueError(
                f"Comparison hand manifest must contain C0 geometry in {world_frame}: "
                f"{comparison_hand_manifest_path}"
            )
        comparison_hand_frames = comparison_manifest.get("frames", [])[:frame_count]
        if len(comparison_hand_frames) < frame_count:
            raise ValueError(
                f"Comparison hand manifest has {len(comparison_hand_frames)} frames, "
                f"expected at least {frame_count}"
            )
        comparison_hand_counts = {
            side: sum(bool(record.get(f"{side}_hand_C0")) for record in comparison_hand_frames)
            for side in HAND_COLORS
        }

    preflight = {
        "viewer": "rigid_rgbd_only_viser",
        "workspace": str(workspace),
        "object_id": args.object_id,
        "frame_count": frame_count,
        "fps": playback_fps,
        "selected_eye": selected_eye,
        "world_frame": world_frame,
        "pose_compensation": "p_C0(t) = T_C0_from_Ct(t) @ p_Ct(t)",
        "poses_path": str(poses_path),
        "pose_key": args.pose_key,
        "rgb_dir": str(rgb_dir),
        "depth_dir": str(depth_dir),
        "hand_pointcloud_dir": (
            str(hand_pointcloud_dir) if hand_pointcloud_dir is not None else None
        ),
        "mask_dir": str(mask_dir),
        "aligned_mesh": str(mesh_path),
        "articulated_object_id": args.articulated_object_id or None,
        "articulated_mesh": (
            str(articulated_mesh_path) if articulated_mesh_path is not None else None
        ),
        "articulated_dynamic_mesh": (
            str(articulated_dynamic_mesh_path)
            if articulated_dynamic_mesh_path is not None
            else None
        ),
        "articulated_tracking_dir": (
            str(articulated_tracking_dir) if articulated_tracking_dir is not None else None
        ),
        "articulated_tracked_pose_frame_count": articulated_tracked_count,
        "articulated_pose_source": articulated_pose_source,
        "tracking_dir": str(tracking_dir),
        "tracked_pose_frame_count": tracked_pose_count,
        "tracked_pose_end_frame_inclusive": tracked_pose_count - 1,
        "post_tracking_pose_policy": (
            f"freeze frame {tracked_pose_count - 1} pose"
            if tracked_pose_count < frame_count
            else "not needed"
        ),
        "track_frame_count": track_frame_count,
        "hand_manifest": str(hand_manifest_path) if hand_manifest is not None else None,
        "hand_side_frame_counts": hand_side_counts,
        "comparison_hand_manifest": (
            str(comparison_hand_manifest_path)
            if comparison_hand_manifest_path is not None
            else None
        ),
        "comparison_hand_side_frame_counts": comparison_hand_counts,
        "pose_source": pose_source,
        "url": f"http://localhost:{args.port}",
    }
    if args.check:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return

    try:
        import viser
    except ImportError as exc:
        raise RuntimeError(
            "viser is not installed in this Python. Use the project runtime, e.g. "
            "/opt/conda/envs/arthoi/bin/python."
        ) from exc

    mesh = load_mesh(mesh_path)
    articulated_mesh = (
        load_mesh(articulated_mesh_path) if articulated_mesh_path is not None else None
    )
    articulated_dynamic_mesh = (
        load_mesh(articulated_dynamic_mesh_path)
        if articulated_dynamic_mesh_path is not None
        else None
    )
    initial_frame = int(np.clip(args.initial_frame, 0, frame_count - 1))
    focus_center = transform_points(
        np.asarray(mesh.centroid, dtype=np.float64)[None], object_delta[initial_frame]
    )[0]
    server = viser.ViserServer(host=args.host, port=args.port)
    scene, gui = server.scene, server.gui
    scene.set_up_direction("-y")

    @server.on_client_connect
    def _frame_camera(client: Any) -> None:
        client.camera.position = focus_center + np.asarray([0.22, -0.16, -0.36])
        client.camera.look_at = focus_center
        client.camera.up_direction = np.asarray([0.0, -1.0, 0.0])

    play = gui.add_checkbox("Play", initial_value=False)
    frame_control = gui.add_slider(
        "Frame", min=0, max=frame_count - 1, step=1, initial_value=initial_frame
    )
    fps_control = gui.add_slider(
        "FPS",
        min=1.0,
        max=max(30.0, float(np.ceil(playback_fps))),
        step=0.5,
        initial_value=float(playback_fps),
    )
    show_rgbd = gui.add_checkbox("Show C0 RGB-D", initial_value=bool(args.pointcloud_only))
    show_object_depth = gui.add_checkbox(
        "Show C0 object depth", initial_value=not args.pointcloud_only
    )
    show_hand_depth = gui.add_checkbox(
        "Show hand depth points", initial_value=hand_pointcloud_dir is not None
    )
    show_tracks = gui.add_checkbox(
        "Show C0 tracks", initial_value=track_frame_count > 0 and not args.pointcloud_only
    )
    show_mesh = gui.add_checkbox("Show tracked mesh", initial_value=not args.pointcloud_only)
    show_articulated_mesh = gui.add_checkbox(
        f"Show {args.articulated_object_id} body",
        initial_value=articulated_mesh is not None and not args.pointcloud_only,
    )
    show_articulated_dynamic_mesh = gui.add_checkbox(
        f"Show tracked {args.articulated_object_id} part",
        initial_value=articulated_dynamic_mesh is not None and not args.pointcloud_only,
    )
    show_left_hand = gui.add_checkbox(
        "Show left hand", initial_value=hand_side_counts["left"] > 0
    )
    show_right_hand = gui.add_checkbox(
        "Show right hand", initial_value=hand_side_counts["right"] > 0
    )
    show_comparison_left_hand = gui.add_checkbox(
        "Show original left hand", initial_value=False
    )
    show_comparison_right_hand = gui.add_checkbox(
        "Show original right hand", initial_value=comparison_hand_counts["right"] > 0
    )
    show_camera_axes = gui.add_checkbox("Show current camera axes", initial_value=True)
    show_c0_axes = gui.add_checkbox("Show C0 axes", initial_value=True)
    status = gui.add_markdown("")

    c0_handle = scene.add_frame(
        "/C0", axes_length=0.09, axes_radius=0.003, visible=bool(show_c0_axes.value)
    )
    camera_handle = scene.add_frame(
        "/current_camera_Ct_in_C0",
        axes_length=0.055,
        axes_radius=0.002,
        visible=bool(show_camera_axes.value),
    )
    mesh_handle = add_mesh_handle(scene, f"/{args.object_id}/aligned_mesh_C0", mesh)
    articulated_mesh_handle = (
        add_mesh_handle(
            scene,
            f"/{args.articulated_object_id}/static_aligned_mesh_C0",
            articulated_mesh,
        )
        if articulated_mesh is not None
        else None
    )
    articulated_dynamic_mesh_handle = (
        add_mesh_handle(
            scene,
            f"/{args.articulated_object_id}/tracked_dynamic_part_C0",
            articulated_dynamic_mesh,
        )
        if articulated_dynamic_mesh is not None
        else None
    )

    frame_handles: list[dict[str, Any]] = []
    point_counts: list[tuple[int, int, int]] = []
    detected_hand_sides: list[list[str]] = []
    print(f"Preloading {frame_count} lightweight RGB-D frames...", flush=True)
    for frame in range(frame_count):
        rgb = np.asarray(Image.open(rgb_paths[frame]).convert("RGB"))
        depth = np.load(depth_paths[frame]).astype(np.float32)
        mask = np.asarray(Image.open(mask_paths[frame]).convert("L")) > 127
        if rgb.shape[:2] != depth.shape or mask.shape != depth.shape:
            raise ValueError(
                f"Frame {frame} RGB/depth/mask mismatch: {rgb.shape[:2]}/{depth.shape}/{mask.shape}"
            )
        rgbd_points, rgbd_colors = backproject_c0(
            depth, rgb, intrinsics, camera_poses[frame], args.rgbd_stride
        )
        object_points, object_colors = backproject_c0(
            depth, rgb, intrinsics, camera_poses[frame], args.object_stride, mask
        )
        if frame < track_frame_count:
            valid = track_valid[frame] & np.isfinite(tracks_c0[frame]).all(axis=1)
            track_points = tracks_c0[frame, valid]
        else:
            track_points = np.empty((0, 3), dtype=np.float32)
        prefix = f"/frames/{frame:06d}"
        handles = {
            "rgbd": scene.add_point_cloud(
                f"{prefix}/rgbd_C0",
                points=rgbd_points,
                colors=rgbd_colors,
                point_size=0.0025,
                point_shape="circle",
                visible=False,
            ),
            "object_depth": scene.add_point_cloud(
                f"{prefix}/object_depth_C0",
                points=object_points,
                colors=object_colors,
                point_size=0.0035,
                point_shape="circle",
                visible=False,
            ),
            "tracks": scene.add_point_cloud(
                f"{prefix}/tracks_C0",
                points=np.ascontiguousarray(track_points),
                colors=np.full((len(track_points), 3), (255, 210, 25), dtype=np.uint8),
                point_size=0.007,
                point_shape="circle",
                visible=False,
            ),
        }
        if hand_pointcloud_dir is not None:
            hand_points_path = hand_pointcloud_dir / f"{frame:06d}.npz"
            if hand_points_path.is_file():
                with np.load(hand_points_path) as loaded:
                    hand_depth_points = loaded["points_C0"].astype(np.float32)
                handles["hand_depth"] = scene.add_point_cloud(
                    f"{prefix}/hand_depth_observations_C0",
                    points=np.ascontiguousarray(hand_depth_points),
                    colors=np.full(
                        (len(hand_depth_points), 3), (30, 225, 220), dtype=np.uint8
                    ),
                    point_size=0.005,
                    point_shape="circle",
                    visible=False,
                )
        frame_sides: list[str] = []
        hand_record = hand_frames[frame]
        for side in HAND_COLORS:
            hand_path_value = hand_record.get(f"{side}_hand_C0")
            if not hand_path_value:
                continue
            hand_path = Path(hand_path_value)
            if not hand_path.is_file():
                raise FileNotFoundError(hand_path)
            handles[f"{side}_hand"] = add_hand_mesh_handle(
                scene,
                f"{prefix}/{side}_hand_C0",
                load_mesh(hand_path),
                side,
            )
            frame_sides.append(side)
        comparison_record = comparison_hand_frames[frame]
        for side in HAND_COLORS:
            comparison_path_value = comparison_record.get(f"{side}_hand_C0")
            if not comparison_path_value:
                continue
            comparison_path = Path(comparison_path_value)
            if not comparison_path.is_file():
                raise FileNotFoundError(comparison_path)
            handles[f"comparison_{side}_hand"] = add_comparison_hand_mesh_handle(
                scene,
                f"{prefix}/original_{side}_hand_C0",
                load_mesh(comparison_path),
                side,
            )
        frame_handles.append(handles)
        point_counts.append((len(rgbd_points), len(object_points), len(track_points)))
        detected_hand_sides.append(frame_sides)
        if (frame + 1) % 10 == 0 or frame + 1 == frame_count:
            print(f"Preloaded {frame + 1}/{frame_count}", flush=True)

    current_frame = initial_frame

    def desired_visibility(key: str) -> bool:
        controls = {
            "rgbd": show_rgbd,
            "object_depth": show_object_depth,
            "hand_depth": show_hand_depth,
            "tracks": show_tracks,
            "left_hand": show_left_hand,
            "right_hand": show_right_hand,
            "comparison_left_hand": show_comparison_left_hand,
            "comparison_right_hand": show_comparison_right_hand,
        }
        return bool(controls[key].value)

    def update_status(frame: int) -> None:
        rgbd_count, object_count, track_count = point_counts[frame]
        pose_text = (
            f"tracked pose {frame}"
            if frame < tracked_pose_count
            else f"frozen pose from {tracked_pose_count - 1}"
        )
        track_text = f"{track_count} observed tracks" if frame < track_frame_count else "tracks unavailable"
        hands_text = ", ".join(detected_hand_sides[frame]) or "none"
        status.content = (
            f"Frame {frame}/{frame_count - 1} | {pose_text} | {track_text} | "
            f"hands {hands_text} | RGB-D {rgbd_count:,} pts | object {object_count:,} pts | "
            "C0 pose compensation applied"
        )

    def show_frame(frame: int) -> None:
        nonlocal current_frame
        frame = int(np.clip(frame, 0, frame_count - 1))
        camera_pose = camera_poses[frame]
        delta = object_delta[frame]
        with server.atomic():
            for handle in frame_handles[current_frame].values():
                handle.visible = False
            for key, handle in frame_handles[frame].items():
                handle.visible = desired_visibility(key)
            mesh_handle.wxyz = wxyz(delta[:3, :3])
            mesh_handle.position = delta[:3, 3].astype(np.float32)
            mesh_handle.visible = bool(show_mesh.value)
            if articulated_mesh_handle is not None:
                articulated_mesh_handle.visible = bool(show_articulated_mesh.value)
            if articulated_dynamic_mesh_handle is not None and articulated_delta is not None:
                articulated_part_delta = articulated_delta[frame]
                articulated_dynamic_mesh_handle.wxyz = wxyz(articulated_part_delta[:3, :3])
                articulated_dynamic_mesh_handle.position = articulated_part_delta[:3, 3].astype(
                    np.float32
                )
                articulated_dynamic_mesh_handle.visible = bool(
                    show_articulated_dynamic_mesh.value
                )
            camera_handle.wxyz = wxyz(camera_pose[:3, :3])
            camera_handle.position = camera_pose[:3, 3].astype(np.float32)
            camera_handle.visible = bool(show_camera_axes.value)
        current_frame = frame
        update_status(frame)

    @frame_control.on_update
    def _(_) -> None:
        show_frame(int(frame_control.value))

    def bind_frame_control(control: Any, key: str) -> None:
        @control.on_update
        def _(_) -> None:
            handle = frame_handles[current_frame].get(key)
            if handle is not None:
                handle.visible = desired_visibility(key)

    bind_frame_control(show_rgbd, "rgbd")
    bind_frame_control(show_object_depth, "object_depth")
    bind_frame_control(show_hand_depth, "hand_depth")
    bind_frame_control(show_tracks, "tracks")
    bind_frame_control(show_left_hand, "left_hand")
    bind_frame_control(show_right_hand, "right_hand")
    bind_frame_control(show_comparison_left_hand, "comparison_left_hand")
    bind_frame_control(show_comparison_right_hand, "comparison_right_hand")

    @show_mesh.on_update
    def _(_) -> None:
        mesh_handle.visible = bool(show_mesh.value)

    @show_articulated_mesh.on_update
    def _(_) -> None:
        if articulated_mesh_handle is not None:
            articulated_mesh_handle.visible = bool(show_articulated_mesh.value)

    @show_articulated_dynamic_mesh.on_update
    def _(_) -> None:
        if articulated_dynamic_mesh_handle is not None:
            articulated_dynamic_mesh_handle.visible = bool(
                show_articulated_dynamic_mesh.value
            )

    @show_camera_axes.on_update
    def _(_) -> None:
        camera_handle.visible = bool(show_camera_axes.value)

    @show_c0_axes.on_update
    def _(_) -> None:
        c0_handle.visible = bool(show_c0_axes.value)

    show_frame(initial_frame)
    print(json.dumps(preflight, ensure_ascii=False, indent=2), flush=True)
    last_tick = time.time()
    while True:
        now = time.time()
        if play.value and now - last_tick >= 1.0 / max(float(fps_control.value), 1.0):
            frame_control.value = (int(frame_control.value) + 1) % frame_count
            last_tick = now
        time.sleep(0.01)


if __name__ == "__main__":
    main()
