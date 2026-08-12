#!/usr/bin/env python3
"""Play dynamic laptop meshes together with EgoForce hand meshes in viser."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import trimesh
import viser

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vlm_sam2_recon.stages.camera_alignment import depth_points_in_right_camera, frame_name  # noqa: E402


DEFAULT_DYNAMIC_DIR = PROJECT_ROOT / "outputs/screen_motion/target_laptop_frames_000000_000019"
DEFAULT_EGOFORCE_DIR = PROJECT_ROOT / "outputs/egoforce_rgb_right"
DEFAULT_EXPORT_ROOT = Path("/code/3DVideo_2026-07-01-19-24-06-667_spatialmp4_export")

PART_COLORS = {"14": (220, 50, 65), "15": (75, 145, 210)}
HAND_COLOR = (245, 190, 135)
ARM_COLOR = (150, 120, 95)
JOINT_COLOR = (255, 40, 40)
TRACK_COLOR = (255, 230, 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve dynamic laptop + EgoForce hands in viser.")
    parser.add_argument("--dynamic-dir", type=Path, default=DEFAULT_DYNAMIC_DIR)
    parser.add_argument("--egoforce-dir", type=Path, default=DEFAULT_EGOFORCE_DIR)
    parser.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT_ROOT)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8098)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--laptop-opacity", type=float, default=0.86)
    parser.add_argument("--hand-opacity", type=float, default=0.95)
    parser.add_argument(
        "--hand-source",
        choices=["manifest", "egoforce"],
        default="manifest",
        help="manifest uses corrected hand meshes when present; egoforce forces raw EgoForce OBJ files.",
    )
    parser.add_argument("--axis-length", type=float, default=0.42)
    parser.add_argument("--rgb-frustum-scale", type=float, default=0.25)
    parser.add_argument("--preload-dynamic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--static-base", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-textured-laptop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--textured-laptop-dir",
        type=Path,
        default=None,
        help="Directory from scripts/build_textured_laptop_parts.py. Defaults to dynamic-dir/textured_laptop_parts if present.",
    )
    parser.add_argument("--hand-side", choices=["left", "right", "both"], default=None)
    parser.add_argument("--show-scene-bounds", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--show-laptop-initial", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-arm-initial", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-rgb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-rgbd", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rgbd-max-points", type=int, default=45000)
    parser.add_argument("--rgbd-point-size", type=float, default=0.002)
    parser.add_argument("--track-point-size", type=float, default=0.012)
    parser.add_argument("--depth-min-m", type=float, default=0.1)
    parser.add_argument("--depth-max-m", type=float, default=3.0)
    parser.add_argument(
        "--depth-convention",
        choices=["camera_to_rig", "rig_to_camera", "direct_same_camera"],
        default=None,
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        geoms = [geom for geom in mesh.geometry.values() if isinstance(geom, trimesh.Trimesh)]
        if not geoms:
            raise ValueError(f"No mesh geometry found in {path}")
        mesh = geoms[0].copy() if len(geoms) == 1 else trimesh.util.concatenate(geoms)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type from {path}: {type(mesh)!r}")
    return mesh


def remove_handle(handle: Any) -> None:
    if handle is None:
        return
    remove = getattr(handle, "remove", None)
    if remove is not None:
        remove()


def add_mesh(
    scene: viser.SceneApi,
    name: str,
    mesh: trimesh.Trimesh,
    color: tuple[int, int, int],
    opacity: float,
) -> Any:
    if mesh_has_texture(mesh):
        return scene.add_mesh_trimesh(
            name,
            mesh=mesh,
            visible=True,
            cast_shadow=True,
            receive_shadow=True,
        )
    return scene.add_mesh_simple(
        name,
        vertices=np.asarray(mesh.vertices, dtype=np.float32),
        faces=np.asarray(mesh.faces, dtype=np.uint32),
        color=color,
        opacity=opacity,
        wireframe=False,
        side="double",
    )


def mesh_has_texture(mesh: trimesh.Trimesh) -> bool:
    uv = getattr(mesh.visual, "uv", None)
    material = getattr(mesh.visual, "material", None)
    return uv is not None and material is not None


def quat_wxyz_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    half = 0.5 * float(angle_rad)
    return np.asarray(
        [np.cos(half), *(np.sin(half) * axis)],
        dtype=np.float32,
    )


def resolve_textured_laptop_dir(dynamic_dir: Path, explicit: Path | None) -> Path | None:
    candidates = []
    if explicit is not None:
        candidates.append(explicit)
    candidates.append(dynamic_dir / "textured_laptop_parts")
    for path in candidates:
        resolved = path.resolve()
        if (resolved / "textured_laptop_manifest.json").exists():
            return resolved
    return None


def add_joint_axis(scene: viser.SceneApi, joint: dict[str, Any], axis_length: float) -> list[Any]:
    origin = np.asarray(joint["origin_xyz"], dtype=np.float32)
    axis = np.asarray(joint["axis_xyz"], dtype=np.float32)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    p0 = origin - axis * axis_length * 0.5
    p1 = origin + axis * axis_length * 0.5
    handles = [
        scene.add_line_segments(
            f"/joints/{joint['name']}/axis",
            points=np.asarray([[p0, p1]], dtype=np.float32),
            colors=JOINT_COLOR,
            line_width=7,
        ),
        scene.add_icosphere(
            f"/joints/{joint['name']}/origin",
            radius=0.012,
            color=JOINT_COLOR,
            position=origin,
            subdivisions=2,
        ),
    ]
    label = (
        f"{joint['name']} angle axis="
        f"[{axis[0]:.2f},{axis[1]:.2f},{axis[2]:.2f}]"
    )
    handles.append(
        scene.add_label(
            f"/joints/{joint['name']}/label",
            text=label,
            position=origin + np.asarray([0.0, -0.035, 0.0], dtype=np.float32),
            font_size_mode="screen",
            font_screen_scale=0.75,
            anchor="bottom-center",
        )
    )
    return handles


def add_rgb_reference(
    scene: viser.SceneApi,
    meta: dict[str, Any],
    export_root: Path,
    frame: int,
    scale: float,
    rgb_path_override: Path | None = None,
    camera_to_world: np.ndarray | None = None,
) -> Any:
    rgb_path = rgb_path_override or (export_root / "rgb_right_png" / f"{frame_name(frame)}.png")
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    intr = meta["rgb_intrinsics_right"]
    width = int(meta["rgb_width_per_eye"])
    height = int(meta["rgb_height_per_eye"])
    fov_y = float(2.0 * np.arctan2(height * 0.5, float(intr["fy"])))
    aspect = float(width / height)
    wxyz = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    position = np.zeros(3, dtype=np.float32)
    if camera_to_world is not None:
        transform = np.asarray(camera_to_world, dtype=np.float64)
        if transform.shape != (4, 4):
            raise ValueError(f"camera_to_world must be 4x4, got {transform.shape}")
        quat_xyzw = Rotation.from_matrix(transform[:3, :3]).as_quat()
        wxyz = np.asarray([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)
        position = transform[:3, 3].astype(np.float32)
    return scene.add_camera_frustum(
        "/input_camera",
        fov=fov_y,
        aspect=aspect,
        scale=float(scale),
        line_width=2.0,
        color=(30, 30, 30),
        image=rgb,
        format="auto",
        variant="wireframe",
        wxyz=wxyz,
        position=position,
    )


def add_current_rgbd_point_cloud(
    scene: viser.SceneApi,
    meta: dict[str, Any],
    export_root: Path,
    frame: int,
    convention: str,
    depth_min_m: float,
    depth_max_m: float,
    max_points: int,
    point_size: float,
    points_to_world: np.ndarray | None = None,
) -> Any:
    rgb_path = export_root / "rgb_right_png" / f"{frame_name(frame)}.png"
    depth_path = export_root / "depth_meters_npy" / f"{frame_name(frame)}.meters.npy"
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    depth_m = np.load(depth_path)
    points_right, u, v, inside = depth_points_in_right_camera(
        meta,
        depth_m,
        convention,
        depth_min_m=depth_min_m,
        depth_max_m=depth_max_m,
    )
    idx = np.flatnonzero(inside)
    if idx.size > max_points:
        rng = np.random.default_rng(frame + 31)
        idx = rng.choice(idx, size=max_points, replace=False)
    width = int(meta["rgb_width_per_eye"])
    height = int(meta["rgb_height_per_eye"])
    ui = np.clip(np.round(u[idx]).astype(np.int64), 0, width - 1)
    vi = np.clip(np.round(v[idx]).astype(np.int64), 0, height - 1)
    colors = np.ascontiguousarray(rgb[vi, ui, :3].astype(np.uint8))
    points = np.ascontiguousarray(points_right[idx].astype(np.float32))
    if points_to_world is not None:
        transform = np.asarray(points_to_world, dtype=np.float64)
        if transform.shape != (4, 4):
            raise ValueError(f"points_to_world must be 4x4, got {transform.shape}")
        points = np.ascontiguousarray(
            (points.astype(np.float64) @ transform[:3, :3].T + transform[:3, 3]).astype(np.float32)
        )
    return scene.add_point_cloud(
        "/current_rgbd/depth_points",
        points=points,
        colors=colors,
        point_size=point_size,
        point_shape="circle",
    )


def load_tracks(dynamic_dir: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    points_path = dynamic_dir / "tracked_points_frame_camera.npy"
    vis_path = dynamic_dir / "tracks_visibility.npy"
    if not points_path.exists() or not vis_path.exists():
        return None, None
    return np.load(points_path), np.load(vis_path).astype(bool)


def hand_paths(egoforce_dir: Path, frame: int) -> dict[str, Path]:
    stem = frame_name(frame)
    return {
        "left_hand": egoforce_dir / f"{stem}_left_hand.obj",
        "left_arm": egoforce_dir / f"{stem}_left_arm.obj",
        "left_hand_arm": egoforce_dir / f"{stem}_left_hand_arm.obj",
        "right_hand": egoforce_dir / f"{stem}_right_hand.obj",
        "right_arm": egoforce_dir / f"{stem}_right_arm.obj",
        "right_hand_arm": egoforce_dir / f"{stem}_right_hand_arm.obj",
    }


def entry_mesh_path(entry: dict[str, Any], side: str, kind: str, fallback: Path) -> Path:
    key = f"{side}_{kind}_mesh"
    if key in entry:
        return Path(entry[key])
    return fallback


def resolve_hand_mesh_path(
    entry: dict[str, Any],
    egoforce_dir: Path,
    frame: int,
    side: str,
    kind: str,
    hand_source: str,
) -> Path:
    fallback = hand_paths(egoforce_dir, frame)[f"{side}_{kind}"]
    if hand_source == "egoforce":
        return fallback
    return entry_mesh_path(entry, side, kind, fallback)


def main() -> None:
    args = parse_args()
    dynamic_dir = args.dynamic_dir.resolve()
    export_root = args.export_root.resolve()
    egoforce_dir = args.egoforce_dir.resolve()
    manifest = read_json(dynamic_dir / "dynamic_manifest.json")
    meta = read_json(export_root / "manifest.json")
    convention = args.depth_convention or manifest.get("depth", {}).get("convention", "camera_to_rig")
    frame_entries = manifest["frames"]
    frame_indices = [int(entry["frame"]) for entry in frame_entries]
    tracked_points, tracked_visibility = load_tracks(dynamic_dir)
    manifest_hand_side = str(manifest.get("hand_side") or "left")
    display_hand_sides = ("left", "right") if args.hand_side == "both" else (args.hand_side or manifest_hand_side,)
    textured_laptop_dir = resolve_textured_laptop_dir(dynamic_dir, args.textured_laptop_dir) if args.use_textured_laptop else None
    textured_laptop_manifest: dict[str, Any] | None = None
    if textured_laptop_dir is not None:
        textured_laptop_manifest = read_json(textured_laptop_dir / "textured_laptop_manifest.json")

    server = viser.ViserServer(host=args.host, port=args.port)
    scene = server.scene
    scene.set_up_direction("-y")
    scene.add_frame("/camera_frame", axes_length=0.08, axes_radius=0.003)

    handles_by_name: dict[str, Any] = {}
    current_index = 0

    gui = server.gui
    play_control = gui.add_checkbox("Play", initial_value=False)
    frame_control = gui.add_slider("Frame", min=0, max=max(0, len(frame_entries) - 1), step=1, initial_value=0)
    fps_control = gui.add_slider("FPS", min=1.0, max=30.0, step=1.0, initial_value=float(args.fps))
    show_laptop = gui.add_checkbox("Show laptop", initial_value=bool(args.show_laptop_initial))
    show_hand = gui.add_checkbox("Show hand", initial_value=True)
    show_arm = gui.add_checkbox("Show arm", initial_value=bool(args.show_arm_initial))
    show_rgbd = gui.add_checkbox("Show RGB-D", initial_value=bool(args.show_rgbd))
    show_rgb = gui.add_checkbox("Show RGB image", initial_value=bool(args.show_rgb))
    show_tracks = gui.add_checkbox("Show tracked points", initial_value=True)
    show_joint = gui.add_checkbox("Show joint", initial_value=True)
    show_labels = gui.add_checkbox("Show labels", initial_value=True)
    prev_button = gui.add_button("Prev frame")
    next_button = gui.add_button("Next frame")

    def remember(handle: Any) -> Any:
        name = getattr(handle, "name", None)
        if isinstance(name, str):
            handles_by_name[name] = handle
            try:
                handle.visible = True
            except Exception:
                pass
        return handle

    def hide_prefix(prefix: str) -> None:
        for name, handle in list(handles_by_name.items()):
            if name == prefix or name.startswith(prefix.rstrip("/") + "/"):
                try:
                    handle.visible = False
                except Exception:
                    remove_handle(handle)

    def show_prefix(prefix: str) -> None:
        for name, handle in list(handles_by_name.items()):
            if name == prefix or name.startswith(prefix.rstrip("/") + "/"):
                try:
                    handle.visible = True
                except Exception:
                    pass

    preloaded_screen: list[Any] = []
    preloaded_hands: dict[tuple[int, str, str], Any] = {}
    preloaded_bounds: dict[str, np.ndarray] = {}
    textured_base_handle: Any | None = None
    textured_screen_handle: Any | None = None
    textured_screen_origin: np.ndarray | None = None
    textured_screen_axis: np.ndarray | None = None
    if args.preload_dynamic:
        if textured_laptop_manifest is not None:
            parts = textured_laptop_manifest.get("parts", {})
            base_label = str(textured_laptop_manifest.get("base_label", manifest.get("base_part_label", "14")))
            screen_label = str(textured_laptop_manifest.get("screen_label", manifest.get("screen_part_label", "15")))
            if base_label in parts:
                base_mesh = load_mesh(Path(parts[base_label]["camera_mesh"]))
                textured_base_handle = remember(
                    add_mesh(scene, f"/laptop/part_{base_label}", base_mesh, PART_COLORS.get(base_label, (180, 180, 180)), 1.0)
                )
                textured_base_handle.visible = False
                preloaded_bounds["base"] = np.asarray(base_mesh.bounds, dtype=np.float32)
            if screen_label in parts:
                screen_mesh = load_mesh(Path(parts[screen_label]["camera_mesh"]))
                joint = textured_laptop_manifest.get("joint_camera") or manifest.get("joint_align_camera")
                textured_screen_origin = np.asarray(joint["origin_xyz"], dtype=np.float32)
                textured_screen_axis = np.asarray(joint["axis_xyz"], dtype=np.float64)
                textured_screen_axis = textured_screen_axis / (np.linalg.norm(textured_screen_axis) + 1e-12)
                screen_local = screen_mesh.copy()
                screen_local.vertices = np.asarray(screen_local.vertices, dtype=np.float64) - textured_screen_origin.astype(np.float64)
                textured_screen_handle = remember(
                    add_mesh(
                        scene,
                        f"/laptop/part_{screen_label}",
                        screen_local,
                        PART_COLORS.get(screen_label, (180, 180, 180)),
                        1.0,
                    )
                )
                textured_screen_handle.position = textured_screen_origin
                textured_screen_handle.wxyz = quat_wxyz_from_axis_angle(textured_screen_axis, 0.0)
                textured_screen_handle.visible = False
        elif args.static_base and frame_entries:
            base_mesh = load_mesh(Path(frame_entries[0]["base_mesh"]))
            handle = remember(add_mesh(scene, "/laptop/part_14", base_mesh, PART_COLORS["14"], args.laptop_opacity))
            handle.visible = False
            preloaded_bounds["base"] = np.asarray(base_mesh.bounds, dtype=np.float32)

        for idx, entry in enumerate(frame_entries):
            if textured_laptop_manifest is None:
                screen_mesh = load_mesh(Path(entry["screen_mesh"]))
                handle = remember(
                    add_mesh(scene, f"/preload/laptop/screen_{idx:06d}", screen_mesh, PART_COLORS["15"], args.laptop_opacity)
                )
                handle.visible = False
                preloaded_screen.append(handle)
            for side in display_hand_sides:
                hand_path = resolve_hand_mesh_path(entry, egoforce_dir, int(entry["frame"]), side, "hand", args.hand_source)
                if hand_path.exists():
                    hand_handle = remember(
                        add_mesh(scene, f"/preload/hand/{side}_hand_{idx:06d}", load_mesh(hand_path), HAND_COLOR, args.hand_opacity)
                    )
                    hand_handle.visible = False
                    preloaded_hands[(idx, side, "hand")] = hand_handle
                arm_path = resolve_hand_mesh_path(entry, egoforce_dir, int(entry["frame"]), side, "arm", args.hand_source)
                if arm_path.exists():
                    arm_handle = remember(
                        add_mesh(scene, f"/preload/hand/{side}_arm_{idx:06d}", load_mesh(arm_path), ARM_COLOR, 0.55)
                    )
                    arm_handle.visible = False
                    preloaded_hands[(idx, side, "arm")] = arm_handle

        if frame_entries:
            for joint in read_json(Path(frame_entries[0]["joint_json"])).get("joints", []):
                for handle in add_joint_axis(scene, joint, args.axis_length):
                    remember(handle).visible = False

    def show_frame(index: int) -> None:
        nonlocal current_index
        current_index = int(index) % len(frame_entries)
        entry = frame_entries[current_index]
        frame = int(entry["frame"])

        rgb_override = Path(entry["rgb_path"]) if entry.get("rgb_path") else None
        camera_to_world = None
        if entry.get("camera_to_frame0_matrix") is not None:
            camera_to_world = np.asarray(entry["camera_to_frame0_matrix"], dtype=np.float64)
        hide_prefix("/input_camera")
        if bool(show_rgb.value):
            remember(
                add_rgb_reference(
                    scene,
                    meta,
                    export_root,
                    frame,
                    args.rgb_frustum_scale,
                    rgb_override,
                    camera_to_world,
                )
            )

        hide_prefix("/current_rgbd")
        if bool(show_rgbd.value):
            standard_only = manifest.get("depth_display_mode") == "standard_frames_only"
            depth_frame_value = entry.get("standard_depth_frame")
            if depth_frame_value is not None or not standard_only:
                depth_frame = int(depth_frame_value if depth_frame_value is not None else entry.get("pose_frame", frame))
                depth_camera_to_world = camera_to_world
                if entry.get("standard_depth_camera_to_frame0_matrix") is not None:
                    depth_camera_to_world = np.asarray(
                        entry["standard_depth_camera_to_frame0_matrix"],
                        dtype=np.float64,
                    )
                remember(
                    add_current_rgbd_point_cloud(
                        scene,
                        meta,
                        export_root,
                        depth_frame,
                        convention,
                        args.depth_min_m,
                        args.depth_max_m,
                        args.rgbd_max_points,
                        args.rgbd_point_size,
                        depth_camera_to_world,
                    )
                )

        if args.preload_dynamic:
            hide_prefix("/laptop")
            hide_prefix("/preload/laptop")
            hide_prefix("/preload/hand")
            hide_prefix("/hand")
            hide_prefix("/labels")
            hide_prefix("/joints")
            hide_prefix("/scene_bounds")
            if bool(show_laptop.value):
                if textured_base_handle is not None or textured_screen_handle is not None:
                    if textured_base_handle is not None:
                        textured_base_handle.visible = True
                    if (
                        textured_screen_handle is not None
                        and textured_screen_origin is not None
                        and textured_screen_axis is not None
                    ):
                        textured_screen_handle.position = textured_screen_origin
                        textured_screen_handle.wxyz = quat_wxyz_from_axis_angle(
                            textured_screen_axis,
                            float(entry.get("angle_rad", 0.0)),
                        )
                        textured_screen_handle.visible = True
                elif args.static_base:
                    show_prefix("/laptop/part_14")
                elif frame_entries:
                    base_mesh = load_mesh(Path(entry["base_mesh"]))
                    remember(add_mesh(scene, "/laptop/part_14", base_mesh, PART_COLORS["14"], args.laptop_opacity))
                if current_index < len(preloaded_screen):
                    preloaded_screen[current_index].visible = True
                if bool(show_labels.value):
                    if "base" in preloaded_bounds:
                        center = preloaded_bounds["base"].mean(axis=0)
                        remember(
                            scene.add_label(
                                "/labels/laptop_part_14",
                                text="part_14",
                                position=center,
                                font_size_mode="screen",
                                font_screen_scale=0.75,
                                anchor="center-center",
                            )
                        )
            if bool(show_joint.value):
                show_prefix("/joints")
            if bool(show_hand.value):
                for side in display_hand_sides:
                    handle = preloaded_hands.get((current_index, side, "hand"))
                    if handle is not None:
                        handle.visible = True
                    arm_handle = preloaded_hands.get((current_index, side, "arm"))
                    if arm_handle is not None:
                        arm_handle.visible = bool(show_arm.value)
            return

        all_bounds = []
        hide_prefix("/laptop")
        hide_prefix("/labels")
        if bool(show_laptop.value):
            for label, path_key in (("14", "base_mesh"), ("15", "screen_mesh")):
                mesh = load_mesh(Path(entry[path_key]))
                remember(
                    add_mesh(
                        scene,
                        f"/laptop/part_{label}",
                        mesh,
                        PART_COLORS.get(label, (180, 180, 180)),
                        args.laptop_opacity,
                    )
                )
                all_bounds.append(np.asarray(mesh.bounds, dtype=np.float32))
                if bool(show_labels.value):
                    center = np.asarray(mesh.bounds, dtype=np.float32).mean(axis=0)
                    remember(
                        scene.add_label(
                            f"/labels/laptop_part_{label}",
                            text=f"part_{label}",
                            position=center,
                            font_size_mode="screen",
                            font_screen_scale=0.75,
                            anchor="center-center",
                        )
                    )

        hide_prefix("/joints")
        if bool(show_joint.value):
            joints = read_json(Path(entry["joint_json"])).get("joints", [])
            for joint in joints:
                for handle in add_joint_axis(scene, joint, args.axis_length):
                    remember(handle)

        hide_prefix("/tracks")
        if bool(show_tracks.value) and tracked_points is not None and tracked_visibility is not None:
            pts = tracked_points[current_index]
            valid = np.isfinite(pts).all(axis=1) & tracked_visibility[current_index]
            if valid.any():
                colors = np.tile(np.asarray(TRACK_COLOR, dtype=np.uint8), (int(valid.sum()), 1))
                remember(
                    scene.add_point_cloud(
                        "/tracks/screen_points",
                        points=np.ascontiguousarray(pts[valid].astype(np.float32)),
                        colors=colors,
                        point_size=args.track_point_size,
                        point_shape="circle",
                    )
                )

        hide_prefix("/hand")
        if bool(show_hand.value):
            paths = hand_paths(egoforce_dir, frame)
            for side in display_hand_sides:
                hand_path = paths[f"{side}_hand"] if args.hand_source == "egoforce" else entry_mesh_path(entry, side, "hand", paths[f"{side}_hand"])
                if hand_path.exists():
                    mesh = load_mesh(hand_path)
                    remember(add_mesh(scene, f"/hand/{side}_hand", mesh, HAND_COLOR, args.hand_opacity))
                    all_bounds.append(np.asarray(mesh.bounds, dtype=np.float32))
                if bool(show_arm.value):
                    arm_path = paths[f"{side}_arm"] if args.hand_source == "egoforce" else entry_mesh_path(entry, side, "arm", paths[f"{side}_arm"])
                    if arm_path.exists():
                        remember(add_mesh(scene, f"/hand/{side}_arm", load_mesh(arm_path), ARM_COLOR, 0.55))

        hide_prefix("/scene_bounds")
        if bool(args.show_scene_bounds) and all_bounds:
            bounds = np.stack(all_bounds)
            xyz_min = bounds[:, 0, :].min(axis=0)
            xyz_max = bounds[:, 1, :].max(axis=0)
            center = (xyz_min + xyz_max) * 0.5
            dims = xyz_max - xyz_min
            remember(
                scene.add_box(
                    "/scene_bounds",
                    dimensions=dims,
                    color=(40, 40, 40),
                    wireframe=True,
                    opacity=0.16,
                    position=center,
                )
            )

    show_frame(0)

    @frame_control.on_update
    def _(_event: Any) -> None:
        show_frame(int(frame_control.value))

    @prev_button.on_click
    def _(_event: Any) -> None:
        next_index = (current_index - 1) % len(frame_entries)
        frame_control.value = next_index
        show_frame(next_index)

    @next_button.on_click
    def _(_event: Any) -> None:
        next_index = (current_index + 1) % len(frame_entries)
        frame_control.value = next_index
        show_frame(next_index)

    for checkbox in (show_laptop, show_hand, show_arm, show_rgbd, show_rgb, show_tracks, show_joint, show_labels):
        @checkbox.on_update
        def _(_event: Any) -> None:
            show_frame(current_index)

    url = f"http://localhost:{args.port}" if args.host == "0.0.0.0" else f"http://{args.host}:{args.port}"
    print(f"Dynamic laptop + hand viser is running: {url}", flush=True)
    print(f"Dynamic dir: {dynamic_dir}", flush=True)
    print(f"EgoForce dir: {egoforce_dir}", flush=True)
    print(f"Frames: {frame_indices[0]}-{frame_indices[-1]} ({len(frame_indices)} frames)", flush=True)
    print(f"Displayed hand side(s): {', '.join(display_hand_sides)}", flush=True)
    print(f"Hand source: {args.hand_source}", flush=True)
    print(f"Preload dynamic meshes: {bool(args.preload_dynamic)}", flush=True)
    if textured_laptop_manifest is not None:
        print(f"Textured laptop: {textured_laptop_dir}", flush=True)
    print("Controls: Play, Prev frame, Next frame, and visibility checkboxes.", flush=True)

    try:
        while True:
            if bool(play_control.value) and len(frame_entries) > 1:
                fps = max(float(fps_control.value), 1e-6)
                time.sleep(1.0 / fps)
                next_index = (current_index + 1) % len(frame_entries)
                frame_control.value = next_index
                show_frame(next_index)
            else:
                time.sleep(0.05)
    except KeyboardInterrupt:
        print("Stopped viser server.")


if __name__ == "__main__":
    main()
