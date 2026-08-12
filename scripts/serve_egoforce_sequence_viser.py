#!/usr/bin/env python3
"""Serve raw EgoForce sequence outputs without any object/contact constraints."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import trimesh
import viser

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vlm_sam2_recon.stages.camera_alignment import depth_points_in_right_camera  # noqa: E402


DEFAULT_EGOFORCE_DIR = PROJECT_ROOT / "outputs/egoforce_rgb_right_15fps"
DEFAULT_RGB_DIR = PROJECT_ROOT / "outputs/tracker_rgb_right_15fps"
DEFAULT_EXPORT_ROOT = Path("/code/3DVideo_2026-07-01-19-24-06-667_spatialmp4_export")

HAND_COLOR = (245, 190, 135)
ARM_COLOR = (150, 120, 95)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve raw EgoForce hand meshes in viser.")
    parser.add_argument("--egoforce-dir", type=Path, default=DEFAULT_EGOFORCE_DIR)
    parser.add_argument("--rgb-dir", type=Path, default=DEFAULT_RGB_DIR)
    parser.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT_ROOT)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8122)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--hand-side", choices=["left", "right", "both"], default="right")
    parser.add_argument("--show-rgb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-rgbd", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-arm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rgb-frustum-scale", type=float, default=0.25)
    parser.add_argument("--tracker-fps", type=float, default=15.0)
    parser.add_argument("--tracker-time-offset-s", type=float, default=None)
    parser.add_argument("--max-rgbd-time-gap-s", type=float, default=0.16)
    parser.add_argument("--rgbd-max-points", type=int, default=45000)
    parser.add_argument("--rgbd-point-size", type=float, default=0.002)
    parser.add_argument("--depth-min-m", type=float, default=0.1)
    parser.add_argument("--depth-max-m", type=float, default=3.0)
    parser.add_argument(
        "--depth-convention",
        choices=["camera_to_rig", "rig_to_camera", "direct_same_camera"],
        default=None,
    )
    parser.add_argument("--hand-opacity", type=float, default=0.95)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def frame_name(frame: int) -> str:
    return f"{int(frame):06d}"


def load_frames_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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


def add_mesh(
    scene: viser.SceneApi,
    name: str,
    mesh: trimesh.Trimesh,
    color: tuple[int, int, int],
    opacity: float,
) -> Any:
    return scene.add_mesh_simple(
        name,
        vertices=np.asarray(mesh.vertices, dtype=np.float32),
        faces=np.asarray(mesh.faces, dtype=np.uint32),
        color=color,
        opacity=opacity,
        wireframe=False,
        side="double",
    )


def remove_handle(handle: Any) -> None:
    if handle is not None and hasattr(handle, "remove"):
        handle.remove()


def add_rgb_reference(
    scene: viser.SceneApi,
    meta: dict[str, Any],
    rgb_path: Path,
    scale: float,
) -> Any | None:
    if not rgb_path.exists():
        return None
    image = np.asarray(Image.open(rgb_path).convert("RGB"))
    intr = meta["rgb_intrinsics_right"]
    height, width = image.shape[:2]
    fy = float(intr["fy"])
    fov = 2.0 * np.arctan2(height * 0.5, max(fy, 1e-6))
    aspect = float(width) / float(max(height, 1))
    return scene.add_camera_frustum(
        "/input_rgb",
        fov=float(fov),
        aspect=aspect,
        scale=float(scale),
        image=image,
    )


def depth_timeline(export_root: Path) -> tuple[np.ndarray, list[dict[str, str]]]:
    rows = load_frames_csv(export_root / "frames.csv")
    usable: list[dict[str, str]] = []
    times: list[float] = []
    for row in rows:
        depth_rel = row.get("depth_meters_npy", "")
        if not depth_rel:
            continue
        if not (export_root / depth_rel).exists():
            continue
        usable.append(row)
        times.append(float(row["depth_timestamp_s"]))
    return np.asarray(times, dtype=np.float64), usable


def tracker_frame_time_s(frame: int, depth_rows: list[dict[str, str]], tracker_fps: float, offset_s: float | None) -> float:
    if not depth_rows:
        return float(frame) / max(float(tracker_fps), 1e-6)
    # The 15fps tracker frames are extracted from the same clip but do not carry
    # their own timestamps.  Anchor frame 0 to the first exported RGB-D timestamp
    # unless the caller explicitly supplies an offset.
    anchor = float(depth_rows[0].get("rgb_timestamp_s") or depth_rows[0]["depth_timestamp_s"])
    offset = 0.0 if offset_s is None else float(offset_s)
    return anchor + offset + float(frame) / max(float(tracker_fps), 1e-6)


def nearest_depth_row(
    frame: int,
    depth_times: np.ndarray,
    depth_rows: list[dict[str, str]],
    tracker_fps: float,
    offset_s: float | None,
    max_gap_s: float,
) -> tuple[int, dict[str, str], float] | None:
    if depth_times.size == 0:
        return None
    hand_time = tracker_frame_time_s(frame, depth_rows, tracker_fps, offset_s)
    idx = int(np.argmin(np.abs(depth_times - hand_time)))
    gap = abs(float(depth_times[idx]) - hand_time)
    if gap > float(max_gap_s):
        return None
    return int(idx), depth_rows[idx], gap


def add_rgbd_point_cloud(
    scene: viser.SceneApi,
    meta: dict[str, Any],
    export_root: Path,
    row: dict[str, str],
    convention: str,
    depth_min_m: float,
    depth_max_m: float,
    max_points: int,
    point_size: float,
) -> Any | None:
    rgb_path = export_root / row["right_rgb_png"]
    depth_path = export_root / row["depth_meters_npy"]
    if not rgb_path.exists() or not depth_path.exists():
        return None
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
    if idx.size == 0:
        return None
    if idx.size > max_points:
        seed = int(row["index"]) + 173
        rng = np.random.default_rng(seed)
        idx = rng.choice(idx, size=int(max_points), replace=False)
    width = int(meta["rgb_width_per_eye"])
    height = int(meta["rgb_height_per_eye"])
    ui = np.clip(np.round(u[idx]).astype(np.int64), 0, width - 1)
    vi = np.clip(np.round(v[idx]).astype(np.int64), 0, height - 1)
    colors = np.ascontiguousarray(rgb[vi, ui, :3].astype(np.uint8))
    points = np.ascontiguousarray(points_right[idx].astype(np.float32))
    return scene.add_point_cloud(
        "/rgbd/depth_points",
        points=points,
        colors=colors,
        point_size=float(point_size),
        point_shape="circle",
    )


def available_frames(egoforce_dir: Path, side: str) -> list[int]:
    frames: set[int] = set()
    sides = ("left", "right") if side == "both" else (side,)
    for current_side in sides:
        for path in egoforce_dir.glob(f"*_{current_side}_hand.obj"):
            try:
                frames.add(int(path.name[:6]))
            except ValueError:
                continue
    return sorted(frames)


def main() -> None:
    args = parse_args()
    egoforce_dir = args.egoforce_dir.resolve()
    rgb_dir = args.rgb_dir.resolve()
    export_root = args.export_root.resolve()
    meta = read_json(export_root / "manifest.json")
    convention = args.depth_convention or meta.get("depth", {}).get("convention", "camera_to_rig")
    depth_times, depth_rows = depth_timeline(export_root)
    frames = available_frames(egoforce_dir, args.hand_side)
    if not frames:
        raise FileNotFoundError(f"No EgoForce hand OBJ files found in {egoforce_dir}")

    server = viser.ViserServer(host=args.host, port=args.port)
    scene = server.scene
    scene.set_up_direction("-y")
    scene.add_frame("/camera_frame", axes_length=0.08, axes_radius=0.003)

    gui = server.gui
    play_control = gui.add_checkbox("Play", initial_value=False)
    frame_control = gui.add_slider("Frame", min=0, max=len(frames) - 1, step=1, initial_value=0)
    fps_control = gui.add_slider("FPS", min=1.0, max=30.0, step=1.0, initial_value=float(args.fps))
    show_rgb_control = gui.add_checkbox("Show RGB", initial_value=bool(args.show_rgb))
    show_rgbd_control = gui.add_checkbox("Show RGB-D depth", initial_value=bool(args.show_rgbd))
    show_hand_control = gui.add_checkbox("Show hand", initial_value=True)
    show_arm_control = gui.add_checkbox("Show arm", initial_value=bool(args.show_arm))
    prev_button = gui.add_button("Prev frame")
    next_button = gui.add_button("Next frame")

    frame_handles: dict[str, Any] = {}
    rgbd_handle: Any | None = None
    current_rgbd_depth_index: int | None = None
    current_index = 0
    sides = ("left", "right") if args.hand_side == "both" else (args.hand_side,)

    def clear_frame_scene() -> None:
        for handle in list(frame_handles.values()):
            remove_handle(handle)
        frame_handles.clear()

    def remember(name: str, handle: Any | None) -> None:
        if handle is not None:
            frame_handles[name] = handle

    def update_rgbd(frame: int, force: bool = False) -> None:
        nonlocal rgbd_handle, current_rgbd_depth_index
        if not bool(show_rgbd_control.value):
            remove_handle(rgbd_handle)
            rgbd_handle = None
            current_rgbd_depth_index = None
            return
        if not bool(meta.get("has_depth", True)):
            return
        match = nearest_depth_row(
            frame,
            depth_times,
            depth_rows,
            tracker_fps=args.tracker_fps,
            offset_s=args.tracker_time_offset_s,
            max_gap_s=args.max_rgbd_time_gap_s,
        )
        if match is None:
            remove_handle(rgbd_handle)
            rgbd_handle = None
            current_rgbd_depth_index = None
            return
        depth_index, depth_row, _gap = match
        if (not force) and current_rgbd_depth_index == depth_index and rgbd_handle is not None:
            return
        remove_handle(rgbd_handle)
        rgbd_handle = add_rgbd_point_cloud(
            scene,
            meta,
            export_root,
            depth_row,
            convention,
            depth_min_m=args.depth_min_m,
            depth_max_m=args.depth_max_m,
            max_points=args.rgbd_max_points,
            point_size=args.rgbd_point_size,
        )
        current_rgbd_depth_index = depth_index if rgbd_handle is not None else None

    def show_frame(index: int) -> None:
        nonlocal current_index
        current_index = int(index) % len(frames)
        frame = frames[current_index]
        stem = frame_name(frame)
        clear_frame_scene()
        if bool(show_rgb_control.value):
            remember("rgb", add_rgb_reference(scene, meta, rgb_dir / f"{stem}.png", args.rgb_frustum_scale))
        if bool(show_hand_control.value):
            for side in sides:
                hand_path = egoforce_dir / f"{stem}_{side}_hand.obj"
                if hand_path.exists():
                    remember(
                        f"{side}_hand",
                        add_mesh(scene, f"/egoforce/{side}_hand", load_mesh(hand_path), HAND_COLOR, args.hand_opacity),
                    )
                arm_path = egoforce_dir / f"{stem}_{side}_arm.obj"
                if bool(show_arm_control.value) and arm_path.exists():
                    remember(
                        f"{side}_arm",
                        add_mesh(scene, f"/egoforce/{side}_arm", load_mesh(arm_path), ARM_COLOR, 0.55),
                    )
        update_rgbd(frame)

    show_frame(0)

    @frame_control.on_update
    def _(_event: Any) -> None:
        show_frame(int(frame_control.value))

    @prev_button.on_click
    def _(_event: Any) -> None:
        frame_control.value = (current_index - 1) % len(frames)
        show_frame(int(frame_control.value))

    @next_button.on_click
    def _(_event: Any) -> None:
        frame_control.value = (current_index + 1) % len(frames)
        show_frame(int(frame_control.value))

    for checkbox in (show_rgb_control, show_hand_control, show_arm_control):
        @checkbox.on_update
        def _(_event: Any) -> None:
            show_frame(current_index)

    @show_rgbd_control.on_update
    def _(_event: Any) -> None:
        update_rgbd(frames[current_index], force=True)

    url = f"http://localhost:{args.port}" if args.host == "0.0.0.0" else f"http://{args.host}:{args.port}"
    print(f"Raw EgoForce viser is running: {url}", flush=True)
    print(f"EgoForce dir: {egoforce_dir}", flush=True)
    print(f"RGB dir: {rgb_dir}", flush=True)
    print(f"RGB-D export root: {export_root}", flush=True)
    print(f"RGB-D depth frames: {len(depth_rows)} at {meta.get('depth_fps', 'unknown')} fps", flush=True)
    print(f"RGB-D convention: {convention}", flush=True)
    print(f"Frames: {frames[0]}-{frames[-1]} ({len(frames)} detected hand frames)", flush=True)

    try:
        while True:
            if bool(play_control.value):
                fps = max(float(fps_control.value), 1e-6)
                time.sleep(1.0 / fps)
                frame_control.value = (current_index + 1) % len(frames)
                show_frame(int(frame_control.value))
            else:
                time.sleep(0.05)
    except KeyboardInterrupt:
        print("Stopped viser server.")


if __name__ == "__main__":
    main()
