#!/usr/bin/env python3
"""Visualize textured PAct parts and predicted articulation in viser."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
import viser


PART_COLORS = [
    (69, 123, 157),
    (230, 57, 70),
    (42, 157, 143),
    (244, 162, 97),
]


def load_glb_meshes(path: Path) -> list[trimesh.Trimesh]:
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Trimesh):
        return [loaded]

    meshes: list[trimesh.Trimesh] = []
    for node_name in loaded.graph.nodes_geometry:
        transform, geometry_name = loaded.graph.get(node_name)
        mesh = loaded.geometry[geometry_name].copy()
        if not isinstance(mesh, trimesh.Trimesh):
            continue
        mesh.apply_transform(transform)
        meshes.append(mesh)
    if not meshes:
        raise ValueError(f"No mesh geometry found in {path}")
    return meshes


def joint_quaternion(axis: np.ndarray, angle_degrees: float) -> np.ndarray:
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-8:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    half_angle = np.deg2rad(angle_degrees) * 0.5
    xyz = axis / axis_norm * np.sin(half_angle)
    return np.asarray([np.cos(half_angle), *xyz], dtype=np.float32)


def add_mesh(
    scene: viser.SceneApi,
    name: str,
    mesh: trimesh.Trimesh,
    color: tuple[int, int, int],
    opacity: float,
) -> Any:
    uv = getattr(mesh.visual, "uv", None)
    material = getattr(mesh.visual, "material", None)
    if uv is not None and material is not None:
        return scene.add_mesh_trimesh(name, mesh=mesh)
    return scene.add_mesh_simple(
        name,
        vertices=np.asarray(mesh.vertices, dtype=np.float32),
        faces=np.asarray(mesh.faces, dtype=np.uint32),
        color=color,
        opacity=opacity,
        side="double",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve textured PAct parts in viser.")
    parser.add_argument("--object-json", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8097)
    parser.add_argument("--opacity", type=float, default=1.0)
    parser.add_argument("--axis-length", type=float, default=1.1)
    args = parser.parse_args()

    object_json = args.object_json.resolve()
    if not object_json.exists():
        raise FileNotFoundError(object_json)
    record = json.loads(object_json.read_text(encoding="utf-8"))
    parts = record.get("diffuse_tree", [])
    if not parts:
        raise ValueError(f"No parts in {object_json}")

    server = viser.ViserServer(host=args.host, port=args.port)
    scene = server.scene
    gui = server.gui
    scene.set_up_direction("+y")
    scene.add_grid(
        "/grid",
        width=1.6,
        height=1.6,
        plane="xz",
        cell_size=0.05,
        cell_thickness=0.001,
        section_size=0.25,
        section_thickness=0.002,
        plane_opacity=0.08,
    )

    all_bounds: list[np.ndarray] = []
    mesh_handles: list[Any] = []
    moving_parts: list[tuple[dict[str, Any], Any, np.ndarray]] = []
    part_stats: list[dict[str, Any]] = []

    for part_index, part in enumerate(parts):
        part_id = int(part["id"])
        part_name = str(part.get("name", f"part_{part_id}"))
        joint = part.get("joint", {})
        joint_type = str(joint.get("type", "fixed"))
        axis_record = joint.get("axis", {})
        origin = np.asarray(axis_record.get("origin", [0.0, 0.0, 0.0]), dtype=np.float32)
        axis = np.asarray(axis_record.get("direction", [0.0, 0.0, 0.0]), dtype=np.float32)
        is_moving = joint_type != "fixed" and float(np.linalg.norm(axis)) > 1e-8

        frame = None
        if is_moving:
            frame = scene.add_frame(
                f"/parts/part_{part_id}_{part_name}_motion",
                position=origin,
                show_axes=False,
            )
            moving_parts.append((part, frame, axis))

        part_vertices = 0
        part_faces = 0
        for glb_index, relative_path in enumerate(part.get("glb", [])):
            glb_path = object_json.parent / relative_path
            if not glb_path.exists():
                raise FileNotFoundError(glb_path)
            for geometry_index, mesh in enumerate(load_glb_meshes(glb_path)):
                all_bounds.append(np.asarray(mesh.bounds, dtype=np.float32))
                part_vertices += len(mesh.vertices)
                part_faces += len(mesh.faces)
                mesh = mesh.copy()
                if is_moving:
                    mesh.vertices = np.asarray(mesh.vertices) - origin
                    node_name = (
                        f"/parts/part_{part_id}_{part_name}_motion/"
                        f"mesh_{glb_index}_{geometry_index}"
                    )
                else:
                    node_name = f"/parts/part_{part_id}_{part_name}/mesh_{glb_index}_{geometry_index}"
                mesh_handles.append(
                    add_mesh(
                        scene,
                        node_name,
                        mesh,
                        PART_COLORS[part_index % len(PART_COLORS)],
                        args.opacity,
                    )
                )

        part_stats.append(
            {
                "id": part_id,
                "name": part_name,
                "vertices": part_vertices,
                "faces": part_faces,
                "joint_type": joint_type,
            }
        )

    if not all_bounds:
        raise ValueError(f"No GLB meshes referenced by {object_json}")

    bounds = np.stack(all_bounds)
    xyz_min = bounds[:, 0].min(axis=0)
    xyz_max = bounds[:, 1].max(axis=0)
    center = (xyz_min + xyz_max) * 0.5
    extent = float(np.max(xyz_max - xyz_min))
    server.initial_camera.position = center + extent * np.asarray([1.4, 0.9, 1.4])
    server.initial_camera.look_at = center
    server.initial_camera.up = np.asarray([0.0, 1.0, 0.0])
    server.initial_camera.fov = float(np.deg2rad(50.0))

    @server.on_client_connect
    def _frame_camera(client: viser.ClientHandle) -> None:
        narrow_screen_scale = max(1.0, 1.0 / client.camera.aspect)
        client.camera.up_direction = np.asarray([0.0, 1.0, 0.0])
        client.camera.position = center + extent * narrow_screen_scale * np.asarray([1.4, 0.9, 1.4])
        client.camera.look_at = center
        client.camera.fov = float(np.deg2rad(50.0))

    scene.add_frame("/object_frame", position=center, axes_length=0.16, axes_radius=0.005)

    joint_handles: list[Any] = []
    for part, _, axis in moving_parts:
        part_id = int(part["id"])
        joint = part["joint"]
        origin = np.asarray(joint["axis"]["origin"], dtype=np.float32)
        axis = axis / np.linalg.norm(axis)
        p0 = origin - axis * args.axis_length * 0.5
        p1 = origin + axis * args.axis_length * 0.5
        joint_handles.append(
            scene.add_line_segments(
                f"/joints/part_{part_id}/axis",
                points=np.asarray([[p0, p1]], dtype=np.float32),
                colors=(230, 57, 70),
                line_width=6,
            )
        )
        joint_handles.append(
            scene.add_icosphere(
                f"/joints/part_{part_id}/origin",
                radius=0.022,
                color=(230, 57, 70),
                position=origin,
                subdivisions=2,
            )
        )

    show_meshes = gui.add_checkbox("Show textured meshes", initial_value=True)
    show_joints = gui.add_checkbox("Show joint axes", initial_value=True)

    @show_meshes.on_update
    def _(_) -> None:
        for handle in mesh_handles:
            handle.visible = bool(show_meshes.value)

    @show_joints.on_update
    def _(_) -> None:
        for handle in joint_handles:
            handle.visible = bool(show_joints.value)

    for part, frame, axis in moving_parts:
        part_id = int(part["id"])
        part_name = str(part.get("name", f"part_{part_id}"))
        joint_range = [float(value) for value in part["joint"].get("range", [0.0, 0.0])]
        lower, upper = min(joint_range), max(joint_range)
        initial = 0.0 if lower <= 0.0 <= upper else joint_range[0]
        angle = gui.add_slider(
            f"{part_name} angle (deg)",
            min=lower,
            max=upper,
            step=0.5,
            initial_value=initial,
        )

        @angle.on_update
        def _(_, angle=angle, frame=frame, axis=axis) -> None:
            frame.wxyz = joint_quaternion(axis, float(angle.value))

    url = f"http://localhost:{args.port}" if args.host == "0.0.0.0" else f"http://{args.host}:{args.port}"
    print(f"PAct viser is running: {url}", flush=True)
    print(f"Object: {object_json}", flush=True)
    print(json.dumps({"parts": part_stats, "bounds": [xyz_min.tolist(), xyz_max.tolist()]}, indent=2), flush=True)

    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
