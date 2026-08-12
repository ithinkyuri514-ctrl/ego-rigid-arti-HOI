#!/usr/bin/env python3
"""Visualize PhysX-Omni part meshes and URDF joints with viser."""

from __future__ import annotations

import argparse
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
import viser


DEFAULT_RESULT_DIR = Path(
    "/code/vlm_sam2_recon/outputs/physxomni/articulated/target_laptop/"
    "physx_base/vlm_sam2_target_laptop"
)

PART_COLORS = [
    (230, 57, 70),
    (69, 123, 157),
    (42, 157, 143),
    (244, 162, 97),
    (131, 56, 236),
    (233, 196, 106),
    (33, 158, 188),
    (255, 183, 3),
]

JOINT_COLOR = (255, 40, 40)
FIXED_JOINT_COLOR = (150, 150, 150)


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


def add_part_mesh(scene: viser.SceneApi, name: str, mesh: trimesh.Trimesh, color: tuple[int, int, int], opacity: float) -> None:
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.uint32)
    scene.add_mesh_simple(
        name,
        vertices=vertices,
        faces=faces,
        color=color,
        opacity=opacity,
        wireframe=False,
        side="double",
    )


def parse_vec(text: str | None, default: tuple[float, float, float]) -> np.ndarray:
    if not text:
        return np.asarray(default, dtype=np.float32)
    return np.asarray([float(x) for x in text.split()], dtype=np.float32)


def parse_joints(urdf_path: Path) -> list[dict]:
    if not urdf_path.exists():
        return []

    joints = []
    root = ET.parse(urdf_path).getroot()
    for joint in root.findall("joint"):
        origin = joint.find("origin")
        axis = joint.find("axis")
        parent = joint.find("parent")
        child = joint.find("child")
        limit = joint.find("limit")
        joints.append(
            {
                "name": joint.attrib.get("name", "joint"),
                "type": joint.attrib.get("type", "unknown"),
                "parent": parent.attrib.get("link") if parent is not None else None,
                "child": child.attrib.get("link") if child is not None else None,
                "origin": parse_vec(origin.attrib.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0)),
                "axis": parse_vec(axis.attrib.get("xyz") if axis is not None else None, (0.0, 0.0, 0.0)),
                "limit": limit.attrib if limit is not None else {},
            }
        )
    return joints


def add_joint_axis(scene: viser.SceneApi, joint: dict, axis_length: float) -> None:
    origin = np.asarray(joint["origin"], dtype=np.float32)
    axis = np.asarray(joint["axis"], dtype=np.float32)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-8:
        scene.add_icosphere(
            f"/joints/{joint['name']}/origin",
            radius=0.018,
            color=FIXED_JOINT_COLOR,
            position=origin,
            subdivisions=2,
        )
        return

    axis = axis / norm
    p0 = origin - axis * axis_length * 0.5
    p1 = origin + axis * axis_length * 0.5
    points = np.asarray([[p0, p1]], dtype=np.float32)
    scene.add_line_segments(
        f"/joints/{joint['name']}/axis",
        points=points,
        colors=JOINT_COLOR,
        line_width=5,
    )
    scene.add_icosphere(
        f"/joints/{joint['name']}/origin",
        radius=0.022,
        color=JOINT_COLOR,
        position=origin,
        subdivisions=2,
    )
    label = f"{joint['type']} axis: [{axis[0]:.2f}, {axis[1]:.2f}, {axis[2]:.2f}]"
    if joint.get("limit"):
        lower = joint["limit"].get("lower")
        upper = joint["limit"].get("upper")
        if lower is not None and upper is not None:
            label += f" limit=({float(lower):.2f}, {float(upper):.2f})"
    scene.add_label(
        f"/joints/{joint['name']}/label",
        text=label,
        position=origin + np.asarray([0.0, 0.0, 0.07], dtype=np.float32),
        font_size_mode="screen",
        font_screen_scale=0.8,
        anchor="bottom-center",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve PhysX-Omni mesh parts and URDF joints in viser.")
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8094)
    parser.add_argument("--opacity", type=float, default=0.92)
    parser.add_argument("--axis-length", type=float, default=1.05)
    args = parser.parse_args()

    result_dir = args.result_dir.resolve()
    info_path = result_dir / "basic_info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing PhysX-Omni basic_info.json: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    parts = info.get("parts", [])
    if not parts:
        raise ValueError(f"No parts in {info_path}")

    server = viser.ViserServer(host=args.host, port=args.port)
    scene = server.scene
    scene.set_up_direction("+z")
    scene.add_grid(
        "/grid",
        width=1.4,
        height=1.4,
        plane="xy",
        cell_size=0.05,
        cell_thickness=0.001,
        section_size=0.25,
        section_thickness=0.002,
        plane_opacity=0.08,
    )

    all_bounds = []
    for idx, part in enumerate(parts):
        label = int(part["label"])
        part_name = str(part.get("name", f"part_{label}"))
        mesh_path = result_dir / "objs" / str(label) / f"{label}.obj"
        if not mesh_path.exists():
            print(f"Skipping missing mesh: {mesh_path}", flush=True)
            continue
        mesh = load_mesh(mesh_path)
        color = PART_COLORS[idx % len(PART_COLORS)]
        add_part_mesh(scene, f"/parts/l_{label}_{part_name}", mesh, color=color, opacity=args.opacity)
        bounds = np.asarray(mesh.bounds, dtype=np.float32)
        all_bounds.append(bounds)
        center = bounds.mean(axis=0)
        scene.add_label(
            f"/labels/l_{label}",
            text=f"l_{label}: {part_name}",
            position=center,
            font_size_mode="screen",
            font_screen_scale=0.75,
            anchor="center-center",
        )

    if not all_bounds:
        raise FileNotFoundError(f"No part OBJ files found under {result_dir / 'objs'}")

    bounds = np.stack(all_bounds)
    xyz_min = bounds[:, 0, :].min(axis=0)
    xyz_max = bounds[:, 1, :].max(axis=0)
    center = (xyz_min + xyz_max) * 0.5
    dims = xyz_max - xyz_min
    scene.add_box(
        "/object_bounds",
        dimensions=dims,
        color=(40, 40, 40),
        wireframe=True,
        opacity=0.25,
        position=center,
    )
    scene.add_frame("/object_frame", position=center, axes_length=0.18, axes_radius=0.006)

    joints = parse_joints(result_dir / "basic.urdf")
    moving_joints = [joint for joint in joints if joint["type"] not in {"fixed"}]
    for joint in moving_joints:
        add_joint_axis(scene, joint, axis_length=args.axis_length)

    url = f"http://localhost:{args.port}" if args.host == "0.0.0.0" else f"http://{args.host}:{args.port}"
    print(f"PhysX-Omni viser is running: {url}", flush=True)
    print(f"Result dir: {result_dir}", flush=True)
    print("Parts:", flush=True)
    for part in parts:
        print(f"  l_{part['label']}: {part.get('name', 'unnamed')}", flush=True)
    print("Moving joints:", flush=True)
    for joint in moving_joints:
        axis = joint["axis"]
        print(
            f"  {joint['name']} type={joint['type']} axis={axis.tolist()} "
            f"origin={joint['origin'].tolist()}",
            flush=True,
        )

    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
