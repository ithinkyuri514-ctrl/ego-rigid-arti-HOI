#!/usr/bin/env python3
"""Visualize Particulate part meshes and URDF joints with viser."""

from __future__ import annotations

import argparse
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
import viser


DEFAULT_RESULT_DIR = Path("/code/vlm_sam2_recon/outputs/particulate/target_laptop_decimated_50000")

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
PRISMATIC_COLOR = (33, 158, 188)
FIXED_JOINT_COLOR = (150, 150, 150)


def latest_matching(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.glob(pattern), key=lambda path: path.stat().st_mtime)
    return matches[-1] if matches else None


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
    scene.add_mesh_simple(
        name,
        vertices=np.asarray(mesh.vertices, dtype=np.float32),
        faces=np.asarray(mesh.faces, dtype=np.uint32),
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


def link_names(urdf_path: Path) -> list[str]:
    root = ET.parse(urdf_path).getroot()
    return [link.attrib["name"] for link in root.findall("link") if "name" in link.attrib]


def compute_link_world_positions(joints: list[dict], links: list[str]) -> dict[str, np.ndarray]:
    child_links = {joint["child"] for joint in joints if joint.get("child")}
    roots = [link for link in links if link not in child_links]
    positions = {link: np.zeros(3, dtype=np.float32) for link in roots}
    unresolved = list(joints)

    while unresolved:
        next_unresolved = []
        progressed = False
        for joint in unresolved:
            parent = joint.get("parent")
            child = joint.get("child")
            if parent in positions and child:
                positions[child] = positions[parent] + joint["origin"]
                joint["world_origin"] = positions[child].copy()
                progressed = True
            else:
                next_unresolved.append(joint)
        if not progressed:
            break
        unresolved = next_unresolved

    for link in links:
        positions.setdefault(link, np.zeros(3, dtype=np.float32))
    for joint in joints:
        joint.setdefault("world_origin", positions.get(joint.get("child"), joint["origin"]))
    return positions


def add_joint_axis(scene: viser.SceneApi, joint: dict, axis_length: float) -> None:
    origin = np.asarray(joint.get("world_origin", joint["origin"]), dtype=np.float32)
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
    color = PRISMATIC_COLOR if joint["type"] == "prismatic" else JOINT_COLOR
    scene.add_line_segments(
        f"/joints/{joint['name']}/axis",
        points=np.asarray([[p0, p1]], dtype=np.float32),
        colors=color,
        line_width=6,
    )
    scene.add_icosphere(
        f"/joints/{joint['name']}/origin",
        radius=0.022,
        color=color,
        position=origin,
        subdivisions=2,
    )
    label = f"{joint['name']} {joint['type']} axis=[{axis[0]:.2f}, {axis[1]:.2f}, {axis[2]:.2f}]"
    if joint.get("limit"):
        lower = joint["limit"].get("lower")
        upper = joint["limit"].get("upper")
        if lower is not None and upper is not None:
            label += f" limit=({float(lower):.2f}, {float(upper):.2f})"
    scene.add_label(
        f"/joints/{joint['name']}/label",
        text=label,
        position=origin + np.asarray([0.0, 0.0, 0.08], dtype=np.float32),
    )


def part_label_from_path(path: Path) -> str:
    return path.stem.removeprefix("part_")


def link_name_from_part_mesh(path: Path) -> str:
    return f"link_{part_label_from_path(path)}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve Particulate part meshes and joints in viser.")
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--urdf", type=Path, default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8095)
    parser.add_argument("--opacity", type=float, default=0.9)
    parser.add_argument("--axis-length", type=float, default=1.05)
    parser.add_argument("--show-source-glb", action="store_true")
    parser.add_argument("--source-mesh", type=Path, default=None)
    parser.add_argument(
        "--up-direction",
        default="+z",
        choices=["+x", "-x", "+y", "-y", "+z", "-z"],
    )
    args = parser.parse_args()

    result_dir = args.result_dir.resolve()
    urdf_path = args.urdf.resolve() if args.urdf else latest_matching(result_dir, "urdf_*/model.urdf")
    if urdf_path is None or not urdf_path.exists():
        raise FileNotFoundError(f"Missing Particulate URDF under: {result_dir}")

    run_json = result_dir / "particulate_run.json"
    run_record = json.loads(run_json.read_text(encoding="utf-8")) if run_json.exists() else {}
    part_meshes = sorted((urdf_path.parent / "meshes").glob("part_*.obj"))
    if not part_meshes:
        raise FileNotFoundError(f"No part OBJ files found under: {urdf_path.parent / 'meshes'}")
    joints = parse_joints(urdf_path)
    link_positions = compute_link_world_positions(joints, link_names(urdf_path))

    server = viser.ViserServer(host=args.host, port=args.port)
    scene = server.scene
    scene.set_up_direction(args.up_direction)
    grid_plane = {"x": "yz", "y": "xz", "z": "xy"}[args.up_direction[-1]]
    scene.add_grid(
        "/grid",
        width=1.4,
        height=1.4,
        plane=grid_plane,
        cell_size=0.05,
        cell_thickness=0.001,
        section_size=0.25,
        section_thickness=0.002,
    )

    all_bounds = []
    source_mesh_path = args.source_mesh.resolve() if args.source_mesh else None
    if source_mesh_path is None and args.show_source_glb and run_record.get("decimated_mesh"):
        source_mesh_path = Path(run_record["decimated_mesh"])
    if source_mesh_path is not None:
        source_mesh = load_mesh(source_mesh_path)
        add_part_mesh(scene, "/source/decimated_mesh", source_mesh, color=(180, 180, 180), opacity=0.18)

    for idx, mesh_path in enumerate(part_meshes):
        mesh = load_mesh(mesh_path)
        label = part_label_from_path(mesh_path)
        link_position = link_positions.get(link_name_from_part_mesh(mesh_path), np.zeros(3, dtype=np.float32))
        mesh.vertices = np.asarray(mesh.vertices) + link_position
        color = PART_COLORS[idx % len(PART_COLORS)]
        add_part_mesh(scene, f"/parts/part_{label}", mesh, color=color, opacity=args.opacity)
        bounds = np.asarray(mesh.bounds, dtype=np.float32)
        all_bounds.append(bounds)
        center = bounds.mean(axis=0)
        scene.add_label(
            f"/labels/part_{label}",
            text=f"part_{label}",
            position=center,
        )

    bounds = np.stack(all_bounds)
    xyz_min = bounds[:, 0, :].min(axis=0)
    xyz_max = bounds[:, 1, :].max(axis=0)
    center = (xyz_min + xyz_max) * 0.5
    scene.add_frame("/object_frame", position=center, axes_length=0.18, axes_radius=0.006)

    moving_joints = [joint for joint in joints if joint["type"] not in {"fixed"}]
    for joint in moving_joints:
        add_joint_axis(scene, joint, axis_length=args.axis_length)

    url = f"http://localhost:{args.port}" if args.host == "0.0.0.0" else f"http://{args.host}:{args.port}"
    print(f"Particulate viser is running: {url}", flush=True)
    print(f"Result dir: {result_dir}", flush=True)
    print(f"URDF: {urdf_path}", flush=True)
    print(f"Up direction: {args.up_direction}", flush=True)
    print("Parts:", flush=True)
    for mesh_path in part_meshes:
        print(f"  part_{part_label_from_path(mesh_path)}: {mesh_path}", flush=True)
    print("Moving joints:", flush=True)
    for joint in moving_joints:
        print(
            f"  {joint['name']} type={joint['type']} axis={joint['axis'].tolist()} "
            f"local_origin={joint['origin'].tolist()} world_origin={joint['world_origin'].tolist()} "
            f"limit={joint.get('limit', {})}",
            flush=True,
        )

    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
