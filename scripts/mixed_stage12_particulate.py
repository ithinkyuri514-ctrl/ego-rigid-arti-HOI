#!/usr/bin/env python3
"""Run Particulate on a mixed-run SAM3D canonical mesh and export parts in C0."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


DEFAULT_WORKSPACE = Path("/code/vlm_sam2_recon/run_mixed_20260728_203734")
DEFAULT_PARTICULATE_ROOT = Path("/code/particulate")
DEFAULT_PARTICULATE_PYTHON = Path("/opt/conda/envs/particulate/bin/python")
SOURCE_FRAME = "sam3d_canonical_z_up"
LEGACY_DESTINATION_FRAME = "frame0_right_camera_opencv_rdf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--object-id", default="microwave")
    parser.add_argument("--source-mesh", type=Path, default=None)
    parser.add_argument("--alignment-report", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--particulate-root", type=Path, default=DEFAULT_PARTICULATE_ROOT)
    parser.add_argument("--python-bin", type=Path, default=DEFAULT_PARTICULATE_PYTHON)
    parser.add_argument("--target-faces", type=int, default=100_000)
    parser.add_argument("--num-points", type=int, default=204_800)
    parser.add_argument("--min-part-confidence", type=float, default=0.0)
    parser.add_argument("--animation-frames", type=int, default=50)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--export-mjcf", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite-input", action="store_true")
    parser.add_argument("--rerun-inference", action="store_true")
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = [
            geometry
            for geometry in loaded.geometry.values()
            if isinstance(geometry, trimesh.Trimesh)
        ]
        if not geometries:
            raise ValueError(f"No triangle mesh in scene: {path}")
        mesh = geometries[0] if len(geometries) == 1 else trimesh.util.concatenate(geometries)
    else:
        mesh = loaded
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.vertices) or not len(mesh.faces):
        raise ValueError(f"No triangle mesh in {path}")
    return mesh


def mesh_stats(mesh: trimesh.Trimesh) -> dict[str, Any]:
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "bounds": np.asarray(mesh.bounds, dtype=np.float64).tolist(),
        "extents": np.asarray(mesh.extents, dtype=np.float64).tolist(),
    }


def decimate_mesh(source_path: Path, output_path: Path, target_faces: int, overwrite: bool) -> dict[str, Any]:
    if output_path.is_file() and not overwrite:
        existing = load_mesh(output_path)
        return {**mesh_stats(existing), "path": str(output_path), "skipped_existing": True}

    import open3d as o3d

    source = load_mesh(source_path)
    o3_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(np.asarray(source.vertices, dtype=np.float64)),
        triangles=o3d.utility.Vector3iVector(np.asarray(source.faces, dtype=np.int32)),
    )
    o3_mesh.remove_duplicated_vertices()
    o3_mesh.remove_duplicated_triangles()
    o3_mesh.remove_degenerate_triangles()
    o3_mesh.remove_unreferenced_vertices()
    if len(o3_mesh.triangles) > target_faces:
        o3_mesh = o3_mesh.simplify_quadric_decimation(target_number_of_triangles=target_faces)
        o3_mesh.remove_duplicated_vertices()
        o3_mesh.remove_duplicated_triangles()
        o3_mesh.remove_degenerate_triangles()
        o3_mesh.remove_unreferenced_vertices()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = trimesh.Trimesh(
        vertices=np.asarray(o3_mesh.vertices),
        faces=np.asarray(o3_mesh.triangles),
        process=False,
    )
    result.export(output_path)
    return {
        **mesh_stats(result),
        "path": str(output_path),
        "source_path": str(source_path),
        "source_faces": int(len(source.faces)),
        "target_faces": int(target_faces),
        "skipped_existing": False,
    }


def particulate_normalization(mesh: trimesh.Trimesh) -> dict[str, Any]:
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    center = bounds.mean(axis=0)
    scale = float(np.max(bounds[1] - bounds[0]))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid canonical mesh scale: {scale}")
    return {
        "source_frame": SOURCE_FRAME,
        "destination_frame": "particulate_normalized_z_up",
        "up_dir": "Z",
        "rotation_source_to_z_up": np.eye(3).tolist(),
        "center_canonical": center.tolist(),
        "scale_canonical": scale,
        "forward_formula": "p_P = (p_K - center_K) / scale_K",
        "inverse_formula": "p_K = p_P * scale_K + center_K",
    }


def latest_matching(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.glob(pattern), key=lambda path: path.stat().st_mtime)
    return matches[-1] if matches else None


def build_particulate_command(
    args: argparse.Namespace,
    input_mesh: Path,
    raw_output_dir: Path,
) -> list[str]:
    command = [
        str(args.python_bin.resolve()),
        "infer.py",
        "--input_mesh",
        str(input_mesh),
        "--output_dir",
        str(raw_output_dir),
        "--up_dir",
        "Z",
        "--num_points",
        str(args.num_points),
        "--min_part_confidence",
        str(args.min_part_confidence),
        "--animation_frames",
        str(args.animation_frames),
        "--export_urdf",
        "--eval",
        "--debug_traceback",
    ]
    if args.amp:
        command.append("--amp")
    if not args.strict:
        command.append("--no_strict")
    if args.export_mjcf:
        command.append("--export_mjcf")

    return command


def run_particulate(args: argparse.Namespace, command: list[str]) -> None:

    env = os.environ.copy()
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    subprocess.run(command, cwd=args.particulate_root.resolve(), env=env, check=True)


def parse_vec(element: ET.Element | None, attribute: str, default: tuple[float, float, float]) -> np.ndarray:
    if element is None or not element.attrib.get(attribute):
        return np.asarray(default, dtype=np.float64)
    values = np.fromstring(element.attrib[attribute], sep=" ", dtype=np.float64)
    if values.shape != (3,):
        raise ValueError(f"Expected three values in {attribute}: {element.attrib[attribute]!r}")
    return values


def parse_part_label(link_name: str) -> int:
    prefix = "link_"
    if not link_name.startswith(prefix):
        raise ValueError(f"Unexpected Particulate link name: {link_name}")
    return int(link_name.removeprefix(prefix))


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float64) @ transform[:3, :3].T + transform[:3, 3]


def transform_direction(direction: np.ndarray, transform: np.ndarray) -> np.ndarray:
    result = transform[:3, :3] @ np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(result))
    if norm <= 1e-10:
        raise ValueError("Joint axis collapsed under canonical-to-C0 transform")
    return result / norm


def parse_urdf_kinematics(urdf_path: Path) -> tuple[list[str], list[dict[str, Any]], dict[str, np.ndarray]]:
    root = ET.parse(urdf_path).getroot()
    links = [link.attrib["name"] for link in root.findall("link")]
    joints: list[dict[str, Any]] = []
    children = set()
    adjacency: dict[str, list[dict[str, Any]]] = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent").attrib["link"]
        child = joint.find("child").attrib["link"]
        record = {
            "name": joint.attrib.get("name", f"joint_{parent}_{child}"),
            "type": joint.attrib.get("type", "fixed"),
            "parent": parent,
            "child": child,
            "origin_parent_P": parse_vec(joint.find("origin"), "xyz", (0.0, 0.0, 0.0)),
            "rpy": parse_vec(joint.find("origin"), "rpy", (0.0, 0.0, 0.0)),
            "axis_P": parse_vec(joint.find("axis"), "xyz", (0.0, 0.0, 1.0)),
            "limit": dict(joint.find("limit").attrib) if joint.find("limit") is not None else {},
        }
        if not np.allclose(record["rpy"], 0.0, atol=1e-8):
            raise ValueError(f"Nonzero Particulate URDF joint rpy is not supported: {record['name']}")
        children.add(child)
        adjacency.setdefault(parent, []).append(record)
        joints.append(record)

    root_links = [link for link in links if link not in children]
    origins_P = {link: np.zeros(3, dtype=np.float64) for link in root_links}
    queue = list(root_links)
    while queue:
        parent = queue.pop(0)
        for joint in adjacency.get(parent, []):
            child = joint["child"]
            origins_P[child] = origins_P[parent] + joint["origin_parent_P"]
            queue.append(child)
    missing = set(links) - set(origins_P)
    if missing:
        raise ValueError(f"URDF contains links not reachable from a root: {sorted(missing)}")
    return root_links, joints, origins_P


def write_c0_urdf(
    output_path: Path,
    root_links: list[str],
    parts_c0: dict[str, trimesh.Trimesh],
    joints: list[dict[str, Any]],
    link_origins_c0: dict[str, np.ndarray],
) -> dict[str, str]:
    mesh_dir = output_path.parent / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    root_set = set(root_links)
    mesh_paths: dict[str, str] = {}
    for link_name, part in parts_c0.items():
        frame_origin = np.zeros(3, dtype=np.float64) if link_name in root_set else link_origins_c0[link_name]
        local = part.copy()
        local.vertices = np.asarray(local.vertices, dtype=np.float64) - frame_origin
        mesh_path = mesh_dir / f"part_{parse_part_label(link_name)}.obj"
        local.export(mesh_path)
        mesh_paths[link_name] = str(mesh_path)

    robot = ET.Element("robot", {"name": "microwave_C0"})
    for link_name in sorted(parts_c0, key=parse_part_label):
        link = ET.SubElement(robot, "link", {"name": link_name})
        for role in ("visual", "collision"):
            role_node = ET.SubElement(link, role)
            ET.SubElement(role_node, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
            geometry = ET.SubElement(role_node, "geometry")
            ET.SubElement(geometry, "mesh", {"filename": f"./meshes/part_{parse_part_label(link_name)}.obj"})

    for record in joints:
        parent, child = record["parent"], record["child"]
        parent_origin = np.zeros(3, dtype=np.float64) if parent in root_set else link_origins_c0[parent]
        child_origin = link_origins_c0[child]
        offset = child_origin - parent_origin
        joint = ET.SubElement(robot, "joint", {"name": record["name"], "type": record["type"]})
        ET.SubElement(joint, "parent", {"link": parent})
        ET.SubElement(joint, "child", {"link": child})
        ET.SubElement(joint, "origin", {"xyz": " ".join(f"{value:.9g}" for value in offset), "rpy": "0 0 0"})
        if record["type"] != "fixed":
            ET.SubElement(
                joint,
                "axis",
                {"xyz": " ".join(f"{value:.9g}" for value in record["axis_C0"])},
            )
            if record["limit"]:
                ET.SubElement(joint, "limit", {key: str(value) for key, value in record["limit"].items()})

    ET.indent(robot, space="  ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(robot).write(output_path, encoding="utf-8", xml_declaration=True)
    return mesh_paths


def export_c0_scene(parts: dict[str, trimesh.Trimesh], joints: list[dict[str, Any]], output_path: Path) -> None:
    colors = np.asarray(
        [[65, 105, 225, 255], [220, 70, 60, 255], [50, 170, 100, 255], [235, 175, 45, 255]],
        dtype=np.uint8,
    )
    scene = trimesh.Scene()
    for index, (link_name, part) in enumerate(sorted(parts.items(), key=lambda item: parse_part_label(item[0]))):
        colored = part.copy()
        colored.visual.face_colors = np.tile(colors[index % len(colors)], (len(colored.faces), 1))
        scene.add_geometry(colored, node_name=link_name, geom_name=link_name)
    for record in joints:
        if record["type"] == "fixed":
            continue
        origin = np.asarray(record["origin_C0"], dtype=np.float64)
        axis = np.asarray(record["axis_C0"], dtype=np.float64)
        length = 0.35
        cylinder = trimesh.creation.cylinder(radius=0.004, height=length, sections=20)
        align = trimesh.geometry.align_vectors([0.0, 0.0, 1.0], axis)
        cylinder.apply_transform(align)
        cylinder.apply_translation(origin)
        cylinder.visual.face_colors = np.tile(np.asarray([250, 210, 30, 255], dtype=np.uint8), (len(cylinder.faces), 1))
        scene.add_geometry(cylinder, node_name=f"{record['name']}_axis", geom_name=f"{record['name']}_axis")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(output_path)


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    workspace = args.workspace.resolve()
    source_mesh = (
        args.source_mesh.resolve()
        if args.source_mesh
        else workspace / f"outputs/03_sam3d_frame0/{args.object_id}/mesh_canonical.glb"
    )
    alignment_report = (
        args.alignment_report.resolve()
        if args.alignment_report
        else workspace / f"outputs/07_alignment/{args.object_id}/frame_000000/alignment_report.json"
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else workspace / f"outputs/12_particulate/{args.object_id}"
    )
    return source_mesh, alignment_report, output_dir


def ensure_worker_environment(args: argparse.Namespace) -> None:
    try:
        import open3d  # noqa: F401
    except ModuleNotFoundError:
        python_bin = args.python_bin.resolve()
        if Path(sys.executable).resolve() == python_bin:
            raise
        os.execv(str(python_bin), [str(python_bin), str(Path(__file__).resolve()), *sys.argv[1:]])


def main() -> int:
    args = parse_args()
    ensure_worker_environment(args)
    source_mesh_path, alignment_report_path, output_dir = resolve_paths(args)
    for path in (source_mesh_path, alignment_report_path, args.particulate_root / "infer.py", args.python_bin):
        if not path.exists():
            raise FileNotFoundError(path)

    stage00_path = args.workspace.resolve() / "outputs/00_rgb_frames/stage00_manifest.json"
    stage00 = json.loads(stage00_path.read_text(encoding="utf-8"))
    selected_eye = str(stage00.get("selected_eye", "right")).lower()
    if selected_eye not in {"left", "right"}:
        raise ValueError(f"Unsupported selected eye: {selected_eye!r}")
    destination_frame = f"frame0_{selected_eye}_camera_opencv_rdf"

    alignment = json.loads(alignment_report_path.read_text(encoding="utf-8"))
    transform_record = alignment.get("transform", {})
    if transform_record.get("source_frame") != SOURCE_FRAME:
        raise ValueError(f"Unexpected alignment source frame: {transform_record.get('source_frame')!r}")
    if transform_record.get("destination_frame") not in {
        destination_frame,
        LEGACY_DESTINATION_FRAME,
    }:
        raise ValueError(f"Unexpected alignment destination frame: {transform_record.get('destination_frame')!r}")
    transform_c0_from_canonical = np.asarray(
        transform_record["T_C0_from_sam3d_canonical"], dtype=np.float64
    )
    if transform_c0_from_canonical.shape != (4, 4):
        raise ValueError("T_C0_from_sam3d_canonical must be 4x4")

    source_mesh = load_mesh(source_mesh_path)
    input_mesh_path = output_dir / "inputs" / f"{args.object_id}_canonical_decimated_{args.target_faces}.glb"
    decimation = decimate_mesh(source_mesh_path, input_mesh_path, args.target_faces, args.overwrite_input)
    input_mesh = load_mesh(input_mesh_path)
    normalization = particulate_normalization(input_mesh)
    raw_output_dir = output_dir / "raw"
    pred_npz = raw_output_dir / "eval/pred.npz"
    command = build_particulate_command(args, input_mesh_path, raw_output_dir)
    inference_executed = args.rerun_inference or not pred_npz.is_file()
    if inference_executed:
        run_particulate(args, command)

    urdf_path = latest_matching(raw_output_dir, "urdf_*/model.urdf")
    pred_obj = raw_output_dir / "eval/pred.obj"
    required = [pred_npz, pred_obj, urdf_path]
    if any(path is None or not path.is_file() for path in required):
        raise RuntimeError(
            "Particulate did not produce the required eval/URDF artifacts. "
            "infer.py catches per-mesh exceptions, so its zero exit status alone is not sufficient."
        )
    assert urdf_path is not None

    root_links, joints, link_origins_P = parse_urdf_kinematics(urdf_path)
    moving_links = {record["child"] for record in joints if record["type"] != "fixed"}
    link_names = sorted(link_origins_P, key=parse_part_label)
    part_labels = [parse_part_label(link) for link in link_names]
    prediction = np.load(pred_npz, allow_pickle=False)
    face_part_ids = np.asarray(prediction["face_part_ids"], dtype=np.int64)
    prediction_mesh = load_mesh(pred_obj)
    if len(face_part_ids) != len(prediction_mesh.faces):
        raise ValueError(
            f"face_part_ids has {len(face_part_ids)} entries for {len(prediction_mesh.faces)} faces"
        )
    eval_part_ids = sorted(int(value) for value in np.unique(face_part_ids))
    if len(eval_part_ids) != len(part_labels):
        raise ValueError(
            f"Prediction has {len(eval_part_ids)} part ids but URDF has {len(part_labels)} links"
        )

    parts_canonical_dir = output_dir / "parts_canonical"
    parts_c0_dir = output_dir / "parts_C0"
    parts_canonical_dir.mkdir(parents=True, exist_ok=True)
    parts_c0_dir.mkdir(parents=True, exist_ok=True)
    parts_c0: dict[str, trimesh.Trimesh] = {}
    part_records = []
    for eval_id, link_name, part_label in zip(eval_part_ids, link_names, part_labels):
        part_canonical = prediction_mesh.submesh([face_part_ids == eval_id], append=True, repair=False)
        canonical_path = parts_canonical_dir / f"part_{part_label}.obj"
        part_canonical.export(canonical_path)
        part_c0 = part_canonical.copy()
        part_c0.vertices = transform_points(part_c0.vertices, transform_c0_from_canonical)
        c0_path = parts_c0_dir / f"part_{part_label}.obj"
        part_c0.export(c0_path)
        parts_c0[link_name] = part_c0
        part_records.append(
            {
                "part_label": part_label,
                "link_name": link_name,
                "eval_part_id": eval_id,
                "semantic_role": "moving_part_candidate" if link_name in moving_links else "base_candidate",
                "canonical_mesh": str(canonical_path),
                "C0_mesh": str(c0_path),
                "canonical_stats": mesh_stats(part_canonical),
                "C0_stats": mesh_stats(part_c0),
            }
        )

    center_K = np.asarray(normalization["center_canonical"], dtype=np.float64)
    scale_K = float(normalization["scale_canonical"])
    link_origins_c0: dict[str, np.ndarray] = {}
    for link_name, origin_P in link_origins_P.items():
        origin_K = origin_P * scale_K + center_K
        link_origins_c0[link_name] = transform_points(origin_K[None], transform_c0_from_canonical)[0]

    joint_records = []
    for record in joints:
        origin_P = link_origins_P[record["child"]]
        origin_K = origin_P * scale_K + center_K
        origin_C0 = transform_points(origin_K[None], transform_c0_from_canonical)[0]
        axis_K = np.asarray(record["axis_P"], dtype=np.float64)
        axis_K /= np.linalg.norm(axis_K) + 1e-12
        axis_C0 = transform_direction(axis_K, transform_c0_from_canonical)
        record["origin_P"] = origin_P
        record["origin_K"] = origin_K
        record["origin_C0"] = origin_C0
        record["axis_K"] = axis_K
        record["axis_C0"] = axis_C0
        joint_records.append(
            {
                "name": record["name"],
                "type": record["type"],
                "parent": record["parent"],
                "child": record["child"],
                "origin_particulate": origin_P.tolist(),
                "origin_sam3d_canonical": origin_K.tolist(),
                "origin_C0": origin_C0.tolist(),
                "axis_particulate": np.asarray(record["axis_P"]).tolist(),
                "axis_sam3d_canonical": axis_K.tolist(),
                "axis_C0": axis_C0.tolist(),
                "limit": record["limit"],
            }
        )

    c0_urdf_path = output_dir / "urdf_C0/model.urdf"
    c0_urdf_meshes = write_c0_urdf(c0_urdf_path, root_links, parts_c0, joints, link_origins_c0)
    c0_scene_path = output_dir / "parts_with_axes_C0.glb"
    export_c0_scene(parts_c0, joints, c0_scene_path)

    mjcf_path = latest_matching(raw_output_dir, "mjcf_*/model.xml")
    axes_glb_path = latest_matching(raw_output_dir, "mesh_parts_with_axes_*.glb")
    animated_glb_path = latest_matching(raw_output_dir, "animated_textured_*.glb")
    manifest = {
        "schema_version": 1,
        "stage": "12_particulate_articulation",
        "status": "completed",
        "object_id": args.object_id,
        "selected_eye": selected_eye,
        "source_mesh": str(source_mesh_path),
        "source_mesh_stats": mesh_stats(source_mesh),
        "source_coordinate_frame": SOURCE_FRAME,
        "alignment_report": str(alignment_report_path),
        "destination_coordinate_frame": destination_frame,
        "T_C0_from_sam3d_canonical": transform_c0_from_canonical.tolist(),
        "particulate_normalization": normalization,
        "decimation": decimation,
        "inference": {
            "command": command,
            "executed_this_invocation": inference_executed,
            "particulate_root": str(args.particulate_root.resolve()),
            "python_bin": str(args.python_bin.resolve()),
            "up_dir": "Z",
            "num_points": args.num_points,
            "target_faces": args.target_faces,
            "min_part_confidence": args.min_part_confidence,
            "strict": args.strict,
            "amp": args.amp,
        },
        "raw_outputs": {
            "result_dir": str(raw_output_dir),
            "pred_npz": str(pred_npz),
            "pred_obj": str(pred_obj),
            "urdf": str(urdf_path),
            "mjcf": str(mjcf_path) if mjcf_path else None,
            "parts_with_axes_glb": str(axes_glb_path) if axes_glb_path else None,
            "animated_glb": str(animated_glb_path) if animated_glb_path else None,
        },
        "C0_outputs": {
            "urdf": str(c0_urdf_path),
            "urdf_meshes": c0_urdf_meshes,
            "parts_with_axes_glb": str(c0_scene_path),
        },
        "root_links": root_links,
        "moving_part_labels": [parse_part_label(link) for link in sorted(moving_links, key=parse_part_label)],
        "base_part_labels": [
            parse_part_label(link) for link in link_names if link not in moving_links
        ],
        "parts": part_records,
        "joints": joint_records,
        "quality_control": {
            "part_face_count": int(sum(record["C0_stats"]["faces"] for record in part_records)),
            "input_face_count": int(len(prediction_mesh.faces)),
            "all_faces_assigned_once": int(sum(record["C0_stats"]["faces"] for record in part_records))
            == int(len(prediction_mesh.faces)),
        },
        "updated_at": utc_now(),
    }
    manifest_path = output_dir / "particulate_manifest.json"
    write_json(manifest_path, manifest)
    write_json(output_dir / "joint_axes_C0.json", {"coordinate_frame": destination_frame, "joints": joint_records})
    print(json.dumps({
        "manifest": str(manifest_path),
        "part_count": len(part_records),
        "joint_count": len(joint_records),
        "parts_with_axes_C0": str(c0_scene_path),
        "urdf_C0": str(c0_urdf_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
