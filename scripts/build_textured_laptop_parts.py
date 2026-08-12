#!/usr/bin/env python3
"""Cut Hunyuan textured laptop parts and align them into the camera frame.

Particulate exports part OBJ files without the original Hunyuan UV/material
data. This script treats Particulate as the segmentation source only: it
transfers the decimated face labels back to the original textured GLB, cuts
textured base/screen submeshes, and applies the already validated alignment
transform into the frame-0 right-camera coordinate system.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from trimesh.visual.color import ColorVisuals
from trimesh.visual.texture import TextureVisuals


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vlm_sam2_recon.stages.camera_alignment import rotate_mesh_about_axis  # noqa: E402


DEFAULT_ALIGNMENT_DIR = (
    PROJECT_ROOT
    / "outputs/object_alignment_hunyuan_base_first_nohinge/target_laptop/frame_000000"
)
DEFAULT_DYNAMIC_DIR = (
    PROJECT_ROOT
    / "outputs/contact_driven_laptop/hunyuan_base_first_fixed_frame0_tight_contact_000000_000057"
)
DEFAULT_SOURCE_MESH = PROJECT_ROOT / "inputs/hunyuan3d_meshes/target_laptop/whole/target_laptop.glb"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build textured base/screen laptop parts from the original Hunyuan GLB."
    )
    parser.add_argument("--alignment-dir", type=Path, default=DEFAULT_ALIGNMENT_DIR)
    parser.add_argument("--dynamic-dir", type=Path, default=DEFAULT_DYNAMIC_DIR)
    parser.add_argument("--source-mesh", type=Path, default=DEFAULT_SOURCE_MESH)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--base-label", default="14")
    parser.add_argument("--screen-label", default="15")
    parser.add_argument("--nearest-k", type=int, default=5)
    parser.add_argument("--normal-weight", type=float, default=0.02)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(data), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def load_single_mesh(path: Path, *, preserve_texture: bool = True) -> trimesh.Trimesh:
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        geoms = [geom.copy() for geom in mesh.geometry.values() if isinstance(geom, trimesh.Trimesh)]
        if not geoms:
            raise ValueError(f"No Trimesh geometry found in {path}")
        if preserve_texture and len(geoms) == 1:
            mesh = geoms[0]
        else:
            mesh = trimesh.util.concatenate(geoms)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type from {path}: {type(mesh)!r}")
    return mesh


def texture_visual_for_vertices(source: trimesh.Trimesh, used_vertices: np.ndarray) -> TextureVisuals | ColorVisuals:
    visual = source.visual
    uv = getattr(visual, "uv", None)
    material = getattr(visual, "material", None)
    if uv is not None and material is not None:
        return TextureVisuals(uv=np.asarray(uv)[used_vertices].copy(), material=material)

    colors = getattr(visual, "vertex_colors", None)
    if colors is not None and len(colors) == len(source.vertices):
        return ColorVisuals(vertex_colors=np.asarray(colors)[used_vertices].copy())
    return ColorVisuals(vertex_colors=np.tile(np.array([[180, 180, 180, 255]], dtype=np.uint8), (len(used_vertices), 1)))


def submesh_with_visual(source: trimesh.Trimesh, face_indices: np.ndarray) -> trimesh.Trimesh:
    face_indices = np.asarray(face_indices, dtype=np.int64)
    if face_indices.size == 0:
        raise ValueError("Cannot build a textured part with zero faces")

    source_faces = np.asarray(source.faces, dtype=np.int64)[face_indices]
    used_vertices, inverse = np.unique(source_faces.reshape(-1), return_inverse=True)
    faces = inverse.reshape((-1, 3))
    vertices = np.asarray(source.vertices, dtype=np.float64)[used_vertices].copy()
    visual = texture_visual_for_vertices(source, used_vertices)
    return trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual, process=False)


def input_to_particulate_matrix(transform: dict[str, Any]) -> np.ndarray:
    rot = np.asarray(transform["rotation_input_to_particulate_z_up"], dtype=np.float64)
    center = np.asarray(transform["center_after_up_rotation"], dtype=np.float64)
    scale = float(transform["scale_after_up_rotation"])
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rot / scale
    out[:3, 3] = -center / scale
    return out


def apply_transform(mesh: trimesh.Trimesh, transform: np.ndarray) -> trimesh.Trimesh:
    out = mesh.copy()
    out.apply_transform(np.asarray(transform, dtype=np.float64))
    return out


def part_label_from_path(path: Path) -> str:
    stem = path.stem
    return stem.split("_", 1)[1] if stem.startswith("part_") else stem


def infer_part_id_to_label(face_part_ids: np.ndarray, urdf_path: Path) -> dict[int, str]:
    id_counts = Counter(int(v) for v in face_part_ids.tolist())
    mesh_dir = urdf_path.parent / "meshes"
    label_counts: dict[str, int] = {}
    for part_path in sorted(mesh_dir.glob("part_*.obj")):
        part_mesh = load_single_mesh(part_path, preserve_texture=False)
        label_counts[part_label_from_path(part_path)] = int(len(part_mesh.faces))

    mapping: dict[int, str] = {}
    used_labels: set[str] = set()
    for part_id, count in sorted(id_counts.items()):
        candidates = [label for label, label_count in label_counts.items() if label_count == count and label not in used_labels]
        if candidates:
            mapping[part_id] = candidates[0]
            used_labels.add(candidates[0])

    if len(mapping) != len(id_counts):
        labels = sorted(label_counts)
        ids = sorted(id_counts)
        if len(labels) != len(ids):
            raise ValueError(
                f"Could not infer part id mapping: face ids={dict(id_counts)}, part meshes={label_counts}"
            )
        mapping = {part_id: labels[i] for i, part_id in enumerate(ids)}
    return mapping


def transfer_labels_to_source_faces(
    source_mesh: trimesh.Trimesh,
    decimated_mesh: trimesh.Trimesh,
    decimated_face_part_ids: np.ndarray,
    *,
    nearest_k: int,
    normal_weight: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if len(decimated_face_part_ids) != len(decimated_mesh.faces):
        raise ValueError(
            "pred.npz face_part_ids length does not match decimated mesh faces: "
            f"{len(decimated_face_part_ids)} vs {len(decimated_mesh.faces)}"
        )

    source_centers = np.asarray(source_mesh.triangles_center, dtype=np.float64)
    decimated_centers = np.asarray(decimated_mesh.triangles_center, dtype=np.float64)
    tree = cKDTree(decimated_centers)
    k = max(1, min(int(nearest_k), len(decimated_centers)))
    distances, nearest = tree.query(source_centers, k=k, workers=-1)
    if k == 1:
        labels = decimated_face_part_ids[np.asarray(nearest, dtype=np.int64)]
        chosen_distances = np.asarray(distances, dtype=np.float64)
    else:
        distances = np.asarray(distances, dtype=np.float64)
        nearest = np.asarray(nearest, dtype=np.int64)
        source_normals = np.asarray(source_mesh.face_normals, dtype=np.float64)
        decimated_normals = np.asarray(decimated_mesh.face_normals, dtype=np.float64)
        normal_dot = np.abs(np.sum(source_normals[:, None, :] * decimated_normals[nearest], axis=2))
        diag = float(np.linalg.norm(source_mesh.extents))
        scores = distances + float(normal_weight) * diag * (1.0 - np.clip(normal_dot, 0.0, 1.0))
        best_local = np.argmin(scores, axis=1)
        rows = np.arange(len(source_centers))
        best = nearest[rows, best_local]
        labels = decimated_face_part_ids[best]
        chosen_distances = distances[rows, best_local]

    metrics = {
        "source_face_count": int(len(source_mesh.faces)),
        "decimated_face_count": int(len(decimated_mesh.faces)),
        "nearest_k": int(k),
        "normal_weight": float(normal_weight),
        "nearest_distance_mean": float(np.mean(chosen_distances)),
        "nearest_distance_median": float(np.median(chosen_distances)),
        "nearest_distance_p95": float(np.quantile(chosen_distances, 0.95)),
        "nearest_distance_max": float(np.max(chosen_distances)),
        "transferred_face_counts": {str(k): int(v) for k, v in Counter(int(x) for x in labels.tolist()).items()},
    }
    return labels.astype(np.int32), metrics


def mesh_bounds(mesh: trimesh.Trimesh) -> dict[str, Any]:
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "bounds": bounds,
        "extents": bounds[1] - bounds[0],
    }


def initial_screen_hinge_refinement(
    alignment_result: dict[str, Any],
    screen_label: str,
) -> dict[str, Any] | None:
    """Return the initial screen-only hinge refinement stored by alignment."""
    alignment = alignment_result.get("alignment", {})
    candidates = (
        alignment.get("hinge_refine"),
        alignment.get("screen_hinge_refine"),
    )
    for item in candidates:
        if not isinstance(item, dict) or item.get("chosen_angle_deg") is None:
            continue
        moving_label = str(
            item.get("screen_part_label")
            or item.get("moving_part_label")
            or screen_label
        )
        if moving_label != str(screen_label):
            continue
        return {
            "angle_deg": float(item["chosen_angle_deg"]),
            "joint_name": item.get("joint_name"),
            "source": "alignment.hinge_refine",
        }
    return None


def main() -> int:
    args = parse_args()
    alignment_dir = args.alignment_dir.resolve()
    dynamic_dir = args.dynamic_dir.resolve()
    source_mesh_path = args.source_mesh.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else dynamic_dir / "textured_laptop_parts"
    )
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output dir already exists. Use --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    alignment = read_json(alignment_dir / "alignment_result.json")
    part_run_path = Path(alignment["canonical_urdf"]).resolve().parents[1] / "particulate_run.json"
    part_run = read_json(part_run_path)
    pred_path = Path(part_run["outputs"]["pred.npz"]).resolve()
    decimated_mesh_path = Path(part_run["decimated_mesh"]).resolve()
    urdf_path = Path(alignment["canonical_urdf"]).resolve()

    print(f"[textured-parts] source textured mesh: {source_mesh_path}", flush=True)
    print(f"[textured-parts] decimated mesh: {decimated_mesh_path}", flush=True)
    print(f"[textured-parts] particulate pred: {pred_path}", flush=True)

    source_mesh = load_single_mesh(source_mesh_path, preserve_texture=True)
    decimated_mesh = load_single_mesh(decimated_mesh_path, preserve_texture=False)
    pred = np.load(pred_path)
    face_part_ids = np.asarray(pred["face_part_ids"], dtype=np.int32)
    part_id_to_label = infer_part_id_to_label(face_part_ids, urdf_path)
    source_face_part_ids, transfer_metrics = transfer_labels_to_source_faces(
        source_mesh,
        decimated_mesh,
        face_part_ids,
        nearest_k=args.nearest_k,
        normal_weight=args.normal_weight,
    )

    input_to_part = input_to_particulate_matrix(part_run["coordinate_transform"])
    part_to_camera = np.asarray(alignment["alignment"]["matrix_canonical_to_align_camera"], dtype=np.float64)
    input_to_camera = part_to_camera @ input_to_part
    joint = read_json(alignment_dir / "joint_camera.json")["joints"][0]
    initial_screen_hinge = initial_screen_hinge_refinement(alignment, args.screen_label)
    screen_metric_refine = alignment.get("alignment", {}).get("screen_metric_refine")
    joint_origin = np.asarray(joint["origin_xyz"], dtype=np.float64)
    joint_axis = np.asarray(joint["axis_xyz"], dtype=np.float64)
    joint_axis /= np.linalg.norm(joint_axis) + 1e-12

    part_outputs: dict[str, Any] = {}
    for part_id, label in sorted(part_id_to_label.items(), key=lambda item: item[1]):
        face_indices = np.flatnonzero(source_face_part_ids == part_id)
        part_input = submesh_with_visual(source_mesh, face_indices)
        part_camera = apply_transform(part_input, input_to_camera)
        applied_initial_hinge_deg = 0.0
        if label == args.screen_label and initial_screen_hinge is not None:
            applied_initial_hinge_deg = float(initial_screen_hinge["angle_deg"])
            part_camera = rotate_mesh_about_axis(
                part_camera,
                joint_origin,
                joint_axis,
                np.deg2rad(applied_initial_hinge_deg),
            )
        if label == args.screen_label and isinstance(screen_metric_refine, dict):
            matrix = np.asarray(screen_metric_refine["screen_affine_matrix"], dtype=np.float64)
            part_camera.apply_transform(matrix)
        input_path = output_dir / f"part_{label}_hunyuan_textured.glb"
        camera_path = output_dir / f"part_{label}_camera_textured.glb"
        part_input.export(input_path)
        part_camera.export(camera_path)
        part_outputs[label] = {
            "part_id": int(part_id),
            "source_face_count": int(face_indices.size),
            "input_mesh": str(input_path),
            "camera_mesh": str(camera_path),
            "input_bounds": mesh_bounds(part_input),
            "camera_bounds": mesh_bounds(part_camera),
            "initial_hinge_angle_deg": applied_initial_hinge_deg,
        }
        print(
            f"[textured-parts] part_{label}: {len(part_camera.vertices)} verts, "
            f"{len(part_camera.faces)} faces -> {camera_path}",
            flush=True,
        )

    scene = trimesh.Scene()
    for label in (args.base_label, args.screen_label):
        if label in part_outputs:
            scene.add_geometry(load_single_mesh(Path(part_outputs[label]["camera_mesh"]), preserve_texture=True), node_name=f"part_{label}")
    laptop_path = output_dir / "laptop_camera_textured.glb"
    scene.export(laptop_path)

    manifest = {
        "type": "textured_hunyuan_part_transfer",
        "source_mesh": str(source_mesh_path),
        "decimated_mesh": str(decimated_mesh_path),
        "particulate_run": str(part_run_path),
        "pred_npz": str(pred_path),
        "alignment_result": str(alignment_dir / "alignment_result.json"),
        "dynamic_dir": str(dynamic_dir),
        "coordinate_frame": "frame0_right_camera",
        "part_id_to_label": part_id_to_label,
        "base_label": args.base_label,
        "screen_label": args.screen_label,
        "input_to_particulate_matrix": input_to_part,
        "particulate_to_camera_matrix": part_to_camera,
        "input_to_camera_matrix": input_to_camera,
        "transfer_metrics": transfer_metrics,
        "parts": part_outputs,
        "laptop_camera_textured": str(laptop_path),
        "joint_camera": joint,
        "initial_screen_hinge_refinement": initial_screen_hinge,
        "screen_metric_refinement": screen_metric_refine,
        "usage": {
            "visualizer": "scripts/serve_dynamic_laptop_hand_viser.py --use-textured-laptop --textured-laptop-dir "
            + str(output_dir),
            "note": "The camera-space textured screen includes the alignment-time hinge angle. Dynamic angles are additional rotations around the fixed joint axis.",
        },
    }
    write_json(output_dir / "textured_laptop_manifest.json", manifest)
    print(f"[textured-parts] manifest: {output_dir / 'textured_laptop_manifest.json'}", flush=True)
    print(f"[textured-parts] combined GLB: {laptop_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
