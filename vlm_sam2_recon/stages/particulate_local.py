"""Local Particulate articulation inference and manifest ingestion."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..manifest_io import read_project_manifest, write_json, write_project_manifest
from ..schemas import (
    CLASS_ARTICULATED,
    ArticulationJoint,
    MeshArtifact,
    PipelineStageRecord,
    STAGE_PARTICULATE,
    STATUS_COMPLETED,
)


DEFAULT_PARTICULATE_ROOT = Path("/code/particulate")
DEFAULT_PYTHON = Path("/opt/conda/envs/particulate/bin/python")


@dataclass
class ParticulateRunConfig:
    project_root: Path
    manifest_path: Path
    particulate_root: Path = DEFAULT_PARTICULATE_ROOT
    python_bin: Path = DEFAULT_PYTHON
    output_root: Path | None = None
    input_root: Path | None = None
    targets: list[str] | None = None
    up_dir: str = "Z"
    num_points: int = 102400
    target_faces: int = 50000
    min_part_confidence: float = 0.0
    no_strict: bool = True
    export_urdf: bool = True
    export_mjcf: bool = True
    eval: bool = True
    hf_disable_xet: bool = True
    overwrite: bool = False
    run_inference: bool = True


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _resolve_project_path(project_root: Path, path: str | Path | None) -> Path | None:
    if path is None:
        return None
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = project_root / resolved
    return resolved.resolve()


def _load_mesh(path: Path):
    import trimesh

    mesh = trimesh.load(path, force="scene", process=False)
    if isinstance(mesh, trimesh.Scene):
        geoms = [geom for geom in mesh.geometry.values() if isinstance(geom, trimesh.Trimesh)]
        if not geoms:
            raise ValueError(f"No mesh geometry found in {path}")
        mesh = trimesh.util.concatenate(geoms)
    return mesh


def mesh_stats(path: Path) -> dict[str, Any]:
    try:
        mesh = _load_mesh(path)
        components = mesh.split(only_watertight=False)
        component_faces = sorted((int(len(comp.faces)) for comp in components), reverse=True)
        return {
            "vertices": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
            "bounds": np.asarray(mesh.bounds).tolist() if mesh.bounds is not None else None,
            "extents": np.asarray(mesh.extents).tolist() if mesh.extents is not None else None,
            "component_count": int(len(components)),
            "largest_component_faces": component_faces[:10],
        }
    except Exception as exc:
        return {"mesh_stats_error": repr(exc)}


def particulate_transform(path: Path, up_dir: str) -> dict[str, Any]:
    mesh = _load_mesh(path)
    rotation = up_dir_rotation(up_dir)
    rotated_vertices = np.asarray(mesh.vertices) @ rotation.T
    bbox_min = rotated_vertices.min(axis=0)
    bbox_max = rotated_vertices.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    scale = float((bbox_max - bbox_min).max())
    if scale <= 0.0:
        raise ValueError(f"Invalid mesh scale for {path}: {scale}")

    return {
        "input_coordinate_frame": "trellis_mesh",
        "particulate_coordinate_frame": "particulate_normalized_z_up",
        "up_dir": up_dir,
        "rotation_input_to_particulate_z_up": rotation.tolist(),
        "center_after_up_rotation": center.tolist(),
        "scale_after_up_rotation": scale,
        "transform_formula": "p_particulate = (R_input_to_particulate @ p_input - center_after_up_rotation) / scale_after_up_rotation",
        "inverse_formula": "p_input = R_input_to_particulate.T @ (p_particulate * scale_after_up_rotation + center_after_up_rotation)",
    }


def up_dir_rotation(up_dir: str) -> np.ndarray:
    if up_dir == "X":
        return np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=np.float64)
    if up_dir == "-X":
        return np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float64)
    if up_dir == "Y":
        return np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
    if up_dir == "-Y":
        return np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
    if up_dir == "Z":
        return np.eye(3, dtype=np.float64)
    if up_dir == "-Z":
        return np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
    raise ValueError(f"Invalid up direction: {up_dir}")


def decimate_mesh(input_mesh: Path, output_mesh: Path, target_faces: int, overwrite: bool = False) -> dict[str, Any]:
    if output_mesh.exists() and not overwrite:
        stats = mesh_stats(output_mesh)
        stats["path"] = str(output_mesh)
        stats["skipped_existing"] = True
        return stats

    import open3d as o3d
    import trimesh

    source = _load_mesh(input_mesh)
    source_vertices = np.asarray(source.vertices)
    source_faces = np.asarray(source.faces)
    o3_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(source_vertices),
        triangles=o3d.utility.Vector3iVector(source_faces),
    )
    o3_mesh.remove_duplicated_vertices()
    o3_mesh.remove_duplicated_triangles()
    o3_mesh.remove_degenerate_triangles()
    o3_mesh.remove_unreferenced_vertices()
    o3_mesh.compute_vertex_normals()

    if np.asarray(o3_mesh.triangles).shape[0] > target_faces:
        o3_mesh = o3_mesh.simplify_quadric_decimation(target_number_of_triangles=target_faces)
        o3_mesh.remove_degenerate_triangles()
        o3_mesh.remove_duplicated_triangles()
        o3_mesh.remove_duplicated_vertices()
        o3_mesh.remove_unreferenced_vertices()
        o3_mesh.compute_vertex_normals()

    output_mesh.parent.mkdir(parents=True, exist_ok=True)
    out_mesh = trimesh.Trimesh(
        vertices=np.asarray(o3_mesh.vertices),
        faces=np.asarray(o3_mesh.triangles),
        process=False,
    )
    out_mesh.export(output_mesh)
    stats = mesh_stats(output_mesh)
    stats["path"] = str(output_mesh)
    stats["source_path"] = str(input_mesh)
    stats["target_faces"] = int(target_faces)
    stats["skipped_existing"] = False
    return stats


def selected_targets(manifest, targets: list[str] | None):
    target_by_id = {target.object_id: target for target in manifest.targets}
    if targets:
        missing = set(targets) - set(target_by_id)
        if missing:
            raise KeyError(f"Targets not found in manifest: {sorted(missing)}")
        return [target_by_id[target_id] for target_id in targets]
    return [target for target in manifest.targets if target.object_class == CLASS_ARTICULATED]


def trellis_mesh_for_target(manifest, target_id: str) -> MeshArtifact | None:
    target = next((item for item in manifest.targets if item.object_id == target_id), None)
    if target and target.mesh_id:
        mesh = next((item for item in manifest.meshes if item.mesh_id == target.mesh_id), None)
        if mesh and mesh.path:
            return mesh
    candidates = [
        mesh
        for mesh in manifest.meshes
        if mesh.target_id == target_id and mesh.source_stage == "trellis2" and mesh.role == "whole" and mesh.path
    ]
    return candidates[0] if candidates else None


def run_particulate(input_mesh: Path, output_dir: Path, config: ParticulateRunConfig) -> None:
    cmd = [
        str(config.python_bin),
        "infer.py",
        "--input_mesh",
        str(input_mesh),
        "--output_dir",
        str(output_dir),
        "--up_dir",
        config.up_dir,
        "--num_points",
        str(config.num_points),
        "--min_part_confidence",
        str(config.min_part_confidence),
    ]
    if config.no_strict:
        cmd.append("--no_strict")
    if config.export_urdf:
        cmd.append("--export_urdf")
    if config.export_mjcf:
        cmd.append("--export_mjcf")
    if config.eval:
        cmd.append("--eval")

    env = os.environ.copy()
    if config.hf_disable_xet:
        env["HF_HUB_DISABLE_XET"] = "1"
    subprocess.run(cmd, cwd=config.particulate_root, env=env, check=True)


def latest_matching(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.glob(pattern), key=lambda path: path.stat().st_mtime)
    return matches[-1] if matches else None


def _parse_vec(text: str | None) -> list[float] | None:
    if not text:
        return None
    return [float(item) for item in text.split()]


def parse_urdf_joints(target_id: str, urdf_path: Path | None) -> list[ArticulationJoint]:
    if urdf_path is None or not urdf_path.exists():
        return []

    root = ET.parse(urdf_path).getroot()
    joints = []
    for joint in root.findall("joint"):
        joint_type = joint.attrib.get("type", "unknown")
        if joint_type == "fixed":
            continue
        origin = joint.find("origin")
        axis = joint.find("axis")
        parent = joint.find("parent")
        child = joint.find("child")
        limit = joint.find("limit")
        name = joint.attrib.get("name", "joint")
        lower = limit.attrib.get("lower") if limit is not None else None
        upper = limit.attrib.get("upper") if limit is not None else None
        joints.append(
            ArticulationJoint(
                joint_id=f"{STAGE_PARTICULATE}:{target_id}:{name}",
                target_id=target_id,
                source_stage=STAGE_PARTICULATE,
                joint_type=joint_type,
                parent=parent.attrib.get("link") if parent is not None else None,
                child=child.attrib.get("link") if child is not None else None,
                origin_xyz=_parse_vec(origin.attrib.get("xyz") if origin is not None else None),
                axis_xyz=_parse_vec(axis.attrib.get("xyz") if axis is not None else None),
                limit_lower=float(lower) if lower is not None else None,
                limit_upper=float(upper) if upper is not None else None,
                metadata={"urdf_joint_name": name, "urdf": str(urdf_path)},
            )
        )
    return joints


def load_npz_summary(npz_path: Path | None) -> dict[str, Any]:
    if npz_path is None or not npz_path.exists():
        return {}

    data = np.load(npz_path, allow_pickle=True)
    summary: dict[str, Any] = {"npz": str(npz_path), "arrays": {}}
    for key in data.files:
        arr = data[key]
        item: dict[str, Any] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
        if key == "face_part_ids":
            unique, counts = np.unique(arr, return_counts=True)
            item["part_face_counts"] = {str(int(pid)): int(count) for pid, count in zip(unique, counts)}
        elif arr.size <= 64:
            item["values"] = arr.tolist()
        summary["arrays"][key] = item
    return summary


def part_mesh_artifacts(
    target_id: str,
    urdf_path: Path | None,
    parent_mesh_id: str,
    run_record: dict[str, Any],
) -> list[MeshArtifact]:
    if urdf_path is None:
        return []

    meshes = []
    for mesh_path in sorted((urdf_path.parent / "meshes").glob("part_*.obj")):
        label_text = mesh_path.stem.removeprefix("part_")
        try:
            part_label = int(label_text)
        except ValueError:
            part_label = None
        meshes.append(
            MeshArtifact(
                mesh_id=f"{STAGE_PARTICULATE}:{target_id}:part_{label_text}",
                target_id=target_id,
                source_stage=STAGE_PARTICULATE,
                status=STATUS_COMPLETED,
                path=str(mesh_path),
                role="articulated_part",
                format="obj",
                part_label=part_label,
                parent_mesh_id=parent_mesh_id,
                stats=mesh_stats(mesh_path),
                metadata=run_record,
            )
        )
    return meshes


def upsert_meshes(manifest, meshes: list[MeshArtifact]) -> None:
    existing = {mesh.mesh_id: idx for idx, mesh in enumerate(manifest.meshes)}
    for mesh in meshes:
        if mesh.mesh_id in existing:
            manifest.meshes[existing[mesh.mesh_id]] = mesh
        else:
            manifest.meshes.append(mesh)


def upsert_joints(manifest, joints: list[ArticulationJoint]) -> None:
    existing = {joint.joint_id: idx for idx, joint in enumerate(manifest.articulation_joints)}
    for joint in joints:
        if joint.joint_id in existing:
            manifest.articulation_joints[existing[joint.joint_id]] = joint
        else:
            manifest.articulation_joints.append(joint)


def ingest_target_result(
    manifest,
    target,
    source_mesh: MeshArtifact,
    decimated_mesh_path: Path,
    run_dir: Path,
    config: ParticulateRunConfig,
    decimation_stats: dict[str, Any],
) -> dict[str, Any]:
    transform = particulate_transform(decimated_mesh_path, config.up_dir)
    pred_npz = run_dir / "eval" / "pred.npz"
    pred_obj = run_dir / "eval" / "pred.obj"
    urdf = latest_matching(run_dir, "urdf_*/model.urdf")
    mjcf = latest_matching(run_dir, "mjcf_*/model.xml")
    axes_glb = latest_matching(run_dir, "mesh_parts_with_axes_*.glb")
    animated_glb = latest_matching(run_dir, "animated_textured_*.glb")
    output_paths = [path for path in [pred_npz, pred_obj, urdf, mjcf, axes_glb, animated_glb] if path and path.exists()]

    run_record = {
        "stage": STAGE_PARTICULATE,
        "target_id": target.object_id,
        "status": STATUS_COMPLETED,
        "source_mesh_id": source_mesh.mesh_id,
        "source_mesh": source_mesh.path,
        "decimated_mesh": str(decimated_mesh_path),
        "output_dir": str(run_dir),
        "outputs": {path.name: str(path) for path in output_paths},
        "config": {
            "particulate_root": str(config.particulate_root),
            "python_bin": str(config.python_bin),
            "up_dir": config.up_dir,
            "num_points": config.num_points,
            "target_faces": config.target_faces,
            "min_part_confidence": config.min_part_confidence,
            "strict": not config.no_strict,
            "hf_disable_xet": config.hf_disable_xet,
        },
        "decimation": decimation_stats,
        "coordinate_transform": transform,
        "prediction": load_npz_summary(pred_npz),
        "updated_at": _utc_now(),
    }
    write_json(run_dir / "particulate_run.json", run_record)

    decimated_mesh_id = f"{STAGE_PARTICULATE}:{target.object_id}:input_decimated_{config.target_faces}"
    decimated_artifact = MeshArtifact(
        mesh_id=decimated_mesh_id,
        target_id=target.object_id,
        source_stage=STAGE_PARTICULATE,
        status=STATUS_COMPLETED,
        path=str(decimated_mesh_path),
        role="particulate_input",
        format=decimated_mesh_path.suffix.lower().lstrip("."),
        parent_mesh_id=source_mesh.mesh_id,
        stats=mesh_stats(decimated_mesh_path),
        metadata=run_record,
    )
    parts = part_mesh_artifacts(target.object_id, urdf, decimated_mesh_id, run_record)
    joints = parse_urdf_joints(target.object_id, urdf)
    for joint in joints:
        joint.metadata["coordinate_frame"] = transform["particulate_coordinate_frame"]
        joint.metadata["coordinate_transform"] = transform
        joint.metadata["prediction_npz"] = str(pred_npz) if pred_npz.exists() else None
    upsert_meshes(manifest, [decimated_artifact] + parts)
    upsert_joints(manifest, joints)

    if joints:
        target.articulation_id = joints[0].joint_id

    return {
        "target_id": target.object_id,
        "source_mesh_id": source_mesh.mesh_id,
        "decimated_mesh_id": decimated_mesh_id,
        "joint_ids": [joint.joint_id for joint in joints],
        "part_mesh_ids": [mesh.mesh_id for mesh in parts],
        "run_record": str(run_dir / "particulate_run.json"),
        "outputs": [str(path) for path in output_paths],
    }


def run_particulate_jobs(config: ParticulateRunConfig) -> dict[str, Any]:
    manifest = read_project_manifest(config.manifest_path)
    targets = selected_targets(manifest, config.targets)
    if not targets:
        raise ValueError("No articulated targets selected for Particulate.")

    input_root = config.input_root or (config.project_root / "outputs" / "particulate_inputs")
    output_root = config.output_root or (config.project_root / "outputs" / "particulate")
    results = []
    for target in targets:
        source_mesh = trellis_mesh_for_target(manifest, target.object_id)
        if source_mesh is None or not source_mesh.path:
            raise FileNotFoundError(f"No TRELLIS whole mesh found for target {target.object_id}")
        source_mesh_path = _resolve_project_path(config.project_root, source_mesh.path)
        if source_mesh_path is None or not source_mesh_path.exists():
            raise FileNotFoundError(f"Missing source mesh for target {target.object_id}: {source_mesh.path}")

        decimated_mesh_path = (
            input_root / target.object_id / f"{target.object_id}_decimated_{config.target_faces}.glb"
        )
        decimation_stats = decimate_mesh(
            source_mesh_path,
            decimated_mesh_path,
            target_faces=config.target_faces,
            overwrite=config.overwrite,
        )

        run_dir = output_root / f"{target.object_id}_decimated_{config.target_faces}"
        pred_npz = run_dir / "eval" / "pred.npz"
        if config.run_inference and (config.overwrite or not pred_npz.exists()):
            run_particulate(decimated_mesh_path, run_dir, config)
        elif not pred_npz.exists():
            raise FileNotFoundError(f"Missing existing Particulate result: {pred_npz}")

        results.append(
            ingest_target_result(manifest, target, source_mesh, decimated_mesh_path, run_dir, config, decimation_stats)
        )

    summary = {
        "stage": STAGE_PARTICULATE,
        "status": STATUS_COMPLETED,
        "updated_at": _utc_now(),
        "results": results,
    }
    summary_path = output_root / "particulate_summary.json"
    write_json(summary_path, summary)

    manifest.upsert_stage(
        PipelineStageRecord(
            stage=STAGE_PARTICULATE,
            status=STATUS_COMPLETED,
            inputs=[item["source_mesh_id"] for item in results],
            outputs=[output for item in results for output in item["outputs"]],
            message="Local Particulate articulation inference completed.",
            metadata={"summary_path": str(summary_path), "target_count": len(results)},
        )
    )
    write_project_manifest(config.manifest_path, manifest)
    return summary


def ensure_manifest_paths(project_root: Path, manifest_path: Path | None) -> tuple[Path, Path]:
    project_root = project_root.resolve()
    if manifest_path is None:
        manifest_path = project_root / "outputs" / "project_manifest.json"
    elif not manifest_path.is_absolute():
        manifest_path = project_root / manifest_path
    return project_root, manifest_path.resolve()


def build_config(args: argparse.Namespace) -> ParticulateRunConfig:
    project_root, manifest_path = ensure_manifest_paths(args.project_root, args.manifest)
    targets = [item.strip() for item in args.targets.split(",") if item.strip()] if args.targets else None
    return ParticulateRunConfig(
        project_root=project_root,
        manifest_path=manifest_path,
        particulate_root=args.particulate_root.resolve(),
        python_bin=args.python_bin.resolve(),
        output_root=args.output_root.resolve() if args.output_root else None,
        input_root=args.input_root.resolve() if args.input_root else None,
        targets=targets,
        up_dir=args.up_dir,
        num_points=args.num_points,
        target_faces=args.target_faces,
        min_part_confidence=args.min_part_confidence,
        no_strict=args.no_strict,
        export_urdf=not args.no_export_urdf,
        export_mjcf=not args.no_export_mjcf,
        eval=not args.no_eval,
        hf_disable_xet=not args.allow_xet,
        overwrite=args.overwrite,
        run_inference=not args.ingest_only,
    )


def config_to_dict(config: ParticulateRunConfig) -> dict[str, Any]:
    out = asdict(config)
    return {key: str(value) if isinstance(value, Path) else value for key, value in out.items()}
