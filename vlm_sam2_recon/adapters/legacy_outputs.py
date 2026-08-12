"""Build a unified manifest from the current script-level outputs."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from ..manifest_io import read_json
from ..paths import ProjectPaths
from ..schemas import (
    CLASS_UNKNOWN,
    SCHEMA_VERSION,
    STAGE_EGOFORCE,
    STAGE_PARTICULATE,
    STAGE_PHYSXOMNI,
    STAGE_SAM2,
    STAGE_TRELLIS2,
    STAGE_VLM,
    STATUS_COMPLETED,
    STATUS_PREPARED,
    STATUS_WAITING,
    ArticulationJoint,
    BBox2D,
    EgoForceSequence,
    FrameRef,
    MaskArtifact,
    MeshArtifact,
    PhysXArtifact,
    PipelineStageRecord,
    ProjectManifest,
    SourceData,
    TargetObject,
    TrellisJob,
)


def _load_json_if_exists(path: Path) -> Any | None:
    if not path.exists():
        return None
    return read_json(path)


def _frame_index_from_name(path: str | Path | None) -> int | None:
    if not path:
        return None
    match = re.search(r"(\d+)", Path(path).stem)
    if not match:
        return None
    return int(match.group(1))


def _object_class(raw: dict[str, Any]) -> str:
    return raw.get("object_class_for_reconstruction") or raw.get("object_class") or CLASS_UNKNOWN


def _vlm_result(vlm_data: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(vlm_data, dict):
        return {}
    result = vlm_data.get("vlm_result", vlm_data)
    return result if isinstance(result, dict) else {}


def _infer_source(paths: ProjectPaths, vlm_json: Path | None, vlm_data: dict[str, Any] | None) -> SourceData:
    metadata = vlm_data.get("metadata", {}) if isinstance(vlm_data, dict) else {}
    frame_dir = metadata.get("frame_dir") or metadata.get("input_frame_dir")
    export_root = None
    sequence_id = metadata.get("sequence_id") or metadata.get("run_id")

    result = _vlm_result(vlm_data)
    targets = result.get("target_objects", []) if isinstance(result, dict) else []
    for target in targets:
        selected = target.get("selected_keyframe") or {}
        frame_path = selected.get("frame_path")
        if frame_path:
            frame_dir = frame_dir or str(Path(frame_path).parent)
            export_root = str(Path(frame_path).parent.parent)
            sequence_id = sequence_id or Path(export_root).name
            break

    if not sequence_id:
        sequence_id = paths.root.name

    return SourceData(
        sequence_id=str(sequence_id),
        frame_dir=str(frame_dir) if frame_dir else None,
        export_root=export_root,
        camera_id=metadata.get("camera_id") or metadata.get("camera"),
        raw_fps=metadata.get("raw_fps"),
        sample_fps=metadata.get("sample_fps"),
        vlm_json=str(vlm_json) if vlm_json else None,
        metadata=metadata,
    )


def _bbox_from_norm1000(selected: dict[str, Any] | None) -> BBox2D | None:
    if not selected:
        return None
    bbox = selected.get("bbox_2d_norm_1000")
    if not bbox:
        return None
    return BBox2D(xyxy=list(bbox), coordinate_space="norm1000", source=STAGE_VLM)


def _targets_from_vlm(vlm_data: dict[str, Any] | None) -> list[TargetObject]:
    result = _vlm_result(vlm_data)
    targets = []
    for raw in result.get("target_objects", []):
        selected = raw.get("selected_keyframe") or {}
        frame = FrameRef(
            frame_index=selected.get("frame_index"),
            frame_file=selected.get("frame_file"),
            frame_path=selected.get("frame_path"),
            timestamp_sec=selected.get("timestamp_sec"),
        )
        targets.append(
            TargetObject(
                object_id=raw.get("object_id", "unknown_target"),
                name_en=raw.get("name_en"),
                name_zh=raw.get("name_zh"),
                category=raw.get("category") or raw.get("semantic_category"),
                object_class=_object_class(raw),
                observed_state=raw.get("observed_state"),
                selected_keyframe=frame,
                selected_bbox=_bbox_from_norm1000(selected),
                relations=raw.get("relations", []),
                raw_vlm=raw,
            )
        )
    return targets


def _mask_id(target_id: str, frame_index: int | str | None, source: str) -> str:
    return f"{source}:{target_id}:frame_{frame_index if frame_index is not None else 'unknown'}"


def _mask_from_summary_item(item: dict[str, Any], source: str) -> MaskArtifact:
    target_id = item["target_id"]
    frame_index = item.get("frame_index")
    image_size = item.get("image_size") or {}
    bbox = item.get("box_xyxy_pixels")
    return MaskArtifact(
        mask_id=_mask_id(target_id, frame_index, source),
        target_id=target_id,
        frame=FrameRef(
            frame_index=frame_index,
            frame_file=item.get("frame_file"),
            frame_path=item.get("image_path"),
        ),
        source_stage=source,
        status=STATUS_COMPLETED,
        mask_png=item.get("mask_png"),
        mask_npy=item.get("mask_npy"),
        overlay_png=item.get("overlay_png"),
        prompt_png=item.get("prompt_overlay_png"),
        score=item.get("sam2_score"),
        area_pixels=item.get("mask_area_pixels"),
        bbox=BBox2D(
            xyxy=list(bbox),
            coordinate_space="pixel",
            image_width=image_size.get("width"),
            image_height=image_size.get("height"),
            source=source,
        )
        if bbox
        else None,
        positive_points_pixels=item.get("positive_points_pixels") or [],
        negative_points_pixels=item.get("negative_points_pixels") or [],
        metadata={
            "name_en": item.get("name_en"),
            "name_zh": item.get("name_zh"),
            "object_class_for_reconstruction": item.get("object_class_for_reconstruction"),
        },
    )


def _masks_from_sam2(paths: ProjectPaths) -> list[MaskArtifact]:
    masks = []
    batch = _load_json_if_exists(paths.sam2_masks / "sam2_mask_summary.json")
    if isinstance(batch, dict):
        for item in batch.get("targets", []):
            masks.append(_mask_from_summary_item(item, "sam2_batch"))

    interactive = _load_json_if_exists(paths.sam2_masks / "sam2_interactive_mask_summary.json")
    if isinstance(interactive, dict):
        targets = interactive.get("targets", {})
        items = targets.values() if isinstance(targets, dict) else targets
        for item in items:
            masks.append(_mask_from_summary_item(item, "sam2_interactive"))
    return masks


def _preferred_mask_ids(masks: list[MaskArtifact]) -> dict[str, str]:
    preferred: dict[str, str] = {}
    for mask in masks:
        if mask.target_id not in preferred:
            preferred[mask.target_id] = mask.mask_id
        if mask.source_stage == "sam2_interactive":
            preferred[mask.target_id] = mask.mask_id
    return preferred


def _trellis_jobs_from_outputs(paths: ProjectPaths, preferred_masks: dict[str, str]) -> list[TrellisJob]:
    jobs_json = _load_json_if_exists(paths.trellis2_interface / "trellis2_jobs.json")
    if not isinstance(jobs_json, list):
        return []

    jobs = []
    for item in jobs_json:
        target_id = item["target_id"]
        metadata_path = Path(item.get("metadata", ""))
        metadata = _load_json_if_exists(metadata_path) if metadata_path.exists() else None
        metadata = metadata if isinstance(metadata, dict) else {}
        bbox = metadata.get("bbox_xyxy")
        inputs = metadata.get("trellis2_inputs", {})
        upload_contract = metadata.get("upload_contract", {})
        jobs.append(
            TrellisJob(
                job_id=f"trellis2:{target_id}",
                target_id=target_id,
                status=STATUS_PREPARED,
                source_mask_id=preferred_masks.get(target_id),
                job_dir=item.get("job_dir"),
                metadata_path=item.get("metadata"),
                cutout_rgba=inputs.get("cutout_rgba") or item.get("preferred_trellis2_input"),
                cutout_white=inputs.get("cutout_white") or item.get("fallback_trellis2_input"),
                cutout_black=inputs.get("cutout_black"),
                crop_rgb=inputs.get("crop_rgb"),
                bbox=BBox2D(xyxy=list(bbox), coordinate_space="pixel", source=STAGE_TRELLIS2) if bbox else None,
                canvas_size=metadata.get("canvas_size"),
                object_scale=metadata.get("object_scale"),
                upload_dir=item.get("upload_dir") or upload_contract.get("upload_dir"),
                metadata={"legacy_metadata": metadata},
            )
        )
    return jobs


def _local_trellis_runs(paths: ProjectPaths) -> dict[str, dict[str, Any]]:
    runs = {}
    root = paths.outputs / "trellis2_local"
    if not root.exists():
        return runs
    for run_path in root.glob("*/trellis2_local_run.json"):
        data = _load_json_if_exists(run_path)
        if isinstance(data, dict) and data.get("target_id"):
            runs[data["target_id"]] = data
    return runs


def _mesh_from_trellis_upload(
    target_id: str,
    item: dict[str, Any],
    local_runs: dict[str, dict[str, Any]],
) -> MeshArtifact:
    path = item.get("path") or item.get("mesh_path")
    role = item.get("role", "whole")
    label = item.get("part_label")
    suffix = Path(path).suffix.lower() if path else None
    local_run = local_runs.get(target_id, {})
    stats = item.get("stats") or local_run.get("stats") or {}
    metadata = dict(item)
    if local_run:
        metadata["trellis2_local_run"] = local_run
    return MeshArtifact(
        mesh_id=f"trellis2:{target_id}:{role}:{label if label is not None else Path(path).stem if path else 'mesh'}",
        target_id=target_id,
        source_stage=STAGE_TRELLIS2,
        path=path,
        role=role,
        format=suffix[1:] if suffix else None,
        part_label=label,
        part_name=item.get("part_name"),
        stats=stats,
        metadata=metadata,
    )


def _trellis_meshes_from_uploads(paths: ProjectPaths) -> list[MeshArtifact]:
    uploaded = _load_json_if_exists(paths.trellis2_interface / "trellis2_uploaded_mesh_manifest.json")
    if not isinstance(uploaded, list):
        return []

    local_runs = _local_trellis_runs(paths)
    meshes = []
    for target in uploaded:
        target_id = target.get("target_id")
        for item in target.get("meshes", []):
            if target_id:
                meshes.append(_mesh_from_trellis_upload(target_id, item, local_runs))
    return meshes


def _parse_vec(text: str | None) -> list[float] | None:
    if not text:
        return None
    return [float(item) for item in text.split()]


def _physx_part_names(basic_info_json: str | None) -> dict[int, str]:
    if not basic_info_json:
        return {}
    data = _load_json_if_exists(Path(basic_info_json))
    if not isinstance(data, dict):
        return {}
    out = {}
    for part in data.get("parts", []):
        if "label" in part:
            out[int(part["label"])] = part.get("name")
    return out


def _joints_from_urdf(target_id: str, urdf_path: str | None) -> list[ArticulationJoint]:
    if not urdf_path or not Path(urdf_path).exists():
        return []

    joints = []
    root = ET.parse(urdf_path).getroot()
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
                joint_id=f"physxomni:{target_id}:{name}",
                target_id=target_id,
                source_stage=STAGE_PHYSXOMNI,
                joint_type=joint_type,
                parent=parent.attrib.get("link") if parent is not None else None,
                child=child.attrib.get("link") if child is not None else None,
                origin_xyz=_parse_vec(origin.attrib.get("xyz") if origin is not None else None),
                axis_xyz=_parse_vec(axis.attrib.get("xyz") if axis is not None else None),
                limit_lower=float(lower) if lower is not None else None,
                limit_upper=float(upper) if upper is not None else None,
                metadata={"urdf_joint_name": name},
            )
        )
    return joints


def _physx_from_outputs(paths: ProjectPaths) -> tuple[list[PhysXArtifact], list[MeshArtifact], list[ArticulationJoint]]:
    data = _load_json_if_exists(paths.physxomni / "reconstruction_dispatch_manifest.json")
    if not isinstance(data, dict):
        return [], [], []

    artifacts = []
    meshes = []
    joints = []
    for item in data.get("articulated", []):
        target_id = item.get("target_id")
        if not target_id:
            continue

        part_names = _physx_part_names(item.get("basic_info_json"))
        part_mesh_ids = []
        for mesh_path in item.get("part_meshes", []):
            part_label = _frame_index_from_name(mesh_path)
            mesh_id = f"physxomni:{target_id}:part_{part_label if part_label is not None else len(part_mesh_ids)}"
            part_mesh_ids.append(mesh_id)
            meshes.append(
                MeshArtifact(
                    mesh_id=mesh_id,
                    target_id=target_id,
                    source_stage=STAGE_PHYSXOMNI,
                    path=mesh_path,
                    role="articulated_part",
                    format=Path(mesh_path).suffix.lower().lstrip("."),
                    part_label=part_label,
                    part_name=part_names.get(part_label),
                    metadata={"physx_run_dir": item.get("physx_run_dir")},
                )
            )

        item_joints = _joints_from_urdf(target_id, item.get("urdf"))
        joints.extend(item_joints)
        artifacts.append(
            PhysXArtifact(
                artifact_id=f"physxomni:{target_id}",
                target_id=target_id,
                status=STATUS_COMPLETED if item.get("urdf") else STATUS_PREPARED,
                run_dir=item.get("physx_run_dir"),
                condition_image=(item.get("condition") or {}).get("condition_image"),
                transparent_cutout=(item.get("condition") or {}).get("transparent_cutout"),
                basic_info_json=item.get("basic_info_json"),
                basic_info_txt=item.get("basic_info_txt"),
                urdf=item.get("urdf"),
                mjcf_xml=item.get("mjcf_xml"),
                part_mesh_ids=part_mesh_ids,
                joint_ids=[joint.joint_id for joint in item_joints],
                metadata=item,
            )
        )

    for item in data.get("rigid", []):
        target_id = item.get("target_id")
        if not target_id:
            continue
        for mesh_item in item.get("meshes", []):
            path = mesh_item.get("path") or mesh_item.get("mesh_path")
            mesh_id = f"rigid:{target_id}:{Path(path).stem if path else 'mesh'}"
            meshes.append(
                MeshArtifact(
                    mesh_id=mesh_id,
                    target_id=target_id,
                    source_stage=STAGE_TRELLIS2,
                    status=STATUS_COMPLETED,
                    path=path,
                    role="whole",
                    format=Path(path).suffix.lower().lstrip(".") if path else None,
                    stats=mesh_item.get("stats", {}),
                    metadata=mesh_item,
                )
            )

    return artifacts, meshes, joints


def _mesh_stats(path: str | Path | None) -> dict[str, Any]:
    if not path or not Path(path).exists():
        return {}
    try:
        import numpy as np
        import trimesh

        mesh = trimesh.load(path, force="scene", process=False)
        if isinstance(mesh, trimesh.Scene):
            geoms = [geom for geom in mesh.geometry.values() if isinstance(geom, trimesh.Trimesh)]
            mesh = trimesh.util.concatenate(geoms) if geoms else mesh
        return {
            "vertices": int(len(getattr(mesh, "vertices", []))),
            "faces": int(len(getattr(mesh, "faces", []))),
            "bounds": np.asarray(mesh.bounds).tolist() if getattr(mesh, "bounds", None) is not None else None,
            "extents": np.asarray(mesh.extents).tolist() if getattr(mesh, "extents", None) is not None else None,
        }
    except Exception as exc:
        return {"mesh_stats_error": repr(exc)}


def _particulate_from_outputs(paths: ProjectPaths) -> tuple[list[MeshArtifact], list[ArticulationJoint], list[dict[str, Any]]]:
    summary = _load_json_if_exists(paths.particulate / "particulate_summary.json")
    if not isinstance(summary, dict):
        return [], [], []

    meshes: list[MeshArtifact] = []
    joints: list[ArticulationJoint] = []
    runs: list[dict[str, Any]] = []
    for item in summary.get("results", []):
        target_id = item.get("target_id")
        run_record_path = item.get("run_record")
        run_record = _load_json_if_exists(Path(run_record_path)) if run_record_path else None
        if not target_id or not isinstance(run_record, dict):
            continue
        runs.append(run_record)

        decimated_mesh = run_record.get("decimated_mesh")
        target_faces = (run_record.get("config") or {}).get("target_faces", "mesh")
        decimated_mesh_id = item.get("decimated_mesh_id") or f"{STAGE_PARTICULATE}:{target_id}:input_decimated_{target_faces}"
        if decimated_mesh:
            meshes.append(
                MeshArtifact(
                    mesh_id=decimated_mesh_id,
                    target_id=target_id,
                    source_stage=STAGE_PARTICULATE,
                    status=STATUS_COMPLETED,
                    path=decimated_mesh,
                    role="particulate_input",
                    format=Path(decimated_mesh).suffix.lower().lstrip("."),
                    parent_mesh_id=run_record.get("source_mesh_id"),
                    stats=_mesh_stats(decimated_mesh),
                    metadata=run_record,
                )
            )

        urdf = None
        outputs = run_record.get("outputs") or {}
        for output_path in outputs.values():
            candidate = Path(output_path)
            if candidate.name == "model.urdf" and candidate.exists():
                urdf = candidate
                break

        if urdf:
            for mesh_path in sorted((urdf.parent / "meshes").glob("part_*.obj")):
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
                        parent_mesh_id=decimated_mesh_id,
                        stats=_mesh_stats(mesh_path),
                        metadata=run_record,
                    )
                )

            for joint in _joints_from_urdf(target_id, str(urdf)):
                joint.joint_id = joint.joint_id.replace(f"{STAGE_PHYSXOMNI}:{target_id}:", f"{STAGE_PARTICULATE}:{target_id}:")
                joint.source_stage = STAGE_PARTICULATE
                joint.metadata["coordinate_frame"] = (run_record.get("coordinate_transform") or {}).get(
                    "particulate_coordinate_frame"
                )
                joint.metadata["coordinate_transform"] = run_record.get("coordinate_transform")
                joints.append(joint)

    return meshes, joints, runs


def _egoforce_from_outputs(paths: ProjectPaths) -> list[EgoForceSequence]:
    root = paths.egoforce_rgb_right
    if not root.exists():
        return []

    per_frame = sorted(root.glob("*_egoforce_meshes.npz"))
    mesh_obj_files = sorted(root.glob("*.obj"))
    sequence_npz = root / "sequence_egoforce_meshes.npz"
    frame_indices = [_frame_index_from_name(path.name) for path in per_frame]
    frame_indices = [item for item in frame_indices if item is not None]

    arrays: dict[str, Any] = {}
    image_paths: list[str] = []
    if sequence_npz.exists():
        try:
            import numpy as np

            data = np.load(sequence_npz, allow_pickle=True)
            arrays = {
                key: {"shape": list(data[key].shape), "dtype": str(data[key].dtype)}
                for key in data.files
                if hasattr(data[key], "shape")
            }
            if "image_paths" in data.files:
                image_paths = [str(item) for item in data["image_paths"].tolist()]
        except Exception as exc:
            arrays = {"read_error": str(exc)}

    status = STATUS_COMPLETED if sequence_npz.exists() or per_frame else STATUS_WAITING
    return [
        EgoForceSequence(
            artifact_id="egoforce:rgb_right",
            status=status,
            sequence_npz=str(sequence_npz) if sequence_npz.exists() else None,
            output_dir=str(root),
            frame_count=len(frame_indices) or (len(image_paths) if image_paths else None),
            frame_indices=frame_indices,
            image_paths=image_paths,
            per_frame_npz=[str(path) for path in per_frame],
            mesh_obj_files=[str(path) for path in mesh_obj_files],
            arrays=arrays,
        )
    ]


def _stage_records(
    paths: ProjectPaths,
    vlm_json: Path | None,
    masks: list[MaskArtifact],
    trellis_jobs: list[TrellisJob],
    trellis_meshes: list[MeshArtifact],
    particulate_runs: list[dict[str, Any]],
    physx_artifacts: list[PhysXArtifact],
    egoforce: list[EgoForceSequence],
) -> list[PipelineStageRecord]:
    return [
        PipelineStageRecord(
            stage=STAGE_VLM,
            status=STATUS_COMPLETED if vlm_json and vlm_json.exists() else STATUS_WAITING,
            inputs=[],
            outputs=[str(vlm_json)] if vlm_json else [],
        ),
        PipelineStageRecord(
            stage=STAGE_SAM2,
            status=STATUS_COMPLETED if masks else STATUS_WAITING,
            inputs=[str(vlm_json)] if vlm_json else [],
            outputs=[str(paths.sam2_masks)],
            metadata={"mask_count": len(masks)},
        ),
        PipelineStageRecord(
            stage=STAGE_TRELLIS2,
            status=STATUS_COMPLETED if trellis_meshes else STATUS_PREPARED if trellis_jobs else STATUS_WAITING,
            inputs=[mask.mask_png for mask in masks if mask.mask_png],
            outputs=[str(paths.trellis2_interface)],
            metadata={"job_count": len(trellis_jobs), "mesh_count": len(trellis_meshes)},
        ),
        PipelineStageRecord(
            stage=STAGE_PARTICULATE,
            status=STATUS_COMPLETED if particulate_runs else STATUS_WAITING,
            inputs=[run.get("source_mesh") for run in particulate_runs if run.get("source_mesh")],
            outputs=[run.get("output_dir") for run in particulate_runs if run.get("output_dir")],
            metadata={"run_count": len(particulate_runs)},
        ),
        PipelineStageRecord(
            stage=STAGE_PHYSXOMNI,
            status=STATUS_COMPLETED if physx_artifacts else STATUS_WAITING,
            inputs=[str(paths.physxomni / "reconstruction_dispatch_manifest.json")],
            outputs=[str(paths.physxomni)],
            metadata={"artifact_count": len(physx_artifacts)},
        ),
        PipelineStageRecord(
            stage=STAGE_EGOFORCE,
            status=egoforce[0].status if egoforce else STATUS_WAITING,
            inputs=[],
            outputs=[str(paths.egoforce_rgb_right)],
            metadata={"sequence_count": len(egoforce)},
        ),
    ]


def build_manifest_from_legacy_outputs(project_root: str | Path) -> ProjectManifest:
    paths = ProjectPaths.from_root(project_root)

    sam2_batch = _load_json_if_exists(paths.sam2_masks / "sam2_mask_summary.json")
    vlm_json = None
    vlm_data = None
    if isinstance(sam2_batch, dict) and sam2_batch.get("vlm_json"):
        candidate = Path(sam2_batch["vlm_json"])
        if candidate.exists():
            vlm_json = candidate
            loaded = _load_json_if_exists(candidate)
            vlm_data = loaded if isinstance(loaded, dict) else None

    source = _infer_source(paths, vlm_json, vlm_data)
    manifest = ProjectManifest(
        schema_version=SCHEMA_VERSION,
        project_root=str(paths.root),
        manifest_id=f"{source.sequence_id}:legacy-bootstrap",
        source=source,
        legacy_refs={},
        notes=[
            "Bootstrapped from script-level outputs. Existing scripts are not migrated yet.",
            "Prefer sam2_interactive masks over sam2_batch masks when both exist for a target.",
        ],
    )

    manifest.targets = _targets_from_vlm(vlm_data)
    manifest.masks = _masks_from_sam2(paths)

    preferred_masks = _preferred_mask_ids(manifest.masks)
    for target in manifest.targets:
        target.selected_mask_id = preferred_masks.get(target.object_id)

    manifest.trellis_jobs = _trellis_jobs_from_outputs(paths, preferred_masks)
    manifest.meshes = _trellis_meshes_from_uploads(paths)

    physx_artifacts, physx_meshes, physx_joints = _physx_from_outputs(paths)
    manifest.physx_artifacts = physx_artifacts
    manifest.meshes.extend(physx_meshes)
    manifest.articulation_joints = physx_joints
    particulate_meshes, particulate_joints, particulate_runs = _particulate_from_outputs(paths)
    manifest.meshes.extend(particulate_meshes)
    manifest.articulation_joints.extend(particulate_joints)
    manifest.egoforce_sequences = _egoforce_from_outputs(paths)

    first_mesh_by_target: dict[str, str] = {}
    for mesh in manifest.meshes:
        first_mesh_by_target.setdefault(mesh.target_id, mesh.mesh_id)
    first_joint_by_target: dict[str, str] = {}
    for source_stage in (STAGE_PARTICULATE, STAGE_PHYSXOMNI):
        for joint in manifest.articulation_joints:
            if joint.source_stage == source_stage:
                first_joint_by_target.setdefault(joint.target_id, joint.joint_id)
    for target in manifest.targets:
        target.mesh_id = first_mesh_by_target.get(target.object_id)
        target.articulation_id = first_joint_by_target.get(target.object_id)

    manifest.stages = _stage_records(
        paths=paths,
        vlm_json=vlm_json,
        masks=manifest.masks,
        trellis_jobs=manifest.trellis_jobs,
        trellis_meshes=[mesh for mesh in manifest.meshes if mesh.source_stage == STAGE_TRELLIS2],
        particulate_runs=particulate_runs,
        physx_artifacts=manifest.physx_artifacts,
        egoforce=manifest.egoforce_sequences,
    )

    for ref in [
        paths.sam2_masks / "sam2_mask_summary.json",
        paths.sam2_masks / "sam2_interactive_mask_summary.json",
        paths.trellis2_interface / "trellis2_interface_manifest.json",
        paths.outputs / "trellis2_local" / "trellis2_local_summary.json",
        paths.particulate / "particulate_summary.json",
        paths.physxomni / "reconstruction_dispatch_manifest.json",
    ]:
        if ref.exists():
            manifest.legacy_refs[ref.stem] = str(ref)

    manifest.touch()
    return manifest
