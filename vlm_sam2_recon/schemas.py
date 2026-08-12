"""Shared manifest schemas for the reconstruction pipeline.

The project currently has several script-local JSON formats. These dataclasses
define the cross-stage contract that future modules should read and write.
They intentionally avoid third-party validation dependencies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "0.1.0"

STAGE_VLM = "vlm"
STAGE_SAM2 = "sam2"
STAGE_TRELLIS2 = "trellis2"
STAGE_HUNYUAN3D = "hunyuan3d"
STAGE_PARTICULATE = "particulate"
STAGE_PHYSXOMNI = "physxomni"
STAGE_EGOFORCE = "egoforce"
STAGE_POSE = "pose"
STAGE_CONTACT = "contact"

STATUS_PENDING = "pending"
STATUS_PREPARED = "prepared"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_WAITING = "waiting"
STATUS_SKIPPED = "skipped"

CLASS_RIGID = "rigid"
CLASS_ARTICULATED = "articulated"
CLASS_UNKNOWN = "unknown"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {k: _jsonable(v) for k, v in asdict(value).items() if v is not None}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items() if v is not None}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _filtered_kwargs(cls: type, data: dict[str, Any]) -> dict[str, Any]:
    allowed = {item.name for item in fields(cls)}
    return {key: value for key, value in data.items() if key in allowed}


class SchemaMixin:
    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None):
        if data is None:
            return None
        return cls(**_filtered_kwargs(cls, data))


@dataclass
class FrameRef(SchemaMixin):
    frame_index: int | None = None
    frame_file: str | None = None
    frame_path: str | None = None
    timestamp_sec: float | None = None
    camera_id: str | None = None


@dataclass
class BBox2D(SchemaMixin):
    xyxy: list[float]
    coordinate_space: str = "pixel"
    image_width: int | None = None
    image_height: int | None = None
    source: str | None = None


@dataclass
class SourceData(SchemaMixin):
    sequence_id: str
    frame_dir: str | None = None
    export_root: str | None = None
    camera_id: str | None = None
    raw_fps: float | None = None
    sample_fps: float | None = None
    vlm_json: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineStageRecord(SchemaMixin):
    stage: str
    status: str
    updated_at: str = field(default_factory=utc_now_iso)
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TargetObject(SchemaMixin):
    object_id: str
    name_en: str | None = None
    name_zh: str | None = None
    category: str | None = None
    object_class: str = CLASS_UNKNOWN
    observed_state: str | None = None
    selected_keyframe: FrameRef | None = None
    selected_bbox: BBox2D | None = None
    selected_mask_id: str | None = None
    mesh_id: str | None = None
    articulation_id: str | None = None
    relations: list[dict[str, Any]] = field(default_factory=list)
    raw_vlm: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None):
        if data is None:
            return None
        clean = _filtered_kwargs(cls, data)
        clean["selected_keyframe"] = FrameRef.from_dict(clean.get("selected_keyframe"))
        clean["selected_bbox"] = BBox2D.from_dict(clean.get("selected_bbox"))
        return cls(**clean)


@dataclass
class MaskArtifact(SchemaMixin):
    mask_id: str
    target_id: str
    frame: FrameRef
    source_stage: str = STAGE_SAM2
    status: str = STATUS_COMPLETED
    mask_png: str | None = None
    mask_npy: str | None = None
    overlay_png: str | None = None
    prompt_png: str | None = None
    score: float | None = None
    area_pixels: int | None = None
    bbox: BBox2D | None = None
    positive_points_pixels: list[list[float]] = field(default_factory=list)
    negative_points_pixels: list[list[float]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None):
        if data is None:
            return None
        clean = _filtered_kwargs(cls, data)
        clean["frame"] = FrameRef.from_dict(clean.get("frame")) or FrameRef()
        clean["bbox"] = BBox2D.from_dict(clean.get("bbox"))
        return cls(**clean)


@dataclass
class TrellisJob(SchemaMixin):
    job_id: str
    target_id: str
    status: str = STATUS_PREPARED
    source_mask_id: str | None = None
    job_dir: str | None = None
    metadata_path: str | None = None
    cutout_rgba: str | None = None
    cutout_white: str | None = None
    cutout_black: str | None = None
    crop_rgb: str | None = None
    bbox: BBox2D | None = None
    canvas_size: int | None = None
    object_scale: float | None = None
    upload_dir: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None):
        if data is None:
            return None
        clean = _filtered_kwargs(cls, data)
        clean["bbox"] = BBox2D.from_dict(clean.get("bbox"))
        return cls(**clean)


@dataclass
class MeshArtifact(SchemaMixin):
    mesh_id: str
    target_id: str
    source_stage: str
    status: str = STATUS_COMPLETED
    path: str | None = None
    role: str = "whole"
    format: str | None = None
    part_label: int | None = None
    part_name: str | None = None
    parent_mesh_id: str | None = None
    stats: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ArticulationJoint(SchemaMixin):
    joint_id: str
    target_id: str
    source_stage: str = STAGE_PHYSXOMNI
    joint_type: str = "unknown"
    parent: str | None = None
    child: str | None = None
    origin_xyz: list[float] | None = None
    axis_xyz: list[float] | None = None
    limit_lower: float | None = None
    limit_upper: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PhysXArtifact(SchemaMixin):
    artifact_id: str
    target_id: str
    status: str
    run_dir: str | None = None
    condition_image: str | None = None
    transparent_cutout: str | None = None
    basic_info_json: str | None = None
    basic_info_txt: str | None = None
    urdf: str | None = None
    mjcf_xml: str | None = None
    part_mesh_ids: list[str] = field(default_factory=list)
    joint_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EgoForceSequence(SchemaMixin):
    artifact_id: str
    status: str
    sequence_npz: str | None = None
    output_dir: str | None = None
    frame_count: int | None = None
    frame_indices: list[int] = field(default_factory=list)
    image_paths: list[str] = field(default_factory=list)
    per_frame_npz: list[str] = field(default_factory=list)
    mesh_obj_files: list[str] = field(default_factory=list)
    arrays: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProjectManifest(SchemaMixin):
    schema_version: str
    project_root: str
    source: SourceData
    manifest_id: str | None = None
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    stages: list[PipelineStageRecord] = field(default_factory=list)
    targets: list[TargetObject] = field(default_factory=list)
    masks: list[MaskArtifact] = field(default_factory=list)
    trellis_jobs: list[TrellisJob] = field(default_factory=list)
    meshes: list[MeshArtifact] = field(default_factory=list)
    articulation_joints: list[ArticulationJoint] = field(default_factory=list)
    physx_artifacts: list[PhysXArtifact] = field(default_factory=list)
    egoforce_sequences: list[EgoForceSequence] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    legacy_refs: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def new(cls, project_root: str | Path, source: SourceData, manifest_id: str | None = None) -> "ProjectManifest":
        return cls(
            schema_version=SCHEMA_VERSION,
            project_root=str(project_root),
            source=source,
            manifest_id=manifest_id,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None):
        if data is None:
            return None
        clean = _filtered_kwargs(cls, data)
        clean["source"] = SourceData.from_dict(clean.get("source")) or SourceData(sequence_id="unknown")
        clean["stages"] = [PipelineStageRecord.from_dict(item) for item in clean.get("stages", [])]
        clean["targets"] = [TargetObject.from_dict(item) for item in clean.get("targets", [])]
        clean["masks"] = [MaskArtifact.from_dict(item) for item in clean.get("masks", [])]
        clean["trellis_jobs"] = [TrellisJob.from_dict(item) for item in clean.get("trellis_jobs", [])]
        clean["meshes"] = [MeshArtifact.from_dict(item) for item in clean.get("meshes", [])]
        clean["articulation_joints"] = [
            ArticulationJoint.from_dict(item) for item in clean.get("articulation_joints", [])
        ]
        clean["physx_artifacts"] = [PhysXArtifact.from_dict(item) for item in clean.get("physx_artifacts", [])]
        clean["egoforce_sequences"] = [EgoForceSequence.from_dict(item) for item in clean.get("egoforce_sequences", [])]
        return cls(**clean)

    def touch(self) -> None:
        self.updated_at = utc_now_iso()

    def stage(self, name: str) -> PipelineStageRecord | None:
        for item in self.stages:
            if item.stage == name:
                return item
        return None

    def upsert_stage(self, record: PipelineStageRecord) -> None:
        for idx, item in enumerate(self.stages):
            if item.stage == record.stage:
                self.stages[idx] = record
                self.touch()
                return
        self.stages.append(record)
        self.touch()

    def validate(self) -> list[str]:
        issues: list[str] = []
        if self.schema_version != SCHEMA_VERSION:
            issues.append(f"schema_version is {self.schema_version}, expected {SCHEMA_VERSION}")

        target_ids = [target.object_id for target in self.targets]
        target_id_set = set(target_ids)
        if len(target_ids) != len(target_id_set):
            issues.append("target object_id values must be unique")

        mask_ids = {mask.mask_id for mask in self.masks}
        mesh_ids = {mesh.mesh_id for mesh in self.meshes}
        joint_ids = {joint.joint_id for joint in self.articulation_joints}

        for target in self.targets:
            if target.selected_mask_id and target.selected_mask_id not in mask_ids:
                issues.append(f"target {target.object_id} references missing mask {target.selected_mask_id}")
            if target.mesh_id and target.mesh_id not in mesh_ids:
                issues.append(f"target {target.object_id} references missing mesh {target.mesh_id}")
            if target.articulation_id and target.articulation_id not in joint_ids:
                issues.append(f"target {target.object_id} references missing joint {target.articulation_id}")

        for mask in self.masks:
            if mask.target_id not in target_id_set:
                issues.append(f"mask {mask.mask_id} references missing target {mask.target_id}")

        for job in self.trellis_jobs:
            if job.target_id not in target_id_set:
                issues.append(f"trellis job {job.job_id} references missing target {job.target_id}")
            if job.source_mask_id and job.source_mask_id not in mask_ids:
                issues.append(f"trellis job {job.job_id} references missing mask {job.source_mask_id}")

        for mesh in self.meshes:
            if mesh.target_id not in target_id_set:
                issues.append(f"mesh {mesh.mesh_id} references missing target {mesh.target_id}")

        for joint in self.articulation_joints:
            if joint.target_id not in target_id_set:
                issues.append(f"joint {joint.joint_id} references missing target {joint.target_id}")

        for artifact in self.physx_artifacts:
            if artifact.target_id not in target_id_set:
                issues.append(f"physx artifact {artifact.artifact_id} references missing target {artifact.target_id}")
            for mesh_id in artifact.part_mesh_ids:
                if mesh_id not in mesh_ids:
                    issues.append(f"physx artifact {artifact.artifact_id} references missing mesh {mesh_id}")
            for joint_id in artifact.joint_ids:
                if joint_id not in joint_ids:
                    issues.append(f"physx artifact {artifact.artifact_id} references missing joint {joint_id}")

        return issues
