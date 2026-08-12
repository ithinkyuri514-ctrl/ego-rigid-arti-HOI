"""Hunyuan3D (浑圆3D) mesh-generation stage.

Drop-in alternative to the local TRELLIS.2 stage. It reuses the exact same
prepared cutout inputs recorded in `manifest.trellis_jobs` (SAM2 mask ->
square RGBA/white cutout), sends each to the Hunyuan3D external API, downloads
the returned mesh, and registers it as a `MeshArtifact` in the ProjectManifest.

Since the whole point is a finer mesh than TRELLIS.2, by default this repoints
`target.mesh_id` to the Hunyuan3D mesh so downstream stages (Particulate /
camera alignment) pick it up automatically. The TRELLIS.2 mesh stays in the
manifest under its own id, so this is reversible (pass --keep-mesh-id to leave
the target pointer untouched).

The API itself is wired in `hunyuan3d_client.py` (that is the file to edit).
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..manifest_io import read_project_manifest, write_json, write_project_manifest
from ..schemas import MeshArtifact, PipelineStageRecord, STAGE_HUNYUAN3D, STATUS_COMPLETED
from .hunyuan3d_client import Hunyuan3DClient, Hunyuan3DClientConfig, Hunyuan3DError


@dataclass
class Hunyuan3DRunConfig:
    project_root: Path
    manifest_path: Path
    output_root: Path | None = None
    upload_root: Path | None = None
    targets: list[str] | None = None
    prefer_input: str = "rgba"  # "rgba" or "white"
    overwrite: bool = False
    update_target_mesh_id: bool = True
    client: Hunyuan3DClientConfig = field(default_factory=Hunyuan3DClientConfig)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def mesh_stats(path: Path) -> dict[str, Any]:
    try:
        import trimesh

        mesh = trimesh.load(path, force="scene", process=False)
        vertices = faces = 0
        bounds = None
        if hasattr(mesh, "geometry"):
            for geom in mesh.geometry.values():
                vertices += int(len(getattr(geom, "vertices", [])))
                faces += int(len(getattr(geom, "faces", [])))
            if vertices:
                bounds = mesh.bounds.tolist()
        else:
            vertices = int(len(getattr(mesh, "vertices", [])))
            faces = int(len(getattr(mesh, "faces", [])))
            bounds = mesh.bounds.tolist()
        return {"vertices": vertices, "faces": faces, "bounds": bounds}
    except Exception as exc:  # trimesh optional / unreadable mesh
        return {"stats_error": str(exc)}


def select_jobs(manifest, target_ids: list[str] | None):
    if not target_ids:
        return manifest.trellis_jobs
    wanted = set(target_ids)
    missing = wanted - {job.target_id for job in manifest.trellis_jobs}
    if missing:
        raise KeyError(f"Targets not found in manifest trellis_jobs: {sorted(missing)}")
    return [job for job in manifest.trellis_jobs if job.target_id in wanted]


MESH_EXTS = (".glb", ".obj", ".ply", ".stl", ".usdz", ".fbx")


def _find_existing_mesh(directory: Path, stem: str) -> Path | None:
    """Return a previously downloaded mesh for `stem` regardless of extension."""
    for ext in MESH_EXTS:
        candidate = directory / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def _pick_input(job, prefer_input: str) -> Path:
    """Reuse the same cutout the TRELLIS.2 stage consumes."""
    order = (job.cutout_rgba, job.cutout_white) if prefer_input == "rgba" else (job.cutout_white, job.cutout_rgba)
    for candidate in order:
        if candidate and Path(candidate).exists():
            return Path(candidate)
    raise FileNotFoundError(f"No cutout input found for {job.target_id} (rgba/white both missing).")


def run_job(job, manifest, client: Hunyuan3DClient, config: Hunyuan3DRunConfig) -> dict[str, Any]:
    input_path = _pick_input(job, config.prefer_input)

    output_root = config.output_root or (config.project_root / "outputs" / "hunyuan3d")
    upload_root = config.upload_root or (config.project_root / "inputs" / "hunyuan3d_meshes")
    target_output_dir = output_root / job.target_id
    upload_whole_dir = upload_root / job.target_id / "whole"
    # Hunyuan3D may return glb or obj; the real extension is decided at download.
    local_stem = target_output_dir / job.target_id

    existing_local = _find_existing_mesh(target_output_dir, job.target_id)
    existing_upload = _find_existing_mesh(upload_whole_dir, job.target_id)
    if existing_local and existing_upload and not config.overwrite:
        print(f"[hunyuan3d] skip existing mesh for {job.target_id}: {existing_local}", flush=True)
        local_mesh_path = existing_local
        upload_mesh_path = existing_upload
        mesh_format = local_mesh_path.suffix.lstrip(".").lower() or "glb"
        api_meta: dict[str, Any] = {"skipped_existing": True}
    else:
        print(f"[hunyuan3d] generating {job.target_id} from {input_path}", flush=True)
        result = client.reconstruct(image_path=input_path, dest_path=local_stem)
        local_mesh_path = result.mesh_path
        mesh_format = result.mesh_format
        api_meta = {
            "task_id": result.task_id,
            "mesh_format": result.mesh_format,
            "raw_response": result.raw_response,
        }
        upload_mesh_path = upload_whole_dir / f"{job.target_id}.{mesh_format}"
        upload_mesh_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_mesh_path, upload_mesh_path)

    stats = mesh_stats(upload_mesh_path)
    run_record = {
        "target_id": job.target_id,
        "status": STATUS_COMPLETED,
        "input_image": str(input_path),
        "source_mask_id": job.source_mask_id,
        "mesh_format": mesh_format,
        "output_mesh": str(local_mesh_path),
        "upload_mesh": str(upload_mesh_path),
        "stats": stats,
        "api": api_meta,
        "config": {
            "base_url": config.client.base_url,
            "model": config.client.model,
            "prefer_input": config.prefer_input,
        },
        "updated_at": _utc_now(),
    }
    write_json(target_output_dir / "hunyuan3d_run.json", run_record)

    mesh_id = f"hunyuan3d:{job.target_id}:whole:{upload_mesh_path.stem}"
    existing = {mesh.mesh_id: idx for idx, mesh in enumerate(manifest.meshes)}
    mesh_artifact = MeshArtifact(
        mesh_id=mesh_id,
        target_id=job.target_id,
        source_stage=STAGE_HUNYUAN3D,
        status=STATUS_COMPLETED,
        path=str(upload_mesh_path),
        role="whole",
        format=mesh_format,
        stats=stats,
        metadata=run_record,
    )
    if mesh_id in existing:
        manifest.meshes[existing[mesh_id]] = mesh_artifact
    else:
        manifest.meshes.append(mesh_artifact)

    if config.update_target_mesh_id:
        for target in manifest.targets:
            if target.object_id == job.target_id:
                target.mesh_id = mesh_id
    return run_record


def run_hunyuan3d_jobs(config: Hunyuan3DRunConfig) -> dict[str, Any]:
    manifest = read_project_manifest(config.manifest_path)
    jobs = select_jobs(manifest, config.targets)
    if not jobs:
        raise ValueError("No mesh jobs selected (manifest.trellis_jobs is empty).")

    config.client.require_configured()
    client = Hunyuan3DClient(config.client)

    results = [run_job(job, manifest, client, config) for job in jobs]

    output_root = config.output_root or (config.project_root / "outputs" / "hunyuan3d")
    run_summary = {
        "stage": STAGE_HUNYUAN3D,
        "status": STATUS_COMPLETED,
        "updated_at": _utc_now(),
        "results": results,
    }
    summary_path = output_root / "hunyuan3d_summary.json"
    write_json(summary_path, run_summary)

    manifest.upsert_stage(
        PipelineStageRecord(
            stage=STAGE_HUNYUAN3D,
            status=STATUS_COMPLETED,
            inputs=[item["input_image"] for item in results],
            outputs=[item["upload_mesh"] for item in results],
            message="Hunyuan3D external-API mesh generation completed.",
            metadata={"summary_path": str(summary_path), "mesh_count": len(results)},
        )
    )
    write_project_manifest(config.manifest_path, manifest)
    return run_summary


def ensure_manifest_paths(project_root: Path, manifest_path: Path | None) -> tuple[Path, Path]:
    project_root = project_root.resolve()
    if manifest_path is None:
        manifest_path = project_root / "outputs" / "project_manifest.json"
    elif not manifest_path.is_absolute():
        manifest_path = project_root / manifest_path
    return project_root, manifest_path.resolve()


def build_config(args: argparse.Namespace) -> Hunyuan3DRunConfig:
    project_root, manifest_path = ensure_manifest_paths(args.project_root, args.manifest)
    targets = [item.strip() for item in args.targets.split(",") if item.strip()] if args.targets else None
    client_cfg = Hunyuan3DClientConfig.from_env(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        poll_interval_sec=args.poll_interval,
        poll_timeout_sec=args.poll_timeout,
    )
    return Hunyuan3DRunConfig(
        project_root=project_root,
        manifest_path=manifest_path,
        output_root=args.output_root.resolve() if args.output_root else None,
        upload_root=args.upload_root.resolve() if args.upload_root else None,
        targets=targets,
        prefer_input=args.prefer_input,
        overwrite=args.overwrite,
        update_target_mesh_id=not args.keep_mesh_id,
        client=client_cfg,
    )


def config_to_dict(config: Hunyuan3DRunConfig) -> dict[str, Any]:
    data = asdict(config)
    for key, value in list(data.items()):
        if isinstance(value, Path):
            data[key] = str(value)
    if isinstance(data.get("client"), dict):
        data["client"].pop("api_key", None)  # never echo the secret
    return data
