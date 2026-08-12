"""Local TRELLIS.2 execution stage.

This module is intended to run inside the `trellis2` conda environment. It
loads TRELLIS.2 once, consumes prepared TrellisJob records, and exports GLB
meshes back into the project's upload/ingest contract.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from ..manifest_io import read_project_manifest, write_json, write_project_manifest
from ..paths import ProjectPaths
from ..schemas import MeshArtifact, PipelineStageRecord, STAGE_TRELLIS2, STATUS_COMPLETED


DEFAULT_TRELLIS_ROOT = Path("/code/TRELLIS.2")
DEFAULT_WEIGHTS_DIR = DEFAULT_TRELLIS_ROOT / "pretrained_weights" / "TRELLIS.2-4B"


@dataclass
class Trellis2RunConfig:
    project_root: Path
    manifest_path: Path
    trellis_root: Path = DEFAULT_TRELLIS_ROOT
    weights_dir: Path = DEFAULT_WEIGHTS_DIR
    output_root: Path | None = None
    upload_root: Path | None = None
    targets: list[str] | None = None
    resolution: str = "1024"
    seed: int = 42
    decimation_target: int = 500000
    texture_size: int = 2048
    max_num_tokens: int = 49152
    ss_steps: int = 12
    shape_steps: int = 12
    tex_steps: int = 12
    ss_decoder: str | None = None
    dinov3_model: str | None = None
    rembg_model: str | None = None
    allow_download: bool = False
    load_rembg: bool = False
    preprocess_image: bool = True
    overwrite: bool = False
    extension_webp: bool = False

    @property
    def pipeline_type(self) -> str:
        return {
            "512": "512",
            "1024": "1024_cascade",
            "1536": "1536_cascade",
        }[self.resolution]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _bootstrap_trellis_imports(trellis_root: Path):
    trellis_root = trellis_root.resolve()
    if not trellis_root.exists():
        raise FileNotFoundError(f"TRELLIS.2 root not found: {trellis_root}")
    if str(trellis_root) not in sys.path:
        sys.path.insert(0, str(trellis_root))

    import app_local
    import torch
    import o_voxel
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    return app_local, torch, o_voxel, Trellis2ImageTo3DPipeline


def _app_local_args(config: Trellis2RunConfig) -> argparse.Namespace:
    return argparse.Namespace(
        weights_dir=str(config.weights_dir),
        ss_decoder=config.ss_decoder,
        dinov3_model=config.dinov3_model,
        rembg_model=config.rembg_model,
        allow_download=config.allow_download,
    )


def load_pipeline(config: Trellis2RunConfig):
    app_local, torch, o_voxel, pipeline_cls = _bootstrap_trellis_imports(config.trellis_root)
    app_local.set_runtime_env(allow_download=config.allow_download)
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    weights_dir, local_config = app_local.build_local_pipeline_config(_app_local_args(config))
    if not config.load_rembg:
        install_noop_rembg(local_config)
    try:
        print(f"[trellis2] loading pipeline from {weights_dir}", flush=True)
        pipeline = pipeline_cls.from_pretrained(str(weights_dir), config_file=local_config.name)
    finally:
        local_config.unlink(missing_ok=True)
    patch_dinov3_layers(pipeline)
    pipeline.cuda()
    return pipeline, torch, o_voxel


def install_noop_rembg(config_path: Path) -> None:
    """Patch the temporary pipeline config to avoid loading RMBG.

    Our inputs are alpha-masked SAM2 cutouts, so TRELLIS.2's preprocessing path
    does not need a background-removal model. Avoiding RMBG also sidesteps a
    local Transformers/BiRefNet compatibility issue.
    """
    from trellis2.pipelines import rembg

    class NoOpRMBG:
        def to(self, device):
            return self

        def cpu(self):
            return self

        def __call__(self, image: Image.Image) -> Image.Image:
            return image.convert("RGBA")

    setattr(rembg, "NoOpRMBG", NoOpRMBG)
    data = json.loads(config_path.read_text(encoding="utf-8"))
    data["args"]["rembg_model"] = {"name": "NoOpRMBG", "args": {}}
    config_path.write_text(json.dumps(data, indent=4) + "\n", encoding="utf-8")


def patch_dinov3_layers(pipeline) -> None:
    image_cond_model = getattr(pipeline, "image_cond_model", None)
    model = getattr(image_cond_model, "model", None)
    nested = getattr(model, "model", None)
    if model is not None and not hasattr(model, "layer") and hasattr(nested, "layer"):
        print("[trellis2] patching DINOv3 layer alias: model.layer -> model.model.layer", flush=True)
        setattr(model, "layer", nested.layer)


def select_jobs(manifest, target_ids: list[str] | None):
    if not target_ids:
        return manifest.trellis_jobs
    wanted = set(target_ids)
    missing = wanted - {job.target_id for job in manifest.trellis_jobs}
    if missing:
        raise KeyError(f"Targets not found in manifest trellis_jobs: {sorted(missing)}")
    return [job for job in manifest.trellis_jobs if job.target_id in wanted]


def mesh_stats(path: Path) -> dict[str, Any]:
    try:
        import trimesh

        mesh = trimesh.load(path, force="scene", process=False)
        vertices = 0
        faces = 0
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
    except Exception as exc:
        return {"stats_error": str(exc)}


def export_glb(
    mesh,
    pipeline,
    o_voxel,
    output_path: Path,
    config: Trellis2RunConfig,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=getattr(pipeline, "pbr_attr_layout", mesh.layout),
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=config.decimation_target,
        texture_size=config.texture_size,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        verbose=True,
    )
    glb.export(output_path, extension_webp=config.extension_webp)


def run_job(job, manifest, pipeline, torch, o_voxel, config: Trellis2RunConfig) -> dict[str, Any]:
    input_path = Path(job.cutout_rgba or job.cutout_white or "")
    if not input_path.exists():
        raise FileNotFoundError(f"Missing TRELLIS input for {job.target_id}: {input_path}")

    output_root = config.output_root or (config.project_root / "outputs" / "trellis2_local")
    upload_root = config.upload_root or (config.project_root / "inputs" / "trellis2_meshes")
    target_output_dir = output_root / job.target_id
    glb_path = target_output_dir / f"{job.target_id}.glb"
    upload_glb_path = upload_root / job.target_id / "whole" / f"{job.target_id}.glb"

    if glb_path.exists() and upload_glb_path.exists() and not config.overwrite:
        print(f"[trellis2] skip existing mesh for {job.target_id}: {glb_path}", flush=True)
    else:
        print(f"[trellis2] running {job.target_id}: {input_path}", flush=True)
        image = Image.open(input_path)
        if config.preprocess_image:
            image = image.convert("RGBA")
        else:
            image = image.convert("RGB")
        outputs = pipeline.run(
            image,
            seed=config.seed,
            preprocess_image=config.preprocess_image,
            pipeline_type=config.pipeline_type,
            max_num_tokens=config.max_num_tokens,
            sparse_structure_sampler_params={"steps": config.ss_steps},
            shape_slat_sampler_params={"steps": config.shape_steps},
            tex_slat_sampler_params={"steps": config.tex_steps},
        )
        mesh = outputs[0]
        mesh.simplify(16777216)
        export_glb(mesh, pipeline, o_voxel, glb_path, config)
        upload_glb_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(glb_path, upload_glb_path)
        torch.cuda.empty_cache()

    stats = mesh_stats(upload_glb_path)
    run_record = {
        "target_id": job.target_id,
        "status": STATUS_COMPLETED,
        "input_image": str(input_path),
        "source_mask_id": job.source_mask_id,
        "output_glb": str(glb_path),
        "upload_glb": str(upload_glb_path),
        "stats": stats,
        "config": {
            "trellis_root": str(config.trellis_root),
            "weights_dir": str(config.weights_dir),
            "resolution": config.resolution,
            "pipeline_type": config.pipeline_type,
            "seed": config.seed,
            "decimation_target": config.decimation_target,
            "texture_size": config.texture_size,
            "max_num_tokens": config.max_num_tokens,
            "ss_steps": config.ss_steps,
            "shape_steps": config.shape_steps,
            "tex_steps": config.tex_steps,
            "preprocess_image": config.preprocess_image,
        },
        "updated_at": _utc_now(),
    }
    write_json(target_output_dir / "trellis2_local_run.json", run_record)

    mesh_id = f"trellis2:{job.target_id}:whole:{upload_glb_path.stem}"
    existing = {mesh.mesh_id: idx for idx, mesh in enumerate(manifest.meshes)}
    mesh_artifact = MeshArtifact(
        mesh_id=mesh_id,
        target_id=job.target_id,
        source_stage=STAGE_TRELLIS2,
        status=STATUS_COMPLETED,
        path=str(upload_glb_path),
        role="whole",
        format="glb",
        stats=stats,
        metadata=run_record,
    )
    if mesh_id in existing:
        manifest.meshes[existing[mesh_id]] = mesh_artifact
    else:
        manifest.meshes.append(mesh_artifact)
    for target in manifest.targets:
        if target.object_id == job.target_id:
            target.mesh_id = mesh_id
    return run_record


def run_trellis2_jobs(config: Trellis2RunConfig) -> dict[str, Any]:
    manifest = read_project_manifest(config.manifest_path)
    jobs = select_jobs(manifest, config.targets)
    if not jobs:
        raise ValueError("No TRELLIS jobs selected.")

    pipeline, torch, o_voxel = load_pipeline(config)
    results = []
    for job in jobs:
        results.append(run_job(job, manifest, pipeline, torch, o_voxel, config))

    run_summary = {
        "stage": STAGE_TRELLIS2,
        "status": STATUS_COMPLETED,
        "updated_at": _utc_now(),
        "results": results,
    }
    summary_path = (config.output_root or (config.project_root / "outputs" / "trellis2_local")) / "trellis2_local_summary.json"
    write_json(summary_path, run_summary)

    manifest.upsert_stage(
        PipelineStageRecord(
            stage=STAGE_TRELLIS2,
            status=STATUS_COMPLETED,
            inputs=[item["input_image"] for item in results],
            outputs=[item["upload_glb"] for item in results],
            message="Local TRELLIS.2 mesh generation completed.",
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


def build_config(args: argparse.Namespace) -> Trellis2RunConfig:
    project_root, manifest_path = ensure_manifest_paths(args.project_root, args.manifest)
    targets = [item.strip() for item in args.targets.split(",") if item.strip()] if args.targets else None
    return Trellis2RunConfig(
        project_root=project_root,
        manifest_path=manifest_path,
        trellis_root=args.trellis_root.resolve(),
        weights_dir=args.weights_dir.resolve(),
        output_root=args.output_root.resolve() if args.output_root else None,
        upload_root=args.upload_root.resolve() if args.upload_root else None,
        targets=targets,
        resolution=args.resolution,
        seed=args.seed,
        decimation_target=args.decimation_target,
        texture_size=args.texture_size,
        max_num_tokens=args.max_num_tokens,
        ss_steps=args.ss_steps,
        shape_steps=args.shape_steps,
        tex_steps=args.tex_steps,
        ss_decoder=args.ss_decoder,
        dinov3_model=args.dinov3_model,
        rembg_model=args.rembg_model,
        allow_download=args.allow_download,
        load_rembg=args.load_rembg,
        preprocess_image=not args.no_preprocess,
        overwrite=args.overwrite,
        extension_webp=bool(args.webp and not args.no_webp),
    )


def config_to_dict(config: Trellis2RunConfig) -> dict[str, Any]:
    data = asdict(config)
    for key, value in list(data.items()):
        if isinstance(value, Path):
            data[key] = str(value)
    return data
