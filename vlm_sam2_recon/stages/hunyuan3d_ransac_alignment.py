"""Hunyuan3D whole-mesh alignment with RANSAC, ICP, and mask-edge scoring."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion, distance_transform_edt
from scipy.spatial import cKDTree

from .camera_alignment import (
    AlignmentConfig,
    apply_se3_to_mesh,
    apply_sim3,
    apply_sim3_to_mesh,
    camera_to_camera_matrix,
    color_by_depth,
    depth_points_in_right_camera,
    export_point_cloud,
    frame_name,
    frame_row,
    mask_bbox_quantiles,
    observed_mask_cloud_with_pixels,
    project_right_camera_points,
    projected_bbox_error,
    projected_bbox_quantiles,
    sample_mesh_points,
    save_binary_mask,
    save_projection_overlay,
    sim3_matrix,
    subsample_points,
    transform_points,
    translation_for_projected_shift,
    trimmed_sim3_icp,
    trimmed_mean,
    umeyama_sim3,
    visible_bidirectional_score,
)
from .screen_hinge_tracking import load_mesh


DEFAULT_EXPORT_ROOT = Path("/code/3DVideo_2026-07-01-19-24-06-667_spatialmp4_export")


@dataclass
class HunyuanRansacAlignmentConfig:
    project_root: Path
    export_root: Path = DEFAULT_EXPORT_ROOT
    target_id: str = "target_laptop"
    mesh_path: Path | None = None
    align_frame: int = 0
    view_frame: int | None = 5
    convention: str = "camera_to_rig"
    output_dir: Path | None = None
    depth_min_m: float = 0.1
    depth_max_m: float = 3.0
    depth_quantile_min: float = 0.03
    depth_quantile_max: float = 0.85
    canonical_samples: int = 30000
    observed_samples: int = 14000
    ransac_iterations: int = 700
    ransac_sample_size: int = 4
    ransac_inlier_threshold_m: float = 0.025
    ransac_trim_fraction: float = 0.60
    ransac_candidate_pool: int = 8000
    ransac_top_k: int = 8
    icp_iterations: int = 45
    icp_trim_fraction: float = 0.65
    edge_refine: bool = True
    edge_scale_min_multiplier: float = 0.90
    edge_scale_max_multiplier: float = 1.16
    edge_scale_steps: int = 53
    edge_shift_max_px: float = 60.0
    edge_shift_steps: int = 7
    edge_boundary_weight: float = 0.9
    edge_outside_weight: float = 8.0
    edge_bbox_weight: float = 0.35
    edge_depth_weight: float = 0.25
    random_seed: int = 42


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


def resolve_selected_mask(project_root: Path, target_id: str) -> tuple[Path, str]:
    manifest = read_json(project_root / "outputs" / "project_manifest.json")
    target = next(item for item in manifest["targets"] if item["object_id"] == target_id)
    mask_item = next(item for item in manifest["masks"] if item["mask_id"] == target["selected_mask_id"])
    return Path(mask_item["mask_npy"]), str(mask_item["mask_id"])


def resolve_hunyuan_mesh(project_root: Path, target_id: str, explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Hunyuan3D mesh not found: {path}")
        return path
    candidates: list[Path] = []
    for root in [
        project_root / "inputs" / "hunyuan3d_meshes" / target_id / "whole",
        project_root / "outputs" / "hunyuan3d" / target_id,
    ]:
        for ext in ("*.glb", "*.obj", "*.ply", "*.stl", "*.fbx"):
            candidates.extend(sorted(root.glob(ext)))
    if not candidates:
        raise FileNotFoundError(
            "No Hunyuan3D mesh found. Run scripts/run_hunyuan3d_local.py first, "
            "or pass --mesh-path to an existing Hunyuan3D mesh."
        )
    return candidates[0].resolve()


def load_whole_mesh(path: Path) -> trimesh.Trimesh:
    mesh = load_mesh(path)
    if len(mesh.faces) == 0:
        raise ValueError(f"Mesh has no faces: {path}")
    return mesh


def ransac_sim3_alignment(
    source: np.ndarray,
    target: np.ndarray,
    iterations: int,
    sample_size: int,
    inlier_threshold_m: float,
    trim_fraction: float,
    candidate_pool: int,
    top_k: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    source_pool = subsample_points(source, min(len(source), candidate_pool), seed + 10)
    target_pool = subsample_points(target, min(len(target), candidate_pool), seed + 11)
    tree = cKDTree(target_pool)
    source_center = source_pool.mean(axis=0)
    target_center = target_pool.mean(axis=0)
    source_diag = float(np.linalg.norm(np.ptp(source_pool, axis=0)))
    target_diag = float(np.linalg.norm(np.ptp(target_pool, axis=0)))
    base_scale = target_diag / max(source_diag, 1e-12)
    base_trans = target_center - base_scale * source_center

    candidates: list[dict[str, Any]] = [
        {
            "scale": base_scale,
            "rotation": np.eye(3, dtype=np.float64),
            "translation": base_trans,
            "source": "bbox_center_identity",
        }
    ]
    min_sample = max(3, int(sample_size))
    if len(source_pool) < min_sample or len(target_pool) < min_sample:
        raise ValueError("Not enough source/target points for RANSAC Sim3.")

    for _ in range(max(1, int(iterations))):
        src_idx = rng.choice(len(source_pool), size=min_sample, replace=False)
        tgt_idx = rng.choice(len(target_pool), size=min_sample, replace=False)
        try:
            scale, rot, trans = umeyama_sim3(source_pool[src_idx], target_pool[tgt_idx], allow_scaling=True)
        except Exception:
            continue
        if not np.isfinite(scale) or scale <= 0.0:
            continue
        if scale < 0.01 or scale > 5.0:
            continue
        transformed = apply_sim3(source_pool, scale, rot, trans)
        dists, _ = tree.query(transformed, k=1, workers=-1)
        inlier_ratio = float(np.mean(dists < inlier_threshold_m))
        score = trimmed_mean(dists, trim_fraction)
        candidates.append(
            {
                "scale": float(scale),
                "rotation": rot,
                "translation": trans,
                "score": float(score),
                "inlier_ratio": inlier_ratio,
                "source": "ransac",
            }
        )

    for item in candidates:
        if "score" not in item:
            dists, _ = tree.query(apply_sim3(source_pool, item["scale"], item["rotation"], item["translation"]), k=1, workers=-1)
            item["score"] = trimmed_mean(dists, trim_fraction)
            item["inlier_ratio"] = float(np.mean(dists < inlier_threshold_m))

    candidates.sort(key=lambda item: (item["score"], -item["inlier_ratio"]))
    refined = []
    for item in candidates[: max(1, int(top_k))]:
        refined_item = trimmed_sim3_icp(
            source_pool,
            target_pool,
            float(item["scale"]),
            np.asarray(item["rotation"], dtype=np.float64),
            np.asarray(item["translation"], dtype=np.float64),
            trim_fraction=trim_fraction,
            iterations=12,
        )
        refined_item["source"] = item.get("source", "ransac")
        refined_item["ransac_score_before_icp"] = float(item["score"])
        refined_item["ransac_inlier_ratio_before_icp"] = float(item["inlier_ratio"])
        refined.append(refined_item)
    refined.sort(key=lambda item: item["trimmed_mean_distance"])
    best = refined[0]
    best["method"] = "ransac_sim3_trimmed_icp_initial"
    best["matrix"] = sim3_matrix(best["scale"], best["rotation"], best["translation"])
    best["top_candidates"] = [
        {
            "source": item.get("source"),
            "score": float(item["score"]),
            "inlier_ratio": float(item["inlier_ratio"]),
            "scale": float(item["scale"]),
        }
        for item in candidates[: min(12, len(candidates))]
    ]
    best["refined_top"] = [
        {
            "trimmed_mean_distance": float(item["trimmed_mean_distance"]),
            "source": item.get("source"),
            "scale": float(item["scale"]),
            "ransac_score_before_icp": float(item.get("ransac_score_before_icp", np.nan)),
            "ransac_inlier_ratio_before_icp": float(item.get("ransac_inlier_ratio_before_icp", np.nan)),
        }
        for item in refined[: min(8, len(refined))]
    ]
    return best


def projected_uv(meta: dict[str, Any], points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    u, v, z = project_right_camera_points(meta, points)
    width = int(meta["rgb_width_per_eye"])
    height = int(meta["rgb_height_per_eye"])
    inside = (z > 1e-6) & np.isfinite(u) & np.isfinite(v) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return u, v, z, inside


def edge_projection_score(
    source_points: np.ndarray,
    observed_points: np.ndarray,
    meta: dict[str, Any],
    mask: np.ndarray,
    scale: float,
    rot: np.ndarray,
    trans: np.ndarray,
    boundary_xy: np.ndarray,
    outside_distance: np.ndarray,
    target_bbox: np.ndarray,
    boundary_trim_fraction: float,
    edge_boundary_weight: float,
    edge_outside_weight: float,
    edge_bbox_weight: float,
    edge_depth_weight: float,
    depth_trim_fraction: float,
) -> dict[str, Any] | None:
    points = apply_sim3(source_points, scale, rot, trans)
    u, v, _, inside = projected_uv(meta, points)
    if int(inside.sum()) < 64:
        return None
    width = int(meta["rgb_width_per_eye"])
    height = int(meta["rgb_height_per_eye"])
    uv = np.column_stack([u[inside], v[inside]])
    ui = np.clip(np.round(uv[:, 0]).astype(np.int64), 0, width - 1)
    vi = np.clip(np.round(uv[:, 1]).astype(np.int64), 0, height - 1)
    target_size = np.maximum(target_bbox[[2, 3]] - target_bbox[[0, 1]], 1.0)
    target_diag = float(np.linalg.norm(target_size))
    outside_score = float(np.mean((outside_distance[vi, ui] / target_diag) ** 2))
    tree = cKDTree(uv)
    boundary_dist_px, _ = tree.query(boundary_xy, k=1, workers=-1)
    keep = max(1, int(len(boundary_dist_px) * boundary_trim_fraction))
    boundary_score = float(np.mean(np.partition(boundary_dist_px, keep - 1)[:keep] / target_diag) ** 2)
    projected_bbox, inside_ratio = projected_bbox_quantiles(meta, points[inside], 0.01, 0.99)
    if projected_bbox is None:
        return None
    bbox_metrics = projected_bbox_error(projected_bbox, target_bbox)
    depth_metrics = visible_bidirectional_score(
        source_points,
        observed_points,
        meta,
        mask,
        scale,
        rot,
        trans,
        grid_px=4,
        trim_fraction=depth_trim_fraction,
    )
    depth_score = float(depth_metrics.get("score", float("inf")))
    if not np.isfinite(depth_score):
        return None
    score = float(
        edge_boundary_weight * boundary_score
        + edge_outside_weight * outside_score
        + edge_bbox_weight * bbox_metrics["score"]
        + edge_depth_weight * depth_score
    )
    return {
        "score": score,
        "boundary_score": boundary_score,
        "outside_score": outside_score,
        "bbox_score": float(bbox_metrics["score"]),
        "depth_score_m": depth_score,
        "projected_bbox": projected_bbox,
        "inside_ratio": inside_ratio,
        "visible_count": int(depth_metrics.get("visible_count", 0)),
    }


def edge_refine_alignment(
    source_points: np.ndarray,
    observed_points: np.ndarray,
    meta: dict[str, Any],
    mask: np.ndarray,
    initial: dict[str, Any],
    config: HunyuanRansacAlignmentConfig,
) -> dict[str, Any]:
    scale0 = float(initial["scale"])
    rot = np.asarray(initial["rotation"], dtype=np.float64)
    trans0 = np.asarray(initial["translation"], dtype=np.float64)
    source_center = source_points.mean(axis=0)
    pivot_cam = scale0 * (rot @ source_center) + trans0
    target_bbox = mask_bbox_quantiles(mask, 0.01, 0.99)
    boundary = mask ^ binary_erosion(mask)
    by, bx = np.nonzero(boundary)
    if len(bx) == 0:
        by, bx = np.nonzero(mask)
    boundary_xy = np.column_stack([bx, by]).astype(np.float64)
    if len(boundary_xy) > 14000:
        rng = np.random.default_rng(config.random_seed + 51)
        boundary_xy = boundary_xy[rng.choice(len(boundary_xy), size=14000, replace=False)]
    outside_distance = distance_transform_edt(~mask)
    initial_points = apply_sim3(source_points, scale0, rot, trans0)
    initial_bbox, _ = projected_bbox_quantiles(meta, initial_points, 0.01, 0.99)
    target_center = 0.5 * (target_bbox[[0, 1]] + target_bbox[[2, 3]])
    if initial_bbox is not None:
        initial_center = 0.5 * (initial_bbox[[0, 1]] + initial_bbox[[2, 3]])
        center_delta = target_center - initial_center
    else:
        center_delta = np.zeros(2, dtype=np.float64)

    scale_values = np.linspace(config.edge_scale_min_multiplier, config.edge_scale_max_multiplier, max(3, config.edge_scale_steps))
    shift_axis = np.linspace(-1.0, 1.0, max(3, config.edge_shift_steps))
    du_values = sorted({0.0, float(np.clip(center_delta[0], -config.edge_shift_max_px, config.edge_shift_max_px))} | {
        float(np.clip(center_delta[0] + a * config.edge_shift_max_px, -config.edge_shift_max_px, config.edge_shift_max_px))
        for a in shift_axis
    })
    dv_values = sorted({0.0, float(np.clip(center_delta[1], -config.edge_shift_max_px, config.edge_shift_max_px))} | {
        float(np.clip(center_delta[1] + a * config.edge_shift_max_px, -config.edge_shift_max_px, config.edge_shift_max_px))
        for a in shift_axis
    })

    best = None
    evaluations = []
    for scale_mult in scale_values:
        scale = scale0 * float(scale_mult)
        scaled_trans = pivot_cam - scale * (rot @ source_center)
        for du in du_values:
            for dv in dv_values:
                trans = scaled_trans + translation_for_projected_shift(meta, pivot_cam, du, dv)
                item = edge_projection_score(
                    source_points,
                    observed_points,
                    meta,
                    mask,
                    scale,
                    rot,
                    trans,
                    boundary_xy,
                    outside_distance,
                    target_bbox,
                    boundary_trim_fraction=0.85,
                    edge_boundary_weight=config.edge_boundary_weight,
                    edge_outside_weight=config.edge_outside_weight,
                    edge_bbox_weight=config.edge_bbox_weight,
                    edge_depth_weight=config.edge_depth_weight,
                    depth_trim_fraction=config.icp_trim_fraction,
                )
                if item is None:
                    continue
                item.update({"scale": scale, "translation": trans, "scale_multiplier": float(scale_mult), "shift_px": [float(du), float(dv)]})
                evaluations.append(item)
                if best is None or item["score"] < best["score"]:
                    best = item
    if best is None:
        out = dict(initial)
        out["method"] = "edge_projection_refined_failed"
        out["edge_error"] = "No valid projection candidates."
        return out
    out = dict(initial)
    out.update(
        {
            "method": "ransac_icp_edge_projection_refined",
            "scale": float(best["scale"]),
            "rotation": rot,
            "translation": best["translation"],
            "matrix": sim3_matrix(float(best["scale"]), rot, best["translation"]),
            "score": best["depth_score_m"],
            "model_to_observed_m": None,
            "observed_to_model_m": None,
            "visible_count": int(best["visible_count"]),
            "edge_projection_refine": {
                "combined_score": float(best["score"]),
                "boundary_score": float(best["boundary_score"]),
                "outside_score": float(best["outside_score"]),
                "bbox_score": float(best["bbox_score"]),
                "depth_score_m": float(best["depth_score_m"]),
                "target_bbox": target_bbox,
                "projected_bbox": best["projected_bbox"],
                "inside_ratio": float(best["inside_ratio"]),
                "scale_multiplier": float(best["scale_multiplier"]),
                "shift_px": best["shift_px"],
                "candidate_count": int(len(evaluations)),
                "top_candidates": [
                    {
                        "combined_score": float(item["score"]),
                        "boundary_score": float(item["boundary_score"]),
                        "outside_score": float(item["outside_score"]),
                        "bbox_score": float(item["bbox_score"]),
                        "depth_score_m": float(item["depth_score_m"]),
                        "scale_multiplier": float(item["scale_multiplier"]),
                        "shift_px": item["shift_px"],
                        "inside_ratio": float(item["inside_ratio"]),
                    }
                    for item in sorted(evaluations, key=lambda x: x["score"])[:8]
                ],
            },
        }
    )
    return out


def save_single_mesh_projection_overlay(
    rgb_path: Path,
    output_path: Path,
    meta: dict[str, Any],
    mesh: trimesh.Trimesh,
    mask: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    image = Image.open(rgb_path).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    mask_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    mask_img = Image.fromarray(mask.astype(np.uint8) * 70, mode="L")
    tint = Image.new("RGBA", image.size, (40, 255, 80, 0))
    tint.putalpha(mask_img)
    image = Image.alpha_composite(image, tint)
    draw = ImageDraw.Draw(overlay)
    boundary = mask ^ binary_erosion(mask)
    by, bx = np.nonzero(boundary)
    for x, y in zip(bx, by):
        draw.point((int(x), int(y)), fill=(30, 255, 80, 230))
    points = sample_mesh_points(mesh, min(45000, max(1500, len(mesh.faces))), seed)
    u, v, z = project_right_camera_points(meta, points)
    width = int(meta["rgb_width_per_eye"])
    height = int(meta["rgb_height_per_eye"])
    inside = (z > 1e-6) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    for x, y in zip(u[inside], v[inside]):
        draw.ellipse((x - 1.1, y - 1.1, x + 1.1, y + 1.1), fill=(255, 80, 45, 210))
    out = Image.alpha_composite(image, mask_layer)
    out = Image.alpha_composite(out, overlay)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.convert("RGB").save(output_path)
    return {"path": str(output_path), "projected_points": int(inside.sum()), "mask_boundary_points": int(len(bx))}


def export_view_dir(
    config: HunyuanRansacAlignmentConfig,
    meta: dict[str, Any],
    mesh_cam: trimesh.Trimesh,
    observed_points: np.ndarray,
    observed_colors: np.ndarray,
    alignment: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any] | None:
    if config.view_frame is None:
        return None
    align_row = frame_row(config.export_root, config.align_frame)
    view_row = frame_row(config.export_root, config.view_frame)
    t_view_from_align = camera_to_camera_matrix(meta, align_row, view_row, camera="right")
    view_dir = out_dir / f"view_frame_{frame_name(config.view_frame)}"
    view_dir.mkdir(parents=True, exist_ok=True)
    mesh_view = apply_se3_to_mesh(mesh_cam, t_view_from_align)
    mesh_view.export(view_dir / "part_whole_view_camera.obj")
    observed_view = transform_points(observed_points, t_view_from_align)
    export_point_cloud(view_dir / "observed_mask_pointcloud_view_camera.ply", observed_view, observed_colors)
    write_json(view_dir / "joint_view_camera.json", {"joints": []})
    return {
        "view_dir": str(view_dir),
        "mesh_view": str(view_dir / "part_whole_view_camera.obj"),
        "observed_pointcloud_view": str(view_dir / "observed_mask_pointcloud_view_camera.ply"),
        "camera_transform_align_to_view": t_view_from_align,
    }


def run_hunyuan_ransac_alignment(config: HunyuanRansacAlignmentConfig) -> dict[str, Any]:
    project_root = config.project_root.resolve()
    export_root = config.export_root.resolve()
    out_dir = (config.output_dir or (project_root / "outputs" / "hunyuan3d_ransac_alignment" / config.target_id / f"frame_{frame_name(config.align_frame)}")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = read_json(export_root / "manifest.json")
    mesh_path = resolve_hunyuan_mesh(project_root, config.target_id, config.mesh_path)
    mesh = load_whole_mesh(mesh_path)
    mask_path, mask_id = resolve_selected_mask(project_root, config.target_id)
    mask = np.load(mask_path).astype(bool)
    depth_path = export_root / "depth_meters_npy" / f"{frame_name(config.align_frame)}.meters.npy"
    rgb_path = export_root / "rgb_right_png" / f"{frame_name(config.align_frame)}.png"
    depth_m = np.load(depth_path)

    validation = {}
    points_right, u, v, inside = depth_points_in_right_camera(meta, depth_m, config.convention, config.depth_min_m, config.depth_max_m)
    validation[config.convention] = save_projection_overlay(
        rgb_path,
        out_dir / f"depth_to_rgb_overlay_{config.convention}.png",
        points_right,
        u,
        v,
        inside,
        mask=mask,
    )
    observed_points, observed_u, observed_v, observed_stats = observed_mask_cloud_with_pixels(
        meta,
        depth_m,
        mask,
        config.convention,
        config.depth_min_m,
        config.depth_max_m,
        config.depth_quantile_min,
        config.depth_quantile_max,
    )
    if len(observed_points) > config.observed_samples:
        rng = np.random.default_rng(config.random_seed)
        idx = rng.choice(len(observed_points), size=config.observed_samples, replace=False)
        observed_points = observed_points[idx]
        observed_u = observed_u[idx]
        observed_v = observed_v[idx]
    observed_colors = color_by_depth(observed_points[:, 2])
    export_point_cloud(out_dir / "observed_mask_pointcloud.ply", observed_points, observed_colors)
    save_binary_mask(mask, out_dir / "target_mask")

    canonical_points = sample_mesh_points(mesh, min(config.canonical_samples, max(2000, len(mesh.faces))), config.random_seed)
    export_point_cloud(out_dir / "hunyuan_canonical_sample_pointcloud.ply", canonical_points)

    ransac = ransac_sim3_alignment(
        canonical_points,
        observed_points,
        iterations=config.ransac_iterations,
        sample_size=config.ransac_sample_size,
        inlier_threshold_m=config.ransac_inlier_threshold_m,
        trim_fraction=config.ransac_trim_fraction,
        candidate_pool=config.ransac_candidate_pool,
        top_k=config.ransac_top_k,
        seed=config.random_seed,
    )
    icp = trimmed_sim3_icp(
        canonical_points,
        observed_points,
        float(ransac["scale"]),
        np.asarray(ransac["rotation"], dtype=np.float64),
        np.asarray(ransac["translation"], dtype=np.float64),
        trim_fraction=config.icp_trim_fraction,
        iterations=config.icp_iterations,
    )
    icp["method"] = "trimmed_icp_after_ransac"
    icp["matrix"] = sim3_matrix(icp["scale"], icp["rotation"], icp["translation"])
    depth_score = visible_bidirectional_score(
        canonical_points,
        observed_points,
        meta,
        mask,
        icp["scale"],
        icp["rotation"],
        icp["translation"],
        grid_px=4,
        trim_fraction=config.icp_trim_fraction,
    )
    icp["visible_score_after_icp"] = depth_score
    alignment = icp
    edge_alignment = None
    if config.edge_refine:
        edge_alignment = edge_refine_alignment(canonical_points, observed_points, meta, mask, icp, config)
        alignment = edge_alignment

    mesh_cam = apply_sim3_to_mesh(mesh, float(alignment["scale"]), np.asarray(alignment["rotation"], dtype=np.float64), np.asarray(alignment["translation"], dtype=np.float64))
    mesh_cam_path = out_dir / "hunyuan3d_camera_aligned.obj"
    mesh_cam.export(mesh_cam_path)
    scene = trimesh.Scene()
    colored = mesh_cam.copy()
    colored.visual.vertex_colors = np.tile(np.asarray([240, 90, 55, 230], dtype=np.uint8), (len(colored.vertices), 1))
    scene.add_geometry(colored, node_name="hunyuan3d_whole")
    scene.export(out_dir / "hunyuan3d_camera_aligned.glb")
    overlay = save_single_mesh_projection_overlay(
        rgb_path,
        out_dir / "hunyuan3d_projection_overlay.png",
        meta,
        mesh_cam,
        mask,
        config.random_seed + 99,
    )
    fit_metrics = visible_bidirectional_score(
        canonical_points,
        observed_points,
        meta,
        mask,
        alignment["scale"],
        alignment["rotation"],
        alignment["translation"],
        grid_px=4,
        trim_fraction=config.icp_trim_fraction,
    )
    view_result = export_view_dir(config, meta, mesh_cam, observed_points, observed_colors, alignment, out_dir)
    result = {
        "type": "hunyuan3d_ransac_alignment",
        "target_id": config.target_id,
        "mesh_path": str(mesh_path),
        "mask_id": mask_id,
        "mask_path": str(mask_path),
        "rgb_path": str(rgb_path),
        "depth_path": str(depth_path),
        "output_dir": str(out_dir),
        "alignment": {
            "method": alignment.get("method"),
            "scale": float(alignment["scale"]),
            "rotation": np.asarray(alignment["rotation"], dtype=np.float64),
            "translation": np.asarray(alignment["translation"], dtype=np.float64),
            "matrix_canonical_to_align_camera": sim3_matrix(alignment["scale"], alignment["rotation"], alignment["translation"]),
            "final_visible_score": fit_metrics,
            "ransac_initial": ransac,
            "icp_after_ransac": icp,
            "edge_projection_refine": edge_alignment.get("edge_projection_refine") if edge_alignment else None,
        },
        "observed_stats": observed_stats,
        "depth_rgb_projection_validation": validation,
        "outputs": {
            "aligned_mesh_obj": str(mesh_cam_path),
            "aligned_mesh_glb": str(out_dir / "hunyuan3d_camera_aligned.glb"),
            "observed_pointcloud": str(out_dir / "observed_mask_pointcloud.ply"),
            "canonical_pointcloud": str(out_dir / "hunyuan_canonical_sample_pointcloud.ply"),
            "projection_overlay": overlay,
            "view": view_result,
        },
        "config": to_jsonable(config.__dict__),
    }
    write_json(out_dir / "alignment_result.json", result)
    return result
