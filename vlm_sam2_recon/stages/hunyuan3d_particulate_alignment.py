"""Align Hunyuan3D + Particulate laptop parts by fitting the base first."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion, binary_fill_holes, distance_transform_edt
from scipy.spatial import cKDTree

from .camera_alignment import (
    apply_se3_to_mesh,
    apply_sim3,
    apply_sim3_to_mesh,
    camera_to_camera_matrix,
    color_by_depth,
    export_colored_scene,
    export_point_cloud,
    frame_name,
    frame_row,
    hinge_angle_refine,
    load_particulate_parts,
    mask_bbox_quantiles,
    observed_mask_cloud_with_pixels,
    PART_COLORS,
    plane_from_points,
    project_right_camera_points,
    projected_bbox_error,
    read_json,
    rotation_from_axis_angle,
    sample_mesh_points,
    save_binary_mask,
    save_mesh_projection_overlay,
    save_projection_overlay,
    sim3_matrix,
    subsample_points,
    transform_joint,
    transform_joint_se3,
    transform_points,
    trimmed_sim3_icp,
    visible_bidirectional_score,
    write_json,
)
from .hunyuan3d_ransac_alignment import ransac_sim3_alignment


DEFAULT_EXPORT_ROOT = Path("/code/3DVideo_2026-07-01-19-24-06-667_spatialmp4_export")


@dataclass
class HunyuanParticulateBaseAlignConfig:
    project_root: Path
    export_root: Path = DEFAULT_EXPORT_ROOT
    target_id: str = "target_laptop"
    align_frame: int = 0
    view_frame: int | None = 5
    convention: str = "camera_to_rig"
    particulate_run_dir: Path | None = None
    output_dir: Path | None = None
    base_mask_path: Path | None = None
    whole_mask_path: Path | None = None
    depth_min_m: float = 0.1
    depth_max_m: float = 3.0
    depth_quantile_min: float = 0.03
    depth_quantile_max: float = 0.85
    canonical_samples_per_part: int = 26000
    observed_samples: int = 14000
    candidate_base_labels: list[str] | None = None
    ransac_iterations: int = 900
    ransac_sample_size: int = 4
    ransac_inlier_threshold_m: float = 0.025
    ransac_trim_fraction: float = 0.62
    ransac_candidate_pool: int = 9000
    ransac_top_k: int = 10
    icp_iterations: int = 50
    icp_trim_fraction: float = 0.68
    silhouette_scale_min_multiplier: float = 1.0
    silhouette_scale_max_multiplier: float = 5.5
    silhouette_scale_steps: int = 51
    silhouette_shift_max_px: float = 260.0
    silhouette_shift_steps: int = 7
    silhouette_min_coverage: float = 0.35
    silhouette_depth_weight: float = 1.0
    silhouette_iou_weight: float = 0.65
    silhouette_coverage_weight: float = 0.85
    silhouette_outside_weight: float = 0.9
    silhouette_bbox_weight: float = 0.18
    silhouette_boundary_weight: float = 0.0
    silhouette_inplane_rotation_max_deg: float = 0.0
    silhouette_inplane_rotation_steps: int = 1
    visible_grid_px: int = 4
    random_seed: int = 42
    hinge_refine: bool = True
    hinge_angle_min_deg: float = -120.0
    hinge_angle_max_deg: float = 120.0
    hinge_angle_steps: int = 241
    hinge_trim_fraction: float = 0.70
    hinge_plane_distance_weight: float = 1.0
    hinge_nn_weight: float = 0.15
    hinge_normal_weight_m_per_deg: float = 0.004
    hinge_angle_regularizer_m_per_deg: float = 0.00015


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
        return [to_jsonable(item) for item in value]
    return value


def write_json_file(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(data), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def latest_matching(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def resolve_particulate_run_dir(project_root: Path, target_id: str, explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Particulate run dir not found: {path}")
        return path
    candidates = [
        project_root / "outputs" / "particulate_hunyuan" / f"{target_id}_decimated_50000",
        project_root / "outputs" / "particulate" / f"{target_id}_decimated_50000",
    ]
    for candidate in candidates:
        if candidate.exists() and latest_matching(candidate, "urdf_*/model.urdf"):
            return candidate.resolve()
    raise FileNotFoundError("No Particulate URDF found. Run Hunyuan Particulate first.")


def resolve_masks(config: HunyuanParticulateBaseAlignConfig, project_root: Path) -> tuple[Path, Path]:
    if config.whole_mask_path is not None:
        whole = config.whole_mask_path.expanduser().resolve()
    else:
        manifest = read_json(project_root / "outputs" / "project_manifest.json")
        target = next(item for item in manifest["targets"] if item["object_id"] == config.target_id)
        mask_item = next(item for item in manifest["masks"] if item["mask_id"] == target["selected_mask_id"])
        whole = Path(mask_item["mask_npy"]).resolve()
    if not whole.exists():
        raise FileNotFoundError(f"Whole mask not found: {whole}")

    if config.base_mask_path is not None:
        base = config.base_mask_path.expanduser().resolve()
    else:
        base = (project_root / "outputs" / "sam2_masks" / config.target_id / f"{config.target_id}_frame_{config.align_frame}_base.mask.npy").resolve()
    if not base.exists():
        raise FileNotFoundError(f"Base mask not found: {base}")
    return whole, base


def part_label_to_int(label: str, fallback: int) -> int:
    try:
        return int(label)
    except ValueError:
        return fallback


def part_color(label: str, base_label: str | None = None, screen_label: str | None = None) -> tuple[int, int, int, int]:
    if label == base_label:
        return (220, 50, 65, 255)
    if label == screen_label:
        return (75, 145, 210, 255)
    return PART_COLORS.get(label, (180, 180, 180, 255))


def rasterized_projection_mask(meta: dict[str, Any], points_cam: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray | None, float]:
    height, width = shape
    u, v, z = project_right_camera_points(meta, points_cam)
    inside = (z > 1e-6) & np.isfinite(u) & np.isfinite(v) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if int(inside.sum()) < 32:
        return np.zeros(shape, dtype=bool), None, float(inside.mean())
    uv = np.column_stack([u[inside], v[inside]]).astype(np.float32)
    bbox = np.asarray(
        [
            np.quantile(uv[:, 0], 0.01),
            np.quantile(uv[:, 1], 0.01),
            np.quantile(uv[:, 0], 0.99),
            np.quantile(uv[:, 1], 0.99),
        ],
        dtype=np.float64,
    )
    hull = cv2.convexHull(np.round(uv).astype(np.int32))
    out = np.zeros(shape, dtype=np.uint8)
    if hull.shape[0] >= 3:
        cv2.fillConvexPoly(out, hull, 1)
    else:
        ui = np.clip(np.round(uv[:, 0]).astype(np.int64), 0, width - 1)
        vi = np.clip(np.round(uv[:, 1]).astype(np.int64), 0, height - 1)
        out[vi, ui] = 1
    return out.astype(bool), bbox, float(inside.mean())


def silhouette_coverage_score(
    canonical_points: np.ndarray,
    observed_points: np.ndarray,
    meta: dict[str, Any],
    mask: np.ndarray,
    scale: float,
    rot: np.ndarray,
    trans: np.ndarray,
    boundary_xy: np.ndarray,
    outside_distance: np.ndarray,
    target_bbox: np.ndarray,
    target_area: int,
    config: HunyuanParticulateBaseAlignConfig,
) -> dict[str, Any] | None:
    points_cam = apply_sim3(canonical_points, scale, rot, trans)
    projected_mask, projected_bbox, inside_ratio = rasterized_projection_mask(meta, points_cam, mask.shape)
    projected_area = int(projected_mask.sum())
    if projected_bbox is None or projected_area < 64 or target_area < 64:
        return None

    intersection = int(np.logical_and(projected_mask, mask).sum())
    union = int(np.logical_or(projected_mask, mask).sum())
    coverage = float(intersection / max(target_area, 1))
    precision = float(intersection / max(projected_area, 1))
    iou = float(intersection / max(union, 1))
    outside_ratio = float((projected_area - intersection) / max(projected_area, 1))
    outside_score = float(np.mean((outside_distance[projected_mask] / max(np.sqrt(target_area), 1.0)) ** 2))

    bbox_metrics = projected_bbox_error(projected_bbox, target_bbox)

    boundary_score = 0.0
    if config.silhouette_boundary_weight > 0.0:
        projected_boundary = projected_mask ^ binary_erosion(projected_mask)
        py, px = np.nonzero(projected_boundary)
        if len(px) >= 16 and len(boundary_xy) >= 16:
            proj_xy = np.column_stack([px, py]).astype(np.float64)
            dists, _ = cKDTree(proj_xy).query(boundary_xy, k=1, workers=-1)
            keep = max(1, int(0.85 * len(dists)))
            target_diag = float(np.linalg.norm(np.maximum(target_bbox[[2, 3]] - target_bbox[[0, 1]], 1.0)))
            boundary_score = float(np.mean(np.partition(dists, keep - 1)[:keep] / target_diag) ** 2)
        else:
            boundary_score = 1.0

    depth_metrics: dict[str, Any]
    if config.silhouette_depth_weight > 0.0:
        depth_metrics = visible_bidirectional_score(
            canonical_points,
            observed_points,
            meta,
            mask,
            scale,
            rot,
            trans,
            grid_px=config.visible_grid_px,
            trim_fraction=config.icp_trim_fraction,
        )
        depth_score = float(depth_metrics.get("score", float("inf")))
        if not np.isfinite(depth_score):
            return None
    else:
        depth_metrics = {"visible_count": 0}
        depth_score = 0.0

    # Lower is better. IoU/coverage are converted to penalties; coverage is
    # intentionally strong so tiny projections cannot win again.
    score = float(
        config.silhouette_depth_weight * depth_score
        + config.silhouette_iou_weight * (1.0 - iou)
        + config.silhouette_coverage_weight * (1.0 - coverage)
        + config.silhouette_outside_weight * outside_ratio
        + config.silhouette_bbox_weight * bbox_metrics["score"]
        + config.silhouette_boundary_weight * boundary_score
    )
    if coverage < config.silhouette_min_coverage:
        score += float((config.silhouette_min_coverage - coverage) * 3.0)

    return {
        "score": score,
        "depth_score_m": depth_score,
        "coverage": coverage,
        "precision": precision,
        "iou": iou,
        "outside_ratio": outside_ratio,
        "outside_score": outside_score,
        "bbox_score": float(bbox_metrics["score"]),
        "boundary_score": boundary_score,
        "projected_bbox": projected_bbox,
        "target_bbox": target_bbox,
        "inside_ratio": inside_ratio,
        "projected_area_px": projected_area,
        "target_area_px": target_area,
        "intersection_px": intersection,
        "visible_count": int(depth_metrics.get("visible_count", 0)),
    }


def refine_with_silhouette_coverage(
    canonical_points: np.ndarray,
    observed_points: np.ndarray,
    meta: dict[str, Any],
    mask: np.ndarray,
    initial: dict[str, Any],
    config: HunyuanParticulateBaseAlignConfig,
) -> dict[str, Any]:
    scale0 = float(initial["scale"])
    rot = np.asarray(initial["rotation"], dtype=np.float64)
    trans0 = np.asarray(initial["translation"], dtype=np.float64)
    center = canonical_points.mean(axis=0)
    pivot = scale0 * (rot @ center) + trans0
    _, observed_normal, _ = plane_from_points(observed_points)
    observed_normal = observed_normal / (np.linalg.norm(observed_normal) + 1e-12)
    target_bbox = mask_bbox_quantiles(mask, 0.01, 0.99)
    target_area = int(mask.sum())
    target_center = 0.5 * (target_bbox[[0, 1]] + target_bbox[[2, 3]])
    points0 = apply_sim3(canonical_points, scale0, rot, trans0)
    _, projected_bbox0, _ = rasterized_projection_mask(meta, points0, mask.shape)
    if projected_bbox0 is not None:
        projected_center = 0.5 * (projected_bbox0[[0, 1]] + projected_bbox0[[2, 3]])
        center_delta = target_center - projected_center
    else:
        center_delta = np.zeros(2, dtype=np.float64)

    boundary = mask ^ binary_erosion(mask)
    by, bx = np.nonzero(boundary)
    if len(bx) == 0:
        by, bx = np.nonzero(mask)
    boundary_xy = np.column_stack([bx, by]).astype(np.float64)
    if len(boundary_xy) > 14000:
        rng = np.random.default_rng(config.random_seed + 33)
        boundary_xy = boundary_xy[rng.choice(len(boundary_xy), size=14000, replace=False)]
    outside_distance = distance_transform_edt(~mask)

    scale_values = np.linspace(
        config.silhouette_scale_min_multiplier,
        config.silhouette_scale_max_multiplier,
        max(3, config.silhouette_scale_steps),
    )
    if config.silhouette_inplane_rotation_steps <= 1 or config.silhouette_inplane_rotation_max_deg <= 0:
        angle_values = [0.0]
    else:
        angle_values = np.linspace(
            -config.silhouette_inplane_rotation_max_deg,
            config.silhouette_inplane_rotation_max_deg,
            max(3, config.silhouette_inplane_rotation_steps),
        )
    shift_axis = np.linspace(-1.0, 1.0, max(3, config.silhouette_shift_steps))
    du_values = sorted(
        {
            0.0,
            float(np.clip(center_delta[0], -config.silhouette_shift_max_px, config.silhouette_shift_max_px)),
        }
        | {
            float(np.clip(center_delta[0] + a * config.silhouette_shift_max_px, -config.silhouette_shift_max_px, config.silhouette_shift_max_px))
            for a in shift_axis
        }
    )
    dv_values = sorted(
        {
            0.0,
            float(np.clip(center_delta[1], -config.silhouette_shift_max_px, config.silhouette_shift_max_px)),
        }
        | {
            float(np.clip(center_delta[1] + a * config.silhouette_shift_max_px, -config.silhouette_shift_max_px, config.silhouette_shift_max_px))
            for a in shift_axis
        }
    )

    best = None
    evaluations = []
    for angle_deg in angle_values:
        angle_rad = float(np.deg2rad(angle_deg))
        twist = rotation_from_axis_angle(observed_normal, angle_rad)
        candidate_rot = twist @ rot
        for scale_mult in scale_values:
            scale = scale0 * float(scale_mult)
            scaled_trans = pivot - scale * (candidate_rot @ center)
            for du in du_values:
                for dv in dv_values:
                    shift = projected_shift_to_camera_translation(meta, pivot, du, dv)
                    trans = scaled_trans + shift
                    item = silhouette_coverage_score(
                        canonical_points,
                        observed_points,
                        meta,
                        mask,
                        scale,
                        candidate_rot,
                        trans,
                        boundary_xy,
                        outside_distance,
                        target_bbox,
                        target_area,
                        config,
                    )
                    if item is None:
                        continue
                    item.update(
                        {
                            "scale": scale,
                            "rotation": candidate_rot,
                            "translation": trans,
                            "scale_multiplier": float(scale_mult),
                            "shift_px": [float(du), float(dv)],
                            "inplane_rotation_deg": float(angle_deg),
                        }
                    )
                    evaluations.append(item)
                    if best is None or item["score"] < best["score"]:
                        best = item

    if best is None:
        out = dict(initial)
        out["method"] = "base_silhouette_coverage_failed"
        out["matrix"] = sim3_matrix(out["scale"], out["rotation"], out["translation"])
        out["silhouette_error"] = "No valid silhouette candidates."
        return out

    out = dict(initial)
    out.update(
        {
            "method": "base_ransac_icp_silhouette_coverage",
            "scale": float(best["scale"]),
            "rotation": best["rotation"],
            "translation": best["translation"],
            "matrix": sim3_matrix(float(best["scale"]), best["rotation"], best["translation"]),
            "score": float(best["score"]),
            "silhouette_coverage": {
                key: best[key]
                for key in (
                    "score",
                    "depth_score_m",
                    "coverage",
                    "precision",
                    "iou",
                    "outside_ratio",
                    "outside_score",
                    "bbox_score",
                    "boundary_score",
                    "projected_bbox",
                    "target_bbox",
                    "inside_ratio",
                    "projected_area_px",
                    "target_area_px",
                    "intersection_px",
                    "visible_count",
                    "scale_multiplier",
                    "shift_px",
                    "inplane_rotation_deg",
                )
            },
            "silhouette_top_candidates": [
                {
                    key: item[key]
                    for key in (
                        "score",
                        "depth_score_m",
                        "coverage",
                        "precision",
                        "iou",
                        "outside_ratio",
                        "bbox_score",
                        "boundary_score",
                        "scale_multiplier",
                        "shift_px",
                        "inplane_rotation_deg",
                    )
                }
                for item in sorted(evaluations, key=lambda x: x["score"])[:10]
            ],
            "silhouette_candidate_count": int(len(evaluations)),
        }
    )
    return out


def projected_shift_to_camera_translation(meta: dict[str, Any], pivot_cam: np.ndarray, du_px: float, dv_px: float) -> np.ndarray:
    kr = meta["rgb_intrinsics_right"]
    z = float(max(pivot_cam[2], 1e-6))
    return np.asarray([du_px * z / kr["fx"], dv_px * z / kr["fy"], 0.0], dtype=np.float64)


def estimate_base_candidate(
    base_part: dict[str, Any],
    observed_base_points: np.ndarray,
    meta: dict[str, Any],
    base_mask: np.ndarray,
    config: HunyuanParticulateBaseAlignConfig,
    seed: int,
) -> dict[str, Any]:
    base_points = sample_mesh_points(
        base_part["mesh"],
        min(config.canonical_samples_per_part, max(4000, len(base_part["mesh"].faces))),
        seed,
    )
    source = subsample_points(base_points, min(len(base_points), config.ransac_candidate_pool), seed + 1)
    target = subsample_points(observed_base_points, min(len(observed_base_points), config.ransac_candidate_pool), seed + 2)
    ransac = ransac_sim3_alignment(
        source,
        target,
        iterations=config.ransac_iterations,
        sample_size=config.ransac_sample_size,
        inlier_threshold_m=config.ransac_inlier_threshold_m,
        trim_fraction=config.ransac_trim_fraction,
        candidate_pool=config.ransac_candidate_pool,
        top_k=config.ransac_top_k,
        seed=seed + 3,
    )
    icp = trimmed_sim3_icp(
        source,
        target,
        float(ransac["scale"]),
        np.asarray(ransac["rotation"], dtype=np.float64),
        np.asarray(ransac["translation"], dtype=np.float64),
        trim_fraction=config.icp_trim_fraction,
        iterations=config.icp_iterations,
    )
    icp["method"] = "base_trimmed_icp_after_ransac"
    icp["matrix"] = sim3_matrix(icp["scale"], icp["rotation"], icp["translation"])
    icp["visible_score_after_icp"] = visible_bidirectional_score(
        base_points,
        observed_base_points,
        meta,
        base_mask,
        icp["scale"],
        icp["rotation"],
        icp["translation"],
        grid_px=config.visible_grid_px,
        trim_fraction=config.icp_trim_fraction,
    )
    refined = refine_with_silhouette_coverage(base_points, observed_base_points, meta, base_mask, icp, config)
    refined["base_part_label"] = base_part["label"]
    refined["base_canonical_sample_count"] = int(len(base_points))
    refined["base_ransac_initial"] = ransac
    refined["base_icp"] = icp
    return refined


def choose_screen_label(parts: list[dict[str, Any]], base_label: str) -> str | None:
    others = [part for part in parts if part["label"] != base_label]
    if not others:
        return None
    # Screens are usually the non-base part with largest non-thin extent.
    others.sort(key=lambda part: float(np.prod(part["mesh"].extents)), reverse=True)
    return others[0]["label"]


def save_base_projection_overlay(
    rgb_path: Path,
    output_path: Path,
    meta: dict[str, Any],
    base_mesh: trimesh.Trimesh,
    base_mask: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    image = Image.open(rgb_path).convert("RGBA")
    mask_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    mask_img = Image.fromarray(base_mask.astype(np.uint8) * 78, mode="L")
    tint = Image.new("RGBA", image.size, (40, 255, 80, 0))
    tint.putalpha(mask_img)
    image = Image.alpha_composite(image, tint)

    points = sample_mesh_points(base_mesh, min(45000, max(1200, len(base_mesh.faces))), seed)
    projected_mask, bbox, inside_ratio = rasterized_projection_mask(meta, points, base_mask.shape)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    boundary = base_mask ^ binary_erosion(base_mask)
    by, bx = np.nonzero(boundary)
    for x, y in zip(bx, by):
        draw.point((int(x), int(y)), fill=(40, 255, 80, 230))
    projected_boundary = projected_mask ^ binary_erosion(projected_mask)
    py, px = np.nonzero(projected_boundary)
    for x, y in zip(px, py):
        draw.point((int(x), int(y)), fill=(255, 60, 50, 230))
    projected_layer = Image.fromarray((projected_mask.astype(np.uint8) * 70), mode="L")
    projected_tint = Image.new("RGBA", image.size, (255, 60, 50, 0))
    projected_tint.putalpha(projected_layer)
    out = Image.alpha_composite(image, projected_tint)
    out = Image.alpha_composite(out, overlay)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.convert("RGB").save(output_path)
    return {
        "path": str(output_path),
        "projected_bbox": bbox,
        "inside_ratio": inside_ratio,
        "projected_area_px": int(projected_mask.sum()),
        "mask_area_px": int(base_mask.sum()),
    }


def run_hunyuan_particulate_base_alignment(config: HunyuanParticulateBaseAlignConfig) -> dict[str, Any]:
    project_root = config.project_root.resolve()
    export_root = config.export_root.resolve()
    run_dir = resolve_particulate_run_dir(project_root, config.target_id, config.particulate_run_dir)
    urdf_path = latest_matching(run_dir, "urdf_*/model.urdf")
    if urdf_path is None:
        raise FileNotFoundError(f"Missing Particulate URDF under {run_dir}")
    urdf_path = urdf_path.resolve()

    out_dir = (
        config.output_dir
        or (project_root / "outputs" / "hunyuan3d_particulate_base_alignment" / config.target_id / f"frame_{frame_name(config.align_frame)}")
    ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = read_json(export_root / "manifest.json")
    rgb_path = export_root / "rgb_right_png" / f"{frame_name(config.align_frame)}.png"
    depth_path = export_root / "depth_meters_npy" / f"{frame_name(config.align_frame)}.meters.npy"
    depth_m = np.load(depth_path)
    whole_mask_path, base_mask_path = resolve_masks(config, project_root)
    whole_mask = np.load(whole_mask_path).astype(bool)
    base_mask = np.load(base_mask_path).astype(bool) & whole_mask
    base_footprint = binary_fill_holes(base_mask).astype(bool) & whole_mask
    screen_mask = whole_mask & ~base_footprint
    if int(base_mask.sum()) < 64:
        raise ValueError(f"Base mask has too few pixels: {int(base_mask.sum())}")

    points_right, u, v, inside = depth_points_for_validation(meta, depth_m, config)
    validation = {
        config.convention: save_projection_overlay(
            rgb_path,
            out_dir / f"depth_to_rgb_overlay_{config.convention}.png",
            points_right,
            u,
            v,
            inside,
            mask=base_mask,
        )
    }

    observed_base_points, _, _, observed_base_stats = observed_mask_cloud_with_pixels(
        meta,
        depth_m,
        base_mask,
        config.convention,
        config.depth_min_m,
        config.depth_max_m,
        config.depth_quantile_min,
        config.depth_quantile_max,
    )
    if len(observed_base_points) > config.observed_samples:
        rng = np.random.default_rng(config.random_seed + 11)
        idx = rng.choice(len(observed_base_points), size=config.observed_samples, replace=False)
        observed_base_points = observed_base_points[idx]
    base_colors = np.tile(np.asarray([220, 50, 65], dtype=np.uint8), (len(observed_base_points), 1))
    export_point_cloud(out_dir / "observed_base_pointcloud.ply", observed_base_points, base_colors)
    observed_screen_points = None
    observed_screen_v = None
    observed_screen_stats = None
    screen_colors = None
    if int(screen_mask.sum()) >= 64:
        observed_screen_points, _, observed_screen_v, observed_screen_stats = observed_mask_cloud_with_pixels(
            meta,
            depth_m,
            screen_mask,
            config.convention,
            config.depth_min_m,
            config.depth_max_m,
            config.depth_quantile_min,
            config.depth_quantile_max,
        )
        if len(observed_screen_points) > config.observed_samples:
            rng = np.random.default_rng(config.random_seed + 12)
            idx = rng.choice(len(observed_screen_points), size=config.observed_samples, replace=False)
            observed_screen_points = observed_screen_points[idx]
            observed_screen_v = observed_screen_v[idx]
        screen_colors = np.tile(np.asarray([75, 145, 210], dtype=np.uint8), (len(observed_screen_points), 1))
        export_point_cloud(out_dir / "observed_screen_pointcloud.ply", observed_screen_points, screen_colors)
    save_binary_mask(base_mask, out_dir / "part_masks" / f"{config.target_id}_frame_{config.align_frame}_base")
    save_binary_mask(screen_mask, out_dir / "part_masks" / f"{config.target_id}_frame_{config.align_frame}_screen")

    parts, joints = load_particulate_parts(urdf_path)
    if len(parts) < 2:
        raise ValueError(f"Expected at least 2 Particulate parts, got {len(parts)} from {urdf_path}")
    export_colored_scene(out_dir / "canonical_parts.glb", parts)

    candidate_labels = config.candidate_base_labels or [part["label"] for part in parts]
    part_by_label = {part["label"]: part for part in parts}
    candidates = []
    for idx, label in enumerate(candidate_labels):
        if label not in part_by_label:
            continue
        candidates.append(
            estimate_base_candidate(
                part_by_label[label],
                observed_base_points,
                meta,
                base_mask,
                config,
                seed=config.random_seed + 1000 + idx * 37 + part_label_to_int(label, idx),
            )
        )
    if not candidates:
        raise ValueError(f"No valid base label candidates from {candidate_labels}")
    candidates.sort(key=lambda item: item.get("score", float("inf")))
    alignment = candidates[0]
    base_label = alignment["base_part_label"]
    screen_label = choose_screen_label(parts, base_label)

    scale = float(alignment["scale"])
    rot = np.asarray(alignment["rotation"], dtype=np.float64)
    trans = np.asarray(alignment["translation"], dtype=np.float64)

    aligned_parts = []
    for part in parts:
        mesh_cam = apply_sim3_to_mesh(part["mesh"], scale, rot, trans)
        color = part_color(part["label"], base_label=base_label, screen_label=screen_label)
        mesh_cam.visual.vertex_colors = np.tile(np.asarray(color, dtype=np.uint8), (len(mesh_cam.vertices), 1))
        out_path = out_dir / f"part_{part['label']}_camera.obj"
        mesh_cam.export(out_path)
        aligned_parts.append({**part, "mesh": mesh_cam, "camera_path": str(out_path), "semantic_role": "base" if part["label"] == base_label else ("screen" if part["label"] == screen_label else "other")})

    camera_joints = [transform_joint(joint, scale, rot, trans) for joint in joints if joint.get("type") != "fixed"]
    hinge_refinement = None
    if (
        config.hinge_refine
        and screen_label is not None
        and observed_screen_points is not None
        and len(observed_screen_points) >= 64
        and camera_joints
    ):
        aligned_parts, hinge_refinement = hinge_angle_refine(
            aligned_parts,
            camera_joints,
            observed_screen_points,
            observed_screen_v,
            screen_label=screen_label,
            base_label=base_label,
            angle_min_deg=config.hinge_angle_min_deg,
            angle_max_deg=config.hinge_angle_max_deg,
            angle_steps=config.hinge_angle_steps,
            trim_fraction=config.hinge_trim_fraction,
            plane_distance_weight=config.hinge_plane_distance_weight,
            nn_weight=config.hinge_nn_weight,
            normal_weight_m_per_deg=config.hinge_normal_weight_m_per_deg,
            angle_regularizer_m_per_deg=config.hinge_angle_regularizer_m_per_deg,
            seed=config.random_seed + 91,
            observed_screen_points=observed_screen_points,
            observed_split_metadata=observed_screen_stats,
        )
        for part in aligned_parts:
            mesh = part["mesh"]
            color = part_color(part["label"], base_label=base_label, screen_label=screen_label)
            mesh.visual.vertex_colors = np.tile(np.asarray(color, dtype=np.uint8), (len(mesh.vertices), 1))
            out_path = out_dir / f"part_{part['label']}_camera.obj"
            mesh.export(out_path)
            part["camera_path"] = str(out_path)

    export_colored_scene(out_dir / "laptop_camera_aligned.glb", aligned_parts)
    write_json(out_dir / "joint_camera.json", {"joints": camera_joints})

    overlay = save_mesh_projection_overlay(rgb_path, out_dir / "aligned_parts_projection_overlay.png", meta, aligned_parts, seed=config.random_seed + 70)
    base_overlay = save_base_projection_overlay(
        rgb_path,
        out_dir / "base_silhouette_coverage_overlay.png",
        meta,
        part_by_camera_label(aligned_parts, base_label)["mesh"],
        base_mask,
        seed=config.random_seed + 71,
    )

    view_result = export_view_outputs(config, meta, aligned_parts, camera_joints, observed_base_points, base_colors, out_dir)

    result = {
        "type": "hunyuan3d_particulate_base_alignment",
        "target_id": config.target_id,
        "particulate_run_dir": str(run_dir),
        "canonical_urdf": str(urdf_path),
        "canonical_part_labels": [part["label"] for part in parts],
        "base_part_label": base_label,
        "screen_part_label": screen_label,
        "rgb_path": str(rgb_path),
        "depth_path": str(depth_path),
        "whole_mask_path": str(whole_mask_path),
        "base_mask_path": str(base_mask_path),
        "observed_base_cloud": observed_base_stats,
        "observed_screen_cloud": observed_screen_stats,
        "depth_rgb_projection_validation": validation,
        "alignment": {
            "method": alignment.get("method"),
            "scale": scale,
            "rotation": rot,
            "translation": trans,
            "matrix_canonical_to_align_camera": sim3_matrix(scale, rot, trans),
            "score": alignment.get("score"),
            "silhouette_coverage": alignment.get("silhouette_coverage"),
            "silhouette_top_candidates": alignment.get("silhouette_top_candidates"),
            "base_ransac_initial": alignment.get("base_ransac_initial"),
            "base_icp": alignment.get("base_icp"),
            "screen_hinge_refine": hinge_refinement,
            "candidate_base_summaries": [
                {
                    "base_part_label": item.get("base_part_label"),
                    "score": item.get("score"),
                    "scale": item.get("scale"),
                    "silhouette_coverage": item.get("silhouette_coverage"),
                    "icp_trimmed_mean_distance": item.get("base_icp", {}).get("trimmed_mean_distance"),
                }
                for item in candidates
            ],
        },
        "joints_camera": camera_joints,
        "outputs": {
            "result_dir": str(out_dir),
            "aligned_glb": str(out_dir / "laptop_camera_aligned.glb"),
            "projection_overlay": overlay,
            "base_silhouette_overlay": base_overlay,
            "observed_base_pointcloud": str(out_dir / "observed_base_pointcloud.ply"),
            "observed_screen_pointcloud": str(out_dir / "observed_screen_pointcloud.ply") if observed_screen_points is not None else None,
            "joint_camera": str(out_dir / "joint_camera.json"),
            "view": view_result,
        },
        "config": to_jsonable(config.__dict__),
    }
    write_json_file(out_dir / "alignment_result.json", result)
    return result


def depth_points_for_validation(
    meta: dict[str, Any],
    depth_m: np.ndarray,
    config: HunyuanParticulateBaseAlignConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from .camera_alignment import depth_points_in_right_camera

    return depth_points_in_right_camera(meta, depth_m, config.convention, config.depth_min_m, config.depth_max_m)


def part_by_camera_label(parts: list[dict[str, Any]], label: str) -> dict[str, Any]:
    for part in parts:
        if part["label"] == label:
            return part
    raise KeyError(f"Missing part label {label}")


def export_view_outputs(
    config: HunyuanParticulateBaseAlignConfig,
    meta: dict[str, Any],
    aligned_parts: list[dict[str, Any]],
    camera_joints: list[dict[str, Any]],
    observed_base_points: np.ndarray,
    observed_base_colors: np.ndarray,
    out_dir: Path,
) -> dict[str, Any] | None:
    if config.view_frame is None:
        return None
    align_row = frame_row(config.export_root, config.align_frame)
    view_row = frame_row(config.export_root, config.view_frame)
    t_view_align = camera_to_camera_matrix(meta, align_row, view_row, camera="right")
    view_dir = out_dir / f"view_frame_{frame_name(config.view_frame)}"
    view_dir.mkdir(parents=True, exist_ok=True)
    view_parts = []
    for part in aligned_parts:
        mesh_view = apply_se3_to_mesh(part["mesh"], t_view_align)
        mesh_view.export(view_dir / f"part_{part['label']}_view_camera.obj")
        view_parts.append({**part, "mesh": mesh_view})
    export_colored_scene(view_dir / "laptop_view_camera_aligned.glb", view_parts)
    observed_base_view = transform_points(observed_base_points, t_view_align)
    export_point_cloud(view_dir / "observed_base_pointcloud_view_camera.ply", observed_base_view, observed_base_colors)
    # Keep a copy under the generic name so existing viser scripts can show it.
    export_point_cloud(view_dir / "observed_mask_pointcloud_view_camera.ply", observed_base_view, observed_base_colors)
    view_joints = [transform_joint_se3(joint, t_view_align) for joint in camera_joints]
    write_json(view_dir / "joint_view_camera.json", {"joints": view_joints})
    return {
        "view_frame": int(config.view_frame),
        "view_dir": str(view_dir),
        "camera_transform_align_to_view": t_view_align,
        "joint_view_json": str(view_dir / "joint_view_camera.json"),
        "observed_base_pointcloud_view": str(view_dir / "observed_base_pointcloud_view_camera.ply"),
    }
