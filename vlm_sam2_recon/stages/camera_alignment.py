"""Align reconstructed object meshes to RGB camera coordinates using depth."""

from __future__ import annotations

import csv
import itertools
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion, binary_fill_holes, distance_transform_edt
from scipy.spatial import cKDTree


PART_COLORS = {
    "14": (220, 50, 65, 255),
    "15": (75, 145, 210, 255),
}


@dataclass
class AlignmentConfig:
    project_root: Path
    export_root: Path
    target_id: str = "target_laptop"
    align_frame: int = 0
    view_frame: int | None = None
    convention: str = "camera_to_rig"
    output_root: Path | None = None
    particulate_run_path: Path | None = None
    base_mask_path: Path | None = None
    depth_min_m: float = 0.1
    depth_max_m: float = 3.0
    depth_quantile_min: float = 0.03
    depth_quantile_max: float = 0.85
    canonical_samples: int = 25000
    observed_samples: int = 12000
    icp_trim_fraction: float = 0.65
    icp_iterations: int = 40
    visible_refine: bool = True
    visible_grid_px: int = 4
    visible_trim_fraction: float = 0.75
    visible_iterations: int = 25
    constrained_refine: bool = True
    constrained_iterations: int = 20
    constrained_trim_fraction: float = 0.75
    constrained_scale_min_multiplier: float = 0.95
    constrained_scale_max_multiplier: float = 1.05
    constrained_rotation_max_deg: float = 5.0
    silhouette_refine: bool = True
    silhouette_quantile_min: float = 0.01
    silhouette_quantile_max: float = 0.99
    silhouette_scale_min_multiplier: float = 0.98
    silhouette_scale_max_multiplier: float = 1.15
    silhouette_scale_steps: int = 211
    silhouette_boundary_trim_fraction: float = 0.85
    silhouette_outside_weight: float = 12.0
    silhouette_boundary_weight: float = 0.9
    silhouette_bbox_weight: float = 0.10
    hinge_refine: bool = True
    hinge_screen_part_label: str = "15"
    hinge_base_part_label: str = "14"
    hinge_angle_min_deg: float = -45.0
    hinge_angle_max_deg: float = 45.0
    hinge_angle_steps: int = 181
    hinge_trim_fraction: float = 0.70
    hinge_plane_distance_weight: float = 1.0
    hinge_nn_weight: float = 0.15
    hinge_normal_weight_m_per_deg: float = 0.004
    hinge_angle_regularizer_m_per_deg: float = 0.00015
    final_alignment_mode: str = "base_first"
    base_first_base_part_label: str = "14"
    base_first_screen_part_label: str = "15"
    screen_first_screen_part_label: str = "15"
    screen_first_base_part_label: str = "14"
    screen_first_axis_twist: bool = True
    screen_first_axis_twist_max_deg: float = 60.0
    screen_projection_refine: bool = True
    screen_projection_scale_min_multiplier: float = 0.94
    screen_projection_scale_max_multiplier: float = 1.08
    screen_projection_shift_max_px: float = 48.0
    screen_projection_depth_weight: float = 0.25
    base_visible_surface_constrain: bool = True
    base_visible_surface_grid_px: int = 4
    base_visible_surface_observed_to_model_weight: float = 0.35
    base_visible_surface_plane_offset_weight: float = 0.15
    base_visible_surface_normal_weight_m_per_deg: float = 0.010
    base_visible_surface_snap_offset: bool = True
    pca_direct_candidates: int = 8
    pca_direct_screen_part_label: str = "15"
    pca_direct_base_part_label: str = "14"
    pca_direct_require_semantic_order: bool = True
    part_aware: bool = False
    screen_part_label: str = "14"
    base_part_label: str = "15"
    part_aware_candidates: int = 16
    part_aware_trim_fraction: float = 0.60
    part_aware_iterations: int = 35
    random_seed: int = 42


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(data), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def save_binary_mask(mask: np.ndarray, output_prefix: Path) -> dict[str, Any]:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    mask = mask.astype(bool)
    png_path = output_prefix.with_suffix(".mask.png")
    npy_path = output_prefix.with_suffix(".mask.npy")
    Image.fromarray(mask.astype(np.uint8) * 255).save(png_path)
    np.save(npy_path, mask)
    return {
        "mask_png": str(png_path),
        "mask_npy": str(npy_path),
        "mask_area_pixels": int(mask.sum()),
    }


def fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    return binary_fill_holes(mask.astype(bool)).astype(bool)


def quantile_crop_mask(mask: np.ndarray, q_min: float, q_max: float, pad_px: int = 6) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return mask.astype(bool)
    height, width = mask.shape
    x0 = max(0, int(np.floor(np.quantile(xs, q_min))) - pad_px)
    y0 = max(0, int(np.floor(np.quantile(ys, q_min))) - pad_px)
    x1 = min(width - 1, int(np.ceil(np.quantile(xs, q_max))) + pad_px)
    y1 = min(height - 1, int(np.ceil(np.quantile(ys, q_max))) + pad_px)
    cropped = np.zeros_like(mask, dtype=bool)
    cropped[y0 : y1 + 1, x0 : x1 + 1] = mask[y0 : y1 + 1, x0 : x1 + 1]
    return cropped


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


def frame_name(frame_index: int) -> str:
    return f"{frame_index:06d}"


def resolve_base_mask_path(config: AlignmentConfig, project_root: Path) -> Path:
    if config.base_mask_path is not None:
        path = config.base_mask_path.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Base mask not found: {path}")
        return path

    mask_dir = project_root / "outputs" / "sam2_masks" / config.target_id
    stem = f"{config.target_id}_frame_{config.align_frame}"
    candidates = [
        mask_dir / f"{stem}_base.mask.npy",
        mask_dir / f"{stem}_interactive.mask.npy",
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(
        "No base mask found. Expected one of: "
        + ", ".join(str(path) for path in candidates)
    )


def load_frames_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def frame_row(export_root: Path, frame_index: int) -> dict[str, str]:
    rows = load_frames_csv(export_root / "frames.csv")
    for row in rows:
        if int(row["index"]) == frame_index:
            return row
    raise KeyError(f"Frame {frame_index} not found in {export_root / 'frames.csv'}")


def qvec_to_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    q = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    q /= np.linalg.norm(q) + 1e-12
    w, x, y, z = q
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def pose_matrix_from_row(row: dict[str, str], prefix: str = "rgb_pose") -> np.ndarray:
    t = np.asarray([float(row[f"{prefix}_x"]), float(row[f"{prefix}_y"]), float(row[f"{prefix}_z"])])
    r = qvec_to_matrix(
        float(row[f"{prefix}_qw"]),
        float(row[f"{prefix}_qx"]),
        float(row[f"{prefix}_qy"]),
        float(row[f"{prefix}_qz"]),
    )
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = r
    out[:3, 3] = t
    return out


def rgb_pose_axis_correction(meta: dict[str, Any]) -> np.ndarray:
    """Map exported RGB image axes to the sensor axes used by the head pose.

    The SpatialMP4 RGB images are rotated clockwise by 90 degrees during export,
    while the camera extrinsics retain the original sensor-axis convention.  The
    correction is intentionally used only for temporal pose composition; the
    independently validated depth-to-RGB calibration continues to use the raw
    camera extrinsics.
    """
    angle_deg = float(meta.get("rgb_pose_image_rotation_deg", -90.0))
    angle_rad = np.deg2rad(angle_deg)
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    correction = np.eye(4, dtype=np.float64)
    correction[:3, :3] = np.asarray(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return correction


def rgb_camera_world_matrix(
    meta: dict[str, Any],
    row: dict[str, str],
    camera: str = "right",
    pose_prefix: str = "rgb_pose",
) -> np.ndarray:
    # Exported camera extrinsics are T_head_camera in the original sensor axes.
    # Post-multiplication preserves the camera center while rotating the exported
    # image axes into the pose convention.
    t_world_head = pose_matrix_from_row(row, pose_prefix)
    t_head_camera = np.asarray(meta[f"rgb_extrinsics_{camera}"], dtype=np.float64)
    t_head_exported_camera = t_head_camera @ rgb_pose_axis_correction(meta)
    return t_world_head @ t_head_exported_camera


def camera_to_camera_matrix(
    meta: dict[str, Any],
    align_row: dict[str, str],
    view_row: dict[str, str],
    camera: str = "right",
    align_pose_prefix: str = "rgb_pose",
    view_pose_prefix: str = "rgb_pose",
) -> np.ndarray:
    t_world_align = rgb_camera_world_matrix(meta, align_row, camera, align_pose_prefix)
    t_world_view = rgb_camera_world_matrix(meta, view_row, camera, view_pose_prefix)
    return np.linalg.inv(t_world_view) @ t_world_align


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def transform_vectors(vectors: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return vectors @ transform[:3, :3].T


def depth_to_rgb_transform(meta: dict[str, Any], convention: str) -> np.ndarray:
    t_depth = np.asarray(meta["depth_extrinsics"], dtype=np.float64)
    t_right = np.asarray(meta["rgb_extrinsics_right"], dtype=np.float64)
    if convention == "camera_to_rig":
        return np.linalg.inv(t_right) @ t_depth
    if convention == "rig_to_camera":
        return t_right @ np.linalg.inv(t_depth)
    if convention == "direct_same_camera":
        return np.eye(4, dtype=np.float64)
    raise ValueError(f"Unknown depth/RGB convention: {convention}")


def depth_points_in_right_camera(
    meta: dict[str, Any],
    depth_m: np.ndarray,
    convention: str,
    depth_min_m: float,
    depth_max_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    kd = meta["depth_intrinsics"]
    kr = meta["rgb_intrinsics_right"]
    width = int(meta["rgb_width_per_eye"])
    height = int(meta["rgb_height_per_eye"])

    yy, xx = np.indices(depth_m.shape)
    z = depth_m.reshape(-1).astype(np.float64)
    x = xx.reshape(-1).astype(np.float64)
    y = yy.reshape(-1).astype(np.float64)
    valid = np.isfinite(z) & (z > depth_min_m) & (z < depth_max_m)
    z = z[valid]
    x = x[valid]
    y = y[valid]

    points_depth = np.stack(
        [
            (x - kd["cx"]) / kd["fx"] * z,
            (y - kd["cy"]) / kd["fy"] * z,
            z,
            np.ones_like(z),
        ],
        axis=1,
    )
    t_right_depth = depth_to_rgb_transform(meta, convention)
    points_right = (t_right_depth @ points_depth.T).T[:, :3]
    u = kr["fx"] * points_right[:, 0] / points_right[:, 2] + kr["cx"]
    v = kr["fy"] * points_right[:, 1] / points_right[:, 2] + kr["cy"]
    inside = (points_right[:, 2] > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return points_right, u, v, inside


def color_by_depth(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z)
    lo, hi = np.quantile(z, [0.02, 0.98]) if z.size else (0.0, 1.0)
    denom = max(float(hi - lo), 1e-6)
    t = np.clip((z - lo) / denom, 0.0, 1.0)
    colors = np.stack([255 * (1.0 - t), 80 + 100 * t, 255 * t], axis=1)
    return colors.astype(np.uint8)


def save_projection_overlay(
    rgb_path: Path,
    output_path: Path,
    points_right: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    inside: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, Any]:
    image = Image.open(rgb_path).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    idx = np.flatnonzero(inside)
    if idx.size > 45000:
        rng = np.random.default_rng(7)
        idx = rng.choice(idx, size=45000, replace=False)
    colors = color_by_depth(points_right[idx, 2])
    for point_idx, color in zip(idx, colors):
        x = int(round(u[point_idx]))
        y = int(round(v[point_idx]))
        draw.point((x, y), fill=(int(color[0]), int(color[1]), int(color[2]), 180))

    if mask is not None:
        mask_overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        mask_img = Image.fromarray((mask.astype(np.uint8) * 80), mode="L")
        mask_overlay.putalpha(mask_img)
        mask_tint = Image.new("RGBA", image.size, (40, 255, 80, 0))
        mask_tint.putalpha(mask_img)
        image = Image.alpha_composite(image, mask_tint)

    out = Image.alpha_composite(image, overlay)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.convert("RGB").save(output_path)
    return {"path": str(output_path), "projected_points": int(idx.size)}


def save_mesh_projection_overlay(
    rgb_path: Path,
    output_path: Path,
    meta: dict[str, Any],
    parts: list[dict[str, Any]],
    seed: int,
    samples_per_part: int = 9000,
) -> dict[str, Any]:
    image = Image.open(rgb_path).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    kr = meta["rgb_intrinsics_right"]
    width = int(meta["rgb_width_per_eye"])
    height = int(meta["rgb_height_per_eye"])
    total = 0
    for idx, part in enumerate(parts):
        points = sample_mesh_points(part["mesh"], min(samples_per_part, max(100, len(part["mesh"].faces))), seed + idx)
        z = points[:, 2]
        valid = z > 1e-6
        u = kr["fx"] * points[:, 0] / z + kr["cx"]
        v = kr["fy"] * points[:, 1] / z + kr["cy"]
        inside = valid & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        color = PART_COLORS.get(part["label"], (255, 255, 255, 255))
        for x, y in zip(u[inside], v[inside]):
            draw.ellipse((x - 1.0, y - 1.0, x + 1.0, y + 1.0), fill=color)
        total += int(inside.sum())
    out = Image.alpha_composite(image, overlay)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.convert("RGB").save(output_path)
    return {"path": str(output_path), "projected_mesh_points": total}


def save_part_projection_diagnostic(
    rgb_path: Path,
    output_path: Path,
    meta: dict[str, Any],
    mesh: trimesh.Trimesh,
    mask: np.ndarray,
    color: tuple[int, int, int, int],
    seed: int,
    observed_points: np.ndarray | None = None,
    samples: int = 22000,
) -> dict[str, Any]:
    image = Image.open(rgb_path).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    height, width = mask.shape

    mask_boundary = mask ^ binary_erosion(mask)
    boundary_y, boundary_x = np.nonzero(mask_boundary)
    for x, y in zip(boundary_x, boundary_y):
        draw.point((int(x), int(y)), fill=(40, 255, 80, 230))

    points = sample_mesh_points(mesh, min(samples, max(1000, len(mesh.faces))), seed)
    plane_threshold = None
    if observed_points is not None and len(observed_points) >= 64:
        observed_center, observed_normal, _ = plane_from_points(observed_points)
        plane_distances = np.abs((points - observed_center) @ observed_normal)
        plane_threshold = float(np.clip(np.quantile(plane_distances, 0.60), 0.003, 0.006))
        filtered_points = points[plane_distances <= plane_threshold]
        if len(filtered_points) >= 512:
            points = filtered_points
    u, v, z = project_right_camera_points(meta, points)
    inside = (z > 1e-6) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    for x, y in zip(u[inside], v[inside]):
        draw.ellipse((x - 1.0, y - 1.0, x + 1.0, y + 1.0), fill=color)

    mask_bbox = mask_bbox_quantiles(mask, 0.01, 0.99)
    mesh_bbox, inside_ratio = projected_bbox_quantiles(meta, points, 0.01, 0.99)
    draw.rectangle(tuple(mask_bbox.tolist()), outline=(40, 255, 80, 255), width=3)
    if mesh_bbox is not None:
        draw.rectangle(tuple(mesh_bbox.tolist()), outline=color, width=3)

    out = Image.alpha_composite(image, overlay)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.convert("RGB").save(output_path)
    return {
        "path": str(output_path),
        "projected_mesh_points": int(inside.sum()),
        "mask_boundary_points": int(len(boundary_x)),
        "mask_bbox": mask_bbox,
        "mesh_bbox": mesh_bbox,
        "inside_ratio": inside_ratio,
        "projection_plane_distance_threshold_m": plane_threshold,
    }


def observed_mask_cloud_with_pixels(
    meta: dict[str, Any],
    depth_m: np.ndarray,
    mask: np.ndarray,
    convention: str,
    depth_min_m: float,
    depth_max_m: float,
    q_min: float,
    q_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    points_right, u, v, inside = depth_points_in_right_camera(
        meta, depth_m, convention, depth_min_m=depth_min_m, depth_max_m=depth_max_m
    )
    width = int(meta["rgb_width_per_eye"])
    height = int(meta["rgb_height_per_eye"])
    valid_idx = np.flatnonzero(inside)
    ui = np.clip(np.round(u[valid_idx]).astype(np.int64), 0, width - 1)
    vi = np.clip(np.round(v[valid_idx]).astype(np.int64), 0, height - 1)
    mask_hits = mask[vi, ui]
    mask_idx = valid_idx[mask_hits]
    points = points_right[mask_idx]
    point_u = u[mask_idx]
    point_v = v[mask_idx]
    raw_count = len(points)
    if raw_count == 0:
        raise ValueError("No projected depth points landed inside the object mask.")

    z_lo, z_hi = np.quantile(points[:, 2], [q_min, q_max])
    keep = (points[:, 2] >= z_lo) & (points[:, 2] <= z_hi)
    points = points[keep]
    point_u = point_u[keep]
    point_v = point_v[keep]
    stats = {
        "raw_mask_point_count": raw_count,
        "filtered_point_count": int(len(points)),
        "z_filter_quantiles": [q_min, q_max],
        "z_filter_m": [float(z_lo), float(z_hi)],
        "bounds": [points.min(axis=0).tolist(), points.max(axis=0).tolist()],
        "extents": np.ptp(points, axis=0).tolist(),
        "pixel_bounds": [[float(point_u.min()), float(point_v.min())], [float(point_u.max()), float(point_v.max())]],
    }
    return points, point_u, point_v, stats


def observed_mask_cloud(
    meta: dict[str, Any],
    depth_m: np.ndarray,
    mask: np.ndarray,
    convention: str,
    depth_min_m: float,
    depth_max_m: float,
    q_min: float,
    q_max: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    points, _, _, stats = observed_mask_cloud_with_pixels(
        meta,
        depth_m,
        mask,
        convention,
        depth_min_m,
        depth_max_m,
        q_min,
        q_max,
    )
    return points, stats


def parse_vec(text: str | None, default: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    if not text:
        return np.asarray(default, dtype=np.float64)
    return np.asarray([float(item) for item in text.split()], dtype=np.float64)


def parse_urdf(urdf_path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    root = ET.parse(urdf_path).getroot()
    links = [link.attrib["name"] for link in root.findall("link") if "name" in link.attrib]
    joints = []
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        origin = joint.find("origin")
        axis = joint.find("axis")
        limit = joint.find("limit")
        joints.append(
            {
                "name": joint.attrib.get("name", "joint"),
                "type": joint.attrib.get("type", "unknown"),
                "parent": parent.attrib.get("link") if parent is not None else None,
                "child": child.attrib.get("link") if child is not None else None,
                "origin": parse_vec(origin.attrib.get("xyz") if origin is not None else None),
                "axis": parse_vec(axis.attrib.get("xyz") if axis is not None else None, (0.0, 0.0, 1.0)),
                "limit": dict(limit.attrib) if limit is not None else {},
            }
        )
    return links, joints


def compute_link_positions(links: list[str], joints: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    child_links = {joint["child"] for joint in joints if joint.get("child")}
    roots = [link for link in links if link not in child_links]
    positions = {link: np.zeros(3, dtype=np.float64) for link in roots}
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
        positions.setdefault(link, np.zeros(3, dtype=np.float64))
    for joint in joints:
        joint.setdefault("world_origin", positions.get(joint.get("child"), joint["origin"]))
    return positions


def part_label_from_path(path: Path) -> str:
    return path.stem.removeprefix("part_")


def load_particulate_parts(urdf_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    links, joints = parse_urdf(urdf_path)
    link_positions = compute_link_positions(links, joints)
    parts = []
    for mesh_path in sorted((urdf_path.parent / "meshes").glob("part_*.obj")):
        label = part_label_from_path(mesh_path)
        link = f"link_{label}"
        mesh = trimesh.load(mesh_path, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(mesh.geometry.values())
        mesh = mesh.copy()
        mesh.vertices = np.asarray(mesh.vertices, dtype=np.float64) + link_positions.get(link, np.zeros(3))
        parts.append({"label": label, "link": link, "mesh": mesh, "source_path": str(mesh_path)})
    return parts, joints


def concatenate_meshes(parts: list[dict[str, Any]]) -> trimesh.Trimesh:
    return trimesh.util.concatenate([part["mesh"] for part in parts])


def sample_mesh_points(mesh: trimesh.Trimesh, count: int, seed: int) -> np.ndarray:
    rng_state = np.random.get_state()
    np.random.seed(seed)
    try:
        points, _ = trimesh.sample.sample_surface(mesh, count)
    finally:
        np.random.set_state(rng_state)
    return np.asarray(points, dtype=np.float64)


def subsample_points(points: np.ndarray, count: int, seed: int) -> np.ndarray:
    if len(points) <= count:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=count, replace=False)
    return points[idx]


def pca_axes(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0)
    cov = np.cov(centered.T)
    _, vecs = np.linalg.eigh(cov)
    axes = vecs[:, ::-1]
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1
    return axes


def umeyama_sim3(source: np.ndarray, target: np.ndarray, allow_scaling: bool = True) -> tuple[float, np.ndarray, np.ndarray]:
    if len(source) < 3:
        raise ValueError("Need at least 3 correspondences for Sim(3).")
    mu_x = source.mean(axis=0)
    mu_y = target.mean(axis=0)
    x = source - mu_x
    y = target - mu_y
    cov = (y.T @ x) / len(source)
    u, singular, vt = np.linalg.svd(cov)
    s_mat = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        s_mat[-1, -1] = -1.0
    rot = u @ s_mat @ vt
    if allow_scaling:
        var_x = np.mean(np.sum(x * x, axis=1))
        scale = float(np.sum(singular * np.diag(s_mat)) / max(var_x, 1e-12))
    else:
        scale = 1.0
    trans = mu_y - scale * (rot @ mu_x)
    return scale, rot, trans


def apply_sim3(points: np.ndarray, scale: float, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    return scale * (points @ rot.T) + trans


def sim3_matrix(scale: float, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = scale * rot
    out[:3, 3] = trans
    return out


def score_alignment(source: np.ndarray, target: np.ndarray, scale: float, rot: np.ndarray, trans: np.ndarray, trim: float) -> float:
    transformed = apply_sim3(source, scale, rot, trans)
    dists, _ = cKDTree(target).query(transformed, k=1, workers=-1)
    keep = max(16, int(len(dists) * trim))
    return float(np.mean(np.partition(dists, keep - 1)[:keep]))


def initial_sim3_candidates(source: np.ndarray, target: np.ndarray, trim: float) -> list[dict[str, Any]]:
    src_axes = pca_axes(source)
    tgt_axes = pca_axes(target)
    src_center = source.mean(axis=0)
    tgt_center = target.mean(axis=0)
    src_diag = np.linalg.norm(np.ptp(source, axis=0))
    tgt_diag = np.linalg.norm(np.ptp(target, axis=0))
    base_scale = float(tgt_diag / max(src_diag, 1e-12))
    candidates = []
    for perm in itertools.permutations(range(3)):
        perm_mat = np.eye(3)[:, perm]
        for signs in itertools.product([-1.0, 1.0], repeat=3):
            sign_mat = np.diag(signs)
            orient = perm_mat @ sign_mat
            rot = tgt_axes @ orient @ src_axes.T
            if np.linalg.det(rot) < 0:
                continue
            trans = tgt_center - base_scale * (rot @ src_center)
            score = score_alignment(source, target, base_scale, rot, trans, trim)
            candidates.append({"scale": base_scale, "rotation": rot, "translation": trans, "score": score})
    candidates.sort(key=lambda item: item["score"])
    return candidates


def trimmed_sim3_icp(
    source: np.ndarray,
    target: np.ndarray,
    scale: float,
    rot: np.ndarray,
    trans: np.ndarray,
    trim_fraction: float,
    iterations: int,
) -> dict[str, Any]:
    tree = cKDTree(target)
    history = []
    for _ in range(iterations):
        transformed = apply_sim3(source, scale, rot, trans)
        dists, indices = tree.query(transformed, k=1, workers=-1)
        keep_count = max(32, int(len(dists) * trim_fraction))
        keep_idx = np.argpartition(dists, keep_count - 1)[:keep_count]
        new_scale, new_rot, new_trans = umeyama_sim3(source[keep_idx], target[indices[keep_idx]], allow_scaling=True)
        new_scale = float(np.clip(new_scale, 0.02, 2.0))
        metric = float(np.mean(dists[keep_idx]))
        history.append(metric)
        delta = abs(metric - history[-2]) if len(history) > 1 else np.inf
        scale, rot, trans = new_scale, new_rot, new_trans
        if delta < 1e-6:
            break
    final_score = score_alignment(source, target, scale, rot, trans, trim_fraction)
    return {
        "scale": scale,
        "rotation": rot,
        "translation": trans,
        "trimmed_mean_distance": final_score,
        "iterations": len(history),
        "history": history,
    }


def estimate_alignment(
    canonical_points: np.ndarray,
    observed_points: np.ndarray,
    seed: int,
    trim_fraction: float,
    iterations: int,
) -> dict[str, Any]:
    source = subsample_points(canonical_points, min(len(canonical_points), 16000), seed)
    target = subsample_points(observed_points, min(len(observed_points), 12000), seed + 1)
    candidates = initial_sim3_candidates(source, target, trim_fraction)[:8]
    refined = []
    for candidate in candidates:
        refined.append(
            trimmed_sim3_icp(
                source,
                target,
                candidate["scale"],
                candidate["rotation"],
                candidate["translation"],
                trim_fraction=trim_fraction,
                iterations=iterations,
            )
        )
    refined.sort(key=lambda item: item["trimmed_mean_distance"])
    best = refined[0]
    best["matrix"] = sim3_matrix(best["scale"], best["rotation"], best["translation"])
    best["initial_candidate_scores"] = [float(item["score"]) for item in candidates]
    return best


def transformed_part_projection_stats(
    parts: list[dict[str, Any]],
    meta: dict[str, Any],
    scale: float,
    rot: np.ndarray,
    trans: np.ndarray,
    seed: int,
) -> dict[str, dict[str, Any]]:
    stats = {}
    width = int(meta["rgb_width_per_eye"])
    height = int(meta["rgb_height_per_eye"])
    for part in parts:
        points = sample_mesh_points(part["mesh"], min(6000, max(512, len(part["mesh"].faces))), seed + int(part["label"]))
        transformed = apply_sim3(points, scale, rot, trans)
        u, v, z = project_right_camera_points(meta, transformed)
        inside = (z > 1e-6) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        if inside.sum() < 16:
            stats[part["label"]] = {"inside_ratio": float(inside.mean())}
            continue
        stats[part["label"]] = {
            "inside_ratio": float(inside.mean()),
            "u_median": float(np.median(u[inside])),
            "v_median": float(np.median(v[inside])),
            "z_median_m": float(np.median(z[inside])),
            "v_quantiles": np.quantile(v[inside], [0.1, 0.5, 0.9]).tolist(),
            "z_quantiles_m": np.quantile(z[inside], [0.1, 0.5, 0.9]).tolist(),
        }
    return stats


def semantic_order_penalty(
    part_stats: dict[str, dict[str, Any]],
    screen_label: str,
    base_label: str,
    require_semantic_order: bool,
) -> float:
    if not require_semantic_order:
        return 0.0
    if screen_label not in part_stats or base_label not in part_stats:
        return 0.0
    screen = part_stats[screen_label]
    base = part_stats[base_label]
    if "v_median" not in screen or "v_median" not in base:
        return 1.0 if require_semantic_order else 0.0
    penalty = 0.0
    if not screen["v_median"] < base["v_median"]:
        penalty += 1.0
    if "z_median_m" in screen and "z_median_m" in base and not screen["z_median_m"] > base["z_median_m"]:
        penalty += 0.25
    if screen.get("inside_ratio", 0.0) < 0.65:
        penalty += 0.25
    if base.get("inside_ratio", 0.0) < 0.65:
        penalty += 0.25
    if penalty > 0.0:
        penalty += 1.0
    return penalty


def estimate_pca_direct_alignment(
    parts: list[dict[str, Any]],
    canonical_points: np.ndarray,
    observed_points: np.ndarray,
    meta: dict[str, Any],
    mask: np.ndarray,
    seed: int,
    trim_fraction: float,
    candidate_count: int,
    screen_label: str,
    base_label: str,
    require_semantic_order: bool,
    grid_px: int,
) -> dict[str, Any]:
    source = subsample_points(canonical_points, min(len(canonical_points), 16000), seed)
    target = subsample_points(observed_points, min(len(observed_points), 12000), seed + 1)
    candidates = initial_sim3_candidates(source, target, trim_fraction)[: max(1, candidate_count)]
    evaluated = []
    for idx, candidate in enumerate(candidates):
        scale = float(candidate["scale"])
        rot = np.asarray(candidate["rotation"], dtype=np.float64)
        trans = np.asarray(candidate["translation"], dtype=np.float64)
        visible = visible_bidirectional_score(
            canonical_points,
            observed_points,
            meta,
            mask,
            scale,
            rot,
            trans,
            grid_px,
            trim_fraction=0.75,
        )
        part_stats = transformed_part_projection_stats(parts, meta, scale, rot, trans, seed + 100 + idx * 17)
        sem_penalty = semantic_order_penalty(part_stats, screen_label, base_label, require_semantic_order)
        score = float(visible.get("score", candidate["score"]) + sem_penalty)
        evaluated.append(
            {
                "candidate_index": idx,
                "score": score,
                "pca_trimmed_score": float(candidate["score"]),
                "visible_score": visible,
                "semantic_penalty": float(sem_penalty),
                "scale": scale,
                "rotation": rot,
                "translation": trans,
                "part_projection_stats": part_stats,
            }
        )

    viable = [item for item in evaluated if item["semantic_penalty"] < 1.0] if require_semantic_order else evaluated
    chosen_pool = viable or evaluated
    chosen_pool.sort(key=lambda item: item["score"])
    best = chosen_pool[0]
    return {
        "method": "pca_direct",
        "scale": best["scale"],
        "rotation": best["rotation"],
        "translation": best["translation"],
        "matrix": sim3_matrix(best["scale"], best["rotation"], best["translation"]),
        "score": best["score"],
        "pca_trimmed_score": best["pca_trimmed_score"],
        "visible_bidirectional_score": best["visible_score"].get("score"),
        "visible_model_to_observed_m": best["visible_score"].get("model_to_observed_m"),
        "visible_observed_to_model_m": best["visible_score"].get("observed_to_model_m"),
        "visible_count": best["visible_score"].get("visible_count"),
        "semantic_penalty": best["semantic_penalty"],
        "screen_part_label": screen_label,
        "base_part_label": base_label,
        "require_semantic_order": require_semantic_order,
        "chosen_candidate_index": best["candidate_index"],
        "part_projection_stats": best["part_projection_stats"],
        "candidate_scores": [
            {
                "candidate_index": item["candidate_index"],
                "score": item["score"],
                "pca_trimmed_score": item["pca_trimmed_score"],
                "visible_score": item["visible_score"],
                "semantic_penalty": item["semantic_penalty"],
                "scale": item["scale"],
                "part_projection_stats": item["part_projection_stats"],
            }
            for item in evaluated
        ],
    }


def estimate_base_first_initial_alignment(
    base_part: dict[str, Any],
    screen_part: dict[str, Any],
    base_canonical_points: np.ndarray,
    screen_canonical_points: np.ndarray,
    observed_base_points: np.ndarray,
    observed_screen_points: np.ndarray,
    meta: dict[str, Any],
    base_mask: np.ndarray,
    screen_mask: np.ndarray,
    seed: int,
    trim_fraction: float,
    candidate_count: int,
    base_label: str,
    screen_label: str,
    grid_px: int,
    screen_score_weight: float = 0.7,
) -> dict[str, Any]:
    source = subsample_points(base_canonical_points, min(len(base_canonical_points), 16000), seed)
    target = subsample_points(observed_base_points, min(len(observed_base_points), 12000), seed + 1)
    candidates = initial_sim3_candidates(source, target, trim_fraction)[: max(1, candidate_count)]
    evaluated = []
    for idx, candidate in enumerate(candidates):
        scale = float(candidate["scale"])
        rot = np.asarray(candidate["rotation"], dtype=np.float64)
        trans = np.asarray(candidate["translation"], dtype=np.float64)
        base_visible = visible_bidirectional_score(
            base_canonical_points,
            observed_base_points,
            meta,
            base_mask,
            scale,
            rot,
            trans,
            grid_px,
            trim_fraction=0.75,
        )
        screen_visible = visible_bidirectional_score(
            screen_canonical_points,
            observed_screen_points,
            meta,
            screen_mask,
            scale,
            rot,
            trans,
            grid_px,
            trim_fraction=0.75,
        )
        part_stats = transformed_part_projection_stats(
            [base_part, screen_part],
            meta,
            scale,
            rot,
            trans,
            seed + 300 + idx * 17,
        )
        sem_penalty = semantic_order_penalty(part_stats, screen_label, base_label, True)
        screen_score = float(screen_visible.get("score", float("inf")))
        base_score = float(base_visible.get("score", candidate["score"]))
        score = float(base_score + screen_score_weight * screen_score + sem_penalty)
        evaluated.append(
            {
                "candidate_index": idx,
                "score": score,
                "base_score": base_score,
                "screen_score": screen_score,
                "pca_trimmed_score": float(candidate["score"]),
                "base_visible_score": base_visible,
                "screen_visible_score": screen_visible,
                "semantic_penalty": float(sem_penalty),
                "screen_score_weight": float(screen_score_weight),
                "scale": scale,
                "rotation": rot,
                "translation": trans,
                "part_projection_stats": part_stats,
            }
        )

    evaluated.sort(key=lambda item: item["score"])
    best = evaluated[0]
    return {
        "method": "base_first_pca_direct",
        "scale": best["scale"],
        "rotation": best["rotation"],
        "translation": best["translation"],
        "matrix": sim3_matrix(best["scale"], best["rotation"], best["translation"]),
        "score": best["score"],
        "base_score": best["base_score"],
        "screen_score": best["screen_score"],
        "pca_trimmed_score": best["pca_trimmed_score"],
        "base_visible_score": best["base_visible_score"],
        "screen_visible_score": best["screen_visible_score"],
        "semantic_penalty": best["semantic_penalty"],
        "screen_score_weight": best["screen_score_weight"],
        "base_part_label": base_label,
        "screen_part_label": screen_label,
        "chosen_candidate_index": best["candidate_index"],
        "part_projection_stats": best["part_projection_stats"],
        "candidate_scores": [
            {
                "candidate_index": item["candidate_index"],
                "score": item["score"],
                "base_score": item["base_score"],
                "screen_score": item["screen_score"],
                "pca_trimmed_score": item["pca_trimmed_score"],
                "base_visible_score": item["base_visible_score"],
                "screen_visible_score": item["screen_visible_score"],
                "semantic_penalty": item["semantic_penalty"],
                "scale": item["scale"],
                "part_projection_stats": item["part_projection_stats"],
            }
            for item in evaluated
        ],
    }


def estimate_screen_first_initial_alignment(
    screen_part: dict[str, Any],
    base_part: dict[str, Any],
    screen_canonical_points: np.ndarray,
    base_canonical_points: np.ndarray,
    observed_screen_points: np.ndarray,
    observed_base_points: np.ndarray,
    meta: dict[str, Any],
    screen_mask: np.ndarray,
    base_mask: np.ndarray,
    seed: int,
    trim_fraction: float,
    candidate_count: int,
    screen_label: str,
    base_label: str,
    grid_px: int,
    base_score_weight: float = 0.7,
) -> dict[str, Any]:
    source = subsample_points(screen_canonical_points, min(len(screen_canonical_points), 16000), seed)
    target = subsample_points(observed_screen_points, min(len(observed_screen_points), 12000), seed + 1)
    candidates = initial_sim3_candidates(source, target, trim_fraction)[: max(1, candidate_count)]
    evaluated = []
    for idx, candidate in enumerate(candidates):
        scale = float(candidate["scale"])
        rot = np.asarray(candidate["rotation"], dtype=np.float64)
        trans = np.asarray(candidate["translation"], dtype=np.float64)
        screen_visible = visible_bidirectional_score(
            screen_canonical_points,
            observed_screen_points,
            meta,
            screen_mask,
            scale,
            rot,
            trans,
            grid_px,
            trim_fraction=0.75,
        )
        base_visible = visible_bidirectional_score(
            base_canonical_points,
            observed_base_points,
            meta,
            base_mask,
            scale,
            rot,
            trans,
            grid_px,
            trim_fraction=0.75,
        )
        part_stats = transformed_part_projection_stats(
            [screen_part, base_part],
            meta,
            scale,
            rot,
            trans,
            seed + 400 + idx * 17,
        )
        sem_penalty = semantic_order_penalty(part_stats, screen_label, base_label, True)
        screen_score = float(screen_visible.get("score", candidate["score"]))
        base_score = float(base_visible.get("score", float("inf")))
        score = float(screen_score + base_score_weight * base_score + sem_penalty)
        evaluated.append(
            {
                "candidate_index": idx,
                "score": score,
                "screen_score": screen_score,
                "base_score": base_score,
                "pca_trimmed_score": float(candidate["score"]),
                "screen_visible_score": screen_visible,
                "base_visible_score": base_visible,
                "semantic_penalty": float(sem_penalty),
                "base_score_weight": float(base_score_weight),
                "scale": scale,
                "rotation": rot,
                "translation": trans,
                "part_projection_stats": part_stats,
            }
        )

    evaluated.sort(key=lambda item: item["score"])
    best = evaluated[0]
    return {
        "method": "screen_first_pca_direct",
        "scale": best["scale"],
        "rotation": best["rotation"],
        "translation": best["translation"],
        "matrix": sim3_matrix(best["scale"], best["rotation"], best["translation"]),
        "score": best["score"],
        "screen_score": best["screen_score"],
        "base_score": best["base_score"],
        "pca_trimmed_score": best["pca_trimmed_score"],
        "screen_visible_score": best["screen_visible_score"],
        "base_visible_score": best["base_visible_score"],
        "semantic_penalty": best["semantic_penalty"],
        "base_score_weight": best["base_score_weight"],
        "screen_part_label": screen_label,
        "base_part_label": base_label,
        "chosen_candidate_index": best["candidate_index"],
        "part_projection_stats": best["part_projection_stats"],
        "candidate_scores": [
            {
                "candidate_index": item["candidate_index"],
                "score": item["score"],
                "screen_score": item["screen_score"],
                "base_score": item["base_score"],
                "pca_trimmed_score": item["pca_trimmed_score"],
                "screen_visible_score": item["screen_visible_score"],
                "base_visible_score": item["base_visible_score"],
                "semantic_penalty": item["semantic_penalty"],
                "scale": item["scale"],
                "part_projection_stats": item["part_projection_stats"],
            }
            for item in evaluated
        ],
    }


def screen_hinge_axis_twist_refine(
    alignment: dict[str, Any],
    joints: list[dict[str, Any]],
    screen_canonical_points: np.ndarray,
    observed_screen_points: np.ndarray,
    observed_base_points: np.ndarray,
    meta: dict[str, Any],
    screen_mask: np.ndarray,
    grid_px: int,
    trim_fraction: float,
    max_abs_deg: float,
) -> dict[str, Any] | None:
    if not joints or observed_screen_points is None or observed_base_points is None:
        return None
    if len(observed_screen_points) < 64 or len(observed_base_points) < 64:
        return None

    scale = float(alignment["scale"])
    rot = np.asarray(alignment["rotation"], dtype=np.float64)
    trans = np.asarray(alignment["translation"], dtype=np.float64)
    screen_center, observed_screen_normal, _ = plane_from_points(observed_screen_points)
    _, observed_base_normal, _ = plane_from_points(observed_base_points)

    _, canonical_screen_normal, _ = plane_from_points(screen_canonical_points)
    current_screen_normal = rot @ canonical_screen_normal
    current_screen_normal = current_screen_normal / (np.linalg.norm(current_screen_normal) + 1e-12)
    if float(observed_screen_normal @ current_screen_normal) < 0.0:
        observed_screen_normal = -observed_screen_normal

    observed_hinge_axis = np.cross(observed_screen_normal, observed_base_normal)
    observed_hinge_norm = float(np.linalg.norm(observed_hinge_axis))
    if observed_hinge_norm < 1e-8:
        return None
    observed_hinge_axis = observed_hinge_axis / observed_hinge_norm

    joint = next((item for item in joints if item.get("type") != "fixed"), joints[0])
    current_joint_axis = rot @ np.asarray(joint["axis"], dtype=np.float64)
    current_joint_axis = current_joint_axis / (np.linalg.norm(current_joint_axis) + 1e-12)

    twist_axis = observed_screen_normal / (np.linalg.norm(observed_screen_normal) + 1e-12)

    def project_to_screen_axis(axis: np.ndarray) -> np.ndarray:
        projected = axis - float(axis @ twist_axis) * twist_axis
        norm = float(np.linalg.norm(projected))
        if norm < 1e-8:
            return projected
        return projected / norm

    current_projected = project_to_screen_axis(current_joint_axis)
    observed_projected = project_to_screen_axis(observed_hinge_axis)
    if np.linalg.norm(current_projected) < 1e-8 or np.linalg.norm(observed_projected) < 1e-8:
        return None
    if float(current_projected @ observed_projected) < 0.0:
        observed_projected = -observed_projected

    signed_angle_rad = float(
        np.arctan2(
            float(twist_axis @ np.cross(current_projected, observed_projected)),
            float(np.clip(current_projected @ observed_projected, -1.0, 1.0)),
        )
    )
    signed_angle_deg = float(np.degrees(signed_angle_rad))
    clipped_angle_deg = float(np.clip(signed_angle_deg, -abs(max_abs_deg), abs(max_abs_deg)))
    clipped_angle_rad = float(np.deg2rad(clipped_angle_deg))
    twist_rot = rotation_from_axis_angle(twist_axis, clipped_angle_rad)
    refined_rot = twist_rot @ rot
    refined_trans = twist_rot @ (trans - screen_center) + screen_center
    refined_axis = refined_rot @ np.asarray(joint["axis"], dtype=np.float64)
    refined_axis = refined_axis / (np.linalg.norm(refined_axis) + 1e-12)
    refined_projected = project_to_screen_axis(refined_axis)
    if np.linalg.norm(refined_projected) >= 1e-8 and float(refined_projected @ observed_projected) < 0.0:
        refined_projected = -refined_projected

    metrics = visible_bidirectional_score(
        screen_canonical_points,
        observed_screen_points,
        meta,
        screen_mask,
        scale,
        refined_rot,
        refined_trans,
        grid_px,
        trim_fraction=trim_fraction,
    )
    out = dict(alignment)
    out.update(
        {
            "method": "screen_hinge_axis_twist_refined",
            "scale": scale,
            "rotation": refined_rot,
            "translation": refined_trans,
            "matrix": sim3_matrix(scale, refined_rot, refined_trans),
            "score": metrics.get("score"),
            "model_to_observed_m": metrics.get("model_to_observed_m"),
            "observed_to_model_m": metrics.get("observed_to_model_m"),
            "visible_count": metrics.get("visible_count"),
            "axis_twist": {
                "joint_name": joint.get("name"),
                "pivot_screen_center_xyz": screen_center,
                "twist_axis_screen_normal_xyz": twist_axis,
                "observed_hinge_axis_xyz": observed_hinge_axis,
                "current_joint_axis_xyz": current_joint_axis,
                "refined_joint_axis_xyz": refined_axis,
                "unclipped_angle_deg": signed_angle_deg,
                "applied_angle_deg": clipped_angle_deg,
                "max_abs_deg": float(max_abs_deg),
                "axis_error_before_deg": acute_angle_deg(current_projected, observed_projected),
                "axis_error_after_deg": acute_angle_deg(refined_projected, observed_projected),
                "screen_visible_score_after": metrics,
            },
        }
    )
    return out


def project_right_camera_points(meta: dict[str, Any], points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kr = meta["rgb_intrinsics_right"]
    z = points[:, 2]
    u = kr["fx"] * points[:, 0] / z + kr["cx"]
    v = kr["fy"] * points[:, 1] / z + kr["cy"]
    return u, v, z


def visible_canonical_subset(
    canonical_points: np.ndarray,
    meta: dict[str, Any],
    mask: np.ndarray,
    scale: float,
    rot: np.ndarray,
    trans: np.ndarray,
    grid_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    transformed = apply_sim3(canonical_points, scale, rot, trans)
    u, v, z = project_right_camera_points(meta, transformed)
    width = int(meta["rgb_width_per_eye"])
    height = int(meta["rgb_height_per_eye"])
    inside = (z > 1e-6) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    idx = np.flatnonzero(inside)
    if idx.size == 0:
        return canonical_points[:0], transformed[:0]

    ui = np.clip(np.round(u[idx]).astype(np.int64), 0, width - 1)
    vi = np.clip(np.round(v[idx]).astype(np.int64), 0, height - 1)
    mask_hits = mask[vi, ui]
    idx = idx[mask_hits]
    if idx.size == 0:
        return canonical_points[:0], transformed[:0]

    ui = (np.clip(np.round(u[idx]).astype(np.int64), 0, width - 1) // grid_px)
    vi = (np.clip(np.round(v[idx]).astype(np.int64), 0, height - 1) // grid_px)
    grid_w = int(np.ceil(width / grid_px))
    keys = vi * grid_w + ui
    order = np.lexsort((z[idx], keys))
    ordered_idx = idx[order]
    ordered_keys = keys[order]
    first = np.r_[True, ordered_keys[1:] != ordered_keys[:-1]]
    selected = ordered_idx[first]
    return canonical_points[selected], transformed[selected]


def visible_projected_surface_points(
    points: np.ndarray,
    meta: dict[str, Any],
    mask: np.ndarray,
    grid_px: int,
) -> np.ndarray:
    if len(points) == 0:
        return points
    u, v, z = project_right_camera_points(meta, points)
    height, width = mask.shape
    inside = (z > 1e-6) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    idx = np.flatnonzero(inside)
    if idx.size == 0:
        return points[:0]
    ui_full = np.clip(np.round(u[idx]).astype(np.int64), 0, width - 1)
    vi_full = np.clip(np.round(v[idx]).astype(np.int64), 0, height - 1)
    idx = idx[mask[vi_full, ui_full]]
    if idx.size == 0:
        return points[:0]
    ui = np.clip(np.round(u[idx]).astype(np.int64), 0, width - 1) // max(1, int(grid_px))
    vi = np.clip(np.round(v[idx]).astype(np.int64), 0, height - 1) // max(1, int(grid_px))
    grid_w = int(np.ceil(width / max(1, int(grid_px))))
    keys = vi * grid_w + ui
    order = np.lexsort((z[idx], keys))
    ordered_idx = idx[order]
    ordered_keys = keys[order]
    first = np.r_[True, ordered_keys[1:] != ordered_keys[:-1]]
    return points[ordered_idx[first]]


def trimmed_mean(values: np.ndarray, fraction: float) -> float:
    if values.size == 0:
        return float("inf")
    keep = max(1, int(values.size * fraction))
    return float(np.mean(np.partition(values, keep - 1)[:keep]))


def visible_bidirectional_score(
    canonical_points: np.ndarray,
    observed_points: np.ndarray,
    meta: dict[str, Any],
    mask: np.ndarray,
    scale: float,
    rot: np.ndarray,
    trans: np.ndarray,
    grid_px: int,
    trim_fraction: float,
) -> dict[str, Any]:
    _, visible_transformed = visible_canonical_subset(canonical_points, meta, mask, scale, rot, trans, grid_px)
    if len(visible_transformed) < 64:
        return {"score": float("inf"), "visible_count": int(len(visible_transformed))}
    obs_tree = cKDTree(observed_points)
    src_to_obs, _ = obs_tree.query(visible_transformed, k=1, workers=-1)
    model_tree = cKDTree(visible_transformed)
    obs_to_src, _ = model_tree.query(observed_points, k=1, workers=-1)
    src_score = trimmed_mean(src_to_obs, trim_fraction)
    obs_score = trimmed_mean(obs_to_src, trim_fraction)
    return {
        "score": 0.5 * (src_score + obs_score),
        "model_to_observed_m": src_score,
        "observed_to_model_m": obs_score,
        "visible_count": int(len(visible_transformed)),
    }


def visible_sim3_refine(
    canonical_points: np.ndarray,
    observed_points: np.ndarray,
    meta: dict[str, Any],
    mask: np.ndarray,
    initial: dict[str, Any],
    grid_px: int,
    trim_fraction: float,
    iterations: int,
) -> dict[str, Any]:
    scale = float(initial["scale"])
    rot = np.asarray(initial["rotation"], dtype=np.float64)
    trans = np.asarray(initial["translation"], dtype=np.float64)
    scale_min = scale * 0.75
    scale_max = scale * 1.25
    obs_tree = cKDTree(observed_points)
    history = []
    best = {
        "scale": scale,
        "rotation": rot,
        "translation": trans,
        **visible_bidirectional_score(canonical_points, observed_points, meta, mask, scale, rot, trans, grid_px, trim_fraction),
    }

    for _ in range(iterations):
        visible_source, visible_transformed = visible_canonical_subset(
            canonical_points, meta, mask, scale, rot, trans, grid_px
        )
        if len(visible_source) < 128:
            break
        dists, indices = obs_tree.query(visible_transformed, k=1, workers=-1)
        keep_count = max(64, int(len(dists) * trim_fraction))
        keep_idx = np.argpartition(dists, keep_count - 1)[:keep_count]
        new_scale, new_rot, new_trans = umeyama_sim3(
            visible_source[keep_idx],
            observed_points[indices[keep_idx]],
            allow_scaling=True,
        )
        new_scale = float(np.clip(new_scale, scale_min, scale_max))
        metrics = visible_bidirectional_score(
            canonical_points, observed_points, meta, mask, new_scale, new_rot, new_trans, grid_px, trim_fraction
        )
        history.append(metrics["score"])
        scale, rot, trans = new_scale, new_rot, new_trans
        if metrics["score"] < best["score"]:
            best = {
                "scale": scale,
                "rotation": rot,
                "translation": trans,
                **metrics,
            }
        if len(history) > 2 and abs(history[-1] - history[-2]) < 1e-6:
            break

    best["matrix"] = sim3_matrix(best["scale"], best["rotation"], best["translation"])
    best["iterations"] = len(history)
    best["history"] = history
    return best


def solve_scale_translation_fixed_rotation(source: np.ndarray, target: np.ndarray, rot: np.ndarray) -> tuple[float, np.ndarray]:
    rotated = source @ rot.T
    mu_x = rotated.mean(axis=0)
    mu_y = target.mean(axis=0)
    x = rotated - mu_x
    y = target - mu_y
    denom = float(np.sum(x * x))
    if denom < 1e-12:
        return 1.0, mu_y - mu_x
    scale = float(np.sum(x * y) / denom)
    trans = mu_y - scale * mu_x
    return scale, trans


def solve_translation_fixed_scale_rotation(
    source: np.ndarray,
    target: np.ndarray,
    scale: float,
    rot: np.ndarray,
) -> np.ndarray:
    return target.mean(axis=0) - scale * (rot @ source.mean(axis=0))


def rotation_angle_deg(rot: np.ndarray) -> float:
    value = np.clip((np.trace(rot) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(value)))


def rotation_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12 or abs(angle_rad) < 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z = axis / norm
    cos_t = float(np.cos(angle_rad))
    sin_t = float(np.sin(angle_rad))
    one_minus = 1.0 - cos_t
    return np.asarray(
        [
            [cos_t + x * x * one_minus, x * y * one_minus - z * sin_t, x * z * one_minus + y * sin_t],
            [y * x * one_minus + z * sin_t, cos_t + y * y * one_minus, y * z * one_minus - x * sin_t],
            [z * x * one_minus - y * sin_t, z * y * one_minus + x * sin_t, cos_t + z * z * one_minus],
        ],
        dtype=np.float64,
    )


def constrain_rotation_to_initial(rot: np.ndarray, initial_rot: np.ndarray, max_angle_deg: float) -> tuple[np.ndarray, float]:
    if max_angle_deg <= 0.0:
        return initial_rot, 0.0
    delta = rot @ initial_rot.T
    angle_rad = float(np.arccos(np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0)))
    angle_deg = float(np.degrees(angle_rad))
    if angle_deg <= max_angle_deg:
        return rot, angle_deg
    axis = np.asarray(
        [
            delta[2, 1] - delta[1, 2],
            delta[0, 2] - delta[2, 0],
            delta[1, 0] - delta[0, 1],
        ],
        dtype=np.float64,
    )
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-12:
        return initial_rot, 0.0
    clipped_delta = rotation_from_axis_angle(axis / axis_norm, np.deg2rad(max_angle_deg))
    return clipped_delta @ initial_rot, float(max_angle_deg)


def constrained_visible_refine(
    canonical_points: np.ndarray,
    observed_points: np.ndarray,
    meta: dict[str, Any],
    mask: np.ndarray,
    initial: dict[str, Any],
    grid_px: int,
    trim_fraction: float,
    iterations: int,
    scale_min_multiplier: float,
    scale_max_multiplier: float,
    rotation_max_deg: float,
) -> dict[str, Any]:
    initial_scale = float(initial["scale"])
    scale = initial_scale
    initial_rot = np.asarray(initial["rotation"], dtype=np.float64)
    rot = initial_rot
    trans = np.asarray(initial["translation"], dtype=np.float64)
    scale_min = initial_scale * scale_min_multiplier
    scale_max = initial_scale * scale_max_multiplier
    obs_tree = cKDTree(observed_points)
    history = []
    best = {
        "scale": scale,
        "rotation": rot,
        "translation": trans,
        **visible_bidirectional_score(canonical_points, observed_points, meta, mask, scale, rot, trans, grid_px, trim_fraction),
    }

    for _ in range(iterations):
        visible_source, visible_transformed = visible_canonical_subset(
            canonical_points, meta, mask, scale, rot, trans, grid_px
        )
        if len(visible_source) < 128:
            break
        dists, indices = obs_tree.query(visible_transformed, k=1, workers=-1)
        keep_count = max(64, int(len(dists) * trim_fraction))
        keep_idx = np.argpartition(dists, keep_count - 1)[:keep_count]
        new_scale, new_rot, new_trans = umeyama_sim3(
            visible_source[keep_idx],
            observed_points[indices[keep_idx]],
            allow_scaling=True,
        )
        new_scale = float(np.clip(new_scale, scale_min, scale_max))
        new_rot, rotation_delta_deg = constrain_rotation_to_initial(new_rot, initial_rot, rotation_max_deg)
        new_trans = solve_translation_fixed_scale_rotation(
            visible_source[keep_idx],
            observed_points[indices[keep_idx]],
            new_scale,
            new_rot,
        )
        metrics = visible_bidirectional_score(
            canonical_points, observed_points, meta, mask, new_scale, new_rot, new_trans, grid_px, trim_fraction
        )
        metrics["rotation_delta_deg"] = rotation_delta_deg
        history.append(metrics["score"])
        scale, rot, trans = new_scale, new_rot, new_trans
        if metrics["score"] < best["score"]:
            best = {
                "scale": scale,
                "rotation": rot,
                "translation": trans,
                **metrics,
            }
        if len(history) > 2 and abs(history[-1] - history[-2]) < 1e-6:
            break

    best["method"] = "constrained_visible_refined"
    best["matrix"] = sim3_matrix(best["scale"], best["rotation"], best["translation"])
    best["iterations"] = len(history)
    best["history"] = history
    best["rotation_locked"] = rotation_max_deg <= 0.0
    best["rotation_max_deg"] = float(rotation_max_deg)
    best["rotation_delta_deg"] = rotation_angle_deg(best["rotation"] @ initial_rot.T)
    best["initial_scale"] = initial_scale
    best["scale_bounds"] = [float(scale_min), float(scale_max)]
    return best


def projected_bbox_quantiles(
    meta: dict[str, Any],
    points: np.ndarray,
    q_min: float,
    q_max: float,
) -> tuple[np.ndarray | None, float]:
    u, v, z = project_right_camera_points(meta, points)
    width = int(meta["rgb_width_per_eye"])
    height = int(meta["rgb_height_per_eye"])
    inside = (z > 1e-6) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if inside.sum() < 64:
        return None, float(inside.mean())
    bbox = np.asarray(
        [
            np.quantile(u[inside], q_min),
            np.quantile(v[inside], q_min),
            np.quantile(u[inside], q_max),
            np.quantile(v[inside], q_max),
        ],
        dtype=np.float64,
    )
    return bbox, float(inside.mean())


def mask_bbox_quantiles(mask: np.ndarray, q_min: float, q_max: float) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("Mask is empty; cannot compute silhouette bounds.")
    return np.asarray(
        [
            np.quantile(xs, q_min),
            np.quantile(ys, q_min),
            np.quantile(xs, q_max),
            np.quantile(ys, q_max),
        ],
        dtype=np.float64,
    )


def silhouette_scale_refine(
    canonical_points: np.ndarray,
    meta: dict[str, Any],
    mask: np.ndarray,
    initial: dict[str, Any],
    q_min: float,
    q_max: float,
    min_multiplier: float,
    max_multiplier: float,
    steps: int,
    boundary_trim_fraction: float,
    outside_weight: float,
    boundary_weight: float,
    bbox_weight: float,
) -> dict[str, Any]:
    scale0 = float(initial["scale"])
    rot = np.asarray(initial["rotation"], dtype=np.float64)
    trans0 = np.asarray(initial["translation"], dtype=np.float64)
    source_center = canonical_points.mean(axis=0)
    pivot_cam = scale0 * (rot @ source_center) + trans0
    target_bbox = mask_bbox_quantiles(mask, q_min, q_max)
    target_size = np.maximum(target_bbox[[2, 3]] - target_bbox[[0, 1]], 1.0)
    target_diag = float(np.linalg.norm(target_size))
    outside_distance = distance_transform_edt(~mask)
    boundary = mask ^ binary_erosion(mask)
    boundary_y, boundary_x = np.nonzero(boundary)
    if len(boundary_x) == 0:
        boundary_y, boundary_x = np.nonzero(mask)
    boundary_xy = np.stack([boundary_x, boundary_y], axis=1).astype(np.float64)
    if len(boundary_xy) > 12000:
        rng = np.random.default_rng(0)
        boundary_xy = boundary_xy[rng.choice(len(boundary_xy), size=12000, replace=False)]
    boundary_keep_count = max(1, int(len(boundary_xy) * boundary_trim_fraction))
    width = int(meta["rgb_width_per_eye"])
    height = int(meta["rgb_height_per_eye"])

    best = None
    evaluations = []
    step_count = max(3, int(steps))
    for multiplier in np.linspace(min_multiplier, max_multiplier, step_count):
        scale = scale0 * float(multiplier)
        trans = pivot_cam - scale * (rot @ source_center)
        transformed = apply_sim3(canonical_points, scale, rot, trans)
        u, v, z = project_right_camera_points(meta, transformed)
        inside = (z > 1e-6) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        if inside.sum() < 64:
            continue
        inside_ratio = float(inside.mean())
        uv = np.stack([u[inside], v[inside]], axis=1)
        bbox = np.asarray(
            [
                np.quantile(uv[:, 0], q_min),
                np.quantile(uv[:, 1], q_min),
                np.quantile(uv[:, 0], q_max),
                np.quantile(uv[:, 1], q_max),
            ],
            dtype=np.float64,
        )
        ui = np.clip(np.round(uv[:, 0]).astype(np.int64), 0, width - 1)
        vi = np.clip(np.round(uv[:, 1]).astype(np.int64), 0, height - 1)
        outside_dist_px = outside_distance[vi, ui]
        outside_score = float(np.mean((outside_dist_px / target_diag) ** 2))

        tree = cKDTree(uv)
        boundary_dist_px, _ = tree.query(boundary_xy, k=1, workers=-1)
        boundary_score = float(
            np.mean(np.partition(boundary_dist_px, boundary_keep_count - 1)[:boundary_keep_count] / target_diag) ** 2
        )
        edge_error = (bbox - target_bbox) / np.asarray(
            [target_size[0], target_size[1], target_size[0], target_size[1]],
            dtype=np.float64,
        )
        center_error = ((bbox[[0, 1]] + bbox[[2, 3]]) - (target_bbox[[0, 1]] + target_bbox[[2, 3]])) * 0.5 / target_size
        size_error = ((bbox[[2, 3]] - bbox[[0, 1]]) - target_size) / target_size
        bbox_score = float(np.mean(edge_error * edge_error) + 0.35 * np.mean(center_error * center_error) + 0.65 * np.mean(size_error * size_error))
        score = float(outside_weight * outside_score + boundary_weight * boundary_score + bbox_weight * bbox_score)
        item = {
            "score": score,
            "scale": scale,
            "multiplier": float(multiplier),
            "translation": trans,
            "projected_bbox": bbox,
            "inside_ratio": inside_ratio,
            "outside_score": outside_score,
            "boundary_score": boundary_score,
            "bbox_score": bbox_score,
            "outside_distance_quantiles_px": np.quantile(outside_dist_px, [0.5, 0.9, 0.98]),
            "center_error_norm": center_error,
            "size_error_norm": size_error,
        }
        evaluations.append(item)
        if best is None or item["score"] < best["score"]:
            best = item

    if best is None:
        out = dict(initial)
        out["method"] = "silhouette_refined_failed"
        out["silhouette_error"] = "No valid projected bbox candidates."
        out["matrix"] = sim3_matrix(out["scale"], out["rotation"], out["translation"])
        return out

    out = dict(initial)
    out["scale"] = best["scale"]
    out["rotation"] = rot
    out["translation"] = best["translation"]
    out["matrix"] = sim3_matrix(out["scale"], rot, out["translation"])
    out["method"] = "silhouette_refined"
    out["silhouette_score"] = best["score"]
    out["silhouette_multiplier"] = best["multiplier"]
    out["silhouette_projected_bbox"] = best["projected_bbox"]
    out["silhouette_target_bbox"] = target_bbox
    out["silhouette_inside_ratio"] = best["inside_ratio"]
    out["silhouette_outside_score"] = best["outside_score"]
    out["silhouette_boundary_score"] = best["boundary_score"]
    out["silhouette_bbox_score"] = best["bbox_score"]
    out["silhouette_outside_distance_quantiles_px"] = best["outside_distance_quantiles_px"]
    out["silhouette_center_error_norm"] = best["center_error_norm"]
    out["silhouette_size_error_norm"] = best["size_error_norm"]
    out["silhouette_boundary_count"] = int(len(boundary_xy))
    out["silhouette_boundary_trim_fraction"] = float(boundary_trim_fraction)
    out["silhouette_weights"] = {
        "outside": float(outside_weight),
        "boundary": float(boundary_weight),
        "bbox": float(bbox_weight),
    }
    out["silhouette_top_candidates"] = [
        {
            "score": item["score"],
            "scale": item["scale"],
            "multiplier": item["multiplier"],
            "projected_bbox": item["projected_bbox"],
            "inside_ratio": item["inside_ratio"],
            "outside_score": item["outside_score"],
            "boundary_score": item["boundary_score"],
            "bbox_score": item["bbox_score"],
            "outside_distance_quantiles_px": item["outside_distance_quantiles_px"],
        }
        for item in sorted(evaluations, key=lambda item: item["score"])[:8]
    ]
    return out


def translation_for_projected_shift(
    meta: dict[str, Any],
    pivot_cam: np.ndarray,
    du_px: float,
    dv_px: float,
) -> np.ndarray:
    kr = meta["rgb_intrinsics_right"]
    z = float(max(pivot_cam[2], 1e-6))
    return np.asarray([du_px * z / kr["fx"], dv_px * z / kr["fy"], 0.0], dtype=np.float64)


def projected_bbox_error(
    projected_bbox: np.ndarray,
    target_bbox: np.ndarray,
) -> dict[str, Any]:
    target_size = np.maximum(target_bbox[[2, 3]] - target_bbox[[0, 1]], 1.0)
    target_center = 0.5 * (target_bbox[[0, 1]] + target_bbox[[2, 3]])
    projected_size = np.maximum(projected_bbox[[2, 3]] - projected_bbox[[0, 1]], 1.0)
    projected_center = 0.5 * (projected_bbox[[0, 1]] + projected_bbox[[2, 3]])
    edge_error = (projected_bbox - target_bbox) / np.asarray(
        [target_size[0], target_size[1], target_size[0], target_size[1]],
        dtype=np.float64,
    )
    center_error = (projected_center - target_center) / target_size
    size_error = (projected_size - target_size) / target_size
    return {
        "edge_error_norm": edge_error,
        "center_error_norm": center_error,
        "size_error_norm": size_error,
        "score": float(
            np.mean(edge_error * edge_error)
            + 0.75 * np.mean(center_error * center_error)
            + 1.25 * np.mean(size_error * size_error)
        ),
    }


def screen_projection_refine_alignment(
    canonical_points: np.ndarray,
    observed_points: np.ndarray,
    meta: dict[str, Any],
    mask: np.ndarray,
    initial: dict[str, Any],
    grid_px: int,
    trim_fraction: float,
    q_min: float,
    q_max: float,
    scale_min_multiplier: float,
    scale_max_multiplier: float,
    shift_max_px: float,
    depth_weight: float,
) -> dict[str, Any]:
    scale0 = float(initial["scale"])
    rot = np.asarray(initial["rotation"], dtype=np.float64)
    trans0 = np.asarray(initial["translation"], dtype=np.float64)
    source_center = canonical_points.mean(axis=0)
    pivot_cam = scale0 * (rot @ source_center) + trans0
    target_bbox = mask_bbox_quantiles(mask, q_min, q_max)
    observed_center, observed_normal, _ = plane_from_points(observed_points)
    initial_all_points = apply_sim3(canonical_points, scale0, rot, trans0)
    plane_distances = np.abs((initial_all_points - observed_center) @ observed_normal)
    plane_threshold = float(np.clip(np.quantile(plane_distances, 0.60), 0.003, 0.006))
    projection_points = canonical_points[plane_distances <= plane_threshold]
    if len(projection_points) < 1024:
        projection_points = canonical_points
        plane_threshold = float("inf")

    initial_points = apply_sim3(projection_points, scale0, rot, trans0)
    initial_bbox, initial_inside_ratio = projected_bbox_quantiles(meta, initial_points, q_min, q_max)
    if initial_bbox is None:
        out = dict(initial)
        out["method"] = "screen_projection_refined_failed"
        out["projection_error"] = "No valid projected screen bbox candidates."
        out["matrix"] = sim3_matrix(out["scale"], out["rotation"], out["translation"])
        return out

    initial_bbox_metrics = projected_bbox_error(initial_bbox, target_bbox)
    initial_depth = visible_bidirectional_score(
        projection_points,
        observed_points,
        meta,
        mask,
        scale0,
        rot,
        trans0,
        grid_px,
        trim_fraction,
    )

    target_size = np.maximum(target_bbox[[2, 3]] - target_bbox[[0, 1]], 1.0)
    target_center = 0.5 * (target_bbox[[0, 1]] + target_bbox[[2, 3]])
    initial_size = np.maximum(initial_bbox[[2, 3]] - initial_bbox[[0, 1]], 1.0)
    initial_center = 0.5 * (initial_bbox[[0, 1]] + initial_bbox[[2, 3]])
    size_ratio = float(np.mean(target_size / initial_size))
    center_delta = target_center - initial_center

    scale_multipliers = sorted(
        set(
            float(np.clip(value, scale_min_multiplier, scale_max_multiplier))
            for value in np.concatenate(
                [
                    np.linspace(scale_min_multiplier, scale_max_multiplier, 37),
                    np.asarray([1.0, size_ratio, 0.5 * (1.0 + size_ratio)], dtype=np.float64),
                ]
            )
        )
    )
    shift_candidates = [-1.0, -0.5, 0.0, 0.5, 1.0]
    du_values = sorted(
        set(float(np.clip(center_delta[0] * factor, -shift_max_px, shift_max_px)) for factor in shift_candidates)
        | {0.0, float(np.clip(center_delta[0], -shift_max_px, shift_max_px))}
    )
    dv_values = sorted(
        set(float(np.clip(center_delta[1] * factor, -shift_max_px, shift_max_px)) for factor in shift_candidates)
        | {0.0, float(np.clip(center_delta[1], -shift_max_px, shift_max_px))}
    )

    best = None
    evaluations = []
    for multiplier in scale_multipliers:
        scale = scale0 * multiplier
        scaled_pivot_trans = pivot_cam - scale * (rot @ source_center)
        for du_px in du_values:
            for dv_px in dv_values:
                trans = scaled_pivot_trans + translation_for_projected_shift(meta, pivot_cam, du_px, dv_px)
                transformed = apply_sim3(projection_points, scale, rot, trans)
                projected_bbox, inside_ratio = projected_bbox_quantiles(meta, transformed, q_min, q_max)
                if projected_bbox is None:
                    continue
                bbox_metrics = projected_bbox_error(projected_bbox, target_bbox)
                depth_metrics = visible_bidirectional_score(
                    projection_points,
                    observed_points,
                    meta,
                    mask,
                    scale,
                    rot,
                    trans,
                    grid_px,
                    trim_fraction,
                )
                depth_score = float(depth_metrics.get("score", float("inf")))
                if not np.isfinite(depth_score):
                    continue
                score = float(bbox_metrics["score"] + depth_weight * depth_score)
                item = {
                    "score": score,
                    "bbox_score": bbox_metrics["score"],
                    "depth_score_m": depth_score,
                    "scale": scale,
                    "multiplier": float(multiplier),
                    "translation": trans,
                    "shift_px": [float(du_px), float(dv_px)],
                    "projected_bbox": projected_bbox,
                    "inside_ratio": inside_ratio,
                    "bbox_metrics": bbox_metrics,
                    "depth_metrics": depth_metrics,
                }
                evaluations.append(item)
                if best is None or item["score"] < best["score"]:
                    best = item

    if best is None:
        out = dict(initial)
        out["method"] = "screen_projection_refined_failed"
        out["projection_error"] = "No valid projected screen bbox candidates."
        out["matrix"] = sim3_matrix(out["scale"], out["rotation"], out["translation"])
        return out

    out = dict(initial)
    out["scale"] = best["scale"]
    out["rotation"] = rot
    out["translation"] = best["translation"]
    out["matrix"] = sim3_matrix(best["scale"], rot, best["translation"])
    out["method"] = "screen_projection_refined"
    out["score"] = best["depth_metrics"].get("score")
    out["model_to_observed_m"] = best["depth_metrics"].get("model_to_observed_m")
    out["observed_to_model_m"] = best["depth_metrics"].get("observed_to_model_m")
    out["visible_count"] = best["depth_metrics"].get("visible_count")
    out["screen_projection_refine"] = {
        "target_bbox": target_bbox,
        "initial_bbox": initial_bbox,
        "projected_bbox": best["projected_bbox"],
        "initial_inside_ratio": initial_inside_ratio,
        "inside_ratio": best["inside_ratio"],
        "initial_bbox_score": initial_bbox_metrics["score"],
        "bbox_score": best["bbox_score"],
        "initial_depth_score_m": initial_depth.get("score"),
        "depth_score_m": best["depth_score_m"],
        "combined_score": best["score"],
        "scale_multiplier": best["multiplier"],
        "shift_px": best["shift_px"],
        "depth_weight": float(depth_weight),
        "projection_source_count": int(len(projection_points)),
        "projection_plane_distance_threshold_m": plane_threshold,
        "bbox_metrics": best["bbox_metrics"],
        "top_candidates": [
            {
                "score": item["score"],
                "bbox_score": item["bbox_score"],
                "depth_score_m": item["depth_score_m"],
                "multiplier": item["multiplier"],
                "shift_px": item["shift_px"],
                "projected_bbox": item["projected_bbox"],
                "inside_ratio": item["inside_ratio"],
            }
            for item in sorted(evaluations, key=lambda item: item["score"])[:8]
        ],
    }
    return out


def plane_from_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    centered = points - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    signed_dist = centered @ normal
    return center, normal, signed_dist


def acute_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(abs(float(a @ b)), -1.0, 1.0))))


def rotate_points_about_axis(points: np.ndarray, origin: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    centered = points - origin
    cos_t = float(np.cos(angle_rad))
    sin_t = float(np.sin(angle_rad))
    rotated = (
        centered * cos_t
        + np.cross(axis, centered) * sin_t
        + axis * (centered @ axis)[:, None] * (1.0 - cos_t)
    )
    return origin + rotated


def rotate_mesh_about_axis(mesh: trimesh.Trimesh, origin: np.ndarray, axis: np.ndarray, angle_rad: float) -> trimesh.Trimesh:
    out = mesh.copy()
    out.vertices = rotate_points_about_axis(np.asarray(out.vertices, dtype=np.float64), origin, axis, angle_rad)
    return out


def point_to_plane_metrics(points: np.ndarray, plane_center: np.ndarray, plane_normal: np.ndarray, trim_fraction: float) -> dict[str, Any]:
    distances = np.abs((points - plane_center) @ plane_normal)
    return {
        "trimmed_mean_m": trimmed_mean(distances, trim_fraction),
        "median_m": float(np.median(distances)),
        "q90_m": float(np.quantile(distances, 0.90)),
        "q98_m": float(np.quantile(distances, 0.98)),
    }


def distance_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {}
    return {
        "mean_m": float(np.mean(values)),
        "median_m": float(np.median(values)),
        "p75_m": float(np.quantile(values, 0.75)),
        "p90_m": float(np.quantile(values, 0.90)),
        "p95_m": float(np.quantile(values, 0.95)),
        "p98_m": float(np.quantile(values, 0.98)),
    }


def part_observed_fit_metrics(
    mesh: trimesh.Trimesh,
    observed_points: np.ndarray,
    meta: dict[str, Any],
    mask: np.ndarray,
    seed: int,
    sample_count: int = 30000,
) -> dict[str, Any]:
    if observed_points is None or len(observed_points) < 64:
        return {"error": "not_enough_observed_points", "observed_count": int(len(observed_points) if observed_points is not None else 0)}
    mesh_points = sample_mesh_points(mesh, min(sample_count, max(1000, len(mesh.faces))), seed)
    u, v, z = project_right_camera_points(meta, mesh_points)
    height, width = mask.shape
    inside = (z > 1e-6) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    inside_idx = np.flatnonzero(inside)
    if inside_idx.size:
        ui = np.clip(np.round(u[inside_idx]).astype(np.int64), 0, width - 1)
        vi = np.clip(np.round(v[inside_idx]).astype(np.int64), 0, height - 1)
        visible_idx = inside_idx[mask[vi, ui]]
    else:
        visible_idx = np.asarray([], dtype=np.int64)
    visible_points = mesh_points[visible_idx] if visible_idx.size >= 64 else mesh_points

    mesh_to_observed, _ = cKDTree(observed_points).query(visible_points, k=1, workers=-1)
    observed_to_mesh, _ = cKDTree(visible_points).query(observed_points, k=1, workers=-1)

    mesh_center, mesh_normal, mesh_plane_dist = plane_from_points(visible_points)
    obs_center, obs_normal, obs_plane_dist = plane_from_points(observed_points)
    if float(mesh_normal @ obs_normal) < 0.0:
        mesh_normal = -mesh_normal
    plane_offsets = (visible_points - obs_center) @ obs_normal

    def uv_summary(points: np.ndarray) -> dict[str, Any]:
        uu, vv, zz = project_right_camera_points(meta, points)
        valid = (zz > 1e-6) & (uu >= 0) & (uu < width) & (vv >= 0) & (vv < height)
        if valid.sum() < 16:
            return {"valid_count": int(valid.sum())}
        return {
            "valid_count": int(valid.sum()),
            "u_q01_q99": np.quantile(uu[valid], [0.01, 0.99]),
            "v_q01_q99": np.quantile(vv[valid], [0.01, 0.99]),
        }

    return {
        "mesh_sample_count": int(len(mesh_points)),
        "visible_mesh_count": int(len(visible_points)),
        "observed_count": int(len(observed_points)),
        "mesh_to_observed": distance_summary(mesh_to_observed),
        "observed_to_mesh": distance_summary(observed_to_mesh),
        "plane_angle_deg": acute_angle_deg(mesh_normal, obs_normal),
        "plane_offset_median_m": float(np.median(plane_offsets)),
        "plane_offset_mean_m": float(np.mean(plane_offsets)),
        "mesh_plane_residual": distance_summary(np.abs(mesh_plane_dist)),
        "observed_plane_residual": distance_summary(np.abs(obs_plane_dist)),
        "mesh_uv": uv_summary(visible_points),
        "observed_uv": uv_summary(observed_points),
    }


def visible_surface_observed_fit_metrics(
    mesh: trimesh.Trimesh,
    observed_points: np.ndarray,
    meta: dict[str, Any],
    mask: np.ndarray,
    seed: int,
    grid_px: int,
    sample_count: int = 60000,
) -> dict[str, Any]:
    if observed_points is None or len(observed_points) < 64:
        return {"error": "not_enough_observed_points", "observed_count": int(len(observed_points) if observed_points is not None else 0)}
    mesh_points = sample_mesh_points(mesh, min(sample_count, max(2000, len(mesh.faces))), seed)
    visible_points = visible_projected_surface_points(mesh_points, meta, mask, grid_px)
    if len(visible_points) < 64:
        return {
            "error": "not_enough_visible_points",
            "visible_count": int(len(visible_points)),
            "mesh_sample_count": int(len(mesh_points)),
        }
    mesh_to_observed, _ = cKDTree(observed_points).query(visible_points, k=1, workers=-1)
    observed_to_mesh, _ = cKDTree(visible_points).query(observed_points, k=1, workers=-1)
    mesh_center, mesh_normal, mesh_plane_dist = plane_from_points(visible_points)
    obs_center, obs_normal, obs_plane_dist = plane_from_points(observed_points)
    if float(mesh_normal @ obs_normal) < 0.0:
        mesh_normal = -mesh_normal
    plane_offsets = (visible_points - obs_center) @ obs_normal
    return {
        "mesh_sample_count": int(len(mesh_points)),
        "visible_mesh_count": int(len(visible_points)),
        "observed_count": int(len(observed_points)),
        "grid_px": int(grid_px),
        "mesh_to_observed": distance_summary(mesh_to_observed),
        "observed_to_mesh": distance_summary(observed_to_mesh),
        "plane_angle_deg": acute_angle_deg(mesh_normal, obs_normal),
        "plane_offset_median_m": float(np.median(plane_offsets)),
        "plane_offset_mean_m": float(np.mean(plane_offsets)),
        "plane_offset_summary": distance_summary(np.abs(plane_offsets)),
        "visible_mesh_plane_residual": distance_summary(np.abs(mesh_plane_dist)),
        "observed_plane_residual": distance_summary(np.abs(obs_plane_dist)),
    }


def hinge_angle_refine(
    aligned_parts: list[dict[str, Any]],
    camera_joints: list[dict[str, Any]],
    observed_points: np.ndarray,
    observed_pixel_v: np.ndarray,
    screen_label: str,
    base_label: str,
    angle_min_deg: float,
    angle_max_deg: float,
    angle_steps: int,
    trim_fraction: float,
    plane_distance_weight: float,
    nn_weight: float,
    normal_weight_m_per_deg: float,
    angle_regularizer_m_per_deg: float,
    seed: int,
    observed_screen_points: np.ndarray | None = None,
    observed_split_metadata: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    part_by_label = {part["label"]: part for part in aligned_parts}
    if screen_label not in part_by_label or base_label not in part_by_label or not camera_joints:
        return aligned_parts, None

    if observed_screen_points is None:
        split = split_laptop_observed_parts(observed_points, observed_pixel_v, seed)
        observed_screen = split["screen_points"]
    else:
        observed_screen = observed_screen_points
        split = {
            "source": "screen_mask",
            "screen_count": int(len(observed_screen)),
            **(observed_split_metadata or {}),
        }
    if len(observed_screen) < 64:
        return aligned_parts, None
    if len(observed_screen) > 9000:
        observed_screen = subsample_points(observed_screen, 9000, seed + 61)
    observed_screen_center, observed_screen_normal, _ = plane_from_points(observed_screen)
    observed_tree = cKDTree(observed_screen)

    screen_mesh = part_by_label[screen_label]["mesh"]
    base_mesh = part_by_label[base_label]["mesh"]
    screen_samples = sample_mesh_points(screen_mesh, min(14000, max(1000, len(screen_mesh.faces))), seed + 62)
    base_samples = sample_mesh_points(base_mesh, min(10000, max(1000, len(base_mesh.faces))), seed + 63)
    _, initial_screen_normal, _ = plane_from_points(screen_samples)
    _, base_normal, _ = plane_from_points(base_samples)

    joint = camera_joints[0]
    origin = np.asarray(joint["origin_xyz"], dtype=np.float64)
    axis = np.asarray(joint["axis_xyz"], dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    step_count = max(3, int(angle_steps))
    evaluations = []
    best = None
    for angle_deg in np.linspace(angle_min_deg, angle_max_deg, step_count):
        angle_rad = float(np.deg2rad(angle_deg))
        candidate_points = rotate_points_about_axis(screen_samples, origin, axis, angle_rad)
        _, candidate_normal, _ = plane_from_points(candidate_points)
        plane_metrics = point_to_plane_metrics(candidate_points, observed_screen_center, observed_screen_normal, trim_fraction)
        nn_distances, _ = observed_tree.query(candidate_points, k=1, workers=-1)
        nn_score = trimmed_mean(nn_distances, trim_fraction)
        normal_error_deg = acute_angle_deg(candidate_normal, observed_screen_normal)
        base_screen_angle_deg = acute_angle_deg(candidate_normal, base_normal)
        score = float(
            plane_distance_weight * plane_metrics["trimmed_mean_m"]
            + nn_weight * nn_score
            + normal_weight_m_per_deg * normal_error_deg
            + angle_regularizer_m_per_deg * abs(float(angle_deg))
        )
        item = {
            "score": score,
            "angle_deg": float(angle_deg),
            "plane_distance_trimmed_mean_m": plane_metrics["trimmed_mean_m"],
            "plane_distance_median_m": plane_metrics["median_m"],
            "plane_distance_q90_m": plane_metrics["q90_m"],
            "plane_distance_q98_m": plane_metrics["q98_m"],
            "nearest_trimmed_mean_m": float(nn_score),
            "normal_error_deg": float(normal_error_deg),
            "base_screen_angle_deg": float(base_screen_angle_deg),
        }
        evaluations.append(item)
        if best is None or item["score"] < best["score"]:
            best = item

    if best is None:
        return aligned_parts, None

    initial_plane_metrics = point_to_plane_metrics(screen_samples, observed_screen_center, observed_screen_normal, trim_fraction)
    initial_nn_distances, _ = observed_tree.query(screen_samples, k=1, workers=-1)
    initial = {
        "angle_deg": 0.0,
        "plane_distance_trimmed_mean_m": initial_plane_metrics["trimmed_mean_m"],
        "plane_distance_median_m": initial_plane_metrics["median_m"],
        "plane_distance_q90_m": initial_plane_metrics["q90_m"],
        "plane_distance_q98_m": initial_plane_metrics["q98_m"],
        "nearest_trimmed_mean_m": trimmed_mean(initial_nn_distances, trim_fraction),
        "normal_error_deg": acute_angle_deg(initial_screen_normal, observed_screen_normal),
        "base_screen_angle_deg": acute_angle_deg(initial_screen_normal, base_normal),
    }

    refined_parts = []
    angle_rad = float(np.deg2rad(best["angle_deg"]))
    for part in aligned_parts:
        if part["label"] == screen_label:
            refined_parts.append({**part, "mesh": rotate_mesh_about_axis(part["mesh"], origin, axis, angle_rad)})
        else:
            refined_parts.append(part)

    result = {
        "method": "hinge_angle_refined",
        "screen_part_label": screen_label,
        "base_part_label": base_label,
        "joint_name": joint.get("name"),
        "joint_origin_xyz": origin,
        "joint_axis_xyz": axis,
        "chosen_angle_deg": best["angle_deg"],
        "initial": initial,
        "best": best,
        "observed_split": {
            key: value
            for key, value in split.items()
            if key not in {"screen_points", "base_points", "screen_indices", "base_indices"}
        },
        "weights": {
            "plane_distance": float(plane_distance_weight),
            "nearest_neighbor": float(nn_weight),
            "normal_m_per_deg": float(normal_weight_m_per_deg),
            "angle_regularizer_m_per_deg": float(angle_regularizer_m_per_deg),
        },
        "top_candidates": sorted(evaluations, key=lambda item: item["score"])[:8],
    }
    return refined_parts, result


def hinge_moving_part_angle_refine(
    aligned_parts: list[dict[str, Any]],
    camera_joints: list[dict[str, Any]],
    observed_moving_points: np.ndarray,
    moving_label: str,
    fixed_label: str,
    angle_min_deg: float,
    angle_max_deg: float,
    angle_steps: int,
    trim_fraction: float,
    plane_distance_weight: float,
    nn_weight: float,
    normal_weight_m_per_deg: float,
    angle_regularizer_m_per_deg: float,
    seed: int,
    observed_metadata: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    part_by_label = {part["label"]: part for part in aligned_parts}
    if moving_label not in part_by_label or fixed_label not in part_by_label or not camera_joints:
        return aligned_parts, None
    if observed_moving_points is None or len(observed_moving_points) < 64:
        return aligned_parts, None

    observed_moving = observed_moving_points
    if len(observed_moving) > 9000:
        observed_moving = subsample_points(observed_moving, 9000, seed + 71)
    observed_center, observed_normal, _ = plane_from_points(observed_moving)
    observed_tree = cKDTree(observed_moving)

    moving_mesh = part_by_label[moving_label]["mesh"]
    fixed_mesh = part_by_label[fixed_label]["mesh"]
    moving_samples = sample_mesh_points(moving_mesh, min(14000, max(1000, len(moving_mesh.faces))), seed + 72)
    fixed_samples = sample_mesh_points(fixed_mesh, min(10000, max(1000, len(fixed_mesh.faces))), seed + 73)
    _, initial_moving_normal, _ = plane_from_points(moving_samples)
    _, fixed_normal, _ = plane_from_points(fixed_samples)

    joint = camera_joints[0]
    origin = np.asarray(joint["origin_xyz"], dtype=np.float64)
    axis = np.asarray(joint["axis_xyz"], dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    step_count = max(3, int(angle_steps))
    evaluations = []
    best = None
    for angle_deg in np.linspace(angle_min_deg, angle_max_deg, step_count):
        angle_rad = float(np.deg2rad(angle_deg))
        candidate_points = rotate_points_about_axis(moving_samples, origin, axis, angle_rad)
        _, candidate_normal, _ = plane_from_points(candidate_points)
        plane_metrics = point_to_plane_metrics(candidate_points, observed_center, observed_normal, trim_fraction)
        nn_distances, _ = observed_tree.query(candidate_points, k=1, workers=-1)
        nn_score = trimmed_mean(nn_distances, trim_fraction)
        normal_error_deg = acute_angle_deg(candidate_normal, observed_normal)
        fixed_moving_angle_deg = acute_angle_deg(candidate_normal, fixed_normal)
        score = float(
            plane_distance_weight * plane_metrics["trimmed_mean_m"]
            + nn_weight * nn_score
            + normal_weight_m_per_deg * normal_error_deg
            + angle_regularizer_m_per_deg * abs(float(angle_deg))
        )
        item = {
            "score": score,
            "angle_deg": float(angle_deg),
            "plane_distance_trimmed_mean_m": plane_metrics["trimmed_mean_m"],
            "plane_distance_median_m": plane_metrics["median_m"],
            "plane_distance_q90_m": plane_metrics["q90_m"],
            "plane_distance_q98_m": plane_metrics["q98_m"],
            "nearest_trimmed_mean_m": float(nn_score),
            "normal_error_deg": float(normal_error_deg),
            "fixed_moving_angle_deg": float(fixed_moving_angle_deg),
        }
        evaluations.append(item)
        if best is None or item["score"] < best["score"]:
            best = item

    if best is None:
        return aligned_parts, None

    initial_plane_metrics = point_to_plane_metrics(moving_samples, observed_center, observed_normal, trim_fraction)
    initial_nn_distances, _ = observed_tree.query(moving_samples, k=1, workers=-1)
    initial = {
        "angle_deg": 0.0,
        "plane_distance_trimmed_mean_m": initial_plane_metrics["trimmed_mean_m"],
        "plane_distance_median_m": initial_plane_metrics["median_m"],
        "plane_distance_q90_m": initial_plane_metrics["q90_m"],
        "plane_distance_q98_m": initial_plane_metrics["q98_m"],
        "nearest_trimmed_mean_m": trimmed_mean(initial_nn_distances, trim_fraction),
        "normal_error_deg": acute_angle_deg(initial_moving_normal, observed_normal),
        "fixed_moving_angle_deg": acute_angle_deg(initial_moving_normal, fixed_normal),
    }

    refined_parts = []
    angle_rad = float(np.deg2rad(best["angle_deg"]))
    for part in aligned_parts:
        if part["label"] == moving_label:
            refined_parts.append({**part, "mesh": rotate_mesh_about_axis(part["mesh"], origin, axis, angle_rad)})
        else:
            refined_parts.append(part)

    result = {
        "method": "hinge_moving_part_angle_refined",
        "moving_part_label": moving_label,
        "fixed_part_label": fixed_label,
        "joint_name": joint.get("name"),
        "joint_origin_xyz": origin,
        "joint_axis_xyz": axis,
        "chosen_angle_deg": best["angle_deg"],
        "initial": initial,
        "best": best,
        "observed_target": {
            "source": "part_mask",
            "count": int(len(observed_moving_points)),
            **(observed_metadata or {}),
        },
        "weights": {
            "plane_distance": float(plane_distance_weight),
            "nearest_neighbor": float(nn_weight),
            "normal_m_per_deg": float(normal_weight_m_per_deg),
            "angle_regularizer_m_per_deg": float(angle_regularizer_m_per_deg),
        },
        "top_candidates": sorted(evaluations, key=lambda item: item["score"])[:8],
    }
    return refined_parts, result


def hinge_visible_surface_angle_refine(
    aligned_parts: list[dict[str, Any]],
    camera_joints: list[dict[str, Any]],
    observed_moving_points: np.ndarray,
    meta: dict[str, Any],
    mask: np.ndarray,
    moving_label: str,
    fixed_label: str,
    angle_min_deg: float,
    angle_max_deg: float,
    angle_steps: int,
    trim_fraction: float,
    plane_distance_weight: float,
    nn_weight: float,
    normal_weight_m_per_deg: float,
    visible_normal_weight_m_per_deg: float,
    angle_regularizer_m_per_deg: float,
    observed_to_model_weight: float,
    plane_offset_weight: float,
    snap_offset: bool,
    grid_px: int,
    seed: int,
    observed_metadata: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    part_by_label = {part["label"]: part for part in aligned_parts}
    if moving_label not in part_by_label or fixed_label not in part_by_label or not camera_joints:
        return aligned_parts, None
    if observed_moving_points is None or len(observed_moving_points) < 64:
        return aligned_parts, None

    observed_moving = observed_moving_points
    if len(observed_moving) > 9000:
        observed_moving = subsample_points(observed_moving, 9000, seed + 81)
    observed_center, observed_normal, observed_plane_dist = plane_from_points(observed_moving)
    observed_tree = cKDTree(observed_moving)

    moving_mesh = part_by_label[moving_label]["mesh"]
    fixed_mesh = part_by_label[fixed_label]["mesh"]
    moving_samples = sample_mesh_points(moving_mesh, min(24000, max(2000, len(moving_mesh.faces))), seed + 82)
    fixed_samples = sample_mesh_points(fixed_mesh, min(10000, max(1000, len(fixed_mesh.faces))), seed + 83)
    _, fixed_normal, _ = plane_from_points(fixed_samples)

    joint = camera_joints[0]
    origin = np.asarray(joint["origin_xyz"], dtype=np.float64)
    axis = np.asarray(joint["axis_xyz"], dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    step_count = max(3, int(angle_steps))
    evaluations = []
    best = None

    def evaluate(angle_deg: float, candidate_points: np.ndarray) -> dict[str, Any] | None:
        visible_points = visible_projected_surface_points(candidate_points, meta, mask, grid_px)
        if len(visible_points) < 128:
            return None
        _, candidate_normal, candidate_plane_dist = plane_from_points(visible_points)
        if float(candidate_normal @ observed_normal) < 0.0:
            candidate_normal = -candidate_normal
            candidate_plane_dist = -candidate_plane_dist
        signed_plane_offsets = (visible_points - observed_center) @ observed_normal
        plane_abs = np.abs(signed_plane_offsets)
        nn_distances, _ = observed_tree.query(visible_points, k=1, workers=-1)
        visible_tree = cKDTree(visible_points)
        obs_to_visible_distances, _ = visible_tree.query(observed_moving, k=1, workers=-1)
        plane_score = trimmed_mean(plane_abs, trim_fraction)
        nn_score = trimmed_mean(nn_distances, trim_fraction)
        obs_score = trimmed_mean(obs_to_visible_distances, trim_fraction)
        normal_error_deg = acute_angle_deg(candidate_normal, observed_normal)
        fixed_moving_angle_deg = acute_angle_deg(candidate_normal, fixed_normal)
        plane_offset_median = float(np.median(signed_plane_offsets))
        plane_offset_score = abs(plane_offset_median)
        score = float(
            plane_distance_weight * plane_score
            + nn_weight * nn_score
            + observed_to_model_weight * obs_score
            + visible_normal_weight_m_per_deg * normal_error_deg
            + plane_offset_weight * plane_offset_score
            + angle_regularizer_m_per_deg * abs(float(angle_deg))
        )
        return {
            "score": score,
            "angle_deg": float(angle_deg),
            "visible_count": int(len(visible_points)),
            "plane_distance_trimmed_mean_m": float(plane_score),
            "plane_distance_median_m": float(np.median(plane_abs)),
            "plane_distance_q90_m": float(np.quantile(plane_abs, 0.90)),
            "plane_distance_q98_m": float(np.quantile(plane_abs, 0.98)),
            "plane_offset_median_m": plane_offset_median,
            "nearest_trimmed_mean_m": float(nn_score),
            "observed_to_visible_trimmed_mean_m": float(obs_score),
            "normal_error_deg": float(normal_error_deg),
            "fixed_moving_angle_deg": float(fixed_moving_angle_deg),
            "visible_plane_residual": distance_summary(np.abs(candidate_plane_dist)),
            "observed_plane_residual": distance_summary(np.abs(observed_plane_dist)),
        }

    for angle_deg in np.linspace(angle_min_deg, angle_max_deg, step_count):
        angle_rad = float(np.deg2rad(angle_deg))
        candidate_points = rotate_points_about_axis(moving_samples, origin, axis, angle_rad)
        item = evaluate(float(angle_deg), candidate_points)
        if item is None:
            continue
        evaluations.append(item)
        if best is None or item["score"] < best["score"]:
            best = item

    if best is None:
        return aligned_parts, None

    initial = evaluate(0.0, moving_samples)
    angle_rad = float(np.deg2rad(best["angle_deg"]))
    snapped_offset = 0.0
    snap_translation = np.zeros(3, dtype=np.float64)
    if snap_offset:
        rotated_samples = rotate_points_about_axis(moving_samples, origin, axis, angle_rad)
        visible_points = visible_projected_surface_points(rotated_samples, meta, mask, grid_px)
        if len(visible_points) >= 128:
            signed_offsets = (visible_points - observed_center) @ observed_normal
            snapped_offset = float(np.median(signed_offsets))
            snap_translation = -snapped_offset * observed_normal

    refined_parts = []
    for part in aligned_parts:
        if part["label"] == moving_label:
            rotated_mesh = rotate_mesh_about_axis(part["mesh"], origin, axis, angle_rad)
            if np.linalg.norm(snap_translation) > 0.0:
                rotated_mesh = rotated_mesh.copy()
                rotated_mesh.vertices = np.asarray(rotated_mesh.vertices, dtype=np.float64) + snap_translation
            refined_parts.append({**part, "mesh": rotated_mesh})
        else:
            refined_parts.append(part)

    result = {
        "method": "hinge_visible_surface_angle_refined",
        "moving_part_label": moving_label,
        "fixed_part_label": fixed_label,
        "joint_name": joint.get("name"),
        "joint_origin_xyz": origin,
        "joint_axis_xyz": axis,
        "chosen_angle_deg": best["angle_deg"],
        "initial": initial,
        "best": best,
        "observed_target": {
            "source": "part_mask",
            "count": int(len(observed_moving_points)),
            **(observed_metadata or {}),
        },
        "weights": {
            "plane_distance": float(plane_distance_weight),
            "nearest_neighbor": float(nn_weight),
            "observed_to_visible": float(observed_to_model_weight),
            "normal_m_per_deg": float(visible_normal_weight_m_per_deg),
            "plane_offset": float(plane_offset_weight),
            "angle_regularizer_m_per_deg": float(angle_regularizer_m_per_deg),
        },
        "visible_surface": {
            "grid_px": int(grid_px),
            "sample_count": int(len(moving_samples)),
            "snap_offset": bool(snap_offset),
            "snapped_plane_offset_m": snapped_offset,
            "snap_translation_xyz": snap_translation,
        },
        "top_candidates": sorted(evaluations, key=lambda item: item["score"])[:8],
    }
    return refined_parts, result


def split_laptop_observed_parts(points: np.ndarray, pixel_v: np.ndarray, seed: int) -> dict[str, Any]:
    if len(points) < 64:
        raise ValueError("Need at least 64 observed laptop points for part-aware alignment.")
    centers = np.quantile(pixel_v, [0.33, 0.67]).astype(np.float64)
    for _ in range(30):
        labels = np.argmin(np.abs(pixel_v[:, None] - centers[None, :]), axis=1)
        next_centers = centers.copy()
        for idx in range(2):
            if np.any(labels == idx):
                next_centers[idx] = float(pixel_v[labels == idx].mean())
        if np.allclose(next_centers, centers):
            break
        centers = next_centers

    centers = np.sort(centers)
    split_v = float(0.5 * (centers[0] + centers[1]))
    screen_mask = pixel_v < split_v
    base_mask = ~screen_mask
    if screen_mask.sum() < 64 or base_mask.sum() < 64:
        split_v = float(np.median(pixel_v))
        screen_mask = pixel_v < split_v
        base_mask = ~screen_mask

    screen_points = points[screen_mask]
    base_points = points[base_mask]
    return {
        "screen_points": screen_points,
        "base_points": base_points,
        "screen_indices": np.flatnonzero(screen_mask),
        "base_indices": np.flatnonzero(base_mask),
        "split_v": split_v,
        "cluster_centers_v": centers,
        "screen_count": int(len(screen_points)),
        "base_count": int(len(base_points)),
        "screen_bounds": [screen_points.min(axis=0).tolist(), screen_points.max(axis=0).tolist()],
        "base_bounds": [base_points.min(axis=0).tolist(), base_points.max(axis=0).tolist()],
        "screen_extents": np.ptp(screen_points, axis=0).tolist(),
        "base_extents": np.ptp(base_points, axis=0).tolist(),
        "screen_depth_median_m": float(np.median(screen_points[:, 2])),
        "base_depth_median_m": float(np.median(base_points[:, 2])),
        "seed": seed,
    }


def uv_depth_summary(meta: dict[str, Any], points: np.ndarray) -> dict[str, Any]:
    if len(points) == 0:
        return {"count": 0}
    u, v, z = project_right_camera_points(meta, points)
    width = int(meta["rgb_width_per_eye"])
    height = int(meta["rgb_height_per_eye"])
    inside = (z > 1e-6) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if not np.any(inside):
        return {"count": int(len(points)), "inside_ratio": 0.0}
    return {
        "count": int(len(points)),
        "inside_ratio": float(inside.mean()),
        "u_quantiles": np.quantile(u[inside], [0.02, 0.1, 0.5, 0.9, 0.98]).tolist(),
        "v_quantiles": np.quantile(v[inside], [0.02, 0.1, 0.5, 0.9, 0.98]).tolist(),
        "z_quantiles_m": np.quantile(z[inside], [0.02, 0.1, 0.5, 0.9, 0.98]).tolist(),
    }


def part_aware_score(
    part_points: dict[str, np.ndarray],
    observed_parts: dict[str, np.ndarray],
    meta: dict[str, Any],
    scale: float,
    rot: np.ndarray,
    trans: np.ndarray,
    screen_label: str,
    base_label: str,
    trim_fraction: float,
) -> dict[str, Any]:
    transformed_screen = apply_sim3(part_points[screen_label], scale, rot, trans)
    transformed_base = apply_sim3(part_points[base_label], scale, rot, trans)
    screen_target = observed_parts["screen"]
    base_target = observed_parts["base"]

    screen_dists, _ = cKDTree(screen_target).query(transformed_screen, k=1, workers=-1)
    base_dists, _ = cKDTree(base_target).query(transformed_base, k=1, workers=-1)
    screen_score = trimmed_mean(screen_dists, trim_fraction)
    base_score = trimmed_mean(base_dists, trim_fraction)

    screen_uv = uv_depth_summary(meta, transformed_screen)
    base_uv = uv_depth_summary(meta, transformed_base)
    semantic_penalty = 0.0
    screen_v = screen_uv.get("v_quantiles", [float("inf"), float("inf"), float("inf")])[2]
    base_v = base_uv.get("v_quantiles", [float("-inf"), float("-inf"), float("-inf")])[2]
    screen_z = screen_uv.get("z_quantiles_m", [float("inf"), float("inf"), float("inf")])[2]
    base_z = base_uv.get("z_quantiles_m", [float("-inf"), float("-inf"), float("-inf")])[2]
    if not screen_v < base_v:
        semantic_penalty += 1.0
    if not screen_z > base_z:
        semantic_penalty += 0.20
    if screen_uv.get("inside_ratio", 0.0) < 0.65:
        semantic_penalty += 0.20
    if base_uv.get("inside_ratio", 0.0) < 0.65:
        semantic_penalty += 0.20

    score = 0.5 * (screen_score + base_score) + semantic_penalty
    return {
        "score": float(score),
        "screen_score_m": float(screen_score),
        "base_score_m": float(base_score),
        "semantic_penalty": float(semantic_penalty),
        "screen_projection": screen_uv,
        "base_projection": base_uv,
    }


def refine_part_aware_sim3(
    part_points: dict[str, np.ndarray],
    observed_parts: dict[str, np.ndarray],
    meta: dict[str, Any],
    initial: dict[str, Any],
    screen_label: str,
    base_label: str,
    trim_fraction: float,
    iterations: int,
) -> dict[str, Any]:
    source = np.concatenate([part_points[screen_label], part_points[base_label]], axis=0)
    target_screen = observed_parts["screen"]
    target_base = observed_parts["base"]
    screen_tree = cKDTree(target_screen)
    base_tree = cKDTree(target_base)
    screen_n = len(part_points[screen_label])

    scale = float(initial["scale"])
    rot = np.asarray(initial["rotation"], dtype=np.float64)
    trans = np.asarray(initial["translation"], dtype=np.float64)
    scale_min = scale * 0.65
    scale_max = scale * 1.35
    history = []
    best_metrics = part_aware_score(
        part_points, observed_parts, meta, scale, rot, trans, screen_label, base_label, trim_fraction
    )
    best = {"scale": scale, "rotation": rot, "translation": trans, **best_metrics}

    for _ in range(iterations):
        screen_transformed = apply_sim3(part_points[screen_label], scale, rot, trans)
        base_transformed = apply_sim3(part_points[base_label], scale, rot, trans)
        screen_dists, screen_indices = screen_tree.query(screen_transformed, k=1, workers=-1)
        base_dists, base_indices = base_tree.query(base_transformed, k=1, workers=-1)

        screen_keep = max(32, int(len(screen_dists) * trim_fraction))
        base_keep = max(32, int(len(base_dists) * trim_fraction))
        screen_keep_idx = np.argpartition(screen_dists, screen_keep - 1)[:screen_keep]
        base_keep_idx = np.argpartition(base_dists, base_keep - 1)[:base_keep]
        source_corr = np.concatenate(
            [part_points[screen_label][screen_keep_idx], part_points[base_label][base_keep_idx]],
            axis=0,
        )
        target_corr = np.concatenate(
            [target_screen[screen_indices[screen_keep_idx]], target_base[base_indices[base_keep_idx]]],
            axis=0,
        )

        new_scale, new_rot, new_trans = umeyama_sim3(source_corr, target_corr, allow_scaling=True)
        new_scale = float(np.clip(new_scale, scale_min, scale_max))
        metrics = part_aware_score(
            part_points, observed_parts, meta, new_scale, new_rot, new_trans, screen_label, base_label, trim_fraction
        )
        history.append(metrics["score"])
        scale, rot, trans = new_scale, new_rot, new_trans
        if metrics["score"] < best["score"]:
            best = {"scale": scale, "rotation": rot, "translation": trans, **metrics}
        if len(history) > 2 and abs(history[-1] - history[-2]) < 1e-6:
            break

    best["matrix"] = sim3_matrix(best["scale"], best["rotation"], best["translation"])
    best["iterations"] = len(history)
    best["history"] = history
    best["source_count"] = int(len(source))
    best["screen_source_count"] = int(screen_n)
    best["base_source_count"] = int(len(source) - screen_n)
    return best


def estimate_part_aware_alignment(
    parts: list[dict[str, Any]],
    observed_points: np.ndarray,
    observed_pixel_v: np.ndarray,
    meta: dict[str, Any],
    screen_label: str,
    base_label: str,
    seed: int,
    candidate_count: int,
    trim_fraction: float,
    iterations: int,
) -> dict[str, Any]:
    part_by_label = {part["label"]: part for part in parts}
    if screen_label not in part_by_label or base_label not in part_by_label:
        raise KeyError(f"Missing required laptop part labels: screen={screen_label} base={base_label}")

    split = split_laptop_observed_parts(observed_points, observed_pixel_v, seed)
    screen_target = subsample_points(split["screen_points"], min(len(split["screen_points"]), 8000), seed + 11)
    base_target = subsample_points(split["base_points"], min(len(split["base_points"]), 8000), seed + 12)
    observed_parts = {"screen": screen_target, "base": base_target}
    part_points = {
        screen_label: sample_mesh_points(part_by_label[screen_label]["mesh"], 9000, seed + 21),
        base_label: sample_mesh_points(part_by_label[base_label]["mesh"], 9000, seed + 22),
    }
    source_all = np.concatenate([part_points[screen_label], part_points[base_label]], axis=0)
    target_all = np.concatenate([screen_target, base_target], axis=0)
    candidates = initial_sim3_candidates(
        subsample_points(source_all, min(len(source_all), 16000), seed + 31),
        subsample_points(target_all, min(len(target_all), 12000), seed + 32),
        trim_fraction,
    )[:candidate_count]

    refined = []
    for candidate in candidates:
        candidate_metrics = part_aware_score(
            part_points,
            observed_parts,
            meta,
            candidate["scale"],
            candidate["rotation"],
            candidate["translation"],
            screen_label,
            base_label,
            trim_fraction,
        )
        candidate_with_metrics = {**candidate, **candidate_metrics}
        refined.append(
            refine_part_aware_sim3(
                part_points,
                observed_parts,
                meta,
                candidate_with_metrics,
                screen_label,
                base_label,
                trim_fraction,
                iterations,
            )
        )

    refined.sort(key=lambda item: item["score"])
    best = refined[0]
    best["method"] = "part_aware"
    best["screen_part_label"] = screen_label
    best["base_part_label"] = base_label
    best["observed_part_split"] = {
        key: value
        for key, value in split.items()
        if key not in {"screen_points", "base_points", "screen_indices", "base_indices"}
    }
    best["candidate_scores"] = [
        {
            "score": float(item["score"]),
            "screen_score_m": float(item["screen_score_m"]),
            "base_score_m": float(item["base_score_m"]),
            "semantic_penalty": float(item["semantic_penalty"]),
            "scale": float(item["scale"]),
        }
        for item in refined[: min(8, len(refined))]
    ]
    return best


def apply_sim3_to_mesh(mesh: trimesh.Trimesh, scale: float, rot: np.ndarray, trans: np.ndarray) -> trimesh.Trimesh:
    out = mesh.copy()
    out.vertices = apply_sim3(np.asarray(out.vertices, dtype=np.float64), scale, rot, trans)
    return out


def apply_se3_to_mesh(mesh: trimesh.Trimesh, transform: np.ndarray) -> trimesh.Trimesh:
    out = mesh.copy()
    out.vertices = transform_points(np.asarray(out.vertices, dtype=np.float64), transform)
    return out


def export_point_cloud(path: Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pc = trimesh.PointCloud(points, colors=colors)
    pc.export(path)


def export_colored_scene(path: Path, parts: list[dict[str, Any]]) -> None:
    scene = trimesh.Scene()
    for idx, part in enumerate(parts):
        mesh = part["mesh"].copy()
        color = PART_COLORS.get(part["label"], (180, 180, 180, 255))
        mesh.visual.vertex_colors = np.tile(np.asarray(color, dtype=np.uint8), (len(mesh.vertices), 1))
        scene.add_geometry(mesh, node_name=f"part_{part['label']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(path)


def transform_joint(joint: dict[str, Any], scale: float, rot: np.ndarray, trans: np.ndarray) -> dict[str, Any]:
    origin = apply_sim3(joint["world_origin"][None, :], scale, rot, trans)[0]
    axis = rot @ joint["axis"]
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    return {
        "name": joint["name"],
        "type": joint["type"],
        "parent": joint.get("parent"),
        "child": joint.get("child"),
        "origin_xyz": origin,
        "axis_xyz": axis,
        "limit": joint.get("limit", {}),
    }


def transform_joint_se3(joint: dict[str, Any], transform: np.ndarray) -> dict[str, Any]:
    origin = transform_points(np.asarray(joint["origin_xyz"])[None, :], transform)[0]
    axis = transform_vectors(np.asarray(joint["axis_xyz"])[None, :], transform)[0]
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    out = dict(joint)
    out["origin_xyz"] = origin
    out["axis_xyz"] = axis
    return out


def write_obj(path: Path, mesh: trimesh.Trimesh) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)


def find_first_hand_frame(egoforce_dir: Path) -> int | None:
    frames = []
    for path in egoforce_dir.glob("*_left_hand.obj"):
        try:
            frames.append(int(path.name.split("_", 1)[0]))
        except ValueError:
            pass
    return min(frames) if frames else None


def run_laptop_alignment(config: AlignmentConfig) -> dict[str, Any]:
    project_root = config.project_root.resolve()
    export_root = config.export_root.resolve()
    output_root = config.output_root or (project_root / "outputs" / "object_alignment")
    out_dir = output_root / config.target_id / f"frame_{frame_name(config.align_frame)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = read_json(export_root / "manifest.json")
    manifest = read_json(project_root / "outputs" / "project_manifest.json")
    target = next(item for item in manifest["targets"] if item["object_id"] == config.target_id)
    mask_item = next(item for item in manifest["masks"] if item["mask_id"] == target["selected_mask_id"])

    rgb_path = export_root / "rgb_right_png" / f"{frame_name(config.align_frame)}.png"
    depth_path = export_root / "depth_meters_npy" / f"{frame_name(config.align_frame)}.meters.npy"
    mask_path = Path(mask_item["mask_npy"])
    depth_m = np.load(depth_path)
    mask = np.load(mask_path).astype(bool)
    alignment_mode = "free_icp" if config.final_alignment_mode == "refined" else config.final_alignment_mode
    part_mask_alignment_modes = {"base_first", "screen_first"}
    semantic_base_label = (
        config.screen_first_base_part_label if alignment_mode == "screen_first" else config.base_first_base_part_label
    )
    semantic_screen_label = (
        config.screen_first_screen_part_label if alignment_mode == "screen_first" else config.base_first_screen_part_label
    )

    base_mask_path = None
    base_mask = None
    base_footprint_mask = None
    screen_mask = None
    screen_projection_mask = None
    if alignment_mode in part_mask_alignment_modes or config.base_mask_path is not None:
        base_mask_path = resolve_base_mask_path(config, project_root)
        base_mask_raw = np.load(base_mask_path).astype(bool)
        if base_mask_raw.shape != mask.shape:
            raise ValueError(f"Base mask shape {base_mask_raw.shape} does not match whole mask shape {mask.shape}.")
        base_mask = base_mask_raw & mask
        base_footprint_mask = fill_mask_holes(base_mask) & mask
        screen_mask = mask & ~base_footprint_mask
        screen_projection_mask = quantile_crop_mask(
            screen_mask,
            config.silhouette_quantile_min,
            config.silhouette_quantile_max,
            pad_px=6,
        )
        if int(base_mask.sum()) < 64:
            raise ValueError(f"Base mask has too few pixels after clipping to whole object: {int(base_mask.sum())}")
        if int(screen_mask.sum()) < 64:
            raise ValueError(f"Screen mask has too few pixels after subtracting filled base footprint: {int(screen_mask.sum())}")
        if int(screen_projection_mask.sum()) < 64:
            raise ValueError(
                f"Screen projection mask has too few pixels after quantile crop: {int(screen_projection_mask.sum())}"
            )

    validation = {}
    for convention in ("camera_to_rig", "rig_to_camera", "direct_same_camera"):
        points_right, u, v, inside = depth_points_in_right_camera(
            meta, depth_m, convention, config.depth_min_m, config.depth_max_m
        )
        overlay_path = out_dir / f"depth_to_rgb_overlay_{convention}.png"
        validation[convention] = save_projection_overlay(rgb_path, overlay_path, points_right, u, v, inside, mask=mask)
        valid_idx = np.flatnonzero(inside)
        if len(valid_idx):
            ui = np.clip(np.round(u[valid_idx]).astype(np.int64), 0, int(meta["rgb_width_per_eye"]) - 1)
            vi = np.clip(np.round(v[valid_idx]).astype(np.int64), 0, int(meta["rgb_height_per_eye"]) - 1)
            validation[convention]["mask_overlap_ratio"] = float(mask[vi, ui].mean())

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
        observed_idx = rng.choice(len(observed_points), size=config.observed_samples, replace=False)
        observed_points = observed_points[observed_idx]
        observed_u = observed_u[observed_idx]
        observed_v = observed_v[observed_idx]
    observed_colors = color_by_depth(observed_points[:, 2])
    export_point_cloud(out_dir / "observed_mask_pointcloud.ply", observed_points, observed_colors)

    observed_base_points = observed_base_u = observed_base_v = observed_base_stats = None
    observed_screen_points = observed_screen_u = observed_screen_v = observed_screen_stats = None
    base_screen_mask_outputs = None
    if base_mask is not None and screen_mask is not None:
        base_screen_mask_outputs = {
            "base": save_binary_mask(base_mask, out_dir / "part_masks" / f"{config.target_id}_frame_{config.align_frame}_base"),
            "base_footprint": save_binary_mask(
                base_footprint_mask,
                out_dir / "part_masks" / f"{config.target_id}_frame_{config.align_frame}_base_footprint",
            ),
            "screen": save_binary_mask(screen_mask, out_dir / "part_masks" / f"{config.target_id}_frame_{config.align_frame}_screen"),
            "screen_projection": save_binary_mask(
                screen_projection_mask,
                out_dir / "part_masks" / f"{config.target_id}_frame_{config.align_frame}_screen_projection",
            ),
        }
        (
            observed_base_points,
            observed_base_u,
            observed_base_v,
            observed_base_stats,
        ) = observed_mask_cloud_with_pixels(
            meta,
            depth_m,
            base_mask,
            config.convention,
            config.depth_min_m,
            config.depth_max_m,
            config.depth_quantile_min,
            config.depth_quantile_max,
        )
        (
            observed_screen_points,
            observed_screen_u,
            observed_screen_v,
            observed_screen_stats,
        ) = observed_mask_cloud_with_pixels(
            meta,
            depth_m,
            screen_mask,
            config.convention,
            config.depth_min_m,
            config.depth_max_m,
            config.depth_quantile_min,
            config.depth_quantile_max,
        )
        if len(observed_base_points) > config.observed_samples:
            rng = np.random.default_rng(config.random_seed + 101)
            base_idx = rng.choice(len(observed_base_points), size=config.observed_samples, replace=False)
            observed_base_points = observed_base_points[base_idx]
            observed_base_u = observed_base_u[base_idx]
            observed_base_v = observed_base_v[base_idx]
        if len(observed_screen_points) > config.observed_samples:
            rng = np.random.default_rng(config.random_seed + 102)
            screen_idx = rng.choice(len(observed_screen_points), size=config.observed_samples, replace=False)
            observed_screen_points = observed_screen_points[screen_idx]
            observed_screen_u = observed_screen_u[screen_idx]
            observed_screen_v = observed_screen_v[screen_idx]
        base_colors = np.tile(
            np.asarray(PART_COLORS.get(semantic_base_label, (75, 145, 210, 255)), dtype=np.uint8),
            (len(observed_base_points), 1),
        )
        screen_colors = np.tile(
            np.asarray(PART_COLORS.get(semantic_screen_label, (220, 50, 65, 255)), dtype=np.uint8),
            (len(observed_screen_points), 1),
        )
        export_point_cloud(out_dir / "observed_base_pointcloud.ply", observed_base_points, base_colors)
        export_point_cloud(out_dir / "observed_screen_pointcloud.ply", observed_screen_points, screen_colors)

    particulate_run_path = config.particulate_run_path or (
        project_root / "outputs" / "particulate" / "target_laptop_decimated_50000" / "particulate_run.json"
    )
    part_run = read_json(particulate_run_path)
    urdf_path = Path(part_run["outputs"]["model.urdf"])
    parts, joints = load_particulate_parts(urdf_path)
    part_by_label = {part["label"]: part for part in parts}
    canonical_mesh = concatenate_meshes(parts)
    canonical_points = sample_mesh_points(canonical_mesh, config.canonical_samples, config.random_seed)
    export_point_cloud(out_dir / "canonical_sample_pointcloud.ply", canonical_points)
    export_colored_scene(out_dir / "canonical_parts.glb", parts)

    base_canonical_points = None
    screen_canonical_points = None
    if alignment_mode in part_mask_alignment_modes:
        if observed_base_points is None or base_mask is None:
            raise ValueError(f"{alignment_mode} alignment requires a base mask and observed base point cloud.")
        if semantic_base_label not in part_by_label:
            raise KeyError(f"Base part label {semantic_base_label!r} not found in Particulate parts.")
        if semantic_screen_label not in part_by_label:
            raise KeyError(f"Screen part label {semantic_screen_label!r} not found in Particulate parts.")
        base_part = part_by_label[semantic_base_label]
        screen_part = part_by_label[semantic_screen_label]
        base_canonical_points = sample_mesh_points(
            base_part["mesh"],
            min(config.canonical_samples, max(4000, len(base_part["mesh"].faces))),
            config.random_seed + 201,
        )
        screen_canonical_points = sample_mesh_points(
            screen_part["mesh"],
            min(config.canonical_samples, max(4000, len(screen_part["mesh"].faces))),
            config.random_seed + 202,
        )
        export_point_cloud(out_dir / "canonical_base_sample_pointcloud.ply", base_canonical_points)
        export_point_cloud(out_dir / "canonical_screen_sample_pointcloud.ply", screen_canonical_points)

    pca_direct_alignment = estimate_pca_direct_alignment(
        parts,
        canonical_points,
        observed_points,
        meta,
        mask,
        seed=config.random_seed,
        trim_fraction=config.icp_trim_fraction,
        candidate_count=config.pca_direct_candidates,
        screen_label=config.pca_direct_screen_part_label,
        base_label=config.pca_direct_base_part_label,
        require_semantic_order=config.pca_direct_require_semantic_order,
        grid_px=config.visible_grid_px,
    )
    coarse_alignment = None
    coarse_visible_score = None
    visible_alignment = None
    constrained_alignment = None
    base_first_alignment = None
    base_pca_alignment = None
    screen_first_alignment = None
    screen_pca_alignment = None
    if alignment_mode == "base_first":
        base_pca_alignment = estimate_base_first_initial_alignment(
            part_by_label[config.base_first_base_part_label],
            part_by_label[config.base_first_screen_part_label],
            base_canonical_points,
            screen_canonical_points,
            observed_base_points,
            observed_screen_points,
            meta,
            base_mask,
            screen_mask,
            seed=config.random_seed + 211,
            trim_fraction=config.icp_trim_fraction,
            candidate_count=config.pca_direct_candidates,
            base_label=config.base_first_base_part_label,
            screen_label=config.base_first_screen_part_label,
            grid_px=config.visible_grid_px,
        )
        alignment = base_pca_alignment
        if config.constrained_refine:
            constrained_alignment = constrained_visible_refine(
                base_canonical_points,
                observed_base_points,
                meta,
                base_mask,
                initial=alignment,
                grid_px=config.visible_grid_px,
                trim_fraction=config.constrained_trim_fraction,
                iterations=config.constrained_iterations,
                scale_min_multiplier=config.constrained_scale_min_multiplier,
                scale_max_multiplier=config.constrained_scale_max_multiplier,
                rotation_max_deg=config.constrained_rotation_max_deg,
            )
            alignment = constrained_alignment
        base_first_alignment = {
            "method": "base_first",
            "base_part_label": config.base_first_base_part_label,
            "screen_part_label": config.base_first_screen_part_label,
            "base_mask_path": str(base_mask_path) if base_mask_path else None,
            "base_mask_area_pixels": int(base_mask.sum()) if base_mask is not None else None,
            "screen_mask_area_pixels": int(screen_mask.sum()) if screen_mask is not None else None,
            "base_pca_initial": base_pca_alignment,
            "base_constrained_refine": constrained_alignment,
        }
    elif alignment_mode == "screen_first":
        screen_pca_alignment = estimate_screen_first_initial_alignment(
            part_by_label[config.screen_first_screen_part_label],
            part_by_label[config.screen_first_base_part_label],
            screen_canonical_points,
            base_canonical_points,
            observed_screen_points,
            observed_base_points,
            meta,
            screen_mask,
            base_mask,
            seed=config.random_seed + 221,
            trim_fraction=config.icp_trim_fraction,
            candidate_count=config.pca_direct_candidates,
            screen_label=config.screen_first_screen_part_label,
            base_label=config.screen_first_base_part_label,
            grid_px=config.visible_grid_px,
        )
        alignment = screen_pca_alignment
        if config.constrained_refine:
            constrained_alignment = constrained_visible_refine(
                screen_canonical_points,
                observed_screen_points,
                meta,
                screen_mask,
                initial=alignment,
                grid_px=config.visible_grid_px,
                trim_fraction=config.constrained_trim_fraction,
                iterations=config.constrained_iterations,
                scale_min_multiplier=config.constrained_scale_min_multiplier,
                scale_max_multiplier=config.constrained_scale_max_multiplier,
                rotation_max_deg=config.constrained_rotation_max_deg,
            )
            alignment = constrained_alignment
        screen_first_alignment = {
            "method": "screen_first",
            "screen_part_label": config.screen_first_screen_part_label,
            "base_part_label": config.screen_first_base_part_label,
            "base_mask_path": str(base_mask_path) if base_mask_path else None,
            "base_mask_area_pixels": int(base_mask.sum()) if base_mask is not None else None,
            "screen_mask_area_pixels": int(screen_mask.sum()) if screen_mask is not None else None,
            "screen_pca_initial": screen_pca_alignment,
            "screen_constrained_refine": constrained_alignment,
        }
    elif alignment_mode in {"pca_constrained", "pca_direct"}:
        alignment = pca_direct_alignment
        if alignment_mode == "pca_constrained" and config.constrained_refine:
            constrained_alignment = constrained_visible_refine(
                canonical_points,
                observed_points,
                meta,
                mask,
                initial=alignment,
                grid_px=config.visible_grid_px,
                trim_fraction=config.constrained_trim_fraction,
                iterations=config.constrained_iterations,
                scale_min_multiplier=config.constrained_scale_min_multiplier,
                scale_max_multiplier=config.constrained_scale_max_multiplier,
                rotation_max_deg=config.constrained_rotation_max_deg,
            )
            alignment = constrained_alignment
    elif alignment_mode == "free_icp":
        alignment = estimate_alignment(
            canonical_points,
            observed_points,
            seed=config.random_seed,
            trim_fraction=config.icp_trim_fraction,
            iterations=config.icp_iterations,
        )
        coarse_alignment = dict(alignment)
        coarse_visible_score = visible_bidirectional_score(
            canonical_points,
            observed_points,
            meta,
            mask,
            alignment["scale"],
            alignment["rotation"],
            alignment["translation"],
            config.visible_grid_px,
            config.visible_trim_fraction,
        )
        if config.visible_refine:
            visible_alignment = visible_sim3_refine(
                canonical_points,
                observed_points,
                meta,
                mask,
                initial=alignment,
                grid_px=config.visible_grid_px,
                trim_fraction=config.visible_trim_fraction,
                iterations=config.visible_iterations,
            )
            if visible_alignment["score"] < coarse_visible_score["score"]:
                alignment = visible_alignment
    else:
        raise ValueError(f"Unsupported final_alignment_mode: {config.final_alignment_mode}")
    pre_silhouette_alignment = dict(alignment)
    silhouette_alignment = None
    if config.silhouette_refine and not config.part_aware and alignment_mode != "pca_direct":
        if alignment_mode == "base_first":
            silhouette_points = base_canonical_points
            silhouette_mask = base_mask
        elif alignment_mode == "screen_first":
            silhouette_points = screen_canonical_points
            silhouette_mask = screen_mask
        else:
            silhouette_points = canonical_points
            silhouette_mask = mask
        silhouette_alignment = silhouette_scale_refine(
            silhouette_points,
            meta,
            silhouette_mask,
            alignment,
            q_min=config.silhouette_quantile_min,
            q_max=config.silhouette_quantile_max,
            min_multiplier=config.silhouette_scale_min_multiplier,
            max_multiplier=config.silhouette_scale_max_multiplier,
            steps=config.silhouette_scale_steps,
            boundary_trim_fraction=config.silhouette_boundary_trim_fraction,
            outside_weight=config.silhouette_outside_weight,
            boundary_weight=config.silhouette_boundary_weight,
            bbox_weight=config.silhouette_bbox_weight,
        )
        alignment = silhouette_alignment
        if base_first_alignment is not None:
            base_first_alignment["base_silhouette_refine"] = silhouette_alignment
        if screen_first_alignment is not None:
            screen_first_alignment["screen_silhouette_refine"] = silhouette_alignment
    screen_axis_twist_alignment = None
    if (
        alignment_mode == "screen_first"
        and config.screen_first_axis_twist
        and not config.part_aware
        and screen_canonical_points is not None
    ):
        screen_axis_twist_alignment = screen_hinge_axis_twist_refine(
            alignment,
            joints,
            screen_canonical_points,
            observed_screen_points,
            observed_base_points,
            meta,
            screen_mask,
            grid_px=config.visible_grid_px,
            trim_fraction=config.constrained_trim_fraction,
            max_abs_deg=config.screen_first_axis_twist_max_deg,
        )
        if screen_axis_twist_alignment is not None:
            alignment = screen_axis_twist_alignment
            if screen_first_alignment is not None:
                screen_first_alignment["screen_axis_twist_refine"] = screen_axis_twist_alignment.get("axis_twist")
    screen_projection_alignment = None
    if (
        alignment_mode == "screen_first"
        and config.screen_projection_refine
        and not config.part_aware
        and screen_canonical_points is not None
    ):
        screen_projection_alignment = screen_projection_refine_alignment(
            screen_canonical_points,
            observed_screen_points,
            meta,
            screen_projection_mask,
            alignment,
            grid_px=config.visible_grid_px,
            trim_fraction=config.constrained_trim_fraction,
            q_min=config.silhouette_quantile_min,
            q_max=config.silhouette_quantile_max,
            scale_min_multiplier=config.screen_projection_scale_min_multiplier,
            scale_max_multiplier=config.screen_projection_scale_max_multiplier,
            shift_max_px=config.screen_projection_shift_max_px,
            depth_weight=config.screen_projection_depth_weight,
        )
        alignment = screen_projection_alignment
        if screen_first_alignment is not None:
            screen_first_alignment["screen_projection_refine"] = screen_projection_alignment.get("screen_projection_refine")
    part_aware_alignment = None
    if config.part_aware:
        part_aware_alignment = estimate_part_aware_alignment(
            parts,
            observed_points,
            observed_v,
            meta,
            screen_label=config.screen_part_label,
            base_label=config.base_part_label,
            seed=config.random_seed,
            candidate_count=config.part_aware_candidates,
            trim_fraction=config.part_aware_trim_fraction,
            iterations=config.part_aware_iterations,
        )
        alignment = part_aware_alignment
    scale = alignment["scale"]
    rot = alignment["rotation"]
    trans = alignment["translation"]

    aligned_parts = []
    for part in parts:
        mesh_cam = apply_sim3_to_mesh(part["mesh"], scale, rot, trans)
        aligned_parts.append({**part, "mesh": mesh_cam})

    camera_joints = [transform_joint(joint, scale, rot, trans) for joint in joints if joint["type"] != "fixed"]
    hinge_refinement = None
    if config.hinge_refine and not config.part_aware and alignment_mode != "pca_direct" and camera_joints:
        if alignment_mode == "screen_first":
            if config.base_visible_surface_constrain and base_mask is not None:
                aligned_parts, hinge_refinement = hinge_visible_surface_angle_refine(
                    aligned_parts,
                    camera_joints,
                    observed_base_points,
                    meta,
                    base_mask,
                    moving_label=config.screen_first_base_part_label,
                    fixed_label=config.screen_first_screen_part_label,
                    angle_min_deg=config.hinge_angle_min_deg,
                    angle_max_deg=config.hinge_angle_max_deg,
                    angle_steps=config.hinge_angle_steps,
                    trim_fraction=config.hinge_trim_fraction,
                    plane_distance_weight=config.hinge_plane_distance_weight,
                    nn_weight=config.hinge_nn_weight,
                    normal_weight_m_per_deg=config.hinge_normal_weight_m_per_deg,
                    visible_normal_weight_m_per_deg=config.base_visible_surface_normal_weight_m_per_deg,
                    angle_regularizer_m_per_deg=config.hinge_angle_regularizer_m_per_deg,
                    observed_to_model_weight=config.base_visible_surface_observed_to_model_weight,
                    plane_offset_weight=config.base_visible_surface_plane_offset_weight,
                    snap_offset=config.base_visible_surface_snap_offset,
                    grid_px=config.base_visible_surface_grid_px,
                    seed=config.random_seed,
                    observed_metadata=observed_base_stats,
                )
            else:
                aligned_parts, hinge_refinement = hinge_moving_part_angle_refine(
                    aligned_parts,
                    camera_joints,
                    observed_base_points,
                    moving_label=config.screen_first_base_part_label,
                    fixed_label=config.screen_first_screen_part_label,
                    angle_min_deg=config.hinge_angle_min_deg,
                    angle_max_deg=config.hinge_angle_max_deg,
                    angle_steps=config.hinge_angle_steps,
                    trim_fraction=config.hinge_trim_fraction,
                    plane_distance_weight=config.hinge_plane_distance_weight,
                    nn_weight=config.hinge_nn_weight,
                    normal_weight_m_per_deg=config.hinge_normal_weight_m_per_deg,
                    angle_regularizer_m_per_deg=config.hinge_angle_regularizer_m_per_deg,
                    seed=config.random_seed,
                    observed_metadata=observed_base_stats,
                )
        else:
            aligned_parts, hinge_refinement = hinge_angle_refine(
                aligned_parts,
                camera_joints,
                observed_points,
                observed_v,
                screen_label=config.hinge_screen_part_label,
                base_label=config.hinge_base_part_label,
                angle_min_deg=config.hinge_angle_min_deg,
                angle_max_deg=config.hinge_angle_max_deg,
                angle_steps=config.hinge_angle_steps,
                trim_fraction=config.hinge_trim_fraction,
                plane_distance_weight=config.hinge_plane_distance_weight,
                nn_weight=config.hinge_nn_weight,
                normal_weight_m_per_deg=config.hinge_normal_weight_m_per_deg,
                angle_regularizer_m_per_deg=config.hinge_angle_regularizer_m_per_deg,
                seed=config.random_seed,
                observed_screen_points=observed_screen_points if alignment_mode == "base_first" else None,
                observed_split_metadata=observed_screen_stats if alignment_mode == "base_first" else None,
            )

    final_part_fit_metrics = {}
    final_part_by_label = {part["label"]: part for part in aligned_parts}
    if alignment_mode in part_mask_alignment_modes:
        if observed_base_points is not None and base_mask is not None and semantic_base_label in final_part_by_label:
            final_part_fit_metrics["base"] = part_observed_fit_metrics(
                final_part_by_label[semantic_base_label]["mesh"],
                observed_base_points,
                meta,
                base_mask,
                seed=config.random_seed + 501,
            )
            final_part_fit_metrics["base_visible_surface"] = visible_surface_observed_fit_metrics(
                final_part_by_label[semantic_base_label]["mesh"],
                observed_base_points,
                meta,
                base_mask,
                seed=config.random_seed + 503,
                grid_px=config.base_visible_surface_grid_px,
            )
        if observed_screen_points is not None and screen_mask is not None and semantic_screen_label in final_part_by_label:
            final_part_fit_metrics["screen"] = part_observed_fit_metrics(
                final_part_by_label[semantic_screen_label]["mesh"],
                observed_screen_points,
                meta,
                screen_mask,
                seed=config.random_seed + 502,
            )

    for part in aligned_parts:
        write_obj(out_dir / f"part_{part['label']}_camera.obj", part["mesh"])
    export_colored_scene(out_dir / "laptop_camera_aligned.glb", aligned_parts)
    alignment_overlay = save_mesh_projection_overlay(
        rgb_path,
        out_dir / "aligned_mesh_projection_overlay.png",
        meta,
        aligned_parts,
        seed=config.random_seed,
    )
    screen_projection_diagnostic = None
    if screen_projection_mask is not None and semantic_screen_label in final_part_by_label:
        screen_projection_diagnostic = save_part_projection_diagnostic(
            rgb_path,
            out_dir / "screen_projection_diagnostic.png",
            meta,
            final_part_by_label[semantic_screen_label]["mesh"],
            screen_projection_mask,
            PART_COLORS.get(semantic_screen_label, (75, 145, 210, 255)),
            seed=config.random_seed + 701,
            observed_points=observed_screen_points,
        )

    write_json(out_dir / "joint_camera.json", {"joints": camera_joints})

    view_frame = config.view_frame
    egoforce_dir = project_root / "outputs" / "egoforce_rgb_right"
    if view_frame is None:
        view_frame = find_first_hand_frame(egoforce_dir)

    view_result = None
    if view_frame is not None:
        align_row = frame_row(export_root, config.align_frame)
        view_row = frame_row(export_root, view_frame)
        t_view_align = camera_to_camera_matrix(meta, align_row, view_row, camera="right")
        view_dir = out_dir / f"view_frame_{frame_name(view_frame)}"
        view_parts = []
        for part in aligned_parts:
            mesh_view = apply_se3_to_mesh(part["mesh"], t_view_align)
            write_obj(view_dir / f"part_{part['label']}_view_camera.obj", mesh_view)
            view_parts.append({**part, "mesh": mesh_view})
        observed_view = transform_points(observed_points, t_view_align)
        export_point_cloud(view_dir / "observed_mask_pointcloud_view_camera.ply", observed_view, observed_colors)
        observed_base_view_path = None
        observed_screen_view_path = None
        if observed_base_points is not None:
            observed_base_view = transform_points(observed_base_points, t_view_align)
            base_view_colors = np.tile(
                np.asarray(PART_COLORS.get(semantic_base_label, (220, 50, 65, 255)), dtype=np.uint8),
                (len(observed_base_view), 1),
            )
            observed_base_view_path = view_dir / "observed_base_pointcloud_view_camera.ply"
            export_point_cloud(observed_base_view_path, observed_base_view, base_view_colors)
        if observed_screen_points is not None:
            observed_screen_view = transform_points(observed_screen_points, t_view_align)
            screen_view_colors = np.tile(
                np.asarray(PART_COLORS.get(semantic_screen_label, (75, 145, 210, 255)), dtype=np.uint8),
                (len(observed_screen_view), 1),
            )
            observed_screen_view_path = view_dir / "observed_screen_pointcloud_view_camera.ply"
            export_point_cloud(observed_screen_view_path, observed_screen_view, screen_view_colors)
        export_colored_scene(view_dir / "laptop_view_camera_aligned.glb", view_parts)
        view_joints = [transform_joint_se3(joint, t_view_align) for joint in camera_joints]
        write_json(view_dir / "joint_view_camera.json", {"joints": view_joints})
        view_result = {
            "view_frame": view_frame,
            "camera_transform_align_to_view": t_view_align,
            "view_dir": str(view_dir),
            "joint_view_json": str(view_dir / "joint_view_camera.json"),
            "observed_pointcloud_view": str(view_dir / "observed_mask_pointcloud_view_camera.ply"),
            "observed_base_pointcloud_view": str(observed_base_view_path) if observed_base_view_path else None,
            "observed_screen_pointcloud_view": str(observed_screen_view_path) if observed_screen_view_path else None,
        }

    result = {
        "target_id": config.target_id,
        "align_frame": config.align_frame,
        "view_frame": view_frame,
        "rgb_path": str(rgb_path),
        "depth_path": str(depth_path),
        "mask_path": str(mask_path),
        "depth_rgb_projection_validation": validation,
        "observed_cloud": observed_stats,
        "observed_base_cloud": observed_base_stats,
        "observed_screen_cloud": observed_screen_stats,
        "part_masks": {
            "base_mask_source": str(base_mask_path) if base_mask_path else None,
            "generated": base_screen_mask_outputs,
        },
        "convention_used": config.convention,
        "canonical_urdf": str(urdf_path),
        "canonical_part_labels": [part["label"] for part in parts],
        "alignment": {
            "alignment_mode": alignment_mode,
            "scale": scale,
            "rotation": rot,
            "translation": trans,
            "matrix_canonical_to_align_camera": alignment["matrix"],
            "final_metric_m": alignment.get("trimmed_mean_distance", alignment.get("score")),
            "trimmed_mean_distance_m": alignment.get("trimmed_mean_distance"),
            "visible_bidirectional_score": alignment.get("score"),
            "visible_model_to_observed_m": alignment.get("model_to_observed_m"),
            "visible_observed_to_model_m": alignment.get("observed_to_model_m"),
            "visible_count": alignment.get("visible_count"),
            "iterations": alignment.get("iterations"),
            "history": alignment.get("history", []),
            "initial_candidate_scores": (
                coarse_alignment["initial_candidate_scores"]
                if coarse_alignment is not None
                else [
                    item["score"]
                    for item in (
                        base_pca_alignment
                        if alignment_mode == "base_first"
                        else screen_pca_alignment
                        if alignment_mode == "screen_first"
                        else pca_direct_alignment
                    ).get("candidate_scores", [])
                ]
            ),
            "method": alignment.get(
                "method",
                "visible_refined" if alignment is visible_alignment else alignment_mode,
            ),
            "coarse_full_mesh": (
                {
                    "scale": coarse_alignment["scale"],
                    "rotation": coarse_alignment["rotation"],
                    "translation": coarse_alignment["translation"],
                    "matrix_canonical_to_align_camera": coarse_alignment["matrix"],
                    "trimmed_mean_distance_m": coarse_alignment["trimmed_mean_distance"],
                    "visible_score": coarse_visible_score,
                }
                if coarse_alignment is not None
                else None
            ),
            "pca_direct_initial": pca_direct_alignment,
            "base_first": base_first_alignment,
            "screen_first": screen_first_alignment,
            "screen_axis_twist_refine": screen_axis_twist_alignment,
            "constrained_refine": constrained_alignment,
            "visible_refine": visible_alignment,
            "pre_silhouette": pre_silhouette_alignment,
            "silhouette_refine": silhouette_alignment,
            "screen_projection_refine": screen_projection_alignment,
            "hinge_refine": hinge_refinement,
            "part_fit_metrics": final_part_fit_metrics,
            "part_aware": part_aware_alignment,
        },
        "outputs": {
            "result_dir": str(out_dir),
            "observed_pointcloud": str(out_dir / "observed_mask_pointcloud.ply"),
            "observed_base_pointcloud": str(out_dir / "observed_base_pointcloud.ply") if observed_base_points is not None else None,
            "observed_screen_pointcloud": str(out_dir / "observed_screen_pointcloud.ply") if observed_screen_points is not None else None,
            "canonical_pointcloud": str(out_dir / "canonical_sample_pointcloud.ply"),
            "canonical_base_pointcloud": str(out_dir / "canonical_base_sample_pointcloud.ply") if base_canonical_points is not None else None,
            "canonical_screen_pointcloud": str(out_dir / "canonical_screen_sample_pointcloud.ply") if screen_canonical_points is not None else None,
            "canonical_parts": str(out_dir / "canonical_parts.glb"),
            "aligned_parts": str(out_dir / "laptop_camera_aligned.glb"),
            "joint_camera": str(out_dir / "joint_camera.json"),
            "aligned_mesh_projection_overlay": alignment_overlay["path"],
            "screen_projection_diagnostic": (
                screen_projection_diagnostic["path"] if screen_projection_diagnostic is not None else None
            ),
        },
        "view": view_result,
    }
    write_json(out_dir / "alignment_result.json", result)
    return result
