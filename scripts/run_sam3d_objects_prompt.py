#!/usr/bin/env python3
"""Run SAM 3D Objects on one RGB/mask prompt and export pose QC artifacts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from copy import deepcopy
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import trimesh
from PIL import Image, ImageDraw
from pytorch3d.transforms import quaternion_to_matrix
from scipy.spatial.transform import Rotation


DEFAULT_SAM3D_ROOT = Path("/code/sam-3d-objects")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sam3d-root", type=Path, default=DEFAULT_SAM3D_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--turntable-frames", type=int, default=72)
    parser.add_argument("--turntable-resolution", type=int, default=512)
    parser.add_argument("--qc-min-size", type=int, default=512)
    return parser.parse_args()


def tensor_list(value: torch.Tensor) -> list:
    return value.detach().float().cpu().numpy().tolist()


def normalize_frame(frame: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    frame = np.asarray(frame)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def color_overlay(rgb: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    overlay = rgb.astype(np.float32).copy()
    gt_color = np.array([50, 205, 90], dtype=np.float32)
    pred_color = np.array([245, 95, 55], dtype=np.float32)
    both_color = np.array([255, 210, 55], dtype=np.float32)
    gt_only = gt & ~pred
    pred_only = pred & ~gt
    both = gt & pred
    alpha = 0.38
    for region, color in ((gt_only, gt_color), (pred_only, pred_color), (both, both_color)):
        overlay[region] = overlay[region] * (1 - alpha) + color * alpha

    gt_edge = cv2.morphologyEx(gt.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    pred_edge = cv2.morphologyEx(pred.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    overlay[gt_edge] = gt_color
    overlay[pred_edge] = pred_color
    return np.clip(overlay, 0, 255).astype(np.uint8)


def labeled_panel(image: np.ndarray, title: str, width: int = 640) -> Image.Image:
    source = Image.fromarray(image)
    height = max(1, round(source.height * width / source.width))
    source = source.resize((width, height), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (width, height + 38), (24, 27, 31))
    panel.paste(source, (0, 38))
    draw = ImageDraw.Draw(panel)
    draw.text((12, 11), title, fill=(235, 238, 242))
    return panel


def make_contact_sheet(
    input_overlay: np.ndarray,
    projection_overlay: np.ndarray,
    turntable: list[np.ndarray],
    iou: float,
    centroid_error_px: float,
) -> Image.Image:
    panels = [
        labeled_panel(input_overlay, "Prompt RGB + SAM2 mask (green)"),
        labeled_panel(
            projection_overlay,
            f"SAM 3D pose projection: IoU={iou:.3f}, centroid error={centroid_error_px:.1f}px",
        ),
    ]
    indices = np.linspace(0, len(turntable) - 1, 4, dtype=int)
    panels.extend(
        labeled_panel(turntable[index], f"SAM 3D mesh view {number + 1}/4")
        for number, index in enumerate(indices)
    )
    cell_w = max(panel.width for panel in panels)
    cell_h = max(panel.height for panel in panels)
    sheet = Image.new("RGB", (cell_w * 2, cell_h * 3), (15, 17, 20))
    for index, panel in enumerate(panels):
        x = (index % 2) * cell_w + (cell_w - panel.width) // 2
        y = (index // 2) * cell_h + (cell_h - panel.height) // 2
        sheet.paste(panel, (x, y))
    return sheet


def main() -> int:
    args = parse_args()
    sam3d_root = args.sam3d_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(sam3d_root))
    sys.path.insert(0, str(sam3d_root / "notebook"))

    from inference import Inference, make_scene, ready_gaussian_for_video_rendering, render_video
    from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform
    from sam3d_objects.pipeline.layout_post_optimization_utils import get_mask_renderer, get_mesh

    image_path = args.image.resolve()
    mask_path = args.mask.resolve()
    rgb = np.asarray(Image.open(image_path).convert("RGB"))
    mask = np.asarray(Image.open(mask_path).convert("L")) > 127
    if rgb.shape[:2] != mask.shape:
        raise ValueError(f"Image/mask shape mismatch: {rgb.shape[:2]} vs {mask.shape}")

    config_path = sam3d_root / "checkpoints/hf/pipeline.yaml"
    inference = Inference(str(config_path), compile=False)
    pipeline = inference._pipeline
    rgba = inference.merge_mask_to_rgba(rgb, mask)

    # Compute MoGe once, retain its intrinsics, and pass the pointmap back into the pipeline.
    pointmap_dict = pipeline.compute_pointmap(rgba)
    pointmap_hwc = pointmap_dict["pointmap"].permute(1, 2, 0).contiguous()
    output = pipeline.run(
        rgba,
        None,
        args.seed,
        stage1_only=False,
        with_mesh_postprocess=False,
        with_texture_baking=False,
        with_layout_postprocess=False,
        use_vertex_color=True,
        pointmap=pointmap_hwc,
    )

    canonical_mesh = output["glb"]
    if not isinstance(canonical_mesh, trimesh.Trimesh):
        canonical_mesh = canonical_mesh.dump(concatenate=True)
    canonical_mesh.export(output_dir / "mesh_canonical.glb")
    canonical_mesh.export(output_dir / "mesh_canonical.ply")
    output["gs"].save_ply(output_dir / "gaussian_canonical.ply")

    quaternion = output["rotation"].detach().float()
    translation = output["translation"].detach().float()
    scale = output["scale"].detach().float()
    rotation = quaternion_to_matrix(quaternion)
    transform = compose_transform(scale=scale, rotation=rotation, translation=translation)

    # SAM 3D mesh decoder is z-up; pose outputs and pointmaps are PyTorch3D camera coordinates.
    z_up_to_y_up = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
        device=translation.device,
    )
    local_vertices = torch.from_numpy(np.asarray(canonical_mesh.vertices)).float().to(translation.device)
    camera_vertices = transform.transform_points((local_vertices @ z_up_to_y_up.T).unsqueeze(0))[0]
    posed_mesh = canonical_mesh.copy()
    posed_mesh.vertices = camera_vertices.detach().cpu().numpy()
    posed_mesh.export(output_dir / "mesh_posed_camera.glb")
    posed_mesh.export(output_dir / "mesh_posed_camera.ply")

    posed_gaussian = make_scene(output)
    posed_gaussian.save_ply(output_dir / "gaussian_posed_camera.ply")

    row_matrix = transform.get_matrix()[0].detach().float().cpu().numpy()
    z_up_to_y_up_4x4 = np.eye(4, dtype=np.float32)
    z_up_to_y_up_4x4[:3, :3] = z_up_to_y_up.detach().cpu().numpy()
    mesh_to_camera_column = row_matrix.T @ z_up_to_y_up_4x4
    rotation_numpy = rotation[0].detach().float().cpu().numpy()
    euler_xyz = Rotation.from_matrix(rotation_numpy).as_euler("xyz", degrees=True)
    intrinsics = pointmap_dict["intrinsics"].detach().float()
    pose = {
        "source_image": str(image_path),
        "source_mask": str(mask_path),
        "seed": args.seed,
        "coordinate_convention": {
            "mesh_canonical": "SAM 3D decoder coordinates (z-up)",
            "mesh_posed_camera": "PyTorch3D camera coordinates (x-left, y-up, z-forward)",
            "quaternion_order": "wxyz",
            "transform_order": "z-up to y-up, then scale, rotation, translation",
            "scale_note": "MoGe monocular scene scale; approximately metric but not guaranteed absolute metric",
        },
        "quaternion_wxyz": tensor_list(quaternion[0]),
        "rotation_matrix": rotation_numpy.tolist(),
        "euler_xyz_degrees": euler_xyz.tolist(),
        "translation": tensor_list(translation[0]),
        "scale_xyz": tensor_list(scale[0]),
        "pose_scale_rotation_translation_matrix_column_vector": row_matrix.T.tolist(),
        "mesh_zup_to_camera_matrix_column_vector": mesh_to_camera_column.tolist(),
        "intrinsics_normalized": tensor_list(intrinsics),
    }

    # Render the posed mesh with the same camera implementation used by SAM 3D layout QC.
    render_mesh, _, _ = get_mesh(deepcopy(canonical_mesh), transform, translation.device)
    resized_mask, renderer = get_mask_renderer(
        torch.from_numpy(mask.astype(np.float32)).to(translation.device),
        args.qc_min_size,
        intrinsics,
        translation.device,
    )
    rendered = renderer(render_mesh)
    pred_mask = rendered[0, ..., 3].detach().cpu().numpy() > 0.5
    gt_mask = resized_mask[0, 0].detach().cpu().numpy() > 0.5
    intersection = np.count_nonzero(pred_mask & gt_mask)
    union = np.count_nonzero(pred_mask | gt_mask)
    iou = float(intersection / union) if union else 0.0

    def centroid(binary: np.ndarray) -> np.ndarray:
        points = np.argwhere(binary)
        return points.mean(axis=0) if len(points) else np.array([math.nan, math.nan])

    centroid_error = float(np.linalg.norm(centroid(pred_mask) - centroid(gt_mask)))
    render_h, render_w = pred_mask.shape
    resized_rgb = cv2.resize(rgb, (render_w, render_h), interpolation=cv2.INTER_AREA)
    input_overlay = color_overlay(resized_rgb, gt_mask, np.zeros_like(gt_mask))
    projection_overlay = color_overlay(resized_rgb, gt_mask, pred_mask)
    Image.fromarray(input_overlay).save(output_dir / "input_mask_overlay.png")
    Image.fromarray(projection_overlay).save(output_dir / "pose_projection_overlay.png")
    Image.fromarray((pred_mask.astype(np.uint8) * 255)).save(output_dir / "projected_mesh_mask.png")

    normalized_gaussian = ready_gaussian_for_video_rendering(posed_gaussian)
    frames = render_video(
        normalized_gaussian,
        r=1.0,
        fov=60,
        pitch_deg=15,
        yaw_start_deg=-45,
        resolution=args.turntable_resolution,
        num_frames=args.turntable_frames,
    )["color"]
    frames = [normalize_frame(frame) for frame in frames]
    imageio.mimsave(output_dir / "mesh_turntable.gif", frames, duration=1000 / 24, loop=0)
    make_contact_sheet(input_overlay, projection_overlay, frames, iou, centroid_error).save(
        output_dir / "sam3d_qc_contact_sheet.jpg", quality=94
    )

    pose["projection_qc"] = {
        "render_size_hw": [render_h, render_w],
        "mask_iou": iou,
        "centroid_error_pixels_at_render_size": centroid_error,
        "gt_mask_pixels": int(np.count_nonzero(gt_mask)),
        "predicted_mask_pixels": int(np.count_nonzero(pred_mask)),
    }
    pose["mesh_stats"] = {
        "vertices": int(len(canonical_mesh.vertices)),
        "faces": int(len(canonical_mesh.faces)),
        "canonical_bounds": np.asarray(canonical_mesh.bounds).tolist(),
        "posed_camera_bounds": np.asarray(posed_mesh.bounds).tolist(),
    }
    (output_dir / "pose.json").write_text(json.dumps(pose, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), **pose["projection_qc"], **pose["mesh_stats"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
