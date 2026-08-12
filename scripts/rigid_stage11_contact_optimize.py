#!/usr/bin/env python3
"""Depth-first EgoForce refinement with VLM/geometry/temporal per-finger contact fusion."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt
import torch
import torch.nn.functional as F
import trimesh
from pytorch3d.loss.point_mesh_distance import point_face_distance
from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.utils.camera_conversions import cameras_from_opencv_projection


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EGOFORCE_ROOT = Path("/code/EgoForce")
for path in (PROJECT_ROOT, EGOFORCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from settings import config as egoforce_config  # noqa: E402
from models import LimbModel  # noqa: E402
from utils.rotations import rotation_6d_to_matrix  # noqa: E402
from vlm_sam2_recon.rigid_pipeline.rigid_contact import (  # noqa: E402
    image_sample,
    project_points_torch,
    sample_sdf_grid,
    second_difference,
    topk_penetration_loss,
    transform_points_torch,
)


REGION_COLUMNS = {
    "index": (1, 2, 3),
    "middle": (4, 5, 6),
    "pinky": (7, 8, 9),
    "ring": (10, 11, 12),
    "thumb": (13, 14, 15),
    "palm": (0,),
}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(type(mesh))
    return mesh


def resolve_bottle_mask_dir(workspace: Path) -> Path:
    candidates = (
        workspace / "outputs/04_object_masks/combined",
        workspace / "outputs/04_object_masks/bottle/objects/bottle",
        workspace / "outputs/02_sam2_frame0_masks/propagated/objects/bottle",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "Bottle mask directory is missing; checked: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def export_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(path)


def transform_np(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def rotation_angle(rotation: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    relative = torch.matmul(rotation, reference.transpose(-1, -2))
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-0.999999, 0.999999)
    return torch.acos(cosine)


def one_sided_point_to_mesh(meshes: Meshes, pointclouds: Pointclouds) -> torch.Tensor:
    points = pointclouds.points_packed()
    triangles = meshes.verts_packed()[meshes.faces_packed()]
    squared = point_face_distance(
        points,
        pointclouds.cloud_to_packed_first_idx(),
        triangles,
        meshes.mesh_to_faces_packed_first_idx(),
        int(pointclouds.num_points_per_cloud().max()),
        1e-12,
    )
    return torch.sqrt(squared.clamp_min(1e-12)).reshape(len(meshes), -1)


def make_cameras(
    count: int,
    intrinsics: dict[str, float],
    height: int,
    width: int,
    scale: int,
    device: torch.device,
):
    camera_matrix = torch.tensor(
        [
            [float(intrinsics["fx"]) / scale, 0.0, float(intrinsics["cx"]) / scale],
            [0.0, float(intrinsics["fy"]) / scale, float(intrinsics["cy"]) / scale],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    )[None].repeat(count, 1, 1)
    return cameras_from_opencv_projection(
        torch.eye(3, device=device)[None].repeat(count, 1, 1),
        torch.zeros(count, 3, device=device),
        camera_matrix,
        torch.tensor([[height, width]], dtype=torch.float32, device=device).repeat(count, 1),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=PROJECT_ROOT / "run_rigid_20260715_215524",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--render-scale", type=int, default=4)
    parser.add_argument("--points-per-frame", type=int, default=768)
    parser.add_argument("--shape-steps", type=int, default=100)
    parser.add_argument("--depth-wrist-steps", type=int, default=260)
    parser.add_argument("--depth-pose-steps", type=int, default=180)
    parser.add_argument("--anchor-steps", type=int, default=240)
    parser.add_argument("--seed-steps", type=int, default=240)
    parser.add_argument("--propagation-steps-per-frame", type=int, default=80)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--overlap-normalization-pixels", type=float, default=8000.0)
    parser.add_argument("--strong-fused-score", type=float, default=0.72)
    parser.add_argument("--trusted-fused-score", type=float, default=0.68)
    parser.add_argument("--strong-relative-speed-m", type=float, default=0.015)
    parser.add_argument(
        "--translation-trust-region-m",
        type=float,
        default=0.015,
        help="Maximum per-frame MANO translation change from EgoForce initialization.",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="Inclusive final frame; mixed rigid optimization must stop at its VLM event boundary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.translation_trust_region_m <= 0.0:
        raise ValueError("--translation-trust-region-m must be positive")
    workspace = args.workspace.resolve()
    output = workspace / "outputs/11_contact_optimization"
    anchor_info = json.loads((output / "vlm_contact_anchor.json").read_text(encoding="utf-8"))
    proxy_manifest = json.loads((output / "collision_proxy_manifest.json").read_text(encoding="utf-8"))
    for directory in (output / "depth_aligned_C0", output / "optimized_C0", output / "hand_pointcloud_C0"):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)
    device = torch.device(args.device)

    hand_manifest = json.loads(
        (workspace / "outputs/09_egoforce/dynamic_manifest.json").read_text(encoding="utf-8")
    )
    object_mask_dir = resolve_bottle_mask_dir(workspace)
    accepted = [
        entry
        for entry in hand_manifest["frames"]
        if entry["status"] == "completed"
        and "right_hand_C0" in entry
        and (args.end_frame is None or int(entry["frame"]) <= args.end_frame)
    ]
    frame_ids = np.asarray([entry["frame"] for entry in accepted], dtype=np.int64)
    if len(frame_ids) < 3 or np.any(np.diff(frame_ids) <= 0):
        raise ValueError(f"Need at least three increasing valid right-hand frames, got {frame_ids.tolist()}")
    output_frame_count = (
        int(args.end_frame) + 1
        if args.end_frame is not None
        else int(hand_manifest["frame_count"])
    )
    raw_records = []
    selected_side_indices = []
    for entry in accepted:
        with np.load(entry["raw_Ct_npz"]) as loaded:
            raw_records.append({key: loaded[key] for key in loaded.files})
        selected_side_indices.append(int(entry.get("selected_raw_side_index", 1)))

    def stacked(key: str, dtype=torch.float32) -> torch.Tensor:
        return torch.as_tensor(
            np.stack(
                [
                    record[key][selected_side_indices[index]]
                    for index, record in enumerate(raw_records)
                ]
            ),
            dtype=dtype,
            device=device,
        )

    betas_raw = stacked("mano_betas")
    global_raw = stacked("mano_global_orient")
    pose_raw = stacked("mano_hand_pose")
    translation_raw = stacked("mano_transl")
    arm_shape = stacked("egoforce_arm_shape")
    arm_rotation = stacked("egoforce_arm_R")
    hand_type = stacked("mano_hand_type", dtype=torch.int64)
    raw_vertices_ct = stacked("hand_vertices")
    raw_joints_ct = stacked("hand_joints")
    target_uv = stacked("egoforce_hand_keypoints_2d")
    keypoint_weight = stacked("egoforce_hand_keypoint_confidence")
    keypoint_weight = (keypoint_weight / keypoint_weight.mean(1, keepdim=True).clamp_min(1e-8)).clamp(0.05, 5.0)

    camera = json.loads((workspace / "outputs/00_rgb_frames/camera.json").read_text(encoding="utf-8"))
    intrinsics = camera["rgb_intrinsics_right"]
    poses_all = np.load(workspace / "outputs/00_rgb_frames/poses.npz")["T_C0_from_Ct"]
    deltas_all = np.load(workspace / "outputs/08_tracking/Delta_C0_object_motion.npy")
    poses = poses_all[frame_ids]
    deltas = deltas_all[frame_ids]
    t_c0_from_ct = torch.as_tensor(poses, dtype=torch.float32, device=device)
    t_c0_from_object = torch.as_tensor(deltas, dtype=torch.float32, device=device)
    t_object_from_c0 = torch.as_tensor(np.linalg.inv(deltas), dtype=torch.float32, device=device)

    model = LimbModel(egoforce_config, device=device, use_pose_pca=False)
    hand_faces_np = np.asarray(model.faces.right_hand, dtype=np.int64)
    arm_faces_np = np.asarray(model.faces.arm, dtype=np.int64)
    hand_faces = torch.as_tensor(hand_faces_np, dtype=torch.long, device=device)
    assignment = model.hand_model.mano_layer_right.lbs_weights.detach().argmax(1).cpu().numpy()
    region_ids = {
        name: torch.as_tensor(np.flatnonzero(np.isin(assignment, columns)), dtype=torch.long, device=device)
        for name, columns in REGION_COLUMNS.items()
    }

    # Fit one physical hand shape before using depth.
    shared_betas = torch.nn.Parameter(torch.median(betas_raw, dim=0).values)
    shape_optimizer = torch.optim.Adam([shared_betas], lr=0.03)
    for _ in range(args.shape_steps):
        limb = model(
            shared_betas[None].expand(len(frame_ids), -1),
            global_raw,
            pose_raw,
            translation_raw,
            hand_type,
            arm_shape,
            arm_rotation,
        )
        residual = torch.linalg.norm(limb.hand.vertices - raw_vertices_ct, dim=-1)
        loss = residual.clamp_max(0.015).mean() + 1e-4 * shared_betas.square().mean()
        shape_optimizer.zero_grad()
        loss.backward()
        shape_optimizer.step()
    shared_betas = shared_betas.detach()

    # Rasterize Raw hand silhouettes to keep only SAM2 depth belonging to the hand mesh, not forearm/background.
    scale = args.render_scale
    full_height, full_width = np.asarray(
        Image.open(workspace / "outputs/00_rgb_frames/right_rgb_png/000024.png")
    ).shape[:2]
    height, width = full_height // scale, full_width // scale
    cameras = make_cameras(len(frame_ids), intrinsics, height, width, scale, device)
    raw_meshes = Meshes(
        verts=list(raw_vertices_ct.unbind(0)),
        faces=[hand_faces for _ in frame_ids],
    )
    raw_fragments = MeshRasterizer(
        cameras=cameras,
        raster_settings=RasterizationSettings(
            image_size=(height, width),
            faces_per_pixel=1,
            blur_radius=0.0,
            cull_backfaces=False,
            max_faces_per_bin=100000,
        ),
    )(raw_meshes)
    raw_render_mask = (raw_fragments.pix_to_face[..., 0] >= 0).detach().cpu().numpy()

    pointclouds_ct = []
    pointclouds_c0 = []
    point_counts = []
    silhouette_distance_np = []
    rng = np.random.default_rng(20260717)
    for local, frame in enumerate(frame_ids):
        mask_path = Path(
            accepted[local].get(
                "sam2_hand_mask_path",
                workspace / f"outputs/02_hand_masks/combined/{frame:06d}.png",
            )
        )
        mask_full = np.asarray(Image.open(mask_path).convert("L")) > 127
        mask = mask_full[::scale, ::scale][:height, :width]
        mask = binary_erosion(mask, iterations=2)
        raw_support = binary_dilation(raw_render_mask[local], iterations=4)
        hand_support = mask & raw_support
        depth = np.load(
            workspace / f"outputs/06_dense_depth/metric_depth_npy/{frame:06d}.npy"
        )[::scale, ::scale][:height, :width]
        confidence = np.load(
            workspace / f"outputs/06_dense_depth/anchor_confidence_npy/{frame:06d}.npy"
        )[::scale, ::scale][:height, :width]
        valid = hand_support & np.isfinite(depth) & (depth > 0.1) & (depth < 2.0)
        yy, xx = np.nonzero(valid)
        z = depth[yy, xx]
        x = (xx - float(intrinsics["cx"]) / scale) * z / (float(intrinsics["fx"]) / scale)
        y = (yy - float(intrinsics["cy"]) / scale) * z / (float(intrinsics["fy"]) / scale)
        points = np.column_stack((x, y, z)).astype(np.float32)
        if len(points) < 64:
            raise RuntimeError(f"Frame {frame} has only {len(points)} reliable hand-depth points")
        weights = (0.25 + 0.75 * confidence[yy, xx]).astype(np.float64)
        weights = weights / weights.sum()
        replace = len(points) < args.points_per_frame
        selected = rng.choice(
            len(points), size=args.points_per_frame, replace=replace, p=weights
        )
        sampled = points[selected]
        pointclouds_ct.append(sampled)
        sampled_c0 = transform_np(sampled, poses[local]).astype(np.float32)
        pointclouds_c0.append(sampled_c0)
        point_counts.append(int(len(points)))
        np.savez_compressed(
            output / f"hand_pointcloud_C0/{frame:06d}.npz",
            points_C0=sampled_c0,
            points_Ct=sampled,
            source_coordinate_frame=np.asarray("current_right_camera_opencv_rdf"),
            coordinate_frame=np.asarray("frame0_right_camera_opencv_rdf"),
        )
        silhouette_distance_np.append(distance_transform_edt(~mask_full).astype(np.float32))
    pointclouds_ct_tensor = torch.as_tensor(np.stack(pointclouds_ct), device=device)
    pointclouds = Pointclouds(points=list(pointclouds_ct_tensor.unbind(0)))
    silhouette_distance = torch.as_tensor(np.stack(silhouette_distance_np), device=device)

    # Pre-render dynamic proxy front/back depths for side-aware contact and occlusion QC.
    proxy = load_mesh(Path(proxy_manifest["collision_proxy"]))
    proxy_faces = torch.as_tensor(np.asarray(proxy.faces), dtype=torch.long, device=device)
    proxy_vertices_ct = []
    for local in range(len(frame_ids)):
        vertices_c0 = transform_np(np.asarray(proxy.vertices), deltas[local])
        proxy_vertices_ct.append(transform_np(vertices_c0, np.linalg.inv(poses[local])))
    proxy_meshes = Meshes(
        verts=[torch.as_tensor(vertices, dtype=torch.float32, device=device) for vertices in proxy_vertices_ct],
        faces=[proxy_faces for _ in frame_ids],
    )
    proxy_fragments = MeshRasterizer(
        cameras=cameras,
        raster_settings=RasterizationSettings(
            image_size=(height, width),
            faces_per_pixel=8,
            blur_radius=0.0,
            cull_backfaces=False,
            max_faces_per_bin=100000,
        ),
    )(proxy_meshes)
    proxy_z = proxy_fragments.zbuf.detach()
    proxy_valid = proxy_z > 0
    proxy_front = torch.where(proxy_valid, proxy_z, torch.full_like(proxy_z, float("inf"))).min(-1).values
    proxy_back = torch.where(proxy_valid, proxy_z, torch.full_like(proxy_z, -float("inf"))).max(-1).values

    sdf_data = np.load(output / "collision_sdf_C0.npz")
    sdf = torch.as_tensor(sdf_data["sdf_xyz"], dtype=torch.float32, device=device)
    sdf_origin = torch.as_tensor(sdf_data["origin_xyz"], dtype=torch.float32, device=device)
    sdf_pitch = float(sdf_data["pitch_m"])

    global6 = torch.nn.Parameter(global_raw.clone())
    pose6 = torch.nn.Parameter(pose_raw.clone())
    translation = torch.nn.Parameter(translation_raw.clone())
    raw_global_matrix = rotation_6d_to_matrix(global_raw.reshape(-1, 6)).reshape(-1, 3, 3).detach()
    raw_pose_matrix = rotation_6d_to_matrix(pose_raw.reshape(-1, 6)).reshape(len(frame_ids), 15, 3, 3).detach()
    reliable_frame = torch.as_tensor(np.asarray(point_counts) >= 128, device=device)
    history = []

    def geometry() -> dict[str, torch.Tensor]:
        limb = model(
            shared_betas[None].expand(len(frame_ids), -1),
            global6,
            pose6,
            translation,
            hand_type,
            arm_shape,
            arm_rotation,
        )
        vertices_c0 = transform_points_torch(limb.hand.vertices, t_c0_from_ct)
        joints_c0 = transform_points_torch(limb.hand.joints, t_c0_from_ct)
        return {
            "vertices_ct": limb.hand.vertices,
            "joints_ct": limb.hand.joints,
            "arm_vertices_ct": limb.arm.vertices,
            "arm_joints_ct": limb.arm.joints,
            "vertices_c0": vertices_c0,
            "joints_c0": joints_c0,
            "vertices_object": transform_points_torch(vertices_c0, t_object_from_c0),
        }

    def observation_losses(
        current: dict[str, torch.Tensor],
        frame_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if frame_mask is not None and not bool(frame_mask.any()):
            raise ValueError("observation_losses frame_mask must select at least one frame")

        def frame_mean(values: torch.Tensor) -> torch.Tensor:
            selected = values if frame_mask is None else values[frame_mask]
            return selected.mean()

        def temporal_mean(values: torch.Tensor) -> torch.Tensor:
            if frame_mask is None:
                return values.mean()
            selected = values[frame_mask[1:-1]]
            return selected.mean() if selected.numel() else values.sum() * 0.0

        def temporal_rms(values: torch.Tensor) -> torch.Tensor:
            if frame_mask is not None:
                values = values[frame_mask[1:-1]]
            return values.square().mean().sqrt() if values.numel() else values.sum() * 0.0

        meshes = Meshes(
            verts=list(current["vertices_ct"].unbind(0)),
            faces=[hand_faces for _ in frame_ids],
        )
        distances = one_sided_point_to_mesh(meshes, pointclouds)
        point_loss = F.smooth_l1_loss(
            distances,
            torch.zeros_like(distances),
            beta=0.008,
            reduction="none",
        ) / 0.008
        uv_joints = project_points_torch(current["joints_ct"], intrinsics)
        uv_vertices = project_points_torch(current["vertices_ct"], intrinsics)
        kp_error = torch.linalg.norm(uv_joints - target_uv, dim=-1)
        keypoint_loss = F.smooth_l1_loss(
            kp_error,
            torch.zeros_like(kp_error),
            beta=12.0,
            reduction="none",
        ) / 12.0
        global_matrix = rotation_6d_to_matrix(global6.reshape(-1, 6)).reshape(-1, 3, 3)
        pose_matrix = rotation_6d_to_matrix(pose6.reshape(-1, 6)).reshape(len(frame_ids), 15, 3, 3)
        wrist_acceleration = torch.linalg.norm(
            second_difference(current["joints_c0"][:, 0]), dim=-1
        )
        rotation_acceleration = second_difference(global_matrix.flatten(1))
        pose_acceleration = second_difference(pose_matrix.flatten(1))
        return {
            "pointcloud": frame_mean(point_loss.clamp_max(4.0)),
            "keypoint": frame_mean(keypoint_loss * keypoint_weight),
            "silhouette": frame_mean(
                image_sample(silhouette_distance, uv_vertices).clamp_max(50.0) / 25.0
            ),
            "translation_anchor": frame_mean(
                torch.linalg.norm(translation - translation_raw, dim=-1)
            ) / args.translation_trust_region_m,
            "rotation_anchor": frame_mean(
                rotation_angle(global_matrix, raw_global_matrix)
            ) / np.deg2rad(15.0),
            "pose_anchor": frame_mean(
                rotation_angle(pose_matrix, raw_pose_matrix)
            ) / np.deg2rad(20.0),
            "wrist_acceleration": temporal_mean(wrist_acceleration) / 0.005,
            "rotation_acceleration": temporal_rms(rotation_acceleration) / 0.05,
            "pose_acceleration": temporal_rms(pose_acceleration) / 0.05,
        }

    def enforce_trust_region(active_pose: torch.Tensor | None = None) -> None:
        with torch.no_grad():
            translation[~reliable_frame] = translation_raw[~reliable_frame]
            global6[~reliable_frame] = global_raw[~reliable_frame]
            if active_pose is not None:
                pose6[~active_pose] = pose_raw[~active_pose]
            delta = translation - translation_raw
            norm = torch.linalg.norm(delta, dim=-1, keepdim=True)
            translation.copy_(
                translation_raw
                + delta
                * torch.clamp(
                    args.translation_trust_region_m / norm.clamp_min(1e-8),
                    max=1.0,
                )
            )

    depth_weights = {
        "pointcloud": 3.0,
        "keypoint": 3.0,
        "silhouette": 0.2,
        "translation_anchor": 0.7,
        "rotation_anchor": 0.7,
        "pose_anchor": 0.8,
        "wrist_acceleration": 0.25,
        "rotation_acceleration": 0.15,
        "pose_acceleration": 0.15,
    }

    def run_observation_phase(name: str, steps: int, include_pose: bool) -> None:
        groups = [
            {"params": [translation], "lr": 4e-4 if not include_pose else 1.5e-4},
            {"params": [global6], "lr": 1.5e-4 if not include_pose else 7e-5},
        ]
        if include_pose:
            groups.append({"params": [pose6], "lr": 7e-5})
        optimizer = torch.optim.Adam(groups)
        for step in range(steps):
            current = geometry()
            terms = observation_losses(current)
            total = sum(depth_weights[key] * value for key, value in terms.items())
            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_([translation, global6, pose6], 5.0)
            optimizer.step()
            enforce_trust_region(torch.ones_like(reliable_frame) if include_pose else None)
            if step % args.log_every == 0 or step == steps - 1:
                record = {"phase": name, "step": step, "total": float(total.detach())}
                record.update({key: float(value.detach()) for key, value in terms.items()})
                history.append(record)
                print(f"[{name} {step + 1}/{steps}] " + " ".join(f"{k}={v:.4f}" for k, v in record.items() if isinstance(v, float)), flush=True)

    run_observation_phase("depth_wrist", args.depth_wrist_steps, include_pose=False)
    run_observation_phase("depth_pose", args.depth_pose_steps, include_pose=True)
    depth_global = global6.detach().clone()
    depth_pose = pose6.detach().clone()
    depth_translation = translation.detach().clone()
    with torch.no_grad():
        depth_geometry = geometry()

    # Fuse VLM semantics with metric geometry and hand/object relative motion per finger.
    finger_names = [finger for finger in ("thumb", "index", "middle", "ring", "pinky") if finger in region_ids]
    vlm_prior = np.full((len(frame_ids), len(finger_names)), 0.05, dtype=np.float32)
    segments = anchor_info.get("contact_segments", [])
    if not segments and "contact_interval" in anchor_info:
        segments = [
            {
                "start_frame": anchor_info["contact_interval"][0],
                "end_frame": anchor_info["contact_interval"][1],
                "finger_probabilities": anchor_info.get("finger_contact_probabilities", {}),
            }
        ]
    for segment in segments:
        start, end = int(segment["start_frame"]), int(segment["end_frame"])
        probabilities = segment.get("finger_probabilities", {})
        for finger_index, finger in enumerate(finger_names):
            probability = float(probabilities.get(finger, 0.0))
            inside = (frame_ids >= start) & (frame_ids <= end)
            vlm_prior[inside, finger_index] = np.maximum(vlm_prior[inside, finger_index], probability)

    overlap_score = np.zeros(len(frame_ids), dtype=np.float32)
    overlap_pixels = np.zeros(len(frame_ids), dtype=np.int64)
    for local, frame in enumerate(frame_ids):
        hand_mask = np.asarray(
            Image.open(workspace / f"outputs/02_hand_masks/combined/{frame:06d}.png")
        ) > 127
        object_mask = np.asarray(
            Image.open(object_mask_dir / f"{frame:06d}.png")
        ) > 127
        overlap_pixels[local] = int((hand_mask & object_mask).sum())
        overlap_score[local] = float(
            np.clip(overlap_pixels[local] / args.overlap_normalization_pixels, 0.0, 1.0)
        )

    tracking_diagnostics = [
        json.loads(line)
        for line in (workspace / "outputs/08_tracking/frame_diagnostics.jsonl").read_text().splitlines()
    ]
    object_motion_score = np.zeros(len(frame_ids), dtype=np.float32)
    for local, frame in enumerate(frame_ids):
        record = tracking_diagnostics[int(frame)]
        translation_score = float(record.get("final_object_center_step_m", 0.0)) / 0.008
        rotation_score = float(record.get("final_rotation_step_deg", 0.0)) / 4.0
        object_motion_score[local] = float(np.clip(max(translation_score, rotation_score), 0.0, 1.0))
    if len(object_motion_score) >= 3:
        object_motion_score = np.maximum.reduce(
            [
                object_motion_score,
                np.r_[object_motion_score[0], object_motion_score[:-1]],
                np.r_[object_motion_score[1:], object_motion_score[-1]],
            ]
        )

    with torch.no_grad():
        depth_signed = sample_sdf_grid(
            sdf,
            depth_geometry["vertices_object"],
            sdf_origin,
            sdf_pitch,
        ).abs()
        finger_gap = np.full((len(frame_ids), len(finger_names)), np.inf, dtype=np.float32)
        finger_centroid_object = np.full(
            (len(frame_ids), len(finger_names), 3), np.nan, dtype=np.float32
        )
        for finger_index, finger in enumerate(finger_names):
            ids = region_ids[finger]
            for local in range(len(frame_ids)):
                values = depth_signed[local, ids]
                chosen = torch.topk(values, min(8, len(values)), largest=False).indices
                selected_ids = ids[chosen]
                finger_gap[local, finger_index] = float(values[chosen].mean().cpu())
                finger_centroid_object[local, finger_index] = (
                    depth_geometry["vertices_object"][local, selected_ids].mean(0).cpu().numpy()
                )

    relative_speed = np.full_like(finger_gap, np.inf)
    steps = np.linalg.norm(np.diff(finger_centroid_object, axis=0), axis=-1)
    if len(frame_ids) > 1:
        relative_speed[0] = steps[0]
        relative_speed[-1] = steps[-1]
    if len(frame_ids) > 2:
        relative_speed[1:-1] = 0.5 * (steps[:-1] + steps[1:])
    geometry_score = np.exp(-finger_gap / 0.012).astype(np.float32)
    temporal_score = np.exp(-relative_speed / 0.012).astype(np.float32)

    # A finger projected into the object mask but absent from the visible hand mask is occluded
    # by the object. This is the metric/mask cue that a hidden finger belongs on the camera-far
    # surface; it avoids inheriting EgoForce's front-side depth initialization.
    finger_occlusion = np.zeros_like(finger_gap, dtype=np.float32)
    finger_visible = np.zeros_like(finger_gap, dtype=np.float32)
    with torch.no_grad():
        projected_q = project_points_torch(depth_geometry["vertices_ct"], intrinsics) / scale
    projected_q = torch.round(projected_q).long().cpu().numpy()
    for local, frame in enumerate(frame_ids):
        hand_mask_q = (
            np.asarray(
                Image.open(workspace / f"outputs/02_hand_masks/combined/{frame:06d}.png")
            )[::scale, ::scale][:height, :width]
            > 127
        )
        object_mask_q = (
            np.asarray(
                Image.open(object_mask_dir / f"{frame:06d}.png")
            )[::scale, ::scale][:height, :width]
            > 127
        )
        for finger_index, finger in enumerate(finger_names):
            px = projected_q[local, region_ids[finger].cpu().numpy()]
            valid = (
                (px[:, 0] >= 0)
                & (px[:, 0] < width)
                & (px[:, 1] >= 0)
                & (px[:, 1] < height)
            )
            if not valid.any():
                continue
            x, y = px[valid, 0], px[valid, 1]
            visible = hand_mask_q[y, x]
            object_only = object_mask_q[y, x] & ~visible
            finger_visible[local, finger_index] = float(visible.mean())
            finger_occlusion[local, finger_index] = float(object_only.mean())

    fused_score = (
        0.15 * vlm_prior
        + 0.35 * geometry_score
        + 0.20 * temporal_score
        + 0.10 * overlap_score[:, None]
        + 0.10 * object_motion_score[:, None]
        + 0.10 * np.clip(finger_occlusion / 0.35, 0.0, 1.0)
    )

    finger_active_np = np.zeros_like(fused_score, dtype=bool)
    for finger_index in range(len(finger_names)):
        trusted = (
            (vlm_prior[:, finger_index] >= 0.55)
            & (fused_score[:, finger_index] >= args.trusted_fused_score)
            & (finger_gap[:, finger_index] < 0.015)
            & (
                (object_motion_score >= 0.25)
                | (finger_occlusion[:, finger_index] >= 0.20)
            )
        )
        if not trusted.any():
            continue
        candidate = (
            (fused_score[:, finger_index] >= 0.58)
            & (finger_gap[:, finger_index] < 0.018)
            & (relative_speed[:, finger_index] < 0.025)
        )
        start, end = int(np.flatnonzero(trusted)[0]), int(np.flatnonzero(trusted)[-1])
        finger_active_np[start : end + 1, finger_index] = trusted[start : end + 1] | candidate[start : end + 1]
        local = start - 1
        while local >= 0 and candidate[local]:
            finger_active_np[local, finger_index] = True
            local -= 1
        local = end + 1
        while local < len(frame_ids) and candidate[local]:
            finger_active_np[local, finger_index] = True
            local += 1

    strong_per_finger = (
        (fused_score >= args.strong_fused_score)
        & (finger_gap < 0.008)
        & (relative_speed < args.strong_relative_speed_m)
        & ((object_motion_score[:, None] >= 0.25) | (finger_occlusion >= 0.20))
    )
    strong_frame = strong_per_finger.sum(axis=1) >= 2
    runs: list[tuple[int, int]] = []
    start = None
    for local, supported in enumerate(np.r_[strong_frame, False]):
        if supported and start is None:
            start = local
        elif not supported and start is not None:
            if local - start >= 3:
                runs.append((start, local - 1))
            start = None
    write_json(
        output / "contact_gate_diagnostics.json",
        {
            "policy": "Diagnostic values after depth-only hand alignment and before contact optimization.",
            "thresholds": {
                "overlap_normalization_pixels": args.overlap_normalization_pixels,
                "strong_fused_score": args.strong_fused_score,
                "trusted_fused_score": args.trusted_fused_score,
                "strong_finger_gap_mm": 8.0,
                "strong_relative_speed_mm_per_frame": args.strong_relative_speed_m * 1000.0,
                "minimum_strong_fingers_per_frame": 2,
                "minimum_consecutive_frames": 3,
            },
            "finger_names": finger_names,
            "frames": [
                {
                    "frame": int(frame),
                    "mask_overlap_pixels": int(overlap_pixels[local]),
                    "object_motion_score": float(object_motion_score[local]),
                    "finger_gap_mm": {
                        finger: float(finger_gap[local, index] * 1000.0)
                        for index, finger in enumerate(finger_names)
                    },
                    "relative_speed_mm_per_frame": {
                        finger: float(relative_speed[local, index] * 1000.0)
                        for index, finger in enumerate(finger_names)
                    },
                    "finger_occlusion": {
                        finger: float(finger_occlusion[local, index])
                        for index, finger in enumerate(finger_names)
                    },
                    "fused_score": {
                        finger: float(fused_score[local, index])
                        for index, finger in enumerate(finger_names)
                    },
                    "strong_fingers": [
                        finger
                        for index, finger in enumerate(finger_names)
                        if strong_per_finger[local, index]
                    ],
                }
                for local, frame in enumerate(frame_ids)
            ],
        },
    )
    if not runs:
        raise RuntimeError("No temporally persistent multi-finger firm-contact segment")
    global_start, global_end = max(runs, key=lambda run: run[1] - run[0])
    global_firm_contact = np.zeros(len(frame_ids), dtype=bool)
    global_firm_contact[global_start : global_end + 1] = True
    finger_active_np &= global_firm_contact[:, None]

    active_counts = finger_active_np.sum(axis=1)
    anchor_candidates = np.flatnonzero(active_counts >= 2)
    if len(anchor_candidates) == 0:
        anchor_candidates = np.flatnonzero(active_counts >= 1)
    if len(anchor_candidates) == 0:
        raise RuntimeError("VLM/depth/motion fusion found no reliable hand-object contact")
    anchor_quality = np.asarray(
        [
            fused_score[local, finger_active_np[local]].mean()
            if finger_active_np[local].any()
            else -np.inf
            for local in range(len(frame_ids))
        ]
    )
    anchor_local = int(anchor_candidates[np.argmax(anchor_quality[anchor_candidates])])
    anchor_frame = int(frame_ids[anchor_local])
    contact_fingers = [
        finger_names[index]
        for index in np.flatnonzero(finger_active_np[anchor_local])
    ]

    # Let metric depth choose front/back independently for every active finger at the anchor.
    anchor_vertices_ct = depth_geometry["vertices_ct"][anchor_local]
    anchor_uv_q = project_points_torch(anchor_vertices_ct, intrinsics) / scale
    contact_correspondences: dict[str, dict[str, object]] = {}
    fx_q, fy_q = float(intrinsics["fx"]) / scale, float(intrinsics["fy"]) / scale
    cx_q, cy_q = float(intrinsics["cx"]) / scale, float(intrinsics["cy"]) / scale
    for finger in contact_fingers:
        ids = region_ids[finger]
        uv = anchor_uv_q[ids]
        rounded = torch.round(uv).long()
        inside = (
            (rounded[:, 0] >= 0)
            & (rounded[:, 0] < width)
            & (rounded[:, 1] >= 0)
            & (rounded[:, 1] < height)
        )
        candidate_ids = ids[inside]
        candidate_uv = uv[inside]
        candidate_px = rounded[inside]
        front = proxy_front[anchor_local, candidate_px[:, 1], candidate_px[:, 0]]
        back = proxy_back[anchor_local, candidate_px[:, 1], candidate_px[:, 0]]
        valid = torch.isfinite(front) & torch.isfinite(back) & ((back - front) > 0.012)
        candidate_ids, candidate_uv, front, back = (
            candidate_ids[valid], candidate_uv[valid], front[valid], back[valid]
        )
        if len(candidate_ids) < 3:
            continue
        vertex_z = anchor_vertices_ct[candidate_ids, 2]
        front_error = torch.abs(vertex_z - front)
        back_error = torch.abs(vertex_z - back)
        front_score = float(torch.topk(front_error, min(3, len(front_error)), largest=False).values.mean().cpu())
        back_score = float(torch.topk(back_error, min(3, len(back_error)), largest=False).values.mean().cpu())
        finger_index = finger_names.index(finger)
        occlusion_fraction = float(finger_occlusion[anchor_local, finger_index])
        side = "back" if occlusion_fraction >= 0.20 else (
            "front" if front_score <= back_score else "back"
        )
        surface_error = front_error if side == "front" else back_error
        chosen = torch.topk(surface_error, min(3, len(surface_error)), largest=False).indices
        selected_ids = candidate_ids[chosen]
        selected_uv = candidate_uv[chosen]
        selected_front = front[chosen]
        selected_back = back[chosen]
        selected_surface = selected_front if side == "front" else selected_back
        target_z = selected_surface + (-0.0015 if side == "front" else 0.0015)
        target_ct = torch.stack(
            (
                (selected_uv[:, 0] - cx_q) * target_z / fx_q,
                (selected_uv[:, 1] - cy_q) * target_z / fy_q,
                target_z,
            ),
            dim=-1,
        )
        target_c0 = transform_points_torch(
            target_ct[None], t_c0_from_ct[anchor_local : anchor_local + 1]
        )[0]
        target_object = transform_points_torch(
            target_c0[None], t_object_from_c0[anchor_local : anchor_local + 1]
        )[0]
        contact_correspondences[finger] = {
            "hand_vertex_indices": selected_ids.detach().cpu().tolist(),
            "finger_region_vertex_indices": ids.detach().cpu().tolist(),
            "target_points_object": target_object.detach().cpu().tolist(),
            "anchor_uv_quarter": selected_uv.detach().cpu().tolist(),
            "front_depth_m": selected_front.detach().cpu().tolist(),
            "back_depth_m": selected_back.detach().cpu().tolist(),
            "selected_surface_depth_m": selected_surface.detach().cpu().tolist(),
            "side": side,
            "front_fit_error_m": front_score,
            "back_fit_error_m": back_score,
            "object_occlusion_fraction": occlusion_fraction,
            "visible_hand_fraction": float(finger_visible[anchor_local, finger_index]),
        }
    if not contact_correspondences:
        raise RuntimeError(f"No metric front/back surface candidates at fused anchor {anchor_frame}")
    contact_fingers = list(contact_correspondences)
    finger_active_np = finger_active_np[:, [finger_names.index(finger) for finger in contact_fingers]]
    fused_score_selected = fused_score[:, [finger_names.index(finger) for finger in contact_fingers]]
    gap_selected = finger_gap[:, [finger_names.index(finger) for finger in contact_fingers]]
    speed_selected = relative_speed[:, [finger_names.index(finger) for finger in contact_fingers]]
    prior_selected = vlm_prior[:, [finger_names.index(finger) for finger in contact_fingers]]
    write_json(output / "anchor_contact_correspondences.json", contact_correspondences)

    target_trajectory = {
        finger: torch.full(
            (len(frame_ids), len(data["hand_vertex_indices"]), 3),
            float("nan"),
            device=device,
        )
        for finger, data in contact_correspondences.items()
    }
    def metric_surface_targets(local: int, finger: str) -> torch.Tensor | None:
        ids = region_ids[finger]
        vertices_ct = depth_geometry["vertices_ct"][local]
        uv = project_points_torch(vertices_ct[ids], intrinsics) / scale
        px = torch.round(uv).long()
        valid = (
            (px[:, 0] >= 0)
            & (px[:, 0] < width)
            & (px[:, 1] >= 0)
            & (px[:, 1] < height)
        )
        ids, uv, px = ids[valid], uv[valid], px[valid]
        front = proxy_front[local, px[:, 1], px[:, 0]]
        back = proxy_back[local, px[:, 1], px[:, 0]]
        ray_valid = torch.isfinite(front) & torch.isfinite(back) & ((back - front) > 0.012)
        ids, uv, front, back = ids[ray_valid], uv[ray_valid], front[ray_valid], back[ray_valid]
        if len(ids) < 3:
            return None
        side = str(contact_correspondences[finger]["side"])
        surface = front if side == "front" else back
        error = torch.abs(vertices_ct[ids, 2] - surface)
        chosen = torch.topk(error, min(3, len(error)), largest=False).indices
        uv, surface = uv[chosen], surface[chosen]
        target_z = surface + (-0.0015 if side == "front" else 0.0015)
        target_ct = torch.stack(
            (
                (uv[:, 0] - cx_q) * target_z / fx_q,
                (uv[:, 1] - cy_q) * target_z / fy_q,
                target_z,
            ),
            dim=-1,
        )
        target_c0 = transform_points_torch(
            target_ct[None], t_c0_from_ct[local : local + 1]
        )[0]
        return transform_points_torch(
            target_c0[None], t_object_from_c0[local : local + 1]
        )[0]

    for finger_index, finger in enumerate(contact_fingers):
        for local in np.flatnonzero(finger_active_np[:, finger_index]):
            targets = metric_surface_targets(int(local), finger)
            if targets is None:
                finger_active_np[local, finger_index] = False
            else:
                target_trajectory[finger][local] = targets

    propagation_supported = finger_active_np.any(axis=1)
    if not propagation_supported[anchor_local]:
        raise RuntimeError("Fused anchor failed metric side-consistency validation")
    # Optimize a small reliable core first, then propagate its residual frame by frame. Treating
    # the entire detected interval as one seed makes the optimizer average incompatible initial
    # finger depths instead of carrying a stable grasp solution through time.
    seed_start_local = max(global_start, anchor_local - 2)
    seed_end_local = min(global_end, anchor_local + 2)
    seed_start, seed_end = int(frame_ids[seed_start_local]), int(frame_ids[seed_end_local])

    write_json(
        output / "contact_target_trajectory.json",
        {
            str(int(frame_ids[local])): {
                finger: target_trajectory[finger][local].detach().cpu().tolist()
                for finger_index, finger in enumerate(contact_fingers)
                if finger_active_np[local, finger_index]
            }
            for local in np.flatnonzero(propagation_supported)
        },
    )
    write_json(
        output / "contact_fusion.json",
        {
            "policy": "VLM per-finger proposal + metric surface gap + object-local relative motion + mask overlap",
            "anchor_frame": anchor_frame,
            "contact_interval": [seed_start, seed_end],
            "multi_finger_strong_interval": [
                int(frame_ids[global_start]),
                int(frame_ids[global_end]),
            ],
            "contact_fingers": contact_fingers,
            "selected_sides": {
                finger: contact_correspondences[finger]["side"] for finger in contact_fingers
            },
            "frames": [
                {
                    "frame": int(frame),
                    "overlap_pixels": int(overlap_pixels[local]),
                    "object_motion_score": float(object_motion_score[local]),
                    "fingers": {
                        finger: {
                            "vlm_probability": float(prior_selected[local, finger_index]),
                            "surface_gap_m": float(gap_selected[local, finger_index]),
                            "object_local_speed_m_per_frame": float(speed_selected[local, finger_index]),
                            "fused_score": float(fused_score_selected[local, finger_index]),
                            "object_occlusion_fraction": float(
                                finger_occlusion[local, finger_names.index(finger)]
                            ),
                            "visible_hand_fraction": float(
                                finger_visible[local, finger_names.index(finger)]
                            ),
                            "firm_contact": bool(finger_active_np[local, finger_index]),
                        }
                        for finger_index, finger in enumerate(contact_fingers)
                    },
                }
                for local, frame in enumerate(frame_ids)
            ],
        },
    )
    finger_active = torch.as_tensor(finger_active_np, dtype=torch.bool, device=device)
    active_contact = torch.zeros(len(frame_ids), dtype=torch.bool, device=device)

    def contact_terms(
        current: dict[str, torch.Tensor],
        optimization_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        selected_contact = (
            active_contact
            if optimization_mask is None
            else active_contact & optimization_mask
        )
        terms = observation_losses(current, selected_contact)
        target_losses = []
        for finger_index, (finger, correspondence) in enumerate(contact_correspondences.items()):
            active_ids = torch.nonzero(
                selected_contact & finger_active[:, finger_index], as_tuple=False
            ).flatten()
            if len(active_ids) == 0:
                continue
            region = torch.as_tensor(
                correspondence["finger_region_vertex_indices"], dtype=torch.long, device=device
            )
            target = target_trajectory[finger][active_ids]
            points = current["vertices_object"][active_ids][:, region]
            # A real finger-object contact is an area constraint, not three permanently glued
            # MANO vertices. Require at least one surface point from each finger region to reach
            # one of the anchor-certified back-surface samples.
            error = torch.cdist(target, points).amin(dim=(-1, -2))
            target_losses.append(
                F.smooth_l1_loss(error, torch.zeros_like(error), beta=0.004) / 0.004
            )
        terms["back_contact"] = (
            torch.stack(target_losses).mean()
            if target_losses
            else current["vertices_object"].sum() * 0.0
        )
        active_ids = torch.nonzero(selected_contact, as_tuple=False).flatten()
        signed = sample_sdf_grid(
            sdf, current["vertices_object"][active_ids], sdf_origin, sdf_pitch
        )
        penetration = F.relu(0.0005 - signed)
        positive_penetration = penetration[penetration > 0]
        terms["penetration"] = (
            positive_penetration.mean() / 0.005
            if positive_penetration.numel()
            else penetration.sum() * 0.0
        )
        return terms

    contact_weights = {
        **depth_weights,
        "pointcloud": 4.0,
        "keypoint": 8.0,
        "back_contact": 12.0,
        "penetration": 10.0,
    }
    contact_optimizer = torch.optim.Adam(
        [
            {"params": [translation], "lr": 1.2e-4},
            {"params": [global6], "lr": 5e-5},
            {"params": [pose6], "lr": 7e-5},
        ]
    )

    def run_contact_steps(label: str, steps: int) -> None:
        for step in range(steps):
            current = geometry()
            terms = contact_terms(current)
            total = sum(contact_weights[key] * value for key, value in terms.items())
            contact_optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_([translation, global6, pose6], 5.0)
            contact_optimizer.step()
            with torch.no_grad():
                inactive = ~active_contact
                translation[inactive] = depth_translation[inactive]
                global6[inactive] = depth_global[inactive]
                pose6[inactive] = depth_pose[inactive]
            # Inactive frames were explicitly restored to the depth-aligned state above. Passing
            # active_contact here would overwrite their pose with raw EgoForce parameters.
            enforce_trust_region(None)
            if step % args.log_every == 0 or step == steps - 1:
                record = {"phase": label, "step": step, "total": float(total.detach()), "active_frames": int(active_contact.sum())}
                record.update({key: float(value.detach()) for key, value in terms.items()})
                history.append(record)
                print(f"[{label} {step + 1}/{steps} active={int(active_contact.sum())}] " + " ".join(f"{k}={v:.4f}" for k, v in record.items() if isinstance(v, float)), flush=True)

    active_contact[anchor_local] = True
    run_contact_steps("contact_anchor", args.anchor_steps)
    for frame in range(seed_start, seed_end + 1):
        local = int(np.where(frame_ids == frame)[0][0])
        active_contact[local] = bool(propagation_supported[local])
    run_contact_steps("contact_seed", args.seed_steps)

    # Expand alternately from the fused seed, copying the neighboring contact residual as initialization.
    left_frames = []
    for frame in range(seed_start - 1, int(frame_ids.min()) - 1, -1):
        local = int(np.where(frame_ids == frame)[0][0])
        if not propagation_supported[local]:
            break
        left_frames.append(frame)
    right_frames = []
    for frame in range(seed_end + 1, int(frame_ids.max()) + 1):
        local = int(np.where(frame_ids == frame)[0][0])
        if not propagation_supported[local]:
            break
        right_frames.append(frame)
    propagation_order = []
    for offset in range(max(len(left_frames), len(right_frames))):
        if offset < len(left_frames):
            propagation_order.append(left_frames[offset])
        if offset < len(right_frames):
            propagation_order.append(right_frames[offset])
    propagated_frames = [
        int(frame_ids[local]) for local in torch.nonzero(active_contact, as_tuple=False).flatten().cpu().tolist()
    ]
    for frame in propagation_order:
        local = int(np.where(frame_ids == frame)[0][0])
        neighbor_frame = frame + 1 if frame < seed_start else frame - 1
        neighbor = int(np.where(frame_ids == neighbor_frame)[0][0])
        with torch.no_grad():
            translation[local] = depth_translation[local] + (translation[neighbor] - depth_translation[neighbor])
            global6[local] = depth_global[local] + (global6[neighbor] - depth_global[neighbor])
            pose6[local] = depth_pose[local] + (pose6[neighbor] - depth_pose[neighbor])
        active_contact[local] = True
        propagated_frames.append(frame)
        run_contact_steps(f"propagate_{frame}", args.propagation_steps_per_frame)

    # Collision cleanup is only allowed to adjust certified contact frames. Non-contact output
    # remains exactly at the depth-aligned solution, including frames that intersect the proxy
    # because the proxy is not a reliable constraint outside the hand-object interaction.
    with torch.no_grad():
        collision_probe = geometry()
        collision_signed = sample_sdf_grid(
            sdf, collision_probe["vertices_object"], sdf_origin, sdf_pitch
        )
        collision_active = active_contact & (
            collision_signed.min(dim=1).values < -0.0005
        )
    if bool(collision_active.any()):
        cleanup_translation = translation.detach().clone()
        cleanup_global = global6.detach().clone()
        cleanup_pose = pose6.detach().clone()
        collision_optimizer = torch.optim.Adam(
            [
                {"params": [translation], "lr": 8e-5},
                {"params": [global6], "lr": 3e-5},
                {"params": [pose6], "lr": 4e-5},
            ]
        )
        collision_ids = torch.nonzero(collision_active, as_tuple=False).flatten()
        cleanup_steps = 260
        for step in range(cleanup_steps):
            current = geometry()
            terms = contact_terms(current, collision_active)
            signed = sample_sdf_grid(
                sdf, current["vertices_object"][collision_ids], sdf_origin, sdf_pitch
            )
            collision_term = topk_penetration_loss(
                signed,
                clearance_m=0.0005,
                vertices_per_frame=24,
                frame_reduction="mean",
            )
            total = (
                4.0 * terms["pointcloud"]
                + 10.0 * terms["keypoint"]
                + 0.2 * terms["silhouette"]
                + 1.0 * terms["translation_anchor"]
                + 1.0 * terms["rotation_anchor"]
                + 1.0 * terms["pose_anchor"]
                + 16.0 * terms["back_contact"]
                + 20.0 * collision_term
            )
            collision_optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_([translation, global6, pose6], 5.0)
            collision_optimizer.step()
            with torch.no_grad():
                inactive = ~collision_active
                translation[inactive] = cleanup_translation[inactive]
                global6[inactive] = cleanup_global[inactive]
                pose6[inactive] = cleanup_pose[inactive]
            enforce_trust_region(None)
            if step % args.log_every == 0 or step == cleanup_steps - 1:
                history.append(
                    {
                        "phase": "collision_cleanup",
                        "step": step,
                        "active_frames": int(collision_active.sum()),
                        "total": float(total.detach()),
                        "penetration": float(collision_term.detach()),
                    }
                )
                print(
                    f"[collision_cleanup {step + 1}/{cleanup_steps} active={int(collision_active.sum())}] "
                    f"total={float(total.detach()):.4f} penetration={float(collision_term.detach()):.4f}",
                    flush=True,
                )

    with torch.no_grad():
        final = geometry()
        depth_vertices_c0_tensor = depth_geometry["vertices_c0"]
        depth_joints_c0_tensor = depth_geometry["joints_c0"]
        depth_arm_c0_tensor = transform_points_torch(
            depth_geometry["arm_vertices_ct"], t_c0_from_ct
        )
        depth_arm_joints_c0_tensor = transform_points_torch(
            depth_geometry["arm_joints_ct"], t_c0_from_ct
        )
        final_vertices_c0_tensor = final["vertices_c0"].clone()
        final_joints_c0_tensor = final["joints_c0"].clone()
        final_arm_c0_tensor = transform_points_torch(
            final["arm_vertices_ct"], t_c0_from_ct
        )
        final_arm_joints_c0_tensor = transform_points_torch(
            final["arm_joints_ct"], t_c0_from_ct
        )
        non_contact = ~active_contact
        non_contact_preexport_drift_m = (
            torch.linalg.norm(
                final_vertices_c0_tensor[non_contact]
                - depth_vertices_c0_tensor[non_contact],
                dim=-1,
            ).max()
            if bool(non_contact.any())
            else final_vertices_c0_tensor.new_tensor(0.0)
        )
        # Enforce the output contract explicitly instead of relying on optimizer restoration and
        # a second MANO forward pass being bit-identical.
        final_vertices_c0_tensor[non_contact] = depth_vertices_c0_tensor[non_contact]
        final_joints_c0_tensor[non_contact] = depth_joints_c0_tensor[non_contact]
        final_arm_c0_tensor[non_contact] = depth_arm_c0_tensor[non_contact]
        final_arm_joints_c0_tensor[non_contact] = depth_arm_joints_c0_tensor[non_contact]
        final_vertices_object = transform_points_torch(
            final_vertices_c0_tensor, t_object_from_c0
        )
        signed_final = sample_sdf_grid(
            sdf, final_vertices_object, sdf_origin, sdf_pitch
        ).cpu().numpy()
        depth_vertices_c0 = depth_vertices_c0_tensor.cpu().numpy()
        depth_joints_c0 = depth_joints_c0_tensor.cpu().numpy()
        depth_arm_c0 = depth_arm_c0_tensor.cpu().numpy()
        final_vertices_c0 = final_vertices_c0_tensor.cpu().numpy()
        final_joints_c0 = final_joints_c0_tensor.cpu().numpy()
        final_arm_c0 = final_arm_c0_tensor.cpu().numpy()
        final_arm_joints_c0 = final_arm_joints_c0_tensor.cpu().numpy()

    frames_manifest = []
    frame_to_local = {int(frame): local for local, frame in enumerate(frame_ids)}
    contact_ids_all = sorted(
        {
            vertex
            for correspondence in contact_correspondences.values()
            for vertex in correspondence["hand_vertex_indices"]
        }
    )
    for frame in range(output_frame_count):
        entry: dict[str, object] = {"frame": frame, "status": "no_valid_right_hand"}
        if frame in frame_to_local:
            local = frame_to_local[frame]
            depth_dir = output / f"depth_aligned_C0/frame_{frame:06d}"
            optimized_dir = output / f"optimized_C0/frame_{frame:06d}"
            depth_hand = depth_dir / "right_hand_depth_aligned_C0.obj"
            depth_arm = depth_dir / "right_arm_depth_aligned_C0.obj"
            optimized_hand = optimized_dir / "right_hand_optimized_C0.obj"
            optimized_arm = optimized_dir / "right_arm_optimized_C0.obj"
            export_obj(depth_hand, depth_vertices_c0[local], hand_faces_np)
            export_obj(depth_arm, depth_arm_c0[local], arm_faces_np)
            export_obj(optimized_hand, final_vertices_c0[local], hand_faces_np)
            export_obj(optimized_arm, final_arm_c0[local], arm_faces_np)
            geometry_path = optimized_dir / "right_hand_arm_optimized_C0.npz"
            np.savez_compressed(
                geometry_path,
                hand_vertices=final_vertices_c0[local],
                hand_joints=final_joints_c0[local],
                arm_vertices=final_arm_c0[local],
                arm_joints=final_arm_joints_c0[local],
                hand_faces=hand_faces_np,
                arm_faces=arm_faces_np,
                signed_distance_to_collision_proxy_m=signed_final[local],
                coordinate_frame=np.asarray("frame0_right_camera_opencv_rdf"),
            )
            contact = bool(active_contact[local])
            entry.update(
                {
                    "status": "completed",
                    "contact": contact,
                    "contact_role": "vlm_anchor" if frame == anchor_frame else ("propagated" if contact else "none"),
                    "hand_pointcloud_C0": str(output / f"hand_pointcloud_C0/{frame:06d}.npz"),
                    "depth_aligned_hand_C0": str(depth_hand),
                    "depth_aligned_arm_C0": str(depth_arm),
                    "optimized_hand_C0": str(optimized_hand),
                    "optimized_arm_C0": str(optimized_arm),
                    "optimized_geometry_C0": str(geometry_path),
                    "contact_vertex_indices": contact_ids_all if contact else [],
                }
            )
        frames_manifest.append(entry)

    write_json(output / "optimization_history.json", history)
    manifest = {
        "schema_version": 3,
        "type": "vlm_geometry_temporal_per_finger_contact_optimization",
        "status": "completed_pending_independent_qc",
        "coordinate_frame": "frame0_right_camera_opencv_rdf",
        "frame_count": output_frame_count,
        "optimization_end_frame_inclusive": output_frame_count - 1,
        "optimized_right_hand_frame_count": len(frame_ids),
        "vlm_anchor": anchor_info,
        "anchor_frame": anchor_frame,
        "vlm_seed_interval": [seed_start, seed_end],
        "propagated_contact_frames": sorted(set(propagated_frames)),
        "contact_fingers": contact_fingers,
        "selected_contact_sides": {
            finger: contact_correspondences[finger]["side"] for finger in contact_fingers
        },
        "contact_fusion": str(output / "contact_fusion.json"),
        "contact_correspondences": str(output / "anchor_contact_correspondences.json"),
        "contact_target_trajectory": str(output / "contact_target_trajectory.json"),
        "collision_proxy": proxy_manifest["collision_proxy"],
        "object_pose_policy": "Stage 08 object scale and pose frozen",
        "collision_correction_policy": (
            "Collision is optimized only through MANO parameters on active contact frames; "
            "no post-hoc per-vertex SDF projection is applied."
        ),
        "collision_cleanup_active_frames": frame_ids[collision_active.cpu().numpy()].tolist(),
        "collision_correction_max_mm": 0.0,
        "collision_corrected_vertex_count": 0,
        "posthoc_vertex_projection_enabled": False,
        "translation_trust_region_m": args.translation_trust_region_m,
        "non_contact_output_policy": "Exact depth-aligned MANO hand, joints, and arm geometry",
        "non_contact_preexport_drift_max_mm": float(
            non_contact_preexport_drift_m.cpu() * 1000.0
        ),
        "depth_alignment_policy": "SAM2 hand mask intersected with dilated Raw-hand raster; calibrated depth backprojected to Ct and fitted one-sided to visible MANO surface",
        "contact_policy": "Object-agnostic VLM per-finger proposals fused with metric surface gap, object-local relative motion, mask overlap, and depth-selected front/back surface hypotheses",
        "frames": frames_manifest,
    }
    write_json(output / "dynamic_manifest.json", manifest)
    print(
        json.dumps(
            {
                "anchor_frame": anchor_frame,
                "seed_interval": [seed_start, seed_end],
                "propagated_contact_frames": manifest["propagated_contact_frames"],
                "contact_fingers": contact_fingers,
                "selected_contact_sides": manifest["selected_contact_sides"],
                "point_count_median": int(np.median(point_counts)),
                "propagation_support_range": [
                    int(min(manifest["propagated_contact_frames"])),
                    int(max(manifest["propagated_contact_frames"])),
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
