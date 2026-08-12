#!/usr/bin/env python3
"""Independent QC for depth-first, far/back-surface Stage 11 refinement."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from pytorch3d.loss.point_mesh_distance import point_face_distance
from pytorch3d.structures import Meshes, Pointclouds
import trimesh

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vlm_sam2_recon.rigid_pipeline.common import read_json, write_json  # noqa: E402


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return mesh


def transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def project(points: np.ndarray, intrinsics: dict[str, float]) -> np.ndarray:
    z = np.maximum(points[:, 2], 1e-8)
    return np.column_stack(
        (
            float(intrinsics["fx"]) * points[:, 0] / z + float(intrinsics["cx"]),
            float(intrinsics["fy"]) * points[:, 1] / z + float(intrinsics["cy"]),
        )
    )


def point_distances(meshes: Meshes, pointclouds: Pointclouds) -> np.ndarray:
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
    return torch.sqrt(squared.clamp_min(1e-12)).reshape(len(meshes), -1).cpu().numpy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=PROJECT_ROOT / "run_rigid_20260715_215524",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def update_pipeline_state(
    workspace: Path,
    status: str,
    summary: dict[str, object],
) -> None:
    state_path = workspace / "pipeline_state.json"
    if not state_path.exists():
        return
    state = read_json(state_path)
    stage_name = "11_rigid_contact_optimization"
    matches = [item for item in state.get("stages", []) if item.get("stage") == stage_name]
    if len(matches) > 1:
        raise KeyError(f"Expected at most one state record for {stage_name}, found {len(matches)}")
    if matches:
        record = matches[0]
    else:
        record = {"stage": stage_name}
        state.setdefault("stages", []).append(record)
    record.update(
        {
            "status": status,
            "inputs": [
                str(workspace / "outputs/08_tracking"),
                str(workspace / "outputs/09_egoforce/dynamic_manifest.json"),
                str(workspace / "outputs/02_hand_masks"),
                str(workspace / "outputs/11_contact_optimization/vlm_contact_anchor.json"),
            ],
            "outputs": [str(workspace / "outputs/11_contact_optimization")],
            "notes": (
                "Rigid cup hand/contact optimization passed independent depth, projection, "
                f"surface-contact, and collision QC over {summary['frame_count']} frames."
                if status == "completed"
                else "Rigid cup hand/contact optimization needs revision; independent QC failed."
            ),
        }
    )
    write_json(state_path, state)


def main() -> None:
    args = parse_args()
    workspace = args.workspace.resolve()
    output = workspace / "outputs/11_contact_optimization"
    manifest = json.loads((output / "dynamic_manifest.json").read_text(encoding="utf-8"))
    correspondences = json.loads(Path(manifest["contact_correspondences"]).read_text(encoding="utf-8"))
    target_trajectory = json.loads(Path(manifest["contact_target_trajectory"]).read_text(encoding="utf-8"))
    camera = json.loads((workspace / "outputs/00_rgb_frames/camera.json").read_text(encoding="utf-8"))
    intrinsics = camera["rgb_intrinsics_right"]
    poses = np.load(workspace / "outputs/00_rgb_frames/poses.npz")["T_C0_from_Ct"]
    deltas = np.load(workspace / "outputs/08_tracking/Delta_C0_object_motion.npy")
    frame_entries = [entry for entry in manifest["frames"] if entry["status"] == "completed"]
    frame_ids = [int(entry["frame"]) for entry in frame_entries]
    hand_manifest = read_json(workspace / "outputs/09_egoforce/dynamic_manifest.json")
    hand_entries = {int(entry["frame"]): entry for entry in hand_manifest["frames"]}
    device = torch.device(args.device)

    raw_verts, depth_verts, optimized_verts, pointclouds = [], [], [], []
    faces_np = None
    raw_joints, depth_joints, optimized_joints, target_uvs = [], [], [], []
    optimized_npz = []
    for entry in frame_entries:
        frame = int(entry["frame"])
        raw = np.load(workspace / f"outputs/09_egoforce/raw_Ct/{frame:06d}_egoforce_meshes.npz")
        selected_side_index = int(hand_entries[frame].get("selected_raw_side_index", 1))
        if selected_side_index not in (0, 1):
            raise ValueError(f"Frame {frame} has invalid selected_raw_side_index={selected_side_index}")
        depth_mesh = load_mesh(Path(entry["depth_aligned_hand_C0"]))
        optimized = np.load(entry["optimized_geometry_C0"])
        pc = np.load(entry["hand_pointcloud_C0"])
        t_ct_from_c0 = np.linalg.inv(poses[frame])
        raw_verts.append(raw["hand_vertices"][selected_side_index])
        depth_verts.append(transform(np.asarray(depth_mesh.vertices), t_ct_from_c0))
        optimized_verts.append(transform(optimized["hand_vertices"], t_ct_from_c0))
        pointclouds.append(pc["points_Ct"])
        faces_np = raw["right_hand_faces"]
        raw_joints.append(raw["hand_joints"][selected_side_index])
        # Optimized NPZ contains final joints; depth joints are recovered from the exported depth hand only for mesh QC.
        optimized_joints.append(transform(optimized["hand_joints"], t_ct_from_c0))
        depth_joints.append(raw["hand_joints"][selected_side_index])
        target_uvs.append(raw["egoforce_hand_keypoints_2d"][selected_side_index])
        optimized_npz.append(optimized)

    faces = torch.as_tensor(faces_np, dtype=torch.long, device=device)
    def meshes(values: list[np.ndarray]) -> Meshes:
        return Meshes(
            verts=[torch.as_tensor(value, dtype=torch.float32, device=device) for value in values],
            faces=[faces for _ in values],
        )
    pcl = Pointclouds(
        points=[torch.as_tensor(value, dtype=torch.float32, device=device) for value in pointclouds]
    )
    with torch.no_grad():
        raw_distance = point_distances(meshes(raw_verts), pcl)
        depth_distance = point_distances(meshes(depth_verts), pcl)
        optimized_distance = point_distances(meshes(optimized_verts), pcl)

    contact_errors = []
    per_frame = []
    penetrating_frames_all = 0
    penetrating_contact_frames = 0
    for local, entry in enumerate(frame_entries):
        frame = int(entry["frame"])
        raw_uv = project(raw_joints[local], intrinsics)
        optimized_uv = project(optimized_joints[local], intrinsics)
        target_uv = target_uvs[local]
        signed = optimized_npz[local]["signed_distance_to_collision_proxy_m"]
        penetrating = int((signed < -0.0005).sum())
        has_penetration = penetrating > 0
        penetrating_frames_all += int(has_penetration)
        if entry["contact"]:
            penetrating_contact_frames += int(has_penetration)
        frame_contact_errors = {}
        if entry["contact"]:
            vertices_object = transform(
                optimized_npz[local]["hand_vertices"], np.linalg.inv(deltas[frame])
            )
            frame_targets = target_trajectory.get(str(frame), {})
            for finger, target_values in frame_targets.items():
                correspondence = correspondences[finger]
                region = np.asarray(
                    correspondence["finger_region_vertex_indices"], dtype=np.int64
                )
                target = np.asarray(target_values, dtype=np.float64)
                pairwise = np.linalg.norm(
                    target[:, None, :] - vertices_object[region][None, :, :], axis=-1
                )
                error = float(pairwise.min() * 1000.0)
                frame_contact_errors[finger] = error
                contact_errors.append(error)
        per_frame.append(
            {
                "frame": frame,
                "contact": bool(entry["contact"]),
                "raw_point_to_mesh_median_mm": float(np.median(raw_distance[local]) * 1000.0),
                "depth_point_to_mesh_median_mm": float(np.median(depth_distance[local]) * 1000.0),
                "optimized_point_to_mesh_median_mm": float(np.median(optimized_distance[local]) * 1000.0),
                "raw_keypoint_median_error_px": float(np.median(np.linalg.norm(raw_uv - target_uv, axis=1))),
                "optimized_keypoint_median_error_px": float(np.median(np.linalg.norm(optimized_uv - target_uv, axis=1))),
                "penetration_qc_required": bool(entry["contact"]),
                "penetrating_vertex_count_beyond_0_5mm": penetrating,
                "maximum_penetration_mm": float(max(0.0, -signed.min() * 1000.0)),
                "surface_contact_error_mm": frame_contact_errors,
            }
        )

    anchor = int(manifest["anchor_frame"])
    anchor_entry = next(entry for entry in per_frame if entry["frame"] == anchor)
    anchor_surface_errors = []
    anchor_opposite_surface_margins = []
    anchor_local = frame_ids.index(anchor)
    anchor_vertices_object = transform(
        optimized_npz[anchor_local]["hand_vertices"], np.linalg.inv(deltas[anchor])
    )
    for finger, correspondence in correspondences.items():
        region = np.asarray(correspondence["finger_region_vertex_indices"], dtype=np.int64)
        target_object = np.asarray(correspondence["target_points_object"], dtype=np.float64)
        pairwise = np.linalg.norm(
            target_object[:, None, :] - anchor_vertices_object[region][None, :, :], axis=-1
        )
        target_index, region_index = np.unravel_index(np.argmin(pairwise), pairwise.shape)
        actual_z = optimized_verts[anchor_local][region[region_index], 2]
        front = float(np.asarray(correspondence["front_depth_m"])[target_index])
        back = float(np.asarray(correspondence["back_depth_m"])[target_index])
        if correspondence.get("side") == "front":
            anchor_surface_errors.append(abs(actual_z - front) * 1000.0)
            anchor_opposite_surface_margins.append((back - actual_z) * 1000.0)
        else:
            anchor_surface_errors.append(abs(actual_z - back) * 1000.0)
            anchor_opposite_surface_margins.append((actual_z - front) * 1000.0)

    raw_pc = float(np.median(raw_distance) * 1000.0)
    depth_pc = float(np.median(depth_distance) * 1000.0)
    optimized_pc = float(np.median(optimized_distance) * 1000.0)
    raw_kp = float(np.median([item["raw_keypoint_median_error_px"] for item in per_frame]))
    optimized_kp = float(np.median([item["optimized_keypoint_median_error_px"] for item in per_frame]))
    summary = {
        "schema_version": 3,
        "passed": bool(
            depth_pc <= raw_pc
            and optimized_pc <= depth_pc + 1.0
            and optimized_kp <= raw_kp + 5.0
            and float(np.median(anchor_surface_errors)) < 12.0
            and float(np.median(anchor_opposite_surface_margins)) > 12.0
            and float(np.median(contact_errors)) < 10.0
            and float(np.percentile(contact_errors, 90)) < 25.0
            and penetrating_contact_frames == 0
        ),
        "frame_count": len(frame_ids),
        "contact_frame_count": sum(item["contact"] for item in per_frame),
        "raw_point_to_mesh_median_mm": raw_pc,
        "depth_aligned_point_to_mesh_median_mm": depth_pc,
        "optimized_point_to_mesh_median_mm": optimized_pc,
        "raw_keypoint_median_error_px": raw_kp,
        "optimized_keypoint_median_error_px": optimized_kp,
        "anchor_selected_surface_error_median_mm": float(np.median(anchor_surface_errors)),
        "anchor_opposite_surface_margin_median_mm": float(np.median(anchor_opposite_surface_margins)),
        "propagated_surface_contact_error_median_mm": float(np.median(contact_errors)),
        "propagated_surface_contact_error_p90_mm": float(np.percentile(contact_errors, 90)),
        "penetration_acceptance_scope": "contact_frames_only",
        "penetration_threshold_mm": 0.5,
        "penetrating_frame_count_beyond_0_5mm_all_frames": penetrating_frames_all,
        "penetrating_contact_frame_count_beyond_0_5mm": penetrating_contact_frames,
        "per_frame": per_frame,
    }
    write_path = output / "independent_qc.json"
    write_path.unlink(missing_ok=True)
    write_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    status = "completed" if summary["passed"] else "needs_revision"
    manifest["status"] = status
    manifest["independent_qc"] = str(write_path)
    manifest["independent_qc_summary"] = {
        key: value for key, value in summary.items() if key != "per_frame"
    }
    write_json(output / "dynamic_manifest.json", manifest)
    update_pipeline_state(workspace, status, summary)

    overlay_dir = output / "projection_overlays"
    overlay_dir.mkdir(exist_ok=True)
    for local, frame in enumerate(frame_ids):
        image = Image.open(
            workspace / f"outputs/00_rgb_frames/right_rgb_png/{frame:06d}.png"
        ).convert("RGB")
        draw = ImageDraw.Draw(image)
        for uv in project(raw_joints[local], intrinsics):
            x, y = map(float, uv)
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(50, 115, 255))
        for uv in project(optimized_joints[local], intrinsics):
            x, y = map(float, uv)
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(40, 220, 95), outline=(255, 255, 255))
        image.save(overlay_dir / f"{frame:06d}.jpg", quality=92)
    print(json.dumps({key: value for key, value in summary.items() if key != "per_frame"}, indent=2))
    if not summary["passed"]:
        raise RuntimeError("Stage 11 failed independent depth/contact-surface QC")


if __name__ == "__main__":
    main()
