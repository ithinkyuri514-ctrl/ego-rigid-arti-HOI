#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--track-output', type=Path, required=True)
    parser.add_argument('--mesh', type=Path, required=True)
    parser.add_argument('--joint-json', type=Path, required=True)
    parser.add_argument('--mask-dir', type=Path, required=True)
    parser.add_argument('--poses', type=Path, required=True)
    parser.add_argument('--camera', type=Path, default=None)
    parser.add_argument('--start-frame', type=int, default=6)
    parser.add_argument('--end-frame', type=int, default=15)
    parser.add_argument('--track-index', type=int, default=0)
    parser.add_argument('--angle-min-deg', type=float, default=-180.0)
    parser.add_argument('--angle-max-deg', type=float, default=180.0)
    parser.add_argument('--coarse-step-deg', type=float, default=2.0)
    parser.add_argument('--refine-step-deg', type=float, default=0.1)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--collision-base-mesh', type=Path, default=None)
    parser.add_argument('--collision-clearance-m', type=float, default=0.001)
    parser.add_argument('--hinge-exclusion-m', type=float, default=0.04)
    parser.add_argument('--collision-step-deg', type=float, default=0.25)
    parser.add_argument('--closing-direction', choices=['negative', 'positive'], default='negative')
    return parser.parse_args()


def read_joint(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text())
    joints = payload.get('joints', payload if isinstance(payload, list) else [])
    if isinstance(payload, dict) and 'origin_C0' in payload:
        joint = payload
    else:
        matches = [item for item in joints if item.get('name') == 'joint_15_14']
        joint = matches[0] if matches else joints[0]
    origin = np.asarray(joint.get('origin_C0', joint.get('origin_xyz')), dtype=np.float64)
    axis = np.asarray(joint.get('axis_C0', joint.get('axis_xyz')), dtype=np.float64)
    axis /= np.linalg.norm(axis)
    return origin, axis


def transform_mesh(vertices: np.ndarray, origin: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    rotation = Rotation.from_rotvec(axis * angle).as_matrix()
    return (vertices - origin) @ rotation.T + origin


def project_mesh(vertices: np.ndarray, faces: np.ndarray, pose_c0_from_ct: np.ndarray, intrinsics: dict, shape: tuple[int, int]) -> np.ndarray:
    del faces
    c0_vertices = np.concatenate([vertices, np.ones((len(vertices), 1))], axis=1)
    ct_vertices = (np.linalg.inv(pose_c0_from_ct) @ c0_vertices.T).T[:, :3]
    z = ct_vertices[:, 2]
    valid = z > 1e-5
    if not np.any(valid):
        return np.zeros(shape, dtype=bool)
    uv = np.column_stack((
        float(intrinsics['fx']) * ct_vertices[valid, 0] / z[valid] + float(intrinsics['cx']),
        float(intrinsics['fy']) * ct_vertices[valid, 1] / z[valid] + float(intrinsics['cy']),
    ))
    height, width = shape
    uv = np.rint(uv).astype(np.int32)
    uv[:, 0] = np.clip(uv[:, 0], 0, width - 1)
    uv[:, 1] = np.clip(uv[:, 1], 0, height - 1)
    hull = cv2.convexHull(uv.reshape(-1, 1, 2))
    rendered = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(rendered, hull, 1)
    return rendered.astype(bool)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / max(union, 1))


def radial_distance_to_axis(points: np.ndarray, origin: np.ndarray, axis: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.cross(points - origin, axis), axis=1)


def make_clearance_evaluator(base_vertices: np.ndarray, lid_vertices: np.ndarray, origin: np.ndarray, axis: np.ndarray, hinge_exclusion_m: float):
    base_keep = radial_distance_to_axis(base_vertices, origin, axis) >= hinge_exclusion_m
    lid_keep = radial_distance_to_axis(lid_vertices, origin, axis) >= hinge_exclusion_m
    sampled_base = base_vertices[base_keep]
    sampled_lid = lid_vertices[lid_keep]
    if len(sampled_base) > 12000:
        sampled_base = sampled_base[np.linspace(0, len(sampled_base) - 1, 12000, dtype=np.int64)]
    if len(sampled_lid) > 3000:
        sampled_lid = sampled_lid[np.linspace(0, len(sampled_lid) - 1, 3000, dtype=np.int64)]
    tree = cKDTree(sampled_base)

    def clearance(angle_rad: float) -> float:
        points = transform_mesh(sampled_lid, origin, axis, angle_rad)
        distances, _ = tree.query(points, k=1, workers=-1)
        return float(np.quantile(distances, 0.01))

    return clearance


def closing_collision_limit(previous_deg: float, closing_direction: float, lower_deg: float, upper_deg: float, clearance, clearance_m: float, step_deg: float) -> tuple[float, bool, float]:
    if closing_direction >= 0.0:
        values = np.arange(previous_deg, upper_deg + step_deg * 0.5, step_deg)
    else:
        values = np.arange(previous_deg, lower_deg - step_deg * 0.5, -step_deg)
    last_safe = previous_deg
    for value in values:
        if clearance(np.deg2rad(value)) < clearance_m:
            safe = float(last_safe)
            colliding = float(value)
            for _ in range(32):
                middle = 0.5 * (safe + colliding)
                if clearance(np.deg2rad(middle)) >= clearance_m:
                    safe = middle
                else:
                    colliding = middle
            contact = 0.5 * (safe + colliding)
            return float(contact), True, float(safe)
        last_safe = float(value)
    return float(last_safe), False, float(last_safe)


def signed_point_angle(point: np.ndarray, reference: np.ndarray, origin: np.ndarray, axis: np.ndarray) -> float:
    r0 = reference - origin
    r1 = point - origin
    r0 = r0 - axis * np.dot(axis, r0)
    r1 = r1 - axis * np.dot(axis, r1)
    r0 /= max(np.linalg.norm(r0), 1e-9)
    r1 /= max(np.linalg.norm(r1), 1e-9)
    return float(np.arctan2(np.dot(axis, np.cross(r0, r1)), np.dot(r0, r1)))


def optimize_angle(vertices, faces, pose, intrinsics, mask, origin, axis, initial, args, lower_bound_deg, upper_bound_deg):
    def score(angle):
        projected = project_mesh(transform_mesh(vertices, origin, axis, angle), faces, pose, intrinsics, mask.shape)
        return iou(projected, mask)

    coarse = np.arange(lower_bound_deg, upper_bound_deg + args.coarse_step_deg * 0.5, args.coarse_step_deg)
    coarse_scores = np.asarray([score(np.deg2rad(value)) for value in coarse])
    best_index = int(np.argmax(coarse_scores))
    center = coarse[best_index]
    lo = max(lower_bound_deg, center - args.coarse_step_deg)
    hi = min(upper_bound_deg, center + args.coarse_step_deg)
    fine = np.arange(lo, hi + args.refine_step_deg * 0.5, args.refine_step_deg)
    fine_scores = np.asarray([score(np.deg2rad(value)) for value in fine])
    index = int(np.argmax(fine_scores))
    return float(fine[index]), float(fine_scores[index]), float(score(initial))


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    camera_path = args.camera or args.workspace / 'outputs/00_rgb_frames/camera.json'
    camera = json.loads(camera_path.read_text())
    intrinsics = camera.get('rgb_intrinsics_selected', camera['rgb_intrinsics_right'])
    pose_data = np.load(args.poses)
    transforms_c0_from_ct = pose_data['T_C0_from_Ct']
    points_c0 = np.load(args.track_output / 'upper_left_track_3d_C0.npy')
    used_depth = np.load(args.track_output / 'upper_left_track_used_depth_m.npy')
    mesh = trimesh.load(args.mesh, force='mesh', process=False)
    base_path = args.collision_base_mesh or args.mesh.with_name('part_15.obj')
    base_mesh = trimesh.load(base_path, force='mesh', process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if len(vertices) > 512:
        vertices = vertices[np.linspace(0, len(vertices) - 1, 512, dtype=np.int64)]
    faces = np.empty((0, 3), dtype=np.int64)
    origin, axis = read_joint(args.joint_json)
    clearance = make_clearance_evaluator(np.asarray(base_mesh.vertices, dtype=np.float64), vertices, origin, axis, args.hinge_exclusion_m)
    reference = points_c0[args.start_frame, args.track_index]
    records = []
    optimized_angles = np.full(len(transforms_c0_from_ct), np.nan, dtype=np.float64)
    previous = 0.0
    contact_reached = False
    closing_direction = -1.0 if args.closing_direction == 'negative' else 1.0
    for frame in range(args.start_frame, args.end_frame + 1):
        mask_path = args.mask_dir / f'{frame:06d}.png'
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 0
        point = points_c0[frame, args.track_index]
        if not np.isfinite(point).all():
            initial = previous
            status = 'missing_lifted_point'
        else:
            initial = signed_point_angle(point, reference, origin, axis)
            status = 'lifted_3d_initial'
        previous_deg = float(np.rad2deg(previous))
        if contact_reached:
            collision_limit_deg = previous_deg
            last_safe_deg = previous_deg
            collision_stop = True
            initial_deg, best_iou, initial_iou = optimize_angle(
                vertices, faces, transforms_c0_from_ct[frame], intrinsics, mask, origin, axis, previous, args,
                previous_deg, previous_deg,
            )
            status = 'hold_contact_angle'
        else:
            collision_limit_deg, collision_stop, last_safe_deg = closing_collision_limit(
                previous_deg, closing_direction, args.angle_min_deg, args.angle_max_deg,
                clearance, args.collision_clearance_m, args.collision_step_deg,
            )
            if closing_direction < 0.0:
                lower_bound_deg = collision_limit_deg
                upper_bound_deg = min(previous_deg, args.angle_max_deg)
            else:
                lower_bound_deg = max(previous_deg, args.angle_min_deg)
                upper_bound_deg = collision_limit_deg
            initial_deg, best_iou, initial_iou = optimize_angle(
                vertices, faces, transforms_c0_from_ct[frame], intrinsics, mask, origin, axis, initial, args,
                lower_bound_deg, upper_bound_deg,
            )
            if collision_stop and abs(initial_deg - collision_limit_deg) < max(args.refine_step_deg * 2.0, 0.2):
                contact_reached = True
        optimized_angles[frame] = np.deg2rad(initial_deg)
        previous = optimized_angles[frame]
        records.append({
            'frame_index': frame,
            'status': 'hold_contact_angle' if contact_reached and frame > args.start_frame and abs(initial_deg - collision_limit_deg) < 0.2 else status,
            'lifted_depth_m': float(used_depth[frame, args.track_index]) if np.isfinite(used_depth[frame, args.track_index]) else None,
            'lifted_initial_angle_deg': float(np.rad2deg(initial)),
            'initial_mesh_iou': initial_iou,
            'optimized_angle_deg': initial_deg,
            'optimized_mesh_iou': best_iou,
            'collision_clearance_m': clearance(optimized_angles[frame]),
            'collision_limit_angle_deg': collision_limit_deg,
            'last_safe_angle_deg': last_safe_deg,
            'contact_delta_from_previous_deg': abs(collision_limit_deg - previous_deg),
            'collision_stop_applied': collision_stop,
        })
    np.save(args.output_dir / 'joint_angles_iou_optimized_rad.npy', optimized_angles)
    (args.output_dir / 'articulate_iou_optimization.json').write_text(json.dumps({
        'method': 'lift_3d_then_mesh_projection_mask_iou_1d_search',
        'mesh': str(args.mesh),
        'mask_dir': str(args.mask_dir),
        'track_output': str(args.track_output),
        'joint_origin_C0': origin.tolist(),
        'joint_axis_C0': axis.tolist(),
        'collision_base_mesh': str(base_path),
        'collision_clearance_threshold_m': args.collision_clearance_m,
        'hinge_exclusion_m': args.hinge_exclusion_m,
        'frames': records,
    }, indent=2) + '\n')
    print('frame initial_deg initial_iou optimized_deg optimized_iou depth')
    for record in records[args.start_frame:args.end_frame + 1]:
        print(record['frame_index'], f"{record['lifted_initial_angle_deg']:.3f}", f"{record['initial_mesh_iou']:.4f}", f"{record['optimized_angle_deg']:.3f}", f"{record['optimized_mesh_iou']:.4f}", record['lifted_depth_m'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
