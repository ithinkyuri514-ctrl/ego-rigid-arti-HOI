#!/usr/bin/env python3
"""Settle a rigid object onto an articulated part's first-contact surface plane."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial import cKDTree


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--mouse-mesh', type=Path, required=True)
    p.add_argument('--mouse-poses', type=Path, required=True)
    p.add_argument('--articulated-part', type=Path, required=True)
    p.add_argument('--joint-json', type=Path, required=True)
    p.add_argument('--iou-output', type=Path, required=True)
    p.add_argument('--placement-start-frame', type=int, default=18)
    p.add_argument('--contact-threshold-m', type=float, default=0.018)
    p.add_argument('--clearance-m', type=float, default=0.0015)
    p.add_argument('--max-correction-m', type=float, default=0.04)
    p.add_argument('--output-dir', type=Path, default=None)
    return p.parse_args()


def load_mesh(path):
    mesh = trimesh.load(path, force='mesh', process=False)
    if isinstance(mesh, trimesh.Scene):
        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        mesh = meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
        raise ValueError(f'Invalid mesh: {path}')
    return mesh


def transform_vertices(vertices, transform):
    return vertices @ transform[:3, :3].T + transform[:3, 3]


def load_joint(path):
    joint = json.loads(path.read_text())['joints'][0]
    origin = np.asarray(joint.get('origin_C0', joint.get('origin_xyz')), dtype=np.float64)
    axis = np.asarray(joint.get('axis_C0', joint.get('axis_xyz')), dtype=np.float64)
    axis /= np.linalg.norm(axis)
    return origin, axis


def load_lid_transforms(iou_output, origin, axis, count):
    payload = json.loads((iou_output / 'articulate_iou_optimization.json').read_text())
    records = {int(x['frame_index']): x for x in payload['frames']}
    transforms = np.repeat(np.eye(4, dtype=np.float64)[None], count, axis=0)
    for frame in range(count):
        record = records.get(frame, records[min(records, key=lambda k: abs(k - frame))])
        transforms[frame] = trimesh.transformations.rotation_matrix(
            np.deg2rad(float(record['optimized_angle_deg'])), axis, origin
        )
    return transforms


def min_vertex_distance(source, target):
    source = source[np.isfinite(source).all(axis=1)]
    target = target[np.isfinite(target).all(axis=1)]
    if len(source) == 0 or len(target) == 0:
        return float('inf')
    source = source[::max(1, len(source) // 12000)]
    target = target[::max(1, len(target) // 30000)]
    return float(cKDTree(target).query(source, k=1)[0].min())


def find_first_contact(mouse_vertices, mouse_poses, lid_vertices_by_frame, start, threshold):
    distances = []
    for frame, lid_vertices in enumerate(lid_vertices_by_frame):
        if not np.isfinite(mouse_poses[frame]).all():
            distances.append(float('inf'))
        else:
            distances.append(min_vertex_distance(transform_vertices(mouse_vertices, mouse_poses[frame]), lid_vertices))
    candidates = [f for f in range(max(0, start), len(distances)) if distances[f] <= threshold]
    if not candidates:
        raise RuntimeError(f'No first contact found; minimum distance={min(distances[start:]):.6f} m')
    frame = candidates[0]
    center = transform_vertices(mouse_vertices, mouse_poses[frame]).mean(axis=0)
    return frame, center, distances[frame], distances


def fit_contact_plane(lid_mesh, lid_transform, mouse_center):
    vertices = transform_vertices(lid_mesh.vertices, lid_transform)
    centroids = vertices[lid_mesh.faces].mean(axis=1)
    normals = lid_mesh.face_normals @ lid_transform[:3, :3].T
    _, candidates = cKDTree(centroids).query(mouse_center, k=min(128, len(centroids)))
    candidates = np.atleast_1d(candidates)
    vectors = mouse_center[None] - centroids[candidates]
    facing = np.sum(normals[candidates] * vectors, axis=1) > 0
    candidates = candidates[facing] if np.any(facing) else candidates
    face = int(candidates[np.argmin(np.linalg.norm(centroids[candidates] - mouse_center, axis=1))])
    point = centroids[face]
    normal = normals[face].astype(np.float64)
    if np.dot(normal, mouse_center - point) < 0:
        normal = -normal
    normal /= np.linalg.norm(normal)
    return point, normal, face


def main():
    args = parse_args()
    workspace = args.workspace.resolve()
    output = (args.output_dir or workspace / 'outputs/11_object_object_support_mouse_laptop').resolve()
    output.mkdir(parents=True, exist_ok=True)
    mouse_mesh = load_mesh(args.mouse_mesh.resolve())
    lid_mesh = load_mesh(args.articulated_part.resolve())
    poses = np.load(args.mouse_poses.resolve()).astype(np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f'Expected [T,4,4] poses, got {poses.shape}')
    origin, axis = load_joint(args.joint_json.resolve())
    lid_transforms = load_lid_transforms(args.iou_output.resolve(), origin, axis, len(poses))
    lid_vertices = [transform_vertices(lid_mesh.vertices, t) for t in lid_transforms]
    contact_frame, center, contact_distance, distances = find_first_contact(
        mouse_mesh.vertices, poses, lid_vertices, args.placement_start_frame, args.contact_threshold_m
    )
    plane_point, plane_normal, face = fit_contact_plane(lid_mesh, lid_transforms[contact_frame], center)
    plane_height = float(np.dot(plane_point, plane_normal))
    target_height = plane_height + args.clearance_m
    corrected = poses.copy()
    records = []
    for frame in range(len(poses)):
        if not np.isfinite(poses[frame]).all():
            records.append({'frame_index': frame, 'support_active': False, 'status': 'invalid_input_pose'})
            continue
        original = transform_vertices(mouse_mesh.vertices, poses[frame])
        before = float(np.quantile(original @ plane_normal, 0.01))
        active = frame >= contact_frame
        correction = 0.0
        if active:
            correction = float(np.clip(target_height - before, -args.max_correction_m, args.max_correction_m))
            corrected[frame, :3, 3] += plane_normal * correction
        after_vertices = transform_vertices(mouse_mesh.vertices, corrected[frame])
        after = float(np.quantile(after_vertices @ plane_normal, 0.01))
        records.append({
            'frame_index': frame,
            'support_active': active,
            'status': 'support_refined' if active else 'pre_contact_tracking',
            'contact_distance_before_m': distances[frame],
            'plane_gap_before_m': before - plane_height,
            'plane_gap_after_m': after - plane_height,
            'translation_correction_m': correction,
        })
    pose_path = output / 'mouse_poses_support_refined.npy'
    np.save(pose_path, corrected)
    manifest = {
        'stage': '11_object_object_support_refinement',
        'status': 'completed',
        'relation': {'supported_object': 'mouse', 'support_object': 'laptop', 'support_part': 'part_14', 'relation_type': 'on_top_of'},
        'first_contact_frame': contact_frame,
        'first_contact_distance_m': contact_distance,
        'support_plane': {
            'coordinate_frame': 'frame0_left_camera_opencv_rdf',
            'point_C0': plane_point.tolist(),
            'normal_C0_toward_supported_object': plane_normal.tolist(),
            'source_face_index': face,
            'source_policy': 'nearest_articulated_part_surface_at_first_contact',
            'locked_from_first_contact': True,
            'post_contact_pose_policy': 'per_frame_tracking_pose_plus_support_plane_correction',
        },
        'params': {'placement_start_frame': args.placement_start_frame, 'contact_threshold_m': args.contact_threshold_m, 'clearance_m': args.clearance_m, 'max_correction_m': args.max_correction_m},
        'inputs': {'mouse_mesh': str(args.mouse_mesh.resolve()), 'mouse_poses': str(args.mouse_poses.resolve()), 'articulated_part': str(args.articulated_part.resolve()), 'iou_output': str(args.iou_output.resolve())},
        'outputs': {'mouse_poses_support_refined': str(pose_path)},
        'frames': records,
    }
    manifest_path = output / 'support_manifest.json'
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'first_contact_frame': contact_frame, 'first_contact_distance_m': contact_distance, 'plane_point_C0': plane_point.tolist(), 'plane_normal_C0': plane_normal.tolist(), 'poses': str(pose_path), 'manifest': str(manifest_path)}, indent=2))


if __name__ == '__main__':
    raise SystemExit(main())
