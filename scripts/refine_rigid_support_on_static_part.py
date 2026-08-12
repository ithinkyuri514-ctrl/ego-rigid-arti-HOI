#!/usr/bin/env python3
"""Refine a rigid trajectory against the first contacted surface of a static articulated part."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial import cKDTree


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--object-id', required=True)
    p.add_argument('--support-object-id', required=True)
    p.add_argument('--support-part-id', required=True)
    p.add_argument('--rigid-mesh', type=Path, required=True)
    p.add_argument('--rigid-poses', type=Path, required=True)
    p.add_argument('--support-mesh', type=Path, required=True)
    p.add_argument('--placement-start-frame', type=int, required=True)
    p.add_argument('--contact-threshold-m', type=float, default=0.005)
    p.add_argument('--clearance-m', type=float, default=0.0015)
    p.add_argument('--max-correction-m', type=float, default=0.04)
    p.add_argument('--output-dir', type=Path, required=True)
    return p.parse_args()


def load_mesh(path):
    mesh = trimesh.load(path, force='mesh', process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)])
    return mesh


def transform(vertices, pose):
    return vertices @ pose[:3, :3].T + pose[:3, 3]


def main():
    args = parse_args()
    rigid = load_mesh(args.rigid_mesh.resolve())
    support = load_mesh(args.support_mesh.resolve())
    poses = np.load(args.rigid_poses.resolve()).astype(np.float64)
    rigid_vertices = np.asarray(rigid.vertices, dtype=np.float64)
    support_vertices = np.asarray(support.vertices, dtype=np.float64)
    support_tree = cKDTree(support_vertices)
    sampled_indices = np.arange(0, len(rigid_vertices), max(1, len(rigid_vertices) // 12000))
    distances = []
    closest_records = []
    for frame, pose in enumerate(poses):
        moved = transform(rigid_vertices[sampled_indices], pose)
        nearest_distance, nearest_index = support_tree.query(moved, k=1)
        local = int(np.argmin(nearest_distance))
        distances.append(float(nearest_distance[local]))
        closest_records.append((moved[local], int(nearest_index[local])))
    contacts = [f for f in range(args.placement_start_frame, len(poses)) if distances[f] <= args.contact_threshold_m]
    if not contacts:
        raise RuntimeError(f'No support contact found; min={min(distances[args.placement_start_frame:]):.6f} m')
    contact_frame = contacts[0]
    rigid_contact_point, support_vertex_index = closest_records[contact_frame]
    support_point = support_vertices[support_vertex_index]
    face_indices = support.vertex_faces[support_vertex_index]
    face_indices = face_indices[face_indices >= 0]
    if not len(face_indices):
        raise RuntimeError('Contact support vertex has no incident faces')
    face_centers = support.triangles_center[face_indices]
    face_index = int(face_indices[np.argmin(np.linalg.norm(face_centers - rigid_contact_point, axis=1))])
    normal = np.array(support.face_normals[face_index], dtype=np.float64, copy=True)
    if np.dot(normal, rigid_contact_point - support_point) < 0.0:
        normal = -normal
    normal /= np.linalg.norm(normal)
    plane_height = float(np.dot(support_point, normal))
    target_height = plane_height + args.clearance_m

    corrected = poses.copy()
    records = []
    for frame, pose in enumerate(poses):
        moved = transform(rigid_vertices, pose)
        before = float(np.quantile(moved @ normal, 0.01))
        correction = 0.0
        active = frame >= contact_frame
        if active:
            correction = float(np.clip(target_height - before, -args.max_correction_m, args.max_correction_m))
            corrected[frame, :3, 3] += normal * correction
        after = transform(rigid_vertices, corrected[frame])
        after_height = float(np.quantile(after @ normal, 0.01))
        records.append({
            'frame_index': frame,
            'support_active': active,
            'contact_distance_before_m': distances[frame],
            'plane_gap_before_m': before - plane_height,
            'plane_gap_after_m': after_height - plane_height,
            'translation_correction_m': correction,
        })

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    pose_path = output / f'{args.object_id}_poses_support_refined.npy'
    np.save(pose_path, corrected)
    manifest = {
        'stage': 'rigid_on_articulated_static_part_support',
        'status': 'completed',
        'relation': {
            'supported_object': args.object_id,
            'support_object': args.support_object_id,
            'support_part': args.support_part_id,
            'relation_type': 'inside_and_supported_by_contacted_surface',
        },
        'first_contact_frame': contact_frame,
        'first_contact_distance_m': distances[contact_frame],
        'support_plane': {
            'point_C0': support_point.tolist(),
            'normal_C0_toward_rigid_object': normal.tolist(),
            'source_vertex_index': support_vertex_index,
            'source_face_index': face_index,
            'source_policy': 'actual_closest_surface_at_first_geometric_contact',
        },
        'params': {
            'placement_start_frame': args.placement_start_frame,
            'contact_threshold_m': args.contact_threshold_m,
            'clearance_m': args.clearance_m,
            'max_correction_m': args.max_correction_m,
        },
        'outputs': {'poses': str(pose_path)},
        'frames': records,
    }
    (output / 'support_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({
        'first_contact_frame': contact_frame,
        'first_contact_distance_m': distances[contact_frame],
        'support_point_C0': support_point.tolist(),
        'support_normal_C0': normal.tolist(),
        'output_poses': str(pose_path),
    }, indent=2))


if __name__ == '__main__':
    raise SystemExit(main())
