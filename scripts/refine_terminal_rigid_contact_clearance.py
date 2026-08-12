#!/usr/bin/env python3
"""Find the terminal contact surface and enforce its clearance over a lookback window."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--object-id', required=True)
    parser.add_argument('--support-object-id', required=True)
    parser.add_argument('--rigid-mesh', type=Path, required=True)
    parser.add_argument('--rigid-poses', type=Path, required=True)
    parser.add_argument('--support-mesh', type=Path, required=True)
    parser.add_argument('--terminal-frame', type=int, required=True)
    parser.add_argument('--lookback-frames', type=int, default=0)
    parser.add_argument('--clearance-m', type=float, default=0.0015)
    parser.add_argument('--max-correction-m', type=float, default=0.01)
    parser.add_argument('--output-dir', type=Path, required=True)
    return parser.parse_args()


def load_mesh(path):
    mesh = trimesh.load(path, force='mesh', process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            [geometry for geometry in mesh.geometry.values() if isinstance(geometry, trimesh.Trimesh)]
        )
    return mesh


def transform(vertices, pose):
    return vertices @ pose[:3, :3].T + pose[:3, 3]


def main():
    args = parse_args()
    rigid = load_mesh(args.rigid_mesh.resolve())
    support = load_mesh(args.support_mesh.resolve())
    poses = np.load(args.rigid_poses.resolve()).astype(np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f'Expected [T,4,4] rigid poses, got {poses.shape}')

    terminal_frame = int(args.terminal_frame)
    lookback_frames = int(args.lookback_frames)
    if not 0 <= terminal_frame < len(poses):
        raise ValueError(f'Invalid terminal frame {terminal_frame}')
    if lookback_frames < 0:
        raise ValueError('lookback-frames must be non-negative')
    start_frame = max(0, terminal_frame - lookback_frames)

    rigid_vertices = np.asarray(rigid.vertices, dtype=np.float64)
    support_vertices = np.asarray(support.vertices, dtype=np.float64)
    sampled_indices = np.arange(0, len(rigid_vertices), max(1, len(rigid_vertices) // 20000))
    terminal_sample = transform(rigid_vertices[sampled_indices], poses[terminal_frame])
    distances, support_indices = cKDTree(support_vertices).query(terminal_sample, k=1)
    nearest_sample = int(np.argmin(distances))
    rigid_point = terminal_sample[nearest_sample]
    support_vertex_index = int(support_indices[nearest_sample])
    support_point = support_vertices[support_vertex_index]
    face_indices = support.vertex_faces[support_vertex_index]
    face_indices = face_indices[face_indices >= 0]
    face_centers = support.triangles_center[face_indices]
    face_index = int(face_indices[np.argmin(np.linalg.norm(face_centers - rigid_point, axis=1))])
    normal = np.array(support.face_normals[face_index], dtype=np.float64, copy=True)
    if np.dot(normal, rigid_point - support_point) < 0.0:
        normal = -normal
    normal /= np.linalg.norm(normal)

    corrected = poses.copy()
    frame_records = []
    for frame in range(start_frame, terminal_frame + 1):
        moved_vertices = transform(rigid_vertices, poses[frame])
        signed_distances = (moved_vertices - support_point) @ normal
        minimum_gap_before = float(signed_distances.min())
        requested_correction = max(float(args.clearance_m) - minimum_gap_before, 0.0)
        applied_correction = min(requested_correction, float(args.max_correction_m))
        corrected[frame, :3, 3] += normal * applied_correction
        minimum_gap_after = minimum_gap_before + applied_correction
        frame_records.append({
            'frame': frame,
            'minimum_plane_gap_before_m': minimum_gap_before,
            'requested_translation_correction_m': requested_correction,
            'applied_translation_correction_m': applied_correction,
            'minimum_plane_gap_after_m': minimum_gap_after,
            'correction_clipped': applied_correction + 1e-12 < requested_correction,
        })

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    pose_path = output / f'{args.object_id}_poses_terminal_contact_refined.npy'
    np.save(pose_path, corrected)
    terminal_record = frame_records[-1]
    manifest = {
        'stage': 'terminal_rigid_contact_surface_window_clearance',
        'status': 'completed',
        'relation': {
            'object': args.object_id,
            'support_object': args.support_object_id,
            'relation_type': 'placed_inside_container',
        },
        'terminal_frame': terminal_frame,
        'lookback_frames': lookback_frames,
        'corrected_frame_interval': [start_frame, terminal_frame],
        'contact_surface': {
            'point_C0': support_point.tolist(),
            'normal_C0_toward_object': normal.tolist(),
            'source_vertex_index': support_vertex_index,
            'source_face_index': face_index,
        },
        'nearest_distance_before_m': float(distances[nearest_sample]),
        'selected_terminal_contact_signed_gap_before_m': float(np.dot(rigid_point - support_point, normal)),
        'translation_correction_m': terminal_record['applied_translation_correction_m'],
        'target_clearance_m': float(args.clearance_m),
        'max_correction_m': float(args.max_correction_m),
        'policy': (
            'select the contacted terminal support plane once, then enforce the same plane clearance '
            'for the terminal frame and its lookback window using translation along the fixed normal'
        ),
        'frames': frame_records,
        'output_poses': str(pose_path),
    }
    manifest_path = output / 'terminal_contact_manifest.json'
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    raise SystemExit(main())
