#!/usr/bin/env python3
"""Lock rigid-object orientation while preserving its tracked center trajectory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-poses', type=Path, required=True)
    parser.add_argument('--mesh', type=Path, required=True)
    parser.add_argument('--reference-frame', type=int, default=0)
    parser.add_argument('--event-start-frame', type=int, default=None)
    parser.add_argument('--event-end-frame', type=int, default=None)
    parser.add_argument('--output-poses', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    return parser.parse_args()


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force='mesh', process=False)
    if isinstance(mesh, trimesh.Scene):
        meshes = [geometry for geometry in mesh.geometry.values() if isinstance(geometry, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f'No triangle mesh found in {path}')
        mesh = trimesh.util.concatenate(meshes)
    return mesh


def main():
    args = parse_args()
    poses = np.load(args.input_poses.resolve()).astype(np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f'Expected [T,4,4], got {poses.shape}')
    if not np.isfinite(poses).all():
        raise ValueError('Input poses contain non-finite values')
    reference_frame = int(args.reference_frame)
    if not 0 <= reference_frame < len(poses):
        raise ValueError(f'Invalid reference frame {reference_frame}')

    mesh = load_mesh(args.mesh.resolve())
    center = np.asarray(mesh.vertices, dtype=np.float64).mean(axis=0)
    reference_rotation = poses[reference_frame, :3, :3].copy()

    tracked_centers = np.einsum('nij,j->ni', poses[:, :3, :3], center) + poses[:, :3, 3]
    constrained = np.repeat(np.eye(4, dtype=np.float64)[None], len(poses), axis=0)
    constrained[:, :3, :3] = reference_rotation
    constrained[:, :3, 3] = tracked_centers - center @ reference_rotation.T

    if (args.event_start_frame is None) != (args.event_end_frame is None):
        raise ValueError('Both event frame arguments must be set together')
    if args.event_start_frame is not None:
        event_start = int(args.event_start_frame)
        event_end = int(args.event_end_frame)
        if not 0 < event_start <= event_end < len(constrained):
            raise ValueError('Invalid event frame interval')
        constrained[:event_start] = constrained[reference_frame]
        constrained[event_end + 1:] = constrained[event_end]
    else:
        event_start = None
        event_end = None

    original_rotations = Rotation.from_matrix(poses[:, :3, :3])
    reference_rotation_object = Rotation.from_matrix(reference_rotation)
    removed_angles_deg = np.degrees((reference_rotation_object.inv() * original_rotations).magnitude())
    constrained_centers = (
        np.einsum('nij,j->ni', constrained[:, :3, :3], center) + constrained[:, :3, 3]
    )
    center_error = np.linalg.norm(constrained_centers - tracked_centers, axis=1)

    output_path = args.output_poses.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, constrained)

    manifest = {
        'stage': 'rigid_pose_translation_only_constraint',
        'status': 'completed',
        'input_poses': str(args.input_poses.resolve()),
        'mesh': str(args.mesh.resolve()),
        'output_poses': str(output_path),
        'frame_count': len(poses),
        'reference_frame': reference_frame,
        'event_interval': None if event_start is None else [event_start, event_end],
        'policy': (
            'lock reference orientation, preserve tracked mesh-center positions during the interaction, '
            'and hold the pose exactly fixed before and after the interaction'
        ),
        'mesh_center_local': center.tolist(),
        'reference_rotation': reference_rotation.tolist(),
        'max_removed_rotation_deg': float(removed_angles_deg.max()),
        'mean_removed_rotation_deg': float(removed_angles_deg.mean()),
        'max_center_preservation_error_m': float(center_error.max()),
    }
    manifest_path = args.manifest.resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    raise SystemExit(main())
