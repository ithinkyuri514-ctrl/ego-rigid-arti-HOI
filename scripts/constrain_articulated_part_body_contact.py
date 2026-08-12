#!/usr/bin/env python3
"""Stop an articulated moving part at first geometric contact with its parent body."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-angles', type=Path, required=True)
    p.add_argument('--moving-mesh', type=Path, required=True)
    p.add_argument('--body-mesh', type=Path, required=True)
    p.add_argument('--joint-json', type=Path, required=True)
    p.add_argument('--start-frame', type=int, required=True)
    p.add_argument('--end-frame', type=int, required=True)
    p.add_argument('--closing-direction', choices=('negative', 'positive'), default='negative')
    p.add_argument('--clearance-m', type=float, default=0.001)
    p.add_argument('--clearance-quantile', type=float, default=0.001)
    p.add_argument('--hinge-exclusion-m', type=float, default=0.04)
    p.add_argument('--output-dir', type=Path, required=True)
    return p.parse_args()


def load_mesh(path):
    mesh = trimesh.load(path, force='mesh', process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)])
    return mesh


def load_joint(path):
    joint = json.loads(path.read_text())['joints'][0]
    origin = np.asarray(joint.get('origin_C0', joint.get('origin_xyz')), dtype=np.float64)
    axis = np.asarray(joint.get('axis_C0', joint.get('axis_xyz')), dtype=np.float64)
    axis /= np.linalg.norm(axis)
    return origin, axis


def radial_distance(points, origin, axis):
    return np.linalg.norm(np.cross(points - origin, axis), axis=1)


def main():
    args = parse_args()
    angles = np.load(args.input_angles.resolve()).astype(np.float64)
    moving = load_mesh(args.moving_mesh.resolve())
    body = load_mesh(args.body_mesh.resolve())
    origin, axis = load_joint(args.joint_json.resolve())
    body_vertices = np.asarray(body.vertices, dtype=np.float64)
    moving_vertices = np.asarray(moving.vertices, dtype=np.float64)
    body_vertices = body_vertices[radial_distance(body_vertices, origin, axis) >= args.hinge_exclusion_m]
    moving_vertices = moving_vertices[radial_distance(moving_vertices, origin, axis) >= args.hinge_exclusion_m]
    body_vertices = body_vertices[::max(1, len(body_vertices) // 30000)]
    moving_vertices = moving_vertices[::max(1, len(moving_vertices) // 12000)]
    tree = cKDTree(body_vertices)

    def clearance(angle):
        rotation = Rotation.from_rotvec(axis * float(angle)).as_matrix()
        points = (moving_vertices - origin) @ rotation.T + origin
        distances = tree.query(points, k=1)[0]
        return float(np.quantile(distances, args.clearance_quantile))

    direction = -1.0 if args.closing_direction == 'negative' else 1.0
    constrained = angles.copy()
    records = []
    contact_angle = None
    previous = float(angles[max(0, args.start_frame - 1)])
    for frame in range(len(angles)):
        candidate = float(angles[frame])
        status = 'outside_close_event'
        applied = candidate
        if frame < args.start_frame:
            previous = candidate
        elif contact_angle is not None:
            applied = contact_angle
            status = 'hold_body_contact_angle'
        elif frame <= args.end_frame:
            candidate_clearance = clearance(candidate)
            closing = direction * (candidate - previous) > 0.0
            if closing and candidate_clearance <= args.clearance_m:
                safe = previous
                colliding = candidate
                for _ in range(32):
                    middle = 0.5 * (safe + colliding)
                    if clearance(middle) > args.clearance_m:
                        safe = middle
                    else:
                        colliding = middle
                contact_angle = 0.5 * (safe + colliding)
                applied = contact_angle
                status = 'first_door_body_contact'
            else:
                status = 'tracked_close_angle'
            previous = applied
        else:
            applied = previous
            status = 'hold_terminal_close_angle'
        constrained[frame] = applied
        records.append({
            'frame_index': frame,
            'input_angle_deg': float(np.rad2deg(candidate)),
            'output_angle_deg': float(np.rad2deg(applied)),
            'door_body_clearance_m': clearance(applied),
            'status': status,
        })

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / 'joint_angles_body_contact_rad.npy', constrained)
    transforms = np.asarray([
        trimesh.transformations.rotation_matrix(float(angle), axis, origin) for angle in constrained
    ], dtype=np.float64)
    np.save(output / 'Delta_C0_part_motion_body_contact.npy', transforms)
    manifest = {
        'stage': 'articulated_part_parent_body_contact',
        'status': 'completed',
        'moving_part': str(args.moving_mesh.resolve()),
        'contact_body': str(args.body_mesh.resolve()),
        'contact_policy': 'moving door stops at first clearance threshold against microwave body; bottle is not a door stop',
        'start_frame': args.start_frame,
        'end_frame': args.end_frame,
        'clearance_m': args.clearance_m,
        'clearance_quantile': args.clearance_quantile,
        'hinge_exclusion_m': args.hinge_exclusion_m,
        'contact_frame': next((r['frame_index'] for r in records if r['status'] == 'first_door_body_contact'), None),
        'contact_angle_deg': None if contact_angle is None else float(np.rad2deg(contact_angle)),
        'frames': records,
    }
    (output / 'body_contact_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({
        'contact_frame': manifest['contact_frame'],
        'contact_angle_deg': manifest['contact_angle_deg'],
        'terminal_clearance_m': records[-1]['door_body_clearance_m'],
        'output_dir': str(output),
    }, indent=2))


if __name__ == '__main__':
    raise SystemExit(main())
