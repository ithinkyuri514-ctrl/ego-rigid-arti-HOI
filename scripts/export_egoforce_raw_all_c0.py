#!/usr/bin/env python3
"""Export every raw-visible EgoForce hand candidate from stored C0 payloads."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SIDES = ('left', 'right')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    return parser.parse_args()


def export_obj(path: Path, vertices: np.ndarray, faces: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    with path.open('w', encoding='ascii') as handle:
        for x, y, z in vertices:
            handle.write(f'v {x:.8f} {y:.8f} {z:.8f}\n')
        for a, b, c in faces + 1:
            handle.write(f'f {int(a)} {int(b)} {int(c)}\n')


def main():
    args = parse_args()
    input_manifest_path = args.input_manifest.resolve()
    source = json.loads(input_manifest_path.read_text())
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    side_counts = {side: 0 for side in SIDES}
    for source_frame in source['frames']:
        frame = int(source_frame['frame'])
        c0_path = Path(source_frame['geometry_C0_npz'])
        payload = np.load(c0_path)
        raw_visible = np.asarray(payload['raw_visible_hand'], dtype=bool)
        frame_dir = output_dir / 'C0' / f'frame_{frame:06d}'
        entry = {
            'frame': frame,
            'timestamp_s': source_frame['timestamp_s'],
            'T_C0_from_Ct': source_frame['T_C0_from_Ct'],
            'source_geometry_C0_npz': str(c0_path.resolve()),
            'raw_detected_sides': [side for index, side in enumerate(SIDES) if raw_visible[index]],
            'status': 'completed' if raw_visible.any() else 'no_raw_egoforce_detection',
        }
        for side_index, side in enumerate(SIDES):
            if not raw_visible[side_index]:
                continue
            hand_path = frame_dir / f'{side}_hand_C0.obj'
            arm_path = frame_dir / f'{side}_arm_C0.obj'
            export_obj(hand_path, payload['hand_vertices'][side_index], payload[f'{side}_hand_faces'])
            export_obj(arm_path, payload['arm_vertices'][side_index], payload['arm_faces'])
            entry[f'{side}_hand_C0'] = str(hand_path)
            entry[f'{side}_arm_C0'] = str(arm_path)
            side_counts[side] += 1
        frames.append(entry)

    manifest = {
        'schema_version': 1,
        'type': 'egoforce_raw_all_pose_compensated_sequence',
        'candidate_policy': 'raw_egoforce_visible_sides_without_sam2_consistency_filter',
        'frame_count': len(frames),
        'detected_frame_count': sum(frame['status'] == 'completed' for frame in frames),
        'side_detected_frame_counts': side_counts,
        'coordinate_frame': source['coordinate_frame'],
        'raw_coordinate_frame': source['raw_coordinate_frame'],
        'transform_rule': source['transform_rule'],
        'source_manifest': str(input_manifest_path),
        'sam2_consistency_filter': False,
        'frames': frames,
    }
    manifest_path = output_dir / 'dynamic_manifest_raw_all.json'
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({
        'manifest': str(manifest_path),
        'frame_count': len(frames),
        'side_detected_frame_counts': side_counts,
        'sam2_consistency_filter': False,
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    raise SystemExit(main())
