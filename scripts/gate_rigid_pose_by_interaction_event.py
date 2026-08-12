#!/usr/bin/env python3
"""Hold a rigid pose still before and after its VLM interaction interval."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-poses', type=Path, required=True)
    parser.add_argument('--event-start-frame', type=int, required=True)
    parser.add_argument('--event-end-frame', type=int, required=True)
    parser.add_argument('--pre-reference-frame', type=int, default=0)
    parser.add_argument('--post-window-end-frame', type=int, default=None)
    parser.add_argument('--frame-count', type=int, default=None)
    parser.add_argument('--output-poses', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    return parser.parse_args()


def robust_final_pose(poses: np.ndarray, start: int, end: int) -> tuple[np.ndarray, list[int], list[int]]:
    frames = [frame for frame in range(start, end + 1) if np.isfinite(poses[frame]).all()]
    if not frames:
        raise ValueError(f'No finite pose in post-event window {start}:{end}')
    translations = np.asarray([poses[frame, :3, 3] for frame in frames], dtype=np.float64)
    median = np.median(translations, axis=0)
    residuals = np.linalg.norm(translations - median[None], axis=1)
    residual_median = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - residual_median)))
    threshold = max(0.004, residual_median + 3.5 * max(mad, 1e-6))
    keep = residuals <= threshold
    kept_frames = [frame for frame, accepted in zip(frames, keep) if accepted]
    rejected_frames = [frame for frame, accepted in zip(frames, keep) if not accepted]
    kept_translations = translations[keep]
    stable_translation = np.median(kept_translations, axis=0)
    rotations = Rotation.from_matrix(np.asarray([poses[frame, :3, :3] for frame in kept_frames]))
    quaternions = rotations.as_quat()
    reference = quaternions[0]
    quaternions[np.sum(quaternions * reference[None], axis=1) < 0.0] *= -1.0
    stable_quaternion = np.mean(quaternions, axis=0)
    stable_quaternion /= np.linalg.norm(stable_quaternion)
    stable_pose = np.eye(4, dtype=np.float64)
    stable_pose[:3, :3] = Rotation.from_quat(stable_quaternion).as_matrix()
    stable_pose[:3, 3] = stable_translation
    return stable_pose, kept_frames, rejected_frames


def main():
    args = parse_args()
    poses = np.load(args.input_poses.resolve()).astype(np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f'Expected [T,4,4], got {poses.shape}')
    source_frame_count = len(poses)
    if args.frame_count is not None:
        if args.frame_count < source_frame_count:
            poses = poses[:args.frame_count]
        elif args.frame_count > source_frame_count:
            poses = np.concatenate([poses, np.repeat(poses[-1][None], args.frame_count - source_frame_count, axis=0)], axis=0)
    start = int(args.event_start_frame)
    end = int(args.event_end_frame)
    post_end = len(poses) - 1 if args.post_window_end_frame is None else int(args.post_window_end_frame)
    if not 0 <= args.pre_reference_frame < start <= end < len(poses):
        raise ValueError('Invalid event or reference frame range')
    if not end <= post_end < len(poses):
        raise ValueError('Invalid post-event window end')
    pre_pose = poses[args.pre_reference_frame].copy()
    if not np.isfinite(pre_pose).all():
        raise ValueError('Pre-event reference pose is invalid')
    final_pose, kept_frames, rejected_frames = robust_final_pose(poses, end, post_end)
    gated = poses.copy()
    gated[:start] = pre_pose
    gated[end + 1:] = final_pose
    output_path = args.output_poses.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, gated)
    manifest = {
        'stage': 'rigid_pose_interaction_event_gate',
        'status': 'completed',
        'input_poses': str(args.input_poses.resolve()),
        'source_frame_count': source_frame_count,
        'output_frame_count': len(poses),
        'output_poses': str(output_path),
        'event_start_frame': start,
        'event_end_frame': end,
        'pre_event_policy': f'hold_frame_{args.pre_reference_frame:06d}_pose',
        'post_event_policy': 'hold_robust_terminal_pose',
        'post_event_estimation_window': [end, post_end],
        'post_event_kept_frames': kept_frames,
        'post_event_rejected_frames': rejected_frames,
        'pre_event_pose': pre_pose.tolist(),
        'terminal_pose': final_pose.tolist(),
    }
    manifest_path = args.manifest.resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({
        'event_interval': [start, end],
        'pre_event_fixed_frames': [0, start - 1],
        'post_event_fixed_frames': [end + 1, len(poses) - 1],
        'terminal_kept_frames': kept_frames,
        'terminal_rejected_frames': rejected_frames,
        'output_poses': str(output_path),
        'manifest': str(manifest_path),
    }, indent=2))


if __name__ == '__main__':
    raise SystemExit(main())
