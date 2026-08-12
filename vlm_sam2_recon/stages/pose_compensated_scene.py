"""Export raw EgoForce hands into frame-0 coordinates beside a static laptop."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vlm_sam2_recon.stages.camera_alignment import (
    apply_se3_to_mesh,
    camera_to_camera_matrix,
    frame_row,
    load_frames_csv,
    write_json,
)
from vlm_sam2_recon.stages.contact_driven_screen import (
    BASE_PART_LABEL,
    SCREEN_PART_LABEL,
    camera_transform_for_resolved_pose,
    load_hand_frame,
    load_pose_timeline,
    read_json,
)
from vlm_sam2_recon.stages.screen_hinge_tracking import load_mesh


@dataclass
class PoseCompensatedSceneConfig:
    alignment_dir: Path
    export_root: Path
    rgb_dir: Path
    hand_dir: Path
    pose_csv: Path
    output_dir: Path
    start_frame: int = 0
    end_frame: int = 135
    fps: float = 15.0
    hand_side: str = "right"


def run_pose_compensated_scene(config: PoseCompensatedSceneConfig) -> dict[str, Any]:
    """Apply only camera-motion compensation; never optimize hand or laptop."""
    for key in ("alignment_dir", "export_root", "rgb_dir", "hand_dir", "pose_csv", "output_dir"):
        setattr(config, key, Path(getattr(config, key)).expanduser().resolve())
    if config.hand_side not in ("left", "right"):
        raise ValueError(f"Unsupported hand side: {config.hand_side}")
    if config.end_frame < config.start_frame:
        raise ValueError(f"Invalid frame range {config.start_frame}..{config.end_frame}")
    for path, label in (
        (config.alignment_dir, "alignment directory"),
        (config.export_root, "SpatialMP4 export"),
        (config.rgb_dir, "RGB directory"),
        (config.hand_dir, "EgoForce directory"),
        (config.pose_csv, "pose CSV"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    meta = read_json(config.export_root / "manifest.json")
    pose_timeline = load_pose_timeline(config.pose_csv)
    if pose_timeline is None:
        raise ValueError("A frame-indexed pose CSV is required for pose-only reconstruction")
    alignment_row = frame_row(config.export_root, 0)

    frames = list(range(config.start_frame, config.end_frame + 1))
    for frame in frames:
        pose_timeline.require_row(frame)

    frame_timestamps = {
        frame: float(pose_timeline.require_row(frame)["timestamp_s"])
        for frame in frames
    }
    depth_assignments: dict[int, dict[str, Any]] = {}
    max_depth_delta_s = 0.5 / max(float(config.fps), 1e-6) + 1e-6
    for row in load_frames_csv(config.export_root / "frames.csv"):
        depth_timestamp = float(row["depth_timestamp_s"])
        nearest_frame = min(frames, key=lambda frame: abs(frame_timestamps[frame] - depth_timestamp))
        delta_s = depth_timestamp - frame_timestamps[nearest_frame]
        if abs(delta_s) > max_depth_delta_s:
            continue
        candidate = {
            "frame": int(row["index"]),
            "timestamp_s": depth_timestamp,
            "tracker_timestamp_s": frame_timestamps[nearest_frame],
            "timestamp_delta_s": delta_s,
            "camera_to_alignment_matrix": np.linalg.inv(
                camera_to_camera_matrix(
                    meta,
                    alignment_row,
                    row,
                    camera="right",
                    view_pose_prefix="depth_pose",
                )
            ).tolist(),
        }
        previous = depth_assignments.get(nearest_frame)
        if previous is None or abs(delta_s) < abs(float(previous["timestamp_delta_s"])):
            depth_assignments[nearest_frame] = candidate

    base_mesh = load_mesh(config.alignment_dir / f"part_{BASE_PART_LABEL}_camera.obj")
    screen_mesh = load_mesh(config.alignment_dir / f"part_{SCREEN_PART_LABEL}_camera.obj")
    joints = read_json(config.alignment_dir / "joint_camera.json").get("joints", [])
    if not joints:
        raise ValueError(f"No joint found in {config.alignment_dir / 'joint_camera.json'}")
    joint = joints[0]

    static_dir = config.output_dir / "static_laptop"
    static_dir.mkdir(parents=True, exist_ok=True)
    base_path = static_dir / f"part_{BASE_PART_LABEL}_camera.obj"
    screen_path = static_dir / f"part_{SCREEN_PART_LABEL}_camera.obj"
    joint_path = static_dir / "joint_camera.json"
    base_mesh.export(base_path)
    screen_mesh.export(screen_path)
    write_json(joint_path, {"joints": [joint]})

    transforms = np.zeros((len(frames), 4, 4), dtype=np.float64)
    frame_entries: list[dict[str, Any]] = []
    detected_frames = 0
    for local_index, frame in enumerate(frames):
        # camera_transform_for_resolved_pose returns T_frame_from_frame0.
        # EgoForce vertices are in the current camera frame, so invert it.
        t_frame_from_frame0 = camera_transform_for_resolved_pose(
            meta,
            config.export_root,
            frame,
            pose_timeline,
        )
        t_frame0_from_frame = np.linalg.inv(t_frame_from_frame0)
        transforms[local_index] = t_frame0_from_frame
        hand = load_hand_frame(config.hand_dir, frame, config.hand_side)
        frame_dir = config.output_dir / f"frame_{frame:06d}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        entry: dict[str, Any] = {
            "frame": frame,
            "local_frame": local_index,
            "pose_frame": frame,
            "rgb_path": str(config.rgb_dir / f"{frame:06d}.png"),
            "angle_rad": 0.0,
            "angle_deg": 0.0,
            "base_mesh": str(base_path),
            "screen_mesh": str(screen_path),
            "joint_json": str(joint_path),
            "status": "raw_egoforce_pose_compensated",
            "camera_to_frame0_matrix": t_frame0_from_frame.tolist(),
            "hand_delta_m": [0.0, 0.0, 0.0],
            "hand_refine_mode": "none",
        }
        depth_assignment = depth_assignments.get(frame)
        if depth_assignment is not None:
            entry["standard_depth_frame"] = int(depth_assignment["frame"])
            entry["standard_depth_timestamp_s"] = float(depth_assignment["timestamp_s"])
            entry["standard_depth_timestamp_delta_s"] = float(depth_assignment["timestamp_delta_s"])
            entry["standard_depth_camera_to_frame0_matrix"] = depth_assignment[
                "camera_to_alignment_matrix"
            ]
        if hand.hand_mesh is not None:
            hand_world = apply_se3_to_mesh(hand.hand_mesh, t_frame0_from_frame)
            hand_path = frame_dir / f"{config.hand_side}_hand_pose_compensated.obj"
            hand_world.export(hand_path)
            entry[f"{config.hand_side}_hand_mesh"] = str(hand_path)
            detected_frames += 1
        if hand.arm_mesh is not None:
            arm_world = apply_se3_to_mesh(hand.arm_mesh, t_frame0_from_frame)
            arm_path = frame_dir / f"{config.hand_side}_arm_pose_compensated.obj"
            arm_world.export(arm_path)
            entry[f"{config.hand_side}_arm_mesh"] = str(arm_path)
        frame_entries.append(entry)

    transforms_path = config.output_dir / "camera_to_frame0_matrices.npy"
    np.save(transforms_path, transforms)
    manifest = {
        "type": "pose_compensated_raw_egoforce_static_laptop",
        "coordinate_frame": "alignment_frame0_right_camera",
        "alignment_dir": str(config.alignment_dir),
        "export_root": str(config.export_root),
        "rgb_dir": str(config.rgb_dir),
        "hand_dir": str(config.hand_dir),
        "output_dir": str(config.output_dir),
        "pose_csv": str(config.pose_csv),
        "fps": float(config.fps),
        "frame_indices": frames,
        "frames": frame_entries,
        "hand_side": config.hand_side,
        "base_part_label": BASE_PART_LABEL,
        "screen_part_label": SCREEN_PART_LABEL,
        "joint_align_camera": joint,
        "camera_to_frame0_matrices": str(transforms_path),
        "detected_hand_frames": detected_frames,
        "depth_display_mode": "standard_frames_only",
        "standard_depth_tracker_frames": sorted(depth_assignments),
        "standard_depth_frame_count": len(depth_assignments),
        "standard_depth_max_timestamp_delta_s": max_depth_delta_s,
        "motion_compensation": {
            "enabled": True,
            "source": str(config.pose_csv),
            "alignment_timestamp_s": float(alignment_row["rgb_pose_timestamp_s"]),
            "rgb_pose_image_rotation_deg": float(meta.get("rgb_pose_image_rotation_deg", -90.0)),
            "description": "Raw EgoForce camera-frame meshes transformed into export frame-0 right-camera coordinates.",
        },
        "optimization": {
            "enabled": False,
            "hand_object_contact": False,
            "hand_refinement": False,
            "screen_motion": False,
        },
        "notes": [
            "Laptop base, screen, and hinge remain fixed in their aligned frame-0 pose.",
            "Hand geometry is unchanged apart from the rigid camera-motion compensation transform.",
        ],
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in config.__dict__.items()},
    }
    write_json(config.output_dir / "dynamic_manifest.json", manifest)
    return manifest
