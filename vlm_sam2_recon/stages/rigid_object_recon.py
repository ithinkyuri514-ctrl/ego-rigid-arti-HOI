"""Scaffolding for egocentric rigid-object reconstruction runs.

This module intentionally contains only path/config/state helpers.  Heavy
stages such as VLM calls, DiffuEraser, SAM2, Hunyuan3D, Video-Depth-Anything,
ICP, EgoForce, and Viser should be implemented as separate, inspectable steps.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


RIGID_STAGE_ORDER = [
    "00_rgb_extract",
    "01_vlm_event_and_keyframes",
    "02_hand_masks",
    "03_diffueraser_hand_removal",
    "04_sam2_object_masks",
    "05_hunyuan_mesh",
    "06_dense_depth_metric_calibration",
    "07_frame0_mesh_alignment",
    "08_rigid_pose_tracking_frame0",
    "09_egoforce_hand_frame0",
    "10_viser_rgbd_playback",
]


@dataclass
class RigidWorkspaceConfig:
    """Serializable config for one rigid-object interaction sequence."""

    run_id: str
    workspace_dir: str
    video_path: str
    spatial_export_root: str
    project_root: str = "/code/vlm_sam2_recon"
    camera_id: str = "right"
    world_frame: str = "frame0_right_camera"
    tracker_fps: float = 15.0
    true_depth_fps: float = 4.0
    dense_depth_model: str = "/code/ArtHOI-4D-Reconstruction/third_party/Video-Depth-Anything"
    dense_depth_checkpoint: str = "/code/ArtHOI-4D-Reconstruction/checkpoints/video_depth_anything_vitl.pth"
    diffueraser_root: str = "/code/ArtHOI-4D-Reconstruction/third_party/diffueraser"
    sam2_root: str = "/code/sam2"
    hunyuan3d_stage: str = "vlm_sam2_recon/stages/hunyuan3d_local.py"
    egoforce_root: str = "/code/EgoForce"
    notes: list[str] = field(default_factory=list)

    @property
    def workspace_path(self) -> Path:
        return Path(self.workspace_dir).expanduser().resolve()

    @property
    def spatial_export_path(self) -> Path:
        return Path(self.spatial_export_root).expanduser().resolve()

    @property
    def video_file(self) -> Path:
        return Path(self.video_path).expanduser().resolve()


def load_rigid_config(path: Path) -> RigidWorkspaceConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    return RigidWorkspaceConfig(**data)


def rigid_workspace_dirs(workspace: Path) -> dict[str, Path]:
    outputs = workspace / "outputs"
    return {
        "configs": workspace / "configs",
        "docs": workspace / "docs",
        "inputs": workspace / "inputs",
        "scratch": workspace / "scratch",
        "logs": outputs / "logs",
        "rgb_frames": outputs / "00_rgb_frames",
        "vlm": outputs / "01_vlm",
        "hand_masks": outputs / "02_hand_masks",
        "diffueraser": outputs / "03_diffueraser",
        "object_masks": outputs / "04_object_masks",
        "hunyuan_mesh": outputs / "05_hunyuan_mesh",
        "dense_depth": outputs / "06_dense_depth",
        "alignment": outputs / "07_alignment",
        "tracking": outputs / "08_tracking",
        "egoforce": outputs / "09_egoforce",
        "visualization": outputs / "10_visualization",
    }


def make_stage_records() -> list[dict[str, Any]]:
    return [
        {
            "stage": stage,
            "status": "pending",
            "inputs": [],
            "outputs": [],
            "notes": "",
        }
        for stage in RIGID_STAGE_ORDER
    ]


def make_initial_state(config: RigidWorkspaceConfig) -> dict[str, Any]:
    return {
        "run_id": config.run_id,
        "type": "rigid_object_interaction_reconstruction",
        "workspace_dir": str(config.workspace_path),
        "video_path": str(config.video_file),
        "spatial_export_root": str(config.spatial_export_path),
        "world_frame": config.world_frame,
        "camera_id": config.camera_id,
        "tracker_fps": config.tracker_fps,
        "true_depth_fps": config.true_depth_fps,
        "target": {
            "object_id": None,
            "name": None,
            "category": None,
            "interaction_event": None,
            "keyframes": [],
        },
        "coordinate_policy": {
            "all_outputs_in": config.world_frame,
            "camera_motion_compensation": "Use high-rate head pose; never treat per-frame camera coordinates as a fixed world.",
            "rigid_motion": "Object pose is one SE(3) transform per frame in frame0_right_camera.",
        },
        "depth_policy": {
            "true_depth": "SpatialMP4 depth frames are metric but sparse in time.",
            "dense_depth": "Video-Depth-Anything fills per-video-frame depth; calibrate scale/shift to true depth frames before use.",
        },
        "stages": make_stage_records(),
    }


def initialize_rigid_workspace(config: RigidWorkspaceConfig, overwrite_state: bool = False) -> Path:
    workspace = config.workspace_path
    for path in rigid_workspace_dirs(workspace).values():
        path.mkdir(parents=True, exist_ok=True)

    state_path = workspace / "pipeline_state.json"
    if overwrite_state or not state_path.exists():
        state_path.write_text(
            json.dumps(make_initial_state(config), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return state_path
