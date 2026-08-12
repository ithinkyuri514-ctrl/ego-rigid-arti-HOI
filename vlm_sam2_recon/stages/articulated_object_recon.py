"""Workspace/state helpers for a timestamped articulated-object run.

This module deliberately owns only the run contract. Heavy model stages stay in
their existing inspectable scripts and receive this workspace explicitly.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


ARTICULATED_STAGE_ORDER = [
    "00_rgb_extract",
    "01_vlm_event_slicing",
    "02_hand_masks",
    "03_diffueraser_hand_removal",
    "04_sam2_object_and_part_masks",
    "05_hunyuan_mesh",
    "06_particulate_parts_and_joint",
    "07_dense_depth_metric_calibration",
    "08_frame0_whole_part_icp_alignment",
    "09_viser_pre_cotracker_inspection",
    "10_cotracker3_articulated_motion",
    "11_egoforce_hand_frame0",
    "12_contact_fusion_and_optimization",
    "13_final_viser",
]


@dataclass
class ArticulatedWorkspaceConfig:
    run_id: str
    workspace_dir: str
    video_path: str
    spatial_export_root: str
    project_root: str = "/code/vlm_sam2_recon"
    camera_id: str = "left"
    world_frame: str = "frame0_left_camera"
    tracker_fps: float = 15.0
    diffueraser_root: str = "/code/ArtHOI-4D-Reconstruction/third_party/diffueraser"
    sam2_root: str = "/code/sam2"
    vda_root: str = "/code/ArtHOI-4D-Reconstruction/third_party/Video-Depth-Anything"
    vda_checkpoint: str = "/code/ArtHOI-4D-Reconstruction/checkpoints/video_depth_anything_vitl.pth"
    egoforce_root: str = "/code/EgoForce"
    notes: list[str] = field(default_factory=list)

    @property
    def workspace_path(self) -> Path:
        return Path(self.workspace_dir).expanduser().resolve()


def articulated_workspace_dirs(workspace: Path) -> dict[str, Path]:
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
        "particulate": outputs / "06_particulate",
        "dense_depth": outputs / "07_dense_depth",
        "alignment": outputs / "08_alignment",
        "visualization": outputs / "09_visualization",
        "tracking": outputs / "10_tracking",
        "egoforce": outputs / "11_egoforce",
        "contact": outputs / "12_contact_optimization",
        "final_visualization": outputs / "13_visualization",
    }


def make_initial_state(config: ArticulatedWorkspaceConfig) -> dict[str, Any]:
    return {
        "run_id": config.run_id,
        "type": "articulated_object_interaction_reconstruction",
        "workspace_dir": str(config.workspace_path),
        "video_path": str(Path(config.video_path).resolve()),
        "spatial_export_root": str(Path(config.spatial_export_root).resolve()),
        "world_frame": config.world_frame,
        "camera_id": config.camera_id,
        "tracker_fps": config.tracker_fps,
        "event_policy": {
            "vlm_is_coarse_router": True,
            "event_indices_are_global": True,
            "event_windows_keep_context_padding": True,
            "same_object_instance_across_events": True,
        },
        "coordinate_policy": {
            "all_outputs_in": config.world_frame,
            "camera_motion_compensation": "Apply interpolated high-rate head/camera pose before 3D motion fitting.",
            "articulated_motion": "Fit observed door motion to one fixed revolute joint; do not export free per-frame SE(3).",
        },
        "depth_policy": {
            "true_depth": "SpatialMP4 depth is sparse metric data and must be timestamp matched.",
            "dense_depth": "Video-Depth-Anything is calibrated against this run's true depth anchors.",
        },
        "stop_before": "10_cotracker3_articulated_motion",
        "stages": [
            {"stage": stage, "status": "pending", "inputs": [], "outputs": [], "notes": ""}
            for stage in ARTICULATED_STAGE_ORDER
        ],
    }


def initialize_articulated_workspace(config: ArticulatedWorkspaceConfig, overwrite_state: bool = False) -> Path:
    workspace = config.workspace_path
    dirs = articulated_workspace_dirs(workspace)
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    state_path = workspace / "pipeline_state.json"
    if state_path.exists() and not overwrite_state:
        return state_path
    state_path.write_text(json.dumps(make_initial_state(config), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    config_path = workspace / "configs" / "articulated_recon_config.json"
    config_path.write_text(json.dumps(asdict(config), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return state_path
