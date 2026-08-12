"""Workspace contract for mixed articulated and rigid interaction reconstruction."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


MIXED_STAGE_ORDER = [
    "00_rgb_extract",
    "01_vlm_mixed_interactions",
    "02_hand_masks",
    "03_diffueraser_hand_removal",
    "04_sam2_object_masks",
    "05_sam3d_frame0_reconstruction",
    "06_dense_depth_metric_calibration",
    "07_frame0_multi_object_alignment",
    "08_viser_frame0_inspection",
]


@dataclass
class MixedWorkspaceConfig:
    run_id: str
    workspace_dir: str
    video_path: str
    spatial_export_root: str
    project_root: str = "/code/vlm_sam2_recon"
    camera_id: str = "right"
    world_frame: str = "frame0_right_camera"
    tracker_fps: float = 15.0
    qwen_model_path: str = "/code/models/Qwen3-VL-8B-Instruct"
    qwen_code_root: str = "/code/Qwen3-VL"
    sam2_root: str = "/code/sam2"
    sam3d_root: str = "/code/sam-3d-objects"
    vda_root: str = "/code/ArtHOI-4D-Reconstruction/third_party/Video-Depth-Anything"
    vda_checkpoint: str = "/code/ArtHOI-4D-Reconstruction/checkpoints/video_depth_anything_vitl.pth"
    notes: list[str] = field(default_factory=list)

    @property
    def workspace_path(self) -> Path:
        return Path(self.workspace_dir).expanduser().resolve()


def mixed_workspace_dirs(workspace: Path) -> dict[str, Path]:
    outputs = workspace / "outputs"
    return {
        "configs": workspace / "configs",
        "docs": workspace / "docs",
        "inputs": workspace / "inputs",
        "scratch": workspace / "scratch",
        "logs": outputs / "logs",
        "rgb_frames": outputs / "00_rgb_frames",
        "vlm": outputs / "01_vlm",
        "masks": outputs / "02_sam2_frame0_masks",
        "sam3d": outputs / "03_sam3d_frame0",
        "dense_depth": outputs / "06_dense_depth",
        "alignment": outputs / "07_alignment",
        "visualization": outputs / "08_visualization",
    }


def make_initial_state(config: MixedWorkspaceConfig) -> dict[str, Any]:
    return {
        "run_id": config.run_id,
        "type": "mixed_articulated_and_rigid_interaction_reconstruction",
        "workspace_dir": str(config.workspace_path),
        "video_path": str(Path(config.video_path).resolve()),
        "spatial_export_root": str(Path(config.spatial_export_root).resolve()),
        "camera_id": "right",
        "world_frame": "frame0_right_camera",
        "modeling_frame_index": 0,
        "event_policy": {
            "vlm_uses_full_right_eye_timeline": True,
            "split_into_articulated_and_rigid_interaction_events": True,
            "event_segments_route_semantics_only": True,
            "all_object_mesh_prompts_use_global_frame_zero": True,
            "sam2_hand_masks_are_human_interactive": True,
            "diffueraser_removes_hands_before_object_sam2": True,
            "sam2_object_masks_are_human_interactive_on_hand_removed_video": True,
        },
        "coordinate_policy": {
            "all_outputs_in": "frame0_right_camera_opencv_rdf",
            "sam3d_pose_is_initialization_only": True,
            "sam3d_camera_conversion": "PyTorch3D x-left/y-up/z-forward to OpenCV x-right/y-down/z-forward",
            "refinement": "Frame-0 metric RGB-D point cloud ICP followed by silhouette IoU refinement.",
        },
        "stages": [
            {"stage": stage, "status": "pending", "inputs": [], "outputs": [], "notes": ""}
            for stage in MIXED_STAGE_ORDER
        ],
    }


def compatible_rigid_config(config: MixedWorkspaceConfig) -> dict[str, Any]:
    workspace = config.workspace_path
    spatial = Path(config.spatial_export_root).resolve()
    depth_timeline = spatial / "depth_frames.csv"
    if not depth_timeline.is_file():
        depth_timeline = spatial / "frames.csv"
    return {
        "run_id": config.run_id,
        "project_root": config.project_root,
        "workspace_dir": str(workspace),
        "video_path": str(Path(config.video_path).resolve()),
        "spatial_export_root": str(spatial),
        "camera_id": "right",
        "world_frame": "frame0_right_camera",
        "tracker_fps": config.tracker_fps,
        "true_depth_fps": 4.0,
        "inputs": {
            "raw_video": str(Path(config.video_path).resolve()),
            "true_depth_dir": str(spatial / "depth_meters_npy"),
            "true_depth_timeline": str(depth_timeline),
            "head_pose_csv": str(spatial / "pose/head_pose.csv"),
            "head_pose_jsonl": str(spatial / "pose/head_pose.jsonl"),
        },
        "models": {
            "qwen3_vl_code": config.qwen_code_root,
            "qwen3_vl_model": config.qwen_model_path,
            "sam2": config.sam2_root,
            "sam3d": config.sam3d_root,
            "video_depth_anything": config.vda_root,
            "video_depth_anything_checkpoint": config.vda_checkpoint,
        },
        "policies": make_initial_state(config)["coordinate_policy"],
    }


def initialize_mixed_workspace(config: MixedWorkspaceConfig, overwrite_state: bool = False) -> Path:
    workspace = config.workspace_path
    for path in mixed_workspace_dirs(workspace).values():
        path.mkdir(parents=True, exist_ok=True)
    state_path = workspace / "pipeline_state.json"
    if overwrite_state or not state_path.exists():
        state_path.write_text(
            json.dumps(make_initial_state(config), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    (workspace / "configs/mixed_recon_config.json").write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (workspace / "configs/rigid_recon_config.json").write_text(
        json.dumps(compatible_rigid_config(config), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return state_path
