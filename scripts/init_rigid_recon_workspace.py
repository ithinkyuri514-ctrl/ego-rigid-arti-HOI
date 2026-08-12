#!/usr/bin/env python3
"""Initialize a workspace for the rigid-object reconstruction pipeline."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vlm_sam2_recon.stages.rigid_object_recon import (  # noqa: E402
    RigidWorkspaceConfig,
    initialize_rigid_workspace,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a rigid reconstruction run workspace.")
    parser.add_argument("--run-id", default="rigid_20260715_151803")
    parser.add_argument("--workspace-dir", type=Path, default=PROJECT_ROOT / "run_rigid_20260715_151803")
    parser.add_argument("--video-path", type=Path, default=Path("/code/3DVideo_2026-07-15-15-18-03-107.mp4"))
    parser.add_argument(
        "--spatial-export-root",
        type=Path,
        default=Path("/code/3DVideo_2026-07-15-15-18-03-107_spatialmp4_depth_pose_export"),
    )
    parser.add_argument("--overwrite-state", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RigidWorkspaceConfig(
        run_id=args.run_id,
        workspace_dir=str(args.workspace_dir.resolve()),
        video_path=str(args.video_path.resolve()),
        spatial_export_root=str(args.spatial_export_root.resolve()),
    )
    state_path = initialize_rigid_workspace(config, overwrite_state=args.overwrite_state)
    config_dir = config.workspace_path / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "rigid_recon_config.json"
    if not config_path.exists() or args.overwrite_state:
        template = PROJECT_ROOT / "configs" / "rigid_20260715_151803.json"
        if template.exists():
            shutil.copyfile(template, config_path)
    print(f"Workspace: {config.workspace_path}")
    print(f"State: {state_path}")
    print(f"Config: {config_path}")


if __name__ == "__main__":
    main()
