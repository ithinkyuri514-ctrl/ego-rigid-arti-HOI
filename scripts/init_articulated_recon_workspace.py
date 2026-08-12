#!/usr/bin/env python3
"""Create an articulated reconstruction workspace without touching old runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vlm_sam2_recon.stages.articulated_object_recon import (  # noqa: E402
    ArticulatedWorkspaceConfig,
    initialize_articulated_workspace,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="articulated_run")
    parser.add_argument(
        "--workspace-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--video-path",
        type=Path,
        required=True,
        help="Input video for this run.",
    )
    parser.add_argument(
        "--spatial-export-root",
        type=Path,
        required=True,
        help="Matching SpatialMP4 export directory.",
    )
    parser.add_argument("--overwrite-state", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--eye", choices=["left", "right"], default="left")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace_dir = args.workspace_dir or (PROJECT_ROOT / f"run_{args.run_id}")
    config = ArticulatedWorkspaceConfig(
        run_id=args.run_id,
        workspace_dir=str(workspace_dir.resolve()),
        video_path=str(args.video_path.resolve()),
        spatial_export_root=str(args.spatial_export_root.resolve()),
        camera_id=args.eye,
        world_frame=f"frame0_{args.eye}_camera",
    )
    state_path = initialize_articulated_workspace(config, overwrite_state=args.overwrite_state)
    print(f"Workspace: {config.workspace_path}")
    print(f"State: {state_path}")
    print(f"Config: {config.workspace_path / 'configs/articulated_recon_config.json'}")


if __name__ == "__main__":
    main()
