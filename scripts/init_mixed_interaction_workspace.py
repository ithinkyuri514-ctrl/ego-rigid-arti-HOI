#!/usr/bin/env python3
"""Initialize a right-eye workspace for mixed articulated and rigid interactions."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vlm_sam2_recon.stages.mixed_interaction_recon import (  # noqa: E402
    MixedWorkspaceConfig,
    initialize_mixed_workspace,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workspace-dir", type=Path, required=True)
    parser.add_argument("--video-path", type=Path, required=True)
    parser.add_argument("--spatial-export-root", type=Path, required=True)
    parser.add_argument("--tracker-fps", type=float, default=15.0)
    parser.add_argument("--overwrite-state", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MixedWorkspaceConfig(
        run_id=args.run_id,
        workspace_dir=str(args.workspace_dir.resolve()),
        video_path=str(args.video_path.resolve()),
        spatial_export_root=str(args.spatial_export_root.resolve()),
        tracker_fps=args.tracker_fps,
    )
    state = initialize_mixed_workspace(config, overwrite_state=args.overwrite_state)
    for name in ("MIXED_PIPELINE_MEMORY.md", "MIXED_PIPELINE_CONTRACT.md"):
        template = PROJECT_ROOT / "docs" / name
        destination = config.workspace_path / "docs" / name
        shutil.copyfile(template, destination)
    print(f"Workspace: {config.workspace_path}")
    print(f"State: {state}")
    print(f"Config: {config.workspace_path / 'configs/mixed_recon_config.json'}")


if __name__ == "__main__":
    main()
