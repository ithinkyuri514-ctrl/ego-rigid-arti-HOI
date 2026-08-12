#!/usr/bin/env python3
"""Run rigid tracking for the first mixed-pipeline rigid interaction only."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--object-id", default=None)
    parser.add_argument("--rgb-dir", type=Path, default=None)
    parser.add_argument(
        "--poses-path",
        type=Path,
        default=None,
        help="Camera trajectory NPZ. Defaults to outputs/00_rgb_frames/poses.npz.",
    )
    parser.add_argument("--exclude-depth-mask-dir", type=Path, default=None)
    parser.add_argument("--disable-pnp", action="store_true")
    parser.add_argument("--icp-samples", type=int, default=None)
    parser.add_argument("--anchor-frames", default=None)
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="Inclusive timeline frame to track. Defaults to the VLM rigid-event end frame.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Separate output directory for a bounded rerun; defaults to outputs/08_tracking.",
    )
    parser.add_argument(
        "--update-stage-state",
        action="store_true",
        help="Update the canonical Stage 08 record. Bounded reruns default to an isolated record.",
    )
    parser.add_argument("--reuse-tracks", action="store_true")
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def rigid_plan(workspace: Path, requested_object_id: str | None) -> tuple[str, int, list[dict]]:
    payload = json.loads((workspace / "outputs/01_vlm/mixed_interactions.json").read_text(encoding="utf-8"))
    events = payload["vlm_result"]["events"]
    rigid_events = [event for event in events if event["interaction_class"] == "rigid"]
    if requested_object_id is not None:
        rigid_events = [event for event in rigid_events if event["object_id"] == requested_object_id]
    if not rigid_events:
        raise ValueError("VLM result contains no matching rigid interaction")
    object_id = rigid_events[0]["object_id"]
    object_events = [event for event in rigid_events if event["object_id"] == object_id]
    return object_id, max(int(event["end_frame"]) for event in object_events), object_events


def ensure_state_stage(workspace: Path) -> None:
    state_path = workspace / "pipeline_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if not any(stage.get("stage") == "08_rigid_pose_tracking_frame0" for stage in state["stages"]):
        state["stages"].append(
            {
                "stage": "08_rigid_pose_tracking_frame0",
                "status": "pending",
                "inputs": [],
                "outputs": [],
                "notes": "Rigid interaction is bounded by the VLM event end frame.",
            }
        )
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    workspace = args.workspace.resolve()
    ensure_state_stage(workspace)
    object_id, planned_end_frame, events = rigid_plan(workspace, args.object_id)
    end_frame = planned_end_frame if args.end_frame is None else int(args.end_frame)
    rgb_dir = (
        args.rgb_dir
        or workspace / "outputs/03_diffueraser/inpainted_frames_png"
    ).absolute()
    full_frame_count = len(list(rgb_dir.glob("*.png")))
    if not 0 <= end_frame < full_frame_count:
        raise ValueError(f"--end-frame must be in [0, {full_frame_count - 1}], got {end_frame}")
    alignment_dir = workspace / "outputs/07_alignment" / object_id / "frame_000000"
    report_path = alignment_dir / "alignment_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    transform = np.asarray(report["transform"]["T_C0_from_sam3d_canonical"], dtype=np.float64)
    output_dir = (args.output_dir or workspace / "outputs/08_tracking").resolve()
    poses_path = (
        args.poses_path.resolve()
        if args.poses_path is not None
        else workspace / "outputs/00_rgb_frames/poses.npz"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    transform_path = output_dir / "T_C0_from_O_initial.npy"
    transform_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(transform_path, transform)

    if args.anchor_frames is None:
        anchors = sorted({0, end_frame, end_frame // 3, 2 * end_frame // 3})
        anchor_frames = ",".join(str(frame) for frame in anchors)
    else:
        anchor_frames = args.anchor_frames
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/rigid_stage08_track_pose.py"),
        "--workspace",
        str(workspace),
        "--output-dir",
        str(output_dir),
        "--rgb-dir",
        str(rgb_dir),
        "--poses-path",
        str(poses_path),
        "--mask-dir",
        str(
            workspace / "outputs/04_object_masks" / object_id / "objects" / object_id
            if (workspace / "outputs/04_object_masks" / object_id / "objects" / object_id).is_dir()
            else workspace / "outputs/02_sam2_frame0_masks/propagated/objects" / object_id
        ),
        "--transform0",
        str(transform_path),
        "--aligned-mesh",
        str(alignment_dir / "sam3d_aligned_C0.glb"),
        "--end-frame",
        str(end_frame),
        "--anchor-frames",
        anchor_frames,
    ]
    exclude_depth_mask_dir = (
        args.exclude_depth_mask_dir
        or workspace / "outputs/02_hand_masks/objects/hand"
    ).resolve()
    if exclude_depth_mask_dir.is_dir():
        command.extend(["--exclude-depth-mask-dir", str(exclude_depth_mask_dir)])
    if args.disable_pnp:
        command.append("--no-enable-pnp")
    if args.icp_samples is not None:
        command.extend(["--icp-samples", str(args.icp_samples)])
    if not args.update_stage_state:
        command.append("--skip-stage-state-update")
    if args.check:
        command.append("--check")
    if args.reuse_tracks:
        command.append("--reuse-tracks")
    plan = {
        "object_id": object_id,
        "rigid_events": events,
        "vlm_planned_end_frame_inclusive": planned_end_frame,
        "optimization_end_frame_inclusive": end_frame,
        "bounded_frame_count": end_frame + 1,
        "full_timeline_frame_count": full_frame_count,
        "tracking_rgb_dir": str(rgb_dir),
        "tracking_rgb_policy": "diffueraser_hand_removed",
        "camera_poses": str(poses_path),
        "exclude_depth_mask_dir": (
            str(exclude_depth_mask_dir) if exclude_depth_mask_dir.is_dir() else None
        ),
        "depth_occluder_policy": (
            "invalidate propagated hand-mask pixels before object RGB-D estimation"
        ),
        "pnp_enabled": not args.disable_pnp,
        "icp_samples": args.icp_samples,
        "reuse_policy": {
            "sam3d_mesh": str(alignment_dir / "sam3d_aligned_C0.glb"),
            "frame0_alignment": str(report_path),
            "object_masks": "existing propagated Stage 04 masks",
            "metric_depth": str(workspace / "outputs/06_dense_depth/metric_depth_npy"),
            "rerun_mesh_reconstruction": False,
            "rerun_frame0_alignment": False,
        },
        "anchor_frames": [int(value) for value in anchor_frames.split(",")],
        "command": command,
    }
    (output_dir / "rigid_interaction_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
