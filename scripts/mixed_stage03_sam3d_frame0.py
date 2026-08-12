#!/usr/bin/env python3
"""Run SAM 3D Objects for every global frame-0 interactive object-mask target."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vlm_sam2_recon.rigid_pipeline.common import update_stage_state, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path("/opt/conda/envs/sam3d-objects/bin/python"))
    parser.add_argument("--turntable-frames", type=int, default=48)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def load_object_prompts(workspace: Path) -> tuple[Path, list[dict]]:
    stage04_root = workspace / "outputs/04_object_masks"
    prompt_manifests = sorted(stage04_root.glob("*/mesh_prompt_frame0/prompt_manifest.json"))
    if prompt_manifests:
        records = []
        for manifest_path in prompt_manifests:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            object_id = str(manifest.get("object_id") or manifest_path.parents[1].name)
            records.append(
                {
                    "object_id": object_id,
                    "object_class": manifest.get("object_class", "unknown"),
                    "rgb": manifest["rgb"],
                    "mask": manifest["mask"],
                    "source_manifest": str(manifest_path),
                }
            )
        return stage04_root, records
    mask_summary_path = workspace / "outputs/02_sam2_frame0_masks/sam2_frame0_summary.json"
    mask_summary = json.loads(mask_summary_path.read_text(encoding="utf-8"))
    return mask_summary_path, mask_summary["objects"]


def update_sam3d_stage(workspace: Path, status: str, *, inputs: list[str], outputs: list[str], notes: str) -> None:
    state_path = workspace / "pipeline_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    names = {item.get("stage") for item in state.get("stages", [])}
    stage_name = "05_sam3d_frame0_reconstruction" if "05_sam3d_frame0_reconstruction" in names else "03_sam3d_frame0_reconstruction"
    update_stage_state(state_path, stage_name, status, inputs=inputs, outputs=outputs, notes=notes)


def main() -> int:
    args = parse_args()
    workspace = args.workspace.resolve()
    prompt_source, object_prompts = load_object_prompts(workspace)
    output_root = workspace / "outputs/03_sam3d_frame0"
    results = []
    for target in object_prompts:
        object_id = target["object_id"]
        output_dir = output_root / object_id
        pose_path = output_dir / "pose.json"
        if args.overwrite or not pose_path.is_file():
            command = [
                str(args.python.resolve()),
                str(PROJECT_ROOT / "scripts/run_sam3d_objects_prompt.py"),
                "--image",
                target["rgb"],
                "--mask",
                target["mask"],
                "--output-dir",
                str(output_dir),
                "--turntable-frames",
                str(args.turntable_frames),
            ]
            print("$ " + " ".join(command), flush=True)
            subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)
        pose = json.loads(pose_path.read_text(encoding="utf-8"))
        results.append(
            {
                "object_id": object_id,
                "object_class": target["object_class"],
                "rgb": target["rgb"],
                "mask": target["mask"],
                "output_dir": str(output_dir),
                "mesh_canonical": str(output_dir / "mesh_canonical.glb"),
                "mesh_posed_sam3d_camera": str(output_dir / "mesh_posed_camera.glb"),
                "pose": str(pose_path),
                "sam3d_projection_qc": pose.get("projection_qc"),
                "sam3d_scale_xyz": pose.get("scale_xyz"),
                "sam3d_translation": pose.get("translation"),
            }
        )
    summary = {
        "stage": "05_sam3d_frame0_reconstruction",
        "status": "completed",
        "modeling_frame_index": 0,
        "source_camera": "right",
        "objects": results,
    }
    summary_path = output_root / "sam3d_frame0_summary.json"
    write_json(summary_path, summary)
    update_sam3d_stage(
        workspace,
        "completed",
        inputs=[str(prompt_source)],
        outputs=[str(output_root)],
        notes=f"Reconstructed {len(results)} meshes from the same global right-eye frame 0; SAM3D pose/scale remain initialization only.",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
