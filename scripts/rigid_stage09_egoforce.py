#!/usr/bin/env python3
"""Run EgoForce and transform its per-frame camera-space geometry into C0."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import trimesh


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vlm_sam2_recon.rigid_pipeline.egoforce import (  # noqa: E402
    joint_reprojection_error,
    make_c0_payload,
    project_points,
    projection_metrics,
)


SIDES = ("left", "right")
SIDE_TO_INDEX = {side: index for index, side in enumerate(SIDES)}
SIDE_COLORS = {"left": (255, 45, 45), "right": (50, 115, 255)}
FRAME_RE = re.compile(r"^(\d{6})_egoforce_meshes\.npz$")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def export_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    with path.open("w", encoding="ascii") as handle:
        for x, y, z in vertices:
            handle.write(f"v {x:.8f} {y:.8f} {z:.8f}\n")
        for a, b, c in faces + 1:
            handle.write(f"f {int(a)} {int(b)} {int(c)}\n")


def draw_projection_overlay(
    rgb_path: Path,
    output_path: Path,
    raw: dict[str, np.ndarray],
    intrinsics: dict[str, float],
    visible: np.ndarray,
) -> None:
    image = Image.open(rgb_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for side_index, side in enumerate(SIDES):
        if not visible[side_index]:
            continue
        uv, valid = project_points(raw["hand_joints"][side_index], intrinsics)
        color = SIDE_COLORS[side]
        for point in uv[valid]:
            x, y = float(point[0]), float(point[1])
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color, outline=(255, 255, 255), width=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def candidate_projection_record(
    vertices: np.ndarray,
    joints: np.ndarray,
    intrinsics: dict[str, float],
    image_width: int,
    image_height: int,
    hand_mask: np.ndarray,
) -> dict[str, object]:
    hand_metrics = projection_metrics(vertices, intrinsics, image_width, image_height)
    projected, positive = project_points(vertices, intrinsics)
    projected_int = np.rint(projected[positive]).astype(np.int64)
    inside = (
        (projected_int[:, 0] >= 0)
        & (projected_int[:, 0] < image_width)
        & (projected_int[:, 1] >= 0)
        & (projected_int[:, 1] < image_height)
    )
    projected_inside = projected_int[inside]
    mask_overlap = (
        float(hand_mask[projected_inside[:, 1], projected_inside[:, 0]].mean())
        if len(projected_inside)
        else 0.0
    )
    bbox_xyxy = None
    bbox_area_ratio = 1.0
    if len(projected_inside):
        x0, y0 = projected_inside.min(axis=0)
        x1, y1 = projected_inside.max(axis=0)
        bbox_xyxy = [int(x0), int(y0), int(x1), int(y1)]
        bbox_area_ratio = float(((x1 - x0 + 1) * (y1 - y0 + 1)) / max(image_width * image_height, 1))
    z = np.asarray(vertices, dtype=np.float64)[:, 2]
    joints_z = np.asarray(joints, dtype=np.float64)[:, 2]
    return {
        "hand_vertices": hand_metrics,
        "median_z_m": float(np.median(z)),
        "joint_median_z_m": float(np.median(joints_z)),
        "bbox_xyxy": bbox_xyxy,
        "bbox_area_ratio": bbox_area_ratio,
        "sam2_projected_vertex_overlap": mask_overlap,
    }


def single_right_score(record: dict[str, object], expected_hand: bool) -> tuple[float, list[str]]:
    metrics = record["hand_vertices"]
    assert isinstance(metrics, dict)
    median_z = float(record["median_z_m"])
    inside = float(metrics["inside_image_ratio"])
    positive = float(metrics["positive_depth_ratio"])
    bbox_area = float(record["bbox_area_ratio"])
    mask_overlap = float(record["sam2_projected_vertex_overlap"])
    reasons: list[str] = []

    score = 3.0 * positive + 3.0 * inside
    if expected_hand:
        score += 1.5 * mask_overlap
    if 0.12 <= median_z <= 1.1:
        score += 2.0
    elif 0.06 <= median_z < 0.12:
        score -= 1.5
        reasons.append("near_camera_depth")
    else:
        score -= 8.0
        reasons.append("invalid_depth")
    if bbox_area <= 0.12:
        score += 2.0
    elif bbox_area <= 0.25:
        score -= 1.0
        reasons.append("large_projection_bbox")
    else:
        score -= 4.0
        reasons.append("huge_projection_bbox")
    if positive < 0.75:
        score -= 6.0
        reasons.append("many_vertices_behind_camera")
    if inside < 0.45:
        score -= 3.0
        reasons.append("mostly_outside_image")
    return score, reasons


def remap_selected_candidate_to_right(payload: dict[str, np.ndarray], selected_index: int) -> None:
    """Expose the chosen EgoForce candidate as the only right hand for downstream code."""
    selected_side = SIDES[selected_index]
    right_index = SIDE_TO_INDEX["right"]
    payload["visible_hand"] = np.asarray([False, True], dtype=bool)
    for key in ("hand_vertices", "arm_vertices", "hand_joints", "arm_joints", "transl", "mano_transl"):
        if key in payload:
            payload[key] = np.asarray(payload[key]).copy()
            payload[key][right_index] = payload[key][selected_index]
    if selected_side == "left" and "left_hand_faces" in payload:
        payload["right_hand_faces"] = np.asarray(payload["left_hand_faces"])


def update_pipeline_state(
    workspace: Path,
    manifest_path: Path,
    poses_path: Path,
    detected_frames: int,
    frame_count: int,
) -> None:
    state_path = workspace / "pipeline_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    matches = [stage for stage in state["stages"] if stage["stage"] == "09_egoforce_hand_frame0"]
    if not matches:
        matches = [
            {
                "stage": "09_egoforce_hand_frame0",
                "status": "pending",
                "inputs": [],
                "outputs": [],
                "notes": "",
            }
        ]
        state["stages"].append(matches[0])
    for stage in matches:
        if stage["stage"] == "09_egoforce_hand_frame0":
            stage.update(
                {
                    "status": "completed",
                    "inputs": [
                        str(workspace / "outputs/00_rgb_frames/right_rgb_png"),
                        str(poses_path),
                        "/code/EgoForce/_DATA/model_weights.pth",
                    ],
                    "outputs": [str(workspace / "outputs/09_egoforce")],
                    "notes": (
                        f"EgoForce geometry detected in {detected_frames}/{frame_count} frames; all exported hand, "
                        "arm, joint, and translation geometry transformed from Ct into C0."
                    ),
                }
            )
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=PROJECT_ROOT / "run_rigid_20260715_215524",
    )
    parser.add_argument("--egoforce-root", type=Path, default=Path("/code/EgoForce"))
    parser.add_argument("--egoforce-python", type=Path, default=Path("/opt/conda/envs/egoforce/bin/python"))
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--hand-mask-dir", type=Path, default=None)
    parser.add_argument(
        "--poses-path",
        type=Path,
        default=None,
        help="Camera trajectory NPZ used to transform Ct hand geometry into C0.",
    )
    parser.add_argument(
        "--single-right-hand",
        action="store_true",
        help=(
            "Treat the scene as one visible right hand: choose the best raw left/right EgoForce "
            "candidate per frame and export it as right_hand_C0 only."
        ),
    )
    parser.add_argument(
        "--single-right-source",
        choices=("best", "left", "right"),
        default="best",
        help=(
            "Raw EgoForce slot used with --single-right-hand. 'best' keeps the mask-based "
            "candidate selection; an explicit side preserves anatomical handedness when both "
            "hands are visible."
        ),
    )
    parser.add_argument(
        "--skip-mask-qc",
        action="store_true",
        help="Accept raw EgoForce detections when no independent SAM2 hand masks exist.",
    )
    parser.add_argument("--reuse-raw", action="store_true", help="Skip EgoForce inference and reuse raw_Ct files.")
    parser.add_argument("--skip-projection-overlays", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = args.workspace.resolve()
    output_dir = workspace / "outputs/09_egoforce"
    raw_dir = output_dir / "raw_Ct"
    c0_dir = output_dir / "C0"
    overlay_dir = output_dir / "projection_overlays"
    rgb_dir = workspace / "outputs/00_rgb_frames/right_rgb_png"
    mask_dir = (args.hand_mask_dir or workspace / "outputs/02_hand_masks/combined").resolve()
    poses_path = (
        args.poses_path.resolve()
        if args.poses_path is not None
        else workspace / "outputs/00_rgb_frames/poses.npz"
    )
    rgb_paths = sorted(rgb_dir.glob("*.png"))
    mask_paths = sorted(mask_dir.glob("*.png"))
    camera = json.loads((workspace / "outputs/00_rgb_frames/camera.json").read_text(encoding="utf-8"))
    selected_eye = str(camera.get("selected_eye", "right")).lower()
    if selected_eye not in {"left", "right"}:
        raise ValueError(f"Unsupported selected eye: {selected_eye!r}")
    intrinsics = camera["rgb_intrinsics_right"]
    image_width, image_height = Image.open(rgb_paths[0]).size
    with np.load(poses_path) as pose_data:
        transforms = pose_data["T_C0_from_Ct"].astype(np.float64)
        timestamps = pose_data["rgb_timestamps_s"].astype(np.float64)
    if not rgb_paths:
        raise FileNotFoundError(rgb_dir)
    if args.skip_mask_qc:
        mask_paths = [None] * len(rgb_paths)
    if len(mask_paths) != len(rgb_paths) or transforms.shape != (len(rgb_paths), 4, 4):
        raise ValueError(
            f"Stage 09 RGB/mask/pose mismatch: {len(rgb_paths)}/{len(mask_paths)}/{transforms.shape}"
        )
    for path, label in (
        (args.egoforce_python, "EgoForce Python"),
        (args.egoforce_root / "scripts/infer_single_rgb_viser.py", "EgoForce inference script"),
        (args.egoforce_root / "_DATA/model_weights.pth", "EgoForce weights"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    camera_path = output_dir / "camera_pinhole.json"
    write_json(
        camera_path,
        {
            "model": "pinhole",
            **{key: float(intrinsics[key]) for key in ("fx", "fy", "cx", "cy")},
        },
    )
    if not args.reuse_raw:
        command = [
            str(args.egoforce_python),
            str(args.egoforce_root / "scripts/infer_single_rgb_viser.py"),
            "--image-dir",
            str(rgb_dir),
            "--image-glob",
            "*.png",
            "--fps",
            str(args.fps),
            "--output-dir",
            str(raw_dir),
            "--camera-json",
            str(camera_path),
            "--no-viser",
        ]
        print("Running:", " ".join(command), flush=True)
        subprocess.run(command, cwd=args.egoforce_root, check=True)

    raw_files: dict[int, Path] = {}
    for path in raw_dir.glob("*_egoforce_meshes.npz"):
        match = FRAME_RE.match(path.name)
        if match:
            raw_files[int(match.group(1))] = path
    if not raw_files:
        raise RuntimeError(f"EgoForce produced no per-frame outputs in {raw_dir}")
    for generated_dir in (c0_dir, overlay_dir):
        if generated_dir.exists():
            shutil.rmtree(generated_dir)

    frames: list[dict[str, object]] = []
    side_counts = {side: 0 for side in SIDES}
    raw_side_counts = {side: 0 for side in SIDES}
    rejected_detections: list[dict[str, object]] = []
    missed_expected: list[int] = []
    qc_records: list[dict[str, object]] = []
    for frame, (rgb_path, mask_path) in enumerate(zip(rgb_paths, mask_paths)):
        hand_mask = (
            np.zeros((image_height, image_width), dtype=bool)
            if mask_path is None
            else np.asarray(Image.open(mask_path).convert("L")) > 127
        )
        mask_area = int(hand_mask.sum())
        expected_hand = bool(args.skip_mask_qc or mask_area >= 100)
        raw_path = raw_files.get(frame)
        entry: dict[str, object] = {
            "frame": frame,
            "timestamp_s": float(timestamps[frame]),
            "rgb_path": str(rgb_path),
            "sam2_hand_mask_path": str(mask_path) if mask_path is not None else None,
            "sam2_hand_mask_area_px": mask_area,
            "sam2_expected_hand": expected_hand,
            "T_C0_from_Ct": transforms[frame].tolist(),
            "detected_sides": [],
            "status": "no_egoforce_detection",
        }
        if raw_path is None:
            if expected_hand:
                missed_expected.append(frame)
            frames.append(entry)
            continue

        with np.load(raw_path) as loaded:
            raw = {key: loaded[key] for key in loaded.files}
        raw_visible = np.asarray(raw["visible_hand"], dtype=bool).reshape(2)
        raw_detected_sides = [side for index, side in enumerate(SIDES) if raw_visible[index]]
        if not raw_detected_sides:
            if expected_hand:
                missed_expected.append(frame)
            frames.append(entry)
            continue
        valid_visible = np.zeros(2, dtype=bool)
        frame_qc: dict[str, object] = {"frame": frame, "sides": {}}
        selected_single_right: dict[str, object] | None = None
        for side_index, side in enumerate(SIDES):
            if not raw_visible[side_index]:
                continue
            raw_side_counts[side] += 1
            candidate_record = candidate_projection_record(
                raw["hand_vertices"][side_index],
                raw["hand_joints"][side_index],
                intrinsics,
                image_width,
                image_height,
                hand_mask,
            )
            target = raw.get("egoforce_hand_keypoints_2d")
            confidence = raw.get("egoforce_hand_keypoint_confidence")
            reprojection = {}
            if target is not None:
                reprojection = joint_reprojection_error(
                    raw["hand_joints"][side_index],
                    target[side_index],
                    None if confidence is None else confidence[side_index],
                    intrinsics,
                )
            score, score_reasons = single_right_score(candidate_record, expected_hand)
            accepted = bool(
                raw_visible[side_index]
                and (args.skip_mask_qc or (expected_hand and candidate_record["sam2_projected_vertex_overlap"] >= 0.5))
            )
            candidate_record.update(
                {
                    "hand_joints": reprojection,
                    "single_right_score": score,
                    "score_reasons": score_reasons,
                    "accepted": accepted,
                }
            )
            frame_qc["sides"][side] = candidate_record
            valid_visible[side_index] = accepted
            if selected_single_right is None or score > float(selected_single_right["score"]):
                selected_single_right = {
                    "side": side,
                    "side_index": side_index,
                    "score": score,
                    "record": candidate_record,
                }
            if accepted:
                side_counts[side] += 1
            else:
                rejected_detections.append(
                    {
                        "frame": frame,
                        "side": side,
                        "sam2_projected_vertex_overlap": candidate_record["sam2_projected_vertex_overlap"],
                    }
                )
        if args.single_right_hand and args.single_right_source != "best":
            preferred_index = SIDE_TO_INDEX[args.single_right_source]
            preferred_record = frame_qc["sides"].get(args.single_right_source)
            if preferred_record is not None:
                selected_single_right = {
                    "side": args.single_right_source,
                    "side_index": preferred_index,
                    "score": float(preferred_record["single_right_score"]),
                    "record": preferred_record,
                }
        if args.single_right_hand:
            original_selected = selected_single_right
            valid_visible[:] = False
            if original_selected is not None and float(original_selected["score"]) >= 6.0:
                valid_visible[SIDE_TO_INDEX["right"]] = True
                selected_side = str(original_selected["side"])
                selected_index = int(original_selected["side_index"])
            else:
                selected_side = None
                selected_index = -1
        else:
            selected_side = None
            selected_index = -1
        detected_sides = [side for index, side in enumerate(SIDES) if valid_visible[index]]
        entry.update(
            {
                "raw_Ct_npz": str(raw_path),
                "raw_detected_sides": raw_detected_sides,
                "detected_sides": detected_sides,
                "status": "completed" if detected_sides else "rejected_by_sam2_consistency",
            }
        )
        if args.single_right_hand:
            entry["single_right_hand_policy"] = (
                "best_raw_candidate_relabelled_to_right"
                if args.single_right_source == "best"
                else f"prefer_raw_{args.single_right_source}_with_best_visible_fallback"
            )
            entry["selected_raw_side"] = selected_side
            entry["selected_raw_side_index"] = selected_index
            entry["selected_raw_side_score"] = (
                None if selected_single_right is None else float(selected_single_right["score"])
            )
        if detected_sides:
            payload = make_c0_payload(raw, transforms[frame])
            payload["raw_visible_hand"] = payload["visible_hand"].copy()
            if args.single_right_hand:
                remap_selected_candidate_to_right(payload, selected_index)
            payload["visible_hand"] = valid_visible
            frame_dir = c0_dir / f"frame_{frame:06d}"
            frame_dir.mkdir(parents=True, exist_ok=True)
            npz_path = frame_dir / "egoforce_geometry_C0.npz"
            np.savez_compressed(npz_path, **payload)
            entry["geometry_C0_npz"] = str(npz_path)
            for side_index, side in enumerate(SIDES):
                if not valid_visible[side_index]:
                    continue
                hand_path = frame_dir / f"{side}_hand_C0.obj"
                arm_path = frame_dir / f"{side}_arm_C0.obj"
                export_obj(
                    hand_path,
                    payload["hand_vertices"][side_index],
                    payload[f"{side}_hand_faces"],
                )
                export_obj(arm_path, payload["arm_vertices"][side_index], payload["arm_faces"])
                entry[f"{side}_hand_C0"] = str(hand_path)
                entry[f"{side}_arm_C0"] = str(arm_path)
        elif expected_hand:
            missed_expected.append(frame)
        if not args.skip_projection_overlays:
            draw_projection_overlay(rgb_path, overlay_dir / f"{frame:06d}.jpg", raw, intrinsics, raw_visible)
        qc_records.append(frame_qc)
        frames.append(entry)

    if args.single_right_hand:
        side_counts = {
            "left": 0,
            "right": sum(
                entry["status"] == "completed" and "right" in entry.get("detected_sides", [])
                for entry in frames
            ),
        }

    manifest = {
        "schema_version": 1,
        "type": "egoforce_pose_compensated_sequence",
        "candidate_policy": (
            (
                "single_visible_right_hand_best_raw_left_or_right_candidate"
                if args.single_right_source == "best"
                else f"single_visible_right_hand_prefer_raw_{args.single_right_source}_with_best_visible_fallback"
            )
            if args.single_right_hand
            else "independent_left_right_sam2_consistency"
        ),
        "frame_count": len(rgb_paths),
        "detected_frame_count": sum(entry["status"] == "completed" for entry in frames),
        "side_detected_frame_counts": side_counts,
        "raw_egoforce_detected_frame_count": len(raw_files),
        "raw_side_detected_frame_counts": raw_side_counts,
        "selected_eye": selected_eye,
        "coordinate_frame": f"frame0_{selected_eye}_camera_opencv_rdf",
        "raw_coordinate_frame": f"current_{selected_eye}_camera_opencv_rdf",
        "transform_rule": "p_C0 = T_C0_from_Ct[frame] @ p_Ct",
        "raw_Ct_policy": "diagnostic_only",
        "independent_hand_mask_qc": not args.skip_mask_qc,
        "camera_json": str(camera_path),
        "pose_source": str(poses_path),
        "missed_sam2_expected_frames": missed_expected,
        "rejected_raw_detections": rejected_detections,
        "frames": frames,
    }
    manifest_path = output_dir / "dynamic_manifest.json"
    write_json(manifest_path, manifest)
    write_json(
        output_dir / "qc_summary.json",
        {
            "detected_frame_count": manifest["detected_frame_count"],
            "side_detected_frame_counts": side_counts,
            "raw_side_detected_frame_counts": raw_side_counts,
            "sam2_expected_frame_count": sum(bool(entry["sam2_expected_hand"]) for entry in frames),
            "missed_sam2_expected_frames": missed_expected,
            "rejected_raw_detections": rejected_detections,
            "projection_records": qc_records,
        },
    )
    update_pipeline_state(
        workspace,
        manifest_path,
        poses_path,
        int(manifest["detected_frame_count"]),
        len(rgb_paths),
    )
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in (
                    "frame_count",
                    "raw_egoforce_detected_frame_count",
                    "detected_frame_count",
                    "side_detected_frame_counts",
                    "missed_sam2_expected_frames",
                    "rejected_raw_detections",
                )
            },
            indent=2,
        )
    )
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
