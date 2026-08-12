#!/usr/bin/env python3
"""Estimate an independent coarse FoundationPose pose for each rigid frame."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import trimesh


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE = PROJECT_ROOT / "run_mixed_20260728_203734"
DEFAULT_FOUNDATIONPOSE_REPO = Path("/code/ArtHOI-4D-Reconstruction")


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = [item for item in loaded.geometry.values() if isinstance(item, trimesh.Trimesh)]
        if not geometries:
            raise ValueError(f"Mesh scene is empty: {path}")
        loaded = geometries[0].copy() if len(geometries) == 1 else trimesh.util.concatenate(geometries)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type from {path}: {type(loaded)!r}")
    return loaded


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def resolve_aligned_mesh(workspace: Path, object_id: str) -> Path:
    summary = read_json(workspace / "outputs/07_alignment/alignment_summary.json")
    matches = [item for item in summary.get("objects", []) if item.get("object_id") == object_id]
    if len(matches) != 1:
        raise KeyError(f"Expected one aligned object {object_id!r}, found {len(matches)}")
    return Path(matches[0]["aligned_mesh"]).resolve()


def resolve_mask_dir(workspace: Path, object_id: str) -> Path:
    candidates = [
        workspace / "outputs/04_object_masks" / object_id / "objects" / object_id,
        workspace / "outputs/02_sam2_frame0_masks/propagated/objects" / object_id,
    ]
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*.png")):
            return candidate.resolve()
    raise FileNotFoundError(f"No propagated masks found for {object_id}: {candidates}")


def resolve_end_frame(workspace: Path, requested: int | None) -> int:
    if requested is not None:
        return requested
    plan_path = workspace / "outputs/08_tracking/rigid_interaction_plan.json"
    if plan_path.is_file():
        return int(read_json(plan_path)["optimization_end_frame_inclusive"])
    return 0


def camera_matrix(camera: dict[str, Any]) -> np.ndarray:
    intrinsics = camera["rgb_intrinsics_right"]
    return np.asarray(
        [
            [intrinsics["fx"], 0.0, intrinsics["cx"]],
            [0.0, intrinsics["fy"], intrinsics["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def rotation_step_deg(previous: np.ndarray | None, current: np.ndarray) -> float | None:
    if previous is None:
        return None
    relative = current[:3, :3] @ previous[:3, :3].T
    return float(np.rad2deg(Rotation.from_matrix(relative).magnitude()))


def transform_point(point: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return np.asarray(point, dtype=np.float64) @ transform[:3, :3].T + transform[:3, 3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--object-id", default="bottle")
    parser.add_argument("--foundationpose-repo", type=Path, default=DEFAULT_FOUNDATIONPOSE_REPO)
    parser.add_argument("--rgb-dir", type=Path, default=None)
    parser.add_argument("--depth-dir", type=Path, default=None)
    parser.add_argument(
        "--poses-path",
        type=Path,
        default=None,
        help="Camera trajectory NPZ used to transform FoundationPose Ct poses into C0.",
    )
    parser.add_argument("--mask-dir", type=Path, default=None)
    parser.add_argument("--hand-mask-dir", type=Path, default=None)
    parser.add_argument(
        "--mask-hand-depth",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Invalidate hand-mask pixels in depth to match hand-removed RGB.",
    )
    parser.add_argument("--aligned-mesh", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--refine-iterations", type=int, default=2)
    parser.add_argument("--max-mesh-faces", type=int, default=30000)
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workspace = args.workspace.resolve()
    foundationpose_repo = args.foundationpose_repo.resolve()
    if str(foundationpose_repo) not in sys.path:
        sys.path.insert(0, str(foundationpose_repo))

    import nvdiffrast.torch as dr
    import torch
    from third_party.foundationpose.Utils import set_logging_format, set_seed
    from third_party.foundationpose.estimater import FoundationPose
    from third_party.foundationpose.learning.training.predict_pose_refine import PoseRefinePredictor
    from third_party.foundationpose.learning.training.predict_score import ScorePredictor

    set_logging_format()
    set_seed(0)

    default_rgb_dir = workspace / "outputs/03_diffueraser/inpainted_frames_png"
    rgb_policy = "diffueraser_hand_removed" if args.rgb_dir is None else "caller_supplied"
    rgb_dir = (args.rgb_dir or default_rgb_dir).resolve()
    depth_dir = (args.depth_dir or workspace / "outputs/06_dense_depth/metric_depth_npy").resolve()
    mask_dir = (args.mask_dir or resolve_mask_dir(workspace, args.object_id)).resolve()
    hand_mask_dir = (
        args.hand_mask_dir or workspace / "outputs/02_hand_masks/combined"
    ).resolve()
    aligned_mesh = (args.aligned_mesh or resolve_aligned_mesh(workspace, args.object_id)).resolve()
    poses_path = (
        args.poses_path.resolve()
        if args.poses_path is not None
        else workspace / "outputs/00_rgb_frames/poses.npz"
    )
    output_dir = (
        args.output_dir or workspace / "outputs/08_foundationpose_independent" / args.object_id
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_output_dir = output_dir / "frames"
    frame_output_dir.mkdir(exist_ok=True)

    rgb_paths = sorted(rgb_dir.glob("*.png"))
    depth_paths = sorted(depth_dir.glob("*.npy"))
    mask_paths = sorted(mask_dir.glob("*.png"))
    hand_mask_paths = sorted(hand_mask_dir.glob("*.png")) if args.mask_hand_depth else []
    if not rgb_paths:
        raise FileNotFoundError(f"No RGB frames in {rgb_dir}")
    if not (len(rgb_paths) == len(depth_paths) == len(mask_paths)):
        raise ValueError(
            f"RGB/depth/mask count mismatch: {len(rgb_paths)}/{len(depth_paths)}/{len(mask_paths)}"
        )
    if args.mask_hand_depth and len(hand_mask_paths) != len(rgb_paths):
        raise ValueError(
            f"Hand-mask/RGB count mismatch: {len(hand_mask_paths)}/{len(rgb_paths)} in {hand_mask_dir}"
        )

    end_frame = resolve_end_frame(workspace, args.end_frame)
    if not 0 <= args.start_frame <= end_frame < len(rgb_paths):
        raise ValueError(f"Invalid frame range {args.start_frame}:{end_frame} for {len(rgb_paths)} frames")

    camera = read_json(workspace / "outputs/00_rgb_frames/camera.json")
    K = camera_matrix(camera)
    with np.load(poses_path) as pose_data:
        if "T_C0_from_Ct" not in pose_data.files:
            raise KeyError(f"T_C0_from_Ct not found in {poses_path}: {pose_data.files}")
        transforms_c0_from_ct = pose_data["T_C0_from_Ct"].astype(np.float64)
    if len(transforms_c0_from_ct) != len(rgb_paths):
        raise ValueError("Camera pose count does not match RGB timeline")

    source_mesh = load_mesh(aligned_mesh)
    estimator_mesh = source_mesh.copy()
    original_face_count = len(estimator_mesh.faces)
    if args.max_mesh_faces > 0 and original_face_count > args.max_mesh_faces:
        estimator_mesh = estimator_mesh.simplify_quadric_decimation(args.max_mesh_faces)
    estimator_mesh.export(output_dir / "foundationpose_estimator_mesh.ply")

    logging.info(
        "Initializing FoundationPose with %d faces (source mesh: %d faces)",
        len(estimator_mesh.faces),
        original_face_count,
    )
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    estimator = FoundationPose(
        model_pts=estimator_mesh.vertices,
        model_normals=estimator_mesh.vertex_normals,
        mesh=estimator_mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=str(output_dir / "debug"),
        debug=args.debug,
        glctx=glctx,
    )

    frame_indices = list(range(args.start_frame, end_frame + 1))
    poses_ct = np.full((len(frame_indices), 4, 4), np.nan, dtype=np.float64)
    poses_c0 = np.full_like(poses_ct, np.nan)
    success = np.zeros(len(frame_indices), dtype=bool)
    diagnostics: list[dict[str, Any]] = []
    previous_pose_c0: np.ndarray | None = None
    previous_center_c0: np.ndarray | None = None

    for output_index, frame in enumerate(frame_indices):
        frame_dir = frame_output_dir / f"{frame:06d}"
        frame_dir.mkdir(exist_ok=True)
        pose_ct_path = frame_dir / "T_Ct_from_aligned_mesh.npy"
        pose_c0_path = frame_dir / "T_C0_from_aligned_mesh.npy"
        diagnostic_path = frame_dir / "diagnostic.json"
        started = time.perf_counter()
        mask = np.asarray(Image.open(mask_paths[frame]).convert("L")) > 127
        valid_mask_pixels = int(mask.sum())

        try:
            if (
                not args.force
                and pose_ct_path.is_file()
                and pose_c0_path.is_file()
                and diagnostic_path.is_file()
            ):
                pose_ct = np.load(pose_ct_path).astype(np.float64)
                pose_c0 = np.load(pose_c0_path).astype(np.float64)
                record = read_json(diagnostic_path)
                record["resumed"] = True
            else:
                if valid_mask_pixels < 4:
                    raise ValueError(f"Object mask has only {valid_mask_pixels} pixels")
                rgb = np.asarray(Image.open(rgb_paths[frame]).convert("RGB"))
                depth = np.load(depth_paths[frame]).astype(np.float32)
                if depth.shape != mask.shape or rgb.shape[:2] != mask.shape:
                    raise ValueError(
                        f"Frame {frame} shape mismatch: rgb={rgb.shape}, depth={depth.shape}, mask={mask.shape}"
                    )
                depth[~np.isfinite(depth) | (depth < 0.001) | (depth > 3.0)] = 0.0
                hand_overlap_pixels = 0
                if args.mask_hand_depth:
                    hand_mask = np.asarray(Image.open(hand_mask_paths[frame]).convert("L")) > 127
                    if hand_mask.shape != depth.shape:
                        raise ValueError(
                            f"Frame {frame} hand-mask shape mismatch: {hand_mask.shape}/{depth.shape}"
                        )
                    hand_overlap_pixels = int((mask & hand_mask).sum())
                    depth[hand_mask] = 0.0
                valid_object_depth_pixels = int((mask & (depth >= 0.001)).sum())
                if valid_object_depth_pixels < 4:
                    raise ValueError(f"Only {valid_object_depth_pixels} valid object depth pixels")

                # register() is intentionally called independently for every frame.
                estimator.pose_last = None
                pose_ct = estimator.register(
                    K=K,
                    rgb=rgb,
                    depth=depth,
                    ob_mask=mask,
                    iteration=args.refine_iterations,
                ).astype(np.float64)
                pose_c0 = transforms_c0_from_ct[frame] @ pose_ct
                if not np.isfinite(pose_c0).all():
                    raise ValueError("FoundationPose returned non-finite pose values")
                score = None
                if getattr(estimator, "scores", None) is not None and len(estimator.scores):
                    score = float(torch.as_tensor(estimator.scores[0]).detach().cpu())
                np.save(pose_ct_path, pose_ct)
                np.save(pose_c0_path, pose_c0)
                record = {
                    "frame": frame,
                    "status": "completed",
                    "resumed": False,
                    "mask_area_px": valid_mask_pixels,
                    "hand_overlap_area_px": hand_overlap_pixels,
                    "valid_object_depth_px": valid_object_depth_pixels,
                    "foundationpose_score": score,
                    "elapsed_s": time.perf_counter() - started,
                    "T_Ct_from_aligned_mesh": pose_ct.tolist(),
                    "T_C0_from_aligned_mesh": pose_c0.tolist(),
                }
                write_json(diagnostic_path, record)

            poses_ct[output_index] = pose_ct
            poses_c0[output_index] = pose_c0
            success[output_index] = True
            center_c0 = transform_point(source_mesh.centroid, pose_c0)
            record["rotation_step_deg"] = rotation_step_deg(previous_pose_c0, pose_c0)
            record["center_step_m"] = (
                None if previous_center_c0 is None else float(np.linalg.norm(center_c0 - previous_center_c0))
            )
            record["mesh_center_C0"] = center_c0.tolist()
            previous_pose_c0 = pose_c0
            previous_center_c0 = center_c0
            logging.info(
                "Frame %d complete: score=%s elapsed=%.2fs",
                frame,
                record.get("foundationpose_score"),
                record.get("elapsed_s", time.perf_counter() - started),
            )
        except Exception as exc:
            record = {
                "frame": frame,
                "status": "failed",
                "resumed": False,
                "mask_area_px": valid_mask_pixels,
                "elapsed_s": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
            }
            write_json(diagnostic_path, record)
            logging.exception("Frame %d failed", frame)
        diagnostics.append(record)
        torch.cuda.empty_cache()

    np.save(output_dir / "frame_indices.npy", np.asarray(frame_indices, dtype=np.int32))
    np.save(output_dir / "success.npy", success)
    np.save(output_dir / "T_Ct_from_aligned_mesh.npy", poses_ct)
    np.save(output_dir / "T_C0_from_aligned_mesh.npy", poses_c0)
    delta_c0 = np.full_like(poses_c0, np.nan)
    if success[0]:
        frame0_inverse = np.linalg.inv(poses_c0[0])
        delta_c0[success] = np.einsum("tij,jk->tik", poses_c0[success], frame0_inverse)
    np.save(output_dir / "Delta_C0_object_motion.npy", delta_c0)
    with (output_dir / "frame_diagnostics.jsonl").open("w", encoding="utf-8") as output_file:
        for record in diagnostics:
            output_file.write(json.dumps(record) + "\n")

    manifest = {
        "stage": "08_foundationpose_independent",
        "status": "completed" if success.all() else "completed_with_failures",
        "object_id": args.object_id,
        "pose_policy": "independent FoundationPose register() on every frame; no temporal/contact optimization",
        "coordinate_convention": {
            "foundationpose_output": "T_Ct_from_aligned_mesh, OpenCV RDF camera coordinates",
            "viewer_output": "T_C0_from_aligned_mesh = T_C0_from_Ct @ T_Ct_from_aligned_mesh",
            "mesh_motion": "Delta_C0(t) = T_C0_from_aligned_mesh(t) @ inv(T_C0_from_aligned_mesh(0))",
        },
        "frame_start": args.start_frame,
        "frame_end_inclusive": end_frame,
        "frame_count": len(frame_indices),
        "successful_frames": int(success.sum()),
        "failed_frames": int((~success).sum()),
        "refine_iterations": args.refine_iterations,
        "rgb_dir": str(rgb_dir),
        "rgb_policy": rgb_policy,
        "depth_dir": str(depth_dir),
        "poses_path": str(poses_path),
        "depth_hand_occlusion_policy": (
            "set SAM2 hand-mask pixels to invalid depth" if args.mask_hand_depth else "unchanged"
        ),
        "hand_mask_dir": str(hand_mask_dir) if args.mask_hand_depth else None,
        "mask_dir": str(mask_dir),
        "aligned_mesh": str(aligned_mesh),
        "estimator_mesh": str(output_dir / "foundationpose_estimator_mesh.ply"),
        "source_mesh_faces": original_face_count,
        "estimator_mesh_faces": len(estimator_mesh.faces),
        "foundationpose_repo": str(foundationpose_repo),
        "output_dir": str(output_dir),
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    return 0 if success.all() else 2


if __name__ == "__main__":
    raise SystemExit(main())
