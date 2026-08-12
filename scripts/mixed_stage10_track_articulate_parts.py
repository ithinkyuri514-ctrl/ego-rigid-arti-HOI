#!/usr/bin/env python3
"""Track articulated parts with CoTracker RGB-D, C0 pose compensation and a fixed revolute axis."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE = PROJECT_ROOT / "run_mixed_20260728_203734"
WORLD_FRAME = "frame0_right_camera_opencv_rdf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument(
        "--parts-json",
        type=Path,
        default=None,
        help="Multi-part manifest. Each entry needs part_id, mask_dir, mesh_c0 and a C0 revolute joint.",
    )
    parser.add_argument("--object-id", default="microwave")
    parser.add_argument(
        "--part-id",
        default=None,
        help="Single-part id, or an optional child-link filter when --parts-json is used.",
    )
    parser.add_argument("--mask-dir", type=Path, default=None)
    parser.add_argument("--part-mesh", type=Path, default=None, help="Reference part mesh already aligned in C0.")
    parser.add_argument("--joint-json", type=Path, default=None, help="C0 joint JSON, directly or under a joints list.")
    parser.add_argument("--joint-name", default=None)
    parser.add_argument("--axis-origin", default=None, help="C0 x,y,z in meters.")
    parser.add_argument("--axis-direction", default=None, help="C0 unit axis x,y,z; normalization is automatic.")
    parser.add_argument("--reference-transform", type=Path, default=None)
    parser.add_argument("--rgb-dir", type=Path, default=None)
    parser.add_argument("--depth-dir", type=Path, default=None)
    parser.add_argument("--poses-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--anchor-frames", default=None, help="Comma-separated global frame indices.")
    parser.add_argument(
        "--interaction-actions",
        default=None,
        help="Comma-separated VLM articulated actions to process, for example 'close'.",
    )
    parser.add_argument(
        "--interaction-end-frame",
        type=int,
        default=None,
        help="Override the end frame of the single selected VLM articulated interval.",
    )
    parser.add_argument(
        "--terminal-joint-limit",
        action="store_true",
        help="Interpolate an overridden interaction tail from the VLM end angle to the matching joint limit.",
    )
    parser.add_argument(
        "--terminal-joint-limit-frame",
        type=int,
        default=None,
        help=(
            "Nominal frame at which terminal-joint-limit interpolation reaches the limit. "
            "When later than --interaction-end-frame, the motion is truncated at the "
            "interaction end while preserving the nominal angular rate."
        ),
    )
    parser.add_argument(
        "--enforce-joint-limits",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Accepted for compatibility; Particulate joint limits are ignored.",
    )
    parser.add_argument("--queries-per-anchor", type=int, default=24)
    parser.add_argument("--query-points-json", type=Path, default=None)
    parser.add_argument(
        "--track-index",
        type=int,
        default=None,
        help="Force one CoTracker query index for every articulated interval.",
    )
    parser.add_argument("--query-mask-margin-px", type=float, default=5.0)
    parser.add_argument("--tracker-confidence", type=float, default=0.75)
    parser.add_argument("--stable-track-count", type=int, default=8)
    parser.add_argument("--min-stable-valid-ratio", type=float, default=0.85)
    parser.add_argument("--min-stable-median-confidence", type=float, default=0.85)
    parser.add_argument("--max-anchor-mesh-distance-m", type=float, default=0.08)
    parser.add_argument("--min-stable-track-separation-px", type=float, default=30.0)
    parser.add_argument("--reuse-tracks", action="store_true")
    parser.add_argument(
        "--enable-raw-icp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable unconstrained ICP before projecting motion onto the fixed joint axis.",
    )
    parser.add_argument("--min-axis-radius-m", type=float, default=0.015)
    parser.add_argument("--max-axis-coordinate-error-m", type=float, default=0.06)
    parser.add_argument("--max-radius-error-m", type=float, default=0.06)
    parser.add_argument("--max-angle-residual-deg", type=float, default=15.0)
    parser.add_argument(
        "--min-angle-points",
        type=int,
        default=1,
        help=(
            "Minimum visible high-confidence tracks for the one-DOF hinge fit. "
            "One 2D point is sufficient because the axis and origin are fixed."
        ),
    )
    parser.add_argument("--angle-sign", type=float, default=1.0)
    parser.add_argument(
        "--enable-contact-angle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop articulated closing at the first lid/base contact angle found by mesh clearance bisection.",
    )
    parser.add_argument("--collision-base-mesh", type=Path, default=None)
    parser.add_argument("--collision-clearance-m", type=float, default=0.001)
    parser.add_argument("--hinge-exclusion-m", type=float, default=0.04)
    parser.add_argument("--collision-step-deg", type=float, default=0.25)
    parser.add_argument(
        "--export-meshes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also write one transformed GLB per frame. Pose arrays are always written.",
    )
    parser.add_argument("--skip-stage-state-update", action="store_true")
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        return value

    path.write_text(json.dumps(convert(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_vec3(value: str | list[float] | tuple[float, ...] | np.ndarray) -> np.ndarray:
    if isinstance(value, str):
        values = [float(item.strip()) for item in value.split(",") if item.strip()]
    else:
        values = [float(item) for item in value]
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"Expected a finite x,y,z vector, got {value!r}")
    return array


def resolve_path(workspace: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else workspace / path).resolve()


def discover_images(directory: Path) -> list[Path]:
    for suffix in ("*.png", "*.jpg", "*.jpeg"):
        paths = sorted(directory.glob(suffix))
        if paths:
            return paths
    return []


def select_joint(payload: dict[str, Any], part: dict[str, Any], requested_name: str | None = None) -> dict[str, Any]:
    if "origin_xyz" in payload or "origin_C0" in payload:
        return payload
    joints = payload.get("joints") or payload.get("articulation_joints") or []
    if not isinstance(joints, list) or not joints:
        raise ValueError("Joint JSON has neither a direct joint nor a non-empty joints list")
    joint_name = requested_name or part.get("joint_name")
    if joint_name is not None:
        matches = [item for item in joints if item.get("name") == joint_name or item.get("joint_id") == joint_name]
        if len(matches) != 1:
            raise ValueError(f"Expected one joint named {joint_name!r}, found {len(matches)}")
        return matches[0]
    part_id = str(part["part_id"])
    matches = [item for item in joints if str(item.get("child", item.get("child_part_id", ""))) == part_id]
    if len(matches) == 1:
        return matches[0]
    if len(joints) == 1:
        return joints[0]
    raise ValueError(f"Could not select a unique joint for part {part_id!r}; specify joint_name")


def normalize_joint(joint: dict[str, Any]) -> dict[str, Any]:
    if str(joint.get("type", "revolute")).lower() not in {"revolute", "continuous"}:
        raise ValueError(f"Only revolute joints are supported, got {joint.get('type')!r}")
    origin_value = joint.get("origin_xyz", joint.get("origin_C0"))
    axis_value = joint.get("axis_xyz", joint.get("axis_C0"))
    if origin_value is None or axis_value is None:
        raise ValueError("Joint must provide origin_xyz/origin_C0 and axis_xyz/axis_C0 in C0")
    origin = parse_vec3(origin_value)
    axis = parse_vec3(axis_value)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        raise ValueError("Joint axis has zero length")
    result = dict(joint)
    result["type"] = "revolute"
    result["origin_xyz"] = origin.tolist()
    result["axis_xyz"] = (axis / norm).tolist()
    result["coordinate_frame"] = WORLD_FRAME
    return result


def load_parts(args: argparse.Namespace, workspace: Path) -> tuple[str, list[dict[str, Any]], Path | None]:
    source_manifest = args.parts_json.resolve() if args.parts_json else None
    if source_manifest is not None:
        payload = read_json(source_manifest)
        selected_eye = payload.get("selected_eye")
        if selected_eye is not None and str(selected_eye).lower() not in {"left", "right"}:
            raise ValueError(f"Unsupported Particulate manifest selected_eye={selected_eye!r}")
        object_id = str(payload.get("object_id", args.object_id))
        values = payload.get("parts")
        if isinstance(values, dict):
            parts = [dict(value, part_id=value.get("part_id", key)) for key, value in values.items()]
        elif isinstance(values, list):
            parts = []
            joints = payload.get("joints") or []
            for value in values:
                item = dict(value)
                item["part_id"] = item.get("part_id", item.get("link_name"))
                if item["part_id"] is None:
                    raise ValueError(f"Particulate part has neither part_id nor link_name: {item}")
                if item.get("mesh_c0") is None and item.get("C0_mesh") is not None:
                    item["mesh_c0"] = item["C0_mesh"]
                if "joint" not in item and "joint_json" not in item:
                    matches = [joint for joint in joints if str(joint.get("child", "")) == str(item["part_id"])]
                    if len(matches) == 1:
                        item["joint"] = matches[0]
                    elif len(matches) == 0:
                        # Root/base links have no motion joint and are not CoTracker targets.
                        continue
                    else:
                        raise ValueError(f"Multiple joints target child {item['part_id']!r}")
                parts.append(item)
        else:
            raise ValueError(f"{source_manifest} must contain a parts list or mapping")
        if args.part_id is not None:
            parts = [part for part in parts if str(part.get("part_id")) == args.part_id]
            if not parts:
                raise ValueError(f"No moving part {args.part_id!r} in {source_manifest}")
    else:
        object_id = args.object_id
        if args.part_id is None or args.part_mesh is None:
            raise ValueError("Single-part mode requires --part-id and --part-mesh")
        part: dict[str, Any] = {
            "part_id": args.part_id,
            "mesh_c0": str(args.part_mesh),
        }
        if args.mask_dir is not None:
            part["mask_dir"] = str(args.mask_dir)
        if args.reference_transform is not None:
            part["reference_transform"] = str(args.reference_transform)
        if args.joint_json is not None:
            part["joint_json"] = str(args.joint_json)
        elif args.axis_origin is not None and args.axis_direction is not None:
            part["joint"] = {
                "type": "revolute",
                "origin_xyz": parse_vec3(args.axis_origin).tolist(),
                "axis_xyz": parse_vec3(args.axis_direction).tolist(),
            }
        else:
            raise ValueError("Single-part mode requires --joint-json or both --axis-origin/--axis-direction")
        if args.joint_name:
            part["joint_name"] = args.joint_name
        parts = [part]

    normalized: list[dict[str, Any]] = []
    for source in parts:
        if "part_id" not in source:
            raise ValueError(f"Part entry has no part_id: {source}")
        part = dict(source)
        part_id = str(part["part_id"])
        mask_default = workspace / "outputs/04_object_masks" / object_id / "parts" / part_id
        if "mask_dir" in part:
            mask_dir = resolve_path(workspace, part["mask_dir"])
        else:
            object_masks = mask_default / "objects" / part_id
            mask_dir = object_masks if object_masks.is_dir() else mask_default / "combined"
        mesh_value = part.get("mesh_c0", part.get("part_mesh_c0", part.get("mesh")))
        if mesh_value is None:
            raise ValueError(f"Part {part_id!r} has no mesh_c0")
        mesh_c0 = resolve_path(workspace, mesh_value)
        if "joint" in part:
            joint_payload = part["joint"]
        elif "joint_json" in part:
            joint_payload = read_json(resolve_path(workspace, part["joint_json"]))
        else:
            raise ValueError(f"Part {part_id!r} has no joint or joint_json")
        joint = normalize_joint(select_joint(joint_payload, part, args.joint_name))
        item = {
            **part,
            "part_id": part_id,
            "mask_dir": mask_dir,
            "mesh_c0": mesh_c0,
            "joint": joint,
        }
        if part.get("reference_transform") is not None:
            item["reference_transform"] = resolve_path(workspace, part["reference_transform"])
        normalized.append(item)
    if not normalized:
        raise ValueError("No articulated parts were configured")
    return object_id, normalized, source_manifest


def vlm_anchor_frames(workspace: Path, object_id: str, frame_count: int) -> list[int]:
    path = workspace / "outputs/01_vlm/mixed_interactions.json"
    if not path.is_file():
        return [0]
    result = read_json(path).get("vlm_result", {})
    events = [
        event
        for event in result.get("events", [])
        if event.get("object_id") == object_id and event.get("interaction_class") == "articulated"
    ]
    anchors = {0}
    for event in events:
        start = int(event["start_frame"])
        end = int(event["end_frame"])
        anchors.update((start, (start + end) // 2, end))
    return sorted(frame for frame in anchors if 0 <= frame < frame_count)


def vlm_interaction_intervals(
    workspace: Path, object_id: str, frame_count: int
) -> list[dict[str, Any]]:
    path = workspace / "outputs/01_vlm/mixed_interactions.json"
    if not path.is_file():
        return [{"start_frame": 0, "end_frame": frame_count - 1, "action": "unknown"}]
    events = read_json(path).get("vlm_result", {}).get("events", [])
    intervals = []
    for event in events:
        if event.get("object_id") != object_id or event.get("interaction_class") != "articulated":
            continue
        start = max(0, int(event["start_frame"]))
        end = min(frame_count - 1, int(event["end_frame"]))
        if start <= end:
            intervals.append(
                {
                    "start_frame": start,
                    "end_frame": end,
                    "action": event.get("action"),
                    "event_id": event.get("event_id"),
                }
            )
    return sorted(intervals, key=lambda item: (item["start_frame"], item["end_frame"]))


def parse_anchor_frames(value: Any, fallback: list[int], frame_count: int) -> list[int]:
    if value is None:
        values = fallback
    elif isinstance(value, str):
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    else:
        values = [int(item) for item in value]
    values = sorted(set(values))
    if any(frame < 0 or frame >= frame_count for frame in values):
        raise ValueError(f"Anchor frames outside [0, {frame_count - 1}]: {values}")
    return values


def wrap_near(values: np.ndarray | float, reference: float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return reference + (array - reference + np.pi) % (2.0 * np.pi) - np.pi


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = np.maximum(weights[order], 0.0)
    total = float(sorted_weights.sum())
    if total <= 1e-12:
        return float(np.median(values))
    index = int(np.searchsorted(np.cumsum(sorted_weights), 0.5 * total, side="left"))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def axis_rotation_transform(origin: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    rotation = Rotation.from_rotvec(axis * float(angle_rad)).as_matrix()
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = origin - rotation @ origin
    return transform


def raw_pose_axis_angle(
    delta_pose: np.ndarray, axis: np.ndarray, previous: float, angle_sign: float
) -> float:
    rotvec = Rotation.from_matrix(delta_pose[:3, :3]).as_rotvec()
    candidate = float(angle_sign) * float(rotvec @ axis)
    return float(wrap_near(candidate, previous))


def estimate_fixed_axis_sequence(
    local_points_c0: np.ndarray,
    observed_points_c0: np.ndarray,
    valid: np.ndarray,
    confidence: np.ndarray,
    raw_delta_poses: np.ndarray,
    origin: np.ndarray,
    axis: np.ndarray,
    *,
    min_axis_radius_m: float,
    max_axis_coordinate_error_m: float,
    max_radius_error_m: float,
    max_angle_residual_deg: float,
    min_angle_points: int,
    angle_sign: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Fit one angle per frame from C0 RGB-D correspondences; output exact line rotations."""
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis) + 1e-12
    origin = np.asarray(origin, dtype=np.float64)
    frame_count = len(observed_points_c0)
    angles = np.zeros(frame_count, dtype=np.float64)
    transforms = np.repeat(np.eye(4, dtype=np.float64)[None], frame_count, axis=0)
    diagnostics: list[dict[str, Any]] = []
    previous = 0.0
    max_residual = math.radians(max_angle_residual_deg)
    for frame in range(frame_count):
        finite = (
            valid[frame]
            & np.isfinite(local_points_c0).all(axis=1)
            & np.isfinite(observed_points_c0[frame]).all(axis=1)
            & np.isfinite(confidence[frame])
        )
        indices = np.flatnonzero(finite)
        point_count_before_geometry = int(len(indices))
        if len(indices):
            reference = local_points_c0[indices] - origin
            observed = observed_points_c0[frame, indices] - origin
            reference_axis = reference @ axis
            observed_axis = observed @ axis
            reference_radial = reference - reference_axis[:, None] * axis
            observed_radial = observed - observed_axis[:, None] * axis
            reference_radius = np.linalg.norm(reference_radial, axis=1)
            observed_radius = np.linalg.norm(observed_radial, axis=1)
            geometry_ok = (
                (reference_radius >= min_axis_radius_m)
                & (observed_radius >= min_axis_radius_m)
                & (np.abs(observed_axis - reference_axis) <= max_axis_coordinate_error_m)
                & (np.abs(observed_radius - reference_radius) <= max_radius_error_m)
            )
            indices = indices[geometry_ok]
            reference_radial = reference_radial[geometry_ok]
            observed_radial = observed_radial[geometry_ok]
            reference_radius = reference_radius[geometry_ok]
        if len(indices):
            sine = np.einsum("ij,j->i", np.cross(reference_radial, observed_radial), axis)
            cosine = np.einsum("ij,ij->i", reference_radial, observed_radial)
            votes = float(angle_sign) * np.arctan2(sine, cosine)
            votes = wrap_near(votes, previous)
            weights = np.clip(confidence[frame, indices].astype(np.float64), 0.0, 1.0)
            weights *= np.maximum(reference_radius, min_axis_radius_m) ** 2
            center = weighted_median(votes, weights)
            residual = np.abs(votes - center)
            mad = float(np.median(residual)) if len(residual) else float("nan")
            robust_gate = min(max_residual, max(math.radians(3.0), 3.5 * 1.4826 * mad))
            inliers = residual <= robust_gate
        else:
            votes = np.empty(0, dtype=np.float64)
            weights = np.empty(0, dtype=np.float64)
            residual = np.empty(0, dtype=np.float64)
            inliers = np.empty(0, dtype=bool)
            mad = float("nan")
            robust_gate = max_residual

        if int(inliers.sum()) >= min_angle_points:
            angle = float(np.average(votes[inliers], weights=np.maximum(weights[inliers], 1e-12)))
            status = "rgbd_axis_fit"
            residual_deg = float(np.rad2deg(np.median(np.abs(votes[inliers] - angle))))
        elif frame == 0:
            angle = 0.0
            status = "reference_identity"
            residual_deg = None
        else:
            angle = previous
            status = "hold_previous_insufficient_axis_points"
            residual_deg = None
        angles[frame] = angle
        transforms[frame] = axis_rotation_transform(origin, axis, angle)
        previous = angle
        diagnostics.append(
            {
                "frame_index": frame,
                "status": status,
                "theta_rad": angle,
                "theta_deg": float(np.rad2deg(angle)),
                "candidate_count_before_geometry_gate": point_count_before_geometry,
                "candidate_count": int(len(indices)),
                "inlier_count": int(inliers.sum()),
                "vote_mad_deg": float(np.rad2deg(mad)) if np.isfinite(mad) else None,
                "robust_gate_deg": float(np.rad2deg(robust_gate)),
                "median_inlier_residual_deg": residual_deg,
            }
        )
    return angles, transforms, diagnostics


def load_reference_transform(part: dict[str, Any]) -> np.ndarray:
    value = part.get("reference_transform")
    if value is None:
        return np.eye(4, dtype=np.float64)
    if isinstance(value, Path):
        transform = np.load(value).astype(np.float64)
    else:
        transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(f"Invalid reference transform for {part['part_id']}: {transform.shape}")
    return transform


def export_dynamic_meshes(
    mesh_path: Path,
    transforms: np.ndarray,
    output_dir: Path,
    part_id: str,
) -> list[str]:
    mesh = trimesh.load(mesh_path, process=False, force="mesh")
    paths: list[str] = []
    for frame, transform in enumerate(transforms):
        path = output_dir / f"frame_{frame:06d}" / f"{part_id}_C0.glb"
        path.parent.mkdir(parents=True, exist_ok=True)
        moved = mesh.copy()
        moved.apply_transform(transform)
        moved.export(path)
        paths.append(str(path))
    return paths


def update_pipeline_state(workspace: Path, output_root: Path, status: str, notes: str) -> None:
    path = workspace / "pipeline_state.json"
    if not path.is_file():
        return
    state = read_json(path)
    stage_name = "10_cotracker3_articulated_motion"
    matches = [stage for stage in state.get("stages", []) if stage.get("stage") == stage_name]
    if not matches:
        record = {"stage": stage_name, "status": "pending", "inputs": [], "outputs": [], "notes": ""}
        state.setdefault("stages", []).append(record)
    elif len(matches) == 1:
        record = matches[0]
    else:
        raise ValueError(f"pipeline_state.json contains {len(matches)} records for {stage_name}")
    record.update({"status": status, "outputs": [str(output_root)], "notes": notes})
    write_json(path, state)


def build_raw_command(
    args: argparse.Namespace,
    workspace: Path,
    rgb_dir: Path,
    part: dict[str, Any],
    anchors: list[int],
    raw_dir: Path,
    transform_path: Path,
    poses_path: Path,
    depth_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/rigid_stage08_track_pose.py"),
        "--workspace",
        str(workspace),
        "--output-dir",
        str(raw_dir),
        "--rgb-dir",
        str(rgb_dir),
        "--depth-dir",
        str(depth_dir),
        "--poses-path",
        str(poses_path),
        "--mask-dir",
        str(part["mask_dir"]),
        "--transform0",
        str(transform_path),
        "--aligned-mesh",
        str(part["mesh_c0"]),
        "--anchor-frames",
        ",".join(str(value) for value in anchors),
        "--queries-per-anchor",
        str(args.queries_per_anchor),
        "--query-mask-margin-px",
        str(args.query_mask_margin_px),
        "--tracker-confidence",
        str(args.tracker_confidence),
        "--allow-nonzero-first-anchor",
        "--exclude-depth-mask-dir",
        str(workspace / "outputs/02_hand_masks/combined"),
        "--no-enable-pnp",
        "--skip-stage-state-update",
    ]
    command.append("--enable-icp" if args.enable_raw_icp else "--no-enable-icp")
    if args.query_points_json is not None:
        command.extend(["--query-points-json", str(args.query_points_json.resolve())])
    if args.reuse_tracks:
        command.append("--reuse-tracks")
    return command


def select_stable_tracks(
    local_points: np.ndarray,
    observed_points: np.ndarray,
    valid: np.ndarray,
    confidence: np.ndarray,
    query_times: np.ndarray,
    *,
    count: int,
    min_valid_ratio: float,
    min_median_confidence: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    finite = np.isfinite(observed_points).all(axis=2)
    usable = valid & finite
    valid_ratio = usable.mean(axis=0)
    median_confidence = np.asarray(
        [np.median(confidence[usable[:, i], i]) if usable[:, i].any() else 0.0 for i in range(valid.shape[1])]
    )
    eligible = (
        (query_times == 0)
        & np.isfinite(local_points).all(axis=1)
        & (valid_ratio >= min_valid_ratio)
        & (median_confidence >= min_median_confidence)
    )
    score = valid_ratio * median_confidence
    indices = np.flatnonzero(eligible)
    indices = indices[np.argsort(score[indices])[::-1]][:count]
    if len(indices) < 3:
        fallback = np.flatnonzero((query_times == 0) & np.isfinite(local_points).all(axis=1))
        indices = fallback[np.argsort(score[fallback])[::-1]][:count]
    records = [
        {
            "track_index": int(index),
            "valid_ratio": float(valid_ratio[index]),
            "median_confidence": float(median_confidence[index]),
            "selection_score": float(score[index]),
            "met_strict_thresholds": bool(eligible[index]),
        }
        for index in indices
    ]
    return indices, records


def select_interval_stable_tracks(
    tracks_xy: np.ndarray,
    observed_points: np.ndarray,
    confidence: np.ndarray,
    query_times: np.ndarray,
    intervals: list[dict[str, Any]],
    anchor_reference_ok: dict[int, np.ndarray],
    *,
    count: int,
    min_valid_ratio: float,
    min_median_confidence: float,
    min_separation_px: float,
) -> tuple[dict[int, np.ndarray], list[dict[str, Any]]]:
    selected: dict[int, np.ndarray] = {}
    records: list[dict[str, Any]] = []
    for interval in intervals:
        start = int(interval["start_frame"])
        end = int(interval["end_frame"])
        segment = slice(start, end + 1)
        finite_2d = np.isfinite(tracks_xy[segment]).all(axis=2)
        visible = finite_2d & (confidence[segment] >= min_median_confidence)
        valid_ratio = visible.mean(axis=0)
        median_confidence = np.median(confidence[segment], axis=0)
        anchor_has_depth = np.isfinite(observed_points[start]).all(axis=1)
        anchor_geometry_ok = anchor_reference_ok[start]
        candidates = np.flatnonzero(
            (query_times == start)
            & anchor_has_depth
            & anchor_geometry_ok
            & (valid_ratio >= min_valid_ratio)
            & (median_confidence >= min_median_confidence)
        )
        score = valid_ratio * median_confidence
        ordered = candidates[np.argsort(score[candidates])[::-1]]
        chosen: list[int] = []
        for index in ordered:
            point = tracks_xy[start, index]
            if all(
                np.linalg.norm(point - tracks_xy[start, other]) >= min_separation_px
                for other in chosen
            ):
                chosen.append(int(index))
            if len(chosen) >= count:
                break
        indices = np.asarray(chosen, dtype=np.int64)
        if len(indices) < min(2, count):
            fallback = np.flatnonzero(
                (query_times == start) & anchor_has_depth & anchor_geometry_ok
            )
            ordered_fallback = fallback[np.argsort(score[fallback])[::-1]]
            chosen = list(indices)
            for index in ordered_fallback:
                if int(index) not in chosen:
                    chosen.append(int(index))
                if len(chosen) >= count:
                    break
            indices = np.asarray(chosen, dtype=np.int64)
        selected[start] = indices
        for index in indices:
            records.append(
                {
                    "event_id": interval.get("event_id"),
                    "interaction_start_frame": start,
                    "interaction_end_frame": end,
                    "track_index": int(index),
                    "valid_ratio_within_interaction": float(valid_ratio[index]),
                    "median_confidence_within_interaction": float(median_confidence[index]),
                    "selection_score": float(score[index]),
                    "met_strict_thresholds": bool(index in candidates),
                }
            )
    return selected, records


def project_c0_points(
    points_c0: np.ndarray, transform_c0_from_ct: np.ndarray, K: np.ndarray
) -> np.ndarray:
    transform_ct_from_c0 = np.linalg.inv(transform_c0_from_ct)
    points_ct = points_c0 @ transform_ct_from_c0[:3, :3].T + transform_ct_from_c0[:3, 3]
    pixels_h = points_ct @ K.T
    return pixels_h[:, :2] / np.maximum(pixels_h[:, 2:3], 1e-8)


def transform_per_frame_points_to_c0(
    points_ct: np.ndarray, transforms_c0_from_ct: np.ndarray
) -> np.ndarray:
    """Transform each frame's depth-lifted Ct points into the common C0 frame."""
    points_ct = np.asarray(points_ct, dtype=np.float64)
    transforms_c0_from_ct = np.asarray(transforms_c0_from_ct, dtype=np.float64)
    if points_ct.ndim != 3 or points_ct.shape[2] != 3:
        raise ValueError(f"Expected points_ct with shape (T, N, 3), got {points_ct.shape}")
    if transforms_c0_from_ct.shape != (len(points_ct), 4, 4):
        raise ValueError(
            "Expected one 4x4 T_C0_from_Ct transform per point frame, got "
            f"{transforms_c0_from_ct.shape} for {len(points_ct)} frames"
        )
    points_c0 = np.full_like(points_ct, np.nan, dtype=np.float64)
    for frame, transform in enumerate(transforms_c0_from_ct):
        finite = np.isfinite(points_ct[frame]).all(axis=1)
        points_c0[frame, finite] = (
            points_ct[frame, finite] @ transform[:3, :3].T + transform[:3, 3]
        )
    return points_c0


def select_left_upper_track(
    tracks_xy: np.ndarray,
    sampled_depth: np.ndarray,
    confidence: np.ndarray,
    query_times: np.ndarray,
    intervals: list[dict[str, Any]],
) -> tuple[dict[int, np.ndarray], list[dict[str, Any]]]:
    selected: dict[int, np.ndarray] = {}
    records: list[dict[str, Any]] = []
    for interval in intervals:
        start = int(interval["start_frame"])
        candidates = np.flatnonzero(
            (query_times == start)
            & np.isfinite(tracks_xy[start]).all(axis=1)
            & np.isfinite(sampled_depth[start])
        )
        if not len(candidates):
            raise ValueError(f"No query at frame {start} has valid anchor depth")
        points = tracks_xy[start, candidates]
        upper_left_score = points[:, 0] + points[:, 1]
        index = int(candidates[np.argmin(upper_left_score)])
        selected[start] = np.asarray([index], dtype=np.int64)
        records.append(
            {
                "interaction_start_frame": start,
                "track_index": index,
                "anchor_xy": tracks_xy[start, index].tolist(),
                "anchor_depth_m": float(sampled_depth[start, index]),
                "anchor_confidence": float(confidence[start, index]),
                "selection_policy": "upper_left_query_with_valid_anchor_depth",
            }
        )
    return selected, records


def lift_tracks_with_depth_interpolation(
    tracks_xy: np.ndarray,
    sampled_depth: np.ndarray,
    depth_rejection_codes: np.ndarray,
    query_times: np.ndarray,
    transforms_c0_from_ct: np.ndarray,
    intrinsics: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Lift each query with accepted current depth or two-sided gap interpolation."""
    tracks_xy = np.asarray(tracks_xy, dtype=np.float64)
    sampled_depth = np.asarray(sampled_depth, dtype=np.float64)
    depth_rejection_codes = np.asarray(depth_rejection_codes, dtype=np.uint8)
    query_times = np.asarray(query_times, dtype=np.int64)
    if depth_rejection_codes.shape != sampled_depth.shape:
        raise ValueError(
            "depth_rejection_codes must have the same [frames, tracks] shape as sampled_depth"
        )
    fx, fy = float(intrinsics["fx"]), float(intrinsics["fy"])
    cx, cy = float(intrinsics["cx"]), float(intrinsics["cy"])
    points_ct = np.full((*tracks_xy.shape[:2], 3), np.nan, dtype=np.float64)
    used_depth = np.full(tracks_xy.shape[:2], np.nan, dtype=np.float64)
    depth_source = np.zeros(tracks_xy.shape[:2], dtype=np.uint8)
    for track, start in enumerate(query_times):
        for frame in range(max(0, int(start)), len(tracks_xy)):
            if not np.isfinite(tracks_xy[frame, track]).all():
                continue
            current_depth = float(sampled_depth[frame, track])
            current_accepted = (
                int(depth_rejection_codes[frame, track]) not in {1, 2, 3, 4, 5, 6}
                and np.isfinite(current_depth)
                and current_depth > 0.0
            )
            if current_accepted:
                depth = current_depth
                depth_source[frame, track] = 1
            else:
                previous_valid = float("nan")
                for previous_frame in range(frame - 1, int(start) - 1, -1):
                    candidate = float(sampled_depth[previous_frame, track])
                    if (
                        int(depth_rejection_codes[previous_frame, track]) not in {1, 2, 3, 4, 5, 6}
                        and np.isfinite(candidate)
                        and candidate > 0.0
                    ):
                        previous_valid = candidate
                        break
                next_valid = float("nan")
                for next_frame in range(frame + 1, len(tracks_xy)):
                    candidate = float(sampled_depth[next_frame, track])
                    if (
                        int(depth_rejection_codes[next_frame, track]) not in {1, 2, 3, 4, 5, 6}
                        and np.isfinite(candidate)
                        and candidate > 0.0
                    ):
                        next_valid = candidate
                        break
                if np.isfinite(previous_valid) and np.isfinite(next_valid):
                    depth = 0.5 * (previous_valid + next_valid)
                    depth_source[frame, track] = 3
                else:
                    continue
            u, v = tracks_xy[frame, track]
            points_ct[frame, track] = (
                (u - cx) * depth / fx,
                (v - cy) * depth / fy,
                depth,
            )
            used_depth[frame, track] = depth
    points_c0 = transform_per_frame_points_to_c0(points_ct, transforms_c0_from_ct)
    return points_ct, points_c0, used_depth, depth_source



def mesh_clearance_evaluator(
    base_mesh: trimesh.Trimesh,
    moving_mesh: trimesh.Trimesh,
    origin: np.ndarray,
    axis: np.ndarray,
    hinge_exclusion_m: float,
):
    base_vertices = np.asarray(base_mesh.vertices, dtype=np.float64)
    moving_vertices = np.asarray(moving_mesh.vertices, dtype=np.float64)
    base_radius = np.linalg.norm(np.cross(base_vertices - origin, axis), axis=1)
    moving_radius = np.linalg.norm(np.cross(moving_vertices - origin, axis), axis=1)
    base_vertices = base_vertices[base_radius >= hinge_exclusion_m]
    moving_vertices = moving_vertices[moving_radius >= hinge_exclusion_m]
    if len(base_vertices) > 12000:
        base_vertices = base_vertices[np.linspace(0, len(base_vertices) - 1, 12000, dtype=np.int64)]
    if len(moving_vertices) > 3000:
        moving_vertices = moving_vertices[np.linspace(0, len(moving_vertices) - 1, 3000, dtype=np.int64)]
    tree = cKDTree(base_vertices)

    def clearance(angle_rad: float) -> float:
        transform = axis_rotation_transform(origin, axis, float(angle_rad))
        moving = moving_vertices @ transform[:3, :3].T + transform[:3, 3]
        distances, _ = tree.query(moving, k=1, workers=-1)
        return float(np.quantile(distances, 0.01))

    return clearance


def find_first_contact_angle(
    previous_deg: float,
    closing_direction: float,
    lower_deg: float,
    upper_deg: float,
    clearance,
    clearance_m: float,
    step_deg: float,
) -> tuple[float, bool, float]:
    values = (
        np.arange(previous_deg, upper_deg + step_deg * 0.5, step_deg)
        if closing_direction >= 0.0
        else np.arange(previous_deg, lower_deg - step_deg * 0.5, -step_deg)
    )
    last_safe = float(previous_deg)
    for value in values:
        if clearance(np.deg2rad(value)) < clearance_m:
            safe = float(last_safe)
            colliding = float(value)
            for _ in range(32):
                middle = 0.5 * (safe + colliding)
                if clearance(np.deg2rad(middle)) >= clearance_m:
                    safe = middle
                else:
                    colliding = middle
            return float(0.5 * (safe + colliding)), True, float(safe)
        last_safe = float(value)
    return float(last_safe), False, float(last_safe)


def apply_contact_angle_constraint(
    angles: np.ndarray,
    interaction_intervals: list[dict[str, Any]],
    moving_mesh: trimesh.Trimesh,
    base_mesh: trimesh.Trimesh,
    origin: np.ndarray,
    axis: np.ndarray,
    *,
    clearance_m: float,
    hinge_exclusion_m: float,
    step_deg: float,
    diagnostics: list[dict[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    constrained = np.asarray(angles, dtype=np.float64).copy()
    clearance = mesh_clearance_evaluator(base_mesh, moving_mesh, origin, axis, hinge_exclusion_m)
    interval_records = []
    for interval in interaction_intervals:
        start = int(interval["start_frame"])
        end = int(interval["end_frame"])
        action = str(interval.get("action", "close")).lower()
        closing_direction = -1.0 if action in {"close", "closing"} else 1.0
        contact_reached = False
        for frame in range(start, min(end, len(constrained) - 1) + 1):
            previous = float(constrained[frame - 1] if frame > start else constrained[start])
            desired = float(constrained[frame])
            desired_deg = float(np.rad2deg(desired))
            previous_deg = float(np.rad2deg(previous))
            if contact_reached:
                contact_angle = previous_deg
                safe_angle = previous_deg
                contact_found = True
                constrained[frame] = previous
                diagnostics[frame]["status"] = "hold_contact_angle"
            else:
                contact_angle, contact_found, safe_angle = find_first_contact_angle(
                    previous_deg,
                    closing_direction,
                    -180.0,
                    180.0,
                    clearance,
                    clearance_m,
                    step_deg,
                )
                crosses_contact = (
                    desired_deg <= contact_angle if closing_direction < 0.0 else desired_deg >= contact_angle
                )
                if contact_found and crosses_contact:
                    constrained[frame] = np.deg2rad(contact_angle)
                    contact_reached = True
                    diagnostics[frame]["status"] = "contact_angle_applied"
                else:
                    contact_angle = float(np.rad2deg(constrained[frame]))
            diagnostics[frame].update(
                {
                    "contact_angle_constraint": "applied" if contact_reached else "not_reached",
                    "contact_angle_deg": float(np.rad2deg(constrained[frame])),
                    "first_contact_angle_deg": float(contact_angle),
                    "last_safe_angle_deg": float(safe_angle),
                    "contact_delta_from_previous_deg": abs(float(np.rad2deg(constrained[frame])) - previous_deg),
                    "lid_base_clearance_m": clearance(float(constrained[frame])),
                    "lid_base_collision_threshold_m": clearance_m,
                }
            )
        interval_records.append(
            {
                "start_frame": start,
                "end_frame": end,
                "action": action,
                "contact_reached": contact_reached,
                "contact_frame": next(
                    (frame for frame in range(start, min(end, len(constrained) - 1) + 1)
                     if diagnostics[frame].get("status") in {"contact_angle_applied", "hold_contact_angle"}),
                    None,
                ),
            }
        )
    return constrained, {
        "enabled": True,
        "method": "lid_base_mesh_clearance_1pct_quantile_with_bisection",
        "clearance_threshold_m": clearance_m,
        "hinge_exclusion_m": hinge_exclusion_m,
        "collision_step_deg": step_deg,
        "intervals": interval_records,
    }


def estimate_single_track_axis_sequence(
    observed_points: np.ndarray,
    confidence: np.ndarray,
    used_depth: np.ndarray,
    depth_source: np.ndarray,
    selected_by_start: dict[int, np.ndarray],
    intervals: list[dict[str, Any]],
    origin: np.ndarray,
    axis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Compute the hinge angle directly from one pose-corrected 3D corner track."""
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis) + 1e-12
    origin = np.asarray(origin, dtype=np.float64)
    angles = np.zeros(len(observed_points), dtype=np.float64)
    diagnostics = [
        {
            "frame_index": frame,
            "status": "hold_previous_outside_vlm_interaction",
            "theta_rad": 0.0,
            "theta_deg": 0.0,
            "inlier_count": 0,
        }
        for frame in range(len(observed_points))
    ]
    previous = 0.0
    cursor = 0
    for interval in intervals:
        start, end = int(interval["start_frame"]), int(interval["end_frame"])
        angles[cursor:start] = previous
        indices = selected_by_start.get(start, np.empty(0, dtype=np.int64))
        if len(indices) != 1:
            raise ValueError(f"Expected exactly one upper-left track at frame {start}, got {indices}")
        index = int(indices[0])
        reference = observed_points[start, index] - origin
        reference_radial = reference - float(reference @ axis) * axis
        if not np.isfinite(reference_radial).all() or np.linalg.norm(reference_radial) < 1e-8:
            raise ValueError(f"Invalid upper-left reference point at frame {start}")
        base_angle = previous
        previous_delta = 0.0
        for frame in range(start, end + 1):
            observed = observed_points[frame, index] - origin
            observed_radial = observed - float(observed @ axis) * axis
            usable = (
                np.isfinite(observed_radial).all()
                and np.isfinite(confidence[frame, index])
                and confidence[frame, index] >= 0.5
            )
            if usable:
                sine = float(np.cross(reference_radial, observed_radial) @ axis)
                cosine = float(reference_radial @ observed_radial)
                delta = float(wrap_near(np.arctan2(sine, cosine), previous_delta))
                previous_delta = delta
                previous = base_angle + delta
                source = int(depth_source[frame, index])
                status_by_source = {
                    1: "single_upper_left_track_current_depth",
                    2: "single_upper_left_track_previous_depth_fallback",
                    3: "single_upper_left_track_interpolated_depth",
                    4: "single_upper_left_track_next_depth_fallback",
                }
                status = status_by_source.get(source, "single_upper_left_track_depth_fallback")
            else:
                status = "hold_previous_unusable_upper_left_track"
            angles[frame] = previous
            diagnostics[frame] = {
                "frame_index": frame,
                "event_id": interval.get("event_id"),
                "action": interval.get("action"),
                "interaction_start_frame": start,
                "track_index": index,
                "status": status,
                "theta_rad": float(previous),
                "theta_deg": float(np.rad2deg(previous)),
                "confidence": float(confidence[frame, index]),
                "used_depth_m": (
                    float(used_depth[frame, index])
                    if np.isfinite(used_depth[frame, index])
                    else None
                ),
                "depth_source": {
                    1: "current_frame",
                    2: "previous_frame_fallback",
                    3: "previous_next_average",
                    4: "next_frame_fallback",
                }.get(int(depth_source[frame, index]), "unavailable"),
                "candidate_count": int(usable),
                "inlier_count": int(usable),
            }
        cursor = end + 1
    angles[cursor:] = previous
    for frame in range(len(angles)):
        diagnostics[frame]["theta_rad"] = float(angles[frame])
        diagnostics[frame]["theta_deg"] = float(np.rad2deg(angles[frame]))
    transforms = np.stack([axis_rotation_transform(origin, axis, angle) for angle in angles])
    return angles, transforms, diagnostics


def estimate_interaction_axis_sequence_reprojection(
    tracks_xy: np.ndarray,
    observed_points: np.ndarray,
    confidence: np.ndarray,
    selected_by_start: dict[int, np.ndarray],
    intervals: list[dict[str, Any]],
    transforms_c0_from_ct: np.ndarray,
    K: np.ndarray,
    origin: np.ndarray,
    axis: np.ndarray,
    *,
    min_angle_points: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Lift points at interaction start, then fit the hinge angle by 2D reprojection."""
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis) + 1e-12
    origin = np.asarray(origin, dtype=np.float64)
    angles = np.zeros(len(tracks_xy), dtype=np.float64)
    diagnostics = [
        {
            "frame_index": frame,
            "status": "hold_previous_outside_vlm_interaction",
            "theta_rad": 0.0,
            "theta_deg": 0.0,
            "inlier_count": 0,
        }
        for frame in range(len(tracks_xy))
    ]
    previous = 0.0
    cursor = 0
    for interval in intervals:
        start, end = int(interval["start_frame"]), int(interval["end_frame"])
        angles[cursor:start] = previous
        indices = selected_by_start.get(start, np.empty(0, dtype=np.int64))
        reference_points = observed_points[start, indices]
        surviving = np.isfinite(reference_points).all(axis=1)
        base_angle = previous
        previous_delta = 0.0
        for frame in range(start, end + 1):
            visible_now = (
                np.isfinite(reference_points).all(axis=1)
                & np.isfinite(tracks_xy[frame, indices]).all(axis=1)
                & (confidence[frame, indices] >= 0.5)
            )
            usable = visible_now
            use = np.flatnonzero(usable)
            if frame == start:
                delta = 0.0
                status = "interaction_reference_depth_lift"
                residual_px = 0.0
            elif len(use) >= min_angle_points:
                points = reference_points[use]
                targets = tracks_xy[frame, indices[use]].astype(np.float64)
                weights = np.sqrt(np.clip(confidence[frame, indices[use]], 0.1, 1.0))

                def residual(value: np.ndarray) -> np.ndarray:
                    transform = axis_rotation_transform(origin, axis, float(value[0]))
                    moved = points @ transform[:3, :3].T + transform[:3, 3]
                    projected = project_c0_points(moved, transforms_c0_from_ct[frame], K)
                    return ((projected - targets) * weights[:, None]).reshape(-1)

                step = math.radians(50.0)
                lower = max(-math.pi, previous_delta - step)
                upper = min(math.pi, previous_delta + step)
                fit = least_squares(
                    residual,
                    np.asarray([np.clip(previous_delta, lower + 1e-6, upper - 1e-6)]),
                    bounds=([lower], [upper]),
                    loss="huber",
                    f_scale=3.0,
                )
                delta = float(fit.x[0])
                residual_px = float(
                    np.median(np.linalg.norm(residual(fit.x).reshape(-1, 2), axis=1))
                )
                status = (
                    "axis_angle_from_single_track_2d_reprojection"
                    if len(use) == 1
                    else "axis_angle_from_2d_reprojection"
                )
            else:
                delta = previous_delta
                residual_px = None
                status = "hold_previous_insufficient_visible_tracks"
            previous_delta = delta
            previous = base_angle + delta
            angles[frame] = previous
            diagnostics[frame] = {
                "frame_index": frame,
                "event_id": interval.get("event_id"),
                "action": interval.get("action"),
                "interaction_start_frame": start,
                "status": status,
                "theta_rad": float(previous),
                "theta_deg": float(np.rad2deg(previous)),
                "candidate_count": int(len(use)),
                "inlier_count": int(len(use)),
                "median_weighted_reprojection_error_px": residual_px,
            }
        cursor = end + 1
    angles[cursor:] = previous
    for interval in intervals:
        start, end = int(interval["start_frame"]), int(interval["end_frame"])
        segment = angles[start : end + 1]
        if len(segment) > 1:
            if segment[-1] >= segment[0]:
                angles[start : end + 1] = np.maximum.accumulate(segment)
            else:
                angles[start : end + 1] = np.minimum.accumulate(segment)
        if end + 1 < len(angles):
            next_start = min(
                [
                    int(other["start_frame"])
                    for other in intervals
                    if int(other["start_frame"]) > end
                ]
                or [len(angles)]
            )
            angles[end + 1 : next_start] = angles[end]
    angles = (angles + np.pi) % (2.0 * np.pi) - np.pi
    for frame in range(len(angles)):
        diagnostics[frame]["theta_rad"] = float(angles[frame])
        diagnostics[frame]["theta_deg"] = float(np.rad2deg(angles[frame]))
    transforms = np.stack([axis_rotation_transform(origin, axis, angle) for angle in angles])
    return angles, transforms, diagnostics


def estimate_interaction_axis_sequence(
    observed_points: np.ndarray,
    confidence: np.ndarray,
    selected_by_start: dict[int, np.ndarray],
    intervals: list[dict[str, Any]],
    origin: np.ndarray,
    axis: np.ndarray,
    *,
    min_axis_radius_m: float,
    max_axis_coordinate_error_m: float,
    max_radius_error_m: float,
    max_angle_residual_deg: float,
    min_angle_points: int,
    angle_sign: float,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Estimate incremental angles per VLM interaction, holding a strict hinge elsewhere."""
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis) + 1e-12
    origin = np.asarray(origin, dtype=np.float64)
    angles = np.zeros(len(observed_points), dtype=np.float64)
    diagnostics = [
        {
            "frame_index": frame,
            "status": "hold_previous_outside_vlm_interaction",
            "theta_rad": 0.0,
            "theta_deg": 0.0,
            "inlier_count": 0,
        }
        for frame in range(len(observed_points))
    ]
    previous = 0.0
    cursor = 0
    max_residual = math.radians(max_angle_residual_deg)
    for interval in intervals:
        start, end = int(interval["start_frame"]), int(interval["end_frame"])
        angles[cursor:start] = previous
        indices = selected_by_start.get(start, np.empty(0, dtype=np.int64))
        reference = observed_points[start, indices] - origin
        reference_axis = reference @ axis if len(indices) else np.empty(0)
        reference_radial = reference - reference_axis[:, None] * axis if len(indices) else reference
        reference_radius = np.linalg.norm(reference_radial, axis=1) if len(indices) else np.empty(0)
        base_angle = previous
        for frame in range(start, end + 1):
            observed = observed_points[frame, indices] - origin
            finite = (
                np.isfinite(reference).all(axis=1)
                & np.isfinite(observed).all(axis=1)
                & np.isfinite(confidence[frame, indices])
                & (confidence[frame, indices] >= 0.5)
            )
            observed_axis = observed @ axis
            observed_radial = observed - observed_axis[:, None] * axis
            observed_radius = np.linalg.norm(observed_radial, axis=1)
            geometry_ok = (
                finite
                & (reference_radius >= min_axis_radius_m)
                & (observed_radius >= min_axis_radius_m)
                & (np.abs(observed_axis - reference_axis) <= max_axis_coordinate_error_m)
                & (np.abs(observed_radius - reference_radius) <= max_radius_error_m)
            )
            use = np.flatnonzero(geometry_ok)
            if len(use):
                sine = np.einsum(
                    "ij,j->i", np.cross(reference_radial[use], observed_radial[use]), axis
                )
                cosine = np.einsum("ij,ij->i", reference_radial[use], observed_radial[use])
                votes = float(angle_sign) * np.arctan2(sine, cosine)
                votes = wrap_near(votes, previous - base_angle)
                weights = np.clip(confidence[frame, indices[use]], 0.0, 1.0)
                weights *= np.maximum(reference_radius[use], min_axis_radius_m) ** 2
                center = weighted_median(votes, weights)
                residual = np.abs(votes - center)
                mad = float(np.median(residual))
                gate = min(max_residual, max(math.radians(2.0), 3.5 * 1.4826 * mad))
                inliers = residual <= gate
            else:
                votes = weights = residual = np.empty(0)
                inliers = np.empty(0, dtype=bool)
                mad, gate = float("nan"), max_residual
            if int(inliers.sum()) >= min_angle_points:
                delta = float(
                    np.average(votes[inliers], weights=np.maximum(weights[inliers], 1e-12))
                )
                previous = base_angle + delta
                status = "rgbd_axis_fit_interaction_relative"
                residual_deg = float(np.rad2deg(np.median(np.abs(votes[inliers] - delta))))
            else:
                status = "hold_previous_insufficient_axis_points"
                residual_deg = None
            angles[frame] = previous
            diagnostics[frame] = {
                "frame_index": frame,
                "event_id": interval.get("event_id"),
                "action": interval.get("action"),
                "interaction_start_frame": start,
                "status": status,
                "theta_rad": float(previous),
                "theta_deg": float(np.rad2deg(previous)),
                "candidate_count": int(len(use)),
                "inlier_count": int(inliers.sum()),
                "vote_mad_deg": float(np.rad2deg(mad)) if np.isfinite(mad) else None,
                "robust_gate_deg": float(np.rad2deg(gate)),
                "median_inlier_residual_deg": residual_deg,
            }
        cursor = end + 1
    angles[cursor:] = previous
    for frame in range(len(angles)):
        diagnostics[frame]["theta_rad"] = float(angles[frame])
        diagnostics[frame]["theta_deg"] = float(np.rad2deg(angles[frame]))
    transforms = np.stack([axis_rotation_transform(origin, axis, angle) for angle in angles])
    return angles, transforms, diagnostics


def apply_vlm_interaction_gating(
    angles: np.ndarray,
    origin: np.ndarray,
    axis: np.ndarray,
    intervals: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    active = np.zeros(len(angles), dtype=bool)
    for interval in intervals:
        active[interval["start_frame"] : interval["end_frame"] + 1] = True
    gated = np.zeros_like(angles)
    previous = 0.0
    for frame, angle in enumerate(angles):
        if active[frame]:
            previous = float(angle)
        gated[frame] = previous
    transforms = np.stack([axis_rotation_transform(origin, axis, angle) for angle in gated])
    return gated, transforms, active


def main() -> int:
    global WORLD_FRAME
    args = parse_args()
    workspace = args.workspace.resolve()
    stage00_path = workspace / "outputs/00_rgb_frames/stage00_manifest.json"
    stage00 = read_json(stage00_path)
    selected_eye = str(stage00.get("selected_eye", "")).lower()
    if selected_eye not in {"left", "right"}:
        raise ValueError(f"Unsupported selected eye in {stage00_path}: {selected_eye!r}")
    WORLD_FRAME = f"frame0_{selected_eye}_camera_opencv_rdf"
    object_id, parts, source_manifest = load_parts(args, workspace)
    rgb_dir = (
        args.rgb_dir or workspace / "outputs/03_diffueraser/inpainted_frames_png"
    ).resolve()
    rgb_paths = discover_images(rgb_dir)
    depth_dir = (
        args.depth_dir.resolve()
        if args.depth_dir is not None
        else workspace / "outputs/06_dense_depth/metric_depth_npy"
    )
    depth_paths = sorted(depth_dir.glob("*.npy"))
    camera_path = workspace / "outputs/00_rgb_frames/camera.json"
    poses_path = (
        args.poses_path.resolve()
        if args.poses_path is not None
        else workspace / "outputs/00_rgb_frames/poses.npz"
    )
    if not rgb_paths:
        raise FileNotFoundError(f"No tracking RGB frames in {rgb_dir}")
    if len(rgb_paths) != len(depth_paths):
        raise ValueError(f"RGB/depth count mismatch: {len(rgb_paths)} vs {len(depth_paths)}")
    for path in (camera_path, poses_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    pose_data = np.load(poses_path)
    transforms_c0_from_ct = pose_data["T_C0_from_Ct"]
    if len(transforms_c0_from_ct) != len(rgb_paths):
        raise ValueError(f"RGB/pose count mismatch: {len(rgb_paths)} vs {len(transforms_c0_from_ct)}")
    camera = read_json(camera_path)
    intrinsics = camera.get("rgb_intrinsics_selected", camera["rgb_intrinsics_right"])
    output_root = (
        args.output_dir or workspace / "outputs/10_articulate_tracking" / object_id
    ).resolve()
    interaction_intervals = vlm_interaction_intervals(workspace, object_id, len(rgb_paths))
    if args.interaction_actions:
        requested_actions = {
            value.strip().lower() for value in args.interaction_actions.split(",") if value.strip()
        }
        interaction_intervals = [
            interval
            for interval in interaction_intervals
            if str(interval.get("action", "")).lower() in requested_actions
        ]
        if not interaction_intervals:
            raise ValueError(
                f"No VLM articulated intervals match --interaction-actions={args.interaction_actions!r}"
            )
    if args.interaction_end_frame is not None:
        if len(interaction_intervals) != 1:
            raise ValueError(
                "--interaction-end-frame requires exactly one selected articulated interval"
            )
        start = int(interaction_intervals[0]["start_frame"])
        end = int(args.interaction_end_frame)
        if not start <= end < len(rgb_paths):
            raise ValueError(
                f"--interaction-end-frame must be in [{start}, {len(rgb_paths) - 1}], got {end}"
            )
        interaction_intervals[0] = {
            **interaction_intervals[0],
            "vlm_end_frame": int(interaction_intervals[0]["end_frame"]),
            "end_frame": end,
            "end_frame_policy": "caller_override",
        }
    fallback_anchors = (
        [interval["start_frame"] for interval in interaction_intervals]
        if interaction_intervals
        else [0]
    )
    plan_parts: list[dict[str, Any]] = []
    for part in parts:
        part_id = part["part_id"]
        for path in (part["mask_dir"], part["mesh_c0"]):
            if not Path(path).exists():
                raise FileNotFoundError(path)
        mask_paths = sorted(Path(part["mask_dir"]).glob("*.png"))
        if len(mask_paths) != len(rgb_paths):
            raise ValueError(f"Part {part_id} RGB/mask count mismatch: {len(rgb_paths)} vs {len(mask_paths)}")
        anchors = parse_anchor_frames(
            part.get("anchor_frames", args.anchor_frames), fallback_anchors, len(rgb_paths)
        )
        part_output = output_root / part_id
        raw_dir = part_output / "raw_se3"
        transform_path = part_output / "T_C0_from_part_reference.npy"
        command = build_raw_command(
            args, workspace, rgb_dir, part, anchors, raw_dir, transform_path, poses_path, depth_dir
        )
        plan_parts.append(
            {
                "part_id": part_id,
                "mask_dir": str(part["mask_dir"]),
                "mesh_c0": str(part["mesh_c0"]),
                "joint_C0": part["joint"],
                "anchor_frames": anchors,
                "raw_command": command,
                "output_dir": str(part_output),
            }
        )
    preflight = {
        "stage": "10_cotracker3_articulated_motion",
        "status": "ready",
        "object_id": object_id,
        "selected_eye": selected_eye,
        "tracking_rgb_dir": str(rgb_dir),
        "tracking_rgb_policy": "diffueraser_hand_removed",
        "frame_count": len(rgb_paths),
        "fps": float(stage00.get("target_fps", 15.0)),
        "world_frame": WORLD_FRAME,
        "pose_compensation": {
            "applied": True,
            "formula": "p_C0(t) = T_C0_from_Ct(t) @ p_Ct(t)",
            "poses": str(poses_path),
        },
        "depth_dir": str(depth_dir),
        "vlm_articulated_intervals": interaction_intervals,
        "track_policy": {
            "count": 1,
            "selection": "upper-left query with valid anchor depth",
            "depth": "accepted current-frame depth; otherwise average nearest accepted previous and next depths",
            "query_frames": [interval["start_frame"] for interval in interaction_intervals],
        },
        "contact_angle_policy": {
            "enabled_by_default": True,
            "method": "first lid/base contact by mesh-clearance bisection, then hold",
            "clearance_threshold_m": args.collision_clearance_m,
            "hinge_exclusion_m": args.hinge_exclusion_m,
            "collision_step_deg": args.collision_step_deg,
        },
        "source_parts_manifest": str(source_manifest) if source_manifest else None,
        "parts": plan_parts,
    }
    if args.check:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "tracking_preflight.json", preflight)
    part_manifests: list[dict[str, Any]] = []
    for part, plan in zip(parts, plan_parts):
        part_id = part["part_id"]
        part_output = Path(plan["output_dir"])
        raw_dir = part_output / "raw_se3"
        transform_path = part_output / "T_C0_from_part_reference.npy"
        part_output.mkdir(parents=True, exist_ok=True)
        reference_transform = load_reference_transform(part)
        np.save(transform_path, reference_transform)
        print("$ " + " ".join(plan["raw_command"]), flush=True)
        completed = subprocess.run(plan["raw_command"], cwd=PROJECT_ROOT, check=False)
        if completed.returncode not in (0, 2):
            raise subprocess.CalledProcessError(completed.returncode, plan["raw_command"])

        confidence = np.load(raw_dir / "track_confidence.npy").astype(np.float64)
        tracks_xy = np.load(raw_dir / "tracks_2d.npy").astype(np.float64)
        sampled_depth = np.load(raw_dir / "track_depth_m.npy").astype(np.float64)
        depth_rejection_codes = np.load(raw_dir / "track_rejection_codes.npy").astype(np.uint8)
        query_times = np.load(raw_dir / "query_times.npy").astype(np.int64)
        observed_points_ct, observed_points, used_depth, depth_source = (
            lift_tracks_with_depth_interpolation(
                tracks_xy,
                sampled_depth,
                depth_rejection_codes,
                query_times,
                transforms_c0_from_ct,
                intrinsics,
            )
        )
        anchor_reference_ok = {
            int(interval["start_frame"]): np.isfinite(
                observed_points[int(interval["start_frame"])]
            ).all(axis=1)
            for interval in interaction_intervals
        }
        if args.track_index is not None:
            if not 0 <= args.track_index < tracks_xy.shape[1]:
                raise ValueError(
                    f"--track-index must be in [0, {tracks_xy.shape[1] - 1}], got {args.track_index}"
                )
            selected_by_start = {}
            stable_track_records = []
            for interval in interaction_intervals:
                start = int(interval["start_frame"])
                if int(query_times[args.track_index]) != start:
                    raise ValueError(
                        f"Track {args.track_index} was queried at frame "
                        f"{query_times[args.track_index]}, not interaction start {start}"
                    )
                if not np.isfinite(observed_points[start, args.track_index]).all():
                    raise ValueError(f"Track {args.track_index} has no valid anchor 3D point at frame {start}")
                selected_by_start[start] = np.asarray([args.track_index], dtype=np.int64)
                stable_track_records.append(
                    {
                        "event_id": interval.get("event_id"),
                        "interaction_start_frame": start,
                        "interaction_end_frame": int(interval["end_frame"]),
                        "track_index": int(args.track_index),
                        "selection_policy": "forced_track_index",
                        "anchor_xy": tracks_xy[start, args.track_index].tolist(),
                        "anchor_depth_m": float(used_depth[start, args.track_index]),
                        "anchor_confidence": float(confidence[start, args.track_index]),
                    }
                )
        else:
            selected_by_start, stable_track_records = select_interval_stable_tracks(
                tracks_xy,
                observed_points,
                confidence,
                query_times,
                interaction_intervals,
                anchor_reference_ok,
                count=args.stable_track_count,
                min_valid_ratio=args.min_stable_valid_ratio,
                min_median_confidence=args.min_stable_median_confidence,
                min_separation_px=args.min_stable_track_separation_px,
            )
        joint = part["joint"]
        origin = np.asarray(joint["origin_xyz"], dtype=np.float64)
        axis = np.asarray(joint["axis_xyz"], dtype=np.float64)
        camera_K = np.asarray(
            [
                [intrinsics["fx"], 0.0, intrinsics["cx"]],
                [0.0, intrinsics["fy"], intrinsics["cy"]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        angles, constrained_delta, diagnostics = estimate_interaction_axis_sequence_reprojection(
            tracks_xy,
            observed_points,
            confidence,
            selected_by_start,
            interaction_intervals,
            transforms_c0_from_ct,
            camera_K,
            origin,
            axis,
            min_angle_points=1,
        )
        np.save(part_output / "upper_left_track_3d_Ct.npy", observed_points_ct)
        np.save(part_output / "upper_left_track_3d_C0.npy", observed_points)
        np.save(part_output / "upper_left_track_used_depth_m.npy", used_depth)
        np.save(part_output / "upper_left_track_depth_source.npy", depth_source)
        angle_application_sign = float(part.get("angle_sign", args.angle_sign))
        if angle_application_sign != 1.0:
            angles = angles * angle_application_sign
            constrained_delta = np.stack(
                [axis_rotation_transform(origin, axis, angle) for angle in angles]
            )
            for frame, record in enumerate(diagnostics):
                record["theta_rad"] = float(angles[frame])
                record["theta_deg"] = float(np.rad2deg(angles[frame]))
                record["angle_application_sign"] = angle_application_sign
        contact_angle_manifest = {"enabled": False}
        if args.enable_contact_angle:
            moving_mesh = trimesh.load(part["mesh_c0"], force="mesh", process=False)
            base_path = args.collision_base_mesh
            if base_path is None:
                candidate = Path(part["mesh_c0"]).with_name("part_15.obj")
                base_path = candidate if candidate.exists() else None
            if base_path is not None and Path(base_path).exists():
                base_mesh = trimesh.load(base_path, force="mesh", process=False)
                angles, contact_angle_manifest = apply_contact_angle_constraint(
                    angles,
                    interaction_intervals,
                    moving_mesh,
                    base_mesh,
                    origin,
                    axis,
                    clearance_m=args.collision_clearance_m,
                    hinge_exclusion_m=args.hinge_exclusion_m,
                    step_deg=args.collision_step_deg,
                    diagnostics=diagnostics,
                )
                constrained_delta = np.stack(
                    [axis_rotation_transform(origin, axis, angle) for angle in angles]
                )
            else:
                contact_angle_manifest = {
                    "enabled": True,
                    "status": "skipped_missing_base_mesh",
                    "base_mesh": str(base_path) if base_path is not None else None,
                }
        # Keep the tracked angle without Particulate terminal interpolation or joint-limit clamping.
        poses = np.einsum("tij,jk->tik", constrained_delta, reference_transform)
        np.save(part_output / "joint_angles_rad.npy", angles.astype(np.float32))
        np.save(part_output / "Delta_C0_part_motion_axis_constrained.npy", constrained_delta)
        np.save(part_output / "Delta_C0_object_motion.npy", constrained_delta)
        np.save(part_output / "T_C0_from_part.npy", poses)
        mesh_paths = (
            export_dynamic_meshes(Path(part["mesh_c0"]), constrained_delta, part_output / "dynamic_meshes", part_id)
            if args.export_meshes
            else []
        )
        fallback_count = sum(
            item["status"] == "hold_previous_unusable_upper_left_track"
            for item in diagnostics
        )
        frame_records = []
        for frame, record in enumerate(diagnostics):
            frame_records.append(
                {
                    **record,
                    "Delta_C0_part_motion_axis_constrained": constrained_delta[frame],
                    "T_C0_from_part": poses[frame],
                    "dynamic_mesh_C0": mesh_paths[frame] if mesh_paths else None,
                }
            )
        manifest = {
            "stage": "10_cotracker3_articulated_motion",
            "status": "completed" if fallback_count / len(diagnostics) <= 0.20 else "needs_revision",
            "object_id": object_id,
            "part_id": part_id,
            "reference_mesh_C0": str(part["mesh_c0"]),
            "mask_dir": str(part["mask_dir"]),
            "tracking_rgb_dir": str(rgb_dir),
            "tracking_rgb_policy": "diffueraser_hand_removed",
            "selected_eye": selected_eye,
            "world_frame": WORLD_FRAME,
            "camera_pose_compensation": {
                "applied": True,
                "source": str(poses_path),
                "transform_key": "T_C0_from_Ct",
                "formula": "p_C0(t) = T_C0_from_Ct(t) @ p_Ct(t)",
            },
            "motion_model": "single_upper_left_anchor_depth_fixed_axis_2d_reprojection",
            "contact_angle_constraint": contact_angle_manifest,
            "depth_lifting_policy": {
                "track_selection": "forced track index" if args.track_index is not None else "up to stable high-confidence anchor points; use one point when only one survives",
                "source": str(raw_dir / "track_depth_m.npy"),
                "sampling": "current-frame depth at the current tracked pixel",
                "missing_depth": "average the nearest accepted previous and next frame depths; leave unavailable when either side is absent",
                "rejected_depth": "codes 1-6 are unavailable for lifting; RANSAC/global trajectory rejection does not by itself invalidate a numeric depth sample",
                "pose_correction": "p_C0(t) = T_C0_from_Ct(t) @ p_Ct(t)",
                "angle": "fixed-axis angle fitted by current-frame 2D reprojection of the lifted interaction-start 3D point",
            },
            "angle_application_sign": angle_application_sign,
            "terminal_joint_limit_policy": "ignored",
            "enforce_joint_limits": False,
            "vlm_articulated_intervals": interaction_intervals,
            "stable_tracks": stable_track_records,
            "selected_track_indices_by_interaction_start": {
                str(start): indices.tolist() for start, indices in selected_by_start.items()
            },
            "joint_C0": joint,
            "raw_se3_output": str(raw_dir),
            "raw_tracker_exit_code": int(completed.returncode),
            "frame_count": len(diagnostics),
            "axis_fit_frame_count": len(diagnostics) - fallback_count,
            "fallback_frame_count": fallback_count,
            "artifacts": {
                "joint_angles_rad": str(part_output / "joint_angles_rad.npy"),
                "upper_left_track_3d_Ct": str(part_output / "upper_left_track_3d_Ct.npy"),
                "upper_left_track_3d_C0": str(part_output / "upper_left_track_3d_C0.npy"),
                "upper_left_track_used_depth_m": str(part_output / "upper_left_track_used_depth_m.npy"),
                "upper_left_track_depth_source": str(part_output / "upper_left_track_depth_source.npy"),
                "constrained_delta_poses": str(part_output / "Delta_C0_part_motion_axis_constrained.npy"),
                "part_poses_C0": str(part_output / "T_C0_from_part.npy"),
                "dynamic_mesh_root": str(part_output / "dynamic_meshes") if mesh_paths else None,
            },
            "frames": frame_records,
        }
        write_json(part_output / "articulate_part_tracking_manifest.json", manifest)
        part_manifests.append(manifest)

    statuses = [item["status"] for item in part_manifests]
    aggregate_status = "completed" if all(value == "completed" for value in statuses) else "needs_revision"
    aggregate = {
        **{key: value for key, value in preflight.items() if key != "status"},
        "status": aggregate_status,
        "motion_model": "per-part fixed revolute axis after C0-compensated RGB-D CoTracker",
        "part_manifests": [
            str(output_root / item["part_id"] / "articulate_part_tracking_manifest.json")
            for item in part_manifests
        ],
    }
    write_json(output_root / "articulate_tracking_manifest.json", aggregate)
    if not args.skip_stage_state_update:
        update_pipeline_state(
            workspace,
            output_root,
            aggregate_status,
            f"Tracked {len(part_manifests)} articulated part(s) in {selected_eye}-eye C0; status={aggregate_status}.",
        )
    print(json.dumps({"status": aggregate_status, "output": str(output_root)}, indent=2))
    return 0 if aggregate_status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
