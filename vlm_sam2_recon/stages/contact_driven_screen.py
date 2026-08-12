#!/usr/bin/env python3
"""Contact-driven laptop screen motion with constrained EgoForce hand correction.

This stage keeps the screen articulation as the trusted structure.  All dynamic
geometry is expressed in the frame-0 right-camera coordinate system, using the
exported headset/camera poses to cancel head motion.  Before contact the screen
is static.  Once a fingertip reaches the screen, the screen contact point is
fixed in the reference screen mesh and every later frame optimizes only a hinge
angle plus a lightweight hand correction.  The optional ``global_rigid`` mode
adds a small global hand rotation increment, which is a lightweight proxy for
refining EgoForce/MANO global orientation and translation before wiring in a
full MANO layer.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from vlm_sam2_recon.stages.camera_alignment import (
    apply_se3_to_mesh,
    camera_to_camera_matrix,
    frame_name,
    frame_row,
    load_frames_csv,
    rotate_mesh_about_axis,
    rotate_points_about_axis,
    transform_points,
    write_json,
)
from vlm_sam2_recon.stages.screen_hinge_tracking import load_mesh, signed_axis_angle
from vlm_sam2_recon.stages.vlm_contact_semantics import normalize_finger_name, normalize_fingers


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ALIGNMENT_DIR = PROJECT_ROOT / "outputs/object_alignment_screen_first_base_visible_snap/target_laptop/frame_000000"
DEFAULT_EXPORT_ROOT = Path("/code/3DVideo_2026-07-01-19-24-06-667_spatialmp4_export")
DEFAULT_RGB_DIR = PROJECT_ROOT / "outputs/tracker_rgb_right_15fps"
DEFAULT_HAND_DIR = PROJECT_ROOT / "outputs/egoforce_rgb_right_15fps"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/contact_driven_laptop/target_laptop_frames_000000_000057"
BASE_PART_LABEL = "14"
SCREEN_PART_LABEL = "15"
FINGERTIP_JOINT_IDS = (4, 8, 12, 16, 20)
FINGERTIP_NAMES = ("thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip")


@dataclass
class ContactDrivenScreenConfig:
    alignment_dir: Path = DEFAULT_ALIGNMENT_DIR
    export_root: Path = DEFAULT_EXPORT_ROOT
    rgb_dir: Path = DEFAULT_RGB_DIR
    hand_dir: Path = DEFAULT_HAND_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    start_frame: int = 0
    end_frame: int = 57
    fps: float = 15.0
    pose_fps: float = 5.0
    pose_csv: Path | None = None
    hand_side: str = "left"
    contact_search_start: int = 0
    contact_search_end: int | None = None
    contact_trend_eps_m: float = 0.002
    contact_force_frame: int | None = None
    vlm_contact_json: Path | None = None
    vlm_contact_target_id: str = "target_laptop"
    vlm_contact_mode: str = "force"  # force, window
    vlm_contact_window_before: int = 0
    vlm_contact_window_after: int = 24
    contact_fingers: tuple[str, ...] | None = None
    contact_distance_threshold_m: float | None = None
    contact_distance_consecutive_frames: int = 3
    contact_distance_min_hits: int = 2
    surface_sample_count: int = 9000
    screen_angle_min_deg: float = -120.0
    screen_angle_max_deg: float = 140.0
    max_hand_translation_m: float = 0.18
    max_solver_nfev: int = 120
    hand_outlier_contact_m: float = 0.16
    hand_outlier_radius_m: float = 0.09
    hand_outlier_axis_m: float = 0.06
    contact_scale_m: float = 0.015
    radius_scale_m: float = 0.025
    axis_scale_m: float = 0.025
    hand_prior_scale_m: float = 0.075
    hand_smooth_scale_m: float = 0.035
    theta_smooth_scale_deg: float = 16.0
    theta_acc_scale_deg: float = 24.0
    penetration_scale_m: float = 0.015
    penetration_margin_m: float = 0.002
    hand_refine_mode: str = "translation"  # translation, global_rigid
    max_hand_rotation_deg: float = 28.0
    hand_rot_prior_scale_deg: float = 18.0
    hand_rot_smooth_scale_deg: float = 12.0
    weight_contact: float = 8.0
    weight_radius: float = 3.5
    weight_axis: float = 3.5
    weight_hand_prior: float = 1.0
    weight_hand_smooth: float = 1.5
    weight_hand_rot_prior: float = 0.4
    weight_hand_rot_smooth: float = 0.8
    weight_theta_smooth: float = 0.6
    weight_theta_acc: float = 0.35
    weight_penetration: float = 1.0
    monotonic_slack_deg: float = 8.0
    enforce_monotonic_after_contact: bool = False
    corrected_hand_suffix: str = "contact_corrected"


@dataclass
class PoseTimeline:
    path: Path
    rows_by_index: dict[int, dict[str, str]]

    def require_row(self, frame_index: int) -> dict[str, str]:
        if frame_index not in self.rows_by_index:
            first = min(self.rows_by_index) if self.rows_by_index else None
            last = max(self.rows_by_index) if self.rows_by_index else None
            raise KeyError(
                f"Pose CSV {self.path} has no row for frame {frame_index}. "
                f"Available index range is {first}..{last}."
            )
        return self.rows_by_index[frame_index]

    @property
    def max_index(self) -> int:
        return max(self.rows_by_index) if self.rows_by_index else 0


@dataclass
class HandFrame:
    frame: int
    hand_mesh: trimesh.Trimesh | None
    arm_mesh: trimesh.Trimesh | None
    hand_joints: np.ndarray | None
    visible: bool
    hand_path: Path | None
    arm_path: Path | None
    npz_path: Path | None


@dataclass
class ContactObservation:
    frame: int
    local_index: int
    distance_m: float
    fingertip_index: int
    fingertip_name: str
    fingertip_point: np.ndarray
    screen_point_align: np.ndarray
    screen_point_frame: np.ndarray
    pose_frame: int


@dataclass
class OptimizedFrame:
    frame: int
    local_index: int
    pose_frame: int
    theta_rad: float
    hand_delta: np.ndarray
    status: str
    distance_before_m: float
    contact_error_m: float
    radius_error_m: float
    axis_error_m: float
    penetration_m: float
    loss: float
    fingertip_point_raw: np.ndarray | None
    fingertip_point_corrected: np.ndarray | None
    screen_contact_point: np.ndarray | None
    hand_rotvec: np.ndarray | None = None
    hand_rotation_center: np.ndarray | None = None


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def side_index(side: str) -> int:
    if side == "left":
        return 0
    if side == "right":
        return 1
    raise ValueError(f"Unsupported hand side: {side}")


def hand_stem(frame: int) -> str:
    return frame_name(frame)


def hand_paths(hand_dir: Path, frame: int, side: str) -> tuple[Path, Path, Path]:
    stem = hand_stem(frame)
    return (
        hand_dir / f"{stem}_{side}_hand.obj",
        hand_dir / f"{stem}_{side}_arm.obj",
        hand_dir / f"{stem}_egoforce_meshes.npz",
    )


def load_hand_frame(hand_dir: Path, frame: int, side: str) -> HandFrame:
    hand_path, arm_path, npz_path = hand_paths(hand_dir, frame, side)
    hand_mesh = load_mesh(hand_path) if hand_path.exists() else None
    arm_mesh = load_mesh(arm_path) if arm_path.exists() else None
    joints = None
    visible = hand_mesh is not None
    if npz_path.exists():
        with np.load(npz_path, allow_pickle=True) as data:
            idx = side_index(side)
            if "visible_hand" in data:
                visible = bool(data["visible_hand"][idx])
            if "hand_joints" in data and data["hand_joints"].shape[0] > idx:
                joints = np.asarray(data["hand_joints"][idx], dtype=np.float64)
    if hand_mesh is None:
        visible = False
    return HandFrame(
        frame=frame,
        hand_mesh=hand_mesh,
        arm_mesh=arm_mesh,
        hand_joints=joints,
        visible=visible,
        hand_path=hand_path if hand_path.exists() else None,
        arm_path=arm_path if arm_path.exists() else None,
        npz_path=npz_path if npz_path.exists() else None,
    )


def fingertip_points(hand: HandFrame) -> tuple[np.ndarray, list[int], list[str]]:
    if hand.hand_joints is not None and hand.hand_joints.shape[0] > max(FINGERTIP_JOINT_IDS):
        points = hand.hand_joints[np.asarray(FINGERTIP_JOINT_IDS, dtype=np.int64)]
        return points.astype(np.float64), list(FINGERTIP_JOINT_IDS), list(FINGERTIP_NAMES)
    if hand.hand_mesh is not None and len(hand.hand_mesh.vertices):
        vertices = np.asarray(hand.hand_mesh.vertices, dtype=np.float64)
        # Fallback for mesh-only outputs: use a few extremal points.  These are
        # weaker than MANO joints, so diagnostics mark only the pseudo index.
        order = np.argsort(vertices[:, 1])[:5]
        return vertices[order], [int(v) for v in order], [f"vertex_{int(v)}" for v in order]
    return np.zeros((0, 3), dtype=np.float64), [], []


def select_contact_fingertips(
    tips: np.ndarray,
    ids: list[int],
    names: list[str],
    contact_fingers: tuple[str, ...] | None,
) -> tuple[np.ndarray, list[int], list[str]]:
    """Restrict MANO tips to VLM-supported fingers while preserving mesh fallback."""
    allowed = set(normalize_fingers(contact_fingers))
    if not allowed or not names:
        return tips, ids, names
    keep = [idx for idx, name in enumerate(names) if normalize_finger_name(name) in allowed]
    if keep:
        indices = np.asarray(keep, dtype=np.int64)
        return tips[indices], [ids[idx] for idx in keep], [names[idx] for idx in keep]
    # Mesh-only pseudo tips do not carry anatomical names, so semantic filtering
    # cannot be applied safely. Keep them as the legacy geometric fallback.
    if all(name.startswith("vertex_") for name in names):
        return tips, ids, names
    return tips[:0], [], []


def mesh_surface_points(mesh: trimesh.Trimesh, max_count: int) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    centers = np.asarray(mesh.triangles_center, dtype=np.float64)
    points = np.vstack([vertices, centers]) if len(centers) else vertices.copy()
    if len(points) > max_count:
        rng = np.random.default_rng(20260706)
        points = points[rng.choice(len(points), size=max_count, replace=False)]
    return points.astype(np.float64)


def pose_frame_for_tracker_frame(frame: int, fps: float, pose_fps: float, pose_frame_max: int) -> int:
    if fps <= 0.0 or pose_fps <= 0.0:
        return min(frame, pose_frame_max)
    mapped = int(round(float(frame) * float(pose_fps) / float(fps)))
    return int(np.clip(mapped, 0, pose_frame_max))


def camera_transform_for_pose(meta: dict[str, Any], export_root: Path, pose_frame: int) -> np.ndarray:
    align_row = frame_row(export_root, 0)
    view_row = frame_row(export_root, pose_frame)
    return camera_to_camera_matrix(meta, align_row, view_row, camera="right")


def load_pose_timeline(pose_csv: Path | None) -> PoseTimeline | None:
    if pose_csv is None:
        return None
    pose_csv = pose_csv.resolve()
    if not pose_csv.exists():
        raise FileNotFoundError(f"Pose CSV not found: {pose_csv}")
    with pose_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    required = {"x", "y", "z", "qw", "qx", "qy", "qz"}
    if not rows:
        raise ValueError(f"Pose CSV is empty: {pose_csv}")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Pose CSV {pose_csv} is missing required columns: {sorted(missing)}")
    rows_by_index: dict[int, dict[str, str]] = {}
    for seq_idx, row in enumerate(rows):
        raw_index = row.get("index", "")
        frame_index = int(raw_index) if str(raw_index).strip() else seq_idx
        rows_by_index[frame_index] = row
    return PoseTimeline(path=pose_csv, rows_by_index=rows_by_index)


def pose_csv_row_to_export_frame_row(row: dict[str, str]) -> dict[str, str]:
    # camera_alignment expects the export frames.csv naming convention.  The
    # 15fps pose file stores the same world-head pose as x/y/z + qw/qx/qy/qz.
    return {
        "rgb_pose_x": row["x"],
        "rgb_pose_y": row["y"],
        "rgb_pose_z": row["z"],
        "rgb_pose_qw": row["qw"],
        "rgb_pose_qx": row["qx"],
        "rgb_pose_qy": row["qy"],
        "rgb_pose_qz": row["qz"],
        "rgb_pose_timestamp_s": row.get("timestamp_s", ""),
    }


def resolved_pose_frame_for_tracker_frame(
    frame: int,
    config: ContactDrivenScreenConfig,
    pose_frame_max: int,
    pose_timeline: PoseTimeline | None,
) -> int:
    if pose_timeline is not None:
        pose_timeline.require_row(frame)
        return int(frame)
    return pose_frame_for_tracker_frame(frame, config.fps, config.pose_fps, pose_frame_max)


def camera_transform_for_resolved_pose(
    meta: dict[str, Any],
    export_root: Path,
    pose_frame: int,
    pose_timeline: PoseTimeline | None,
) -> np.ndarray:
    if pose_timeline is None:
        return camera_transform_for_pose(meta, export_root, pose_frame)
    # The laptop mesh is expressed in the camera coordinates of export frame 0,
    # whose timestamp is generally not the same as tracker/pose frame 0.
    align_row = frame_row(export_root, 0)
    view_row = pose_csv_row_to_export_frame_row(pose_timeline.require_row(pose_frame))
    return camera_to_camera_matrix(meta, align_row, view_row, camera="right")


def standard_depth_assignments(
    config: ContactDrivenScreenConfig,
    meta: dict[str, Any],
    frames: list[int],
    pose_timeline: PoseTimeline | None,
) -> dict[int, dict[str, Any]]:
    """Associate sparse standard-depth frames with tracker frames and exact poses."""
    if pose_timeline is None:
        return {}
    frame_timestamps = {
        frame: float(pose_timeline.require_row(frame)["timestamp_s"])
        for frame in frames
    }
    max_delta_s = 0.5 / max(float(config.fps), 1e-6) + 1e-6
    alignment_row = frame_row(config.export_root, 0)
    assignments: dict[int, dict[str, Any]] = {}
    for row in load_frames_csv(config.export_root / "frames.csv"):
        depth_timestamp = float(row["depth_timestamp_s"])
        nearest_frame = min(frames, key=lambda frame: abs(frame_timestamps[frame] - depth_timestamp))
        delta_s = depth_timestamp - frame_timestamps[nearest_frame]
        if abs(delta_s) > max_delta_s:
            continue
        candidate = {
            "frame": int(row["index"]),
            "timestamp_s": depth_timestamp,
            "timestamp_delta_s": delta_s,
            "camera_to_alignment_matrix": np.linalg.inv(
                camera_to_camera_matrix(
                    meta,
                    alignment_row,
                    row,
                    camera="right",
                    view_pose_prefix="depth_pose",
                )
            ).tolist(),
        }
        previous = assignments.get(nearest_frame)
        if previous is None or abs(delta_s) < abs(float(previous["timestamp_delta_s"])):
            assignments[nearest_frame] = candidate
    return assignments


def max_export_frame_index(export_root: Path) -> int:
    rows = load_frames_csv(export_root / "frames.csv")
    if not rows:
        return 0
    return max(int(row["index"]) for row in rows)


def axis_decompose(point: np.ndarray, origin: np.ndarray, axis: np.ndarray) -> tuple[float, float]:
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    vec = np.asarray(point, dtype=np.float64) - origin
    s = float(vec @ axis)
    radial = vec - s * axis
    return s, float(np.linalg.norm(radial))


def median_filter_1d(values: np.ndarray, radius: int = 1) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    out = values.copy()
    for idx in range(len(values)):
        lo = max(0, idx - radius)
        hi = min(len(values), idx + radius + 1)
        window = values[lo:hi]
        finite = window[np.isfinite(window)]
        if finite.size:
            out[idx] = float(np.median(finite))
    return out


def find_contact_index(observations: list[ContactObservation], config: ContactDrivenScreenConfig) -> int:
    if config.contact_force_frame is not None:
        for obs in observations:
            if obs.frame == config.contact_force_frame:
                return obs.local_index
        raise ValueError(f"Forced contact frame {config.contact_force_frame} is outside the requested sequence")
    distances = np.asarray([obs.distance_m for obs in observations], dtype=np.float64)
    smooth = median_filter_1d(distances, radius=1)
    end = len(smooth) - 1 if config.contact_search_end is None else min(config.contact_search_end, len(smooth) - 1)
    start = max(0, int(config.contact_search_start))
    finite = np.isfinite(smooth)
    if config.contact_distance_threshold_m is not None:
        threshold = float(config.contact_distance_threshold_m)
        consecutive = max(1, int(config.contact_distance_consecutive_frames))
        min_hits = max(1, min(consecutive, int(config.contact_distance_min_hits)))
        for idx in range(start, end + 1):
            hi = min(end + 1, idx + consecutive)
            window = smooth[idx:hi]
            finite_window = np.isfinite(window)
            if int(finite_window.sum()) < min_hits:
                continue
            hits = np.flatnonzero(finite_window & (window <= threshold))
            if len(hits) >= min_hits:
                return int(idx + hits[0])

    for idx in range(max(start + 2, 2), max(start + 2, end - 1)):
        if not finite[idx - 2 : idx + 3].all():
            continue
        decreasing = smooth[idx - 2] >= smooth[idx - 1] - config.contact_trend_eps_m and smooth[idx - 1] >= smooth[idx] - config.contact_trend_eps_m
        increasing = smooth[idx + 1] >= smooth[idx] + config.contact_trend_eps_m and smooth[idx + 2] >= smooth[idx + 1] - config.contact_trend_eps_m
        if decreasing and increasing:
            lo = max(start, idx - 2)
            hi = min(end + 1, idx + 3)
            return int(lo + np.nanargmin(smooth[lo:hi]))
    search = smooth[start : end + 1]
    if not np.isfinite(search).any():
        raise RuntimeError("Could not find any finite hand-screen distance for contact detection")
    return int(start + np.nanargmin(search))


def _target_matches(candidate: dict[str, Any], target_id: str) -> bool:
    target_keys = (
        "target_object_id",
        "target_id",
        "object_id",
        "object",
        "object_name",
        "name_en",
    )
    values = [candidate.get(key) for key in target_keys if candidate.get(key) not in (None, "")]
    if not values:
        return True
    target_norm = str(target_id).lower()
    for value in values:
        text = str(value).lower()
        if text == target_norm or target_norm in text:
            return True
        if target_norm.endswith("laptop") and "laptop" in text:
            return True
    return False


def _looks_like_contact_event(candidate: dict[str, Any]) -> bool:
    event_keys = ("event_type", "relation", "action", "type", "name", "description")
    values = [str(candidate.get(key, "")).lower() for key in event_keys]
    joined = " ".join(values)
    if not joined.strip():
        return True
    return any(token in joined for token in ("contact", "touch", "press", "push", "grab", "close", "first"))


def _frame_value_to_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and np.isfinite(value):
        return int(round(float(value)))
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        stem = Path(text).stem
        for item in (text, stem):
            if item.lstrip("+-").isdigit():
                return int(item)
    if isinstance(value, list):
        frames = [_frame_value_to_int(item) for item in value]
        frames = [item for item in frames if item is not None]
        return min(frames) if frames else None
    if isinstance(value, dict):
        for key in (
            "frame_index",
            "frame",
            "first_contact_frame_index",
            "first_contact_frame",
            "contact_frame_index",
            "contact_frame",
            "start_frame_index",
            "start_frame",
            "frame_file",
            "frame_path",
        ):
            frame = _frame_value_to_int(value.get(key))
            if frame is not None:
                return frame
    return None


def _extract_candidate_frame(candidate: dict[str, Any]) -> tuple[int | None, str | None]:
    for key in (
        "first_contact_frame",
        "first_contact",
        "first_touch_frame",
        "first_touch",
        "contact_start",
        "contact",
        "touch",
        "frame_index",
        "frame",
        "start_frame",
        "evidence_frames",
    ):
        if key not in candidate:
            continue
        frame = _frame_value_to_int(candidate.get(key))
        if frame is not None:
            return frame, key
    return None, None


def _extract_candidate_contact_semantics(candidate: dict[str, Any]) -> dict[str, Any]:
    sources = [candidate]
    for key in ("first_contact_frame", "first_contact", "contact"):
        value = candidate.get(key)
        if isinstance(value, dict):
            sources.insert(0, value)
    fingers: list[str] = []
    primary: str | None = None
    hand_side: str | None = None
    contacted_part: str | None = None
    for source in sources:
        if not fingers:
            fingers = normalize_fingers(source.get("contact_fingers", source.get("fingers")))
        if primary is None:
            primary = normalize_finger_name(source.get("primary_contact_finger"))
        if hand_side is None and source.get("hand_side") in ("left", "right", "both"):
            hand_side = str(source["hand_side"])
        if contacted_part is None and source.get("contacted_part") not in (None, ""):
            contacted_part = str(source["contacted_part"])
    if primary is not None and primary not in fingers:
        fingers = normalize_fingers([primary, *fingers])
    return {
        "contact_fingers": fingers,
        "primary_contact_finger": primary,
        "hand_side": hand_side,
        "contacted_part": contacted_part,
    }


def resolve_vlm_contact_frame(vlm_json_path: Path, target_id: str, frames: list[int]) -> tuple[int, dict[str, Any]]:
    data = read_json(vlm_json_path)
    result = data.get("vlm_result", data) if isinstance(data, dict) else {}
    if not isinstance(result, dict):
        raise ValueError(f"VLM contact JSON has no object result: {vlm_json_path}")

    candidates: list[tuple[str, dict[str, Any]]] = []
    for target in result.get("target_objects", []) or []:
        if not isinstance(target, dict) or not _target_matches(target, target_id):
            continue
        for key in (
            "hand_object_interaction",
            "contact_analysis",
            "first_contact",
            "first_contact_frame",
            "interaction",
        ):
            value = target.get(key)
            if isinstance(value, dict):
                merged = dict(value)
                merged.setdefault("target_object_id", target.get("object_id", target_id))
                candidates.append((f"target_objects.{target.get('object_id', target_id)}.{key}", merged))

    for key in (
        "hand_object_events",
        "hand_object_interaction_events",
        "interaction_events",
        "temporal_events",
        "global_relations",
        "first_contact_events",
    ):
        value = result.get(key)
        if isinstance(value, list):
            for idx, item in enumerate(value):
                if isinstance(item, dict):
                    candidates.append((f"{key}[{idx}]", item))
        elif isinstance(value, dict):
            candidates.append((key, value))

    for key in ("hand_object_interaction", "contact_analysis", "first_contact", "reconstruction_plan"):
        value = result.get(key)
        if isinstance(value, dict):
            candidates.append((key, value))

    for source, candidate in candidates:
        if not _target_matches(candidate, target_id):
            continue
        if not _looks_like_contact_event(candidate):
            continue
        frame, frame_key = _extract_candidate_frame(candidate)
        if frame is None:
            continue
        if frame not in frames:
            raise ValueError(
                f"VLM contact frame {frame} from {vlm_json_path} is outside requested frame range "
                f"{frames[0]}..{frames[-1]}"
            )
        return frame, {
            "type": "vlm_first_contact",
            "vlm_json": str(vlm_json_path),
            "target_id": target_id,
            "source": source,
            "frame_key": frame_key,
            "raw_candidate": candidate,
            **_extract_candidate_contact_semantics(candidate),
        }

    raise ValueError(
        f"Could not find a first-contact frame for {target_id!r} in VLM JSON: {vlm_json_path}"
    )


def build_contact_observations(
    config: ContactDrivenScreenConfig,
    meta: dict[str, Any],
    frames: list[int],
    hand_frames: list[HandFrame],
    screen_surface_align: np.ndarray,
    pose_frame_max: int,
    pose_timeline: PoseTimeline | None,
) -> list[ContactObservation]:
    observations: list[ContactObservation] = []
    for local_idx, frame in enumerate(frames):
        pose_frame = resolved_pose_frame_for_tracker_frame(frame, config, pose_frame_max, pose_timeline)
        t_frame_from_align = camera_transform_for_resolved_pose(meta, config.export_root, pose_frame, pose_timeline)
        t_align_from_frame = np.linalg.inv(t_frame_from_align)
        tree = cKDTree(screen_surface_align)
        hand = hand_frames[local_idx]
        tips, ids, names = fingertip_points(hand)
        tips, ids, names = select_contact_fingertips(tips, ids, names, config.contact_fingers)
        if len(tips) == 0:
            observations.append(
                ContactObservation(
                    frame=frame,
                    local_index=local_idx,
                    distance_m=float("inf"),
                    fingertip_index=-1,
                    fingertip_name="missing",
                    fingertip_point=np.full(3, np.nan),
                    screen_point_align=np.full(3, np.nan),
                    screen_point_frame=np.full(3, np.nan),
                    pose_frame=pose_frame,
                )
            )
            continue
        tips_align = transform_points(tips, t_align_from_frame)
        distances, nearest = tree.query(tips_align, k=1)
        best_tip = int(np.nanargmin(distances))
        best_surface = int(nearest[best_tip])
        observations.append(
            ContactObservation(
                frame=frame,
                local_index=local_idx,
                distance_m=float(distances[best_tip]),
                fingertip_index=int(ids[best_tip]),
                fingertip_name=str(names[best_tip]),
                fingertip_point=tips_align[best_tip].astype(np.float64),
                screen_point_align=screen_surface_align[best_surface].astype(np.float64),
                screen_point_frame=screen_surface_align[best_surface].astype(np.float64),
                pose_frame=pose_frame,
            )
        )
    return observations


def choose_screen_normal_toward_hand(
    screen_mesh0: trimesh.Trimesh,
    contact_align: np.ndarray,
    fingertip_align: np.ndarray,
) -> np.ndarray:
    vertices = np.asarray(screen_mesh0.vertices, dtype=np.float64)
    center = vertices.mean(axis=0)
    _, _, vt = np.linalg.svd(vertices - center, full_matrices=False)
    normal = vt[-1]
    normal /= np.linalg.norm(normal) + 1e-12
    if float((fingertip_align - contact_align) @ normal) < 0.0:
        normal = -normal
    return normal


def signed_penetration_depth(
    hand_vertices: np.ndarray,
    screen_contact_frame: np.ndarray,
    screen_normal_frame: np.ndarray,
    margin_m: float,
) -> np.ndarray:
    if len(hand_vertices) == 0:
        return np.zeros(0, dtype=np.float64)
    signed = (hand_vertices - screen_contact_frame[None, :]) @ screen_normal_frame
    # Penalize only points that moved behind the contact-side plane.  This is a
    # conservative screen-plane proxy, not a full watertight SDF.
    return np.maximum(0.0, margin_m - signed)


def hand_rotation_center(hand: HandFrame) -> np.ndarray:
    if hand.hand_joints is not None and np.isfinite(hand.hand_joints).all() and len(hand.hand_joints):
        return np.asarray(hand.hand_joints[0], dtype=np.float64)
    if hand.hand_mesh is not None and len(hand.hand_mesh.vertices):
        return np.asarray(hand.hand_mesh.vertices, dtype=np.float64).mean(axis=0)
    return np.zeros(3, dtype=np.float64)


def apply_hand_rigid(points: np.ndarray, center: np.ndarray, rotvec: np.ndarray, delta: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    rotvec = np.asarray(rotvec, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    if points.size == 0:
        return points.reshape(-1, 3)
    if float(np.linalg.norm(rotvec)) < 1e-12:
        rotated = points.copy()
    else:
        rotated = Rotation.from_rotvec(rotvec).apply(points - center[None, :]) + center[None, :]
    return rotated + delta[None, :]


def transform_hand_point(point: np.ndarray, center: np.ndarray, rotvec: np.ndarray, delta: np.ndarray) -> np.ndarray:
    return apply_hand_rigid(np.asarray(point, dtype=np.float64).reshape(1, 3), center, rotvec, delta)[0]


def optimize_after_contact(
    config: ContactDrivenScreenConfig,
    meta: dict[str, Any],
    frames: list[int],
    hand_frames: list[HandFrame],
    contact_obs: ContactObservation,
    contact_index: int,
    joint: dict[str, Any],
    screen_mesh0: trimesh.Trimesh,
    screen_normal_align: np.ndarray,
    pose_frame_max: int,
    pose_timeline: PoseTimeline | None,
) -> list[OptimizedFrame]:
    origin = np.asarray(joint["origin_xyz"], dtype=np.float64)
    axis = np.asarray(joint["axis_xyz"], dtype=np.float64)
    axis /= np.linalg.norm(axis) + 1e-12
    contact_align = np.asarray(contact_obs.screen_point_align, dtype=np.float64)
    s0, r0 = axis_decompose(contact_align, origin, axis)
    theta_min = math.radians(config.screen_angle_min_deg)
    theta_max = math.radians(config.screen_angle_max_deg)
    theta_smooth_scale = math.radians(max(config.theta_smooth_scale_deg, 1e-3))
    theta_acc_scale = math.radians(max(config.theta_acc_scale_deg, 1e-3))
    hand_rot_prior_scale = math.radians(max(config.hand_rot_prior_scale_deg, 1e-3))
    hand_rot_smooth_scale = math.radians(max(config.hand_rot_smooth_scale_deg, 1e-3))
    max_hand_rotation = math.radians(max(config.max_hand_rotation_deg, 0.0))
    monotonic_slack = math.radians(max(0.0, config.monotonic_slack_deg))
    refine_mode = config.hand_refine_mode.lower()
    if refine_mode not in {"translation", "global_rigid"}:
        raise ValueError(f"Unsupported hand_refine_mode={config.hand_refine_mode!r}; expected translation or global_rigid")
    use_global_rigid = refine_mode == "global_rigid"
    entries: list[OptimizedFrame] = []
    prev_theta = 0.0
    prev_prev_theta = 0.0
    prev_delta = np.zeros(3, dtype=np.float64)
    prev_rotvec = np.zeros(3, dtype=np.float64)

    for local_idx, frame in enumerate(frames):
        pose_frame = resolved_pose_frame_for_tracker_frame(frame, config, pose_frame_max, pose_timeline)
        t_frame_from_align = camera_transform_for_resolved_pose(meta, config.export_root, pose_frame, pose_timeline)
        t_align_from_frame = np.linalg.inv(t_frame_from_align)
        hand = hand_frames[local_idx]
        obs_distance = float("inf")
        contact_error = radius_error = axis_error = penetration = loss = 0.0
        raw_tip = corrected_tip = screen_contact_frame = None
        status = "free_before_contact"
        theta = 0.0
        delta = np.zeros(3, dtype=np.float64)
        rotvec = np.zeros(3, dtype=np.float64)
        hand_center = transform_points(hand_rotation_center(hand)[None, :], t_align_from_frame)[0]

        if local_idx < contact_index:
            entries.append(
                OptimizedFrame(frame, local_idx, pose_frame, theta, delta, status, obs_distance, contact_error, radius_error, axis_error, penetration, loss, raw_tip, corrected_tip, screen_contact_frame, rotvec, hand_center)
            )
            continue

        tips, ids, _names = fingertip_points(hand)
        if len(tips) == 0 or contact_obs.fingertip_index not in ids:
            theta = prev_theta
            delta = prev_delta.copy()
            rotvec = prev_rotvec.copy()
            status = "hold_missing_hand"
            screen_contact_align = rotate_points_about_axis(contact_align[None, :], origin, axis, theta)[0]
            screen_contact_frame = screen_contact_align
            entries.append(
                OptimizedFrame(frame, local_idx, pose_frame, theta, delta, status, obs_distance, contact_error, radius_error, axis_error, penetration, loss, raw_tip, corrected_tip, screen_contact_frame, rotvec, hand_center)
            )
            prev_prev_theta, prev_theta, prev_delta, prev_rotvec = prev_theta, theta, delta, rotvec
            continue

        tip_idx = ids.index(contact_obs.fingertip_index)
        raw_tip = transform_points(tips[tip_idx][None, :], t_align_from_frame)[0]
        raw_tip_align = raw_tip
        prev_screen_contact_align = rotate_points_about_axis(contact_align[None, :], origin, axis, prev_theta)[0]
        prev_screen_contact_frame = prev_screen_contact_align
        raw_s, raw_r = axis_decompose(raw_tip_align, origin, axis)
        raw_contact_dist = float(np.linalg.norm(raw_tip - prev_screen_contact_frame))
        raw_radius_error = float(raw_r - r0)
        raw_axis_error = float(raw_s - s0)
        if (
            local_idx > contact_index
            and raw_contact_dist > config.hand_outlier_contact_m
            and (abs(raw_radius_error) > config.hand_outlier_radius_m or abs(raw_axis_error) > config.hand_outlier_axis_m)
        ):
            theta = prev_theta
            rotvec = prev_rotvec.copy() if use_global_rigid else np.zeros(3, dtype=np.float64)
            screen_contact_align = prev_screen_contact_align
            screen_contact_frame = screen_contact_align
            raw_tip_rigid = transform_hand_point(raw_tip, hand_center, rotvec, np.zeros(3, dtype=np.float64))
            delta = np.asarray(screen_contact_frame - raw_tip_rigid, dtype=np.float64)
            delta_norm = float(np.linalg.norm(delta))
            if delta_norm > config.max_hand_translation_m:
                delta *= config.max_hand_translation_m / max(delta_norm, 1e-12)
            corrected_tip = transform_hand_point(raw_tip, hand_center, rotvec, delta)
            s, r = axis_decompose(corrected_tip, origin, axis)
            contact_error = float(np.linalg.norm(corrected_tip - screen_contact_frame))
            radius_error = float(r - r0)
            axis_error = float(s - s0)
            normal_align = rotate_points_about_axis((contact_align + screen_normal_align)[None, :], origin, axis, theta)[0] - screen_contact_align
            normal_align /= np.linalg.norm(normal_align) + 1e-12
            hand_vertices = np.asarray(hand.hand_mesh.vertices, dtype=np.float64) if hand.hand_mesh is not None else np.zeros((0, 3), dtype=np.float64)
            if len(hand_vertices):
                hand_vertices = transform_points(hand_vertices, t_align_from_frame)
            if len(hand_vertices) > 1200:
                hand_vertices_for_pen = hand_vertices[:: max(1, len(hand_vertices) // 800)]
            else:
                hand_vertices_for_pen = hand_vertices
            pen_vertices = apply_hand_rigid(hand_vertices_for_pen, hand_center, rotvec, delta)
            pen_values = signed_penetration_depth(pen_vertices, screen_contact_align, normal_align, config.penetration_margin_m)
            penetration = float(np.max(pen_values)) if pen_values.size else 0.0
            loss = float((contact_error / config.contact_scale_m) ** 2)
            obs_distance = raw_contact_dist
            status = "hand_outlier_hold_screen"
            entries.append(
                OptimizedFrame(frame, local_idx, pose_frame, theta, delta, status, obs_distance, contact_error, radius_error, axis_error, penetration, loss, raw_tip, corrected_tip, screen_contact_frame, rotvec, hand_center)
            )
            prev_prev_theta, prev_theta, prev_delta, prev_rotvec = prev_theta, theta, delta, rotvec
            continue

        theta_obs = signed_axis_angle(contact_align, raw_tip_align, origin, axis)
        if not np.isfinite(theta_obs):
            theta_obs = prev_theta
        theta_init = float(np.clip(0.65 * theta_obs + 0.35 * prev_theta, theta_min, theta_max))
        if local_idx == contact_index:
            theta_init = 0.0
        if use_global_rigid:
            x0 = np.concatenate([[theta_init], prev_rotvec, prev_delta]).astype(np.float64)
            bounds_lo = np.asarray(
                [theta_min, -max_hand_rotation, -max_hand_rotation, -max_hand_rotation, -config.max_hand_translation_m, -config.max_hand_translation_m, -config.max_hand_translation_m],
                dtype=np.float64,
            )
            bounds_hi = np.asarray(
                [theta_max, max_hand_rotation, max_hand_rotation, max_hand_rotation, config.max_hand_translation_m, config.max_hand_translation_m, config.max_hand_translation_m],
                dtype=np.float64,
            )
        else:
            x0 = np.concatenate([[theta_init], prev_delta]).astype(np.float64)
            bounds_lo = np.asarray([theta_min, -config.max_hand_translation_m, -config.max_hand_translation_m, -config.max_hand_translation_m], dtype=np.float64)
            bounds_hi = np.asarray([theta_max, config.max_hand_translation_m, config.max_hand_translation_m, config.max_hand_translation_m], dtype=np.float64)
        hand_vertices = np.asarray(hand.hand_mesh.vertices, dtype=np.float64) if hand.hand_mesh is not None else np.zeros((0, 3), dtype=np.float64)
        if len(hand_vertices):
            hand_vertices = transform_points(hand_vertices, t_align_from_frame)
        if len(hand_vertices) > 1200:
            hand_vertices_for_pen = hand_vertices[:: max(1, len(hand_vertices) // 800)]
        else:
            hand_vertices_for_pen = hand_vertices

        def residuals(x: np.ndarray) -> np.ndarray:
            theta_x = float(x[0])
            if use_global_rigid:
                rotvec_x = np.asarray(x[1:4], dtype=np.float64)
                delta_x = np.asarray(x[4:7], dtype=np.float64)
            else:
                rotvec_x = np.zeros(3, dtype=np.float64)
                delta_x = np.asarray(x[1:4], dtype=np.float64)
            screen_contact_align = rotate_points_about_axis(contact_align[None, :], origin, axis, theta_x)[0]
            screen_contact = screen_contact_align
            finger = transform_hand_point(raw_tip, hand_center, rotvec_x, delta_x)
            finger_align = finger
            s, r = axis_decompose(finger_align, origin, axis)
            normal_align = rotate_points_about_axis((contact_align + screen_normal_align)[None, :], origin, axis, theta_x)[0] - screen_contact_align
            normal_align /= np.linalg.norm(normal_align) + 1e-12
            hand_vertices_corr = apply_hand_rigid(hand_vertices_for_pen, hand_center, rotvec_x, delta_x)
            penetration_values = signed_penetration_depth(hand_vertices_corr, screen_contact, normal_align, config.penetration_margin_m)
            if len(penetration_values) > 80:
                penetration_values = np.sort(penetration_values)[-80:]
            res = [
                *(math.sqrt(config.weight_contact) * (finger - screen_contact) / config.contact_scale_m),
                math.sqrt(config.weight_radius) * (r - r0) / config.radius_scale_m,
                math.sqrt(config.weight_axis) * (s - s0) / config.axis_scale_m,
                *(math.sqrt(config.weight_hand_prior) * delta_x / config.hand_prior_scale_m),
                *(math.sqrt(config.weight_hand_smooth) * (delta_x - prev_delta) / config.hand_smooth_scale_m),
                math.sqrt(config.weight_theta_smooth) * (theta_x - prev_theta) / theta_smooth_scale,
                math.sqrt(config.weight_theta_acc) * (theta_x - 2.0 * prev_theta + prev_prev_theta) / theta_acc_scale,
            ]
            if use_global_rigid:
                res.extend((math.sqrt(config.weight_hand_rot_prior) * rotvec_x / hand_rot_prior_scale).tolist())
                res.extend((math.sqrt(config.weight_hand_rot_smooth) * (rotvec_x - prev_rotvec) / hand_rot_smooth_scale).tolist())
            if penetration_values.size:
                # Keep penetration influence independent of mesh sampling density.
                penetration_weight = math.sqrt(
                    config.weight_penetration / max(int(penetration_values.size), 1)
                )
                res.extend((penetration_weight * penetration_values / config.penetration_scale_m).tolist())
            return np.asarray(res, dtype=np.float64)

        if local_idx == contact_index:
            contact_point_frame = contact_align
            delta_contact = np.asarray(contact_point_frame - raw_tip, dtype=np.float64)
            delta_norm = float(np.linalg.norm(delta_contact))
            if delta_norm > config.max_hand_translation_m:
                delta_contact *= config.max_hand_translation_m / max(delta_norm, 1e-12)
            if use_global_rigid:
                x0 = np.concatenate([[0.0], np.zeros(3, dtype=np.float64), delta_contact])
            else:
                x0 = np.concatenate([[0.0], delta_contact])
            # Establish the sticky contact exactly once. Optimizing this frame
            # allowed the many penetration samples to pull the fingertip away
            # from the very contact point that defines all later motion.
            theta = 0.0
            rotvec = np.zeros(3, dtype=np.float64)
            delta = delta_contact
            screen_contact_frame = contact_align.copy()
            corrected_tip = transform_hand_point(raw_tip, hand_center, rotvec, delta)
            s, r = axis_decompose(corrected_tip, origin, axis)
            contact_error = float(np.linalg.norm(corrected_tip - screen_contact_frame))
            radius_error = float(r - r0)
            axis_error = float(s - s0)
            normal_align = screen_normal_align / (np.linalg.norm(screen_normal_align) + 1e-12)
            pen_values = signed_penetration_depth(
                apply_hand_rigid(hand_vertices_for_pen, hand_center, rotvec, delta),
                screen_contact_frame,
                normal_align,
                config.penetration_margin_m,
            )
            penetration = float(np.max(pen_values)) if pen_values.size else 0.0
            obs_distance = float(np.linalg.norm(raw_tip - screen_contact_frame))
            loss = float((contact_error / config.contact_scale_m) ** 2)
            status = "contact_lock"
            entries.append(
                OptimizedFrame(
                    frame,
                    local_idx,
                    pose_frame,
                    theta,
                    delta,
                    status,
                    obs_distance,
                    contact_error,
                    radius_error,
                    axis_error,
                    penetration,
                    loss,
                    raw_tip,
                    corrected_tip,
                    screen_contact_frame,
                    rotvec,
                    hand_center,
                )
            )
            prev_prev_theta, prev_theta, prev_delta, prev_rotvec = prev_theta, theta, delta, rotvec
            continue
        try:
            result = least_squares(
                residuals,
                x0,
                bounds=(bounds_lo, bounds_hi),
                loss="huber",
                f_scale=1.0,
                max_nfev=int(config.max_solver_nfev),
            )
            theta = float(result.x[0])
            if use_global_rigid:
                rotvec = np.asarray(result.x[1:4], dtype=np.float64)
                delta = np.asarray(result.x[4:7], dtype=np.float64)
            else:
                rotvec = np.zeros(3, dtype=np.float64)
                delta = np.asarray(result.x[1:4], dtype=np.float64)
            loss = float(np.mean(result.fun * result.fun)) if result.fun.size else 0.0
            suffix = "_global_rigid" if use_global_rigid else ""
            status = f"contact_optimized{suffix}" if local_idx > contact_index else f"contact_lock{suffix}"
        except ValueError:
            theta = prev_theta
            delta = prev_delta.copy()
            rotvec = prev_rotvec.copy() if use_global_rigid else np.zeros(3, dtype=np.float64)
            status = "hold_solver_bounds"
            loss = float("inf")

        if config.enforce_monotonic_after_contact and local_idx > contact_index and theta < prev_theta - monotonic_slack:
            theta = prev_theta - monotonic_slack
            status = "contact_optimized_monotonic_clamped"

        screen_contact_align = rotate_points_about_axis(contact_align[None, :], origin, axis, theta)[0]
        screen_contact_frame = screen_contact_align
        corrected_tip = transform_hand_point(raw_tip, hand_center, rotvec, delta)
        corrected_tip_align = corrected_tip
        s, r = axis_decompose(corrected_tip_align, origin, axis)
        contact_error = float(np.linalg.norm(corrected_tip - screen_contact_frame))
        radius_error = float(r - r0)
        axis_error = float(s - s0)
        normal_align = rotate_points_about_axis((contact_align + screen_normal_align)[None, :], origin, axis, theta)[0] - screen_contact_align
        normal_align /= np.linalg.norm(normal_align) + 1e-12
        pen_values = signed_penetration_depth(
            apply_hand_rigid(hand_vertices_for_pen, hand_center, rotvec, delta),
            screen_contact_frame,
            normal_align,
            config.penetration_margin_m,
        )
        penetration = float(np.max(pen_values)) if pen_values.size else 0.0
        obs_distance = float(np.linalg.norm(raw_tip - screen_contact_frame))
        entries.append(
            OptimizedFrame(frame, local_idx, pose_frame, theta, delta, status, obs_distance, contact_error, radius_error, axis_error, penetration, loss, raw_tip, corrected_tip, screen_contact_frame, rotvec, hand_center)
        )
        prev_prev_theta, prev_theta, prev_delta, prev_rotvec = prev_theta, theta, delta, rotvec
    return entries


def translated_mesh(mesh: trimesh.Trimesh | None, delta: np.ndarray) -> trimesh.Trimesh | None:
    if mesh is None:
        return None
    out = mesh.copy()
    out.vertices = np.asarray(out.vertices, dtype=np.float64) + np.asarray(delta, dtype=np.float64)[None, :]
    return out


def refined_hand_mesh(
    mesh: trimesh.Trimesh | None,
    center: np.ndarray,
    rotvec: np.ndarray | None,
    delta: np.ndarray,
) -> trimesh.Trimesh | None:
    if mesh is None:
        return None
    out = mesh.copy()
    rot = np.zeros(3, dtype=np.float64) if rotvec is None else np.asarray(rotvec, dtype=np.float64)
    out.vertices = apply_hand_rigid(np.asarray(out.vertices, dtype=np.float64), center, rot, np.asarray(delta, dtype=np.float64))
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def export_contact_sequence(
    config: ContactDrivenScreenConfig,
    meta: dict[str, Any],
    frames: list[int],
    hand_frames: list[HandFrame],
    optimized: list[OptimizedFrame],
    contact_obs: ContactObservation,
    joint: dict[str, Any],
    base_mesh0: trimesh.Trimesh,
    screen_mesh0: trimesh.Trimesh,
    pose_timeline: PoseTimeline | None,
) -> list[dict[str, Any]]:
    origin = np.asarray(joint["origin_xyz"], dtype=np.float64)
    axis = np.asarray(joint["axis_xyz"], dtype=np.float64)
    axis /= np.linalg.norm(axis) + 1e-12
    entries: list[dict[str, Any]] = []
    for opt, hand in zip(optimized, hand_frames):
        frame_dir = config.output_dir / f"frame_{frame_name(opt.frame)}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        t_frame_from_align = camera_transform_for_resolved_pose(meta, config.export_root, opt.pose_frame, pose_timeline)
        t_align_from_frame = np.linalg.inv(t_frame_from_align)
        base_frame = base_mesh0.copy()
        screen_rot = rotate_mesh_about_axis(screen_mesh0, origin, axis, opt.theta_rad)
        screen_frame = screen_rot
        joint_frame = joint
        base_path = frame_dir / f"part_{BASE_PART_LABEL}_dynamic.obj"
        screen_path = frame_dir / f"part_{SCREEN_PART_LABEL}_dynamic.obj"
        joint_path = frame_dir / "joint_dynamic.json"
        base_frame.export(base_path)
        screen_frame.export(screen_path)
        write_json(joint_path, {"joints": [joint_frame]})
        left_hand_path = left_arm_path = None
        center = opt.hand_rotation_center if opt.hand_rotation_center is not None else hand_rotation_center(hand)
        hand_mesh_align = apply_se3_to_mesh(hand.hand_mesh, t_align_from_frame) if hand.hand_mesh is not None else None
        arm_mesh_align = apply_se3_to_mesh(hand.arm_mesh, t_align_from_frame) if hand.arm_mesh is not None else None
        corrected_hand = refined_hand_mesh(hand_mesh_align, center, opt.hand_rotvec, opt.hand_delta)
        corrected_arm = refined_hand_mesh(arm_mesh_align, center, opt.hand_rotvec, opt.hand_delta)
        if corrected_hand is not None:
            left_hand_path = frame_dir / f"{config.hand_side}_hand_{config.corrected_hand_suffix}.obj"
            corrected_hand.export(left_hand_path)
        if corrected_arm is not None:
            left_arm_path = frame_dir / f"{config.hand_side}_arm_{config.corrected_hand_suffix}.obj"
            corrected_arm.export(left_arm_path)
        rgb_path = config.rgb_dir / f"{frame_name(opt.frame)}.png"
        entry = {
            "frame": int(opt.frame),
            "local_frame": int(opt.local_index),
            "pose_frame": int(opt.pose_frame),
            "rgb_path": str(rgb_path),
            "angle_rad": float(opt.theta_rad),
            "angle_deg": float(np.rad2deg(opt.theta_rad)),
            "base_mesh": str(base_path),
            "screen_mesh": str(screen_path),
            "joint_json": str(joint_path),
            "status": opt.status,
            "contact_error_m": float(opt.contact_error_m),
            "radius_error_m": float(opt.radius_error_m),
            "axis_error_m": float(opt.axis_error_m),
            "penetration_m": float(opt.penetration_m),
            "hand_delta_m": [float(v) for v in opt.hand_delta],
            "hand_refine_mode": config.hand_refine_mode,
            "hand_rotvec_rad": [float(v) for v in (opt.hand_rotvec if opt.hand_rotvec is not None else np.zeros(3, dtype=np.float64))],
            "hand_rotvec_deg_norm": float(np.rad2deg(np.linalg.norm(opt.hand_rotvec if opt.hand_rotvec is not None else np.zeros(3, dtype=np.float64)))),
            "hand_rotation_center_camera": [float(v) for v in center],
            "camera_to_frame0_matrix": t_align_from_frame.tolist(),
        }
        if left_hand_path is not None:
            entry[f"{config.hand_side}_hand_mesh"] = str(left_hand_path)
        if left_arm_path is not None:
            entry[f"{config.hand_side}_arm_mesh"] = str(left_arm_path)
        if opt.fingertip_point_corrected is not None:
            entry["fingertip_point_camera"] = [float(v) for v in opt.fingertip_point_corrected]
        if opt.screen_contact_point is not None:
            entry["screen_contact_point_camera"] = [float(v) for v in opt.screen_contact_point]
        entries.append(entry)
    return entries


def run_contact_driven_screen(config: ContactDrivenScreenConfig) -> dict[str, Any]:
    config.alignment_dir = config.alignment_dir.resolve()
    config.export_root = config.export_root.resolve()
    config.rgb_dir = config.rgb_dir.resolve()
    config.hand_dir = config.hand_dir.resolve()
    config.output_dir = config.output_dir.resolve()
    if config.vlm_contact_json is not None:
        config.vlm_contact_json = config.vlm_contact_json.resolve()
    if config.pose_csv is not None:
        config.pose_csv = config.pose_csv.resolve()
    if config.contact_fingers is not None:
        config.contact_fingers = tuple(normalize_fingers(config.contact_fingers)) or None
    config.output_dir.mkdir(parents=True, exist_ok=True)
    frames = list(range(config.start_frame, config.end_frame + 1))
    if not frames:
        raise ValueError("No frames requested")
    contact_frame_source: dict[str, Any] = {"type": "geometric_nearest_fingertip_search"}
    if config.contact_force_frame is not None:
        contact_frame_source = {
            "type": "manual_contact_force_frame",
            "frame": int(config.contact_force_frame),
        }
    elif config.vlm_contact_json is not None:
        contact_frame, contact_frame_source = resolve_vlm_contact_frame(
            config.vlm_contact_json,
            config.vlm_contact_target_id,
            frames,
        )
        if config.contact_fingers is None:
            semantic_fingers = normalize_fingers(contact_frame_source.get("contact_fingers"))
            if semantic_fingers:
                config.contact_fingers = tuple(semantic_fingers)
        mode = config.vlm_contact_mode.lower()
        if mode == "force":
            config.contact_force_frame = int(contact_frame)
        elif mode == "window":
            window_start = max(frames[0], int(contact_frame) - max(0, int(config.vlm_contact_window_before)))
            window_end = min(frames[-1], int(contact_frame) + max(0, int(config.vlm_contact_window_after)))
            if config.contact_search_start > frames[0]:
                window_start = max(window_start, int(config.contact_search_start))
            if config.contact_search_end is not None:
                window_end = min(window_end, int(config.contact_search_end))
            config.contact_search_start = window_start
            config.contact_search_end = window_end
            contact_frame_source = {
                **contact_frame_source,
                "type": "vlm_window_geometric_refine",
                "semantic_frame": int(contact_frame),
                "window_start": int(window_start),
                "window_end": int(window_end),
                "distance_threshold_m": config.contact_distance_threshold_m,
                "consecutive_frames": int(config.contact_distance_consecutive_frames),
                "min_hits": int(config.contact_distance_min_hits),
            }
        else:
            raise ValueError(f"Unsupported vlm_contact_mode={config.vlm_contact_mode!r}; expected force or window")
    meta = read_json(config.export_root / "manifest.json")
    pose_timeline = load_pose_timeline(config.pose_csv)
    if pose_timeline is not None:
        for frame in frames:
            pose_timeline.require_row(frame)
        pose_frame_max = pose_timeline.max_index
    else:
        pose_frame_max = max(0, int(math.floor(frames[-1] * config.pose_fps / max(config.fps, 1e-6))))
        pose_frame_max = min(pose_frame_max, max_export_frame_index(config.export_root))
    joints = read_json(config.alignment_dir / "joint_camera.json").get("joints", [])
    if not joints:
        raise ValueError(f"No joint found in {config.alignment_dir / 'joint_camera.json'}")
    joint = joints[0]
    base_mesh0 = load_mesh(config.alignment_dir / f"part_{BASE_PART_LABEL}_camera.obj")
    screen_mesh0 = load_mesh(config.alignment_dir / f"part_{SCREEN_PART_LABEL}_camera.obj")
    screen_surface_align = mesh_surface_points(screen_mesh0, config.surface_sample_count)
    hand_frames = [load_hand_frame(config.hand_dir, frame, config.hand_side) for frame in frames]
    observations = build_contact_observations(config, meta, frames, hand_frames, screen_surface_align, pose_frame_max, pose_timeline)
    contact_index = find_contact_index(observations, config)
    contact_obs = observations[contact_index]
    screen_normal_align = choose_screen_normal_toward_hand(
        screen_mesh0,
        contact_obs.screen_point_align,
        contact_obs.fingertip_point,
    )
    optimized = optimize_after_contact(
        config,
        meta,
        frames,
        hand_frames,
        contact_obs,
        contact_index,
        joint,
        screen_mesh0,
        screen_normal_align,
        pose_frame_max,
        pose_timeline,
    )
    frame_entries = export_contact_sequence(config, meta, frames, hand_frames, optimized, contact_obs, joint, base_mesh0, screen_mesh0, pose_timeline)
    depth_assignments = standard_depth_assignments(config, meta, frames, pose_timeline)
    for entry in frame_entries:
        assignment = depth_assignments.get(int(entry["frame"]))
        if assignment is None:
            continue
        entry["standard_depth_frame"] = int(assignment["frame"])
        entry["standard_depth_timestamp_s"] = float(assignment["timestamp_s"])
        entry["standard_depth_timestamp_delta_s"] = float(assignment["timestamp_delta_s"])
        entry["standard_depth_camera_to_frame0_matrix"] = assignment["camera_to_alignment_matrix"]
    contact_points = np.full((len(frames), 2, 3), np.nan, dtype=np.float32)
    visibility = np.zeros((len(frames), 2), dtype=bool)
    for opt in optimized:
        if opt.fingertip_point_corrected is not None and opt.screen_contact_point is not None:
            contact_points[opt.local_index, 0] = opt.fingertip_point_corrected.astype(np.float32)
            contact_points[opt.local_index, 1] = opt.screen_contact_point.astype(np.float32)
            visibility[opt.local_index, :] = True
    np.save(config.output_dir / "contact_points_frame_camera.npy", contact_points)
    # Reuse the existing Viser tracked-points convention: two points are shown,
    # the corrected fingertip and the screen-side sticky contact point.
    np.save(config.output_dir / "tracked_points_frame_camera.npy", contact_points.reshape(len(frames), 2, 3))
    np.save(config.output_dir / "tracks_visibility.npy", visibility)
    np.save(config.output_dir / "screen_angles_rad.npy", np.asarray([o.theta_rad for o in optimized], dtype=np.float32))
    np.save(
        config.output_dir / "hand_refine_rotvec_rad.npy",
        np.asarray([o.hand_rotvec if o.hand_rotvec is not None else np.zeros(3, dtype=np.float64) for o in optimized], dtype=np.float32),
    )
    np.save(config.output_dir / "hand_refine_delta_m.npy", np.asarray([o.hand_delta for o in optimized], dtype=np.float32))
    obs_rows = [
        {
            "frame": obs.frame,
            "local_frame": obs.local_index,
            "pose_frame": obs.pose_frame,
            "distance_m": obs.distance_m,
            "fingertip_index": obs.fingertip_index,
            "fingertip_name": obs.fingertip_name,
            "is_contact_frame": obs.local_index == contact_index,
        }
        for obs in observations
    ]
    write_csv(
        config.output_dir / "contact_search.csv",
        obs_rows,
        ["frame", "local_frame", "pose_frame", "distance_m", "fingertip_index", "fingertip_name", "is_contact_frame"],
    )
    diag_rows = [
        {
            "frame": opt.frame,
            "local_frame": opt.local_index,
            "pose_frame": opt.pose_frame,
            "theta_rad": opt.theta_rad,
            "theta_deg": float(np.rad2deg(opt.theta_rad)),
            "status": opt.status,
            "distance_before_m": opt.distance_before_m,
            "contact_error_m": opt.contact_error_m,
            "radius_error_m": opt.radius_error_m,
            "axis_error_m": opt.axis_error_m,
            "penetration_m": opt.penetration_m,
            "loss": opt.loss,
            "hand_delta_norm_m": float(np.linalg.norm(opt.hand_delta)),
            "hand_delta_x_m": float(opt.hand_delta[0]),
            "hand_delta_y_m": float(opt.hand_delta[1]),
            "hand_delta_z_m": float(opt.hand_delta[2]),
            "hand_rotvec_norm_deg": float(np.rad2deg(np.linalg.norm(opt.hand_rotvec if opt.hand_rotvec is not None else np.zeros(3, dtype=np.float64)))),
            "hand_rotvec_x_rad": float((opt.hand_rotvec if opt.hand_rotvec is not None else np.zeros(3, dtype=np.float64))[0]),
            "hand_rotvec_y_rad": float((opt.hand_rotvec if opt.hand_rotvec is not None else np.zeros(3, dtype=np.float64))[1]),
            "hand_rotvec_z_rad": float((opt.hand_rotvec if opt.hand_rotvec is not None else np.zeros(3, dtype=np.float64))[2]),
        }
        for opt in optimized
    ]
    write_csv(
        config.output_dir / "contact_optimization.csv",
        diag_rows,
        [
            "frame",
            "local_frame",
            "pose_frame",
            "theta_rad",
            "theta_deg",
            "status",
            "distance_before_m",
            "contact_error_m",
            "radius_error_m",
            "axis_error_m",
            "penetration_m",
            "loss",
            "hand_delta_norm_m",
            "hand_delta_x_m",
            "hand_delta_y_m",
            "hand_delta_z_m",
            "hand_rotvec_norm_deg",
            "hand_rotvec_x_rad",
            "hand_rotvec_y_rad",
            "hand_rotvec_z_rad",
        ],
    )
    motion_source = str(config.pose_csv) if config.pose_csv is not None else str(config.export_root / "frames.csv")
    motion_description = (
        "Each EgoForce hand mesh is transformed from its same-index 15fps pose CSV camera coordinates into frame-0 right-camera coordinates before contact optimization and export."
        if config.pose_csv is not None
        else "Each EgoForce hand mesh is transformed from its pose frame camera coordinates into frame-0 right-camera coordinates before contact optimization and export."
    )
    manifest = {
        "type": "contact_driven_laptop_screen",
        "alignment_dir": str(config.alignment_dir),
        "export_root": str(config.export_root),
        "rgb_dir": str(config.rgb_dir),
        "hand_dir": str(config.hand_dir),
        "output_dir": str(config.output_dir),
        "coordinate_frame": "alignment_frame0_right_camera",
        "motion_compensation": {
            "enabled": True,
            "source": motion_source,
            "pose_csv_indexed_by_tracker_frame": config.pose_csv is not None,
            "pose_frame_max": int(pose_frame_max),
            "alignment_timestamp_s": float(frame_row(config.export_root, 0)["rgb_pose_timestamp_s"]),
            "rgb_pose_image_rotation_deg": float(meta.get("rgb_pose_image_rotation_deg", -90.0)),
            "description": motion_description,
        },
        "depth_display_mode": "standard_frames_only" if depth_assignments else "none",
        "standard_depth_tracker_frames": sorted(depth_assignments),
        "standard_depth_frame_count": len(depth_assignments),
        "frames": frame_entries,
        "frame_indices": frames,
        "fps": float(config.fps),
        "pose_fps": float(config.pose_fps),
        "hand_side": config.hand_side,
        "hand_refine_mode": config.hand_refine_mode,
        "base_part_label": BASE_PART_LABEL,
        "screen_part_label": SCREEN_PART_LABEL,
        "joint_align_camera": joint,
        "contact": {
            "frame": int(contact_obs.frame),
            "local_frame": int(contact_obs.local_index),
            "pose_frame": int(contact_obs.pose_frame),
            "frame_source": contact_frame_source,
            "distance_m": float(contact_obs.distance_m),
            "fingertip_index": int(contact_obs.fingertip_index),
            "fingertip_name": str(contact_obs.fingertip_name),
            "semantic_candidate_fingers": list(config.contact_fingers or ()),
            "screen_point_align": [float(v) for v in contact_obs.screen_point_align],
            "screen_point_frame": [float(v) for v in contact_obs.screen_point_frame],
        },
        "contact_search_csv": str(config.output_dir / "contact_search.csv"),
        "contact_optimization_csv": str(config.output_dir / "contact_optimization.csv"),
        "optimization_csv": str(config.output_dir / "contact_optimization.csv"),
        "contact_points_npy": str(config.output_dir / "contact_points_frame_camera.npy"),
        "hand_refine_rotvec_npy": str(config.output_dir / "hand_refine_rotvec_rad.npy"),
        "hand_refine_delta_npy": str(config.output_dir / "hand_refine_delta_m.npy"),
        "contact_manifest": str(config.output_dir / "contact_manifest.json"),
        "dynamic_manifest": str(config.output_dir / "dynamic_manifest.json"),
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in config.__dict__.items()},
        "notes": [
            "Screen contact point is fixed in reference screen mesh coordinates.",
            "Screen motion is one-DoF hinge rotation.",
            "translation mode preserves the original per-frame hand translation MVP.",
            "global_rigid mode refines a small EgoForce/MANO global orientation and translation proxy, not finger pose.",
            "The loss terms are structured so a full MANO layer can replace the mesh-level proxy later.",
        ],
    }
    write_json(config.output_dir / "contact_manifest.json", manifest)
    write_json(config.output_dir / "dynamic_manifest.json", manifest)
    return manifest
