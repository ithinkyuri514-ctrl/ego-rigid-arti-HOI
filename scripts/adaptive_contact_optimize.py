#!/usr/bin/env python3
"""Conservative geometry-level adaptive hand-object contact correction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter1d, map_coordinates
from scipy.sparse import coo_matrix, diags, eye
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
import trimesh


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGION_JOINTS = {
    "index": (1, 2, 3),
    "middle": (4, 5, 6),
    "pinky": (7, 8, 9),
    "ring": (10, 11, 12),
    "thumb": (13, 14, 15),
    "palm": (0,),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--object-id", default="bottle")
    parser.add_argument("--object-mesh", type=Path, required=True)
    parser.add_argument("--tracking-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--hand-manifest", type=Path, default=None)
    parser.add_argument("--object-mask-dir", type=Path, default=None)
    parser.add_argument(
        "--interaction-class",
        choices=("rigid", "articulated"),
        default="rigid",
    )
    parser.add_argument("--depth-dir", type=Path, default=None)
    parser.add_argument("--poses-path", type=Path, default=None)
    parser.add_argument("--hand-side", choices=("left", "right"), default="right")
    parser.add_argument("--voxel-pitch-m", type=float, default=0.0025)
    parser.add_argument("--voxel-padding-m", type=float, default=0.06)
    parser.add_argument("--max-global-translation-m", type=float, default=0.035)
    parser.add_argument("--max-local-offset-m", type=float, default=0.010)
    parser.add_argument("--contact-vertices-per-region", type=int, default=12)
    parser.add_argument("--spatial-regularization", type=float, default=2.5)
    parser.add_argument("--offset-regularization", type=float, default=0.35)
    parser.add_argument("--depth-global-weight", type=float, default=2.0)
    parser.add_argument("--depth-local-weight", type=float, default=3.0)
    parser.add_argument("--depth-max-correspondence-m", type=float, default=0.035)
    parser.add_argument("--depth-max-points", type=int, default=2500)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError(f"Invalid mesh: {path}")
    return mesh


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return np.asarray(points) @ transform[:3, :3].T + transform[:3, 3]


def clip_vectors(vectors: np.ndarray, maximum: float) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors * np.minimum(1.0, maximum / np.maximum(norms, 1e-12))


class VoxelSDF:
    def __init__(self, mesh: trimesh.Trimesh, pitch: float, padding_m: float) -> None:
        voxel = mesh.voxelized(pitch=pitch).fill()
        occupancy = np.asarray(voxel.matrix, dtype=bool)
        padding = max(3, int(np.ceil(padding_m / pitch)))
        occupancy = np.pad(occupancy, padding, mode="constant", constant_values=False)
        outside = distance_transform_edt(~occupancy)
        inside = distance_transform_edt(occupancy)
        self.sdf = np.where(occupancy, -(inside - 0.5), outside - 0.5).astype(np.float32) * pitch
        self.origin = np.asarray(voxel.transform[:3, 3], dtype=np.float64) - padding * pitch
        self.pitch = float(pitch)
        self.gradient = np.stack(np.gradient(self.sdf, self.pitch), axis=0).astype(np.float32)

    def query(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        coordinates = ((np.asarray(points) - self.origin) / self.pitch).T
        distance = map_coordinates(
            self.sdf,
            coordinates,
            order=1,
            mode="constant",
            cval=float(self.sdf.max()),
        )
        gradient = np.stack(
            [
                map_coordinates(
                    self.gradient[axis], coordinates, order=1, mode="nearest"
                )
                for axis in range(3)
            ],
            axis=1,
        )
        gradient /= np.maximum(np.linalg.norm(gradient, axis=1, keepdims=True), 1e-8)
        return distance.astype(np.float64), gradient.astype(np.float64)


def interaction_interval(
    vlm: dict[str, Any], object_id: str, interaction_class: str
) -> tuple[int, int]:
    events = [
        event
        for event in vlm["vlm_result"]["events"]
        if event.get("interaction_class") == interaction_class
        and event.get("object_id") == object_id
    ]
    if not events:
        raise ValueError(f"No {interaction_class} VLM events for {object_id!r}")
    return min(int(event["start_frame"]) for event in events), max(
        int(event["end_frame"]) for event in events
    )


def build_region_ids(vertices: np.ndarray, joints: np.ndarray) -> dict[str, np.ndarray]:
    assignment = np.linalg.norm(
        vertices[:, None, :] - joints[None, :16, :], axis=2
    ).argmin(axis=1)
    regions = {
        name: np.flatnonzero(np.isin(assignment, columns))
        for name, columns in REGION_JOINTS.items()
    }
    palm = regions["palm"]
    if len(palm):
        thenar_count = min(32, len(palm))
        order = np.argsort(np.linalg.norm(vertices[palm] - joints[13], axis=1))
        regions["thenar"] = palm[order[:thenar_count]]
    else:
        regions["thenar"] = np.empty(0, dtype=np.int64)
    return regions


def graph_laplacian(mesh: trimesh.Trimesh):
    edges = np.asarray(mesh.edges_unique, dtype=np.int64)
    row = np.r_[edges[:, 0], edges[:, 1]]
    col = np.r_[edges[:, 1], edges[:, 0]]
    adjacency = coo_matrix(
        (np.ones(len(row), dtype=np.float64), (row, col)),
        shape=(len(mesh.vertices), len(mesh.vertices)),
    ).tocsr()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    return diags(degree) - adjacency


def rasterize_mesh_support(
    vertices_ct: np.ndarray,
    faces: np.ndarray,
    intrinsics: dict[str, float],
    shape: tuple[int, int],
) -> np.ndarray:
    height, width = shape
    z = vertices_ct[:, 2]
    uv = np.column_stack(
        [
            float(intrinsics["fx"]) * vertices_ct[:, 0] / np.maximum(z, 1e-8)
            + float(intrinsics["cx"]),
            float(intrinsics["fy"]) * vertices_ct[:, 1] / np.maximum(z, 1e-8)
            + float(intrinsics["cy"]),
        ]
    )
    support = np.zeros((height, width), dtype=np.uint8)
    valid_vertex = z > 0.05
    for face in faces:
        if not valid_vertex[face].all():
            continue
        polygon = np.rint(uv[face]).astype(np.int32)
        if (
            polygon[:, 0].max() < 0
            or polygon[:, 0].min() >= width
            or polygon[:, 1].max() < 0
            or polygon[:, 1].min() >= height
        ):
            continue
        cv2.fillConvexPoly(support, polygon, 1)
    return cv2.dilate(support, np.ones((13, 13), np.uint8), iterations=1) > 0


def extract_hand_depth_points_c0(
    depth: np.ndarray,
    hand_mask: np.ndarray,
    object_mask: np.ndarray,
    vertices_c0: np.ndarray,
    faces: np.ndarray,
    pose_c0_from_ct: np.ndarray,
    intrinsics: dict[str, float],
    maximum: int,
    seed: int,
) -> np.ndarray:
    inverse_pose = np.linalg.inv(pose_c0_from_ct)
    vertices_ct = transform_points(vertices_c0, inverse_pose)
    support = rasterize_mesh_support(vertices_ct, faces, intrinsics, depth.shape)
    object_exclusion = cv2.dilate(
        object_mask.astype(np.uint8), np.ones((7, 7), np.uint8), iterations=1
    ) > 0
    valid = (
        support
        & hand_mask
        & ~object_exclusion
        & np.isfinite(depth)
        & (depth > 0.1)
        & (depth < 2.0)
    )
    y, x = np.nonzero(valid)
    if not len(x):
        return np.empty((0, 3), dtype=np.float64)
    z = depth[y, x].astype(np.float64)
    points_ct = np.column_stack(
        [
            (x - float(intrinsics["cx"])) * z / float(intrinsics["fx"]),
            (y - float(intrinsics["cy"])) * z / float(intrinsics["fy"]),
            z,
        ]
    )
    points_c0 = transform_points(points_ct, pose_c0_from_ct)
    if len(points_c0) > maximum:
        rng = np.random.default_rng(seed)
        points_c0 = points_c0[rng.choice(len(points_c0), maximum, replace=False)]
    return points_c0


def depth_correspondences(
    points_c0: np.ndarray,
    vertices_c0: np.ndarray,
    maximum_distance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if len(points_c0) < 32:
        return (
            np.empty(0, dtype=np.int64),
            np.empty((0, 3), dtype=np.float64),
            np.empty(0, dtype=np.float64),
            0.0,
        )
    distance, vertex_index = cKDTree(vertices_c0).query(points_c0, k=1)
    valid = np.isfinite(distance) & (distance <= maximum_distance)
    if np.count_nonzero(valid) < 32:
        return (
            np.empty(0, dtype=np.int64),
            np.empty((0, 3), dtype=np.float64),
            np.empty(0, dtype=np.float64),
            0.0,
        )
    distance = distance[valid]
    vertex_index = vertex_index[valid]
    residual = points_c0[valid] - vertices_c0[vertex_index]
    trim = distance <= np.quantile(distance, 0.85)
    distance = distance[trim]
    vertex_index = vertex_index[trim]
    residual = residual[trim]
    quality = float(
        np.clip(len(distance) / 600.0, 0.0, 1.0)
        * np.exp(-np.median(distance) / 0.018)
    )
    return vertex_index, residual, distance, quality


def select_opposing_fingers(
    gaps: np.ndarray, names: list[str], start: int, end: int, switch_penalty_m: float = 0.008
) -> list[str]:
    count, finger_count = gaps.shape
    cost = np.full((count, finger_count), np.inf, dtype=np.float64)
    back = np.zeros((count, finger_count), dtype=np.int64)
    cost[0] = np.clip(gaps[0], 0.0, 0.08)
    for frame in range(1, count):
        for current in range(finger_count):
            transition = cost[frame - 1] + switch_penalty_m * (
                np.arange(finger_count) != current
            )
            previous = int(np.argmin(transition))
            cost[frame, current] = transition[previous] + np.clip(gaps[frame, current], 0.0, 0.08)
            back[frame, current] = previous
    states = np.zeros(count, dtype=np.int64)
    states[-1] = int(np.argmin(cost[-1]))
    for frame in range(count - 1, 0, -1):
        states[frame - 1] = back[frame, states[frame]]
    chosen = [names[index] for index in states]
    if start > 0:
        chosen[:start] = [chosen[start]] * start
    if end + 1 < count:
        chosen[end + 1 :] = [chosen[end]] * (count - end - 1)
    return chosen


def main() -> None:
    args = parse_args()
    workspace = args.workspace.resolve()
    output = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else workspace / "outputs/11_adaptive_contact_depth_v2"
    )
    output.mkdir(parents=True, exist_ok=True)
    optimized_root = output / "optimized_C0"
    optimized_root.mkdir(parents=True, exist_ok=True)

    hand_manifest_path = (
        args.hand_manifest.resolve()
        if args.hand_manifest is not None
        else workspace / "outputs/09_egoforce/dynamic_manifest.json"
    )
    hand_manifest = read_json(hand_manifest_path)
    frames = hand_manifest["frames"]
    side_index = 0 if args.hand_side == "left" else 1
    object_mesh = load_mesh(args.object_mesh.resolve())
    sdf = VoxelSDF(object_mesh, args.voxel_pitch_m, args.voxel_padding_m)
    delta_path = args.tracking_dir.resolve() / "Delta_C0_object_motion.npy"
    object_delta = np.load(delta_path).astype(np.float64)
    camera = read_json(workspace / "outputs/00_rgb_frames/camera.json")
    intrinsics = camera.get("rgb_intrinsics_selected", camera["rgb_intrinsics_right"])
    depth_dir = (
        args.depth_dir.resolve()
        if args.depth_dir is not None
        else workspace / "outputs/06_dense_depth/raw_projected_npy"
    )
    poses_path = (
        args.poses_path.resolve()
        if args.poses_path is not None
        else workspace / "outputs/00_pose_refinement/poses_refined.npz"
    )
    poses_c0_from_ct = np.load(poses_path)["T_C0_from_Ct"].astype(np.float64)
    object_mask_dir = (
        args.object_mask_dir.resolve()
        if args.object_mask_dir is not None
        else workspace / f"outputs/04_object_masks/{args.object_id}/objects/{args.object_id}"
    )
    tracked_count = min(len(object_delta), len(frames))
    start, end = interaction_interval(
        read_json(workspace / "outputs/01_vlm/mixed_interactions.json"),
        args.object_id,
        args.interaction_class,
    )
    start = int(np.clip(start, 0, tracked_count - 1))
    end = int(np.clip(end, start, tracked_count - 1))

    geometry: list[dict[str, Any]] = []
    for frame in range(tracked_count):
        geometry_path = Path(frames[frame]["geometry_C0_npz"])
        with np.load(geometry_path) as loaded:
            record = {
                "vertices": loaded["hand_vertices"][side_index].astype(np.float64),
                "joints": loaded["hand_joints"][side_index].astype(np.float64),
                "faces": loaded[f"{args.hand_side}_hand_faces"].astype(np.int64),
                "arm_vertices": loaded["arm_vertices"][side_index].astype(np.float64),
                "arm_faces": loaded["arm_faces"].astype(np.int64),
            }
        previous_path = frames[frame].get("adaptive_contact_geometry_C0")
        if previous_path:
            with np.load(previous_path) as previous:
                record["vertices"] = previous["hand_vertices"].astype(np.float64)
                record["faces"] = previous["hand_faces"].astype(np.int64)
                record["arm_vertices"] = previous["arm_vertices"].astype(np.float64)
                record["arm_faces"] = previous["arm_faces"].astype(np.int64)
        geometry.append(record)

    reference_frame = start
    region_ids = build_region_ids(
        geometry[reference_frame]["vertices"], geometry[reference_frame]["joints"]
    )
    reference_mesh = trimesh.Trimesh(
        vertices=geometry[reference_frame]["vertices"],
        faces=geometry[reference_frame]["faces"],
        process=False,
    )
    laplacian = graph_laplacian(reference_mesh)

    finger_names = ["index", "middle", "ring", "pinky"]
    gaps = np.zeros((tracked_count, len(finger_names)), dtype=np.float64)
    thumb_gap = np.zeros(tracked_count, dtype=np.float64)
    raw_sdf: list[np.ndarray] = []
    raw_gradient: list[np.ndarray] = []
    vertices_object: list[np.ndarray] = []
    for frame in range(tracked_count):
        inverse = np.linalg.inv(object_delta[frame])
        points = transform_points(geometry[frame]["vertices"], inverse)
        distance, gradient = sdf.query(points)
        vertices_object.append(points)
        raw_sdf.append(distance)
        raw_gradient.append(gradient)
        thumb_gap[frame] = float(
            np.mean(np.sort(np.abs(distance[region_ids["thumb"]]))[: args.contact_vertices_per_region])
        )
        for index, name in enumerate(finger_names):
            gaps[frame, index] = float(
                np.mean(np.sort(np.abs(distance[region_ids[name]]))[: args.contact_vertices_per_region])
            )

    opposing = select_opposing_fingers(gaps, finger_names, start, end)
    opposing_gap = np.asarray(
        [gaps[frame, finger_names.index(opposing[frame])] for frame in range(tracked_count)]
    )
    geometry_score = np.exp(-0.5 * (np.minimum(thumb_gap, opposing_gap) / 0.025) ** 2)
    vlm_weight = np.zeros(tracked_count, dtype=np.float64)
    vlm_weight[start : end + 1] = 1.0
    ramp = min(2, max(0, (end - start) // 3))
    for offset in range(ramp):
        value = float(offset + 1) / float(ramp + 1)
        vlm_weight[start + offset] *= value
        vlm_weight[end - offset] *= value
    activation = vlm_weight * (0.35 + 0.65 * geometry_score)

    contact_translation_target = np.zeros((tracked_count, 3), dtype=np.float64)
    for frame in range(tracked_count):
        ids = []
        for region in ("thumb", opposing[frame]):
            candidates = region_ids[region]
            order = np.argsort(np.abs(raw_sdf[frame][candidates]))
            ids.extend(candidates[order[: args.contact_vertices_per_region]].tolist())
        ids_array = np.unique(np.asarray(ids, dtype=np.int64))
        displacement = -raw_sdf[frame][ids_array, None] * raw_gradient[frame][ids_array]
        contact_translation_target[frame] = np.median(displacement, axis=0)
    contact_translation_target = clip_vectors(
        contact_translation_target, args.max_global_translation_m
    )

    depth_points_c0: list[np.ndarray] = []
    depth_vertex_index: list[np.ndarray] = []
    depth_residual: list[np.ndarray] = []
    depth_distance: list[np.ndarray] = []
    depth_quality = np.zeros(tracked_count, dtype=np.float64)
    depth_translation_target = np.zeros((tracked_count, 3), dtype=np.float64)
    pointcloud_root = output / "hand_depth_pointcloud_C0"
    pointcloud_root.mkdir(parents=True, exist_ok=True)
    for frame in range(tracked_count):
        depth = np.load(depth_dir / f"{frame:06d}.npy").astype(np.float32)
        hand_mask = np.asarray(
            Image.open(frames[frame]["sam2_hand_mask_path"]).convert("L")
        ) > 127
        object_mask = np.asarray(
            Image.open(object_mask_dir / f"{frame:06d}.png").convert("L")
        ) > 127
        points = extract_hand_depth_points_c0(
            depth,
            hand_mask,
            object_mask,
            geometry[frame]["vertices"],
            geometry[frame]["faces"],
            poses_c0_from_ct[frame],
            intrinsics,
            args.depth_max_points,
            seed=20260803 + frame,
        )
        vertex_index, residual, distance, quality = depth_correspondences(
            points, geometry[frame]["vertices"], args.depth_max_correspondence_m
        )
        depth_points_c0.append(points)
        depth_vertex_index.append(vertex_index)
        depth_residual.append(residual)
        depth_distance.append(distance)
        depth_quality[frame] = quality
        if len(residual):
            depth_translation_target[frame] = (
                np.median(residual, axis=0) @ object_delta[frame][:3, :3]
            )
        np.savez_compressed(
            pointcloud_root / f"{frame:06d}.npz",
            points_C0=points.astype(np.float32),
            correspondence_vertex_index=vertex_index,
            correspondence_residual_C0=residual.astype(np.float32),
            correspondence_distance_m=distance.astype(np.float32),
        )

    contact_weight = activation
    point_weight = args.depth_global_weight * depth_quality
    total_weight = contact_weight + point_weight
    translation_target = np.divide(
        contact_weight[:, None] * contact_translation_target
        + point_weight[:, None] * depth_translation_target,
        np.maximum(total_weight[:, None], 1e-8),
    )
    translation_target = clip_vectors(translation_target, args.max_global_translation_m)
    global_translation_object = gaussian_filter1d(
        translation_target, sigma=1.0, axis=0, mode="nearest"
    )
    global_translation_object *= np.clip(total_weight[:, None], 0.0, 1.0)
    global_translation_object = clip_vectors(
        global_translation_object, args.max_global_translation_m
    )

    optimized_records: list[dict[str, Any]] = []
    output_manifest_frames: list[dict[str, Any]] = []
    for frame, original_record in enumerate(frames):
        record = dict(original_record)
        if frame >= tracked_count or not original_record.get(f"{args.hand_side}_hand_C0"):
            output_manifest_frames.append(record)
            continue
        rotation = object_delta[frame][:3, :3]
        global_c0 = global_translation_object[frame] @ rotation.T
        vertices_global = geometry[frame]["vertices"] + global_c0
        arm_global = geometry[frame]["arm_vertices"] + global_c0
        points_object = vertices_object[frame] + global_translation_object[frame]
        distance, gradient = sdf.query(points_object)

        target_offset_object = np.zeros_like(points_object)
        confidence = np.zeros(len(points_object), dtype=np.float64)
        if len(depth_points_c0[frame]):
            local_index, local_residual, _, local_quality = depth_correspondences(
                depth_points_c0[frame], vertices_global, args.depth_max_correspondence_m
            )
            if len(local_index):
                for vertex_index in np.unique(local_index):
                    samples = local_residual[local_index == vertex_index]
                    target_offset_object[vertex_index] = (
                        np.median(samples, axis=0) @ rotation
                    )
                    confidence[vertex_index] = max(
                        confidence[vertex_index],
                        args.depth_local_weight * local_quality,
                    )
        if activation[frame] > 0.0:
            for region, region_weight in (
                ("thumb", 1.0),
                (opposing[frame], 1.0),
                ("thenar", 0.35),
            ):
                candidates = region_ids[region]
                if not len(candidates):
                    continue
                count = min(args.contact_vertices_per_region, len(candidates))
                chosen = candidates[np.argsort(np.abs(distance[candidates]))[:count]]
                target_offset_object[chosen] = -distance[chosen, None] * gradient[chosen]
                confidence[chosen] = np.maximum(
                    confidence[chosen], activation[frame] * region_weight
                )
        penetrating = distance < -0.0015
        target_offset_object[penetrating] = -(
            distance[penetrating, None] + 0.0005
        ) * gradient[penetrating]
        confidence[penetrating] = np.maximum(confidence[penetrating], 1.0)
        target_offset_c0 = target_offset_object @ rotation.T
        target_offset_c0 = clip_vectors(target_offset_c0, args.max_local_offset_m)

        weights = confidence * 12.0
        system = (
            diags(weights)
            + args.spatial_regularization * laplacian
            + args.offset_regularization * eye(len(points_object), format="csr")
        )
        local_offset = np.column_stack(
            [spsolve(system, weights * target_offset_c0[:, axis]) for axis in range(3)]
        )
        local_offset = clip_vectors(local_offset, args.max_local_offset_m)
        optimized_vertices = vertices_global + local_offset

        optimized_object = transform_points(optimized_vertices, np.linalg.inv(object_delta[frame]))
        final_sdf, _ = sdf.query(optimized_object)
        if len(depth_points_c0[frame]):
            _, _, final_depth_distance, _ = depth_correspondences(
                depth_points_c0[frame],
                optimized_vertices,
                args.depth_max_correspondence_m,
            )
        else:
            final_depth_distance = np.empty(0, dtype=np.float64)
        frame_dir = optimized_root / f"frame_{frame:06d}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        hand_path = frame_dir / f"{args.hand_side}_hand_adaptive_contact_C0.obj"
        arm_path = frame_dir / f"{args.hand_side}_arm_adaptive_contact_C0.obj"
        trimesh.Trimesh(
            vertices=optimized_vertices,
            faces=geometry[frame]["faces"],
            process=False,
        ).export(hand_path)
        trimesh.Trimesh(
            vertices=arm_global,
            faces=geometry[frame]["arm_faces"],
            process=False,
        ).export(arm_path)
        npz_path = frame_dir / "adaptive_contact_geometry_C0.npz"
        np.savez_compressed(
            npz_path,
            hand_vertices=optimized_vertices.astype(np.float32),
            hand_faces=geometry[frame]["faces"],
            arm_vertices=arm_global.astype(np.float32),
            arm_faces=geometry[frame]["arm_faces"],
            global_translation_C0=global_c0.astype(np.float32),
            local_vertex_offsets_C0=local_offset.astype(np.float32),
            signed_distance_before_m=distance.astype(np.float32),
            signed_distance_after_m=final_sdf.astype(np.float32),
        )
        record[f"{args.hand_side}_hand_C0"] = str(hand_path)
        record[f"{args.hand_side}_arm_C0"] = str(arm_path)
        record["adaptive_contact_geometry_C0"] = str(npz_path)
        output_manifest_frames.append(record)

        contact_ids = np.unique(
            np.r_[region_ids["thumb"], region_ids[opposing[frame]]]
        )
        optimized_records.append(
            {
                "frame": frame,
                "activation": float(activation[frame]),
                "opposing_finger": opposing[frame],
                "global_translation_mm": (global_c0 * 1000.0).tolist(),
                "global_translation_norm_mm": float(np.linalg.norm(global_c0) * 1000.0),
                "local_offset_max_mm": float(np.linalg.norm(local_offset, axis=1).max() * 1000.0),
                "depth_point_count": int(len(depth_points_c0[frame])),
                "depth_correspondence_count": int(len(depth_distance[frame])),
                "depth_alignment_quality": float(depth_quality[frame]),
                "pointcloud_distance_before_median_mm": (
                    float(np.median(depth_distance[frame]) * 1000.0)
                    if len(depth_distance[frame])
                    else None
                ),
                "pointcloud_distance_after_median_mm": (
                    float(np.median(final_depth_distance) * 1000.0)
                    if len(final_depth_distance)
                    else None
                ),
                "penetrating_vertices_before": int(np.count_nonzero(distance < -0.0015)),
                "penetrating_vertices_after": int(np.count_nonzero(final_sdf < -0.0015)),
                "penetration_max_before_mm": float(max(0.0, -distance.min()) * 1000.0),
                "penetration_max_after_mm": float(max(0.0, -final_sdf.min()) * 1000.0),
                "contact_gap_before_mm": float(
                    np.mean(np.sort(np.abs(distance[contact_ids]))[: 2 * args.contact_vertices_per_region])
                    * 1000.0
                ),
                "contact_gap_after_mm": float(
                    np.mean(np.sort(np.abs(final_sdf[contact_ids]))[: 2 * args.contact_vertices_per_region])
                    * 1000.0
                ),
            }
        )

    optimized_manifest = dict(hand_manifest)
    optimized_manifest.update(
        {
            "type": "egoforce_adaptive_contact_sequence",
            "source_hand_manifest": str(hand_manifest_path),
            "adaptive_contact_output": str(output),
            "frames": output_manifest_frames,
        }
    )
    optimized_manifest_path = output / "dynamic_manifest.json"
    write_json(optimized_manifest_path, optimized_manifest)
    active_records = [record for record in optimized_records if record["activation"] > 0.0]
    summary = {
        "stage": "11_adaptive_contact_geometry_correction",
        "status": "completed",
        "workspace": str(workspace),
        "object_id": args.object_id,
        "interaction_class": args.interaction_class,
        "hand_side": args.hand_side,
        "object_mesh": str(args.object_mesh.resolve()),
        "object_tracking": str(delta_path),
        "depth_dir": str(depth_dir),
        "depth_policy": "native projected metric depth intersected with SAM2 hand mask and right-hand mesh support; object mask excluded",
        "poses_path": str(poses_path),
        "object_pose_policy": "fixed; no object correction",
        "mano_policy": "unchanged; exported EgoForce geometry receives bounded translation and local vertex offsets",
        "vlm_prior_interval": [start, end],
        "tracked_frame_count": tracked_count,
        "optimized_hand_manifest": str(optimized_manifest_path),
        "limits": {
            "global_translation_mm": args.max_global_translation_m * 1000.0,
            "local_offset_mm": args.max_local_offset_m * 1000.0,
        },
        "active_frame_count": len(active_records),
        "active_metrics": {
            "contact_gap_before_median_mm": float(
                np.median([record["contact_gap_before_mm"] for record in active_records])
            ),
            "contact_gap_after_median_mm": float(
                np.median([record["contact_gap_after_mm"] for record in active_records])
            ),
            "penetrating_vertices_before_total": int(
                sum(record["penetrating_vertices_before"] for record in active_records)
            ),
            "penetrating_vertices_after_total": int(
                sum(record["penetrating_vertices_after"] for record in active_records)
            ),
            "global_translation_median_mm": float(
                np.median([record["global_translation_norm_mm"] for record in active_records])
            ),
            "global_translation_max_mm": float(
                max(record["global_translation_norm_mm"] for record in active_records)
            ),
            "local_offset_max_mm": float(
                max(record["local_offset_max_mm"] for record in active_records)
            ),
            "pointcloud_distance_before_median_mm": float(
                np.median(
                    [
                        record["pointcloud_distance_before_median_mm"]
                        for record in active_records
                        if record["pointcloud_distance_before_median_mm"] is not None
                    ]
                )
            ),
            "pointcloud_distance_after_median_mm": float(
                np.median(
                    [
                        record["pointcloud_distance_after_median_mm"]
                        for record in active_records
                        if record["pointcloud_distance_after_median_mm"] is not None
                    ]
                )
            ),
        },
        "frames": optimized_records,
    }
    write_json(output / "adaptive_contact_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
