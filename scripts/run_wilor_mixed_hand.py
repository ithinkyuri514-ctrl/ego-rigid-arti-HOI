#!/usr/bin/env python3
"""Run WiLoR on a native RGB timeline and export C0/IoU hand geometry."""
from __future__ import annotations
import argparse, csv, json, subprocess
from pathlib import Path
import cv2
import numpy as np
import trimesh
from scipy.optimize import minimize

SIDES = ("left", "right")

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--wilor-root", type=Path, required=True)
    p.add_argument("--wilor-python", type=Path, default=Path("/opt/conda/envs/wilor/bin/python"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--det-conf", type=float, default=0.3)
    p.add_argument("--rescale-factor", type=float, default=2.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--skip-inference", action="store_true")
    return p.parse_args()

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def transform_points(points, transform):
    h = np.concatenate([points, np.ones((len(points), 1), dtype=points.dtype)], axis=1)
    return (h @ transform.T)[:, :3]

def rasterize(vertices, faces, intrinsics, shape):
    height, width = shape
    z = vertices[:, 2]
    valid = z > 0.03
    uv = np.column_stack([
        float(intrinsics["fx"]) * vertices[:, 0] / np.maximum(z, 1e-8) + float(intrinsics["cx"]),
        float(intrinsics["fy"]) * vertices[:, 1] / np.maximum(z, 1e-8) + float(intrinsics["cy"]),
    ])
    mask = np.zeros((height, width), dtype=np.uint8)
    for face in faces:
        if not valid[face].all():
            continue
        polygon = np.rint(uv[face]).astype(np.int32)
        if polygon[:, 0].max() < 0 or polygon[:, 0].min() >= width:
            continue
        if polygon[:, 1].max() < 0 or polygon[:, 1].min() >= height:
            continue
        cv2.fillConvexPoly(mask, polygon, 1)
    return mask.astype(bool)

def iou(projected, target):
    union = np.logical_or(projected, target).sum()
    return float(np.logical_and(projected, target).sum() / union) if union else 0.0

def optimize_translation(vertices, faces, mask, intrinsics, pose_c0_from_ct):
    pose_ct_from_c0 = np.linalg.inv(pose_c0_from_ct)
    initial = iou(rasterize(transform_points(vertices, pose_ct_from_c0), faces, intrinsics, mask.shape), mask)
    def objective(delta):
        candidate_ct = transform_points(vertices + delta[None, :], pose_ct_from_c0)
        value = iou(rasterize(candidate_ct, faces, intrinsics, mask.shape), mask)
        return -value + 0.015 * float(np.linalg.norm(delta) / 0.03)
    result = minimize(objective, np.zeros(3), method="Nelder-Mead",
                      options={"maxiter": 100, "xatol": 2e-4, "fatol": 1e-4})
    delta = np.clip(result.x, -0.04, 0.04).astype(np.float64)
    optimized = vertices + delta[None, :]
    final = iou(rasterize(transform_points(optimized, pose_ct_from_c0), faces, intrinsics, mask.shape), mask)
    return (optimized, initial, final) if final >= initial else (vertices, initial, initial)

def estimate_metric_scale(vertices_ct, depth, intrinsics, mask):
    z = vertices_ct[:, 2]
    uv_x = np.rint(float(intrinsics["fx"]) * vertices_ct[:, 0] / np.maximum(z, 1e-8) + float(intrinsics["cx"])).astype(int)
    uv_y = np.rint(float(intrinsics["fy"]) * vertices_ct[:, 1] / np.maximum(z, 1e-8) + float(intrinsics["cy"])).astype(int)
    height, width = depth.shape
    valid = ((z > 0) & (uv_x >= 0) & (uv_x < width) & (uv_y >= 0) & (uv_y < height))
    if valid.sum() < 20:
        return 0.04
    valid_indices = np.flatnonzero(valid)
    valid_indices = valid_indices[mask[uv_y[valid], uv_x[valid]]]
    if len(valid_indices) < 20:
        return 0.04
    sampled = depth[uv_y[valid_indices], uv_x[valid_indices]].astype(np.float64)
    good = np.isfinite(sampled) & (sampled > 0.1) & (sampled < 2.0)
    if good.sum() < 10:
        return 0.04
    ratios = sampled[good] / np.maximum(z[valid_indices][good], 1e-6)
    return float(np.clip(np.median(ratios), 0.005, 0.2))

def export_obj(path, vertices, faces):
    path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(path)

def main():
    args = parse_args()
    workspace = args.workspace.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rgb_dir = workspace / "outputs/00_rgb_frames/right_rgb_png"
    with (workspace / "outputs/00_rgb_frames/timeline.csv").open(newline="", encoding="utf-8") as stream:
        timeline = list(csv.DictReader(stream))
    camera = read_json(workspace / "outputs/00_rgb_frames/camera.json")
    intrinsics = camera["rgb_intrinsics_selected"]
    poses = np.load(workspace / "outputs/00_pose_refinement/poses_refined.npz")["T_C0_from_Ct"].astype(np.float64)
    frame_count = min(len(timeline), len(poses))
    if args.max_frames > 0:
        frame_count = min(frame_count, args.max_frames)
    cache_dir = output / "wilor_raw"
    v3d_path, faces_path = cache_dir / "v3d.npy", cache_dir / "faces.npy"
    if not args.skip_inference:
        cache_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            str(args.wilor_python), str(args.wilor_root / "WiLoR_ArtHOI.py"),
            "--img_folder", str(rgb_dir), "--seq_path", str(workspace), "--out_dir", str(cache_dir),
            "--det_conf", str(args.det_conf), "--rescale_factor", str(args.rescale_factor),
            "--batch_size", str(args.batch_size), "--max_frames", str(frame_count),
        ], check=True)
    raw = np.load(v3d_path, allow_pickle=True).item()
    faces = np.load(faces_path).astype(np.int64)
    height, width = int(camera["rgb_height_per_eye"]), int(camera["rgb_width_per_eye"])
    shape = (height, width)
    wilor_focal = 5000.0 / 256.0 * max(width, height)
    output_frames, qc_frames = [], []
    side_counts = {side: 0 for side in SIDES}
    for frame in range(frame_count):
        rgb_path = rgb_dir / f"{frame:06d}.png"
        mask_path = workspace / "outputs/02_hand_masks/combined" / f"{frame:06d}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(mask_path)
        mask_bool = mask > 127
        depth = np.load(workspace / "outputs/06_dense_depth/raw_projected_npy" / f"{frame:06d}.npy")
        record = {"frame": frame, "timestamp_s": float(timeline[frame]["rgb_timestamp_s"]),
                  "rgb_path": str(rgb_path), "sam2_hand_mask_path": str(mask_path),
                  "T_C0_from_Ct": poses[frame].tolist(), "detected_sides": [],
                  "raw_detected_sides": [], "status": "completed"}
        qc_record = {"frame": frame, "sides": {}}
        frame_geometry = {}
        for side in SIDES:
            vertices = np.asarray(raw[f"v3d.{side}"][frame], dtype=np.float64)
            if not np.isfinite(vertices).all():
                continue
            record["raw_detected_sides"].append(side)
            side_counts[side] += 1
            vertices_ct = np.column_stack([
                vertices[:, 0] * wilor_focal / float(intrinsics["fx"]),
                vertices[:, 1] * wilor_focal / float(intrinsics["fy"]),
                vertices[:, 2],
            ])
            scale = estimate_metric_scale(vertices_ct, depth, intrinsics, mask_bool)
            vertices_c0 = transform_points(vertices_ct * scale, poses[frame])
            vertices_c0, initial_iou, final_iou = optimize_translation(vertices_c0, faces, mask_bool, intrinsics, poses[frame])
            joints = vertices_c0[np.linspace(0, len(vertices_c0) - 1, 16).round().astype(int)]
            frame_dir = output / "C0" / f"frame_{frame:06d}"
            hand_path, arm_path = frame_dir / f"{side}_hand_C0.obj", frame_dir / f"{side}_arm_C0.obj"
            geometry_path = frame_dir / f"{side}_geometry_C0.npz"
            export_obj(hand_path, vertices_c0, faces)
            export_obj(arm_path, vertices_c0, faces)
            np.savez_compressed(geometry_path, hand_vertices=vertices_c0.astype(np.float32),
                                hand_joints=joints.astype(np.float32), hand_faces=faces,
                                arm_vertices=vertices_c0.astype(np.float32), arm_faces=faces,
                                scale_ct_to_metric=np.float32(scale), iou_initial=np.float32(initial_iou),
                                iou_final=np.float32(final_iou))
            record[f"{side}_hand_C0"], record[f"{side}_arm_C0"] = str(hand_path), str(arm_path)
            record[f"{side}_geometry_C0_npz"] = str(geometry_path)
            frame_geometry[side] = {"vertices": vertices_c0, "joints": joints}
            record["detected_sides"].append(side)
            qc_record["sides"][side] = {"scale_ct_to_metric": scale, "iou_initial": initial_iou,
                                         "iou_final": final_iou, "iou_gain": final_iou - initial_iou}
        if frame_geometry:
            template_vertices = next(iter(frame_geometry.values()))["vertices"]
            template_joints = next(iter(frame_geometry.values()))["joints"]
            combined_vertices = np.stack([frame_geometry.get(side, {"vertices": np.full_like(template_vertices, np.nan)})["vertices"] for side in SIDES])
            combined_joints = np.stack([frame_geometry.get(side, {"joints": np.full_like(template_joints, np.nan)})["joints"] for side in SIDES])
            combined_path = output / "C0" / f"frame_{frame:06d}" / "wilor_geometry_C0.npz"
            np.savez_compressed(combined_path, hand_vertices=combined_vertices.astype(np.float32),
                                hand_joints=combined_joints.astype(np.float32),
                                left_hand_faces=faces, right_hand_faces=faces,
                                arm_vertices=combined_vertices.astype(np.float32), arm_faces=faces,
                                raw_visible_hand=np.asarray([side in frame_geometry for side in SIDES], dtype=bool))
            record["geometry_C0_npz"] = str(combined_path)
        output_frames.append(record)
        qc_frames.append(qc_record)
    manifest = {
        "schema_version": 1, "type": "wilor_pose_compensated_sequence_iou_optimized",
        "candidate_policy": "WiLoR_detection_then_metric_depth_scale_then_C0_then_SAM2_mask_IoU_translation",
        "frame_count": frame_count, "detected_frame_count": sum(bool(x["detected_sides"]) for x in output_frames),
        "side_detected_frame_counts": side_counts, "coordinate_frame": "frame0_right_camera_opencv_rdf",
        "raw_coordinate_frame": "current_right_camera_opencv_rdf_weak_perspective",
        "transform_rule": "WiLoR weak-perspective camera -> selected RGB intrinsics, metric depth scale, then T_C0_from_Ct",
        "camera_json": str((workspace / "outputs/00_rgb_frames/camera.json").resolve()),
        "pose_source": str((workspace / "outputs/00_pose_refinement/poses_refined.npz").resolve()),
        "sam2_consistency_filter": "mask_IoU_translation_optimization",
        "source_wilor_v3d": str(v3d_path.resolve()), "source_wilor_faces": str(faces_path.resolve()),
        "frames": output_frames,
    }
    write_json(output / "dynamic_manifest.json", manifest)
    write_json(output / "wilor_iou_qc.json", {"status": "completed", "frame_count": frame_count,
              "wilor_focal_px": wilor_focal, "intrinsics": intrinsics, "frames": qc_frames,
              "side_detected_frame_counts": side_counts})
    print(json.dumps({"manifest": str(output / "dynamic_manifest.json"), "frames": frame_count,
                      "side_counts": side_counts}))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
