#!/usr/bin/env python3
"""Jointly refine WiLoR hand geometry with SAM2 IoU and native metric depth."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import cv2
import numpy as np
import trimesh
from scipy.optimize import minimize

SIDES = ("left", "right")

def args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--input-manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--iou-weight", type=float, default=2.0)
    p.add_argument("--depth-weight", type=float, default=1.0)
    p.add_argument("--depth-scale-m", type=float, default=0.025)
    p.add_argument("--max-frames", type=int, default=0)
    return p.parse_args()

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def transform(points, matrix):
    h = np.concatenate([points, np.ones((len(points), 1), dtype=points.dtype)], axis=1)
    return (h @ matrix.T)[:, :3]

def rasterize(vertices, faces, K, shape):
    h, w = shape
    z = vertices[:, 2]
    valid = z > 0.03
    uv = np.column_stack([
        K["fx"] * vertices[:, 0] / np.maximum(z, 1e-8) + K["cx"],
        K["fy"] * vertices[:, 1] / np.maximum(z, 1e-8) + K["cy"],
    ])
    output = np.zeros((h, w), np.uint8)
    for face in faces:
        if not valid[face].all():
            continue
        poly = np.rint(uv[face]).astype(np.int32)
        if poly[:, 0].max() < 0 or poly[:, 0].min() >= w or poly[:, 1].max() < 0 or poly[:, 1].min() >= h:
            continue
        cv2.fillConvexPoly(output, poly, 1)
    return output.astype(bool)

def mask_iou(projected, target):
    union = np.logical_or(projected, target).sum()
    return float(np.logical_and(projected, target).sum() / union) if union else 0.0

def depth_residual(vertices_ct, depth, K, hand_mask):
    z = vertices_ct[:, 2]
    x = np.rint(K["fx"] * vertices_ct[:, 0] / np.maximum(z, 1e-8) + K["cx"]).astype(int)
    y = np.rint(K["fy"] * vertices_ct[:, 1] / np.maximum(z, 1e-8) + K["cy"]).astype(int)
    h, w = depth.shape
    valid = (z > 0.05) & (x >= 0) & (x < w) & (y >= 0) & (y < h)
    if valid.sum() == 0:
        return 1.0, np.nan, 0
    ids = np.flatnonzero(valid)
    ids = ids[hand_mask[y[ids], x[ids]]]
    if len(ids) == 0:
        return 1.0, np.nan, 0
    observed = depth[y[ids], x[ids]].astype(np.float64)
    good = np.isfinite(observed) & (observed > 0.05) & (observed < 3.0)
    if good.sum() < 10:
        return 1.0, np.nan, int(good.sum())
    residual = z[ids][good] - observed[good]
    # Median absolute error is robust to sparse projected-depth collisions.
    return float(np.clip(np.median(np.abs(residual)) / 0.025, 0.0, 4.0)), float(np.median(residual)), int(good.sum())

def export_obj(path, vertices, faces):
    path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(path)

def main():
    a = args()
    workspace = a.workspace.resolve()
    source = read_json(a.input_manifest.resolve())
    output = a.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    camera = read_json(workspace / "outputs/00_rgb_frames/camera.json")
    K = camera["rgb_intrinsics_selected"]
    shape = (int(camera["rgb_height_per_eye"]), int(camera["rgb_width_per_eye"]))
    poses = np.load(workspace / "outputs/00_pose_refinement/poses_refined.npz")["T_C0_from_Ct"].astype(np.float64)
    with (workspace / "outputs/00_rgb_frames/timeline.csv").open(newline="", encoding="utf-8") as stream:
        timeline = list(csv.DictReader(stream))
    count = min(len(source["frames"]), len(poses), len(timeline))
    if a.max_frames > 0:
        count = min(count, a.max_frames)
    frames, qc = [], []
    side_counts = {side: 0 for side in SIDES}
    for frame in range(count):
        src = source["frames"][frame]
        geometry = np.load(src["geometry_C0_npz"])
        base_vertices = geometry["hand_vertices"].astype(np.float64)
        faces = {"left": geometry["left_hand_faces"].astype(np.int64), "right": geometry["right_hand_faces"].astype(np.int64)}
        mask_path = Path(src["sam2_hand_mask_path"])
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(mask_path)
        mask = mask > 127
        depth = np.load(workspace / "outputs/06_dense_depth/raw_projected_npy" / f"{frame:06d}.npy").astype(np.float32)
        pose_ct_from_c0 = np.linalg.inv(poses[frame])
        record = dict(src)
        record["frame"] = frame
        record["timestamp_s"] = float(timeline[frame]["rgb_timestamp_s"])
        record["T_C0_from_Ct"] = poses[frame].tolist()
        record["detected_sides"] = []
        frame_qc = {"frame": frame, "sides": {}}
        refined = {}
        for index, side in enumerate(SIDES):
            vertices = base_vertices[index]
            if not np.isfinite(vertices).all():
                continue
            side_faces = faces[side]
            centroid = vertices.mean(axis=0)
            def candidate(params):
                delta = np.asarray(params[:3], dtype=np.float64)
                scale = float(params[3])
                return centroid[None, :] + scale * (vertices - centroid[None, :]) + delta[None, :]
            def metrics(params):
                c0 = candidate(params)
                ct = transform(c0, pose_ct_from_c0)
                projected = rasterize(ct, side_faces, K, shape)
                score_iou = mask_iou(projected, mask)
                score_depth, bias, point_count = depth_residual(ct, depth, K, mask)
                return score_iou, score_depth, bias, point_count
            initial = metrics([0.0, 0.0, 0.0, 1.0])
            def objective(params):
                score_iou, score_depth, _, _ = metrics(params)
                delta_norm = np.linalg.norm(np.asarray(params[:3])) / 0.03
                scale_penalty = abs(float(params[3]) - 1.0)
                return -a.iou_weight * score_iou + a.depth_weight * score_depth + 0.015 * delta_norm + 0.03 * scale_penalty
            result = minimize(objective, np.array([0.0, 0.0, 0.0, 1.0]), method="Powell",
                              bounds=[(-0.035, 0.035), (-0.035, 0.035), (-0.035, 0.035), (0.82, 1.18)],
                              options={"maxiter": 70, "xtol": 2e-3, "ftol": 2e-3})
            final = metrics(result.x)
            if final[0] + 1e-5 < initial[0] and final[1] > initial[1]:
                result.x = np.array([0.0, 0.0, 0.0, 1.0])
                final = initial
            vertices_out = candidate(result.x)
            refined[side] = {"vertices": vertices_out, "joints": vertices_out[np.linspace(0, len(vertices_out) - 1, 16).round().astype(int)]}
            side_counts[side] += 1
            record["detected_sides"].append(side)
            frame_qc["sides"][side] = {
                "initial_iou": initial[0], "final_iou": final[0], "iou_gain": final[0] - initial[0],
                "initial_depth_loss": initial[1], "final_depth_loss": final[1],
                "initial_depth_bias_m": initial[2], "final_depth_bias_m": final[2],
                "depth_point_count": final[3], "delta_C0_m": np.asarray(result.x[:3]).tolist(),
                "scale_factor": float(result.x[3]), "objective": float(result.fun),
            }
        if refined:
            template = next(iter(refined.values()))
            empty_v = np.full_like(template["vertices"], np.nan)
            empty_j = np.full_like(template["joints"], np.nan)
            combined_v = np.stack([refined.get(side, {"vertices": empty_v})["vertices"] for side in SIDES])
            combined_j = np.stack([refined.get(side, {"joints": empty_j})["joints"] for side in SIDES])
            frame_dir = output / "C0" / f"frame_{frame:06d}"
            frame_dir.mkdir(parents=True, exist_ok=True)
            combined_path = frame_dir / "wilor_depth_iou_geometry_C0.npz"
            np.savez_compressed(combined_path, hand_vertices=combined_v.astype(np.float32), hand_joints=combined_j.astype(np.float32),
                                left_hand_faces=faces["left"], right_hand_faces=faces["right"], arm_vertices=combined_v.astype(np.float32),
                                arm_faces=faces["right"], raw_visible_hand=np.asarray([side in refined for side in SIDES]))
            for side in SIDES:
                if side not in refined:
                    continue
                hand_path = frame_dir / f"{side}_hand_C0.obj"
                arm_path = frame_dir / f"{side}_arm_C0.obj"
                export_obj(hand_path, refined[side]["vertices"], faces[side])
                export_obj(arm_path, refined[side]["vertices"], faces[side])
                record[f"{side}_hand_C0"] = str(hand_path)
                record[f"{side}_arm_C0"] = str(arm_path)
            record["geometry_C0_npz"] = str(combined_path)
        frames.append(record)
        qc.append(frame_qc)
    manifest = dict(source)
    manifest.update({
        "type": "wilor_pose_compensated_sequence_depth_iou_optimized",
        "candidate_policy": "WiLoR C0 geometry jointly refined by SAM2 mask IoU and native metric depth",
        "optimization": {"iou_weight": a.iou_weight, "depth_weight": a.depth_weight, "depth_scale_m": a.depth_scale_m,
                         "optimized_parameters": "C0 translation plus isotropic hand scale", "object_trajectory_changed": False},
        "source_hand_manifest": str(a.input_manifest.resolve()), "frames": frames,
        "side_detected_frame_counts": side_counts,
    })
    write_json(output / "dynamic_manifest.json", manifest)
    write_json(output / "wilor_depth_iou_qc.json", {"status": "completed", "frame_count": count,
              "optimization": manifest["optimization"], "side_detected_frame_counts": side_counts, "frames": qc})
    print(json.dumps({"manifest": str(output / "dynamic_manifest.json"), "frames": count, "side_counts": side_counts}))

if __name__ == "__main__":
    main()
