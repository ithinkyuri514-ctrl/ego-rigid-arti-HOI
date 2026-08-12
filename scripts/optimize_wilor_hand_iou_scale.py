#!/usr/bin/env python3
"""Refine WiLoR C0 hand geometry with SAM2 IoU, translation, and scale."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import cv2
import numpy as np
import trimesh
from scipy.optimize import minimize

SIDES = ("left", "right")

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--input-manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--scale-min", type=float, default=0.75)
    p.add_argument("--scale-max", type=float, default=1.25)
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

def iou(projected, target):
    union = np.logical_or(projected, target).sum()
    return float(np.logical_and(projected, target).sum() / union) if union else 0.0

def export_obj(path, vertices, faces):
    path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(path)

def main():
    a = parse_args()
    workspace = a.workspace.resolve()
    source = read_json(a.input_manifest.resolve())
    output = a.output_dir.resolve()
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
        faces = {side: geometry[f"{side}_hand_faces"].astype(np.int64) for side in SIDES}
        mask_path = Path(src["sam2_hand_mask_path"])
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(mask_path)
        mask = mask > 127
        pose_ct_from_c0 = np.linalg.inv(poses[frame])
        record = dict(src)
        record.update({"frame": frame, "timestamp_s": float(timeline[frame]["rgb_timestamp_s"]),
                       "T_C0_from_Ct": poses[frame].tolist(), "detected_sides": []})
        frame_qc = {"frame": frame, "sides": {}}
        refined = {}
        for index, side in enumerate(SIDES):
            vertices = base_vertices[index]
            if not np.isfinite(vertices).all():
                continue
            side_faces = faces[side]
            center = vertices.mean(axis=0)
            def candidate(params):
                delta = np.asarray(params[:3], dtype=np.float64)
                scale = float(params[3])
                return center[None, :] + scale * (vertices - center[None, :]) + delta[None, :]
            def score(params):
                c0 = candidate(params)
                ct = transform(c0, pose_ct_from_c0)
                return iou(rasterize(ct, side_faces, K, shape), mask)
            initial = score([0.0, 0.0, 0.0, 1.0])
            def objective(params):
                value = score(params)
                delta_penalty = 0.01 * np.linalg.norm(np.asarray(params[:3])) / 0.03
                scale_penalty = 0.01 * abs(float(params[3]) - 1.0)
                return -value + delta_penalty + scale_penalty
            result = minimize(objective, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64), method="Powell",
                              bounds=[(-0.04, 0.04), (-0.04, 0.04), (-0.04, 0.04), (a.scale_min, a.scale_max)],
                              options={"maxiter": 100, "xtol": 1e-3, "ftol": 1e-4})
            params = np.asarray(result.x, dtype=np.float64)
            final = score(params)
            if final < initial:
                params = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
                final = initial
            vertices_out = candidate(params)
            joints = vertices_out[np.linspace(0, len(vertices_out) - 1, 16).round().astype(int)]
            refined[side] = {"vertices": vertices_out, "joints": joints}
            side_counts[side] += 1
            record["detected_sides"].append(side)
            frame_qc["sides"][side] = {"initial_iou": initial, "final_iou": final,
                                         "iou_gain": final - initial, "delta_C0_m": params[:3].tolist(),
                                         "scale_factor": float(params[3]), "objective": float(objective(params))}
        if refined:
            template = next(iter(refined.values()))
            empty_v = np.full_like(template["vertices"], np.nan)
            empty_j = np.full_like(template["joints"], np.nan)
            combined_v = np.stack([refined.get(side, {"vertices": empty_v})["vertices"] for side in SIDES])
            combined_j = np.stack([refined.get(side, {"joints": empty_j})["joints"] for side in SIDES])
            frame_dir = output / "C0" / f"frame_{frame:06d}"
            frame_dir.mkdir(parents=True, exist_ok=True)
            combined_path = frame_dir / "wilor_iou_scale_geometry_C0.npz"
            np.savez_compressed(combined_path, hand_vertices=combined_v.astype(np.float32), hand_joints=combined_j.astype(np.float32),
                                left_hand_faces=faces["left"], right_hand_faces=faces["right"],
                                arm_vertices=combined_v.astype(np.float32), arm_faces=faces["right"],
                                raw_visible_hand=np.asarray([side in refined for side in SIDES]))
            for side in SIDES:
                if side not in refined:
                    continue
                hand_path = frame_dir / f"{side}_hand_C0.obj"
                arm_path = frame_dir / f"{side}_arm_C0.obj"
                export_obj(hand_path, refined[side]["vertices"], faces[side])
                export_obj(arm_path, refined[side]["vertices"], faces[side])
                record[f"{side}_hand_C0"], record[f"{side}_arm_C0"] = str(hand_path), str(arm_path)
            record["geometry_C0_npz"] = str(combined_path)
        frames.append(record)
        qc.append(frame_qc)
    manifest = dict(source)
    manifest.update({"type": "wilor_pose_compensated_sequence_iou_scale_optimized",
                     "candidate_policy": "WiLoR C0 geometry refined by SAM2 mask IoU with C0 translation and isotropic scale",
                     "optimization": {"objective": "SAM2 mask IoU", "optimized_parameters": "C0 translation plus isotropic hand scale",
                                      "scale_bounds": [a.scale_min, a.scale_max], "depth_used": False},
                     "source_hand_manifest": str(a.input_manifest.resolve()), "frames": frames,
                     "side_detected_frame_counts": side_counts})
    write_json(output / "dynamic_manifest.json", manifest)
    write_json(output / "wilor_iou_scale_qc.json", {"status": "completed", "frame_count": count,
              "optimization": manifest["optimization"], "side_detected_frame_counts": side_counts, "frames": qc})
    print(json.dumps({"manifest": str(output / "dynamic_manifest.json"), "frames": count, "side_counts": side_counts}))

if __name__ == "__main__":
    main()
