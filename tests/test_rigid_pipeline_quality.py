from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from vlm_sam2_recon.rigid_pipeline.calibration_validation import normalized_sensor_extrinsics
from vlm_sam2_recon.rigid_pipeline.depth_fusion import evaluate_metric_depth
from vlm_sam2_recon.rigid_pipeline.mesh_alignment import (
    Similarity,
    fixed_scale_point_to_plane_icp,
)


def test_normalized_sensor_extrinsics_inverts_head_to_sensor() -> None:
    raw = np.eye(4)
    raw[:3, 3] = [0.1, -0.2, 0.3]
    assert np.allclose(normalized_sensor_extrinsics(raw, "sensor_to_head"), raw)
    assert np.allclose(normalized_sensor_extrinsics(raw, "head_to_sensor"), np.linalg.inv(raw))


def test_metric_depth_evaluation_reports_coverage_and_error() -> None:
    true = np.asarray([[1.0, 2.0, 0.0], [1.5, 2.5, 3.0]])
    estimated = np.asarray([[1.1, 1.9, 4.0], [1.5, 0.0, 3.2]])
    result = evaluate_metric_depth(estimated, true, depth_min_m=0.1, depth_max_m=5.0)
    assert result["true_valid_count"] == 5
    assert result["valid_count"] == 4
    assert np.isclose(result["valid_ratio"], 0.8)
    assert result["rmse_m"] > 0


def test_point_to_plane_icp_reduces_synthetic_pose_error_without_scale_drift() -> None:
    rng = np.random.default_rng(7)
    # Three non-coplanar faces provide constraints for all six pose axes.
    xy = rng.uniform(-0.2, 0.2, size=(900, 2))
    yz = rng.uniform(-0.2, 0.2, size=(900, 2))
    xz = rng.uniform(-0.2, 0.2, size=(900, 2))
    source = np.vstack(
        [
            np.column_stack([xy, np.full(len(xy), 0.2)]),
            np.column_stack([np.full(len(yz), 0.2), yz]),
            np.column_stack([xz[:, 0], np.full(len(xz), -0.2), xz[:, 1]]),
        ]
    )
    true_rotation = Rotation.from_rotvec([0.025, -0.035, 0.02]).as_matrix()
    true_translation = np.asarray([0.018, -0.012, 0.025])
    target = source @ true_rotation.T + true_translation
    initial = Similarity(
        scale=1.0,
        rotation=Rotation.from_rotvec([0.015, -0.02, 0.01]).as_matrix(),
        translation=np.asarray([0.01, -0.006, 0.014]),
    )
    before = np.mean(np.linalg.norm(initial.transform(source) - target, axis=1))
    result, history = fixed_scale_point_to_plane_icp(
        source,
        target,
        initial,
        max_distances_m=(0.08, 0.04, 0.02),
        iterations_per_level=10,
    )
    after = np.mean(np.linalg.norm(result.transform(source) - target, axis=1))
    assert history
    assert np.isclose(result.scale, 1.0)
    assert after < before * 0.35
