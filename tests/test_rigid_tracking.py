import numpy as np

from vlm_sam2_recon.rigid_pipeline.rigid_tracking import (
    estimate_pairwise_pose_sequence,
    identify_bad_pairwise_tracks,
    kabsch_se3,
    ransac_se3,
    reject_temporal_spikes,
    transform_points,
)


def test_ransac_se3_rejects_large_3d_outliers():
    rng = np.random.default_rng(5)
    source = rng.normal(size=(40, 3)) * 0.08
    angle = 0.24
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]],
        dtype=np.float64,
    )
    expected = np.eye(4)
    expected[:3, :3] = rotation
    expected[:3, 3] = (0.04, -0.02, 0.015)
    target = transform_points(source, expected)
    target[:8] += rng.normal(size=(8, 3)) * 0.3
    result = ransac_se3(source, target, threshold_m=0.01, iterations=300, min_inliers=20, seed=9)
    assert result is not None
    assert result.inliers.sum() == 32
    assert np.allclose(result.matrix, expected, atol=1e-6)


def test_kabsch_keeps_unit_scale_and_proper_rotation():
    source = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    target = source + np.asarray([0.2, -0.1, 0.4])
    result = kabsch_se3(source, target)
    assert np.isclose(np.linalg.det(result[:3, :3]), 1.0)
    assert np.allclose(transform_points(source, result), target)


def test_temporal_spike_filter_removes_isolated_depth_jump():
    xyz = np.zeros((5, 1, 3), dtype=float)
    xyz[:, 0, 2] = [0.5, 0.51, 1.2, 0.52, 0.53]
    depths = xyz[..., 2].copy()
    rejection = np.zeros((5, 1), dtype=np.uint8)
    filtered = reject_temporal_spikes(xyz, depths, rejection, np.asarray([0]))
    assert filtered[2, 0] == 6
    assert np.count_nonzero(filtered) == 1


def test_pairwise_pose_sequence_accumulates_small_rigid_steps():
    rng = np.random.default_rng(8)
    reference = rng.normal(size=(30, 3)) * 0.05
    points = np.empty((5, 30, 3), dtype=float)
    expected = np.repeat(np.eye(4)[None], 5, axis=0)
    points[0] = reference
    for frame in range(1, 5):
        step = np.eye(4)
        step[:3, 3] = (0.01, -0.002, 0.0)
        expected[frame] = step @ expected[frame - 1]
        points[frame] = transform_points(reference, expected[frame])
    poses, inliers, diagnostics = estimate_pairwise_pose_sequence(
        points,
        np.zeros((5, 30), dtype=np.uint8),
        ransac_threshold_m=1e-4,
        min_inliers=10,
    )
    assert all(item["status"] == "completed" for item in diagnostics)
    assert np.allclose(poses, expected, atol=1e-7)
    assert inliers[1:].all()


def test_pairwise_motion_gate_uses_object_center_not_world_origin():
    rng = np.random.default_rng(19)
    reference = rng.normal(size=(40, 3)) * 0.04 + np.asarray([0.0, 0.0, 0.8])
    angle = np.deg2rad(20.0)
    rotation = np.asarray(
        [[np.cos(angle), 0.0, np.sin(angle)], [0.0, 1.0, 0.0], [-np.sin(angle), 0.0, np.cos(angle)]],
        dtype=np.float64,
    )
    center = reference.mean(axis=0)
    step = np.eye(4)
    step[:3, :3] = rotation
    step[:3, 3] = center - rotation @ center
    points = np.stack([reference, transform_points(reference, step)])

    poses, _, diagnostics = estimate_pairwise_pose_sequence(
        points,
        np.zeros((2, len(reference)), dtype=np.uint8),
        ransac_threshold_m=1e-4,
        min_inliers=10,
        max_step_translation_m=0.05,
    )

    assert diagnostics[1]["status"] == "completed"
    assert diagnostics[1]["translation_step_m"] < 1e-6
    assert diagnostics[1]["transform_translation_norm_m"] > 0.2
    assert np.allclose(poses[1], step, atol=1e-7)


def test_final_frame_anchor_is_not_globally_rejected_without_future_pairs():
    points = np.zeros((3, 4, 3), dtype=np.float64)
    rejection = np.ones((3, 4), dtype=np.uint8)
    rejection[2] = 0
    diagnostics = [
        {"status": "completed"},
        {"status": "completed"},
        {"status": "completed"},
    ]

    rejected, records = identify_bad_pairwise_tracks(
        points,
        rejection,
        np.repeat(np.eye(4)[None], 3, axis=0),
        diagnostics,
        np.full(4, 2, dtype=np.int64),
    )

    assert not rejected.any()
    assert all(record["active_pairs"] == 0 for record in records)
    assert all(record["valid_pair_ratio"] == 1.0 for record in records)
