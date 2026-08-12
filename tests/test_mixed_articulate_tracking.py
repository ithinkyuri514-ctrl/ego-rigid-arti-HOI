from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from scripts.mixed_stage10_track_articulate_parts import (
    axis_rotation_transform,
    estimate_fixed_axis_sequence,
    estimate_interaction_axis_sequence,
    estimate_single_track_axis_sequence,
    lift_tracks_with_previous_depth_fallback,
    transform_per_frame_points_to_c0,
)


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def test_fixed_axis_fit_recovers_nonzero_origin_rotation() -> None:
    origin = np.asarray([0.25, -0.15, 0.8], dtype=np.float64)
    axis = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    local = origin + np.asarray(
        [
            [0.10, -0.05, 0.00],
            [0.12, 0.00, 0.08],
            [-0.09, 0.03, 0.10],
            [-0.11, -0.02, -0.06],
            [0.06, 0.06, -0.12],
            [-0.08, -0.04, 0.13],
        ],
        dtype=np.float64,
    )
    expected_angles = np.asarray([0.0, 0.2, 0.55, 0.9], dtype=np.float64)
    expected_transforms = np.stack(
        [axis_rotation_transform(origin, axis, angle) for angle in expected_angles]
    )
    observed = np.stack([transform_points(local, transform) for transform in expected_transforms])
    valid = np.ones(observed.shape[:2], dtype=bool)
    confidence = np.ones(observed.shape[:2], dtype=np.float64)

    # One geometrically inconsistent point must not pull the one-DoF result away.
    observed[2, 0] += np.asarray([0.4, 0.3, -0.2])
    raw_delta = expected_transforms.copy()
    angles, transforms, diagnostics = estimate_fixed_axis_sequence(
        local,
        observed,
        valid,
        confidence,
        raw_delta,
        origin,
        axis,
        min_axis_radius_m=0.01,
        max_axis_coordinate_error_m=0.05,
        max_radius_error_m=0.05,
        max_angle_residual_deg=10.0,
        min_angle_points=3,
    )

    np.testing.assert_allclose(angles, expected_angles, atol=1e-6)
    np.testing.assert_allclose(transforms, expected_transforms, atol=1e-6)
    np.testing.assert_allclose(transform_points(origin[None], transforms[3]), origin[None], atol=1e-9)
    assert diagnostics[2]["status"] == "rgbd_axis_fit"


def test_fixed_axis_fit_falls_back_to_projected_raw_rotation() -> None:
    origin = np.zeros(3, dtype=np.float64)
    axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    local = np.asarray([[0.1, 0.0, 1.0]], dtype=np.float64)
    raw = np.stack(
        [axis_rotation_transform(origin, axis, angle) for angle in (0.0, -0.3, -0.7)]
    )
    observed = np.full((3, 1, 3), np.nan, dtype=np.float64)
    valid = np.zeros((3, 1), dtype=bool)
    confidence = np.zeros((3, 1), dtype=np.float64)
    angles, _, diagnostics = estimate_fixed_axis_sequence(
        local,
        observed,
        valid,
        confidence,
        raw,
        origin,
        axis,
        min_axis_radius_m=0.01,
        max_axis_coordinate_error_m=0.05,
        max_radius_error_m=0.05,
        max_angle_residual_deg=10.0,
        min_angle_points=3,
    )

    np.testing.assert_allclose(angles, [0.0, -0.3, -0.7], atol=1e-6)
    assert diagnostics[1]["status"] == "raw_rotation_axis_fallback"


def test_interaction_fit_uses_each_frames_depth_with_pose_compensation() -> None:
    origin = np.asarray([0.2, -0.1, 1.1], dtype=np.float64)
    axis = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    reference_points = origin + np.asarray(
        [
            [0.16, -0.04, 0.02],
            [0.13, 0.03, 0.11],
            [-0.12, -0.02, 0.14],
            [-0.15, 0.05, -0.08],
        ],
        dtype=np.float64,
    )
    expected_angles = np.asarray([0.0, 0.18, 0.43, 0.72], dtype=np.float64)
    observed_c0 = np.stack(
        [
            transform_points(reference_points, axis_rotation_transform(origin, axis, angle))
            for angle in expected_angles
        ]
    )

    transforms_c0_from_ct = np.repeat(np.eye(4, dtype=np.float64)[None], 4, axis=0)
    transforms_c0_from_ct[1, :3, :3] = Rotation.from_euler("z", 8.0, degrees=True).as_matrix()
    transforms_c0_from_ct[1, :3, 3] = [0.08, -0.03, 0.01]
    transforms_c0_from_ct[2, :3, :3] = Rotation.from_euler("x", -6.0, degrees=True).as_matrix()
    transforms_c0_from_ct[2, :3, 3] = [0.14, 0.02, -0.04]
    transforms_c0_from_ct[3, :3, :3] = Rotation.from_euler("zy", [10.0, 5.0], degrees=True).as_matrix()
    transforms_c0_from_ct[3, :3, 3] = [0.21, -0.05, 0.03]

    observed_ct = np.empty_like(observed_c0)
    for frame, transform in enumerate(transforms_c0_from_ct):
        inverse = np.linalg.inv(transform)
        observed_ct[frame] = transform_points(observed_c0[frame], inverse)

    pose_corrected = transform_per_frame_points_to_c0(observed_ct, transforms_c0_from_ct)
    np.testing.assert_allclose(pose_corrected, observed_c0, atol=1e-10)

    confidence = np.ones(pose_corrected.shape[:2], dtype=np.float64)
    selected_by_start = {0: np.arange(len(reference_points), dtype=np.int64)}
    intervals = [{"start_frame": 0, "end_frame": 3, "event_id": "open", "action": "open"}]
    angles, _, diagnostics = estimate_interaction_axis_sequence(
        pose_corrected,
        confidence,
        selected_by_start,
        intervals,
        origin,
        axis,
        min_axis_radius_m=0.01,
        max_axis_coordinate_error_m=0.01,
        max_radius_error_m=0.01,
        max_angle_residual_deg=5.0,
        min_angle_points=3,
        angle_sign=1.0,
    )

    np.testing.assert_allclose(angles, expected_angles, atol=1e-8)
    assert diagnostics[3]["status"] == "rgbd_axis_fit_interaction_relative"


def test_single_track_lift_reuses_previous_depth_at_current_pixel() -> None:
    tracks_xy = np.asarray([[[10.0, 20.0]], [[12.0, 24.0]], [[15.0, 30.0]]])
    sampled_depth = np.asarray([[2.0], [np.nan], [3.0]])
    query_times = np.asarray([0])
    transforms = np.repeat(np.eye(4, dtype=np.float64)[None], 3, axis=0)
    transforms[1, :3, 3] = [0.5, -0.25, 0.1]
    intrinsics = {"fx": 10.0, "fy": 20.0, "cx": 5.0, "cy": 10.0}

    points_ct, points_c0, used_depth, depth_source = lift_tracks_with_previous_depth_fallback(
        tracks_xy,
        sampled_depth,
        query_times,
        transforms,
        intrinsics,
    )

    np.testing.assert_allclose(used_depth[:, 0], [2.0, 2.0, 3.0])
    np.testing.assert_array_equal(depth_source[:, 0], [1, 2, 1])
    np.testing.assert_allclose(points_ct[1, 0], [1.4, 1.4, 2.0])
    np.testing.assert_allclose(points_c0[1, 0], [1.9, 1.15, 2.1])


def test_single_track_axis_angle_reports_previous_depth_fallback() -> None:
    origin = np.asarray([0.1, -0.2, 0.8], dtype=np.float64)
    axis = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    reference = origin + np.asarray([0.2, 0.03, 0.08])
    expected_angles = np.asarray([0.0, -0.35, -0.7])
    observed = np.full((3, 1, 3), np.nan, dtype=np.float64)
    for frame, angle in enumerate(expected_angles):
        observed[frame, 0] = transform_points(
            reference[None], axis_rotation_transform(origin, axis, angle)
        )[0]
    confidence = np.ones((3, 1), dtype=np.float64)
    used_depth = np.asarray([[1.0], [1.0], [0.9]])
    depth_source = np.asarray([[1], [2], [1]], dtype=np.uint8)

    angles, _, diagnostics = estimate_single_track_axis_sequence(
        observed,
        confidence,
        used_depth,
        depth_source,
        {0: np.asarray([0], dtype=np.int64)},
        [{"start_frame": 0, "end_frame": 2, "action": "close"}],
        origin,
        axis,
    )

    np.testing.assert_allclose(angles, expected_angles, atol=1e-8)
    assert diagnostics[1]["status"] == "single_upper_left_track_previous_depth_fallback"
