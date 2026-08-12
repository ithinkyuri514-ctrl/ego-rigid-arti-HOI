import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from vlm_sam2_recon.stages.screen_hinge_tracking import (
    DepthFrameSampler,
    ScreenHingeTrackingConfig,
    estimate_three_point_direct_theta,
    huber_loss,
    mad_inlier_mask,
    select_good_features,
    signed_axis_angle,
)


class ScreenHingeTrackingTest(unittest.TestCase):
    def test_good_features_stay_inside_eroded_mask(self):
        rgb = np.zeros((120, 160, 3), dtype=np.uint8)
        mask = np.zeros((120, 160), dtype=bool)
        mask[20:100, 30:140] = True
        for y in range(28, 96, 12):
            for x in range(38, 136, 12):
                rgb[y - 2 : y + 3, x - 2 : x + 3] = 255
        pts = select_good_features(rgb, mask, max_points=30, min_distance_px=6, quality_level=0.01, erode_px=5)
        self.assertGreaterEqual(len(pts), 10)
        for x, y in pts:
            self.assertTrue(mask[int(round(y)), int(round(x))])
            self.assertGreaterEqual(x, 35)
            self.assertLessEqual(x, 134)
            self.assertGreaterEqual(y, 25)
            self.assertLessEqual(y, 94)

    def test_aligned_depth_sampler_uses_neighbor_median(self):
        meta = {
            "rgb_intrinsics_right": {"fx": 100.0, "fy": 100.0, "cx": 10.0, "cy": 10.0},
            "rgb_width_per_eye": 20,
            "rgb_height_per_eye": 20,
        }
        depth = np.ones((20, 20), dtype=np.float32)
        depth[9:12, 9:12] = np.asarray([[1.0, 2.0, 1.0], [2.0, 2.0, 2.0], [1.0, 2.0, 1.0]], dtype=np.float32)
        sampler = DepthFrameSampler(meta, depth, "direct_same_camera", (20, 20), "aligned_rgb", 0.1, 3.0)
        sample = sampler.sample(np.asarray([10.0, 10.0]), radius_px=1)
        self.assertIsNotNone(sample)
        assert sample is not None
        self.assertAlmostEqual(sample.depth_m, 2.0, places=5)
        np.testing.assert_allclose(sample.point_frame, np.asarray([0.0, 0.0, 2.0]), atol=1e-6)

    def test_robust_helpers(self):
        values = np.asarray([0.0, 0.1, 0.2, 0.1, 20.0], dtype=np.float64)
        inliers = mad_inlier_mask(values, sigma=3.0)
        self.assertFalse(bool(inliers[-1]))
        losses = huber_loss(np.asarray([0.0, 1.0, 3.0]), delta=1.0)
        np.testing.assert_allclose(losses, np.asarray([0.0, 0.5, 2.5]))

    def test_three_point_direct_theta_uses_consistent_votes(self):
        origin = np.zeros(3, dtype=np.float64)
        axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        theta = np.deg2rad(35.0)
        refs = np.asarray([[1.0, 0.0, 0.0], [1.2, 0.2, 0.0], [0.9, -0.3, 0.0]], dtype=np.float64)
        rot = np.asarray(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        observed = refs @ rot.T
        votes = np.asarray([signed_axis_angle(refs[i], observed[i], origin, axis) for i in range(3)], dtype=np.float64)
        obs = {
            "ids": np.asarray([4, 7, 9], dtype=np.int64),
            "per_point_theta": votes,
            "per_point_delta": votes,
            "confidence": np.asarray([0.95, 0.9, 0.93], dtype=np.float64),
            "axis_radius": np.asarray([1.0, 1.1, 0.95], dtype=np.float64),
            "reproj_pred": np.asarray([1.0, 2.0, 1.5], dtype=np.float64),
            "plane_dist": np.asarray([0.003, 0.002, 0.004], dtype=np.float64),
            "depth_residual": np.asarray([0.004, 0.005, 0.004], dtype=np.float64),
        }
        config = ScreenHingeTrackingConfig(angle_method="three_point_direct", three_point_count=3, angle_min_deg=-90.0, angle_max_deg=90.0)
        got, _, diag = estimate_three_point_direct_theta(obs, 0.0, 0.0, 0.0, 60.0, config)
        self.assertEqual(diag["used_points"], 3)
        self.assertAlmostEqual(np.rad2deg(got), 35.0, places=4)
        self.assertEqual(diag["mode"], "incremental_delta")

    def test_three_point_increment_adds_to_previous_angle(self):
        obs = {
            "ids": np.asarray([0, 1, 2], dtype=np.int64),
            "per_point_theta": np.deg2rad(np.asarray([50.0, 51.0, 49.0], dtype=np.float64)),
            "per_point_delta": np.deg2rad(np.asarray([10.0, 11.0, 9.0], dtype=np.float64)),
            "confidence": np.asarray([0.9, 0.95, 0.92], dtype=np.float64),
            "axis_radius": np.asarray([0.20, 0.25, 0.22], dtype=np.float64),
            "reproj_pred": np.asarray([1.0, 1.0, 1.0], dtype=np.float64),
            "plane_dist": np.asarray([0.002, 0.002, 0.002], dtype=np.float64),
            "depth_residual": np.asarray([0.002, 0.002, 0.002], dtype=np.float64),
        }
        config = ScreenHingeTrackingConfig(
            angle_method="three_point_direct",
            three_point_count=3,
            angle_min_deg=-90.0,
            angle_max_deg=90.0,
        )
        got, _, diag = estimate_three_point_direct_theta(
            obs,
            np.deg2rad(50.0),
            np.deg2rad(40.0),
            np.deg2rad(30.0),
            30.0,
            config,
        )
        self.assertEqual(diag["mode"], "incremental_delta")
        self.assertAlmostEqual(np.rad2deg(got), 50.0, places=4)

    def test_signed_axis_angle_is_independent_of_radius(self):
        origin = np.zeros(3, dtype=np.float64)
        axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        theta = np.deg2rad(60.0)
        ref = np.asarray([0.25, 0.0, 0.0], dtype=np.float64)
        obs = np.asarray([0.25 * np.cos(theta), 0.25 * np.sin(theta), 0.0], dtype=np.float64)
        self.assertAlmostEqual(np.rad2deg(signed_axis_angle(ref, obs, origin, axis)), 60.0, places=5)


if __name__ == "__main__":
    unittest.main()
