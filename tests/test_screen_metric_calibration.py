import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vlm_sam2_recon.stages.screen_metric_calibration import (  # noqa: E402
    local_scale_shift_matrix,
    transform_points,
)


class ScreenMetricCalibrationTest(unittest.TestCase):
    def test_local_scale_preserves_normal_coordinate(self):
        pivot = np.zeros(3)
        axis = np.asarray([1.0, 0.0, 0.0])
        radial = np.asarray([0.0, 1.0, 0.0])
        points = np.asarray([[1.0, 2.0, 3.0]])
        matrix = local_scale_shift_matrix(pivot, axis, radial, 2.0, 0.5, 0.1, -0.2)
        got = transform_points(points, matrix)
        np.testing.assert_allclose(got, [[2.1, 0.8, 3.0]], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
