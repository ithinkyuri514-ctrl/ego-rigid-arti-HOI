import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vlm_sam2_recon.stages.camera_alignment import (  # noqa: E402
    camera_to_camera_matrix,
    rgb_camera_world_matrix,
    rgb_pose_axis_correction,
)
from vlm_sam2_recon.stages.contact_driven_screen import (  # noqa: E402
    PoseTimeline,
    camera_transform_for_resolved_pose,
)


def pose_fields(prefix: str, x: float = 0.0) -> dict[str, str]:
    return {
        f"{prefix}_x": str(x),
        f"{prefix}_y": "0",
        f"{prefix}_z": "0",
        f"{prefix}_qw": "1",
        f"{prefix}_qx": "0",
        f"{prefix}_qy": "0",
        f"{prefix}_qz": "0",
    }


class CameraPoseCompensationTest(unittest.TestCase):
    def test_default_exported_rgb_axis_correction_is_minus_90_degrees(self):
        correction = rgb_pose_axis_correction({})
        expected = np.asarray(
            [
                [0.0, 1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        np.testing.assert_allclose(correction, expected, atol=1e-12)

    def test_pose_axis_correction_does_not_move_camera_center(self):
        extrinsic = np.eye(4)
        extrinsic[:3, 3] = [0.1, -0.2, 0.3]
        meta = {"rgb_extrinsics_right": extrinsic.tolist()}
        world_camera = rgb_camera_world_matrix(meta, pose_fields("rgb_pose"))
        np.testing.assert_allclose(world_camera[:3, 3], extrinsic[:3, 3], atol=1e-12)

    def test_camera_matrix_accepts_depth_pose_prefix(self):
        meta = {
            "rgb_extrinsics_right": np.eye(4).tolist(),
            "rgb_pose_image_rotation_deg": 0.0,
        }
        align = pose_fields("rgb_pose", x=1.0)
        view = pose_fields("depth_pose", x=3.0)
        got = camera_to_camera_matrix(
            meta,
            align,
            view,
            view_pose_prefix="depth_pose",
        )
        np.testing.assert_allclose(got[:3, 3], [-2.0, 0.0, 0.0], atol=1e-12)

    def test_resolved_pose_anchors_to_export_alignment_frame(self):
        meta = {
            "rgb_extrinsics_right": np.eye(4).tolist(),
            "rgb_pose_image_rotation_deg": 0.0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = {"index": "0", **pose_fields("rgb_pose", x=1.0)}
            with (root / "frames.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            timeline = PoseTimeline(
                path=root / "pose.csv",
                rows_by_index={
                    0: {"x": "0", "y": "0", "z": "0", "qw": "1", "qx": "0", "qy": "0", "qz": "0"},
                    1: {"x": "3", "y": "0", "z": "0", "qw": "1", "qx": "0", "qy": "0", "qz": "0"},
                },
            )
            got = camera_transform_for_resolved_pose(meta, root, 1, timeline)
        np.testing.assert_allclose(got[:3, 3], [-2.0, 0.0, 0.0], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
