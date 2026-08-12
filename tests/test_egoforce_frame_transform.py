import numpy as np

from vlm_sam2_recon.rigid_pipeline.egoforce import make_c0_payload, project_points, transform_points


def test_transform_points_supports_batched_hand_geometry():
    points = np.zeros((2, 4, 3), dtype=np.float32)
    points[..., 2] = 1.0
    transform = np.eye(4)
    transform[:3, 3] = (0.1, -0.2, 0.3)
    transformed = transform_points(points, transform)
    assert transformed.shape == points.shape
    assert np.allclose(transformed[..., 0], 0.1)
    assert np.allclose(transformed[..., 1], -0.2)
    assert np.allclose(transformed[..., 2], 1.3)


def test_c0_payload_transforms_geometry_but_does_not_relabel_local_pose_parameters():
    raw = {
        "visible_hand": np.array([True, False]),
        "hand_vertices": np.array([[[0.0, 0.0, 1.0]], [[0.0, 0.0, 2.0]]]),
        "mano_global_orient": np.ones((2, 6)),
        "left_hand_faces": np.zeros((0, 3), dtype=np.int64),
    }
    transform = np.eye(4)
    transform[0, 3] = 0.25
    payload = make_c0_payload(raw, transform)
    assert np.allclose(payload["hand_vertices"][..., 0], 0.25)
    assert "mano_global_orient" not in payload
    assert payload["coordinate_frame"].item() == "frame0_right_camera_opencv_rdf"


def test_project_points_uses_opencv_rdf_pinhole_convention():
    points = np.array([[0.0, 0.0, 2.0], [1.0, -0.5, 2.0], [0.0, 0.0, -1.0]])
    uv, valid = project_points(points, {"fx": 100.0, "fy": 120.0, "cx": 10.0, "cy": 20.0})
    assert np.allclose(uv[:2], [[10.0, 20.0], [60.0, -10.0]])
    assert valid.tolist() == [True, True, False]
    assert np.isnan(uv[2]).all()
