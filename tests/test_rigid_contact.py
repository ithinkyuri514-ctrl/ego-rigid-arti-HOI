import numpy as np
import torch

from vlm_sam2_recon.rigid_pipeline.rigid_contact import (
    longest_true_interval,
    project_vertices_outside_sdf,
    sample_sdf_grid,
    topk_penetration_loss,
    transform_points_torch,
)


def test_longest_contact_interval_closes_one_frame_gap():
    values = np.array([False, True, True, False, True, True, False, False, True])
    assert longest_true_interval(values, max_gap=1) == (1, 5)


def test_batched_transform_points():
    points = torch.zeros(2, 3, 3)
    transforms = torch.eye(4).repeat(2, 1, 1)
    transforms[:, :3, 3] = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]])
    output = transform_points_torch(points, transforms)
    assert torch.allclose(output[:, 0], transforms[:, :3, 3])


def test_sample_sdf_grid_respects_xyz_axis_order():
    x, y, z = torch.meshgrid(torch.arange(4), torch.arange(5), torch.arange(6), indexing="ij")
    sdf = x.float() + 10.0 * y.float() + 100.0 * z.float()
    points = torch.tensor([[2.0, 3.0, 4.0]])
    sampled = sample_sdf_grid(sdf, points, torch.zeros(3), pitch=1.0)
    assert torch.allclose(sampled, torch.tensor([432.0]), atol=1e-4)


def test_topk_penetration_loss_focuses_on_deepest_vertices_per_frame():
    signed = torch.tensor(
        [
            [-0.0045, -0.0015, 0.0010, 0.0200],
            [-0.0005, 0.0005, 0.0030, 0.0200],
        ]
    )
    loss = topk_penetration_loss(
        signed,
        clearance_m=0.0005,
        vertices_per_frame=2,
    )
    expected = torch.tensor(((0.005 + 0.002) / 2 + (0.001 + 0.0) / 2) / 2 / 0.005)
    assert torch.allclose(loss, expected)
    summed = topk_penetration_loss(
        signed,
        clearance_m=0.0005,
        vertices_per_frame=2,
        frame_reduction="sum",
    )
    assert torch.allclose(summed, expected * 2)


def test_project_vertices_outside_sdf_moves_only_points_below_clearance():
    x, _, _ = torch.meshgrid(
        torch.arange(8), torch.arange(4), torch.arange(4), indexing="ij"
    )
    sdf = (x.float() - 3.0) * 0.001
    points = torch.tensor([[[0.0020, 0.0015, 0.0015], [0.0050, 0.0015, 0.0015]]])
    corrected = project_vertices_outside_sdf(
        sdf,
        points,
        torch.zeros(3),
        0.001,
        clearance_m=0.0005,
    )
    signed = sample_sdf_grid(sdf, corrected, torch.zeros(3), 0.001)
    assert signed[0, 0] >= 0.0005
    assert torch.allclose(corrected[0, 1], points[0, 1])
