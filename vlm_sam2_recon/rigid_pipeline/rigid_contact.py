"""Deterministic geometry helpers for rigid hand-object contact refinement."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def close_short_false_gaps(values: np.ndarray, max_gap: int = 1) -> np.ndarray:
    values = np.asarray(values, dtype=bool).copy()
    start = None
    for index, value in enumerate(values):
        if not value and start is None:
            start = index
        if value and start is not None:
            if start > 0 and index - start <= max_gap:
                values[start:index] = True
            start = None
    return values


def longest_true_interval(values: np.ndarray, max_gap: int = 1) -> tuple[int, int] | None:
    values = close_short_false_gaps(values, max_gap=max_gap)
    best = None
    start = None
    for index, value in enumerate(np.r_[values, False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            candidate = (start, index - 1)
            if best is None or candidate[1] - candidate[0] > best[1] - best[0]:
                best = candidate
            start = None
    return best


def transform_points_torch(points: torch.Tensor, transforms: torch.Tensor) -> torch.Tensor:
    """Apply batched destination-from-source transforms to row-vector points."""
    return torch.matmul(points, transforms[..., :3, :3].transpose(-1, -2)) + transforms[..., None, :3, 3]


def project_points_torch(points: torch.Tensor, intrinsics: dict[str, float]) -> torch.Tensor:
    z = points[..., 2].clamp_min(1e-6)
    u = float(intrinsics["fx"]) * points[..., 0] / z + float(intrinsics["cx"])
    v = float(intrinsics["fy"]) * points[..., 1] / z + float(intrinsics["cy"])
    return torch.stack((u, v), dim=-1)


def image_sample(images: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample [B,H,W] images at [B,N,2] pixel coordinates."""
    height, width = images.shape[-2:]
    grid = uv.clone()
    grid[..., 0] = 2.0 * grid[..., 0] / max(width - 1, 1) - 1.0
    grid[..., 1] = 2.0 * grid[..., 1] / max(height - 1, 1) - 1.0
    sampled = F.grid_sample(
        images[:, None],
        grid[:, None],
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled[:, 0, 0]


def sample_sdf_grid(
    sdf_xyz: torch.Tensor,
    points: torch.Tensor,
    origin_xyz: torch.Tensor,
    pitch: float,
) -> torch.Tensor:
    """Trilinearly sample an XYZ-ordered SDF at [...,3] world points."""
    shape = torch.as_tensor(sdf_xyz.shape, device=points.device, dtype=points.dtype)
    indices = (points - origin_xyz) / float(pitch)
    normalized = 2.0 * indices / (shape - 1.0) - 1.0
    flat = normalized.reshape(1, 1, 1, -1, 3)
    # grid_sample volumes are [D=z,H=y,W=x], while the saved grid is [x,y,z].
    volume = sdf_xyz.permute(2, 1, 0)[None, None]
    sampled = F.grid_sample(
        volume,
        flat,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled.reshape(points.shape[:-1])


def topk_penetration_loss(
    signed_distance: torch.Tensor,
    *,
    clearance_m: float = 0.0005,
    vertices_per_frame: int = 24,
    frame_reduction: str = "mean",
) -> torch.Tensor:
    """Penalize the deepest collision vertices without dilution by the full mesh."""
    if signed_distance.ndim != 2:
        raise ValueError(
            f"Expected [frames, vertices] signed distances, got {tuple(signed_distance.shape)}"
        )
    if vertices_per_frame <= 0:
        raise ValueError("vertices_per_frame must be positive")
    violations = F.relu(float(clearance_m) - signed_distance)
    count = min(int(vertices_per_frame), violations.shape[1])
    per_frame = violations.topk(count, dim=1).values.mean(dim=1)
    if frame_reduction == "mean":
        reduced = per_frame.mean()
    elif frame_reduction == "sum":
        reduced = per_frame.sum()
    else:
        raise ValueError(f"Unsupported frame_reduction: {frame_reduction}")
    return reduced / 0.005


def project_vertices_outside_sdf(
    sdf_xyz: torch.Tensor,
    points: torch.Tensor,
    origin_xyz: torch.Tensor,
    pitch: float,
    *,
    clearance_m: float = 0.0005,
    max_steps: int = 12,
) -> torch.Tensor:
    """Project only colliding vertices outward along the sampled SDF gradient."""
    corrected = points.detach().clone()
    target = float(clearance_m) + 1e-5
    for _ in range(max_steps):
        corrected.requires_grad_(True)
        signed = sample_sdf_grid(sdf_xyz, corrected, origin_xyz, pitch)
        active = signed < target
        if not bool(active.any()):
            return corrected.detach()
        gradient = torch.autograd.grad(signed[active].sum(), corrected)[0]
        squared_norm = gradient.square().sum(dim=-1).clamp_min(1e-8)
        distance = (target - signed).clamp_min(0.0)
        step = distance[..., None] * gradient / squared_norm[..., None]
        movable = active & (squared_norm > 1e-7)
        corrected = (corrected + torch.where(movable[..., None], step, torch.zeros_like(step))).detach()
    return corrected


def geman_mcclure(residual: torch.Tensor, sigma: float) -> torch.Tensor:
    squared = residual.square()
    sigma_squared = float(sigma) ** 2
    return sigma_squared * squared / (sigma_squared + squared)


def second_difference(values: torch.Tensor) -> torch.Tensor:
    if values.shape[0] < 3:
        return values.new_zeros((0, *values.shape[1:]))
    return values[2:] - 2.0 * values[1:-1] + values[:-2]
