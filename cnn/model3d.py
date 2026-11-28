"""Native 3D CNN predicting cubic B-spline control displacements."""

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from data.geometry import bspline_basis


def warp_torch3d(image: Tensor, field: Tensor, *, padding_mode: str = "zeros") -> Tensor:
    """Pull [B,C,D,H,W] moving data with (dz,dy,dx) voxel displacements."""
    if image.ndim != 5 or field.shape != (image.shape[0], *image.shape[-3:], 3):
        raise ValueError("expected image [B,C,D,H,W] and field [B,D,H,W,3]")
    depth, height, width = image.shape[-3:]
    if min(depth, height, width) < 2:
        raise ValueError("spatial dimensions must be at least 2")
    z = torch.arange(depth, dtype=field.dtype, device=field.device).view(1, depth, 1, 1)
    y = torch.arange(height, dtype=field.dtype, device=field.device).view(1, 1, height, 1)
    x = torch.arange(width, dtype=field.dtype, device=field.device).view(1, 1, 1, width)
    grid = torch.stack((2 * (x + field[..., 2]) / (width - 1) - 1,
                        2 * (y + field[..., 1]) / (height - 1) - 1,
                        2 * (z + field[..., 0]) / (depth - 1) - 1), dim=-1)
    return F.grid_sample(image, grid, mode="bilinear", padding_mode=padding_mode, align_corners=True)


def bending_energy3d(field: Tensor) -> Tensor:
    """Mean squared 3D second differences and mixed derivatives."""
    if field.ndim != 5 or field.shape[-1] != 3 or min(field.shape[1:4]) < 3:
        raise ValueError("field must be [B,D,H,W,3] with each side >= 3")
    dzz = field[:, 2:] - 2 * field[:, 1:-1] + field[:, :-2]
    dyy = field[:, :, 2:] - 2 * field[:, :, 1:-1] + field[:, :, :-2]
    dxx = field[:, :, :, 2:] - 2 * field[:, :, :, 1:-1] + field[:, :, :, :-2]
    dzy = field[:, 1:, 1:] - field[:, 1:, :-1] - field[:, :-1, 1:] + field[:, :-1, :-1]
    dzx = field[:, 1:, :, 1:] - field[:, 1:, :, :-1] - field[:, :-1, :, 1:] + field[:, :-1, :, :-1]
    dyx = field[:, :, 1:, 1:] - field[:, :, 1:, :-1] - field[:, :, :-1, 1:] + field[:, :, :-1, :-1]
    return (dzz.square().mean() + dyy.square().mean() + dxx.square().mean()
            + 2 * (dzy.square().mean() + dzx.square().mean() + dyx.square().mean()))


def ncc_loss3d(fixed: Tensor, warped: Tensor, mask: Tensor | None = None) -> Tensor:
    if fixed.shape != warped.shape or fixed.ndim != 5:
        raise ValueError("fixed and warped must have matching [B,C,D,H,W] shape")
    weight = (fixed > 0 if mask is None else mask != 0).to(fixed.dtype)
    if weight.shape != fixed.shape:
        raise ValueError("mask must match image shape")
    axes = (1, 2, 3, 4)
    count = weight.sum(dim=axes).clamp_min(1)
    a = (fixed - ((fixed * weight).sum(dim=axes) / count)[:, None, None, None, None]) * weight
    b = (warped - ((warped * weight).sum(dim=axes) / count)[:, None, None, None, None]) * weight
    numerator = (a * b).sum(dim=axes)
    # Clamp before sqrt: where() alone does not prevent NaN gradients at zero variance.
    denominator = torch.sqrt((a.square().sum(dim=axes) * b.square().sum(dim=axes)).clamp_min(1e-24))
    correlation = torch.where(denominator > 1e-12, numerator / denominator.clamp_min(1e-12), 0)
    return (1 - correlation).mean()


class RegistrationCNN3D(nn.Module):
    """One-stage 3D B-spline CNN with identity initialization."""

    def __init__(self, shape: tuple[int, int, int], spacing: int = 32):
        super().__init__()
        if len(shape) != 3 or min(shape) < 3 or not isinstance(spacing, int) or spacing < 2:
            raise ValueError("shape must be 3D and spacing >= 2")
        self.shape = tuple(int(v) for v in shape)
        self.spacing = spacing
        bases = [bspline_basis(length, spacing) for length in self.shape]
        self.grid_shape = tuple(b.shape[1] for b in bases)
        for name, basis in zip(("basis_z", "basis_y", "basis_x"), bases):
            self.register_buffer(name, torch.from_numpy(basis.astype(np.float32)), persistent=False)
        self.encoder = nn.Sequential(
            nn.Conv3d(2, 8, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(8, 16, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),
        )
        self.head = nn.Conv3d(32, 3, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, fixed: Tensor, moving: Tensor) -> tuple[Tensor, Tensor]:
        if fixed.shape != moving.shape or fixed.ndim != 5 or fixed.shape[1] != 1:
            raise ValueError("fixed and moving must match [B,1,D,H,W]")
        if tuple(fixed.shape[-3:]) != self.shape:
            raise ValueError(f"model expects shape {self.shape}")
        features = self.encoder(torch.cat((fixed, moving), dim=1))
        control = self.head(F.adaptive_avg_pool3d(features, self.grid_shape))
        field = torch.einsum("za,bcade,yd,xe->bzyxc", self.basis_z, control,
                             self.basis_y, self.basis_x)
        return field, warp_torch3d(moving, field)
