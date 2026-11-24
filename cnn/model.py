"""Unsupervised 2D CNN that predicts cubic B-spline control displacements.

Fields are on the fixed grid in voxel units with channels (dy, dx). Warping
samples the moving image at (y + dy, x + dx), as in data.geometry.warp.
"""

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from data.geometry import bspline_basis


def warp_torch(image: Tensor, field: Tensor, *, mode: str = "bilinear") -> Tensor:
    """Pull an image or label map into fixed space; use ``mode='nearest'`` for labels."""
    if image.ndim != 4 or field.shape != (image.shape[0], *image.shape[-2:], 2):
        raise ValueError("expected image [B,C,H,W] and field [B,H,W,2]")
    if mode not in ("bilinear", "nearest"):
        raise ValueError("mode must be 'bilinear' or 'nearest'")
    height, width = image.shape[-2:]
    if height < 2 or width < 2:
        raise ValueError("height and width must be at least 2")
    y = torch.arange(height, device=field.device, dtype=field.dtype).view(1, height, 1)
    x = torch.arange(width, device=field.device, dtype=field.dtype).view(1, 1, width)
    if mode == "nearest":
        # Match NumPy's float64 floor(q + 0.5), including half-voxel ties.
        iy = torch.floor(y.double() + field[..., 0].double() + 0.5).long()
        ix = torch.floor(x.double() + field[..., 1].double() + 0.5).long()
        valid = (iy >= 0) & (iy < height) & (ix >= 0) & (ix < width)
        linear = (iy.clamp(0, height - 1) * width + ix.clamp(0, width - 1))
        index = linear.flatten(1).unsqueeze(1).expand(-1, image.shape[1], -1)
        sampled = image.flatten(2).gather(2, index).reshape_as(image)
        return sampled * valid.unsqueeze(1)
    grid_x = 2 * (x + field[..., 1]) / (width - 1) - 1
    grid_y = 2 * (y + field[..., 0]) / (height - 1) - 1
    grid = torch.stack((grid_x, grid_y), dim=-1)
    return F.grid_sample(image, grid, mode=mode, padding_mode="zeros", align_corners=True)


def bending_energy_torch(field: Tensor) -> Tensor:
    """Mean squared 2D second differences, matching the NumPy FFD penalty."""
    if field.ndim != 4 or field.shape[-1] != 2 or min(field.shape[1:3]) < 3:
        raise ValueError("field must have shape [B,H,W,2], with H,W >= 3")
    dyy = field[:, 2:] - 2 * field[:, 1:-1] + field[:, :-2]
    dxx = field[:, :, 2:] - 2 * field[:, :, 1:-1] + field[:, :, :-2]
    dyx = field[:, 1:, 1:] - field[:, 1:, :-1] - field[:, :-1, 1:] + field[:, :-1, :-1]
    return dyy.square().mean() + dxx.square().mean() + 2 * dyx.square().mean()


def ncc_loss(fixed: Tensor, warped: Tensor, mask: Tensor | None = None) -> Tensor:
    """Mean batch loss 1-NCC over the fixed-image foreground (or supplied mask)."""
    if fixed.shape != warped.shape or fixed.ndim != 4:
        raise ValueError("fixed and warped must have matching [B,C,H,W] shape")
    if mask is None:
        mask = torch.ones_like(fixed)
    elif mask.shape != fixed.shape:
        raise ValueError("mask must match fixed image shape")
    weight = (mask != 0).to(dtype=fixed.dtype)
    count = weight.sum(dim=(1, 2, 3)).clamp_min(1)
    mean_fixed = (fixed * weight).sum(dim=(1, 2, 3)) / count
    mean_warped = (warped * weight).sum(dim=(1, 2, 3)) / count
    a = (fixed - mean_fixed[:, None, None, None]) * weight
    b = (warped - mean_warped[:, None, None, None]) * weight
    numerator = (a * b).sum(dim=(1, 2, 3))
    denominator = torch.sqrt(a.square().sum(dim=(1, 2, 3)) * b.square().sum(dim=(1, 2, 3)))
    correlation = torch.where(denominator > 1e-12, numerator / denominator.clamp_min(1e-12), 0)
    return (1 - correlation).mean()


class RegistrationCNN(nn.Module):
    """One-stage deformable network; its final layer starts at identity warp."""

    def __init__(self, shape: tuple[int, int], spacing: int = 32):
        super().__init__()
        if (len(shape) != 2 or any(not isinstance(v, int) or v < 3 for v in shape)
                or not isinstance(spacing, int) or spacing < 2):
            raise ValueError("shape must be two integers >= 3 and spacing an integer >= 2")
        self.shape = shape
        self.spacing = spacing
        by = bspline_basis(self.shape[0], self.spacing)
        bx = bspline_basis(self.shape[1], self.spacing)
        self.grid_shape = (by.shape[1], bx.shape[1])
        self.register_buffer("basis_y", torch.from_numpy(by.astype(np.float32)), persistent=False)
        self.register_buffer("basis_x", torch.from_numpy(bx.astype(np.float32)), persistent=False)
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(64, 2, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, fixed: Tensor, moving: Tensor) -> tuple[Tensor, Tensor]:
        if fixed.shape != moving.shape or fixed.ndim != 4 or fixed.shape[1] != 1:
            raise ValueError("fixed and moving must have matching [B,1,H,W] shape")
        if tuple(fixed.shape[-2:]) != self.shape:
            raise ValueError(f"model expects spatial shape {self.shape}")
        features = self.encoder(torch.cat((fixed, moving), dim=1))
        control = self.head(F.adaptive_avg_pool2d(features, self.grid_shape))
        field = torch.einsum("hg,bcgk,wk->bhwc", self.basis_y, control, self.basis_x)
        warped = warp_torch(moving, field)
        return field, warped
