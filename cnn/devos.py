"""3D DLIR stages following de Vos et al. (MedIA 2019).

The network predicts affine parameters or cubic B-spline control displacements.
Stages can be trained sequentially; the preceding transformations are frozen.
Displacements are fixed-grid pull fields in array order (z, y, x), in voxels.
"""

from collections.abc import Sequence
import math

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from data.geometry import bspline_basis
from cnn.model3d import warp_torch3d


def _triple(values: Sequence[float], name: str) -> tuple[float, float, float]:
    if len(values) != 3 or any(not math.isfinite(float(v)) or float(v) <= 0 for v in values):
        raise ValueError(f"{name} must contain three positive finite values in z,y,x order")
    return tuple(float(v) for v in values)


def _glorot(module: nn.Module) -> None:
    for layer in module.modules():
        if isinstance(layer, (nn.Conv3d, nn.Linear)):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


def _rotations(angles: Tensor) -> Tensor:
    """Euler rotations around array z, y, x axes, composed in that order."""
    a, b, c = angles.unbind(dim=1)
    ca, sa, cb, sb, cc, sc = a.cos(), a.sin(), b.cos(), b.sin(), c.cos(), c.sin()
    zero, one = torch.zeros_like(a), torch.ones_like(a)
    rz = torch.stack((torch.stack((one, zero, zero), -1),
                      torch.stack((zero, ca, -sa), -1),
                      torch.stack((zero, sa, ca), -1)), dim=1)
    ry = torch.stack((torch.stack((cb, zero, sb), -1),
                      torch.stack((zero, one, zero), -1),
                      torch.stack((-sb, zero, cb), -1)), dim=1)
    rx = torch.stack((torch.stack((cc, -sc, zero), -1),
                      torch.stack((sc, cc, zero), -1),
                      torch.stack((zero, zero, one), -1)), dim=1)
    return rz @ ry @ rx


class AffineStage3D(nn.Module):
    """Two shared-weight image pipelines and a constrained 12-parameter head."""

    def __init__(self) -> None:
        super().__init__()
        channels = (8, 16, 32, 64, 128)
        previous = 1
        self.blocks = nn.ModuleList()
        for width in channels:
            self.blocks.append(nn.Sequential(nn.Conv3d(previous, width, 3, padding=1),
                                             nn.ReLU(inplace=True)))
            previous = width
        self.head = nn.Sequential(nn.Linear(2 * channels[-1], 128), nn.ReLU(inplace=True),
                                  nn.Linear(128, 12))
        _glorot(self)

    def _encode(self, image: Tensor) -> Tensor:
        for block in self.blocks:
            image = block(image)
            kernel = tuple(2 if side >= 2 else 1 for side in image.shape[-3:])
            image = F.avg_pool3d(image, kernel_size=kernel, stride=kernel)
        return image.mean(dim=(-3, -2, -1))

    def forward(self, fixed: Tensor, moving: Tensor,
                voxel_spacing: Sequence[float]) -> Tensor:
        if fixed.shape != moving.shape or fixed.ndim != 5 or fixed.shape[1] != 1:
            raise ValueError("affine stage expects matching [B,1,D,H,W] images")
        spacing = _triple(voxel_spacing, "voxel_spacing")
        raw = self.head(torch.cat((self._encode(fixed), self._encode(moving)), dim=1))
        translation_mm = raw[:, :3]
        rotation = math.pi * raw[:, 3:6].tanh()
        scale = 0.5 + raw[:, 6:9].sigmoid()
        shear = math.pi * raw[:, 9:12].tanh()
        one, zero = torch.ones_like(shear[:, 0]), torch.zeros_like(shear[:, 0])
        sh = torch.stack((torch.stack((one, shear[:, 0], shear[:, 1]), -1),
                          torch.stack((zero, one, shear[:, 2]), -1),
                          torch.stack((zero, zero, one), -1)), dim=1)
        matrix = _rotations(rotation) @ sh @ torch.diag_embed(scale)
        shape = fixed.shape[-3:]
        axes = [torch.arange(n, dtype=fixed.dtype, device=fixed.device) for n in shape]
        grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
        physical = (grid - grid.new_tensor([(n - 1) / 2 for n in shape]))
        physical = physical * grid.new_tensor(spacing)
        mapped = torch.einsum("bij,dhwj->bdhwi", matrix, physical) + translation_mm[:, None, None, None]
        return (mapped - physical) / physical.new_tensor(spacing)


class DeformableStage3D(nn.Module):
    """Fully convolutional patch predictor with support matched to B-spline spacing."""

    def __init__(self, spacing_mm: Sequence[float], pool_counts: Sequence[int]) -> None:
        super().__init__()
        self.spacing_mm = _triple(spacing_mm, "spacing_mm")
        if len(pool_counts) != 3 or any(not isinstance(v, int) or v < 0 or v > 6 for v in pool_counts):
            raise ValueError("pool_counts must contain three integers in 0..6")
        self.pool_counts = tuple(pool_counts)
        n_pool = max(self.pool_counts)
        self.blocks = nn.ModuleList()
        previous = 2
        for level in range(max(1, n_pool)):
            width = min(16 * 2 ** level, 64)
            self.blocks.append(nn.Sequential(nn.Conv3d(previous, width, 3, padding=1),
                                             nn.ReLU(inplace=True)))
            previous = width
        self.refine = nn.Sequential(
            nn.Conv3d(previous, previous, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(previous, previous, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(previous, previous, 1), nn.ReLU(inplace=True),
            nn.Conv3d(previous, previous, 1), nn.ReLU(inplace=True),
            nn.Conv3d(previous, 3, 1),
        )
        _glorot(self)

    def forward(self, fixed: Tensor, moving: Tensor,
                voxel_spacing: Sequence[float]) -> Tensor:
        if fixed.shape != moving.shape or fixed.ndim != 5 or fixed.shape[1] != 1:
            raise ValueError("deformable stage expects matching [B,1,D,H,W] images")
        spacing = _triple(voxel_spacing, "voxel_spacing")
        shape = fixed.shape[-3:]
        gaps = tuple(mm / voxel for mm, voxel in zip(self.spacing_mm, spacing))
        if min(gaps) < 1:
            raise ValueError("B-spline grid spacing is below one voxel at this image resolution")
        bases = [bspline_basis(size, gap).astype(np.float32)
                 for size, gap in zip(shape, gaps)]
        grid_shape = tuple(b.shape[1] for b in bases)
        strides = tuple(2 ** count for count in self.pool_counts)
        padding = []
        for size, count, gap, stride in zip(shape, grid_shape, gaps, strides):
            extra = count * stride - size
            left = min(max(1, round(gap)), extra)
            padding.append((left, extra - left))
        x = F.pad(torch.cat((fixed, moving), dim=1),
                  (padding[2][0], padding[2][1], padding[1][0], padding[1][1],
                   padding[0][0], padding[0][1]))
        for level, block in enumerate(self.blocks):
            x = block(x)
            if level < max(self.pool_counts):
                kernel = tuple(2 if level < count else 1 for count in self.pool_counts)
                x = F.avg_pool3d(x, kernel_size=kernel, stride=kernel)
        control = self.refine(x)
        if control.shape[-3:] != grid_shape:
            raise RuntimeError("control grid and B-spline basis disagree")
        bz, by, bx = (torch.as_tensor(b, dtype=control.dtype, device=control.device)
                      for b in bases)
        return torch.einsum("za,bcade,yd,xe->bzyxc", bz, control, by, bx)


def resize_field(field: Tensor, shape: Sequence[int]) -> Tensor:
    """Trilinearly resize a pull field and convert its displacement units."""
    target = tuple(int(v) for v in shape)
    source = field.shape[1:4]
    if source == target:
        return field
    resized = F.interpolate(field.permute(0, 4, 1, 2, 3), size=target,
                            mode="trilinear", align_corners=True).permute(0, 2, 3, 4, 1)
    scale = resized.new_tensor([(new - 1) / (old - 1) for old, new in zip(source, target)])
    return resized * scale


def compose_pull_fields(base: Tensor, residual: Tensor) -> Tensor:
    """Apply residual to an image already warped by base."""
    sampled = warp_torch3d(base.permute(0, 4, 1, 2, 3), residual)
    return residual + sampled.permute(0, 2, 3, 4, 1)


def bending_energy_mm(field: Tensor, voxel_spacing: Sequence[float]) -> Tensor:
    """Mean 3D bending energy using physical displacement and coordinates."""
    spacing = _triple(voxel_spacing, "voxel_spacing")
    physical = field * field.new_tensor(spacing)
    second = []
    for axis, step in enumerate(spacing, start=1):
        difference = physical.diff(n=2, dim=axis) / step ** 2
        second.append(difference.square().mean())
    mixed = []
    for a, b in ((1, 2), (1, 3), (2, 3)):
        difference = physical.diff(dim=a).diff(dim=b) / (spacing[a - 1] * spacing[b - 1])
        mixed.append(difference.square().mean())
    return sum(second) + 2 * sum(mixed)


class DLIR3D(nn.Module):
    """Affine and/or deformable stages with exact pull-field composition."""

    def __init__(self, stage_specs: Sequence[dict]) -> None:
        super().__init__()
        if not stage_specs:
            raise ValueError("at least one stage is required")
        self.stage_specs = [dict(spec) for spec in stage_specs]
        modules = []
        for spec in self.stage_specs:
            if spec["kind"] == "affine":
                modules.append(AffineStage3D())
            elif spec["kind"] == "deformable":
                modules.append(DeformableStage3D(spec["spacing_mm"], spec["pool_counts"]))
            else:
                raise ValueError(f"unknown DLIR stage: {spec['kind']}")
        self.stages = nn.ModuleList(modules)

    def train_only_stage(self, index: int) -> None:
        for stage_number, stage in enumerate(self.stages):
            for parameter in stage.parameters():
                parameter.requires_grad_(stage_number == index)

    def forward(self, fixed: Tensor, moving: Tensor, *,
                voxel_spacing: Sequence[float], upto_stage: int | None = None) -> tuple[Tensor, Tensor]:
        if fixed.shape != moving.shape or fixed.ndim != 5 or fixed.shape[1] != 1:
            raise ValueError("DLIR expects matching [B,1,D,H,W] images")
        spacing = _triple(voxel_spacing, "voxel_spacing")
        last = len(self.stages) - 1 if upto_stage is None else upto_stage
        if last < 0 or last >= len(self.stages):
            raise ValueError("invalid stage index")
        native_shape = fixed.shape[-3:]
        total = fixed.new_zeros((fixed.shape[0], *native_shape, 3))
        warped = moving
        for stage_number in range(last + 1):
            spec = self.stage_specs[stage_number]
            if "target_spacing_mm" in spec:
                desired = _triple(spec["target_spacing_mm"], "target_spacing_mm")
                stage_shape = tuple(max(3, round((size - 1) * step / goal) + 1)
                                    for size, step, goal in zip(native_shape, spacing, desired))
            else:
                factors = _triple(spec.get("downsample", (1, 1, 1)), "downsample")
                stage_shape = tuple(max(3, round((size - 1) / factor) + 1)
                                    for size, factor in zip(native_shape, factors))
            stage_spacing = tuple(step * (size - 1) / (reduced - 1)
                                  for step, size, reduced in zip(spacing, native_shape, stage_shape))
            if stage_shape == native_shape:
                stage_fixed, stage_moving = fixed, warped
            else:
                stage_fixed = F.interpolate(fixed, size=stage_shape, mode="trilinear", align_corners=True)
                stage_moving = F.interpolate(warped, size=stage_shape, mode="trilinear", align_corners=True)
            residual = self.stages[stage_number](stage_fixed, stage_moving, stage_spacing)
            total = compose_pull_fields(total, resize_field(residual, native_shape))
            warped = warp_torch3d(moving, total)
        return total, warped


def sunnybrook_stage_specs(mode: str = "multi") -> list[dict]:
    """de Vos cardiac Table 1: array order z,y,x in physical millimetres."""
    fine = {"kind": "deformable", "downsample": [1, 1, 1],
            "spacing_mm": [8.0, 10.0, 10.0], "pool_counts": [0, 3, 3]}
    if mode == "single":
        return [fine]
    if mode == "multi":
        coarse = {"kind": "deformable", "downsample": [2, 2, 2],
                  "spacing_mm": [16.0, 20.0, 20.0], "pool_counts": [0, 3, 3]}
        return [coarse, fine]
    raise ValueError("mode must be single or multi")


def nlst_stage_specs() -> list[dict]:
    """de Vos chest-CT stage schedule; requires the original NLST protocol."""
    return [{"kind": "affine", "target_spacing_mm": [8.0, 8.0, 8.0]},
            {"kind": "deformable", "target_spacing_mm": [8.0, 8.0, 8.0],
             "spacing_mm": [64.0, 64.0, 64.0], "pool_counts": [3, 3, 3]},
            {"kind": "deformable", "target_spacing_mm": [4.0, 4.0, 4.0],
             "spacing_mm": [32.0, 32.0, 32.0], "pool_counts": [3, 3, 3]},
            {"kind": "deformable", "target_spacing_mm": [2.0, 2.0, 2.0],
             "spacing_mm": [16.0, 16.0, 16.0], "pool_counts": [3, 3, 3]}]
