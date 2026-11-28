"""3D DLIR stages following de Vos et al. (MedIA 2019).

The network predicts affine parameters or cubic B-spline control displacements.
Stages can be trained sequentially; the preceding transformations are frozen.
Displacements are fixed-grid pull fields in array order (z, y, x), in voxels.
"""

from collections.abc import Sequence
from contextlib import nullcontext
import math

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from cnn.bspline import interpolate_control_grid
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

    def __init__(self, channels: int = 32, depth: int = 5,
                 downsampling: str = "average") -> None:
        super().__init__()
        if channels < 1 or depth < 1 or downsampling not in ("average", "strided"):
            raise ValueError("invalid affine network configuration")
        self.blocks = nn.ModuleList()
        previous = 1
        for _ in range(depth):
            layers = [nn.Conv3d(previous, channels, 3, padding=1), nn.ReLU()]
            if downsampling == "average":
                layers.append(nn.AvgPool3d(2, ceil_mode=True))
            else:
                layers.extend([nn.Conv3d(channels, channels, 4, stride=2, padding=1), nn.ReLU()])
            self.blocks.append(nn.Sequential(*layers))
            previous = channels
        # Hidden widths are explicit implementation choices, not specified in Fig. 2.
        self.head = nn.Sequential(nn.Linear(2 * channels, 128), nn.ReLU(),
                                  nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 12))
        _glorot(self)

    def _encode(self, image: Tensor) -> Tensor:
        for block in self.blocks:
            for layer in block:
                if isinstance(layer, nn.AvgPool3d):
                    kernel = tuple(2 if size >= 2 else 1 for size in image.shape[-3:])
                    image = F.avg_pool3d(image, kernel, kernel, ceil_mode=True)
                else:
                    if isinstance(layer, nn.Conv3d) and layer.stride == (2, 2, 2):
                        pads = [max(0, 2 - size) for size in image.shape[-3:]]
                        image = F.pad(image, (0, pads[2], 0, pads[1], 0, pads[0]))
                    image = layer(image)
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

    def __init__(self, spacing_mm: Sequence[float], pool_counts: Sequence[int],
                 channels: int = 32, downsampling: str = "average") -> None:
        super().__init__()
        self.spacing_mm = _triple(spacing_mm, "spacing_mm")
        if len(pool_counts) != 3 or any(not isinstance(v, int) or not 0 <= v <= 6 for v in pool_counts):
            raise ValueError("pool_counts must contain three integers in 0..6")
        if channels < 1 or downsampling not in ("average", "strided"):
            raise ValueError("invalid deformable network configuration")
        self.pool_counts = tuple(pool_counts)
        self.strides = tuple(2 ** count for count in pool_counts)
        self.blocks = nn.ModuleList()
        previous = 2
        for level in range(max(self.pool_counts)):
            stride = tuple(2 if level < count else 1 for count in self.pool_counts)
            layers = [nn.Conv3d(previous, channels, 3, padding=1), nn.ReLU()]
            if downsampling == "average":
                layers.append(nn.AvgPool3d(stride, stride))
            else:
                kernel = tuple(4 if step == 2 else 1 for step in stride)
                padding = tuple(1 if step == 2 else 0 for step in stride)
                layers.extend([nn.Conv3d(channels, channels, kernel, stride, padding), nn.ReLU()])
            self.blocks.append(nn.Sequential(*layers))
            previous = channels
        self.refine = nn.Sequential(
            nn.Conv3d(previous, channels, 3, padding=1), nn.ReLU(),
            nn.Conv3d(channels, channels, 3, padding=1), nn.ReLU(),
            nn.Conv3d(channels, channels, 1), nn.ReLU(),
            nn.Conv3d(channels, channels, 1), nn.ReLU(),
            nn.Conv3d(channels, 3, 1),
        )
        _glorot(self)

    def forward(self, fixed: Tensor, moving: Tensor,
                voxel_spacing: Sequence[float]) -> Tensor:
        if fixed.shape != moving.shape or fixed.ndim != 5 or fixed.shape[1] != 1:
            raise ValueError("deformable stage expects matching [B,1,D,H,W] images")
        spacing = _triple(voxel_spacing, "voxel_spacing")
        actual = tuple(step * voxel for step, voxel in zip(self.strides, spacing))
        if not np.allclose(actual, self.spacing_mm, rtol=1e-4, atol=1e-4):
            raise ValueError(f"grid spacing {actual} mm differs from configured {self.spacing_mm}; "
                             "resample inputs to the preset resolution or use a custom stage config")
        shape = fixed.shape[-3:]
        grid_shape = tuple((n - 1) // step + 4 for n, step in zip(shape, self.strides))
        pads = [(step, count * step - n - step)
                for n, count, step in zip(shape, grid_shape, self.strides)]
        x = F.pad(torch.cat((fixed, moving), dim=1),
                  (pads[2][0], pads[2][1], pads[1][0], pads[1][1], pads[0][0], pads[0][1]))
        for block in self.blocks:
            x = block(x)
        control = self.refine(x)
        return interpolate_control_grid(control, shape, self.strides)


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
    sampled = warp_torch3d(base.permute(0, 4, 1, 2, 3), residual, padding_mode="border")
    return residual + sampled.permute(0, 2, 3, 4, 1)


def sample_grid(image: Tensor, shape: Sequence[int], source_spacing: Sequence[float],
                target_spacing: Sequence[float], origin: Sequence[float] = (0, 0, 0),
                *, padding_mode: str = "border") -> Tensor:
    """Resample a tensor on physical grids; origin is in source voxel units."""
    axes = [(torch.arange(n, dtype=image.dtype, device=image.device) * target / source + offset)
            for n, source, target, offset in zip(shape, source_spacing, target_spacing, origin)]
    normalized = [2 * axis / (n - 1) - 1 for axis, n in zip(axes, image.shape[-3:])]
    zz, yy, xx = torch.meshgrid(*normalized, indexing="ij")
    grid = torch.stack((xx, yy, zz), -1)[None].expand(image.shape[0], -1, -1, -1, -1)
    return F.grid_sample(image, grid, mode="bilinear", padding_mode=padding_mode, align_corners=True)


def bending_energy_mm(field: Tensor, voxel_spacing: Sequence[float]) -> Tensor:
    """Mean 3D bending energy using physical displacement and coordinates."""
    spacing = _triple(voxel_spacing, "voxel_spacing")
    if field.ndim != 5 or field.shape[-1] != 3 or min(field.shape[1:4]) < 3:
        raise ValueError("field must be [B,D,H,W,3] with spatial sides >= 3")
    physical = field * field.new_tensor(spacing)
    second = []
    for axis, step in enumerate(spacing, start=1):
        difference = physical.diff(n=2, dim=axis) / step ** 2
        second.append(difference.square().sum(-1).mean())
    mixed = []
    for a, b in ((1, 2), (1, 3), (2, 3)):
        difference = physical.diff(dim=a).diff(dim=b) / (spacing[a - 1] * spacing[b - 1])
        mixed.append(difference.square().sum(-1).mean())
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
                modules.append(AffineStage3D(spec.get("channels", 32), spec.get("depth", 5),
                                             spec.get("downsampling", "average")))
            elif spec["kind"] == "deformable":
                modules.append(DeformableStage3D(spec["spacing_mm"], spec["pool_counts"],
                                                 spec.get("channels", 32), spec.get("downsampling", "average")))
            else:
                raise ValueError(f"unknown DLIR stage: {spec['kind']}")
        self.stages = nn.ModuleList(modules)

    def train_only_stage(self, index: int) -> None:
        if not 0 <= index < len(self.stages):
            raise ValueError("invalid stage index")
        for stage_number, stage in enumerate(self.stages):
            for parameter in stage.parameters():
                parameter.requires_grad_(stage_number == index)

    def forward(self, fixed: Tensor, moving: Tensor, *,
                voxel_spacing: Sequence[float], upto_stage: int | None = None,
                return_details: bool = False):
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
                stage_spacing = _triple(spec["target_spacing_mm"], "target_spacing_mm")
                stage_shape = tuple(max(3, math.ceil((size - 1) * step / goal) + 1)
                                    for size, step, goal in zip(native_shape, spacing, stage_spacing))
                stage_fixed = sample_grid(fixed, stage_shape, spacing, stage_spacing)
                stage_moving = sample_grid(warped, stage_shape, spacing, stage_spacing)
                origin = (0.0, 0.0, 0.0)
            else:
                factors = _triple(spec.get("downsample", (1, 1, 1)), "downsample")
                if any(factor != int(factor) for factor in factors):
                    raise ValueError("downsample must contain integer pooling factors")
                kernel = tuple(int(factor) for factor in factors)
                stage_fixed = F.avg_pool3d(fixed, kernel, kernel)
                stage_moving = F.avg_pool3d(warped, kernel, kernel)
                if min(stage_fixed.shape[-3:]) < 3:
                    raise ValueError("stage image is too small; each pooled side must be >= 3")
                stage_spacing = tuple(step * factor for step, factor in zip(spacing, factors))
                # Pooling samples at the centers of windows, not at voxel zero.
                origin = tuple((factor - 1) / 2 for factor in factors)
            frozen = not any(parameter.requires_grad for parameter in self.stages[stage_number].parameters())
            with torch.no_grad() if frozen else nullcontext():
                residual = self.stages[stage_number](stage_fixed, stage_moving, stage_spacing)
            field_origin = tuple(-offset * step / reduced
                                 for offset, step, reduced in zip(origin, spacing, stage_spacing))
            native_residual = sample_grid(residual.permute(0, 4, 1, 2, 3), native_shape,
                                          stage_spacing, spacing, field_origin).permute(0, 2, 3, 4, 1)
            native_residual = native_residual * residual.new_tensor(stage_spacing) / residual.new_tensor(spacing)
            total = compose_pull_fields(total, native_residual)
            warped = warp_torch3d(moving, total)
        if return_details:
            return total, warped, {"fixed": stage_fixed,
                                   "warped": warp_torch3d(stage_moving, residual),
                                   "local_field": residual, "spacing": stage_spacing}
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
    return [{"kind": "affine", "target_spacing_mm": [8.0, 8.0, 8.0],
             "depth": 3, "channels": 32, "downsampling": "strided"},
            {"kind": "deformable", "target_spacing_mm": [8.0, 8.0, 8.0],
             "spacing_mm": [64.0, 64.0, 64.0], "pool_counts": [3, 3, 3], "downsampling": "strided"},
            {"kind": "deformable", "target_spacing_mm": [4.0, 4.0, 4.0],
             "spacing_mm": [32.0, 32.0, 32.0], "pool_counts": [3, 3, 3], "downsampling": "strided"},
            {"kind": "deformable", "target_spacing_mm": [2.0, 2.0, 2.0],
             "spacing_mm": [16.0, 16.0, 16.0], "pool_counts": [3, 3, 3], "downsampling": "strided"}]
