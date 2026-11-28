"""Affine-first 3D B-spline registration following Rueckert et al. (1999).

The image metric is differentiable Parzen-window NMI. A single cubic lattice
is subdivided exactly as the control spacing is halved between image levels.
Arrays and pull displacements use (z, y, x) order. Physical coordinates below
are millimetres along those array axes, not scanner-world coordinates.
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter
import torch
from torch import Tensor
from torch.nn import functional as F

from data.geometry import bspline_basis
from data.geometry3d import resize_volume, warp3d


@dataclass
class RegistrationResult:
    field: np.ndarray
    warped: np.ndarray
    history: list[float]
    affine_matrix: np.ndarray
    working_shapes: tuple[tuple[int, int, int], ...]


def _pyramid(volume: np.ndarray, mask: np.ndarray, max_dim: int, levels: int):
    factor = min(1.0, max_dim / max(volume.shape))
    finest = tuple(max(3, round(n * factor)) for n in volume.shape)
    shapes = [tuple(max(3, round(n / 2 ** (levels - i - 1))) for n in finest)
              for i in range(levels)]
    images, masks = [], []
    for shape in shapes:
        if shape == volume.shape:
            images.append(volume)
            masks.append(mask)
            continue
        scale = [(original - 1) / (reduced - 1)
                 for original, reduced in zip(volume.shape, shape)]
        sigma = tuple(max(0.0, (s - 1) / 2) for s in scale)
        images.append(resize_volume(gaussian_filter(volume, sigma), shape))
        masks.append(resize_volume(mask.astype(np.uint8), shape, order=0).astype(bool))
    return shapes, images, masks


def _grid(shape: tuple[int, int, int], device: torch.device) -> Tensor:
    axes = [torch.arange(n, dtype=torch.float32, device=device) for n in shape]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)


def _warp(moving: Tensor, grid: Tensor, displacement: Tensor) -> Tensor:
    shape = moving.shape
    point = grid + displacement
    sample_grid = torch.stack([2 * point[..., axis] / (shape[axis] - 1) - 1
                               for axis in (2, 1, 0)], dim=-1)[None]
    return F.grid_sample(moving[None, None], sample_grid, mode="bilinear",
                         padding_mode="zeros", align_corners=True)[0, 0]


def _affine_field(theta: Tensor, grid: Tensor, spacing: Tensor) -> Tensor:
    """12 dimensionless parameters: 3x3 delta matrix and 3 translations."""
    center = (torch.as_tensor(grid.shape[:3], device=grid.device) - 1) / 2
    centered_mm = (grid - center) * spacing
    linear = theta[:9].reshape(3, 3) * 0.05
    translation_mm = theta[9:] * 5.0
    return (torch.einsum("ij,dhwj->dhwi", linear, centered_mm)
            + translation_mm) / spacing


def _expand(control: Tensor, bases: tuple[Tensor, Tensor, Tensor]) -> Tensor:
    bz, by, bx = bases
    partial = torch.einsum("za,abcd->zbcd", bz, control)
    partial = torch.einsum("yb,zbcd->zycd", by, partial)
    return torch.einsum("xc,zycd->zyxd", bx, partial)


def subdivide_lattice(control: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    """Exact cubic subdivision for lattices whose first point is at -spacing.

    Fine even indices lie halfway between coarse points; odd indices coincide
    with coarse points. The masks are (1,1)/2 and (1,6,1)/8 respectively.
    """
    refined = control
    for axis, count in enumerate(target_shape):
        old_count = refined.shape[axis]
        weights = np.zeros((count, old_count), dtype=np.float32)
        for index in range(count):
            first = index // 2
            coefficients = (0.5, 0.5) if index % 2 == 0 else (0.125, 0.75, 0.125)
            if first + len(coefficients) > old_count:
                raise ValueError("target grid is incompatible with half-spacing subdivision")
            weights[index, first:first + len(coefficients)] = coefficients
        moved = np.moveaxis(refined, axis, 0)
        moved = (weights @ moved.reshape(old_count, -1)).reshape(count, *moved.shape[1:])
        refined = np.moveaxis(moved, 0, axis)
    return refined


def _fixed_bins(fixed: Tensor, mask: Tensor, bins: int):
    scaled = fixed[mask].clamp(0, 1) * (bins - 1)
    lower = torch.floor(scaled).long()
    upper = (lower + 1).clamp_max(bins - 1)
    high_weight = scaled - lower
    return lower, upper, 1 - high_weight, high_weight


def _nmi(warped: Tensor, mask: Tensor, fixed_bins, bins: int) -> Tensor:
    """NMI from a differentiable triangular Parzen joint histogram."""
    lower_f, upper_f, low_f, high_f = fixed_bins
    scaled = warped[mask].clamp(0, 1) * (bins - 1)
    lower_m = torch.floor(scaled).long()
    upper_m = (lower_m + 1).clamp_max(bins - 1)
    high_m = scaled - lower_m
    low_m = 1 - high_m
    joint = warped.new_zeros(bins * bins)
    for fi, fw in ((lower_f, low_f), (upper_f, high_f)):
        for mi, mw in ((lower_m, low_m), (upper_m, high_m)):
            joint.scatter_add_(0, fi * bins + mi, fw * mw)
    joint = (joint / len(lower_f)).reshape(bins, bins)

    def entropy(probability: Tensor) -> Tensor:
        return -(probability * probability.clamp_min(1e-12).log()).sum()

    h_joint = entropy(joint)
    return (entropy(joint.sum(0)) + entropy(joint.sum(1))) / h_joint.clamp_min(1e-12)


def _bending(displacement_mm: Tensor, spacing_mm: Tensor) -> Tensor:
    """Mean squared second physical derivatives of the local transform."""
    terms = []
    for axis in range(3):
        terms.append(torch.diff(displacement_mm, n=2, dim=axis).square().sum(-1).mean()
                     / spacing_mm[axis] ** 4)
    for axis_a, axis_b in ((0, 1), (0, 2), (1, 2)):
        mixed = torch.diff(torch.diff(displacement_mm, dim=axis_a), dim=axis_b)
        terms.append(2 * mixed.square().sum(-1).mean()
                     / (spacing_mm[axis_a] * spacing_mm[axis_b]) ** 2)
    return sum(terms)


def _descent(parameter: Tensor, objective, iterations: int, step: float,
             tolerance: float, history: list[float]) -> None:
    """Normalized gradient descent with backtracking and a stopping test."""
    for _ in range(iterations):
        if parameter.grad is not None:
            parameter.grad = None
        current = objective()
        current.backward()
        gradient = parameter.grad
        if gradient is None or not torch.isfinite(gradient).all():
            raise RuntimeError("NMI gradient is nonfinite")
        norm = gradient.norm()
        if norm.item() < 1e-10:
            break
        direction = -gradient / norm
        original = parameter.detach().clone()
        baseline = float(current.detach())
        accepted = False
        trial_step = step
        for _ in range(12):
            with torch.no_grad():
                parameter.copy_(original + trial_step * direction)
                candidate = float(objective())
            if np.isfinite(candidate) and candidate < baseline - tolerance:
                history.append(candidate)
                accepted = True
                break
            trial_step *= 0.5
        if not accepted:
            with torch.no_grad():
                parameter.copy_(original)
            break


def register(
    fixed: np.ndarray, moving: np.ndarray, *,
    voxel_spacing: tuple[float, float, float],
    mask: np.ndarray | None = None,
    final_spacing_mm: float | tuple[float, float, float] = 10.0,
    levels: int = 3, affine_iterations: int = 40, ffd_iterations: int = 80,
    bins: int = 32, max_dim: int = 64, bending_weight: float = 0.01,
    affine_step: float = 0.02, ffd_step_mm: float = 0.25,
    tolerance: float = 1e-5, device: str = "cpu",
    max_output_voxels: int = 20_000_000,
) -> RegistrationResult:
    """Register a 3D pair, returning a full-resolution fixed-to-moving pull field.

    The final control spacing is in mm. Coarser levels double it while image
    resolution increases from coarse to fine. An ROI mask should be supplied
    for a breast-paper experiment; the default uses positive fixed voxels.
    """
    fixed, moving = np.asarray(fixed, np.float32), np.asarray(moving, np.float32)
    if (fixed.ndim != 3 or moving.shape != fixed.shape or min(fixed.shape) < 3
            or not np.isfinite(fixed).all() or not np.isfinite(moving).all()):
        raise ValueError("fixed and moving must be finite 3D arrays on one grid")
    original_moving = moving.copy()
    if fixed.size > max_output_voxels:
        raise ValueError("full-resolution output exceeds max_output_voxels")
    spacing = np.asarray(voxel_spacing, dtype=float)
    final_gap = np.broadcast_to(np.asarray(final_spacing_mm, dtype=float), (3,))
    if (spacing.shape != (3,) or not np.isfinite(spacing).all() or not np.isfinite(final_gap).all()
            or np.any(spacing <= 0) or np.any(final_gap <= 0)
            or levels < 1 or affine_iterations < 0 or ffd_iterations < 0
            or bins < 8 or max_dim < 3 or not np.isfinite(bending_weight)
            or bending_weight < 0 or not np.isfinite(affine_step)
            or not np.isfinite(ffd_step_mm) or not np.isfinite(tolerance)
            or affine_step <= 0 or ffd_step_mm <= 0 or tolerance < 0):
        raise ValueError("invalid registration settings")
    region = fixed > 0 if mask is None else np.asarray(mask, dtype=bool)
    if region.shape != fixed.shape or not region.any():
        raise ValueError("mask must match the fixed image and contain foreground")
    # Histograms use a common intensity range; NMI tolerates contrast changes.
    low = min(float(np.percentile(fixed[region], 1)), float(np.percentile(moving, 1)))
    high = max(float(np.percentile(fixed[region], 99)), float(np.percentile(moving, 99)))
    if high <= low:
        raise ValueError("image intensities have no usable range")
    fixed = np.clip((fixed - low) / (high - low), 0, 1)
    moving = np.clip((moving - low) / (high - low), 0, 1)

    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    shapes, fixed_pyramid, masks = _pyramid(fixed, region, max_dim, levels)
    _, moving_pyramid, _ = _pyramid(moving, region, max_dim, levels)
    if not all(item.any() for item in masks):
        raise ValueError("ROI disappeared at a coarse image resolution")
    history: list[float] = []
    affine = torch.zeros(12, device=torch_device, requires_grad=True)

    def level_data(index: int):
        shape = shapes[index]
        local_spacing = torch.as_tensor(spacing * (np.asarray(fixed.shape) - 1)
                                        / (np.asarray(shape) - 1),
                                        dtype=torch.float32, device=torch_device)
        grid = _grid(shape, torch_device)
        fixed_tensor = torch.as_tensor(fixed_pyramid[index], device=torch_device)
        moving_tensor = torch.as_tensor(moving_pyramid[index], device=torch_device)
        mask_tensor = torch.as_tensor(masks[index], device=torch_device)
        return shape, local_spacing, grid, fixed_tensor, moving_tensor, mask_tensor

    # Global affine registration first, using all image pyramid levels.
    for level in range(levels):
        _, local_spacing, grid, fixed_level, moving_level, region_level = level_data(level)
        bins_fixed = _fixed_bins(fixed_level, region_level, bins)

        def affine_objective():
            warped = _warp(moving_level, grid, _affine_field(affine, grid, local_spacing))
            return -_nmi(warped, region_level, bins_fixed, bins)

        history.append(float(affine_objective().detach()))
        _descent(affine, affine_objective, affine_iterations,
                 affine_step, tolerance, history)

    control_np = None
    for level in range(levels):
        shape, local_spacing, grid, fixed_level, moving_level, region_level = level_data(level)
        bins_fixed = _fixed_bins(fixed_level, region_level, bins)
        gap_mm = final_gap * 2 ** (levels - level - 1)
        # Compute lattice extents in float64 to avoid an extra knot from
        # float32 rounding when the physical field of view is exactly divisible.
        voxel_mm = spacing * (np.asarray(fixed.shape) - 1) / (np.asarray(shape) - 1)
        bases = tuple(torch.as_tensor(bspline_basis(n, float(gap / voxel)),
                                      dtype=torch.float32, device=torch_device)
                      for n, gap, voxel in zip(shape, gap_mm, voxel_mm))
        grid_shape = tuple(basis.shape[1] for basis in bases)
        control_np = (np.zeros((*grid_shape, 3), np.float32) if control_np is None
                      else subdivide_lattice(control_np, grid_shape))
        control = torch.tensor(control_np, device=torch_device, requires_grad=True)
        affine_field = _affine_field(affine.detach(), grid, local_spacing)

        def ffd_objective():
            local_mm = _expand(control, bases)
            warped = _warp(moving_level, grid, affine_field + local_mm / local_spacing)
            return (-_nmi(warped, region_level, bins_fixed, bins)
                    + bending_weight * _bending(local_mm, local_spacing))

        history.append(float(ffd_objective().detach()))
        _descent(control, ffd_objective, ffd_iterations,
                 ffd_step_mm, tolerance, history)
        control_np = control.detach().cpu().numpy()

    theta = affine.detach().cpu().numpy()
    linear = np.eye(3, dtype=np.float32) + theta[:9].reshape(3, 3) * 0.05
    translation_mm = theta[9:] * 5.0
    center_mm = (np.asarray(fixed.shape, np.float32) - 1) * spacing / 2
    affine_matrix = np.eye(4, dtype=np.float32)
    affine_matrix[:3, :3] = linear
    affine_matrix[:3, 3] = translation_mm + center_mm - linear @ center_mm
    field = np.zeros((*fixed.shape, 3), np.float32)
    control, gap_mm = control_np, final_gap
    bases_np = [bspline_basis(n, float(gap / voxel)).astype(np.float32)
                for n, gap, voxel in zip(fixed.shape, gap_mm, spacing)]
    bz, by, bx = bases_np
    for start in range(0, fixed.shape[0], 8):
        stop = min(fixed.shape[0], start + 8)
        local_mm = np.einsum("za,abcd,yb,xc->zyxd", bz[start:stop], control, by, bx,
                             optimize=True)
        field[start:stop] += local_mm / spacing
    for start in range(0, fixed.shape[0], 8):
        stop = min(fixed.shape[0], start + 8)
        zz, yy, xx = np.meshgrid(np.arange(start, stop),
                                 np.arange(fixed.shape[1]), np.arange(fixed.shape[2]),
                                 indexing="ij")
        centered_mm = (np.stack((zz, yy, xx), axis=-1) - (np.asarray(fixed.shape) - 1) / 2) * spacing
        field[start:stop] += ((np.einsum("ij,...j->...i", linear - np.eye(3), centered_mm)
                               + translation_mm) / spacing).astype(np.float32)
    return RegistrationResult(field, warp3d(original_moving, field), history,
                              affine_matrix, tuple(shapes))
