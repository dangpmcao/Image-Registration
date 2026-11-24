"""Native 3D cubic B-spline FFD for one volume pair.

Volumes use (z,y,x), pull fields use (dz,dy,dx) in voxels. Optimisation may
run on a reduced 3D grid; the result is evaluated on the original grid.
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates

from data.geometry import bspline_basis
from data.geometry3d import resize_field, resize_volume, warp3d


@dataclass
class RegistrationResult:
    field: np.ndarray
    warped: np.ndarray
    history: list[float]
    working_shape: tuple[int, int, int]


def _expand(control: np.ndarray, bases: tuple[np.ndarray, ...]) -> np.ndarray:
    bz, by, bx = bases
    p = np.einsum("za,abcd->zbcd", bz, control, optimize=True)
    p = np.einsum("yb,zbcd->zycd", by, p, optimize=True)
    return np.einsum("xc,zycd->zyxd", bx, p, optimize=True)


def _adjoint(gradient: np.ndarray, bases: tuple[np.ndarray, ...]) -> np.ndarray:
    bz, by, bx = bases
    p = np.einsum("xc,zyxd->zycd", bx, gradient, optimize=True)
    p = np.einsum("yb,zycd->zbcd", by, p, optimize=True)
    return np.einsum("za,zbcd->abcd", bz, p, optimize=True)


def _bending_value_grad(field: np.ndarray) -> tuple[float, np.ndarray]:
    gradient = np.zeros_like(field)
    value = 0.0
    for axis in range(3):
        difference = np.diff(field, n=2, axis=axis)
        value += float(np.mean(difference ** 2))
        scaled = 2 * difference / difference.size
        for position, sign in ((slice(None, -2), 1), (slice(1, -1), -2),
                               (slice(2, None), 1)):
            index = [slice(None)] * 4
            index[axis] = position
            gradient[tuple(index)] += sign * scaled
    for axis_a, axis_b in ((0, 1), (0, 2), (1, 2)):
        difference = np.diff(np.diff(field, axis=axis_a), axis=axis_b)
        value += 2 * float(np.mean(difference ** 2))
        scaled = 4 * difference / difference.size
        for a, b, sign in ((0, 0, 1), (1, 0, -1), (0, 1, -1), (1, 1, 1)):
            index = [slice(None)] * 4
            index[axis_a] = slice(a, -1 if a == 0 else None)
            index[axis_b] = slice(b, -1 if b == 0 else None)
            gradient[tuple(index)] += sign * scaled
    return value, gradient


def _similarity(fixed: np.ndarray, warped: np.ndarray, mask: np.ndarray,
                metric: str) -> tuple[float, np.ndarray]:
    target, source = fixed[mask], warped[mask]
    gradient = np.zeros_like(warped)
    if metric == "ssd":
        difference = source - target
        gradient[mask] = 2 * difference / difference.size
        return float(np.mean(difference ** 2)), gradient
    a, b = target - target.mean(), source - source.mean()
    aa = float(np.dot(a, a))
    bb = float(np.dot(b, b))
    if aa < 1e-12 or bb < 1e-12:
        return 1.0, gradient
    correlation = float(np.dot(a, b) / np.sqrt(aa * bb))
    gradient[mask] = -a / np.sqrt(aa * bb) + correlation * b / bb
    return 1.0 - correlation, gradient


def _value_grad(fixed: np.ndarray, moving: np.ndarray,
                moving_grads: list[np.ndarray], mask: np.ndarray,
                grid: np.ndarray, base: np.ndarray, control: np.ndarray,
                bases: tuple[np.ndarray, ...], metric: str,
                bending_weight: float) -> tuple[float, np.ndarray]:
    field = base + _expand(control, bases)
    coords = grid + np.moveaxis(field, -1, 0)
    warped = map_coordinates(moving, coords, order=1, mode="grid-constant",
                             cval=0, prefilter=False)
    loss, grad_warped = _similarity(fixed, warped, mask, metric)
    sampled_grads = [map_coordinates(g, coords, order=1, mode="grid-constant",
                                     cval=0, prefilter=False) for g in moving_grads]
    grad_field = grad_warped[..., None] * np.stack(sampled_grads, axis=-1)
    if bending_weight:
        penalty, grad_penalty = _bending_value_grad(field)
        loss += bending_weight * penalty
        grad_field += bending_weight * grad_penalty
    return loss, _adjoint(grad_field, bases)


def register(
    fixed: np.ndarray, moving: np.ndarray, *,
    voxel_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    spacing_mm: float | tuple[float, float, float] = 32.0,
    levels: int = 2, iterations: int = 100, metric: str = "ncc",
    bending_weight: float = 50.0, mask: np.ndarray | None = None,
    learning_rate: float = 0.3, max_dim: int = 128,
    max_output_voxels: int = 16_000_000,
) -> RegistrationResult:
    """Fit coarse-to-fine 3D FFD and return a full-grid pull displacement.

    Optimisation uses at most ``max_dim`` voxels on the longest axis. Full-grid
    output above ``max_output_voxels`` is rejected explicitly to avoid OOM.
    """
    fixed, moving = np.asarray(fixed, np.float32), np.asarray(moving, np.float32)
    if (fixed.ndim != 3 or moving.shape != fixed.shape or min(fixed.shape) < 3
            or not np.isfinite(fixed).all() or not np.isfinite(moving).all()):
        raise ValueError("fixed and moving must be finite 3D volumes on the same grid")
    if fixed.size > max_output_voxels:
        raise ValueError(f"volume has {fixed.size:,} voxels; max_output_voxels={max_output_voxels:,}. Downsample/crop explicitly or raise the limit on a memory-rich server")
    voxel_spacing = tuple(float(v) for v in voxel_spacing)
    if isinstance(spacing_mm, (float, int)):
        spacing_mm = (float(spacing_mm),) * 3
    if (len(voxel_spacing) != 3 or len(spacing_mm) != 3
            or any(not np.isfinite(v) or v <= 0 for v in (*voxel_spacing, *spacing_mm))
            or levels < 1 or iterations < 1 or max_dim < 3 or max_output_voxels < 1
            or metric not in ("ncc", "ssd") or not np.isfinite(bending_weight)
            or bending_weight < 0 or not np.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError("invalid spacing, level, iteration, metric, or optimisation parameter")
    mask = np.ones_like(fixed, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if mask.shape != fixed.shape or not mask.any():
        raise ValueError("mask must match volume shape and contain foreground")

    factor = min(1.0, max_dim / max(fixed.shape))
    working_shape = tuple(max(3, round(size * factor)) for size in fixed.shape)
    if working_shape != fixed.shape:
        work_fixed = resize_volume(fixed, working_shape, 1)
        work_moving = resize_volume(moving, working_shape, 1)
        work_mask = resize_volume(mask.astype(np.uint8), working_shape, 0).astype(bool)
    else:
        work_fixed, work_moving, work_mask = fixed, moving, mask
    if not work_mask.any():
        raise ValueError("foreground mask disappeared at working resolution")
    working_spacing = tuple(v * (original - 1) / (reduced - 1)
                            for v, original, reduced in zip(voxel_spacing, fixed.shape, working_shape))
    moving_grads = np.gradient(work_moving)
    grid = np.indices(working_shape, dtype=np.float32)
    field = np.zeros((*working_shape, 3), dtype=np.float32)
    history: list[float] = []
    for level in range(levels):
        lattice_spacing = tuple(max(2.0, mm / voxel / (2 ** level))
                                for mm, voxel in zip(spacing_mm, working_spacing))
        bases = tuple(bspline_basis(size, gap).astype(np.float32)
                      for size, gap in zip(working_shape, lattice_spacing))
        control = np.zeros((*[basis.shape[1] for basis in bases], 3), dtype=np.float32)
        first, second = np.zeros_like(control), np.zeros_like(control)
        best, _ = _value_grad(work_fixed, work_moving, moving_grads, work_mask,
                             grid, field, control, bases, metric, bending_weight)
        best_control = control.copy()
        history.append(best)
        for step in range(1, iterations + 1):
            _, gradient = _value_grad(work_fixed, work_moving, moving_grads, work_mask,
                                      grid, field, control, bases, metric, bending_weight)
            first = 0.9 * first + 0.1 * gradient
            second = 0.999 * second + 0.001 * gradient ** 2
            control -= learning_rate * (first / (1 - 0.9 ** step)) / (np.sqrt(second / (1 - 0.999 ** step)) + 1e-8)
            loss, _ = _value_grad(work_fixed, work_moving, moving_grads, work_mask,
                                  grid, field, control, bases, metric, bending_weight)
            history.append(loss)
            if loss < best:
                best, best_control = loss, control.copy()
        field += _expand(best_control, bases)
    full_field = resize_field(field, fixed.shape)
    return RegistrationResult(full_field, warp3d(moving, full_field), history, working_shape)
