"""2D cubic B-spline free-form deformation, optimized per image pair."""

from dataclasses import dataclass

import numpy as np

from data.geometry import _check_image, _samples, bspline_basis, warp


@dataclass
class RegistrationResult:
    field: np.ndarray
    warped: np.ndarray
    history: list[float]


def bspline_field(control: np.ndarray, shape: tuple[int, int], spacing: tuple[float, float]) -> np.ndarray:
    """Expand a (control_y, control_x, 2) lattice to a dense displacement."""
    by, bx = bspline_basis(shape[0], spacing[0]), bspline_basis(shape[1], spacing[1])
    control = np.asarray(control, dtype=np.float64)
    if control.shape != (by.shape[1], bx.shape[1], 2) or not np.isfinite(control).all():
        raise ValueError("control lattice must match image shape and spacing and be finite")
    return np.stack([by @ control[..., axis] @ bx.T for axis in range(2)], axis=-1)


def _bending_value_grad(field: np.ndarray) -> tuple[float, np.ndarray]:
    """Discrete 2D bending energy and its gradient in voxel coordinates."""
    dyy = field[2:] - 2 * field[1:-1] + field[:-2]
    dxx = field[:, 2:] - 2 * field[:, 1:-1] + field[:, :-2]
    dyx = field[1:, 1:] - field[1:, :-1] - field[:-1, 1:] + field[:-1, :-1]
    value = float(np.mean(dyy ** 2) + np.mean(dxx ** 2) + 2 * np.mean(dyx ** 2))
    grad = np.zeros_like(field)
    syy = 2 * dyy / dyy.size
    sxx = 2 * dxx / dxx.size
    syx = 4 * dyx / dyx.size
    grad[2:] += syy
    grad[1:-1] -= 2 * syy
    grad[:-2] += syy
    grad[:, 2:] += sxx
    grad[:, 1:-1] -= 2 * sxx
    grad[:, :-2] += sxx
    grad[1:, 1:] += syx
    grad[1:, :-1] -= syx
    grad[:-1, 1:] -= syx
    grad[:-1, :-1] += syx
    return value, grad


def bending_energy(field: np.ndarray) -> float:
    """Mean squared second differences; zero for an affine displacement."""
    field = np.asarray(field, dtype=np.float64)
    if field.ndim != 3 or field.shape[-1] != 2 or min(field.shape[:2]) < 3 or not np.isfinite(field).all():
        raise ValueError("field must be finite with shape (height, width, 2), at least 3x3")
    return _bending_value_grad(field)[0]


def _similarity(fixed: np.ndarray, warped: np.ndarray, mask: np.ndarray, metric: str) -> tuple[float, np.ndarray]:
    target, source = fixed[mask], warped[mask]
    grad = np.zeros_like(warped)
    if metric == "ssd":
        delta = source - target
        grad[mask] = 2 * delta / delta.size
        return float(np.mean(delta ** 2)), grad
    if metric != "ncc":
        raise ValueError("metric must be 'ncc' or 'ssd'")
    a, b = target - target.mean(), source - source.mean()
    aa, bb = float(a @ a), float(b @ b)
    if aa < 1e-12 or bb < 1e-12:
        return 1.0, grad
    correlation = float((a @ b) / np.sqrt(aa * bb))
    grad[mask] = -a / np.sqrt(aa * bb) + correlation * b / bb
    return 1.0 - correlation, grad


def _value_and_grad(
    fixed: np.ndarray, moving: np.ndarray, mask: np.ndarray,
    base: np.ndarray, control: np.ndarray, by: np.ndarray, bx: np.ndarray,
    metric: str, bending_weight: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    field = base + np.stack([by @ control[..., axis] @ bx.T for axis in range(2)], axis=-1)
    warped, grad_y, grad_x = _samples(moving, field)
    loss, grad_warped = _similarity(fixed, warped, mask, metric)
    grad_field = grad_warped[..., None] * np.stack((grad_y, grad_x), axis=-1)
    if bending_weight:
        penalty, grad_penalty = _bending_value_grad(field)
        loss += bending_weight * penalty
        grad_field += bending_weight * grad_penalty
    grad_control = np.stack([by.T @ grad_field[..., axis] @ bx for axis in range(2)], axis=-1)
    return loss, grad_control, field


def register(
    fixed: np.ndarray,
    moving: np.ndarray,
    *,
    spacing: int | tuple[int, int] = (32, 32),
    levels: int = 2,
    iterations: int = 100,
    metric: str = "ncc",
    bending_weight: float = 50.0,
    mask: np.ndarray | None = None,
    learning_rate: float = 0.3,
) -> RegistrationResult:
    """Register one 2D pair using additive coarse-to-fine cubic B-spline FFD.

    ``spacing`` is the coarsest control spacing; each next level halves it.
    An independent residual lattice is fitted at each level on full-resolution
    images. The best loss at each level is retained, including the zero update.
    """
    fixed, moving = _check_image(fixed, "fixed"), _check_image(moving, "moving")
    if fixed.shape != moving.shape:
        raise ValueError("fixed and moving must have the same shape")
    if isinstance(spacing, int):
        spacing = (spacing, spacing)
    if len(spacing) != 2 or any(not np.isfinite(s) or s < 2 for s in spacing):
        raise ValueError("spacing must contain two finite values >= 2")
    if (levels < 1 or iterations < 1 or not np.isfinite(learning_rate)
            or not np.isfinite(bending_weight) or learning_rate <= 0 or bending_weight < 0):
        raise ValueError("levels, iterations and learning_rate must be positive and finite; bending_weight finite and >= 0")
    if metric not in ("ncc", "ssd"):
        raise ValueError("metric must be 'ncc' or 'ssd'")
    if mask is None:
        mask = np.ones(fixed.shape, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != fixed.shape or not mask.any():
            raise ValueError("mask must match image shape and contain at least one voxel")

    field = np.zeros((*fixed.shape, 2), dtype=np.float64)
    history: list[float] = []
    for level in range(levels):
        level_spacing = tuple(max(2.0, s / (2 ** level)) for s in spacing)
        by, bx = (bspline_basis(fixed.shape[0], level_spacing[0]),
                  bspline_basis(fixed.shape[1], level_spacing[1]))
        control = np.zeros((by.shape[1], bx.shape[1], 2), dtype=np.float64)
        first_moment = np.zeros_like(control)
        second_moment = np.zeros_like(control)
        best_loss, _, _ = _value_and_grad(fixed, moving, mask, field, control, by, bx, metric, bending_weight)
        best_control = control.copy()
        history.append(best_loss)
        for step in range(1, iterations + 1):
            _, gradient, _ = _value_and_grad(fixed, moving, mask, field, control, by, bx, metric, bending_weight)
            first_moment = 0.9 * first_moment + 0.1 * gradient
            second_moment = 0.999 * second_moment + 0.001 * gradient ** 2
            corrected_first = first_moment / (1 - 0.9 ** step)
            corrected_second = second_moment / (1 - 0.999 ** step)
            control -= learning_rate * corrected_first / (np.sqrt(corrected_second) + 1e-8)
            loss, _, _ = _value_and_grad(fixed, moving, mask, field, control, by, bx, metric, bending_weight)
            history.append(loss)
            if loss < best_loss:
                best_loss = loss
                best_control = control.copy()
        field += np.stack([by @ best_control[..., axis] @ bx.T for axis in range(2)], axis=-1)
    return RegistrationResult(field=field, warped=warp(moving, field), history=history)
