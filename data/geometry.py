"""Shared cubic B-spline basis and 2D image geometry."""

import numpy as np


def bspline_basis(length: int, spacing: float) -> np.ndarray:
    """Cubic cardinal B-spline basis; lattice centers start at -spacing."""
    if length < 1 or not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("image length and grid spacing must be positive and finite")
    positions = np.arange(length, dtype=np.float64) / spacing
    cell = np.floor(positions).astype(int)
    u = positions - cell
    weights = np.stack(((1 - u) ** 3 / 6,
                        (3 * u ** 3 - 6 * u ** 2 + 4) / 6,
                        (-3 * u ** 3 + 3 * u ** 2 + 3 * u + 1) / 6,
                        u ** 3 / 6), axis=1)
    basis = np.zeros((length, cell[-1] + 4), dtype=np.float64)
    rows = np.arange(length)
    for k in range(4):
        basis[rows, cell + k] = weights[:, k]
    return basis


def _check_image(image: np.ndarray, name: str) -> np.ndarray:
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2 or min(image.shape) < 3 or not np.isfinite(image).all():
        raise ValueError(f"{name} must be a finite 2D image of at least 3x3")
    return image


def _check_field(field: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    field = np.asarray(field, dtype=np.float64)
    if field.shape != (*shape, 2) or not np.isfinite(field).all():
        raise ValueError("field must have shape (height, width, 2) and be finite")
    return field


def _samples(image: np.ndarray, field: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bilinear values and coordinate derivatives, using zero padding."""
    height, width = image.shape
    yy, xx = np.indices(image.shape, dtype=np.float64)
    qy = yy + field[..., 0]
    qx = xx + field[..., 1]
    y0 = np.floor(qy).astype(np.int64)
    x0 = np.floor(qx).astype(np.int64)
    fy = qy - y0
    fx = qx - x0

    def at(y: np.ndarray, x: np.ndarray) -> np.ndarray:
        valid = (y >= 0) & (y < height) & (x >= 0) & (x < width)
        return np.where(valid, image[np.clip(y, 0, height - 1), np.clip(x, 0, width - 1)], 0.0)

    v00 = at(y0, x0)
    v01 = at(y0, x0 + 1)
    v10 = at(y0 + 1, x0)
    v11 = at(y0 + 1, x0 + 1)
    value = (1 - fy) * ((1 - fx) * v00 + fx * v01) + fy * ((1 - fx) * v10 + fx * v11)
    grad_y = (1 - fx) * (v10 - v00) + fx * (v11 - v01)
    grad_x = (1 - fy) * (v01 - v00) + fy * (v11 - v10)
    return value, grad_y, grad_x


def warp(image: np.ndarray, field: np.ndarray, *, order: int = 1) -> np.ndarray:
    """Pull moving into fixed space at (y + dy, x + dx); order 0 for labels."""
    input_dtype = np.asarray(image).dtype
    image = _check_image(image, "image")
    field = _check_field(field, image.shape)
    if order == 1:
        return _samples(image, field)[0]
    if order != 0:
        raise ValueError("order must be 0 or 1")
    yy, xx = np.indices(image.shape, dtype=np.float64)
    qy = np.floor(yy + field[..., 0] + 0.5).astype(np.int64)
    qx = np.floor(xx + field[..., 1] + 0.5).astype(np.int64)
    valid = (qy >= 0) & (qy < image.shape[0]) & (qx >= 0) & (qx < image.shape[1])
    sampled = image[np.clip(qy, 0, image.shape[0] - 1), np.clip(qx, 0, image.shape[1] - 1)]
    return np.where(valid, sampled, 0).astype(input_dtype)


def jacobian_determinant(field: np.ndarray) -> np.ndarray:
    """det(d(x + displacement)/dx), where x is ordered (y, x)."""
    field = np.asarray(field, dtype=np.float64)
    if field.ndim != 3 or field.shape[-1] != 2 or min(field.shape[:2]) < 3 or not np.isfinite(field).all():
        raise ValueError("field must be finite with shape (height, width, 2), at least 3x3")
    uy_y, uy_x = np.gradient(field[..., 0])
    ux_y, ux_x = np.gradient(field[..., 1])
    return (1 + uy_y) * (1 + ux_x) - uy_x * ux_y


def dice(fixed_labels: np.ndarray, warped_labels: np.ndarray, labels=None) -> dict[int, float]:
    """Per-label Dice, excluding background 0 by default; absent labels are NaN."""
    fixed_labels, warped_labels = np.asarray(fixed_labels), np.asarray(warped_labels)
    if fixed_labels.shape != warped_labels.shape:
        raise ValueError("label arrays must have the same shape")
    if labels is None:
        labels = np.union1d(np.unique(fixed_labels), np.unique(warped_labels))
        labels = labels[labels != 0]
    scores = {}
    for label in labels:
        a, b = fixed_labels == label, warped_labels == label
        count = int(a.sum() + b.sum())
        scores[int(label)] = float(2 * np.count_nonzero(a & b) / count) if count else float("nan")
    return scores
