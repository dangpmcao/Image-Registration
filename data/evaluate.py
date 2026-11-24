"""Evaluation and result export shared by B-spline and CNN."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data.pairs import Pair
from data.geometry import dice, jacobian_determinant, warp
from data.geometry3d import jacobian_determinant3d, warp3d


def _ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    a, b = a[mask].astype(float), b[mask].astype(float)
    a -= a.mean()
    b -= b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denominator) if denominator > 0 else 0.0


def _dice(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    scores = np.fromiter(dice(a, b).values(), dtype=float)
    finite = scores[np.isfinite(scores)]
    return float(finite.mean()) if finite.size else None


def _preview(path: Path, fixed: np.ndarray, moving: np.ndarray,
             warped: np.ndarray, determinant: np.ndarray, mask: np.ndarray) -> None:
    if fixed.ndim == 3:
        z = int(np.argmax(mask.sum(axis=(1, 2))))
        fixed, moving, warped, determinant = (array[z] for array in
                                              (fixed, moving, warped, determinant))
    panels = [fixed, moving, warped, abs(fixed - moving),
              abs(fixed - warped), determinant]
    titles = ["Fixed", "Moving", "Registered", "Before: |difference|",
              "After: |difference|", "det(J)"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 7), constrained_layout=True)
    for ax, panel, title in zip(axes.flat, panels, titles):
        ax.imshow(panel, cmap="coolwarm" if title == "det(J)" else "gray")
        ax.set_title(title)
        ax.axis("off")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_result(pair: Pair, field: np.ndarray, warped: np.ndarray, *,
                method: str, split: str, seed: int, config: dict,
                seconds: float, output: Path) -> dict:
    """Compute comparable metrics and write one result directory."""
    if pair.fixed.ndim not in (2, 3) or field.shape != (*pair.fixed.shape, pair.fixed.ndim):
        raise ValueError("field must match the pair's native image shape and dimension")
    if (warped.shape != pair.fixed.shape or not np.isfinite(warped).all()
            or not np.isfinite(field).all()):
        raise ValueError("warped image or field has invalid shape or values")
    mask = (pair.fixed_mask > 0 if pair.fixed_mask is not None else pair.fixed > 0)
    if not mask.any():
        raise ValueError("fixed image has no evaluation foreground")
    if pair.fixed.ndim == 2:
        determinant = jacobian_determinant(field)
        warped_seg = warp(pair.moving_seg, field, order=0) if pair.moving_seg is not None else None
    else:
        determinant = jacobian_determinant3d(field)
        warped_seg = warp3d(pair.moving_seg, field, order=0) if pair.moving_seg is not None else None
    field_error = None
    if pair.ground_truth_field is not None:
        difference_mm = (field - pair.ground_truth_field) * np.asarray(pair.spacing)
        field_error = float(np.linalg.norm(difference_mm, axis=-1)[mask].mean())
    metrics = {
        "dataset": pair.dataset, "fixed": pair.fixed_id, "moving": pair.moving_id,
        "method": method, "split": split, "split_seed": seed,
        "spacing_mm": pair.spacing,
        "field_convention": "fixed-grid pull displacement in array order, voxel units",
        "config": config,
        "ncc_before": _ncc(pair.fixed, pair.moving, mask),
        "ncc_after": _ncc(pair.fixed, warped, mask),
        "dice_before": _dice(pair.fixed_seg, pair.moving_seg),
        "dice_after": _dice(pair.fixed_seg, warped_seg),
        "folding_percent_foreground": float(100 * np.mean(determinant[mask] <= 0)),
        "folding_percent_full": float(100 * np.mean(determinant <= 0)),
        "field_error_mm": field_error,
        "registration_seconds": seconds,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n",
                                         encoding="utf-8")
    np.savez_compressed(output / "field.npz", field=field.astype(np.float32))
    _preview(output / "preview.png", pair.fixed, pair.moving, warped, determinant, mask)
    return metrics
