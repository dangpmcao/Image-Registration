"""Run the first reproducible 2D OASIS B-spline experiment."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common.data import load_pair, split_subjects, subject_ids
from common.geometry import dice, jacobian_determinant, warp
from b_spline.ffd import register


def _ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    a, b = a[mask].astype(float), b[mask].astype(float)
    a -= a.mean()
    b -= b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denominator) if denominator > 0 else 0.0


def _mean_dice(fixed: np.ndarray, moving: np.ndarray, labels: np.ndarray) -> float:
    scores = np.array(list(dice(fixed, moving, labels).values()), dtype=float)
    return float(np.mean(scores[np.isfinite(scores)])) if np.isfinite(scores).any() else float("nan")


def _save_preview(path: Path, fixed: np.ndarray, moving: np.ndarray,
                  warped: np.ndarray, determinant: np.ndarray) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="OASIS 2D directory (defaults to ../dataset/oasis_2d_neurite)")
    parser.add_argument("--fixed", help="Fixed subject ID; defaults to first subject in the validation split")
    parser.add_argument("--moving", help="Moving subject ID; defaults to second subject in the validation split")
    parser.add_argument("--output", type=Path, default=Path("results/oasis_2d"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--spacing", type=int, default=32, help="Coarsest control grid spacing in voxels")
    parser.add_argument("--levels", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=100, help="Iterations per level")
    parser.add_argument("--metric", choices=("ncc", "ssd"), default="ncc")
    parser.add_argument("--bending", type=float, default=50.0)
    parser.add_argument("--learning-rate", type=float, default=0.3)
    args = parser.parse_args()

    if (args.fixed is None) != (args.moving is None):
        parser.error("supply both --fixed and --moving, or neither")
    splits = split_subjects(subject_ids(args.root), seed=args.seed)
    fixed_id = args.fixed or splits["val"][0]
    moving_id = args.moving or splits["val"][1]
    if fixed_id == moving_id:
        parser.error("fixed and moving must be different subjects")
    pair_split = next((name for name, ids in splits.items()
                       if fixed_id in ids and moving_id in ids), None)
    if pair_split is None:
        parser.error("fixed and moving must both belong to the same subject split")
    pair = load_pair(args.root, fixed_id, moving_id)
    mask = pair.fixed > 0
    if not mask.any():
        parser.error("fixed image has no nonzero foreground")

    start = perf_counter()
    result = register(pair.fixed, pair.moving, spacing=args.spacing,
                      levels=args.levels, iterations=args.iterations,
                      metric=args.metric, bending_weight=args.bending,
                      learning_rate=args.learning_rate, mask=mask)
    elapsed = perf_counter() - start
    warped_seg = warp(pair.moving_seg, result.field, order=0).astype(pair.fixed_seg.dtype)
    determinant = jacobian_determinant(result.field)
    labels = np.union1d(np.unique(pair.fixed_seg), np.unique(pair.moving_seg))
    labels = labels[labels != 0]
    metrics = {
        "fixed": fixed_id,
        "moving": moving_id,
        "split": pair_split,
        "split_seed": args.seed,
        "spacing_mm": pair.spacing,
        "field_convention": "fixed-grid pull displacement (dy, dx), voxel units",
        "config": {"spacing_voxels": args.spacing, "levels": args.levels,
                   "iterations_per_level": args.iterations, "metric": args.metric,
                   "bending_weight": args.bending, "learning_rate": args.learning_rate},
        "ncc_before": _ncc(pair.fixed, pair.moving, mask),
        "ncc_after": _ncc(pair.fixed, result.warped, mask),
        "dice_before": _mean_dice(pair.fixed_seg, pair.moving_seg, labels),
        "dice_after": _mean_dice(pair.fixed_seg, warped_seg, labels),
        "folding_percent_foreground": float(100 * np.mean(determinant[mask] <= 0)),
        "folding_percent_full": float(100 * np.mean(determinant <= 0)),
        "registration_seconds": elapsed,
    }
    setting = (f"{args.metric}_s{args.spacing}_l{args.levels}_i{args.iterations}"
               f"_b{args.bending:g}_lr{args.learning_rate:g}")
    out = args.output / f"{fixed_id}__{moving_id}" / setting
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    np.savez_compressed(out / "field.npz", field=result.field.astype(np.float32))
    _save_preview(out / "preview.png", pair.fixed, pair.moving, result.warped, determinant)
    print(json.dumps(metrics, indent=2, allow_nan=False))
    print(f"Saved results to {out.resolve()}")


if __name__ == "__main__":
    main()
