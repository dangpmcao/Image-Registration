"""Train a dataset-specific 2D or 3D CNN registration model."""

import argparse
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import torch

from cnn import (RegistrationCNN, RegistrationCNN3D, bending_energy_torch,
                 bending_energy3d, ncc_loss, ncc_loss3d)
from data.geometry3d import resize_volume
from data.registry import DATASET_FOLDERS, list_pairs, load_pair


def _loss(model, fixed: torch.Tensor, moving: torch.Tensor,
          mask: torch.Tensor, bending: float) -> torch.Tensor:
    field, warped = model(fixed, moving)
    if fixed.ndim == 4:
        similarity, penalty = ncc_loss, bending_energy_torch
    else:
        similarity, penalty = ncc_loss3d, bending_energy3d
    return similarity(fixed, warped, mask) + bending * penalty(field)


def _save(path: Path, model, optimizer, step: int, best_loss: float,
          config: dict, rng: np.random.Generator) -> None:
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "step": step, "best_validation_loss": best_loss, "config": config,
             "rng_state": rng.bit_generator.state}
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASET_FOLDERS, default="oasis_2d")
    parser.add_argument("--root", type=Path, help="Override this dataset's directory")
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--run-name", help="Checkpoint directory under results/<dataset>/cnn")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, help="Default: 4 for 2D, 1 for 3D")
    parser.add_argument("--val-every", type=int, default=200)
    parser.add_argument("--val-pairs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--spacing", type=int, default=32, help="Control spacing on the CNN working grid")
    parser.add_argument("--bending", type=float, default=50.0)
    parser.add_argument("--shape", type=int, nargs="+", help="CNN working shape; default 2D native or 3D capped at --max-dim")
    parser.add_argument("--max-dim", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if (args.steps < 1 or args.val_every < 1 or args.val_pairs < 1 or args.spacing < 2
            or args.max_dim < 3 or (args.batch_size is not None and args.batch_size < 1)
            or not np.isfinite(args.learning_rate) or args.learning_rate <= 0
            or not np.isfinite(args.bending) or args.bending < 0):
        parser.error("invalid positive steps, batch size, validation interval, spacing, or learning rate")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")

    train_specs = list_pairs(args.dataset, "train", root=args.root, seed=args.seed)
    val_specs = list_pairs(args.dataset, "val", root=args.root, seed=args.seed)[:args.val_pairs]
    first = load_pair(train_specs[0])
    dimension = first.fixed.ndim
    if dimension not in (2, 3):
        raise ValueError("CNN supports native 2D and 3D images only")
    shape = tuple(args.shape) if args.shape else (first.fixed.shape if dimension == 2 else
                                                   tuple(min(side, args.max_dim) for side in first.fixed.shape))
    if len(shape) != dimension or min(shape) < 3:
        parser.error("--shape must match the dataset dimension and have sides >= 3")
    batch_size = args.batch_size or (4 if dimension == 2 else 1)

    @lru_cache(maxsize=8)
    def arrays(spec):
        pair = load_pair(spec)
        if pair.fixed.ndim != dimension:
            raise ValueError("Pair dimension differs within this dataset")
        mask = pair.fixed_mask if pair.fixed_mask is not None else pair.fixed > 0
        if dimension == 3:
            return (resize_volume(pair.fixed, shape), resize_volume(pair.moving, shape),
                    resize_volume(mask.astype(np.uint8), shape, order=0).astype(bool))
        if pair.fixed.shape != shape:
            raise ValueError("2D pair shape differs from the CNN working shape")
        return pair.fixed, pair.moving, mask

    def batch(specs):
        fixed, moving, masks = zip(*(arrays(spec) for spec in specs))
        return (torch.from_numpy(np.stack(fixed)[:, None].astype(np.float32)).to(device),
                torch.from_numpy(np.stack(moving)[:, None].astype(np.float32)).to(device),
                torch.from_numpy(np.stack(masks)[:, None]).to(device))

    def validation_loss(model) -> float:
        model.eval()
        losses = []
        with torch.inference_mode():
            for spec in val_specs:
                fixed, moving, mask = batch([spec])
                losses.append(float(_loss(model, fixed, moving, mask, args.bending)))
        model.train()
        return float(np.mean(losses))

    torch.manual_seed(args.seed)
    model = (RegistrationCNN(shape, args.spacing) if dimension == 2 else
             RegistrationCNN3D(shape, args.spacing)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    config = {"dataset": args.dataset, "dimension": dimension, "shape": shape,
              "spacing": args.spacing, "bending": args.bending, "seed": args.seed,
              "learning_rate": args.learning_rate, "batch_size": batch_size,
              "validation_pairs": len(val_specs)}
    run_name = args.run_name or f"train_s{args.spacing}_seed{args.seed}"
    if Path(run_name).name != run_name or run_name in (".", ".."):
        parser.error("--run-name must be a single directory name")
    dataset_output = args.output / args.dataset
    for branch in ("cnn", "b-spline"):
        (dataset_output / branch).mkdir(parents=True, exist_ok=True)
    run_dir = dataset_output / "cnn" / run_name
    if (run_dir / "last.pt").exists() and not args.resume:
        parser.error("run has a checkpoint; use --resume or a new --run-name")
    if args.resume and not (run_dir / "last.pt").is_file():
        parser.error("--resume requires the run's last.pt")
    start, best_loss = 1, float("inf")
    rng = np.random.default_rng(args.seed)
    if args.resume:
        checkpoint = torch.load(run_dir / "last.pt", map_location=device, weights_only=True)
        if checkpoint["config"] != config:
            parser.error("checkpoint config differs from requested training config")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start, best_loss = checkpoint["step"] + 1, checkpoint["best_validation_loss"]
        rng.bit_generator.state = checkpoint["rng_state"]
    run_dir.mkdir(parents=True, exist_ok=True)

    for step in range(start, args.steps + 1):
        indices = rng.integers(len(train_specs), size=batch_size)
        fixed, moving, mask = batch([train_specs[i] for i in indices])
        optimizer.zero_grad(set_to_none=True)
        loss = _loss(model, fixed, moving, mask, args.bending)
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite training loss at step {step}")
        loss.backward()
        optimizer.step()
        if step % args.val_every == 0 or step == args.steps:
            val_loss = validation_loss(model)
            if not np.isfinite(val_loss):
                raise RuntimeError(f"nonfinite validation loss at step {step}")
            if val_loss < best_loss:
                best_loss = val_loss
                _save(run_dir / "best.pt", model, optimizer, step, best_loss, config, rng)
            _save(run_dir / "last.pt", model, optimizer, step, best_loss, config, rng)
            record = {"step": step, "train_loss": float(loss.detach()),
                      "validation_loss": val_loss, "best_validation_loss": best_loss}
            with (run_dir / "history.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
