"""Sequential DLIR training, following paper/cnn.pdf Sections 2, 5 and 6.

Variable-size pairs are accumulated one at a time into a logical mini-batch;
this preserves physical spacing without padding all subjects to one shape.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from cnn.devos import DLIR3D, bending_energy_mm, nlst_stage_specs, sunnybrook_stage_specs
from cnn.dlir_data import augment, load_dlir_pair, pair_specs, tensors
from cnn.model3d import ncc_loss3d
from data.registry import DATASET_FOLDERS


def stage_loss(model, fixed, moving, spacing, stage: int, bending: float):
    _, _, details = model(fixed, moving, voxel_spacing=spacing, upto_stage=stage,
                           return_details=True)
    # Full image NCC, not a segmentation/foreground-weighted training objective.
    loss = ncc_loss3d(details["fixed"], details["warped"], torch.ones_like(details["fixed"])) - 1
    if model.stage_specs[stage]["kind"] == "deformable" and bending:
        loss = loss + bending * bending_energy_mm(details["local_field"], details["spacing"])
    return loss


def save_checkpoint(path, model, optimizer, config, stage, step, best, rng):
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "config": config, "stage": stage, "step": step,
             "best_validation_loss": best, "rng_state": rng.bit_generator.state,
             "torch_rng_state": torch.get_rng_state(),
             "cuda_rng_states": (torch.cuda.get_rng_state_all()
                                 if next(model.parameters()).device.type == "cuda" else [])}
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASET_FOLDERS, required=True)
    parser.add_argument("--preset", choices=("sunnybrook-single", "sunnybrook-multi", "chest-ct", "custom"),
                        required=True)
    parser.add_argument("--stages-config", type=Path, help="JSON list of stage specifications for custom preset")
    parser.add_argument("--fold", type=int, choices=(1, 2, 3))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--steps-per-stage", type=int, nargs="+", default=[10000])
    parser.add_argument("--batch-size", type=int, nargs="+", help="Logical batch size per stage")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="Explicit implementation choice; the paper does not specify Adam's LR")
    parser.add_argument("--bending", type=float, default=0.05)
    parser.add_argument("--val-every", type=int, default=200)
    parser.add_argument("--val-pairs", type=int, default=8)
    parser.add_argument("--crop-shape", type=int, nargs=3, help="Additional training crop z y x; changes paper FOV")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if Path(args.run_name).name != args.run_name or args.run_name in (".", ".."):
        parser.error("--run-name must be a single directory name")
    if (args.val_every < 1 or args.val_pairs < 1 or args.learning_rate <= 0
            or not np.isfinite(args.learning_rate) or args.bending < 0 or not np.isfinite(args.bending)
            or (args.crop_shape is not None and min(args.crop_shape) < 6)):
        parser.error("invalid training settings; crops must have sides >= 6")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable; use a scheduler-assigned GPU")
    if args.preset.startswith("sunnybrook"):
        specs = sunnybrook_stage_specs(args.preset.removeprefix("sunnybrook-"))
        default_batch = [8] if len(specs) == 1 else [16, 8]
    elif args.preset == "chest-ct":
        specs, default_batch = nlst_stage_specs(), [16, 8, 4, 2]
    else:
        if args.stages_config is None:
            parser.error("--preset custom requires --stages-config")
        specs = json.loads(args.stages_config.read_text())
        default_batch = [1] * len(specs)

    def expand(values, name):
        if len(values) == 1:
            values = values * len(specs)
        if len(values) != len(specs) or min(values) < 1:
            parser.error(f"{name} expects one positive integer or one per stage")
        return values

    budgets = expand(args.steps_per_stage, "--steps-per-stage")
    batch_sizes = expand(args.batch_size or default_batch, "--batch-size")
    train_specs = pair_specs(args.dataset, "train", args.preset, args.root, args.seed, args.fold)
    val_specs = pair_specs(args.dataset, "val", args.preset, args.root, args.seed, args.fold)[:args.val_pairs]
    manifest = {"train": [asdict(spec) for spec in train_specs],
                "validation": [asdict(spec) for spec in val_specs]}
    manifest_text = json.dumps(manifest, default=str, sort_keys=True, indent=2) + "\n"
    paper_dir = Path(__file__).resolve().parents[1] / "paper"
    paper_hashes = {name: hashlib.sha256((paper_dir / name).read_bytes()).hexdigest()
                    for name in ("cnn.pdf", "b-spline.pdf") if (paper_dir / name).is_file()}
    source_root = paper_dir.parent
    source_files = [source_root / "train.py", *sorted((source_root / "cnn").glob("*.py")),
                    *sorted((source_root / "data").glob("*.py"))]
    source_hashes = {str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in source_files}
    config = {"architecture": "dlir", "format_version": 1, "dataset": args.dataset,
              "preset": args.preset, "stage_specs": specs, "fold": args.fold, "seed": args.seed,
              "steps_per_stage": budgets, "batch_sizes": batch_sizes,
              "learning_rate": args.learning_rate, "bending": args.bending,
              "crop_shape": args.crop_shape, "val_every": args.val_every,
              "pair_manifest_sha256": hashlib.sha256(manifest_text.encode()).hexdigest(),
              "paper_sha256": paper_hashes,
              "source_sha256": source_hashes, "torch_version": str(torch.__version__),
              "validation_protocol": ("training-subject ED/ES monitoring; not independent"
                                      if args.preset.startswith("sunnybrook") else "subject-disjoint validation"),
              "cohort_note": ("released Sunnybrook parts used as reproducible folds"
                              if args.preset.startswith("sunnybrook") else
                              "local dataset adaptation; not the original NLST/DIR-Lab experiment")}
    run = args.output / args.dataset / "dlir" / args.run_name
    if run.exists() and any(run.iterdir()) and not args.resume:
        parser.error("run directory is nonempty; use --resume or a new --run-name")
    if args.resume and not (run / "last.pt").is_file():
        parser.error("--resume requires last.pt")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = DLIR3D(specs).to(device)
    resume = None
    if args.resume:
        resume = torch.load(run / "last.pt", map_location=device, weights_only=True)
        if resume["config"] != config:
            parser.error("resume configuration, pair manifest, or paper hashes differ")
        model.load_state_dict(resume["model"])
        rng.bit_generator.state = resume["rng_state"]
        torch.set_rng_state(resume["torch_rng_state"].cpu())
        if device.type == "cuda" and resume["cuda_rng_states"]:
            torch.cuda.set_rng_state_all([state.cpu() for state in resume["cuda_rng_states"]])
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (run / "pairs.json").write_text(manifest_text)

    def validation(stage):
        model.eval()
        values = []
        with torch.no_grad():
            for spec in val_specs:
                fixed, moving, spacing = tensors(load_dlir_pair(spec, args.preset), args.preset, device)
                # A smoke crop is also applied deterministically to validation.
                if args.crop_shape:
                    slices = tuple(slice(0, min(n, limit)) for n, limit in zip(fixed.shape[-3:], args.crop_shape))
                    fixed, moving = (image[(slice(None), slice(None), *slices)] for image in (fixed, moving))
                values.append(float(stage_loss(model, fixed, moving, spacing, stage, args.bending)))
        model.train()
        return float(np.mean(values))

    first_stage = resume["stage"] if resume else 0
    for stage in range(first_stage, len(specs)):
        model.train_only_stage(stage)
        optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
        start, best = 1, float("inf")
        if resume is not None and stage == first_stage:
            optimizer.load_state_dict(resume["optimizer"])
            start, best = resume["step"] + 1, resume["best_validation_loss"]
        for step in range(start, budgets[stage] + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            train_loss = 0.0
            for index in rng.integers(len(train_specs), size=batch_sizes[stage]):
                fixed, moving, spacing = tensors(load_dlir_pair(train_specs[int(index)], args.preset),
                                                 args.preset, device)
                fixed, moving, spacing = augment(fixed, moving, spacing, rng, args.preset,
                                                 tuple(args.crop_shape) if args.crop_shape else None)
                loss = stage_loss(model, fixed, moving, spacing, stage, args.bending)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"nonfinite loss at stage={stage}, step={step}")
                (loss / batch_sizes[stage]).backward()
                train_loss += float(loss.detach()) / batch_sizes[stage]
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError(f"nonfinite gradient at stage={stage}, step={step}")
            optimizer.step()
            if step % args.val_every == 0 or step == budgets[stage]:
                val_loss = validation(stage)
                if not np.isfinite(val_loss):
                    raise RuntimeError("nonfinite validation loss")
                if val_loss < best:
                    best = val_loss
                    save_checkpoint(run / f"stage{stage}_best.pt", model, optimizer, config, stage, step, best, rng)
                    save_checkpoint(run / "best.pt", model, optimizer, config, stage, step, best, rng)
                save_checkpoint(run / "last.pt", model, optimizer, config, stage, step, best, rng)
                record = {"stage": stage, "step": step, "train_loss": train_loss,
                          "validation_loss": val_loss, "best_validation_loss": best}
                with (run / "history.jsonl").open("a") as stream:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                print(json.dumps(record), flush=True)
        # Freeze the selected preceding stage before training the next one.
        selected = torch.load(run / f"stage{stage}_best.pt", map_location=device, weights_only=True)
        model.load_state_dict(selected["model"])
    print(f"Completed all {len(specs)} stages. Checkpoint: {run / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
