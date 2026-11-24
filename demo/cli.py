"""Register one dataset pair with B-spline or a trained CNN."""

import argparse
import hashlib
from importlib import import_module
import json
from pathlib import Path
from time import perf_counter

from data.pairs import load_pair as load_oasis_pair, split_subjects, subject_ids
from data.evaluate import save_result
from data.geometry3d import resize_field, resize_volume, warp3d
from data.registry import DATASET_FOLDERS, dataset_root, list_pairs, load_pair


def _cnn_infer(pair, checkpoint_path: Path, device_name: str,
               max_output_voxels: int, split_seed: int):
    import torch
    from cnn import RegistrationCNN, RegistrationCNN3D

    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else
                          "cpu" if device_name == "auto" else device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = checkpoint["config"]
    if config.get("dataset", "oasis_2d") != pair.dataset:
        raise ValueError("checkpoint was trained on a different dataset")
    if config["seed"] != split_seed:
        raise ValueError("--seed must match the CNN checkpoint's split seed")
    shape = tuple(config["shape"])
    if len(shape) != pair.fixed.ndim:
        raise ValueError("checkpoint and pair dimensions differ")
    if pair.fixed.size > max_output_voxels:
        raise ValueError("full-resolution field exceeds --max-output-voxels")
    model_type = RegistrationCNN if pair.fixed.ndim == 2 else RegistrationCNN3D
    model = model_type(shape, config["spacing"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    start = perf_counter()
    if pair.fixed.ndim == 3:
        fixed_image = resize_volume(pair.fixed, shape)
        moving_image = resize_volume(pair.moving, shape)
    else:
        if pair.fixed.shape != shape:
            raise ValueError("2D checkpoint image shape differs from this dataset")
        fixed_image, moving_image = pair.fixed, pair.moving
    fixed = torch.from_numpy(fixed_image[None, None]).to(device)
    moving = torch.from_numpy(moving_image[None, None]).to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    with torch.inference_mode():
        field_tensor, warped_tensor = model(fixed, moving)
    if device.type == "cuda":
        torch.cuda.synchronize()
    field = field_tensor[0].cpu().numpy()
    if pair.fixed.ndim == 3:
        field = resize_field(field, pair.fixed.shape)
        warped = warp3d(pair.moving, field)
    else:
        warped = warped_tensor[0, 0].cpu().numpy()
    elapsed = perf_counter() - start
    checkpoint_id = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()[:12]
    return field, warped, elapsed, {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_step": checkpoint["step"], "training": config,
        "device": str(device), "working_shape": shape,
    }, f"ckpt_{checkpoint_id}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASET_FOLDERS, default="oasis_2d")
    parser.add_argument("--method", choices=("b-spline", "cnn"), default="b-spline")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--pair-index", type=int, default=0)
    parser.add_argument("--fixed", help="Optional OASIS 2D fixed subject ID")
    parser.add_argument("--moving", help="Optional OASIS 2D moving subject ID")
    parser.add_argument("--root", type=Path, help="Override this dataset's directory")
    parser.add_argument("--output", type=Path, default=Path("results"),
                        help="Root result directory; dataset and method are appended")
    parser.add_argument("--checkpoint", type=Path, help="CNN best.pt or last.pt")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--spacing", type=int, default=32, help="2D control spacing in voxels")
    parser.add_argument("--spacing-mm", type=float, nargs="+", default=[32.0],
                        help="One or three 3D control spacings in millimetres (z y x)")
    parser.add_argument("--levels", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--metric", choices=("ncc", "ssd"), default="ncc")
    parser.add_argument("--bending", type=float, default=50.0)
    parser.add_argument("--learning-rate", type=float, default=0.3)
    parser.add_argument("--max-dim", type=int, default=128,
                        help="Largest 3D working-grid side")
    parser.add_argument("--max-output-voxels", type=int, default=20_000_000)
    args = parser.parse_args()
    if len(args.spacing_mm) not in (1, 3):
        parser.error("--spacing-mm expects one value or z y x values")
    spacing_mm = args.spacing_mm[0] if len(args.spacing_mm) == 1 else tuple(args.spacing_mm)

    if (args.fixed is None) != (args.moving is None):
        parser.error("supply both --fixed and --moving, or neither")
    if args.fixed is not None:
        if args.dataset != "oasis_2d":
            parser.error("--fixed/--moving currently apply only to oasis_2d")
        ids = split_subjects(subject_ids(dataset_root(args.dataset, args.root)), seed=args.seed)[args.split]
        if args.fixed == args.moving or args.fixed not in ids or args.moving not in ids:
            parser.error("fixed and moving must be distinct subjects in the selected split")
        pair = load_oasis_pair(dataset_root(args.dataset, args.root), args.fixed, args.moving)
        pair_id = f"{args.fixed}__{args.moving}"
    else:
        specs = list_pairs(args.dataset, args.split, root=args.root, seed=args.seed)
        if not 0 <= args.pair_index < len(specs):
            parser.error(f"--pair-index must be between 0 and {len(specs) - 1}")
        spec = specs[args.pair_index]
        pair, pair_id = load_pair(spec), spec.pair_id

    if args.method == "cnn":
        if args.checkpoint is None or not args.checkpoint.is_file():
            parser.error("--method cnn requires an existing --checkpoint")
        field, warped, elapsed, config, setting = _cnn_infer(
            pair, args.checkpoint, args.device, args.max_output_voxels, args.seed)
    else:
        if args.checkpoint is not None:
            parser.error("--checkpoint only applies to CNN")
        mask = pair.fixed_mask if pair.fixed_mask is not None else pair.fixed > 0
        start = perf_counter()
        if pair.fixed.ndim == 2:
            register = import_module("b-spline.ffd").register
            result = register(pair.fixed, pair.moving, spacing=args.spacing,
                              levels=args.levels, iterations=args.iterations,
                              metric=args.metric, bending_weight=args.bending,
                              learning_rate=args.learning_rate, mask=mask)
            setting = (f"{args.metric}_s{args.spacing}_l{args.levels}_i{args.iterations}"
                       f"_b{args.bending:g}_lr{args.learning_rate:g}")
        else:
            register = import_module("b-spline.ffd3d").register
            result = register(pair.fixed, pair.moving, voxel_spacing=pair.spacing,
                              spacing_mm=spacing_mm, levels=args.levels,
                              iterations=args.iterations, metric=args.metric,
                              bending_weight=args.bending, learning_rate=args.learning_rate,
                              mask=mask, max_dim=args.max_dim,
                              max_output_voxels=args.max_output_voxels)
            mm_tag = "-".join(f"{v:g}" for v in args.spacing_mm)
            setting = (f"{args.metric}_mm{mm_tag}_l{args.levels}_i{args.iterations}"
                       f"_maxdim{args.max_dim}"
                       f"_b{args.bending:g}_lr{args.learning_rate:g}")
        elapsed = perf_counter() - start
        field, warped = result.field, result.warped
        config = {"spacing_voxels": args.spacing if pair.fixed.ndim == 2 else None,
                  "spacing_mm": spacing_mm if pair.fixed.ndim == 3 else None,
                  "levels": args.levels, "iterations_per_level": args.iterations,
                  "metric": args.metric, "bending_weight": args.bending,
                  "learning_rate": args.learning_rate}
        if pair.fixed.ndim == 3:
            config.update(max_dim=args.max_dim, working_shape=result.working_shape)
    dataset_output = args.output / args.dataset
    for branch in ("cnn", "b-spline"):
        (dataset_output / branch).mkdir(parents=True, exist_ok=True)
    output = dataset_output / args.method / f"{args.split}_seed{args.seed}" / pair_id / setting
    metrics = save_result(pair, field, warped, method=args.method,
                          split=args.split, seed=args.seed, config=config,
                          seconds=elapsed, output=output)
    print(json.dumps(metrics, indent=2, allow_nan=False))
    print(f"Saved results to {output.resolve()}")


if __name__ == "__main__":
    main()
