"""Explicit data protocols for DLIR; baseline preprocessing remains unchanged."""

import math

import numpy as np
import torch

from cnn.devos import sample_grid
from data.pairs import Pair, PairSpec
from data.registry import dataset_root, list_pairs, load_pair


def pair_specs(dataset: str, split: str, preset: str, root, seed: int,
               fold: int | None) -> list[PairSpec]:
    if preset.startswith("sunnybrook"):
        if dataset != "sunnybrook" or fold not in (1, 2, 3):
            raise ValueError("Sunnybrook presets require --dataset sunnybrook and --fold 1/2/3")
        from data.dicom_data import list_sunnybrook_paper_pairs
        return list_sunnybrook_paper_pairs(dataset_root(dataset, root), split, fold, seed)
    return list_pairs(dataset, split, root=root, seed=seed)


def load_dlir_pair(spec: PairSpec, preset: str) -> Pair:
    if preset.startswith("sunnybrook"):
        from data.dicom_data import _sunnybrook_volume
        return _sunnybrook_volume(spec, paper_normalization=True)
    if preset == "chest-ct":
        if spec.dataset not in ("lungct", "nlst2023"):
            raise ValueError("chest-ct expects lungct or nlst2023 (adaptations of the original cohort)")
        from data.nifti_data import load_pair as load_nifti
        return load_nifti(spec, intensity_range=(-1000.0, 3095.0))
    return load_pair(spec)


def tensors(pair: Pair, preset: str, device: torch.device):
    if pair.fixed.ndim != 3:
        raise ValueError("DLIR paper methods require 3D volumes; use the baseline for 2D")
    fixed, moving = (torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32))[None, None].to(device)
                     for image in (pair.fixed, pair.moving))
    spacing = tuple(pair.spacing)
    if preset.startswith("sunnybrook"):
        target = (8.0, 1.25, 1.25)  # Table 1, converted from x,y,z to z,y,x.
        shape = tuple(max(3, math.ceil((n - 1) * old / new) + 1)
                      for n, old, new in zip(pair.fixed.shape, spacing, target))
        fixed = sample_grid(fixed, shape, spacing, target)
        moving = sample_grid(moving, shape, spacing, target)
        spacing = target
    return fixed, moving, spacing


def augment(fixed, moving, spacing, rng, preset: str,
            crop_shape: tuple[int, int, int] | None = None):
    """Corresponding augmentations; labels are never used in the training loss."""
    if preset.startswith("sunnybrook"):
        k = int(rng.integers(4))
        fixed, moving = (torch.rot90(image, k, (-2, -1)) for image in (fixed, moving))
        if k % 2:
            spacing = (spacing[0], spacing[2], spacing[1])
        maximum = (0, 16, 16)
    elif preset == "chest-ct":
        maximum = tuple(round(32 / step) for step in spacing)
    else:
        maximum = (0, 0, 0)
    slices = []
    for axis, (size, limit) in enumerate(zip(fixed.shape[-3:], maximum)):
        if crop_shape is not None:
            length = min(crop_shape[axis], size)
            start = int(rng.integers(size - length + 1))
            slices.append(slice(start, start + length))
        else:
            # Keep at least six voxels for a two-level cardiac pyramid.
            limit = min(limit, max(0, (size - 6) // 2))
            left, right = rng.integers(limit + 1, size=2)
            slices.append(slice(int(left), size - int(right)))
    index = (slice(None), slice(None), *slices)
    return fixed[index], moving[index], spacing
