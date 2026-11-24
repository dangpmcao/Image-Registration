"""MNIST images paired with reproducible, known synthetic translations."""

import gzip
import struct
from functools import lru_cache
from pathlib import Path

import numpy as np

from data.pairs import Pair, PairSpec
from data.geometry import warp


_FILES = {"train": "train-images-idx3-ubyte.gz", "test": "t10k-images-idx3-ubyte.gz"}


@lru_cache(maxsize=2)
def _images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as stream:
        magic, count, height, width = struct.unpack(">IIII", stream.read(16))
        if magic != 2051:
            raise ValueError(f"Not an MNIST IDX image file: {path}")
        values = np.frombuffer(stream.read(), dtype=np.uint8)
    if values.size != count * height * width:
        raise ValueError(f"MNIST image count does not match header: {path}")
    return values.reshape(count, height, width).astype(np.float32) / 255


def list_pairs(root: Path, split: str = "val", seed: int = 42) -> list[PairSpec]:
    if split not in ("train", "val", "test"):
        raise ValueError("split must be train, val, or test")
    path = root / _FILES["test" if split == "test" else "train"]
    if not path.is_file():
        raise FileNotFoundError(path)
    count = len(_images(path))
    if split == "train":
        indices = range(0, int(count * 0.9))
    elif split == "val":
        indices = range(int(count * 0.9), count)
    else:
        indices = range(count)
    rng = np.random.default_rng(seed)
    shifts = rng.integers(-3, 4, size=(count, 2))
    return [PairSpec("mnist", f"{split}_{i}", f"{split}:{i}",
                     f"shift:{int(shifts[i, 0])}:{int(shifts[i, 1])}", path, path)
            for i in indices]


def load_pair(spec: PairSpec) -> Pair:
    if spec.dataset != "mnist":
        raise ValueError("Expected an MNIST pair specification")
    index = int(spec.fixed_id.split(":")[-1])
    _, dy_text, dx_text = spec.moving_id.split(":")
    dy, dx = int(dy_text), int(dx_text)
    fixed = _images(spec.fixed_path)[index]
    field = np.empty((*fixed.shape, 2), dtype=np.float32)
    field[...] = (dy, dx)
    moving = warp(fixed, -field).astype(np.float32)
    return Pair(fixed=fixed, moving=moving, fixed_seg=None, moving_seg=None,
                spacing=(1.0, 1.0), fixed_id=spec.fixed_id,
                moving_id=spec.moving_id, dataset="mnist", ground_truth_field=field)
