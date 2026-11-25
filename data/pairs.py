"""Pair records and the OASIS 2D dataset loader."""

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence

import numpy as np


DEFAULT_OASIS_ROOT = Path(__file__).resolve().parents[2] / "dataset" / "oasis_2d_neurite"
_IMAGE = "slice_norm.nii.gz"
_LABEL = "slice_seg24.nii.gz"


@dataclass(frozen=True)
class Pair:
    """Images and optional labels on one fixed grid, with spacing in array order."""

    fixed: np.ndarray
    moving: np.ndarray
    fixed_seg: np.ndarray | None
    moving_seg: np.ndarray | None
    spacing: tuple[float, ...]
    fixed_id: str
    moving_id: str
    dataset: str = "oasis_2d"
    landmarks_fixed: np.ndarray | None = None
    landmarks_moving: np.ndarray | None = None
    ground_truth_field: np.ndarray | None = None
    fixed_mask: np.ndarray | None = None
    moving_mask: np.ndarray | None = None
    evaluation_mask: np.ndarray | None = None


@dataclass(frozen=True)
class PairSpec:
    """A reproducible fixed/moving pairing; paths are resolved by its loader."""

    dataset: str
    pair_id: str
    fixed_id: str
    moving_id: str
    fixed_path: Path
    moving_path: Path
    fixed_label_path: Path | None = None
    moving_label_path: Path | None = None


def _root(path: str | Path | None) -> Path:
    root = Path(path) if path is not None else DEFAULT_OASIS_ROOT
    if not root.is_dir():
        raise FileNotFoundError(f"OASIS 2D directory does not exist: {root}")
    return root


def subject_ids(root: str | Path | None = None) -> list[str]:
    """Return complete OASIS 2D subject directories in a stable order."""

    data_root = _root(root)
    subjects = sorted(p for p in data_root.glob("OASIS_OAS1_*_MR1") if p.is_dir())
    incomplete = [p.name for p in subjects if not (p / _IMAGE).is_file() or not (p / _LABEL).is_file()]
    if incomplete:
        raise FileNotFoundError(f"Missing OASIS slice or label for: {', '.join(incomplete[:5])}")
    if not subjects:
        raise FileNotFoundError(f"No OASIS 2D subjects found in {data_root}")
    return [p.name for p in subjects]


def split_subjects(
    ids: Sequence[str],
    *,
    seed: int = 42,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
) -> dict[str, list[str]]:
    """Make a reproducible split before forming image pairs, avoiding subject leakage."""

    if not (0 < train_fraction < 1 and 0 < val_fraction < 1 and train_fraction + val_fraction < 1):
        raise ValueError("train_fraction and val_fraction must be positive and sum to less than 1")
    subjects = sorted(ids)
    if len(subjects) != len(set(subjects)):
        raise ValueError("Subject IDs must be unique")
    order = np.random.default_rng(seed).permutation(len(subjects))
    shuffled = [subjects[i] for i in order]
    n_train = round(len(shuffled) * train_fraction)
    n_val = round(len(shuffled) * val_fraction)
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def _load_subject(root: Path, subject_id: str):
    if Path(subject_id).name != subject_id or subject_id in (".", ".."):
        raise ValueError(f"Expected a subject ID, got {subject_id!r}")
    subject = root / subject_id
    if not subject.is_dir():
        raise FileNotFoundError(f"OASIS subject does not exist: {subject}")

    import nibabel as nib

    image_nii = nib.load(str(subject / _IMAGE))
    label_nii = nib.load(str(subject / _LABEL))
    if not np.allclose(image_nii.affine, label_nii.affine, atol=1e-4):
        raise ValueError(f"Image and label geometry disagree for {subject_id}")

    def to_rows_cols(nii, dtype):
        array = np.asanyarray(nii.dataobj)
        if array.ndim == 3 and array.shape[2] == 1:
            array = array[:, :, 0]
        if array.ndim != 2:
            raise ValueError(f"Expected a 2D slice for {subject_id}, got {array.shape}")
        # NIfTI stores these slices as (x, y, 1); downstream arrays use (row=y, col=x).
        return np.ascontiguousarray(array.T, dtype=dtype)

    image = to_rows_cols(image_nii, np.float32)
    label = to_rows_cols(label_nii, np.int16)
    if image.shape != label.shape:
        raise ValueError(f"Image and label shapes disagree for {subject_id}")
    spacing = tuple(float(v) for v in image_nii.header.get_zooms()[:2][::-1])
    if len(spacing) != 2 or any(not np.isfinite(v) or v <= 0 for v in spacing):
        raise ValueError(f"Invalid slice spacing for {subject_id}: {spacing}")
    return image, label, spacing, image_nii.affine


def load_pair(
    root: str | Path | None, fixed_id: str, moving_id: str
) -> Pair:
    """Load two aligned OASIS slices as fixed and moving images.

    Images and labels have shape (height, width). Spacing is (row_mm, col_mm).
    A downstream pull warp should use displacement channels (drow, dcol).
    """

    data_root = _root(root)
    fixed, fixed_label, spacing, fixed_affine = _load_subject(data_root, fixed_id)
    moving, moving_label, moving_spacing, moving_affine = _load_subject(data_root, moving_id)
    if fixed.shape != moving.shape or not np.allclose(spacing, moving_spacing):
        raise ValueError("Fixed and moving slice grids disagree")
    if not np.allclose(fixed_affine, moving_affine, atol=1e-4):
        raise ValueError("Fixed and moving slices are not in the same aligned space")
    return Pair(fixed, moving, fixed_label, moving_label, spacing, fixed_id, moving_id)
