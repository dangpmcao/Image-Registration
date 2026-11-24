"""One entry point for the datasets available in ../dataset."""

from pathlib import Path

from data.pairs import Pair, PairSpec, load_pair as load_oasis_pair, split_subjects, subject_ids


DATASET_FOLDERS = {
    "mnist": "mnist",
    "oasis_2d": "oasis_2d_neurite",
    "acdc": "acdc",
    "sunnybrook": "sunnybrook",
    "sunnybrook_2d": "sunnybrook",
    "lungct": "learn2reg_lungct",
    "nlst2023": "learn2reg_nlst2023",
    "oasis_3d": "learn2reg_oasis",
    "abdomenctct": "learn2reg_abdomenctct",
    "tcia_breast_nact": "tcia_breast_nact",
    "ixi": "ixi",
}

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / "dataset"


def dataset_root(dataset: str, root: str | Path | None = None) -> Path:
    if dataset not in DATASET_FOLDERS:
        raise ValueError(f"Unknown dataset {dataset!r}; choose from {', '.join(DATASET_FOLDERS)}")
    path = Path(root) if root is not None else DEFAULT_DATA_ROOT / DATASET_FOLDERS[dataset]
    if not path.is_dir():
        raise FileNotFoundError(f"Dataset is absent: {path}")
    return path


def list_pairs(dataset: str, split: str = "val", *, root: str | Path | None = None,
               seed: int = 42) -> list[PairSpec]:
    """Return deterministic pair specs for one dataset and split."""
    path = dataset_root(dataset, root)
    if dataset == "mnist":
        from data import mnist_data
        pairs = mnist_data.list_pairs(path, split, seed)
    elif dataset == "oasis_2d":
        if split not in ("train", "val", "test"):
            raise ValueError("split must be train, val, or test")
        ids = split_subjects(subject_ids(path), seed=seed)[split]
        matches = ([(ids[i], ids[(i + offset) % len(ids)])
                    for offset in (1, 2, 3, 4) for i in range(len(ids))]
                   if split == "train" else
                   [(ids[i], ids[i + 1]) for i in range(0, len(ids) - 1, 2)])
        pairs = [PairSpec(dataset, f"{fixed}__{moving}", fixed, moving,
                          path / fixed / "slice_norm.nii.gz",
                          path / moving / "slice_norm.nii.gz",
                          path / fixed / "slice_seg24.nii.gz",
                          path / moving / "slice_seg24.nii.gz")
                 for fixed, moving in matches]
    elif dataset in ("acdc", "lungct", "nlst2023", "oasis_3d", "abdomenctct", "ixi"):
        from data import nifti_data
        pairs = nifti_data.list_pairs(dataset, path, split, seed)
    else:
        from data import dicom_data
        pairs = dicom_data.list_pairs(dataset, path, split, seed)
    if not pairs:
        raise ValueError(f"No {split} pairs available for {dataset}")
    return pairs


def load_pair(spec: PairSpec) -> Pair:
    """Load a pair in its native 2D or 3D array shape."""
    if spec.dataset == "mnist":
        from data import mnist_data
        return mnist_data.load_pair(spec)
    if spec.dataset == "oasis_2d":
        return load_oasis_pair(spec.fixed_path.parent.parent, spec.fixed_id, spec.moving_id)
    if spec.dataset in ("acdc", "lungct", "nlst2023", "oasis_3d", "abdomenctct", "ixi"):
        from data import nifti_data
        return nifti_data.load_pair(spec)
    if spec.dataset in ("sunnybrook", "sunnybrook_2d", "tcia_breast_nact"):
        from data import dicom_data
        return dicom_data.load_pair(spec)
    raise ValueError(f"Unknown dataset {spec.dataset!r}")
