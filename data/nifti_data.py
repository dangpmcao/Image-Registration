"""Native 3D NIfTI pairs from ACDC, Learn2Reg, and IXI.

Arrays are (z, y, x). MRI uses each image's nonzero 1st/99th percentiles;
chest CT clips HU to [-1000, 400], abdomen CT to [-200, 300].
"""

import json
from pathlib import Path

import nibabel as nib
import numpy as np

from data.pairs import Pair, PairSpec, split_subjects


_L2R = {
    "lungct": "LungCT",
    "nlst2023": "NLST",
    "oasis_3d": "OASIS",
    "abdomenctct": "AbdomenCTCT",
}


def _base(root: Path, inner: str | None = None) -> Path:
    root = Path(root)
    path = root / inner if inner and (root / inner).is_dir() else root
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def _local(base: Path, path: str) -> Path:
    target = (base / path).resolve()
    if not target.is_relative_to(base.resolve()):
        raise ValueError(f"Manifest path escapes dataset root: {path}")
    return target


def _id(path: Path) -> str:
    return path.name.removesuffix(".nii.gz")


def _spec(dataset: str, fixed: Path, moving: Path, *, prefix: str = "") -> PairSpec:
    def label_for(image: Path) -> Path | None:
        if dataset not in ("oasis_3d", "abdomenctct"):
            return None
        folder = image.parent.name.replace("images", "labels")
        candidate = image.parent.parent / folder / image.name
        return candidate if candidate.is_file() else None

    fixed_id, moving_id = _id(fixed), _id(moving)
    return PairSpec(dataset, f"{prefix}{fixed_id}__{moving_id}", fixed_id, moving_id,
                    fixed, moving, label_for(fixed), label_for(moving))


def _records(base: Path, rows: list[dict]) -> list[tuple[Path, Path]]:
    pairs = [(_local(base, item["fixed"]), _local(base, item["moving"])) for item in rows]
    return [(fixed, moving) for fixed, moving in pairs if fixed.is_file() and moving.is_file()]


def _learn2reg(dataset: str, root: Path, split: str, seed: int) -> list[PairSpec]:
    name = _L2R[dataset]
    base = _base(root, name)
    manifest = json.loads((base / f"{name}_dataset.json").read_text(encoding="utf-8"))
    validation = _records(base, manifest["registration_val"])
    official_test = _records(base, manifest["registration_test"])
    if dataset in ("lungct", "nlst2023"):
        training = _records(base, manifest["training_paired_images"])
        val_paths = {path for pair in validation for path in pair}
        training = [pair for pair in training if not any(path in val_paths for path in pair)]
        if dataset == "nlst2023" and not official_test:
            # The public local NLST package has no imagesTs. Reserve subject pairs
            # from the remaining imagesTr instead of pretending official test exists.
            order = np.random.default_rng(seed).permutation(len(training))
            n_test = max(1, round(0.15 * len(training)))
            held_out = set(order[:n_test])
            internal_test = [pair for i, pair in enumerate(training) if i in held_out]
            training = [pair for i, pair in enumerate(training) if i not in held_out]
        else:
            internal_test = []
    else:
        val_paths = {path for pair in validation for path in pair}
        images = [_local(base, row["image"]) for row in manifest["training"]]
        images = [path for path in images if path.is_file() and path not in val_paths]
        order = np.random.default_rng(seed).permutation(len(images))
        images = [images[i] for i in order]
        training = list(zip(images[::2], images[1::2]))
        internal_test = []

    selected = {"train": training, "val": validation,
                "test": official_test or internal_test}[split]
    prefix = "internal_test_" if split == "test" and not official_test else ""
    return [_spec(dataset, fixed, moving, prefix=prefix) for fixed, moving in selected]


def _acdc(root: Path, split: str, seed: int) -> list[PairSpec]:
    base = _base(root)
    images = sorted((base / "Images").glob("patient*_frame*.nii.gz"))
    by_patient: dict[str, list[Path]] = {}
    for image in images:
        by_patient.setdefault(image.name.split("_")[0], []).append(image)
    selected = split_subjects(list(by_patient), seed=seed)[split]
    pairs = []
    for patient in selected:
        frames = sorted(by_patient[patient])
        if len(frames) != 2:
            raise ValueError(f"ACDC expected two ED/ES frames for {patient}")
        labels = [base / "Masks" / p.name.replace(".nii.gz", "_gt.nii.gz") for p in frames]
        if not all(p.is_file() for p in labels):
            raise FileNotFoundError(f"ACDC mask missing for {patient}")
        # ED has the larger left-ventricular cavity (ACDC label 3).
        cavity_sizes = [int(np.count_nonzero(np.asanyarray(nib.load(str(p)).dataobj) == 3))
                        for p in labels]
        if cavity_sizes[0] == cavity_sizes[1]:
            raise ValueError(f"Cannot distinguish ACDC ED/ES for {patient}")
        ed = int(np.argmax(cavity_sizes))
        es = 1 - ed
        pairs.append(PairSpec("acdc", patient, _id(frames[ed]), _id(frames[es]),
                              frames[ed], frames[es], labels[ed], labels[es]))
    return pairs


def _ixi(root: Path, split: str, seed: int) -> list[PairSpec]:
    base = _base(root)
    t1 = {p.name.removesuffix("-T1.nii.gz"): p for p in (base / "T1").glob("*-T1.nii.gz")}
    t2 = {p.name.removesuffix("-T2.nii.gz"): p for p in (base / "T2").glob("*-T2.nii.gz")}
    keys = split_subjects(sorted(t1.keys() & t2.keys()), seed=seed)[split]
    return [PairSpec("ixi", key, f"{key}-T1", f"{key}-T2", t1[key], t2[key]) for key in keys]


def list_pairs(dataset: str, root: Path, split: str = "val", seed: int = 42) -> list[PairSpec]:
    """List subject-disjoint pairs, using Learn2Reg manifests when available."""
    if split not in ("train", "val", "test"):
        raise ValueError("split must be train, val, or test")
    if dataset in _L2R:
        return _learn2reg(dataset, root, split, seed)
    if dataset == "acdc":
        return _acdc(root, split, seed)
    if dataset == "ixi":
        return _ixi(root, split, seed)
    raise ValueError(f"Unsupported NIfTI dataset: {dataset}")


def _on_grid(image: nib.spatialimages.SpatialImage, fixed: nib.spatialimages.SpatialImage,
             order: int) -> np.ndarray:
    if image.shape != fixed.shape or not np.allclose(image.affine, fixed.affine, atol=1e-4):
        from nibabel.processing import resample_from_to

        image = resample_from_to(image, (fixed.shape, fixed.affine), order=order)
    return image.get_fdata(dtype=np.float32)


def _normalize(image: np.ndarray, dataset: str) -> np.ndarray:
    if dataset in ("lungct", "nlst2023"):
        lo, hi = -1000.0, 400.0
    elif dataset == "abdomenctct":
        lo, hi = -200.0, 300.0
    elif dataset == "oasis_3d":
        return np.clip(image, 0, 1).astype(np.float32, copy=False)
    else:
        foreground = image[image > 0]
        if not foreground.size:
            return np.zeros_like(image, dtype=np.float32)
        lo, hi = np.percentile(foreground, (1, 99))
        if hi <= lo:
            return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - lo) / (hi - lo), 0, 1).astype(np.float32, copy=False)


def _mask_for(path: Path, fixed: nib.spatialimages.SpatialImage) -> np.ndarray | None:
    folder = path.parent.name.replace("images", "masks")
    candidate = path.parent.parent / folder / path.name
    if folder == path.parent.name or not candidate.is_file():
        return None
    return (_on_grid(nib.load(str(candidate)), fixed, order=0) > 0).transpose(2, 1, 0).copy()


def load_pair(spec: PairSpec, *, intensity_range: tuple[float, float] | None = None) -> Pair:
    """Load and align moving/labels to the fixed grid; return (z,y,x) arrays."""
    if spec.dataset not in (*_L2R, "acdc", "ixi"):
        raise ValueError(f"Unsupported NIfTI dataset: {spec.dataset}")
    fixed_nii = nib.load(str(spec.fixed_path))
    moving_nii = nib.load(str(spec.moving_path))
    if fixed_nii.ndim != 3 or moving_nii.ndim != 3:
        raise ValueError("Expected 3D NIfTI images")
    def normalize(image):
        if intensity_range is None:
            return _normalize(image, spec.dataset)
        low, high = intensity_range
        if not np.isfinite([low, high]).all() or high <= low:
            raise ValueError("intensity_range must be finite and increasing")
        return np.clip((image - low) / (high - low), 0, 1).astype(np.float32)

    fixed = normalize(_on_grid(fixed_nii, fixed_nii, 1).transpose(2, 1, 0).copy())
    moving = normalize(_on_grid(moving_nii, fixed_nii, 1).transpose(2, 1, 0).copy())

    def label(path: Path | None) -> np.ndarray | None:
        if path is None:
            return None
        values = _on_grid(nib.load(str(path)), fixed_nii, order=0)
        return np.rint(values).astype(np.int16).transpose(2, 1, 0).copy()

    # ACDC's redistributed sform has unit spacing despite pixdim storing the
    # original 1.56 x 1.56 x 10 mm; use pixdim for this dataset's voxel spacing.
    spacing = (tuple(float(v) for v in fixed_nii.header.get_zooms()[:3][::-1])
               if spec.dataset == "acdc"
               else tuple(float(np.linalg.norm(fixed_nii.affine[:3, axis])) for axis in (2, 1, 0)))
    if any(not np.isfinite(v) or v <= 0 for v in spacing):
        raise ValueError("Invalid NIfTI spacing")
    return Pair(fixed, moving, label(spec.fixed_label_path), label(spec.moving_label_path),
                spacing, spec.fixed_id, spec.moving_id, dataset=spec.dataset,
                fixed_mask=_mask_for(spec.fixed_path, fixed_nii),
                moving_mask=_mask_for(spec.moving_path, fixed_nii))
