"""Sunnybrook cine MRI and TCIA breast DCE-MRI pair adapters."""

from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import re

import numpy as np
import pydicom

from data.pairs import Pair, PairSpec, split_subjects


def _normalise(fixed: np.ndarray, moving: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Use one intensity scale for both images to retain contrast differences."""
    low = min(float(np.percentile(fixed, 1)), float(np.percentile(moving, 1)))
    high = max(float(np.percentile(fixed, 99)), float(np.percentile(moving, 99)))
    scale = max(high - low, 1e-6)
    return (np.clip((fixed - low) / scale, 0, 1).astype(np.float32),
            np.clip((moving - low) / scale, 0, 1).astype(np.float32))


def _read_image(path: Path):
    ds = pydicom.dcmread(str(path))
    pixels = ds.pixel_array.astype(np.float32)
    pixels = pixels * float(ds.get("RescaleSlope", 1)) + float(ds.get("RescaleIntercept", 0))
    return pixels, ds


def _phase_contours(path: Path) -> tuple[int, int, dict[tuple[int, int], Path]]:
    """ED is the annotated phase with the larger median LV endocardial area."""
    areas: dict[int, list[float]] = defaultdict(list)
    contours: dict[tuple[int, int], Path] = {}
    for file in path.glob("IM-*-icontour-manual.txt"):
        match = re.search(r"-(\d+)-icontour-manual\.txt$", file.name)
        if not match:
            continue
        number = int(match.group(1))
        phase, slice_index = (number - 1) % 20, (number - 1) // 20
        xy = np.loadtxt(file, dtype=np.float64)
        if xy.ndim != 2 or xy.shape[0] < 3 or xy.shape[1] != 2:
            raise ValueError(f"Invalid contour: {file}")
        area = 0.5 * abs(np.dot(xy[:, 0], np.roll(xy[:, 1], -1))
                         - np.dot(xy[:, 1], np.roll(xy[:, 0], -1)))
        areas[phase].append(float(area))
        contours[(slice_index, phase)] = file
    if len(areas) != 2:
        raise ValueError(f"Expected two annotated cardiac phases in {path}, found {sorted(areas)}")
    phases = sorted(areas, key=lambda key: np.median(areas[key]))
    if np.median(areas[phases[1]]) <= 1.02 * np.median(areas[phases[0]]):
        raise ValueError(f"Cannot distinguish ED and ES from contours in {path}")
    return phases[1], phases[0], contours


def _sunnybrook_cases(root: Path, split: str):
    part_for_split = {"train": 3, "val": 2, "test": 1}
    if split not in (*part_for_split, "all"):
        raise ValueError("split must be train, val, test, or all")
    for dicom_dir in sorted(root.glob("dicom/*/*/SC-*/DICOM")):
        part_match = re.search(r"Part(\d+)$", dicom_dir.parents[2].name)
        if not part_match:
            continue
        part = int(part_match.group(1))
        if split != "all" and part != part_for_split[split]:
            continue
        contour_dirs = list(root.glob(
            f"contours/*Part{part}/*/{dicom_dir.parent.name}/contours-manual/IRCCI-expert"))
        if len(contour_dirs) != 1:
            raise FileNotFoundError(f"Expected contours for {dicom_dir.parent.name}")
        yield dicom_dir, contour_dirs[0]


def list_sunnybrook_paper_pairs(root: Path, split: str, fold: int,
                                seed: int = 42) -> list[PairSpec]:
    """de Vos cardiac protocol: 30 training scans, 15 held-out scans per fold.

    ``train`` has every ordered pair of distinct cine phases within each
    training scan (30 × 20 × 19 = 11,400). ``test`` has both ED/ES directions
    on the held-out scans. ``val`` monitors ED/ES on training scans; it shares
    subjects with train and must not be reported as an independent test set.
    """
    if split not in ("train", "val", "test") or fold not in (1, 2, 3):
        raise ValueError("Sunnybrook paper split must be train/val/test and fold 1..3")
    cases = []
    for dicom_dir, contour_dir in _sunnybrook_cases(Path(root), "all"):
        match = re.search(r"Part([123])$", dicom_dir.parents[2].name)
        if match is None:
            raise ValueError(f"Cannot determine Sunnybrook challenge part: {dicom_dir}")
        cases.append((int(match.group(1)), dicom_dir, contour_dir))
    if any(sum(part == fold_id for part, _, _ in cases) != 15 for fold_id in (1, 2, 3)):
        raise ValueError("Expected exactly 15 Sunnybrook scans in each released part")
    selected = [(dicom_dir, contour_dir) for part, dicom_dir, contour_dir in cases
                if (part == fold) == (split == "test")]
    pairs = []
    for dicom_dir, contour_dir in selected:
        case = dicom_dir.parent.name
        if split == "train":
            phases = ((fixed, moving) for fixed in range(20)
                      for moving in range(20) if fixed != moving)
            label_dir = None
        else:
            ed, es, _ = _phase_contours(contour_dir)
            phases = ((ed, es), (es, ed))
            label_dir = contour_dir
        for fixed, moving in phases:
            pairs.append(PairSpec("sunnybrook",
                                  f"fold{fold}_{case}_p{fixed:02}_p{moving:02}",
                                  f"{case}:p{fixed:02}", f"{case}:p{moving:02}",
                                  dicom_dir, dicom_dir, label_dir, label_dir))
    if split == "train":
        order = np.random.default_rng(seed).permutation(len(pairs))
        pairs = [pairs[index] for index in order]
    return pairs


def list_pairs(dataset: str, root: Path, split: str = "val", seed: int = 42) -> list[PairSpec]:
    """List stable DICOM pairs; Sunnybrook uses its released challenge split."""
    root = Path(root)
    if dataset in ("sunnybrook", "sunnybrook_2d"):
        pairs = []
        for dicom_dir, contour_dir in _sunnybrook_cases(root, split):
            case = dicom_dir.parent.name
            ed, es, contours = _phase_contours(contour_dir)
            if dataset == "sunnybrook":
                pairs.append(PairSpec(dataset, case, f"{case}:ED{ed}", f"{case}:ES{es}",
                                      dicom_dir, dicom_dir))
            else:
                slices = sorted({index for index, phase in contours if phase == ed}
                                & {index for index, phase in contours if phase == es})
                for index in slices:
                    ed_number, es_number = 20 * index + ed + 1, 20 * index + es + 1
                    ed_file = dicom_dir / f"IM-0001-{ed_number:04}.dcm"
                    es_file = dicom_dir / f"IM-0001-{es_number:04}.dcm"
                    if not ed_file.is_file() or not es_file.is_file():
                        raise FileNotFoundError(f"Missing cine frame for {case} slice {index}")
                    pairs.append(PairSpec(dataset, f"{case}_z{index:02}",
                                          f"{case}:z{index}:ED", f"{case}:z{index}:ES",
                                          ed_file, es_file,
                                          contours[(index, ed)], contours[(index, es)]))
        if not pairs:
            raise FileNotFoundError(f"No {dataset} pairs for {split} in {root}")
        return pairs
    if dataset != "tcia_breast_nact":
        raise ValueError(f"Unknown DICOM dataset: {dataset}")
    if split not in ("train", "val", "test", "all"):
        raise ValueError("split must be train, val, test, or all")
    subjects = sorted(p for p in root.glob("UCSF-BR-*") if p.is_dir())
    selected = subjects if split == "all" else [root / sid for sid in split_subjects(
        [p.name for p in subjects], seed=seed)[split]]
    pairs = []
    for patient in selected:
        series = list(patient.glob("MR_*"))
        if len(series) != 1 or len(list(series[0].glob("*.dcm"))) < 2:
            raise ValueError(f"Expected one multi-phase DCE series for {patient.name}")
        pairs.append(PairSpec(dataset, patient.name, f"{patient.name}:t0",
                              f"{patient.name}:t1", series[0], series[0]))
    if not pairs:
        raise FileNotFoundError(f"No {dataset} pairs for {split} in {root}")
    return pairs


def _slice_geometry(ds) -> tuple[float, float]:
    spacing = tuple(float(v) for v in ds.PixelSpacing)
    if len(spacing) != 2 or any(not np.isfinite(v) or v <= 0 for v in spacing):
        raise ValueError("Invalid DICOM pixel spacing")
    return spacing


def _stack(frames: list[tuple[np.ndarray, object]]):
    """Sort by physical slice position; return volume and full grid geometry."""
    first = frames[0][1]
    orientation = np.asarray(first.ImageOrientationPatient, dtype=float)
    normal = np.cross(orientation[:3], orientation[3:])
    normal /= np.linalg.norm(normal)
    spacing_2d = _slice_geometry(first)
    with_positions = []
    for pixels, ds in frames:
        if not np.allclose(ds.ImageOrientationPatient, orientation, atol=1e-4):
            raise ValueError("DICOM slice orientations differ")
        if not np.allclose(_slice_geometry(ds), spacing_2d, atol=1e-4):
            raise ValueError("DICOM pixel spacings differ")
        position = np.asarray(ds.ImagePositionPatient, dtype=float)
        with_positions.append((float(position @ normal), position, pixels))
    with_positions.sort(key=lambda item: item[0])
    positions = np.stack([item[1] for item in with_positions])
    distances = np.diff([item[0] for item in with_positions])
    if len(distances) == 0 or np.min(distances) <= 0 or not np.allclose(distances, np.median(distances), atol=0.15):
        raise ValueError("DICOM slices do not have a regular physical grid")
    volume = np.stack([item[2] for item in with_positions])
    return volume, (float(np.median(distances)), *spacing_2d), positions, orientation


@lru_cache(maxsize=64)
def load_sunnybrook_phase(dicom_dir: Path, phase_index: int):
    """Load one 3D cine timepoint; cache a bounded number for random pair training."""
    dicom_dir = Path(dicom_dir)
    if not 0 <= phase_index < 20:
        raise ValueError("Sunnybrook phase index must be 0..19")
    count = len(list(dicom_dir.glob("IM-*.dcm")))
    if count == 0 or count % 20:
        raise ValueError(f"Expected 20 phases per Sunnybrook slice: {dicom_dir}")
    frames = [_read_image(dicom_dir / f"IM-0001-{20 * index + phase_index + 1:04}.dcm")
              for index in range(count // 20)]
    volume, spacing, positions, orientation = _stack(frames)
    source_positions = np.stack([np.asarray(ds.ImagePositionPatient, dtype=float)
                                 for _, ds in frames])
    for array in (volume, positions, orientation, source_positions):
        array.setflags(write=False)
    return volume, spacing, positions, orientation, source_positions


def _contour_volume(contour_dir: Path, phase: int, shape: tuple[int, int, int],
                    sorted_positions: np.ndarray, source_positions: np.ndarray):
    labels = np.zeros(shape, dtype=np.uint8)
    covered = np.zeros(shape[0], dtype=bool)
    for source_index, position in enumerate(source_positions):
        number = 20 * source_index + phase + 1
        contour = contour_dir / f"IM-0001-{number:04}-icontour-manual.txt"
        if not contour.is_file():
            continue
        distances = np.linalg.norm(sorted_positions - position, axis=1)
        target_index = int(np.argmin(distances))
        if distances[target_index] > 0.1 or covered[target_index]:
            raise ValueError(f"Contour slice does not match DICOM volume: {contour}")
        labels[target_index] = _contour_mask(contour, shape[1:])
        covered[target_index] = True
    return labels, covered


def _sunnybrook_volume(spec: PairSpec) -> Pair:
    fixed_match = re.search(r":(?:ED|ES|p)(\d+)$", spec.fixed_id)
    moving_match = re.search(r":(?:ED|ES|p)(\d+)$", spec.moving_id)
    if not fixed_match or not moving_match:
        raise ValueError("Sunnybrook pair needs two cine phase indices")
    fixed_phase, moving_phase = int(fixed_match.group(1)), int(moving_match.group(1))
    fixed, spacing, fixed_pos, fixed_orientation, fixed_source_pos = load_sunnybrook_phase(
        spec.fixed_path, fixed_phase)
    moving, moving_spacing, moving_pos, moving_orientation, moving_source_pos = load_sunnybrook_phase(
        spec.moving_path, moving_phase)
    if (fixed.shape != moving.shape or not np.allclose(spacing, moving_spacing, atol=1e-4)
            or not np.allclose(fixed_pos, moving_pos, atol=0.1)
            or not np.allclose(fixed_orientation, moving_orientation, atol=1e-4)):
        raise ValueError("Sunnybrook ED and ES grids differ")
    fixed_seg = moving_seg = evaluation_mask = None
    if spec.fixed_label_path is not None and spec.moving_label_path is not None:
        fixed_seg, fixed_covered = _contour_volume(
            spec.fixed_label_path, fixed_phase, fixed.shape, fixed_pos, fixed_source_pos)
        moving_seg, moving_covered = _contour_volume(
            spec.moving_label_path, moving_phase, moving.shape, moving_pos, moving_source_pos)
        common_slices = fixed_covered & moving_covered
        if not common_slices.any():
            raise ValueError("Sunnybrook phases have no jointly annotated slices")
        evaluation_mask = np.broadcast_to(common_slices[:, None, None], fixed.shape).copy()
    fixed, moving = _normalise(fixed, moving)
    return Pair(fixed, moving, fixed_seg, moving_seg, spacing,
                spec.fixed_id, spec.moving_id, spec.dataset,
                evaluation_mask=evaluation_mask)


def _contour_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    from matplotlib.path import Path as MplPath

    contour = np.loadtxt(path, dtype=float)
    yy, xx = np.indices(shape)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    return MplPath(contour).contains_points(points).reshape(shape).astype(np.uint8)


def _sunnybrook_slice(spec: PairSpec) -> Pair:
    fixed, fixed_ds = _read_image(spec.fixed_path)
    moving, moving_ds = _read_image(spec.moving_path)
    if (fixed.shape != moving.shape
            or not np.allclose(fixed_ds.ImagePositionPatient, moving_ds.ImagePositionPatient, atol=0.1)
            or not np.allclose(fixed_ds.ImageOrientationPatient, moving_ds.ImageOrientationPatient, atol=1e-4)):
        raise ValueError("Sunnybrook annotated cine slices differ in geometry")
    spacing = _slice_geometry(fixed_ds)
    if not np.allclose(spacing, _slice_geometry(moving_ds), atol=1e-4):
        raise ValueError("Sunnybrook annotated cine pixel spacings differ")
    fixed_seg = _contour_mask(spec.fixed_label_path, fixed.shape)
    moving_seg = _contour_mask(spec.moving_label_path, moving.shape)
    fixed, moving = _normalise(fixed, moving)
    return Pair(fixed, moving, fixed_seg, moving_seg, spacing,
                spec.fixed_id, spec.moving_id, spec.dataset)


def _breast_volume(spec: PairSpec) -> Pair:
    series = list(spec.fixed_path.glob("*.dcm"))
    if len(series) < 2:
        raise ValueError(f"DCE series is empty: {spec.fixed_path}")
    headers = {}
    for file in series:
        ds = pydicom.dcmread(str(file), stop_before_pixels=True)
        number = int(ds.InstanceNumber)
        if number in headers:
            raise ValueError(f"Duplicate DCE instance number {number}")
        headers[number] = (file, ds)
    if set(headers) != set(range(1, len(series) + 1)):
        raise ValueError("DCE instance numbers must be contiguous")
    positions = {tuple(round(float(x), 3) for x in ds.ImagePositionPatient)
                 for _, ds in headers.values()}
    slices_per_phase = len(positions)
    if slices_per_phase < 2 or len(series) % slices_per_phase or len(series) < 2 * slices_per_phase:
        raise ValueError("DCE series cannot be divided into complete phases")
    phase_ranges = [range(k * slices_per_phase + 1, (k + 1) * slices_per_phase + 1)
                    for k in range(len(series) // slices_per_phase)]
    for numbers in phase_ranges:
        block_positions = {tuple(round(float(x), 3) for x in headers[i][1].ImagePositionPatient)
                           for i in numbers}
        if block_positions != positions:
            raise ValueError("DCE phase has missing or duplicate slice positions")
    t0 = [headers[i][1] for i in phase_ranges[0]]
    t1 = [headers[i][1] for i in phase_ranges[1]]
    time_key = ("AcquisitionTime" if all(ds.get("AcquisitionTime") for ds in t0 + t1)
                else "ContentTime")
    if not all(ds.get(time_key) for ds in t0 + t1):
        raise ValueError("DCE phase lacks AcquisitionTime and ContentTime")
    if max(str(ds.get(time_key)) for ds in t0) >= min(str(ds.get(time_key)) for ds in t1):
        raise ValueError("DCE acquisition times do not separate t0 and t1")
    fixed, spacing, fixed_pos, fixed_orientation = _stack(
        [_read_image(headers[i][0]) for i in phase_ranges[0]])
    moving, moving_spacing, moving_pos, moving_orientation = _stack(
        [_read_image(headers[i][0]) for i in phase_ranges[1]])
    if (fixed.shape != moving.shape or not np.allclose(spacing, moving_spacing, atol=1e-4)
            or not np.allclose(fixed_pos, moving_pos, atol=0.1)
            or not np.allclose(fixed_orientation, moving_orientation, atol=1e-4)):
        raise ValueError("Breast DCE phases are not on the same physical grid")
    fixed, moving = _normalise(fixed, moving)
    return Pair(fixed, moving, None, None, spacing, spec.fixed_id, spec.moving_id, spec.dataset)


def load_pair(spec: PairSpec) -> Pair:
    """Read one Sunnybrook or breast DCE pair on a common physical grid."""
    if spec.dataset == "sunnybrook":
        return _sunnybrook_volume(spec)
    if spec.dataset == "sunnybrook_2d":
        return _sunnybrook_slice(spec)
    if spec.dataset == "tcia_breast_nact":
        return _breast_volume(spec)
    raise ValueError(f"Unknown DICOM dataset: {spec.dataset}")
