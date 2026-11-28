"""Registration metrics in physical units on a common fixed grid."""

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, map_coordinates


def normalized_mutual_information(a, b, mask, bins: int = 32) -> float:
    """Histogram NMI, (H(a)+H(b))/H(a,b); not the optimizer's Parzen estimate."""
    a, b = np.asarray(a)[mask], np.asarray(b)[mask]
    if not a.size or bins < 2:
        raise ValueError("NMI requires a nonempty mask and at least two bins")
    joint = np.histogram2d(a, b, bins=bins)[0]
    joint /= joint.sum()

    def entropy(values):
        positive = values[values > 0]
        return float(-(positive * np.log(positive)).sum())

    denominator = entropy(joint)
    return (entropy(joint.sum(0)) + entropy(joint.sum(1))) / denominator if denominator > 0 else 0.0


def landmark_errors(fixed, moving, field, spacing) -> dict:
    """Pull-map TRE: compare fixed + u(fixed) with paired moving landmarks.

    Coordinates are continuous voxels in array order on the common grid.
    A field maps fixed to moving; sampling it at moving landmarks is incorrect.
    """
    fixed, moving = np.asarray(fixed, float), np.asarray(moving, float)
    dimension = field.shape[-1]
    if (fixed.ndim != 2 or fixed.shape != moving.shape or fixed.shape[1] != dimension
            or not len(fixed) or not np.isfinite(fixed).all() or not np.isfinite(moving).all()):
        raise ValueError("landmarks must be matching nonempty finite [N,dimension] arrays")
    if np.any(fixed < 0) or np.any(fixed > np.asarray(field.shape[:-1]) - 1):
        raise ValueError("fixed landmarks lie outside the field")
    displacement = np.stack([map_coordinates(field[..., axis], fixed.T, order=1,
                                             mode="nearest", prefilter=False)
                             for axis in range(dimension)], -1)
    scale = np.asarray(spacing)
    before = np.linalg.norm((fixed - moving) * scale, axis=1)
    after = np.linalg.norm((fixed + displacement - moving) * scale, axis=1)
    return {"count": len(fixed), "before_mean_mm": float(before.mean()),
            "after_mean_mm": float(after.mean()), "after_std_mm": float(after.std()),
            "after_per_landmark_mm": after.tolist()}


def surface_distances(fixed_labels, moving_labels, spacing) -> dict | None:
    """Per-label ASD and Hausdorff in mm; requires fully observed labels."""
    if fixed_labels is None or moving_labels is None:
        return None
    result = {}
    for label in np.union1d(np.unique(fixed_labels), np.unique(moving_labels)):
        if label == 0:
            continue
        a, b = fixed_labels == label, moving_labels == label
        if not a.any() or not b.any():
            result[str(int(label))] = {"asd_mm": None, "hausdorff_mm": None}
            continue
        surface_a, surface_b = a & ~binary_erosion(a), b & ~binary_erosion(b)
        distances = np.concatenate((distance_transform_edt(~surface_b, sampling=spacing)[surface_a],
                                    distance_transform_edt(~surface_a, sampling=spacing)[surface_b]))
        result[str(int(label))] = {"asd_mm": float(distances.mean()),
                                  "hausdorff_mm": float(distances.max())}
    return result
