"""Shared 3D pull warping and Jacobian on arrays ordered (z, y, x)."""

import numpy as np
from scipy.ndimage import map_coordinates


def warp3d(volume: np.ndarray, field: np.ndarray, *, order: int = 1,
           chunk_slices: int = 8) -> np.ndarray:
    """Sample ``volume[z+dz, y+dy, x+dx]`` with zero padding.

    ``field`` is [D,H,W,3] in voxel units. Order 0 preserves label dtype;
    order 1 returns float32. Slices are processed in chunks to limit memory.
    """
    source = np.asarray(volume, dtype=np.float32 if order == 1 else None)
    displacement = np.asarray(field)
    if (source.ndim != 3 or displacement.shape != (*source.shape, 3)
            or not np.isfinite(displacement).all() or chunk_slices < 1):
        raise ValueError("expected a finite [D,H,W,3] field on a 3D volume")
    if order not in (0, 1):
        raise ValueError("order must be 0 or 1")
    out = np.empty(source.shape, dtype=source.dtype if order == 0 else np.float32)
    depth, height, width = source.shape
    yy, xx = np.meshgrid(np.arange(height, dtype=np.float32),
                         np.arange(width, dtype=np.float32), indexing="ij")
    for start in range(0, depth, chunk_slices):
        stop = min(depth, start + chunk_slices)
        block = displacement[start:stop]
        qz = np.arange(start, stop, dtype=np.float32)[:, None, None] + block[..., 0]
        qy = yy[None] + block[..., 1]
        qx = xx[None] + block[..., 2]
        if order == 0:
            iz = np.floor(qz + 0.5).astype(np.intp)
            iy = np.floor(qy + 0.5).astype(np.intp)
            ix = np.floor(qx + 0.5).astype(np.intp)
            valid = ((iz >= 0) & (iz < depth) & (iy >= 0) & (iy < height)
                     & (ix >= 0) & (ix < width))
            out[start:stop] = np.where(valid, source[np.clip(iz, 0, depth - 1),
                                                     np.clip(iy, 0, height - 1),
                                                     np.clip(ix, 0, width - 1)], 0)
        else:
            out[start:stop] = map_coordinates(source, [qz, qy, qx], order=1,
                                               mode="grid-constant", cval=0,
                                               prefilter=False)
    return out


def jacobian_determinant3d(field: np.ndarray, *, chunk_slices: int = 8) -> np.ndarray:
    """Determinant of I + grad(field), with voxel coordinates in (z,y,x)."""
    field = np.asarray(field)
    if (field.ndim != 4 or field.shape[-1] != 3 or min(field.shape[:3]) < 3
            or not np.isfinite(field).all() or chunk_slices < 1):
        raise ValueError("field must be finite with shape [D,H,W,3], each axis >= 3")
    out = np.empty(field.shape[:3], dtype=np.float32)
    depth = field.shape[0]
    for start in range(0, depth, chunk_slices):
        stop = min(depth, start + chunk_slices)
        lo, hi = max(0, start - 1), min(depth, stop + 1)
        block = field[lo:hi]
        derivatives = [np.gradient(block[..., axis]) for axis in range(3)]
        select = slice(start - lo, stop - lo)
        a, b, c = (derivatives[0][0][select] + 1,
                   derivatives[0][1][select], derivatives[0][2][select])
        d, e, f = (derivatives[1][0][select],
                   derivatives[1][1][select] + 1, derivatives[1][2][select])
        g, h, i = (derivatives[2][0][select],
                   derivatives[2][1][select], derivatives[2][2][select] + 1)
        out[start:stop] = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    return out


def resize_volume(volume: np.ndarray, shape: tuple[int, int, int], order: int = 1) -> np.ndarray:
    """Resample a 3D array to a new grid using aligned corner coordinates."""
    axes = [np.linspace(0, size - 1, count, dtype=np.float32)
            for size, count in zip(volume.shape, shape)]
    coords = np.meshgrid(*axes, indexing="ij")
    return map_coordinates(volume, coords, order=order, mode="nearest",
                           prefilter=False).astype(np.float32 if order else volume.dtype)


def resize_field(field: np.ndarray, shape: tuple[int, int, int],
                 chunk_slices: int = 8) -> np.ndarray:
    """Resample a pull field and convert displacements to the new voxel units."""
    if field.shape[:3] == shape:
        return np.asarray(field, dtype=np.float32)
    old_shape = field.shape[:3]
    factors = [(old - 1) / (new - 1) for old, new in zip(old_shape, shape)]
    scales = [1 / factor for factor in factors]
    out = np.empty((*shape, 3), dtype=np.float32)
    yy, xx = np.meshgrid(np.arange(shape[1], dtype=np.float32) * factors[1],
                         np.arange(shape[2], dtype=np.float32) * factors[2],
                         indexing="ij")
    for start in range(0, shape[0], chunk_slices):
        stop = min(shape[0], start + chunk_slices)
        zz = np.arange(start, stop, dtype=np.float32)[:, None, None] * factors[0]
        coords = [np.broadcast_to(zz, (stop - start, *shape[1:])),
                  np.broadcast_to(yy, (stop - start, *shape[1:])),
                  np.broadcast_to(xx, (stop - start, *shape[1:]))]
        for axis in range(3):
            out[start:stop, ..., axis] = map_coordinates(field[..., axis], coords,
                                                         order=1, mode="nearest",
                                                         prefilter=False) * scales[axis]
    return out
