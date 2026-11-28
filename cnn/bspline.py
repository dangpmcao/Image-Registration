"""Fixed cubic B-spline transposed convolutions (de Vos, Section 2.2).

Control point j lies at (j - 1) * stride, matching data.geometry.bspline_basis.
The three separable convolutions avoid allocating a dense 3D spline kernel.
"""

from collections.abc import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F


def interpolate_control_grid(control: Tensor, shape: Sequence[int],
                             strides: Sequence[int]) -> Tensor:
    """[B,3,Gz,Gy,Gx] voxel displacements -> [B,D,H,W,3]."""
    if control.ndim != 5 or control.shape[1] != 3 or len(shape) != 3 or len(strides) != 3:
        raise ValueError("expected a 3D control grid with three displacement channels")
    expected = tuple((int(n) - 1) // int(s) + 4 for n, s in zip(shape, strides))
    if tuple(control.shape[2:]) != expected:
        raise ValueError(f"expected control grid {expected}, got {tuple(control.shape[2:])}")
    dense = control
    for axis in (2, 1, 0):
        step = int(strides[axis])
        if step < 1:
            raise ValueError("spline strides must be positive integers")
        distance = torch.arange(-2 * step + 1, 2 * step,
                                device=control.device, dtype=control.dtype).abs() / step
        kernel = torch.where(distance < 1,
                             (4 - 6 * distance.square() + 3 * distance.pow(3)) / 6,
                             (2 - distance).clamp_min(0).pow(3) / 6)
        kernel_shape = [1, 1, 1]
        kernel_shape[axis] = kernel.numel()
        weights = kernel.reshape(1, 1, *kernel_shape).expand(3, 1, *kernel_shape)
        stride, padding = [1, 1, 1], [0, 0, 0]
        stride[axis], padding[axis] = step, 3 * step - 1
        dense = F.conv_transpose3d(dense, weights, stride=tuple(stride),
                                   padding=tuple(padding), groups=3)
    return dense[:, :, :shape[0], :shape[1], :shape[2]].permute(0, 2, 3, 4, 1)
