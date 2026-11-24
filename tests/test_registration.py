"""Checks that guard coordinate and interpolation conventions."""

import unittest

import numpy as np

from common.data import split_subjects
from common.geometry import dice, jacobian_determinant, warp
from b_spline.ffd import bspline_field, register


class RegistrationTests(unittest.TestCase):
    def test_identity_warp_and_labels(self):
        image = np.arange(20 * 24, dtype=np.float64).reshape(20, 24)
        field = np.zeros((*image.shape, 2))
        np.testing.assert_allclose(warp(image, field), image)
        np.testing.assert_array_equal(warp((image % 3).astype(np.int16), field, order=0), image % 3)
        np.testing.assert_allclose(jacobian_determinant(field), 1)

    def test_constant_control_translation(self):
        control = np.empty((6, 6, 2), dtype=float)
        control[...] = (2.0, -3.0)
        field = bspline_field(control, (20, 20), (8, 8))
        np.testing.assert_allclose(field[..., 0], 2.0, atol=1e-12)
        np.testing.assert_allclose(field[..., 1], -3.0, atol=1e-12)
        np.testing.assert_allclose(jacobian_determinant(field), 1, atol=1e-12)

    def test_synthetic_pair_improves(self):
        yy, xx = np.indices((48, 48))
        def image_at(y, x):
            return (np.exp(-((y - 15) ** 2 + (x - 16) ** 2) / 50)
                    + 0.8 * np.exp(-((y - 31) ** 2 + (x - 29) ** 2) / 80)
                    + 0.4 * np.exp(-((y - 26) ** 2 + (x - 12) ** 2) / 30))
        fixed = image_at(yy, xx)
        moving = image_at(yy + 2, xx - 3)
        before = np.mean((fixed - moving) ** 2)
        result = register(fixed, moving, spacing=16, levels=1,
                          iterations=80, metric="ssd", bending_weight=1)
        after = np.mean((fixed - result.warped) ** 2)
        self.assertLess(after, before / 5)
        self.assertLess(abs(np.median(result.field[..., 0]) + 2), 1.5)
        self.assertLess(abs(np.median(result.field[..., 1]) - 3), 1.5)

    def test_subject_split_and_dice(self):
        splits = split_subjects([f"subject-{i}" for i in range(20)])
        self.assertEqual(sum(map(len, splits.values())), 20)
        self.assertEqual(len(set(sum(splits.values(), []))), 20)
        fixed = np.array([[0, 1], [1, 2]])
        moving = np.array([[0, 1], [2, 2]])
        self.assertEqual(dice(fixed, moving), {1: 2 / 3, 2: 2 / 3})


if __name__ == "__main__":
    unittest.main()
