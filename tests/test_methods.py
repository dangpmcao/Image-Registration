"""Algorithmic checks, independent of model accuracy or downloaded datasets."""

from importlib import import_module
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cnn.bspline import interpolate_control_grid
from cnn.devos import (AffineStage3D, DLIR3D, bending_energy_mm, compose_pull_fields,
                       nlst_stage_specs, sample_grid, sunnybrook_stage_specs)
from cnn.dlir_training import save_checkpoint, stage_loss
from cnn.model import RegistrationCNN, ncc_loss
from cnn.model3d import RegistrationCNN3D, ncc_loss3d, warp_torch3d
from data.geometry import bspline_basis
from data.geometry3d import warp3d
from data.metrics import landmark_errors, normalized_mutual_information, surface_distances

rueckert = import_module("b-spline.rueckert")


class MethodsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(7)
        self.rng = np.random.default_rng(7)

    def test_transposed_spline_matches_tensor_product_and_gradient(self):
        shape, strides = (7, 10, 13), (1, 2, 4)
        bases = [torch.tensor(bspline_basis(n, s), dtype=torch.float64) for n, s in zip(shape, strides)]
        control = torch.randn(2, 3, *[b.shape[1] for b in bases], dtype=torch.float64, requires_grad=True)
        dense = interpolate_control_grid(control, shape, strides)
        expected = torch.einsum("za,bcade,yd,xe->bzyxc", bases[0], control, bases[1], bases[2])
        torch.testing.assert_close(dense, expected)
        actual_grad = torch.autograd.grad(dense.square().sum(), control)[0]
        expected_grad = torch.autograd.grad(expected.square().sum(), control)[0]
        torch.testing.assert_close(actual_grad, expected_grad)

    def test_exact_cubic_subdivision(self):
        shape = (17, 20, 23)
        coarse = [bspline_basis(n, 8) for n in shape]
        fine = [bspline_basis(n, 4) for n in shape]
        control = self.rng.normal(size=(*[b.shape[1] for b in coarse], 3)).astype(np.float32)
        refined = rueckert.subdivide_lattice(control, tuple(b.shape[1] for b in fine))
        before = np.einsum("za,abcd,yb,xc->zyxd", coarse[0], control, coarse[1], coarse[2])
        after = np.einsum("za,abcd,yb,xc->zyxd", fine[0], refined, fine[1], fine[2])
        np.testing.assert_allclose(before, after, atol=2e-7)

    def test_physical_bending_has_known_value(self):
        spacing = (2.0, 3.0, 4.0)
        field = torch.zeros(1, 7, 8, 9, 3)
        z_mm = torch.arange(7) * spacing[0]
        field[0, ..., 0] = z_mm[:, None, None].square() / spacing[0]
        self.assertAlmostEqual(float(bending_energy_mm(field, spacing)), 4.0)
        physical = field[0] * field.new_tensor(spacing)
        self.assertAlmostEqual(float(rueckert._bending(physical, field.new_tensor(spacing))), 4.0)

    def test_affine_bending_is_zero(self):
        grid = torch.stack(torch.meshgrid(torch.arange(5), torch.arange(6), torch.arange(7), indexing="ij"), -1).float()
        field = (grid * grid.new_tensor((0.1, -0.2, 0.3)) + 2)[None]
        self.assertLess(float(bending_energy_mm(field, (2, 3, 4))), 1e-10)

    def test_composition_is_pull_composition(self):
        grid = torch.arange(7).view(1, 7, 1, 1).expand(1, 7, 8, 9)
        base = torch.zeros(1, 7, 8, 9, 3)
        base[..., 0] = grid * 0.1
        residual = torch.zeros_like(base)
        residual[..., 0] = 1
        result = compose_pull_fields(base, residual)
        torch.testing.assert_close(result[:, :-1, ..., 0], 1 + 0.1 * (grid[:, :-1] + 1))
        self.assertFalse(torch.allclose(result, base + residual))

    def test_torch_and_numpy_pull_warps_agree_at_boundaries(self):
        image = self.rng.random((6, 7, 8)).astype(np.float32)
        field = self.rng.uniform(-1.5, 1.5, (*image.shape, 3)).astype(np.float32)
        actual = warp_torch3d(torch.from_numpy(image)[None, None], torch.from_numpy(field)[None])
        np.testing.assert_allclose(actual[0, 0].numpy(), warp3d(image, field), atol=8e-7)

    def test_affine_identity_for_batch(self):
        model = AffineStage3D()
        for parameter in model.parameters():
            torch.nn.init.zeros_(parameter)
        field = model(torch.rand(2, 1, 6, 9, 12), torch.rand(2, 1, 6, 9, 12), (2, 1, 1))
        self.assertEqual(field.shape, (2, 6, 9, 12, 3))
        torch.testing.assert_close(field, torch.zeros_like(field))

    def test_sequential_stage_freezes_preceding_parameters(self):
        model = DLIR3D(sunnybrook_stage_specs("multi"))
        model.train_only_stage(1)
        fixed, moving = torch.rand(2, 1, 6, 24, 24), torch.rand(2, 1, 6, 24, 24)
        loss = stage_loss(model, fixed, moving, (8, 1.25, 1.25), 1, 0.05)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is None for p in model.stages[0].parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.stages[1].parameters()))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_spacing_mismatch_is_explicit(self):
        model = DLIR3D(sunnybrook_stage_specs("single"))
        with self.assertRaisesRegex(ValueError, "grid spacing"):
            model(torch.rand(1, 1, 6, 24, 24), torch.rand(1, 1, 6, 24, 24), voxel_spacing=(10, 1.4, 1.4))

    def test_checkpoint_restores_optimizer_and_rng_mid_stage(self):
        specs = sunnybrook_stage_specs("single")
        model = DLIR3D(specs)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        rng = np.random.default_rng(17)

        def step(network, optim):
            fixed = torch.rand(1, 1, 6, 16, 16)
            moving = torch.rand_like(fixed) * float(rng.uniform(0.8, 1.2))
            optim.zero_grad()
            loss = stage_loss(network, fixed, moving, (8, 1.25, 1.25), 0, 0.05)
            loss.backward()
            optim.step()
            return loss.detach()

        step(model, optimizer)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            save_checkpoint(path, model, optimizer, {"stage_specs": specs}, 0, 1, -0.5, rng)
            expected_loss = step(model, optimizer)
            restored = DLIR3D(specs)
            restored_optimizer = torch.optim.Adam(restored.parameters(), lr=1e-4)
            checkpoint = torch.load(path, weights_only=True)
            restored.load_state_dict(checkpoint["model"])
            restored_optimizer.load_state_dict(checkpoint["optimizer"])
            rng.bit_generator.state = checkpoint["rng_state"]
            torch.set_rng_state(checkpoint["torch_rng_state"])
            actual_loss = step(restored, restored_optimizer)
            torch.testing.assert_close(actual_loss, expected_loss, atol=0, rtol=0)
            for expected, actual in zip(model.parameters(), restored.parameters()):
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_chest_architecture_has_strided_downsampling(self):
        specs = nlst_stage_specs()
        model = DLIR3D(specs[:1])
        x = torch.rand(1, 1, 9, 10, 11)
        field, warped = model(x, x, voxel_spacing=(8, 8, 8))
        self.assertEqual(field.shape, (1, 9, 10, 11, 3))
        self.assertEqual(warped.shape, x.shape)
        self.assertEqual(specs[0]["depth"], 3)
        self.assertTrue(all(spec["downsampling"] == "strided" for spec in specs))

    def test_zero_variance_ncc_has_finite_gradient(self):
        for shape, function in [((1, 1, 6, 7), ncc_loss), ((1, 1, 5, 6, 7), ncc_loss3d)]:
            moving = torch.zeros(shape, requires_grad=True)
            loss = function(torch.zeros(shape), moving, torch.ones(shape))
            loss.backward()
            self.assertTrue(torch.isfinite(moving.grad).all())

    def test_baselines_still_start_at_identity(self):
        for model, shape in [(RegistrationCNN((12, 16), 4), (2, 1, 12, 16)),
                             (RegistrationCNN3D((6, 12, 16), 4), (2, 1, 6, 12, 16))]:
            x = torch.rand(shape)
            field, warped = model(x, x)
            torch.testing.assert_close(field, torch.zeros_like(field))
            torch.testing.assert_close(warped, x, atol=1e-6, rtol=1e-6)

    def test_rueckert_zero_iterations_and_multiresolution(self):
        x = self.rng.random((9, 12, 13)).astype(np.float32)
        result = rueckert.register(x, x, voxel_spacing=(2, 1, 1), max_dim=10, levels=3,
                                   affine_iterations=0, ffd_iterations=0)
        np.testing.assert_array_equal(result.field, np.zeros((*x.shape, 3)))
        np.testing.assert_array_equal(result.warped, x)
        np.testing.assert_array_equal(result.affine_matrix, np.eye(4))

    def test_rueckert_optimization_improves_its_objective(self):
        z, y, x = np.indices((9, 11, 13), dtype=np.float32)
        fixed = np.exp(-((z-4)**2 + (y-5)**2 + (x-6)**2) / 8).astype(np.float32)
        moving = np.roll(fixed, 1, axis=2)
        result = rueckert.register(fixed, moving, voxel_spacing=(2, 1, 1), levels=1,
                                   max_dim=13, affine_iterations=3, ffd_iterations=3)
        self.assertTrue(np.isfinite(result.field).all())
        self.assertLess(result.history[-1], result.history[0])

    def test_landmark_direction_and_physical_units(self):
        field = np.zeros((6, 7, 8, 3), np.float32)
        field[..., 2] = 2
        result = landmark_errors([[2, 3, 2]], [[2, 3, 4]], field, (3, 2, 1.5))
        self.assertEqual(result["before_mean_mm"], 3)
        self.assertEqual(result["after_mean_mm"], 0)

    def test_nmi_and_surface_distance(self):
        x = np.arange(120).reshape(4, 5, 6).astype(float)
        self.assertAlmostEqual(normalized_mutual_information(x, x, np.ones_like(x, bool)), 2)
        a, b = np.zeros((7, 8, 9), int), np.zeros((7, 8, 9), int)
        a[3, 4, 3], b[3, 4, 4] = 1, 1
        result = surface_distances(a, b, (3, 2, 1.5))["1"]
        self.assertEqual(result, {"asd_mm": 1.5, "hausdorff_mm": 1.5})


if __name__ == "__main__":
    unittest.main()
