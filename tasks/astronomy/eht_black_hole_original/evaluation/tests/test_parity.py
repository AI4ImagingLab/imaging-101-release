"""
Parity Tests: Cleaned Code vs ehtim Reference
==============================================

These tests verify that our self-contained implementation produces
numerically identical results to ehtim at every pipeline stage.

Fixtures are frozen numeric outputs from the archived reference run.
"""

import os
import unittest
import numpy as np

TASK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FIX_DIR = os.path.join(TASK_DIR, 'evaluation', 'fixtures')

import sys
sys.path.insert(0, TASK_DIR)


class TestDFTMatrix(unittest.TestCase):
    """Verify our DFT matrix matches ehtim's ftmatrix exactly."""

    def setUp(self):
        f = np.load(os.path.join(FIX_DIR, 'physics_model', 'ftmatrix.npz'))
        self.ehtim_rows = f['A_rows']
        self.uv_rows = f['uv_rows']
        self.psize = float(f['pixel_size_rad'])
        self.N = int(f['N'])

    def test_dft_matrix_matches_ehtim(self):
        from src.physics_model import ClosureForwardModel
        model = ClosureForwardModel(
            uv_coords=self.uv_rows,
            N=self.N,
            pixel_size_rad=self.psize,
            triangles=np.zeros((0, 3), dtype=int),
            quadrangles=np.zeros((0, 4), dtype=int),
        )
        np.testing.assert_allclose(
            model.A, self.ehtim_rows,
            rtol=1e-10, atol=1e-15,
            err_msg="DFT matrix does not match ehtim ftmatrix"
        )


class TestFullDFTAndForward(unittest.TestCase):
    """Verify full A matrix and forward model output."""

    def setUp(self):
        with np.load(os.path.join(TASK_DIR, 'data', 'raw_data.npz'), allow_pickle=False) as raw:
            self.uv_coords = raw['uv_coords'][0]
        f2 = np.load(os.path.join(FIX_DIR, 'physics_model', 'forward.npz'))
        self.gt_jy = f2['input_image']
        self.vis_model = f2['output_vis']

    def test_forward_model(self):
        from src.physics_model import ClosureForwardModel
        f_meta = np.load(os.path.join(FIX_DIR, 'physics_model', 'ftmatrix.npz'))
        psize = float(f_meta['pixel_size_rad'])
        N = int(f_meta['N'])
        model = ClosureForwardModel(
            uv_coords=self.uv_coords,
            N=N, pixel_size_rad=psize,
            triangles=np.zeros((0, 3), dtype=int),
            quadrangles=np.zeros((0, 4), dtype=int),
        )
        vis_out = model.forward(self.gt_jy)
        np.testing.assert_allclose(
            vis_out, self.vis_model, rtol=1e-10,
            err_msg="Forward model output mismatch"
        )


class TestClosurePhaseChiSquared(unittest.TestCase):
    """Verify closure phase chi-squared and gradient match ehtim."""

    def setUp(self):
        f = np.load(os.path.join(FIX_DIR, 'physics_model', 'chisq_cphase.npz'))
        self.gt_img = f['input_image_gt']
        self.pert_img = f['input_image_pert']
        self.cp_values_deg = f['cp_values_deg']
        self.cp_sigmas_deg = f['cp_sigmas_deg']
        self.cp_u1 = f['cp_u1']
        self.cp_u2 = f['cp_u2']
        self.cp_u3 = f['cp_u3']
        self.chisq_gt = float(f['output_chisq_gt'])
        self.grad_gt = f['output_grad_gt']
        self.chisq_pert = float(f['output_chisq_pert'])
        self.grad_pert = f['output_grad_pert']

        f_meta = np.load(os.path.join(FIX_DIR, 'physics_model', 'ftmatrix.npz'))
        self.psize = float(f_meta['pixel_size_rad'])
        self.N = int(f_meta['N'])

    def _build_model(self):
        from src.physics_model import ClosureForwardModel
        # Build with empty triangles/quadrangles — we test chi-squared directly
        # using the per-uv Amatrices
        return ClosureForwardModel(
            uv_coords=np.zeros((0, 2)),  # dummy
            N=self.N, pixel_size_rad=self.psize,
            triangles=np.zeros((0, 3), dtype=int),
            quadrangles=np.zeros((0, 4), dtype=int),
        )

    def test_chisq_cphase_gt(self):
        from src.physics_model import ClosureForwardModel
        chisq = ClosureForwardModel.chisq_cphase_from_uv(
            self.gt_img.flatten(), self.N, self.psize,
            self.cp_u1, self.cp_u2, self.cp_u3,
            self.cp_values_deg, self.cp_sigmas_deg
        )
        np.testing.assert_allclose(chisq, self.chisq_gt, rtol=1e-8)

    def test_chisq_cphase_pert(self):
        from src.physics_model import ClosureForwardModel
        chisq = ClosureForwardModel.chisq_cphase_from_uv(
            self.pert_img.flatten(), self.N, self.psize,
            self.cp_u1, self.cp_u2, self.cp_u3,
            self.cp_values_deg, self.cp_sigmas_deg
        )
        np.testing.assert_allclose(chisq, self.chisq_pert, rtol=1e-8)

    def test_grad_cphase_gt(self):
        from src.physics_model import ClosureForwardModel
        grad = ClosureForwardModel.chisqgrad_cphase_from_uv(
            self.gt_img.flatten(), self.N, self.psize,
            self.cp_u1, self.cp_u2, self.cp_u3,
            self.cp_values_deg, self.cp_sigmas_deg
        )
        # Slightly looser tolerance for gt image (near-zero residuals amplify float noise)
        np.testing.assert_allclose(grad, self.grad_gt, rtol=1e-7, atol=1e-10)

    def test_grad_cphase_pert(self):
        from src.physics_model import ClosureForwardModel
        grad = ClosureForwardModel.chisqgrad_cphase_from_uv(
            self.pert_img.flatten(), self.N, self.psize,
            self.cp_u1, self.cp_u2, self.cp_u3,
            self.cp_values_deg, self.cp_sigmas_deg
        )
        np.testing.assert_allclose(grad, self.grad_pert, rtol=1e-8, atol=1e-15)


class TestLogClosureAmpChiSquared(unittest.TestCase):
    """Verify log closure amplitude chi-squared and gradient match ehtim."""

    def setUp(self):
        f = np.load(os.path.join(FIX_DIR, 'physics_model', 'chisq_logcamp.npz'))
        self.gt_img = f['input_image_gt']
        self.pert_img = f['input_image_pert']
        self.lca_values = f['lca_values']
        self.lca_sigmas = f['lca_sigmas']
        self.lca_u1 = f['lca_u1']
        self.lca_u2 = f['lca_u2']
        self.lca_u3 = f['lca_u3']
        self.lca_u4 = f['lca_u4']
        self.chisq_gt = float(f['output_chisq_gt'])
        self.grad_gt = f['output_grad_gt']
        self.chisq_pert = float(f['output_chisq_pert'])
        self.grad_pert = f['output_grad_pert']

        f_meta = np.load(os.path.join(FIX_DIR, 'physics_model', 'ftmatrix.npz'))
        self.psize = float(f_meta['pixel_size_rad'])
        self.N = int(f_meta['N'])

    def test_chisq_logcamp_gt(self):
        from src.physics_model import ClosureForwardModel
        chisq = ClosureForwardModel.chisq_logcamp_from_uv(
            self.gt_img.flatten(), self.N, self.psize,
            self.lca_u1, self.lca_u2, self.lca_u3, self.lca_u4,
            self.lca_values, self.lca_sigmas
        )
        np.testing.assert_allclose(chisq, self.chisq_gt, rtol=1e-8)

    def test_chisq_logcamp_pert(self):
        from src.physics_model import ClosureForwardModel
        chisq = ClosureForwardModel.chisq_logcamp_from_uv(
            self.pert_img.flatten(), self.N, self.psize,
            self.lca_u1, self.lca_u2, self.lca_u3, self.lca_u4,
            self.lca_values, self.lca_sigmas
        )
        np.testing.assert_allclose(chisq, self.chisq_pert, rtol=1e-8)

    def test_grad_logcamp_gt(self):
        from src.physics_model import ClosureForwardModel
        grad = ClosureForwardModel.chisqgrad_logcamp_from_uv(
            self.gt_img.flatten(), self.N, self.psize,
            self.lca_u1, self.lca_u2, self.lca_u3, self.lca_u4,
            self.lca_values, self.lca_sigmas
        )
        np.testing.assert_allclose(grad, self.grad_gt, rtol=1e-8, atol=1e-15)

    def test_grad_logcamp_pert(self):
        from src.physics_model import ClosureForwardModel
        grad = ClosureForwardModel.chisqgrad_logcamp_from_uv(
            self.pert_img.flatten(), self.N, self.psize,
            self.lca_u1, self.lca_u2, self.lca_u3, self.lca_u4,
            self.lca_values, self.lca_sigmas
        )
        np.testing.assert_allclose(grad, self.grad_pert, rtol=1e-8, atol=1e-15)


class TestRegularizers(unittest.TestCase):
    """Verify regularizers match ehtim's implementations."""

    def setUp(self):
        f = np.load(os.path.join(FIX_DIR, 'solvers', 'regularizers.npz'))
        self.test_img = f['input_image']
        self.prior_img = f['prior_image']
        self.gs_val = float(f['gs_val'])
        self.gs_grad = f['gs_grad']
        self.simple_val = float(f['simple_val'])
        self.simple_grad = f['simple_grad']
        self.total_flux = float(f['total_flux'])

    def test_gs_entropy_value(self):
        from src.solvers import GullSkillingRegularizer
        reg = GullSkillingRegularizer(prior=self.prior_img)
        val, grad = reg.value_and_grad(self.test_img)
        np.testing.assert_allclose(val, self.gs_val, rtol=1e-10)

    def test_gs_entropy_grad(self):
        from src.solvers import GullSkillingRegularizer
        reg = GullSkillingRegularizer(prior=self.prior_img)
        val, grad = reg.value_and_grad(self.test_img)
        np.testing.assert_allclose(grad.ravel(), self.gs_grad, rtol=1e-10)

    def test_simple_entropy_value(self):
        from src.solvers import SimpleEntropyRegularizer
        reg = SimpleEntropyRegularizer(prior=self.prior_img)
        val, grad = reg.value_and_grad(self.test_img)
        np.testing.assert_allclose(val, self.simple_val, rtol=1e-10)

    def test_simple_entropy_grad(self):
        from src.solvers import SimpleEntropyRegularizer
        reg = SimpleEntropyRegularizer(prior=self.prior_img)
        val, grad = reg.value_and_grad(self.test_img)
        np.testing.assert_allclose(grad.ravel(), self.simple_grad, rtol=1e-10)


if __name__ == '__main__':
    unittest.main()
