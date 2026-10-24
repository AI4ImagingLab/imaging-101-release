from pathlib import Path
import numpy as np
import pytest
import torch
from src.physics_model import wrapped_laplacian
from src.solvers import primal_dual, primal_dual_torch

FIX = Path(__file__).resolve().parents[1]/'fixtures'


def test_independent_wrapped_laplacian_capture():
    with np.load(FIX/'upstream_input.npz', allow_pickle=False) as z:
        np.testing.assert_allclose(wrapped_laplacian(z['phase'], z['spacing']), z['laplacian'], atol=2e-13, rtol=0)


@pytest.mark.parametrize('iterations', [1, 2, 10, 50])
@pytest.mark.parametrize('backend', ['numpy', 'torch_cpu'])
def test_independent_corrected_c_state_capture(iterations, backend):
    torch.set_num_threads(1)
    with np.load(FIX/'upstream_input.npz', allow_pickle=False) as z:
        kwargs = dict(spacing=z['spacing'], lam=.001, ratio=3, tolerance=0,
                      max_iterations=iterations, return_state=True)
        state = primal_dual(-z['laplacian'], z['mask'], **kwargs) if backend == 'numpy' else primal_dual_torch(-z['laplacian'], z['mask'], device='cpu', **kwargs)
    with np.load(FIX/f'corrected_step_{iterations}.npz', allow_pickle=False) as ref:
        for key in ref.files:
            np.testing.assert_allclose(state[key], ref[key], rtol=0, atol=5e-11, err_msg=key)


@pytest.mark.parametrize('iterations', [1, 2, 10, 50])
def test_independent_unmodified_tv_c_capture(iterations):
    with np.load(FIX/'upstream_input.npz', allow_pickle=False) as z:
        result = primal_dual(-z['laplacian'], z['mask'], z['spacing'], .001, 0, 0, iterations)
    with np.load(FIX/f'tv_step_{iterations}.npz', allow_pickle=False) as ref:
        np.testing.assert_allclose(result['chi'], ref['chi'], rtol=0, atol=5e-11)
