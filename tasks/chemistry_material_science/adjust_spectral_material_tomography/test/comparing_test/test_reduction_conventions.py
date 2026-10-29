"""Independent original-Octave numeric call roles, captured before extraction."""
from pathlib import Path
import numpy as np
import pytest
from src.solvers import frobenius_norm, _dot, _sum

FIXTURE = Path(__file__).resolve().parents[1] / 'fixtures/upstream_reduction_conventions.npz'


def test_original_serial_scaled_frobenius_norm():
    with np.load(FIXTURE, allow_pickle=False) as reference:
        for name, expected in zip(reference['names'], reference['expected_norms']):
            assert frobenius_norm(reference['input_' + name]) == expected


def test_original_native_vector_spg_scalar_roles():
    with np.load(FIXTURE, allow_pickle=False) as reference:
        g, d = reference['input_spg_gradient'], reference['input_spg_direction']
        s, y = reference['input_spg_step'], reference['input_spg_gradient_change']
        for value, name in [(_dot(g, d), 'vector_dot'), (_dot(s, s), 'squared_step'),
                            (_dot(s, y), 'curvature')]:
            np.testing.assert_allclose(value, reference['expected_' + name], rtol=2e-14, atol=2e-14)
        assert _sum(abs(g)) == reference['expected_gradient_l1']


def test_hybrid_sparse_boundary_and_original_two_steps():
    import torch
    from scipy import sparse
    from src.solvers import LinearMap, adjust
    if not torch.cuda.is_available():
        pytest.skip('CUDA sparse backend requires a CUDA device')
    path = FIXTURE.parent / 'upstream_four_stages.npz'
    with np.load(path, allow_pickle=False) as reference:
        matrix = sparse.csr_matrix(reference['W'])
        operator = LinearMap(matrix, 'cuda_sparse')
        value = operator.array(np.ones((matrix.shape[1], 4)))
        assert value.flags.f_contiguous
        forward = operator.multiply(value)
        assert forward.flags.f_contiguous
        np.testing.assert_allclose(forward, matrix @ value, rtol=2e-14, atol=2e-14)
        adjoint = operator.transpose(forward)
        assert adjoint.flags.f_contiguous
        np.testing.assert_allclose(adjoint, matrix.T @ forward, rtol=2e-14, atol=2e-14)
        result = adjust(reference['Y'], matrix, reference['T'], backend='cuda_sparse',
                        iterations=2, verbose=False)
        for local, original in [('maps', 'Ac'), ('spectra', 'Fc'), ('weights', 'Rc')]:
            np.testing.assert_allclose(result[local], reference[original], rtol=2e-10, atol=2e-11)
