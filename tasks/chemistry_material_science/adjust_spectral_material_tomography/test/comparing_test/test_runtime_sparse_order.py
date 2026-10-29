"""Verify CSR labels at the operator boundary after actual solver conversion."""
from pathlib import Path
import numpy as np
import pytest
from scipy.sparse import csr_matrix
import src.solvers as solvers
from src.physics_model import build_system_matrix, geometry_vectors


@pytest.mark.parametrize('backend', ['cpu', 'cpu_sparse', 'cuda_sparse'])
def test_adjust_actual_sparse_boundary_keeps_source_order(monkeypatch, backend):
    if backend == 'cuda_sparse':
        import torch
        if not torch.cuda.is_available():
            pytest.skip('CUDA sparse boundary requires a CUDA device')
    fixture = Path(__file__).resolve().parents[1] / 'fixtures/upstream_line_operator.npz'
    with np.load(fixture, allow_pickle=False) as reference:
        matrix = build_system_matrix(geometry_vectors(reference['param_angles']))
        expected = {name: reference[key].copy() for name, key in
            [('indptr', 'param_indptr'), ('indices', 'param_indices'), ('data', 'param_weights')]}

    class BoundaryReached(Exception):
        pass

    def inspect(self, value):
        if backend in ('cpu', 'cpu_sparse'):
            assert self.matrix.dtype == np.float64
            for name, array in expected.items():
                np.testing.assert_array_equal(getattr(self.matrix, name), array)
        else:
            actual = {'indptr': self.matrix.crow_indices().cpu().numpy(),
                      'indices': self.matrix.col_indices().cpu().numpy(),
                      'data': self.matrix.values().cpu().numpy()}
            assert actual['data'].dtype == np.float64
            for name, array in expected.items():
                np.testing.assert_array_equal(actual[name], array)
        raise BoundaryReached

    monkeypatch.setattr(solvers.LinearMap, 'multiply', inspect)
    with pytest.raises(BoundaryReached):
        solvers.adjust(np.ones((matrix.shape[0], 4)), matrix,
                       np.eye(4), iterations=1, backend=backend, verbose=False)


def test_dtype_change_never_sorts_csr():
    matrix = csr_matrix((np.array([1., 2., 3.], dtype=np.float32),
                         np.array([2, 0, 1]), np.array([0, 3])), shape=(1, 3))
    assert not matrix.has_sorted_indices
    converted = solvers.cast_sparse_preserving_order(matrix, np.float64)
    np.testing.assert_array_equal(converted.indices, matrix.indices)
    np.testing.assert_array_equal(converted.indptr, matrix.indptr)
    np.testing.assert_array_equal(converted.data, matrix.data)
    assert not converted.has_sorted_indices


def test_cpu_sparse_preserves_original_relay_dense_layout():
    matrix = csr_matrix([[1., 0., 2.], [0., 3., 4.]])
    operator = solvers.LinearMap(matrix, 'cpu_sparse')
    values = operator.array(np.arange(12.).reshape(3, 4))
    assert values.flags.f_contiguous
    assert operator.zeros((3, 4)).flags.f_contiguous
    forward = operator.multiply(values)
    assert forward.flags.f_contiguous
    np.testing.assert_array_equal(forward, matrix @ np.ascontiguousarray(values))
    np.testing.assert_array_equal(operator.multiply(values), forward)
    backward = operator.transpose(forward)
    assert backward.flags.f_contiguous
    np.testing.assert_array_equal(backward, matrix.T.tocsr() @ np.ascontiguousarray(forward))
