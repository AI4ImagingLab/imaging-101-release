"""Check general ordered CUDA arithmetic against independent serial CPU sums."""
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.deterministic_sparse import OrderedCsr


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_unsorted_csr_repeats_and_transpose(dtype):
    cp = pytest.importorskip('cupy')
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip('ordered sparse arithmetic requires a CUDA device')
    matrix = csr_matrix((np.array([1e10, -1e10, 1.2, 4., .5, -.5], dtype=dtype),
                         np.array([2, 0, 1, 3, 2, 0]),
                         np.array([0, 3, 4, 6])), shape=(3, 4))
    values = np.random.default_rng(743).normal(size=(4, 5)).astype(dtype)
    operator = OrderedCsr(matrix)
    expected = matrix @ values
    for _ in range(4):
        result = operator.multiply(np.asfortranarray(values))
        assert result.flags.f_contiguous
        np.testing.assert_array_equal(result, expected)
    transpose = OrderedCsr(matrix.T.tocsr())
    np.testing.assert_array_equal(transpose.multiply(expected), matrix.T.tocsr() @ expected)
    with pytest.raises(TypeError, match='dtype'):
        operator.multiply(values.astype(np.int32))
    with pytest.raises(ValueError, match='compatible'):
        operator.multiply(values[:2])
    operator.close()
    transpose.close()
    with pytest.raises(RuntimeError, match='closed'):
        operator.multiply(values)
