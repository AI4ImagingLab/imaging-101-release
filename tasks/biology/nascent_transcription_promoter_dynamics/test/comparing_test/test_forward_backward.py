from pathlib import Path
import numpy as np
import pytest
from src.solvers import forward_backward
F=Path(__file__).resolve().parents[1]/'fixtures'
@pytest.mark.parametrize('index',[0,62,63])
def test_corrected_oracle_forward_backward(index):
    with np.load(F/f'forward_backward_{index}.npz',allow_pickle=False) as z:
        args=[[] if i==8 else z[f'input_{i}'].copy() for i in range(19)]
        result=forward_backward(*args)
        for j,gamma in enumerate(result['Gamma']):
            np.testing.assert_array_equal(list(gamma),z[f'output_gamma_states_{j}'])
            np.testing.assert_allclose(list(gamma.values()),z[f'output_gamma_log_{j}'],rtol=1e-6,atol=1e-7)
        for key,value in result.items():
            if key!='Gamma':np.testing.assert_allclose(value,z['output_'+key],rtol=1e-6,atol=1e-7)
