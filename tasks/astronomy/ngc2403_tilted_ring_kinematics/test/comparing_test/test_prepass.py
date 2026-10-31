from pathlib import Path
import numpy as np
import pytest
from src import solvers
T=Path(__file__).resolve().parents[2]
F=T/'test/fixtures'

@pytest.mark.parametrize('number',[1,2])
def test_original_prepass_normal_equations(number):
    function=getattr(solvers,'prepass_normal_equations',None)
    assert callable(function),'prepass linearization is not implemented'
    with np.load(F/f'prepass_lsq_{number}.npz',allow_pickle=False) as z:
        chi,matrix,vector=function(z['input_coordinates'],z['input_velocity'],z['input_weight'],z['input_parameters'])
        np.testing.assert_allclose(chi,z['output_chi'],rtol=0,atol=1e-5)
        np.testing.assert_allclose(matrix,z['output_matrix'],rtol=0,atol=1e-5)
        np.testing.assert_allclose(vector,z['output_vector'],rtol=0,atol=1e-5)

def test_original_complete_2d_initialization():
    function=getattr(solvers,'initialize_rings',None)
    assert callable(function),'original 2D initialization is not implemented'
    with np.load(F/'preprocessing.npz',allow_pickle=False) as z: moment=z['output_moment']
    actual,details=function(moment)
    with np.load(F/'prepass_full.npz',allow_pickle=False) as z:
        np.testing.assert_allclose(details['parameters'],z['output_parameters'],rtol=0,atol=5e-4)
        np.testing.assert_array_equal(details['counts'],z['output_counts'])
    assert actual.shape==(41,3)
    assert actual[0,0]==120 and actual[-1,0]==120
    assert actual[-1,2]==np.float32(123.7)
