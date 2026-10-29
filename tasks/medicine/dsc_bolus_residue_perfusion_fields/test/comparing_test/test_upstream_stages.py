"""Source inverse stages frozen before extraction; independently identified matrices."""
from pathlib import Path
import numpy as np
import pytest
from src.preprocessing import signal_to_concentration
from src.physics_model import convolution_matrix
from src.solvers import deconvolve,leakage_volume
T=Path(__file__).resolve().parents[2]
@pytest.fixture(scope='module')
def fixture():
 with np.load(T/'test/fixtures/upstream_stages.npz',allow_pickle=False) as z:return {k:z[k] for k in z.files}
@pytest.mark.parametrize('circulant',[False,True])
def test_matrix(fixture,circulant):
 np.testing.assert_allclose(convolution_matrix(fixture['aif'],circulant),fixture['circulant_matrix' if circulant else 'causal_matrix'],rtol=0,atol=1e-14)
 with np.load(T/'test/fixtures/native_matrices.npz',allow_pickle=False) as native:
  np.testing.assert_allclose(convolution_matrix(fixture['aif'],circulant),native['cSVD' if circulant else 'SVD'],rtol=0,atol=1e-14)
  if circulant:np.testing.assert_allclose(convolution_matrix(fixture['aif'],True),native['oSVD'],rtol=0,atol=1e-14)
@pytest.mark.parametrize('method',['svd','csvd','osvd'])
def test_inverse_stages(fixture,method):
 result=deconvolve(fixture['concentration'],fixture['aif'],method=method)
 for name in ['map','residual']:
  np.testing.assert_allclose(result[name],fixture[method+'_'+name],rtol=2e-7,atol=2e-9)
 if method=='osvd':np.testing.assert_allclose(result['oscillation'],fixture['osvd_oscillation'],rtol=2e-7,atol=2e-9)
 np.testing.assert_allclose(fixture['cbv_lc']/result['map'],fixture[method+'_mtt'],rtol=2e-7,atol=2e-9)
def test_complete_preprocessing_and_leakage(fixture):
 with np.load(T/'data/raw_data.npz',allow_pickle=False) as raw:
  concentration,s0,bolus=signal_to_concentration(raw['signal'][0],raw['support'][0])
  result=leakage_volume(concentration,raw['arterial_input'][0],raw['support'][0],bolus)
 index=fixture['coordinates'];sel=(index[:,2],index[:,0],index[:,1])
 np.testing.assert_array_equal(bolus,fixture['bolus'])
 np.testing.assert_allclose(s0[sel],fixture['s0'],rtol=2e-7,atol=2e-9)
 np.testing.assert_allclose(concentration[sel],fixture['concentration'],rtol=2e-7,atol=2e-9)
 for key in ['cbv','cbv_lc','K1','K2','K1cv','K2cv']:
  np.testing.assert_allclose(result[key][sel],fixture[key],rtol=2e-7,atol=2e-9)
