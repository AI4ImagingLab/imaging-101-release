from pathlib import Path
import numpy as np
from src.preprocessing import normalize_and_bin,load_observations
from src.physics_model import model_jac,equations
from src.solvers import gauss_jordan
T=Path(__file__).resolve().parents[2]
def test_native_source_normalization_before_binning():
 with np.load(T/'test/fixtures/native_preprocessing.npz',allow_pickle=False) as z:
  raw=z['raw'].reshape(2,2,8,1920,51)
  normalized=raw/raw.max(axis=-1,keepdims=True)
  assert np.max(abs(normalized.reshape(2,2,-1,51)-z['original_normalized']))<2e-14
  result=normalize_and_bin(raw)
  assert np.max(abs(result.reshape(2,2,-1,51)-z['original_binned']))<2e-14
  with np.load(T/'data/raw_data.npz',allow_pickle=False) as public:
   np.testing.assert_allclose(result,public['spectra'][0,:,:,100:102],rtol=0,atol=2e-14)
def test_captured_source_forward_jacobian():
 with np.load(T/'test/fixtures/source_model.npz',allow_pickle=False) as z:
  for window in range(2):
   index=z['range_index']==window
   y,j=model_jac(z['frequency_ghz'][window],z['parameters'][index])
   assert np.max(abs(y-z['values'][index]))<2e-6
   assert np.linalg.norm(j-z['jacobian'][index])/np.linalg.norm(z['jacobian'][index])<1e-7
def test_captured_source_normal_equations_and_steps():
 with np.load(T/'test/fixtures/source_lm.npz',allow_pickle=False) as z:
  for window in range(2):
   index=z['range_index']==window
   _,g,h=equations(z['frequency_ghz'][window],z['parameters'][index],z['data'][index])
   np.testing.assert_array_equal(g,z['gradient'][index])
   np.testing.assert_array_equal(h,z['hessian'][index])
   step=gauss_jordan(z['scaled_hessian'][index],g)
   np.testing.assert_array_equal(step,z['step'][index])

