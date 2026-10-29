from pathlib import Path
import numpy as np
from scipy import sparse
from src.preprocessing import prepare
from src.physics_model import DopplerModel
ROOT=Path(__file__).resolve().parents[2]
def test_spectral_preprocessing():
 raw=dict(np.load(ROOT/'data/raw_data.npz'));out=prepare(raw);f=np.load(ROOT/'test/fixtures/original_spectral_preprocessing.npz')
 np.testing.assert_allclose(out['velocity_cm_s'],np.array([float(f'{x*1e5:.5e}') for x in f['velocity']]),atol=1e-9)
 np.testing.assert_allclose(out['unquantized_line_spectra'],f['normalised_flux'],atol=1e-9,rtol=1e-12)
def test_projection_and_preprocessing_stages():
 raw=dict(np.load(ROOT/'data/raw_data.npz'));prepared=prepare(raw);model=DopplerModel(prepared);f=np.load(ROOT/'test/fixtures/original_geometry.npz');q=np.load(ROOT/'test/fixtures/original_projection_columns.npz')
 np.testing.assert_array_equal(model.ix,f['index_x']);np.testing.assert_array_equal(model.iy,f['index_y'])
 reference=sparse.csc_matrix((q['weights'],q['rows'],q['indptr']),shape=tuple(q['shape']));difference=model.P_csc[:,q['columns']]-reference
 assert difference.nnz==0 or np.max(np.abs(difference.data))<2e-7
 state=model.initialize_data(prepared['line_spectra']);original=np.load(ROOT/'test/fixtures/original_preprocessing.npz')
 for key in ['positive_data','baseline_data','baseline_field','weights']:np.testing.assert_allclose(state[key],original[key],atol=1e-9,rtol=2e-7)
def test_complete_original_pipeline():
 from src.solvers import reconstruct
 output=reconstruct(dict(np.load(ROOT/'data/raw_data.npz')));ref=np.load(ROOT/'evaluation/baseline_reference.npz')['reconstruction']
 error=np.sqrt(np.mean((output['reconstruction']-ref)**2))/np.ptp(ref)
 assert error<=7.691681073121665e-6
 assert output['iterations']==27

def test_first_two_source_updates():
 from src.solvers import reconstruct
 output=reconstruct(dict(np.load(ROOT/'data/raw_data.npz')))
 for i in range(2):
  f=np.load(ROOT/f'test/fixtures/original_iteration{i+1}.npz')
  for key in ['field_after_update','forward_before_update','likelihood_gradient_projected','entropy_gradient_projected','floating_default_before_update','new_direction','saved_direction_after_update']:
   a=output['first_iterations'][i][key];b=f[key]
   error=np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-300)
   tolerance=2e-6 if 'gradient' in key else 2e-7 if key in ['forward_before_update','floating_default_before_update'] else 5e-6
   assert error<tolerance,(i,key,error,tolerance)
