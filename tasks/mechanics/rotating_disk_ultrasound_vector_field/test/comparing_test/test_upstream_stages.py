from pathlib import Path
import numpy as np
from scipy import sparse
from src.preprocessing import load_observations
from src.physics_model import demodulate,automatic_fnumber,beamformer,doppler_velocity
from src.solvers import robust_smooth,dct_filter

ROOT=Path(__file__).resolve().parents[2]
def fixture():
 with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False) as d:return {k:d[k] for k in d.files}

def test_original_demodulation_complete_selected_traces():
 f=fixture();data=load_observations(ROOT/'data/raw_data.npz')
 iq=demodulate(data['rf'][:,f['trace_channels'],:])
 np.testing.assert_allclose(iq,f['iq_traces'],rtol=2e-10,atol=1e-8)

def test_original_receive_operator_rows_and_directivity():
 f=fixture();idx=f['beam_sample_indices_fortran'];x=f['x'].ravel(order='F')[idx];z=f['z'].ravel(order='F')[idx]
 assert abs(automatic_fnumber(np.pi/12)-3.365971459145058)<1e-14
 for theta,name in [(np.pi/12,'Mp_rows'),(-np.pi/12,'Mm_rows')]:
  actual=beamformer(x,z,theta)
  expected=sparse.csr_matrix((f[name+'_data'],f[name+'_indices'],f[name+'_indptr']),shape=tuple(f[name+'_shape']))
  np.testing.assert_array_equal(actual.indptr,expected.indptr);np.testing.assert_array_equal(actual.indices,expected.indices)
  np.testing.assert_allclose(actual.data,expected.data,rtol=2e-10,atol=1e-10)

def test_original_phase_velocity_and_variance_known_case():
 f=fixture();velocity,variance,ac=doppler_velocity(f['known_iq'],window=3)
 np.testing.assert_allclose(velocity,f['known_velocity'],atol=2e-12)
 np.testing.assert_allclose(variance,f['known_variance'],atol=2e-12)
 np.testing.assert_allclose(velocity,.74*.4/np.pi,atol=2e-12)

def test_original_first_smoothing_update_and_unweighted_filter():
 f=fixture();actual,diag=robust_smooth(f['raw_vector'],f['roi'],robust=False,max_iter=1)
 np.testing.assert_allclose(actual,f['first_smoothing_update'],rtol=2e-10,atol=2e-9)
 np.testing.assert_allclose(dct_filter(f['small'],2.5),f['small_filtered'],rtol=2e-12,atol=2e-12)

def test_original_complete_robust_smoothing():
 f=fixture();actual,diag=robust_smooth(f['raw_vector'],f['roi'])
 np.testing.assert_allclose(actual,f['smoothed_vector'],rtol=2e-10,atol=2e-9)
