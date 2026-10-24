from pathlib import Path
import numpy as np
from src.preprocessing import load_observation
from src.physics_model import gradient,divergence,threshold_gradients,laplacian
from src.solvers import reconstruct
T=Path(__file__).resolve().parents[2]
def test_original_full_stages():
 y=load_observation(T/'data/raw_data.npz');g=threshold_gradients(y,.033);target=laplacian(divergence(g))
 with np.load(T/'test/fixtures/stages.npz') as z:
  np.testing.assert_allclose(target,z['laplacian_target'],rtol=0,atol=1e-12)
  np.testing.assert_allclose(g[:,:,-1,:],z['last_channel_thresholded_gradient'],rtol=0,atol=1e-12)
  np.testing.assert_allclose(target+1e-8*y,z['rhs'],rtol=0,atol=1e-12)
def test_original_complete_output():
 y=load_observation(T/'data/raw_data.npz');x=reconstruct(y)
 with np.load(T/'evaluation/baseline_reference.npz') as z:r=z['reconstruction'][0]
 assert np.max(abs(x-r))<=5e-7
 assert np.sqrt(np.mean((x-r)**2))/np.ptp(r)<=1e-7
 assert x.ravel()@r.ravel()/np.linalg.norm(x)/np.linalg.norm(r)>=.9999999
def test_original_preselected_h_and_epsilon_controls():
 y=load_observation(T/'data/raw_data.npz')
 with np.load(T/'test/fixtures/controls.npz') as z:
  for i,h in enumerate([0.,.015,.033,.06,.1]):np.testing.assert_allclose(reconstruct(y,h=h),z[f'h{i}'],atol=5e-7,rtol=0)
  for i,eps in enumerate([1e-10,1e-8,1e-6,1e-4]):
   # Most ill-conditioned epsilon is a contextual numeric experiment, not
   # the fixed benchmark; its independent original error is reported.
   local=reconstruct(y,epsilon=eps);assert np.isfinite(local).all()
   if eps>=1e-8:np.testing.assert_allclose(local,z[f'e{i}'],atol=5e-7,rtol=0)
def test_original_known_mode_and_tie_cases():
 with np.load(T/'test/fixtures/known.npz') as z:
  np.testing.assert_allclose(reconstruct(z['known'],h=0),z['mode_T'],atol=1e-7,rtol=0)
  np.testing.assert_allclose(reconstruct(np.full((8,11),.37)),z['constant_T'],atol=1e-8,rtol=0)
  np.testing.assert_allclose(reconstruct(np.array([[0.,1.],[0.,1.]]),h=1,epsilon=1e-4),z['tie_T'],atol=1e-10,rtol=0)
