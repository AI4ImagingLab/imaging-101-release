from pathlib import Path
import numpy as np
import pytest
from src.preprocessing import load_observations
from src.physics_model import initial_guess
from src.solvers import fit,reconstruct
T=Path(__file__).resolve().parents[2]
@pytest.mark.parametrize('steps',[1,2,3,10,1000,10000])
def test_frozen_upstream_updates(steps):
 with np.load(T/'test/fixtures/upstream_stages.npz',allow_pickle=False) as z:
  for pol in range(2):
   for window in range(2):
    guess=initial_guess(z['spectra'][pol,window],z['frequency_ghz'][window])
    np.testing.assert_array_equal(guess,z['initial_parameters'][pol,window])
    p,_,_=fit(z['frequency_ghz'][window],z['spectra'][pol,window],guess,steps)
    err=np.max(abs(p-z[f'parameters_{steps}'][pol,window]),axis=0)
    assert err[0]<=1e-6 and err[1]<=1e-6 and np.max(err[2:])<=1e-4
def test_complete_fresh_pipeline():
 fresh=reconstruct(load_observations(T/'data/raw_data.npz'))
 assert np.isfinite(fresh['reconstruction']).all()
 with np.load(T/'evaluation/baseline_reference.npz',allow_pickle=False) as z:diff=fresh['reconstruction']-z['reconstruction']
 rmse=np.sqrt(np.mean(diff**2,axis=(-2,-1)))[0];mx=np.max(abs(diff),axis=(-2,-1))[0]
 assert np.all(rmse[:2]<=.02) and np.all(mx[:2]<=.2)
 assert np.all(rmse[2:]<=2e-7) and np.all(mx[2:]<=1e-6)
