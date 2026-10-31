from pathlib import Path
import importlib.util
import numpy as np
import pytest
from src.physics_model import epg_signal,rf_matrix
from src.solvers import active_set_nnls,objective_and_gap
ROOT=Path(__file__).resolve().parents[2]

def test_independent_bloch_all_cases():
    z=np.load(ROOT/'test/fixtures/independent_bloch.npz')
    out=np.array([epg_signal(t1,[t2],kap,False)[0] for t1,t2,kap in zip(z['input_t1_ms'],z['input_t2_ms'],z['input_kappa'])])
    np.testing.assert_allclose(out,z['output_independent_bloch'],atol=2e-12,rtol=2e-11)
    np.testing.assert_allclose(out,z['output_corrected_source_v2'],atol=2e-12,rtol=2e-11)
    assert np.max(abs(abs(z['output_original_v1'])-abs(z['output_independent_bloch'])))>.09

def test_phase_independent_of_amplitude_and_ideal_decay():
    t2=80.;t1=1000.;s=epg_signal(t1,[t2],1.)[0]
    np.testing.assert_allclose(s,(1-np.exp(-1))*np.exp(-np.arange(1,33)*9.2/t2),atol=2e-14)
    np.testing.assert_array_equal(epg_signal(t1,[t2],0.),np.zeros((1,32)))

@pytest.mark.parametrize('x',[np.array([0.,0.]),np.array([4/3,0.]),np.array([.5,2.]),np.array([2.,.1])])
def test_stable_duality_equals_definition(x):
    A=np.diag([2.,1.]);y=np.array([3.,-1.]);lam=.5
    # Repeat two echo columns only for generic-certificate routine.
    J,gap=objective_and_gap(A[None],x[None],y[None],np.array([0]),lam)
    u=A@x-y;g=A.T@u;D=-u@u-2*y@u-np.minimum(g,0)@np.minimum(g,0)/lam
    assert gap[0]>=0
    np.testing.assert_allclose(gap[0],J[0]-D,atol=1e-13)
    assert D<=2.+1e-13
