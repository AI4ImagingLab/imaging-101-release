
from pathlib import Path
import numpy as np
from src.preprocessing import load_data
from src.solvers import reconstruct
ROOT=Path(__file__).resolve().parents[2]
def test_full_upstream_parity():
    result=reconstruct(load_data(ROOT/'data/raw_data.npz'))
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as f:
        for k in f.files:
            assert result[k].shape==f[k].shape
            assert np.linalg.norm(result[k]-f[k])/np.linalg.norm(f[k])<1e-5,k

def test_frozen_preprocessing_and_kernel():
    from src.preprocessing import phase_echo
    from src.physics_model import decay_kernel,RELAXATION_TIME
    data=load_data(ROOT/'data/raw_data.npz');f=np.load(ROOT/'test/fixtures/upstream_stages.npz')
    np.testing.assert_allclose(phase_echo(data['echo']),f['output_phased'],rtol=1e-13,atol=1e-13);np.testing.assert_array_equal(RELAXATION_TIME,f['param_relaxation_time']);np.testing.assert_allclose(decay_kernel(data['echo_time'][1:]),f['output_kernel'][:255],rtol=1e-13)
    from scipy.optimize import nnls
    k=decay_kernel(data['echo_time'][1:]);average=nnls(np.vstack([k,.01*np.eye(255)]),np.r_[phase_echo(data['echo']).mean(axis=0),np.zeros(255)])[0]
    np.testing.assert_allclose(average,f['output_average_distribution'],rtol=1e-12,atol=1e-12);np.testing.assert_allclose(k@average,f['output_average_echo'],rtol=1e-12,atol=1e-12)
def test_analytic_decay_gradient_and_nonnegativity():
    from src.physics_model import decay_kernel,objective
    k=decay_kernel(np.array([0.,1.,2.]),np.array([1.]));np.testing.assert_allclose(k[:,0],[1,np.exp(-1),np.exp(-2)])
    x=np.array([2.]);y=np.array([1.,.5,.1]);eps=1e-6;fd=(objective(x+eps,y,k)-objective(x-eps,y,k))/(2*eps);np.testing.assert_allclose(fd,2*k.T@(k@x-y)+.0002*x,rtol=1e-8)
    z=reconstruct(load_data(ROOT/'data/raw_data.npz'));assert z['reconstruction'].min()>=0;assert (np.diff(z['fitted_echo'][0],axis=1)<=1e-12).all()
