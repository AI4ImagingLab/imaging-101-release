
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

def test_frozen_operator_and_quadratic_penalty():
    from src.physics_model import operators
    d=load_data(ROOT/'data/raw_data.npz');a,l,t=operators(d['frequency']);f=np.load(ROOT/'test/fixtures/upstream_stages.npz')
    np.testing.assert_allclose(a.real,f['output_A_re'],rtol=1e-13,atol=1e-14);np.testing.assert_allclose(a.imag,f['output_A_im'],rtol=1e-13,atol=1e-14);np.testing.assert_allclose(l.T@l,f['output_M'],rtol=1e-12,atol=1e-12)
    result=reconstruct(d);assert np.linalg.norm(result['reconstruction'][0]-f['output_gamma'])/np.linalg.norm(f['output_gamma'])<1e-5;np.testing.assert_allclose(result['fitted_impedance'][0],f['output_fit'],rtol=1e-7,atol=1e-10)
def test_physics_limits_adjoint_and_regularizer():
    from src.physics_model import operators,forward,objective
    f=np.array([1e5,1.,1e-5]);a,l,t=operators(f);x=np.array([0.,3.,0.,0.,0.]);np.testing.assert_allclose(forward(x,f),3.)
    np.testing.assert_allclose(l@np.ones(len(x)),0,atol=1e-12)
    x=np.arange(len(x),dtype=float);z=np.array([1+2j,3-1j,2j]);np.testing.assert_allclose(np.vdot(a@x,z),np.vdot(x,a.conj().T@z))
    direction=np.linspace(-1,1,len(x));eps=1e-6;y=np.ones(3,dtype=complex);fd=(objective(x+eps*direction,f,y)-objective(x-eps*direction,f,y))/(2*eps);analytic=2*np.real(np.vdot(a@direction,a@x-y))+.002*np.dot(l@direction,l@x);np.testing.assert_allclose(fd,analytic,rtol=1e-8)
