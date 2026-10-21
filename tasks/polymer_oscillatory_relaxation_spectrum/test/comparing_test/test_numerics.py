
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

def test_resampling_kernel_and_spectrum_stages():
    from src.preprocessing import prepare
    from src.physics_model import get_kern_mat
    from src.solvers import solve
    d=load_data(ROOT/'data/raw_data.npz');w,g,weights=prepare(d);f=np.load(ROOT/'test/fixtures/upstream_stages.npz')
    np.testing.assert_array_equal(w,f['output_frequency']);np.testing.assert_allclose(g,f['output_moduli'],rtol=1e-14);np.testing.assert_array_equal(weights,f['output_weights']);np.testing.assert_allclose(get_kern_mat(f['param_relaxation_time'],w),f['output_kernel'],rtol=1e-14)
    _,c,_=solve(d);np.testing.assert_allclose(c.H,f['output_log_spectrum'],rtol=1e-10,atol=1e-10)
def test_maxwell_limit_and_analytic_jacobian():
    from src.solvers import _maxwell_kernel,_residualLM,_jacobianLM
    from src.physics_model import get_kern_mat
    np.testing.assert_allclose(_maxwell_kernel(np.array([2.]),np.array([.5]))[:,0],[.5,.5])
    s=np.geomspace(.1,10,6);w=np.geomspace(.01,100,7);k=get_kern_mat(s,w);h=np.linspace(-2,1,6);g=k@np.exp(h)*1.1;wt=np.ones(14);v=np.linspace(1,-1,6);e=1e-6
    fd=(_residualLM(h+e*v,.01,g,wt,k)-_residualLM(h-e*v,.01,g,wt,k))/(2*e);np.testing.assert_allclose(fd,_jacobianLM(h,.01,g,wt,k)@v,rtol=1e-8,atol=1e-9)
