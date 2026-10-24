from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose
from src.preprocessing import load_data,observation_weights
from src.physics_model import regularizer,weighted_system,forward,adjoint,objective_gradient
from src.solvers import reconstruct
ROOT=Path(__file__).resolve().parents[2]
def data():return load_data(ROOT/'data/raw_data.npz')
def test_independent_original_stages():
    d=data();A,b,R,w=weighted_system(d)
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False) as f:
        for a,k in [(w,'output_weights'),(R,'output_regularizer'),(A,'output_weighted_operator'),(b,'output_weighted_data')]:assert_allclose(a,f[k],rtol=0,atol=0)
def test_independent_original_full_pipeline():
    m,p,r=reconstruct(data())
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False) as f:
        assert_allclose(m.ravel(),f['output_reconstruction'],rtol=1e-7,atol=1e-9)
        assert_allclose(p,f['output_prediction'],rtol=1e-9,atol=1e-7)
def test_adjoint_and_linearity():
    d=data();G=d['green_matrix'];x=np.sin(np.arange(289));y=np.cos(np.arange(204));assert_allclose(forward(2*x,G),2*forward(x,G),rtol=1e-14);assert_allclose(forward(x,G)@y,x@adjoint(y,G),rtol=1e-13)
def test_graph_physics():
    d=data();R=regularizer(d['source_coordinates']);assert R.shape==(528,289);assert_allclose(R@np.ones(289),0);eig=np.linalg.eigvalsh(R.T@R);assert abs(eig[0])<1e-12 and eig[1]>0
    for row in R:
        ix=np.flatnonzero(row);assert len(ix)==2;assert_allclose(np.linalg.norm(d['source_coordinates'][ix[0]]-d['source_coordinates'][ix[1]]),.03,atol=1e-10)
def test_gradient_and_kkt():
    d=data();A,b,R,w=weighted_system(d);x=np.full(289,.003);cost,g=objective_gradient(x,A,b);v=np.sin(np.arange(289));eps=1e-7;fd=(objective_gradient(x+eps*v,A,b)[0]-objective_gradient(x-eps*v,A,b)[0])/(2*eps);assert_allclose(fd,g@v,rtol=1e-8)
    m,p,r=reconstruct(d);assert 0<=m.min() and m.max()<=1;assert abs(m.sum()-1)<.002;assert r.optimality<1e-5
def test_readme_and_weight_cap():
    text=(ROOT/'README.md').read_text(encoding='utf8');assert '0.030' in text;assert 'simulated' in text
    for key in data():assert '| '+key+' |' in text
    assert_allclose(observation_weights(np.array([0.,.01,4.,-16.])),[1,1,.5,.25])
