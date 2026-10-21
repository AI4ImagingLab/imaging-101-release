import importlib.util
from pathlib import Path
import numpy as np
from src.physics_model import parallel_ray,forward,adjoint
from src.solvers import reconstruct
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)
def test_axis_aligned_path_lengths():
    a=parallel_ray(8,1.,np.array([0.,90.]),8,1.).tocsr()
    np.testing.assert_allclose(a.dot(np.ones(64)),8,rtol=0,atol=1e-6)
def test_independent_projector():
    angles=np.array([-75.,-31.,1.,27.,75.]);a=parallel_ray(16,1.,angles,16,1.).tocsr();b=judge.ray_matrix(16,angles)
    np.testing.assert_allclose(a.toarray(),b.toarray(),rtol=2e-6,atol=2e-6)
def test_adjoint_identity_and_gradient():
    rng=np.random.default_rng(193);a=parallel_ray(8,1.,np.array([-63.,1.,37.]),8,1.).tocsr();x=rng.normal(size=(2,8,8));y=rng.normal(size=(2,8,3))
    np.testing.assert_allclose(np.vdot(forward(x,a),y),np.vdot(x,adjoint(y,a)),rtol=1e-12,atol=1e-12)
    f=lambda z:.5*np.sum((forward(z,a)-y)**2);direction=rng.normal(size=x.shape);eps=1e-5
    np.testing.assert_allclose((f(x+eps*direction)-f(x-eps*direction))/(2*eps),np.vdot(adjoint(forward(x,a)-y,a),direction),rtol=1e-8)
def test_positivity_and_descent():
    angles=np.array([-73.,-41.,-9.,23.,55.,75.]);a=parallel_ray(8,1.,angles,8,1.);x=np.zeros((1,8,8));x[:,2:6,3:5]=1;d=forward(x,a)
    v,h=reconstruct(d,angles,iterations=100,step=.005,record_every=10)
    assert np.min(v)>=0 and h[-1,1]<h[0,1] and h[-1,1]<.2

def test_known_metrics():
    r=np.array([0.,1.,2.]);m=judge.compare(r,r);assert abs(m['ncc']-1)<1e-14 and m['nrmse']==0
    m=judge.compare(r*2,r);np.testing.assert_allclose(m['nrmse'],np.sqrt(5/3)/2);assert abs(m['ncc']-1)<1e-14

def test_bound_acquisition_constants_and_input_contract(tmp_path):
    import pytest
    from src.preprocessing import load_data,PIXEL_NM,NITER,STEP
    data,angles=load_data(ROOT/'data/raw_data.npz')
    expected=np.delete(np.arange(-75,76,2,dtype=float),[49,50])
    np.testing.assert_array_equal(angles,expected)
    assert PIXEL_NM==6.39 and NITER==100 and STEP==.0001
    text=(ROOT/'README.md').read_text(encoding='utf8')
    assert '| Nominal voxel sampling | 6.39 |' in text
    assert not any(x in text.lower() for x in ['github.com','github.io','githubusercontent.com','codeload.github.com','git.io'])
    with np.load(ROOT/'data/raw_data.npz',allow_pickle=False) as z:assert set(z.files)=={'measurements','angles'}
    np.savez(tmp_path/'bad.npz',measurements=np.zeros((1,74,128,128),dtype='float32'),angles=angles[None])
    with pytest.raises(ValueError,match='float64'):load_data(tmp_path/'bad.npz')
