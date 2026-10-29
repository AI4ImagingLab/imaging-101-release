from pathlib import Path
import numpy as np
from src.main import reconstruct
from src.physics_model import forward,residual,bounded_weights,affine_jacobian
from src.solvers import MODELS
ROOT=Path(__file__).resolve().parents[2]
def test_all_native_stages_and_eleven_fits():
    e,y,s,r=reconstruct(ROOT/'data/raw_data.npz')
    with np.load(ROOT/'test/fixtures/native.npz') as q:
        np.testing.assert_array_equal(e,q['energy']);np.testing.assert_array_equal(y,q['observation'])
        np.testing.assert_allclose(s,q['standards'],atol=1e-13,rtol=0)
        np.testing.assert_allclose(r['weights'],q['weights'],atol=1e-5,rtol=0)
        np.testing.assert_allclose(r['fits'],q['fits'],atol=1e-6,rtol=0)
        np.testing.assert_allclose(r['statistics'][:,:4],q['statistics'][:,:4],atol=1e-3,rtol=0)
        for m,initial in zip(MODELS,q['initials']):
            u=np.arcsin(2*initial[list(m[:-1])]-1)
            np.testing.assert_allclose(bounded_weights(u,m),initial,atol=1e-14)
            np.testing.assert_allclose(residual(u,m,y,s),(y-initial@s)/.001,atol=1e-10)
    assert r['best']==7
def test_selected_affine_solution_independent():
    e,y,s,r=reconstruct(ROOT/'data/raw_data.npz');m=MODELS[7]
    free=np.linalg.lstsq((s[list(m[:-1])]-s[m[-1]]).T,y-s[m[-1]],rcond=None)[0]
    expected=np.zeros(6);expected[list(m[:-1])]=free;expected[m[-1]]=1-free.sum()
    np.testing.assert_allclose(r['weights'][7],expected,atol=2e-7)
    assert np.min(expected)>=0 and abs(sum(expected)-1)<1e-14
    assert np.sqrt(np.mean((r['fits'][7]-y)**2))<.0037
def test_linearity_and_physical_jacobian():
    _,y,s,_=reconstruct(ROOT/'data/raw_data.npz');a=np.arange(6)/20;b=np.arange(6)[::-1]/30
    np.testing.assert_allclose(forward(a+b,s),forward(a,s)+forward(b,s),atol=1e-15)
    m=MODELS[7];w=np.array([.3,.4,.1,0,0,.2]);h=1e-5
    for j,ix in enumerate(m[:-1]):
        wp=w.copy();wm=w.copy();wp[ix]+=h;wp[m[-1]]-=h;wm[ix]-=h;wm[m[-1]]+=h
        numerical=-(forward(wp,s)-forward(wm,s))/(2*h*.001)
        np.testing.assert_allclose(numerical,affine_jacobian(m,s)[:,j],atol=1e-7)
