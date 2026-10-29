from pathlib import Path
import numpy as np
from src.preprocessing import load_data,prepare,NM_PER_PIXEL
from src.physics_model import lockin,adaptive_lockin,weighted_projection
from src.solvers import reconstruct,integrate_gradient
ROOT=Path(__file__).resolve().parents[2]

def test_frozen_original_stages():
    f=np.load(ROOT/'test/fixtures/upstream.npz');d=load_data(ROOT/'data/raw_data.npz')
    np.testing.assert_array_equal(d['image'],f['input_image']);np.testing.assert_array_equal(d['reference_vectors'],f['input_kvecs'])
    ks=d['reference_vectors'];sigma=int(np.ceil(1/np.linalg.norm(ks,axis=1).min()));kw=np.linalg.norm(ks,axis=1).mean()/2.5
    value,selected=adaptive_lockin(prepare(d['image']),ks[0],sigma,kw,kw/3)
    np.testing.assert_allclose(value,f['output_lockin'][0],rtol=1e-12,atol=1e-10)
    np.testing.assert_array_equal(selected,f['output_selected'][0])
    np.testing.assert_allclose(reconstruct(d),f['output_u']*.88,rtol=1e-10,atol=1e-10)

def test_constant_wave_lockin():
    n=64;y,x=np.mgrid[:n,:n];a=np.cos(2*np.pi*(y/8+x/8))
    q=lockin(a,(1/8,1/8),5)
    np.testing.assert_allclose(q,.5,atol=1e-12)

def test_independent_weighted_projection():
    ks=np.array([[.1,0],[0,.1],[.1,.1]]);v=np.array([.07,-.03]);b=np.einsum('jc,c->j',2*np.pi*ks,v)[:,None,None]*np.ones((3,4,5));w=np.ones((3,4,5));w[0]*=3
    got=weighted_projection(b,ks,w);np.testing.assert_allclose(got,np.broadcast_to(v[:,None,None],got.shape),atol=1e-15)

def test_neumann_integration_and_gauge():
    n=32;y,x=np.mgrid[:n,:n];truth=.03*x+.02*y;truth-=truth.mean()
    got=integrate_gradient(np.diff(truth,axis=1),np.diff(truth,axis=0),np.ones_like(truth))
    np.testing.assert_allclose(got,truth,atol=1e-12)
    assert abs(got.mean())<1e-13
    assert np.all(integrate_gradient(np.zeros((n,n-1)),np.zeros((n-1,n)),np.ones((n,n)))==0)
    assert NM_PER_PIXEL==.88 and '0.88' in (ROOT/'README.md').read_text(encoding='utf8')


def test_cartesian_rewrap_is_confined_to_negligible_edges():
    f=np.load(ROOT/'test/fixtures/upstream.npz');signals=f['output_lockin'];phase=np.angle(signals);ks=f['input_kvecs']
    from src.physics_model import wrap_phase
    from src.solvers import divergence
    mask=np.zeros(phase.shape[1:]);mask[66:-66,66:-66]=1
    w=np.abs(signals)*(mask+1e-6);weight=np.linalg.norm(w,axis=0)
    gx=weighted_projection(wrap_phase(np.diff(phase,axis=2)),ks,w);gy=weighted_projection(wrap_phase(np.diff(phase,axis=1)),ks,w)
    assert abs(gx[:,66:-66,66:-66]).max()<np.pi and abs(gy[:,66:-66,66:-66]).max()<np.pi
    ww=weight**2;wx=np.minimum(ww[:,:-1],ww[:,1:]);wy=np.minimum(ww[:-1],ww[1:])
    for i in range(2):
        rhs=divergence(wx*gx[i],wy*gy[i]);alter=divergence(wx*(wrap_phase(gx[i])-gx[i]),wy*(wrap_phase(gy[i])-gy[i]))
        assert np.linalg.norm(alter)/np.linalg.norm(rhs)<1e-10
