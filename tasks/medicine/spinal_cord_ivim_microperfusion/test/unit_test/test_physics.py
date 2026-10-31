import numpy as np
from src.physics_model import signal,jacobian
from src.solvers import from_internal,to_internal

def test_physics_and_derivative():
    b=np.array([0.,5.,50.,800.]);p=np.array([100.,.001,.1,.02]);J=jacobian(b,*p)
    assert signal(b,*p)[0]==100
    assert np.all(np.diff(signal(b,*p))<0)
    for j,eps in enumerate([1e-3,1e-8,1e-6,1e-7]):
        plus=p.copy();minus=p.copy();plus[j]+=eps;minus[j]-=eps
        np.testing.assert_allclose((signal(b,*plus)-signal(b,*minus))/(2*eps),J[:,j],rtol=1e-6,atol=1e-6)
    np.testing.assert_array_equal(signal(b,100,.001,0,.02),signal(b,100,.001,0,.08))
    np.testing.assert_allclose(signal(b,100,.001,.8,0),100*np.exp(-b*.001),rtol=1e-15)

def test_box_transform():
    lower=np.array([0.,.1]);upper=np.array([2.,.2]);p=np.array([1.,.12])
    np.testing.assert_allclose(from_internal(to_internal(p,lower,upper),lower,upper),p,rtol=1e-15)

def test_public_constants():
    from src.preprocessing import load_data
    from pathlib import Path
    root=Path(__file__).resolve().parents[2];im,mask,b,affine=load_data(root/'data/raw_data.npz')
    np.testing.assert_array_equal(b,[5,10,20,30,50,75,150,250,600,700,800]);assert mask.sum()==2523
    text=(root/'README.md').read_text(encoding='utf-8');assert '128' in text and '134' in text and '2523' in text
