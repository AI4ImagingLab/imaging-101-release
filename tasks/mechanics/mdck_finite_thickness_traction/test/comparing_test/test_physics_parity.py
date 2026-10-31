from pathlib import Path
import numpy as np
from src.preprocessing import load_data,prepare_displacement
from src.physics_model import stiffness,apply_spectral
from src.solvers import reconstruct
ROOT=Path(__file__).resolve().parents[2]

def test_frozen_upstream_full_benchmark():
    with np.load(ROOT/'test/fixtures/upstream.npz',allow_pickle=False) as f:
        u=load_data(ROOT/'data/raw_data.npz')
        np.testing.assert_array_equal(u,np.stack([f['input_u'],f['input_v']]))
        for mode,key in [('finite','output_traction'),('infinite','output_infinite')]:
            np.testing.assert_allclose(reconstruct(u,mode=mode),f[key],rtol=1e-10,atol=1e-8)
        np.testing.assert_allclose(reconstruct(u,smoothing_um=0),f['output_unfiltered'],rtol=1e-10,atol=1e-8)

def test_elastic_spectral_eigenvalues_positive_work_and_roundtrip():
    kxx,kxy,kyy=stiffness(50,8.040000000000001,300.,49000.,.49)
    assert np.all(kxx>=0) and np.all(kyy>=0)
    assert np.min(kxx*kyy-kxy*kxy)>=-1e-8
    rng=np.random.default_rng(6);u=rng.normal(size=(2,50,50));u-=u.mean(axis=(1,2),keepdims=True)
    t=apply_spectral(u,(kxx,kxy,kyy))
    assert np.sum(u*t)>0
    det=kxx*kyy-kxy*kxy;det[0,0]=1
    inv=(kyy/det,-kxy/det,kxx/det)
    recovered=apply_spectral(t,inv)
    np.testing.assert_allclose(recovered,u,atol=1e-12)

def test_infinite_limit_and_rigid_translation():
    finite=stiffness(50,8.,3000.,49000.,.49)
    infinite=stiffness(50,8.,None,49000.,.49)
    # Upstream half-space explicitly removes cross-coupling at the
    # double-Nyquist mode; finite-layer real-IFFT retains that real mode.
    mask=np.ones((50,50),bool);mask[25,25]=False
    for a,b in zip(finite,infinite):np.testing.assert_allclose(a[mask],b[mask],rtol=1e-12,atol=1e-10)
    u=load_data(ROOT/'data/raw_data.npz')
    np.testing.assert_allclose(reconstruct(u+np.array([11.,-9.])[:,None,None]),reconstruct(u),atol=1e-10)

def test_constants_and_component_linearity():
    u=load_data(ROOT/'data/raw_data.npz')
    assert u.shape==(2,48,49) and u.dtype==np.float64
    np.testing.assert_allclose(reconstruct(2*u),2*reconstruct(u),rtol=1e-13,atol=1e-10)
    assert np.allclose(reconstruct(np.zeros_like(u)),0)
    from src.preprocessing import GRID_PITCH_UM,PIXEL_SIZE_UM,YOUNG_PA,POISSON,THICKNESS_UM
    assert GRID_PITCH_UM==(100-60)*PIXEL_SIZE_UM
    text=(ROOT/'README.md').read_text(encoding='utf8')
    for v in [str(GRID_PITCH_UM),str(PIXEL_SIZE_UM),str(YOUNG_PA),str(POISSON),str(THICKNESS_UM)]:assert v in text

def test_preparation_and_analytic_longitudinal_transverse_response():
    from src.preprocessing import PIXEL_SIZE_UM
    input_field=np.arange(24.,dtype=float).reshape(2,3,4)
    out=prepare_displacement(input_field)
    assert out.shape==(2,4,4)
    np.testing.assert_array_equal(out[:,0],0)
    np.testing.assert_allclose(out[:,-3:,:],(input_field-input_field.mean(axis=(1,2),keepdims=True))*PIXEL_SIZE_UM)
    n=50;pitch=8.;e=49000.;nu=.49;h=300.
    tensor=stiffness(n,pitch,h,e,nu)
    k=2*np.pi/(n*pitch);r=k*h
    gamma=((3-4*nu)+(1-2*nu)**2/np.cosh(r)**2+r*r/np.cosh(r)**2)/((3-4*nu)*np.tanh(r)+r/np.cosh(r)**2)
    wave=np.cos(2*np.pi*np.arange(n)/n)[None,:]*np.ones((n,1))
    for component,coefficient in [(0,e*k*gamma/(2*(1-nu**2))),(1,e*k/(2*(1+nu)*np.tanh(r)))]:
        u=np.zeros((2,n,n));u[component]=wave
        expected=u*coefficient
        np.testing.assert_allclose(apply_spectral(u,tensor),expected,atol=1e-11)
