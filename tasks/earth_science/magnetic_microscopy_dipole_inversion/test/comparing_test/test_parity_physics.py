from pathlib import Path
import numpy as np
from src.preprocessing import load_data,prepare,detect_windows,upward
from src.solvers import reconstruct,euler,invert_moment
from src.physics_model import dipole_jacobian,forward

ROOT=Path(__file__).resolve().parents[2]

def test_independent_upstream_stages_and_pipeline():
    with np.load(ROOT/'test/fixtures/upstream.npz',allow_pickle=False) as f:
        data,x,y=load_data(ROOT/'data/raw_data.npz')
        np.testing.assert_array_equal(data,f['input_bz'])
        result=reconstruct(data,x,y)
        for a,k in zip(result[6],('up','dx','dy','dz','tga','stretched')):
            np.testing.assert_allclose(a,f['output_'+k],rtol=1e-9,atol=5e-9)
        np.testing.assert_array_equal(result[7],f['output_windows'])
        for a,k in zip(result[:6],('positions','moments','bases','scores','valid_windows','covariance')):
            # Scale-aware comparison: moments are ~1e-12, covariances ~1e-25.
            scale=np.max(np.abs(f['output_'+k]))
            assert np.max(np.abs(a-f['output_'+k])) <= 1e-9*scale

def test_fourier_harmonic_physics_and_zero_mode():
    n=64;s=2.35;h=5.;x=np.arange(n)*s
    f=np.cos(2*np.pi*3*x/(n*s))[None,:]*np.ones((n,1))
    np.testing.assert_allclose(upward(f,s,h),f*np.exp(-2*np.pi*3/(n*s)*h),atol=1e-14)
    np.testing.assert_allclose(upward(np.ones((n,n)),s,h),1,atol=1e-15)

def test_dipole_axis_sign_units_and_superposition():
    x=np.zeros((1,1));y=x.copy()
    # On axis at 10 um: Bz=2e-7*m/r^3 T, horizontal moments give zero.
    a=dipole_jacobian(x,y,10.,[0,0,0])
    np.testing.assert_allclose(a,[[0,0,2e8]],rtol=1e-9)
    b=forward(x,y,10.,[[0,0,0]],[[0,0,1e-12]])
    np.testing.assert_allclose(b,2e5,rtol=1e-9)
    np.testing.assert_allclose(forward(x,y,10.,[[0,0,0]]*2,[[0,0,1e-12]]*2),2*b)

def test_euler_analytic_homogeneity_and_moment_recovery():
    x=np.linspace(-30,30,31);y=np.linspace(-30,30,31);xx,yy=np.meshgrid(x,y)
    p=np.array([1.,-2.,-10.]);m=np.array([1e-13,2e-13,-3e-13]);z=10.
    f=forward(xx,yy,z,[p],[m]);eps=1e-3
    dx=(forward(xx+eps,yy,z,[p],[m])-forward(xx-eps,yy,z,[p],[m]))/(2*eps)
    dy=(forward(xx,yy+eps,z,[p],[m])-forward(xx,yy-eps,z,[p],[m]))/(2*eps)
    dz=(forward(xx,yy,z+eps,[p],[m])-forward(xx,yy,z-eps,[p],[m]))/(2*eps)
    pos,base=euler(f+17,dx,dy,dz,x,y,z)
    np.testing.assert_allclose(pos,p,atol=2e-7);assert abs(base-17)<1e-5
    estimated,*_=invert_moment(f,x,y,p,z)
    np.testing.assert_allclose(estimated,m,rtol=1e-14,atol=0)

def test_data_constants_and_readme():
    data,x,y=load_data(ROOT/'data/raw_data.npz')
    assert data.dtype==np.float64 and data.shape==(600,960)
    np.testing.assert_array_equal(x,np.arange(960)*2.35)
    np.testing.assert_array_equal(y,np.arange(600)*2.35)
    text=(ROOT/'README.md').read_text(encoding='utf-8')
    for value in ['2.35','600','960','5.0','float64','75']:assert value in text
