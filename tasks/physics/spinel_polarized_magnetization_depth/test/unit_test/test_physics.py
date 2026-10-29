from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.physics_model import *
from src.preprocessing import load_data
from src.solvers import bounded_simplex

def test_fresnel_lossless_limit():
    q=np.linspace(.005,.2,200);rho=np.array([0.,5.377]);k0=np.sqrt(q*q/4+4j*np.pi*1e-16);ks=np.sqrt(q*q/4-4*np.pi*5.377e-6+4j*np.pi*1e-16)
    expected=abs((k0-ks)/(k0+ks))**2
    actual=layer_reflectivity(q,np.zeros(2),rho,np.zeros(2),np.zeros(1),1)
    np.testing.assert_allclose(actual,expected,rtol=1e-13);assert actual.min()>=0 and actual.max()<=1+1e-12

def test_magnetization_reversal_and_zero_channel_equality():
    q=np.linspace(.006,.2,84);p=np.array([1.,.026,0,5,5,10,.3,6.,140.]);w,r,m,s=layer_stack(p)
    np.testing.assert_array_equal(layer_reflectivity(q,w,r,m,s,1),layer_reflectivity(q,w,r,-m,s,-1))
    np.testing.assert_array_equal(layer_reflectivity(q,w,r,m*0,s,1),layer_reflectivity(q,w,r,m*0,s,-1))

def test_gaussian_normalization_and_line_integral():
    q=np.linspace(.006,.2,84);sigma=np.full(84,.003)
    np.testing.assert_allclose(gaussian_resolution(q,np.ones(84),sigma),1,rtol=1e-14)
    # A linear function integrated by a symmetric interior Gaussian retains its centre.
    actual=gaussian_resolution(q,3*q+2,sigma)
    np.testing.assert_allclose(actual[10:-10],(3*q+2)[10:-10],rtol=1e-14)

def test_profile_area_and_roughness_limits():
    p=np.array([1.,.026,0,5,5,10,.3,6.,140.]);z=np.linspace(-200,400,10001);profile=depth_profiles(p,z)
    np.testing.assert_allclose(np.trapezoid(profile[:,1],z),.3*(140-10),rtol=1e-12)
    assert abs(profile[0,0]-5.377)<1e-12 and abs(profile[-1,0])<1e-12

def test_bounded_optimizer_and_failure():
    import pytest
    result=bounded_simplex(lambda x:np.sum((x-np.array([.2,.7]))**2),np.array([.5,.5]),np.zeros(2),np.ones(2))
    np.testing.assert_allclose(result['parameters'],[.2,.7],atol=1e-6)
    with pytest.raises(RuntimeError):bounded_simplex(lambda x:np.sum(x*x),np.array([.5,.5]),np.zeros(2),np.ones(2),maxiter=2)

def test_complete_measured_arrays_and_constants():
    d=load_data(ROOT/'data/raw_data.npz');assert d['reflectivity'].shape==(2,84) and np.all(d['uncertainty']>0)
    assert SUBSTRATE_SLD==5.377 and d['depth'].shape==(521,)
    np.testing.assert_array_equal(d['wavelength'],np.full((2,84),4.75))

def test_asymmetry_and_uncertainty_known_case():
    from src.visualization import spin_asymmetry,asymmetry_uncertainty
    channels=np.array([[3.,2.],[1.,2.]]);noise=np.ones_like(channels)*.1
    np.testing.assert_allclose(spin_asymmetry(channels),[.5,0.],atol=1e-15)
    np.testing.assert_allclose(asymmetry_uncertainty(channels,noise),[np.sqrt(.1)/8,np.sqrt(.08)/8],atol=1e-15)

def test_readme_constants():
    text=(ROOT/'README.md').read_text(encoding='utf8')
    for value in ['SLD5.377','| Neutron wavelength | 4.75 |','0.029999999329447746','1e-16']:
        assert value in text
