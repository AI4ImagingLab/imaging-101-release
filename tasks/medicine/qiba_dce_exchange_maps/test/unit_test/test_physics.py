from fractions import Fraction
from pathlib import Path
import numpy as np
import pytest
from src.physics_model import conv,fma_square,julia_square_sum
from src.preprocessing import signal_to_concentration,load_observations
from src.generate_data import perturb_signal

ROOT=Path(__file__).resolve().parents[2]

def test_analytic_exponential_integrals():
    t=np.linspace(0,2,41);b=.4
    np.testing.assert_allclose(conv(np.ones(41),np.array([b]),t)[0],-np.expm1(-b*t)/b,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(conv(t,np.array([b]),t)[0],t/b+np.expm1(-b*t)/b**2,rtol=1e-12,atol=1e-12)
    assert conv(t,np.array([b]),t)[0,0]==0

def test_nonuniform_time_and_time_units():
    seconds=np.array([0.,1.,4.,9.,20.,40.]);minutes=seconds/60
    # Rates expressed per minute and per second give the same concentrations
    # when transfer and washout are both transformed consistently.
    cp=np.ones(6)
    a=.2*conv(cp,np.array([.6]),minutes)
    b=(.2/60)*conv(cp,np.array([.6/60]),seconds)
    np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-14)

def test_known_spgr_roundtrip():
    r0=1.;tr=.005;angle=np.pi/6
    rates=np.array([[1.,1.,1.1,1.5,2.0]])
    signal=50000*np.sin(angle)*(1-np.exp(-tr*rates))/(1-np.exp(-tr*rates)*np.cos(angle))
    r,ct=signal_to_concentration(signal,np.array([r0]))
    np.testing.assert_allclose(r,rates,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(ct,(rates-r0)/4.5,rtol=1e-10,atol=1e-12)

def test_fma_halfway_uses_exact_integer_fallback():
    eps=np.finfo(float).eps;a=np.array([1+eps]);c=np.array([1-eps])
    exact=float(Fraction(float(a[0]))**2+Fraction(float(c[0])))
    assert (a*a+c)[0]!=exact
    assert fma_square(a,c)[0]==exact

def test_fma_against_exact_rational_cases():
    rng=np.random.RandomState(73)
    a=np.ldexp(rng.uniform(.5,1,300),rng.randint(-400,400,300))
    c=np.ldexp(rng.uniform(.5,1,300),rng.randint(-400,400,300))
    expected=np.array([float(Fraction(float(x))**2+Fraction(float(z))) for x,z in zip(a,c)])
    np.testing.assert_array_equal(fma_square(a,c),expected)

def test_square_sum_known_integer_energy():
    a=np.arange(1321,dtype=float)[None]
    assert julia_square_sum(a)[0]==1320*1321*2641/6

def test_perturbation_statistics_and_calibration_independence():
    original=load_observations(ROOT/'data/raw_data.npz')
    changed=perturb_signal(original,.03,20261022)
    scale=np.mean(original['signal'][:,:2],axis=1,keepdims=True)*.03
    noise=(changed['signal']-original['signal'])/scale
    assert abs(noise.mean())<.003 and abs(noise.std()-1)<.003
    for key in ['relaxation_rate','time_seconds','plasma_concentration']:
        np.testing.assert_array_equal(changed[key],original[key])
    assert not np.shares_memory(changed['signal'],original['signal'])
    np.testing.assert_array_equal(perturb_signal(original,0,4)['signal'],original['signal'])

@pytest.mark.parametrize('sigma',[-1.,float('nan'),float('inf')])
def test_invalid_perturbation_sigma(sigma):
    with pytest.raises(ValueError):perturb_signal({'signal':np.ones((1,2,1,1))},sigma,4)
