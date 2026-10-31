import importlib.util
from pathlib import Path
import numpy as np
import pytest
from src import physics_model as pm
from src.preprocessing import load_observations,estimate_noise
from src.solvers import fit_spectrum
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('independent_judge',ROOT/'evaluation/judge.py')
judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

def test_zero_opacity_and_thermal_equilibrium():
    frequency=np.linspace(93170e6,93178e6,4001)
    assert np.array_equal(pm.brightness(frequency,[8,0,3,.3]),np.zeros(4001))
    assert np.array_equal(pm.brightness(frequency,[2.73,2,3,.3]),np.zeros(4001))

def test_opacity_linear_and_saturation_physical_limits():
    f=np.linspace(93170e6,93178e6,4001)
    p=[8,1,3,.3];a=pm.optical_depth(f,p)
    np.testing.assert_allclose(pm.optical_depth(f,[8,2,3,.3]),2*a)
    thermal=pm.H_OVER_K*f*(1/np.expm1(pm.H_OVER_K*f/8)-1/np.expm1(pm.H_OVER_K*f/2.73))
    value=pm.brightness(f,[8,1e5,3,.3])
    assert (value>=0).all();assert (value<=thermal+1e-12).all()
    assert pm.brightness(f,[8,1e-5,3,.3]).max()<pm.brightness(f,p).max()

def test_component_integrated_opacity_and_redshift():
    # Independent Gaussian integral and Doppler sign, including upstream width convention.
    f=np.linspace(93165e6,93185e6,100001);p=[8,2,3,.3]
    measured=np.trapezoid(pm.optical_depth(f,p),f)
    widths=.3/pm.C_KMS*pm.FREQUENCIES*(1-pm.OFFSETS/pm.C_KMS)
    expected=2*np.sum(pm.STRENGTHS/pm.STRENGTHS.sum()*np.sqrt(2*np.pi)*widths)
    assert measured==pytest.approx(expected,rel=1e-10)
    assert np.argmax(pm.optical_depth(f,[8,2,4,.3]))<np.argmax(pm.optical_depth(f,[8,2,3,.3]))

def test_nonphysical_width_rejected():
    with pytest.raises(ValueError):pm.brightness(np.array([93173e6]),[8,2,3,0])
    with pytest.raises(ValueError):pm.brightness(np.array([93173e6]),[0,2,3,.2])

def test_public_constants_and_metadata_agree():
    with np.load(ROOT/'data/raw_data.npz',allow_pickle=False) as d:
        np.testing.assert_array_equal(d['hyperfine_velocity_offsets'][0],pm.OFFSETS)
        np.testing.assert_array_equal(d['hyperfine_relative_strengths'][0],pm.STRENGTHS)
        v=(np.arange(501)+1-403.0960083008)*(-.06284131109715)+2.5
        np.testing.assert_allclose(d['frequency'][0],93176265000*(1-v/299792.458),rtol=0,atol=2e-5)
    readme=(ROOT/'README.md').read_text()
    for constant in ['93176265000','93173763700','299792.458','2.73','20 through 28','0.06284131109715']:
        assert constant in readme

def test_noise_population_normalization():
    cube,f=load_observations(ROOT/'data/raw_data.npz')
    noise=estimate_noise(cube,f)
    v=pm.C_KMS*(1-f/pm.REST_FREQUENCY);m=(v>=20)&(v<=28)
    manual=np.sqrt(((cube[m]-cube[m].mean(axis=0))**2).mean(axis=0))
    np.testing.assert_allclose(noise,manual,rtol=1e-14)
    with pytest.raises(ValueError):estimate_noise(cube[-2:],f[-2:])

def test_fit_independent_noiseless_case_and_jacobian_rank():
    f=np.linspace(93170e6,93178e6,1201);truth=np.array([9,2.3,3.1,.22])
    result,chi=fit_spectrum(f,pm.brightness(f,truth),.2)
    np.testing.assert_allclose(result,truth,rtol=2e-7);assert chi<1e-12
    jac=[]
    for i in range(4):
        step=1e-5*truth[i];a=truth.copy();b=truth.copy();a[i]+=step;b[i]-=step
        jac.append((pm.brightness(f,a)-pm.brightness(f,b))/(2*step))
    assert np.linalg.matrix_rank(np.array(jac))==4

def test_metric_known_case():
    assert judge.relative_max(np.array([1.1,3.6]),np.array([1.,4.]))==pytest.approx(.1)
    with pytest.raises(ValueError):judge.relative_max(np.array([0.]),np.array([0.]))

