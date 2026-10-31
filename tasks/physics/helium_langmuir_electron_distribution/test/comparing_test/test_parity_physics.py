from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose,assert_array_equal
from src.preprocessing import load_data,PROBE_AREA_M2,unique_characteristic
from src.solvers import reconstruct,fit_temperature
from src.physics_model import normalized_eedf,densities,ELEMENTARY_CHARGE,ELECTRON_MASS,HELIUM_ION_MASS,ion_current,electron_current
from src.physics_model import distribution_current

ROOT=Path(__file__).resolve().parents[2]

def test_full_native_pipeline_parity():
    v,i=load_data(ROOT/'data/raw_data.npz');out,d=reconstruct(v,i)
    with np.load(ROOT/'test/fixtures/upstream.npz',allow_pickle=False) as a:
        assert_array_equal(v,a['bias']);assert_array_equal(i,a['current'])
        assert_allclose(out['reconstruction'][0],a['eedf'],rtol=1e-12,atol=1e-14)
        assert_array_equal(out['energy'][0],a['energy'])
        for k in ['T_e','n_e','n_i','n_i_OML']:assert_allclose(out[k].ravel(),a[k].ravel(),rtol=1e-7)
        for k in ['V_P','V_F','I_es','I_is']:assert_array_equal(out[k].ravel(),a[k].ravel())
        assert_allclose(d['ion_fit'],a['ion_fit'],rtol=1e-12,atol=1e-12)
        assert_allclose(d['ion_current'],a['ion_current'],rtol=1e-12,atol=1e-15)
        assert_array_equal(v[d['first_mask']],a['section1_bias'])
        assert_array_equal(v[d['second_mask']],a['section2_bias'])
        assert_allclose(i[d['second_mask']]-d['ion_current'][d['second_mask']],a['section2_current'],rtol=1e-12)
        # Upstream masks extrapolation above maximum measured current solely for display.
        finite=np.isfinite(a['electron_current'])
        assert_allclose(d['electron_current'][finite],a['electron_current'][finite],rtol=1e-10)
        # x0 and y0 separately have a gauge ambiguity; compare physical log-current.
        assert_allclose(electron_current(v,a['electron_fit']),d['electron_current'],rtol=1e-10)

def test_quadratic_derivative_sign_and_normalization():
    v=np.linspace(-10,4,281);current=.7*v*v+2*v+3
    energy,p,first,second=normalized_eedf(v,current,3.,-7.)
    assert_allclose(second[2:-2],1.4,rtol=1e-10,atol=1e-10)
    assert np.all(np.diff(energy)<0)
    target=np.sqrt(energy);target/=abs(np.trapezoid(target,x=energy))
    assert_allclose(p,target,rtol=1e-10,atol=1e-10)
    assert_allclose(abs(np.trapezoid(p,x=energy)),1.,atol=1e-14)

def test_negative_curvature_is_not_clipped():
    v=np.linspace(-10,4,281);current=-v*v
    energy,p,*_=normalized_eedf(v,current,3.,-7.)
    assert np.all(p<0)

def test_temperature_from_known_exponential():
    v=np.linspace(-2,3,80);current=4e-3*np.exp((v-3)/2.75)
    fit,_=fit_temperature(v,current)
    assert_allclose(fit[2],2.75,rtol=1e-8)
    assert_allclose(electron_current(v,fit),current,rtol=1e-8)

def test_independent_energy_integral_matches_maxwellian_current():
    energy=np.linspace(1e-6,80,40001);temperature=2.75;density=2e16;vp=3.
    p=2*np.sqrt(energy)*np.exp(-energy/temperature)/(np.sqrt(np.pi)*temperature**1.5)
    v=np.linspace(-2,2,11)
    current=distribution_current(v,energy,p,vp,density,PROBE_AREA_M2)
    saturation=PROBE_AREA_M2*ELEMENTARY_CHARGE*density*np.sqrt(8*temperature*ELEMENTARY_CHARGE/(np.pi*ELECTRON_MASS))/4
    expected=saturation*np.exp((v-vp)/temperature)
    assert_allclose(current,expected,rtol=2e-7)

def test_density_units_and_scaling():
    area=PROBE_AREA_M2;temperature=2.75;ne=2e16;ni=3e16
    ies=area*ELEMENTARY_CHARGE*ne*np.sqrt(8*temperature*ELEMENTARY_CHARGE/(np.pi*ELECTRON_MASS))/4
    iis=-.6*ELEMENTARY_CHARGE*area*ni*np.sqrt(temperature*ELEMENTARY_CHARGE/HELIUM_ION_MASS)
    slope=-2*(area*ELEMENTARY_CHARGE**1.5*ni)**2/(np.pi**2*HELIUM_ION_MASS)*1e6
    measured=densities(iis,ies,temperature,slope,area)
    assert_allclose(measured,[ne,ni,ni],rtol=1e-14)
    assert_allclose(densities(iis*2,ies*2,temperature,slope*4,area),np.array(measured)*2,rtol=1e-14)
    assert_allclose(densities(iis,ies,temperature,slope,area*2),np.array(measured)/2,rtol=1e-14)

def test_preprocessing_duplicate_average():
    v,i=unique_characteristic([2,1,1,3],[7,2,4,8])
    assert_array_equal(v,[1,2,3]);assert_array_equal(i,[3,7,8])

def test_full_observation_sweep_not_cropped():
    v,i=load_data(ROOT/'data/raw_data.npz');out,d=reconstruct(v,i)
    assert len(v)==304 and d['gradient'].shape==(304,) and d['second_derivative'].shape==(304,)
    assert len(out['energy'][0])==64
    assert_allclose(PROBE_AREA_M2,np.pi*1.145e-3*1.57e-3+np.pi*(1.57e-3)**2/4,rtol=1e-15)
