from pathlib import Path
import sys,numpy as np,pytest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.preprocessing import load_data,normalize_irf
from src.solvers import zero_norm,circular_correlation,matched_peaks,reconstruct
from src.physics_model import expected_histogram,shifts_to_depth,fit_nuisance,BIN_DEPTH_MM
@pytest.fixture(scope='module')
def data():return load_data(ROOT/'data/raw_data.npz')
def test_exact_original_full_pipeline(data):
    d,p=reconstruct(*data)
    with np.load(ROOT/'test/fixtures/upstream.npz') as z:np.testing.assert_array_equal(p,z['peaks']);np.testing.assert_array_equal(d,z['depth_mm'])
def test_original_normalization_correlation(data):
    counts,h=data
    with np.load(ROOT/'test/fixtures/upstream.npz') as z:
        np.testing.assert_allclose(h,z['irf'],rtol=1e-15,atol=1e-19)
        np.testing.assert_allclose(zero_norm(h),z['example_normalized_template'],atol=1e-14)
        np.testing.assert_allclose(zero_norm(counts[25]),z['example_normalized_signal'],atol=1e-14)
        np.testing.assert_allclose(circular_correlation(zero_norm(counts[25]),zero_norm(h)),z['example_correlation'],atol=1e-12)
def test_complete_small_original_constructor():
    with np.load(ROOT/'test/fixtures/upstream.npz') as z:p,_=matched_peaks(z['small_constructor_cases'],z['small_constructor_irf']);np.testing.assert_array_equal(p,z['small_constructor_peaks'])
def test_original_known_shifts_across_circular_boundary(data):
    with np.load(ROOT/'test/fixtures/upstream_physics.npz') as z:p,_=matched_peaks(z['known_pulses'],data[1]);np.testing.assert_array_equal(p,z['known_shifts']);np.testing.assert_array_equal(p,z['original_known_peaks'])
def test_independent_direct_lag_sum(data):
    y=zero_norm(data[0][25]);h=zero_norm(data[1]);corr=circular_correlation(y,h)
    with np.load(ROOT/'test/fixtures/upstream_physics.npz') as z:
        lags=z['direct_lags'];direct=np.stack([y@np.roll(h,int(s)) for s in lags],axis=-1);np.testing.assert_allclose(corr[:,lags],direct,atol=1e-12);np.testing.assert_allclose(direct,z['direct_correlation'],atol=1e-12)
@pytest.mark.parametrize('shift',[0,1,31])
def test_photon_forward_inverse_and_boundary(shift):
    h=np.zeros(32);h[:3]=[.6,.3,.1];y=expected_histogram(h,shift,10000.,2.);p,_=matched_peaks(y,h);assert p==shift
    assert y.sum()==pytest.approx(10000+64)
def test_independent_roundtrip_time_units_and_gauge():
    s=np.array([[7,8],[10,2]]);d=shifts_to_depth(s);expected=np.array([[0,1],[3,-5]])*.149896229;np.testing.assert_allclose(d,expected,atol=1e-15)
    np.testing.assert_array_equal(d,shifts_to_depth(s,time_bin_s=1.*1e-12));assert d[0,0]==0 and BIN_DEPTH_MM==pytest.approx(.149896229,abs=1e-15)
def test_exact_nonnegative_nuisance_forward():
    h=np.exp(-np.arange(30)**2/8);h/=h.sum();y=expected_histogram(h,8,500.,3.);A,b,p=fit_nuisance(y,h,8);assert A==pytest.approx(500);assert b==pytest.approx(3);np.testing.assert_allclose(y,p,atol=1e-13)
def test_background_and_positive_gain_invariance():
    rng=np.random.default_rng(25);h=rng.random(32);h/=h.sum();y=4*np.roll(h,7)+2;p,_=matched_peaks(y,h);p2,_=matched_peaks(5*y+20,h);assert p==p2==7
def test_reversal_is_not_same_operator():
    h=np.array([.7,.2,.1]+[0]*29);y=np.roll(h,5);corr=circular_correlation(y,h);reverse=circular_correlation(h,y);assert np.argmax(corr)==5 and np.argmax(reverse)==27
def test_no_target_calibration_in_input():
    with np.load(ROOT/'data/raw_data.npz') as z:assert set(z.files)=={'photon_counts','instrument_response'}
