import numpy as np
import pytest
from src.physics_model import phase_conductivity, full_helmholtz, second_derivative_kernels, MU0, OMEGA, EPS0
from src.preprocessing import average_repeats
from src.visualization import agreement


def test_known_uniform_conductivity_sign_spacing_factor_two():
    z,y,x=(np.indices((19,19,19))-9)*.002
    sigma=.7
    phase=sigma*MU0*OMEGA/3*(x*x+y*y+z*z)+.3*x-.5*y+2
    estimate, mask=phase_conductivity(phase, 'cube', 3)
    np.testing.assert_allclose(estimate[mask], sigma, rtol=0, atol=2e-11)
    wrong,_=phase_conductivity(phase, 'cube', 3, spacing=(.004,.004,.004))
    np.testing.assert_allclose(wrong[mask], sigma/4, rtol=0, atol=2e-11)
    doubled,_=phase_conductivity(phase, 'cube', 3, frequency=256e6)
    np.testing.assert_allclose(doubled[mask], sigma/2, rtol=0, atol=2e-11)


def test_complex_wave_physics_and_global_magnitude_scale():
    z,y,x=(np.indices((19,19,19))-9)*.002
    a,b=25.,6.
    phase=2*a*x;mag=np.exp(b*x)
    sigma,epsr,mask=full_helmholtz(phase,mag,'cube',2)
    expected_sigma=2*a*b/(MU0*OMEGA)
    expected_epsr=(a*a-b*b)/(MU0*OMEGA*OMEGA*EPS0)
    # Finite-window local polynomial approximates the continuum Laplacian.
    np.testing.assert_allclose(sigma[mask],expected_sigma,rtol=.005)
    np.testing.assert_allclose(epsr[mask],expected_epsr,rtol=.005)
    s2,e2,m2=full_helmholtz(phase,mag*7.3,'cube',2)
    np.testing.assert_allclose(sigma[mask],s2[mask],rtol=1e-10,atol=2e-11)
    np.testing.assert_allclose(epsr[mask],e2[mask],rtol=1e-10,atol=2e-9)


def test_source_support_has_no_target_selection():
    phase=np.zeros((15,15,15));phase[7,7,7]=np.nan
    result,mask=phase_conductivity(phase,'cube',2)
    assert not mask[5:10,5:10,5:10].any()
    assert np.isnan(result[~mask]).all()
    assert not mask[:2].any() and mask[3,3,3]


def test_polynomial_moments_and_offset_invariance():
    for d,kernel in enumerate(second_derivative_kernels('cube',5)):
        xyz=(np.indices(kernel.shape)-5)[::-1]
        assert abs(kernel.sum())<1e-12
        for axis in range(3):
            assert abs((kernel*xyz[axis]**2).sum()-(2 if axis==d else 0))<1e-12
    rng=np.random.default_rng(19);phase=rng.normal(size=(15,15,15))*.01
    a,mask=phase_conductivity(phase,'cross',2)
    b,_=phase_conductivity(phase+3,'cross',2)
    np.testing.assert_allclose(a[mask],b[mask],rtol=1e-10,atol=2e-10)


def test_metric_known_case_and_source_order_average():
    reference=np.array([1.,2.,4.]);estimate=np.array([1.,2.,3.]);mask=np.ones(3,bool)
    measured=agreement(estimate,reference,mask)
    assert measured['nrmse']==pytest.approx(np.sqrt(1/3)/3)
    assert measured['ncc']==pytest.approx(17/np.sqrt(14*21))
    with pytest.raises(ValueError):agreement(reference,np.ones(3),mask)
    x=np.array([[1.,np.nan],[2.,np.nan],[3.,np.nan]])
    np.testing.assert_array_equal(average_repeats(x),np.array([2.,np.nan]))
