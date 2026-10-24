import numpy as np
import pytest
from src.preprocessing import split_channels, POLARIZER_ORDER, OFFSET_ADU
from src.physics_model import analyzer_matrix,forward,adjoint,polarization
from src.solvers import estimate_stokes,reconstruct

def test_independent_known_analyzer_case():
    # S=(10,6,8) gives perfect 100% polarization at atan2(8,6)/2.
    s=np.array([10.,6.,8.])[:,None,None]
    measured=np.array([8.,9.,2.,1.])[:,None,None]
    np.testing.assert_array_equal(forward(s),measured)
    np.testing.assert_array_equal(estimate_stokes(measured),s)
    angle,degree=polarization(s)
    np.testing.assert_allclose(angle,0.4636476090008061,atol=1e-15)
    np.testing.assert_allclose(degree,1.,atol=1e-15)

def test_independent_adjoint_and_normal_equations():
    rng=np.random.default_rng(31);s=rng.normal(size=(3,5,7));i=rng.normal(size=(4,5,7))
    assert np.vdot(forward(s),i)==pytest.approx(np.vdot(s,adjoint(i)),rel=1e-14,abs=1e-14)
    np.testing.assert_allclose(adjoint(forward(estimate_stokes(i))-i),0,atol=1e-15)
    A=analyzer_matrix();np.testing.assert_array_equal(A.T@A,np.diag([1,.5,.5]))

def test_gradient_by_independent_finite_differences():
    rng=np.random.default_rng(7);s=rng.normal(size=(3,3,4));i=rng.normal(size=(4,3,4));d=rng.normal(size=s.shape);h=1e-5
    E=lambda x:.5*np.sum((forward(x)-i)**2)
    assert (E(s+h*d)-E(s-h*d))/(2*h)==pytest.approx(np.sum(adjoint(forward(s)-i)*d),rel=1e-9)

def test_mosaic_signed_and_no_unsigned_overflow():
    a=np.array([[1,9],[3,7]],dtype=np.uint16)
    np.testing.assert_array_equal(split_channels(a)[:,0,0],[9,7,3,1])
    np.testing.assert_array_equal(estimate_stokes(split_channels(a))[:,0,0],[10,6,6])
    assert POLARIZER_ORDER==(-45,0,90,45) and OFFSET_ADU==0.0

def test_pedestal_invariance_and_degree_bias():
    rng=np.random.default_rng(61);a=rng.integers(200,700,size=(4,6),dtype=np.uint16);x=reconstruct(a);y=reconstruct(a.astype(float)+100)
    np.testing.assert_array_equal(y['reconstruction'][0]-x['reconstruction'][0],np.full((2,3),200.))
    np.testing.assert_array_equal(y['reconstruction'][1:],x['reconstruction'][1:])
    np.testing.assert_array_equal(y['orientation'],x['orientation'])
    assert np.all(y['degree']<=x['degree'])

def test_isotropic_zero_and_angle_periodicity():
    a,d=polarization(np.zeros((3,2,2)));np.testing.assert_array_equal(a,0);np.testing.assert_array_equal(d,0)
    with pytest.raises(ValueError):polarization(np.array([0.,1.,0.])[:,None,None])
    # Two signs of a linear dipole represent the same pi-periodic orientation.
    delta=np.array([0.,np.pi,-np.pi]);np.testing.assert_allclose(np.arctan2(np.sin(2*delta),np.cos(2*delta))/2,0,atol=2e-16)

@pytest.mark.parametrize('shape',[(3,4),(4,3)])
def test_odd_mosaic_is_rejected(shape):
    with pytest.raises(ValueError):split_channels(np.ones(shape))


def test_visualization_metric_definition():
    from src.visualization import ncc_nrmse
    n,r=ncc_nrmse(np.array([0.,1.,0.]),np.array([0.,1.,2.]))
    assert n==pytest.approx(1/np.sqrt(5))
    assert r==pytest.approx(np.sqrt(4/3)/2)


def test_public_acquisition_constants_match_source():
    from pathlib import Path
    from src.preprocessing import RAW_SHAPE
    readme=(Path(__file__).resolve().parents[2]/'README.md').read_text(encoding='utf8')
    assert str(RAW_SHAPE) in readme
    for literal in ['[-45, 0; 90, 45]', '0.0', '69', '138', '500.004']:
        assert literal in readme
