import numpy as np
from src import preprocessing as pre,physics_model as physics

def test_monoclinic_metric_independent_cartesian():
    basis=physics.cell_basis();h=np.array([[2,1,-3],[-1,2,4],[0,1,0]])
    reciprocal=np.linalg.inv(basis).T
    expected=1/np.linalg.norm(h@reciprocal.T,axis=1)
    np.testing.assert_allclose(pre.dspacings(h),expected,rtol=2e-14)
    assert abs(abs(np.linalg.det(basis))-2000.756611312398)<1e-9

def test_known_fourier_pair_and_physical_scale():
    shape=(8,6,10);rho=np.zeros(shape);rho[1,2,3]=2.;rho-=rho.mean()
    h=np.array([[1,2,3],[2,-1,1],[0,0,0]])
    F=physics.structure_factors(rho,h)
    expected=20/np.prod(shape)*np.exp(2j*np.pi*(h@np.array([1/8,2/6,3/10])))
    expected[-1]=0
    np.testing.assert_allclose(F,expected,atol=1e-15)

def test_periodic_translation_inversion_preserves_intensities():
    rng=np.random.default_rng(17);rho=rng.normal(size=(8,6,10));h=rng.integers(-3,3,(19,3))
    inverted=rho[np.ix_(*[(-np.arange(n))%n for n in rho.shape])]
    for other in [np.roll(rho,(2,-1,3),axis=(0,1,2)),inverted]:
        np.testing.assert_allclose(abs(physics.structure_factors(other,h)),abs(physics.structure_factors(rho,h)),atol=3e-15)

def test_flip_preserves_pointwise_magnitude_and_norm():
    rho=np.array([-12.,-.4,0.,.1,2.,100.]).reshape(1,2,3)
    flipped=physics.flip_density(rho,.1,30.)
    np.testing.assert_array_equal(abs(flipped),abs(rho))
    assert flipped[0,0,0]>0

def test_p21_screw_absence_independent():
    # At0k0, translation by b/2 contributes exp(pi*i*k).
    for k in range(-3,4):
        absent,_,_,_=pre.equivalent_reflections([0,k,0])
        assert bool(absent)==bool(k%2)

def test_negative_observation_noise_floor_is_not_discarded():
    obs=np.array([1.,-2.,3.]);pred=np.sqrt(np.maximum(obs,0))
    assert physics.intensity_relative_error(pred,obs)==2/np.sqrt(14)
