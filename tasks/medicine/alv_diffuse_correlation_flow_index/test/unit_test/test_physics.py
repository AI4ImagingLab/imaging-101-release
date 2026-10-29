import numpy as np
from src.physics_model import correlation,field_correlation
def test_static_and_zero_lag():
    tau=np.geomspace(1e-10,.01,100)
    np.testing.assert_array_equal(correlation(tau,.5,0),np.full(100,1.5))
    assert correlation(np.array([0.]),.53,1000)[0]==1.53
def test_normalization_decay_and_siegert():
    tau=np.r_[0,np.geomspace(1e-10,.1,500)]
    g=field_correlation(tau,1000)
    assert g[0]==1 and np.all(g>=0) and np.all(np.diff(g)<=0)
    np.testing.assert_allclose(correlation(tau,.48,1000),1+.48*g*g)
def test_medium_wavevector_dimensional_scaling():
    tau=np.geomspace(1e-9,1e-3,30)
    # Keeping n/lambda fixed preserves k while boundary n is unchanged.
    np.testing.assert_allclose(field_correlation(tau,4000,wavelength_mm=.00157),field_correlation(tau,1000),rtol=1e-14)
