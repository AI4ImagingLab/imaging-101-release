import numpy as np
from src.physics_model import concentrations, bound_complex, association_constants
def test_mass_action_and_conservation():
    s=np.geomspace(1e-6,1e-3,30); t=np.geomspace(1e-7,1e-2,30)
    c=bound_complex(s,t,1e4)
    assert np.all(c>=0) and np.all(c<=np.minimum(s,t))
    np.testing.assert_allclose(c,1e4*(s-c)*(t-c),rtol=1e-10,atol=1e-18)
def test_zero_ligand():
    np.testing.assert_allclose(bound_complex(np.array([1e-4]),np.array([0.]),1e4),0,atol=1e-18)
def test_injection_mass_balance():
    s,t=concentrations(np.array([2.,2.]),1.,3.,100.)
    np.testing.assert_allclose(s,[1.,.98,.9604],rtol=1e-15)
    np.testing.assert_allclose(t,[0.,.06,.1188],rtol=1e-15,atol=3e-16)
def test_vant_hoff_derivative():
    temp=np.array([298.149,298.151]); k=association_constants(-6000.,10000.,temp)
    derivative=np.diff(np.log(k))/np.diff(1/temp)
    np.testing.assert_allclose(derivative,-(-6000.)/1.9872036,rtol=1e-9)
