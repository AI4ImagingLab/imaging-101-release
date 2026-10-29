import numpy as np
from src.generate_data import gaussian_blur,synthetic_mixture
def test_known_gaussian_constant_conservation():
 x=np.full((17,23),.4);np.testing.assert_allclose(gaussian_blur(x,3),x,atol=1e-15)
def test_known_mixture_and_mean_ambiguity():
 z=synthetic_mixture(noise=0)
 np.testing.assert_array_equal(z['observation'],z['transmitted_component']+(1-z['alpha'])*z['blurred_reflection'])
 np.testing.assert_allclose(z['observation'].mean(),z['mean_convention_target'].mean(),atol=1e-15)
