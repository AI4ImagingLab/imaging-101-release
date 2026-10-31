import numpy as np
from src.solvers import sample_grid

def test_multinomial_mass_and_uniform_within_cell_statistics():
    # No fixed seed or expected random sequence. Test distributional properties
    # at tolerances wider than seven standard errors for50,000samples.
    mass=np.zeros((35,35,35,35))
    first=(10,12,14,16);second=(22,20,18,16)
    mass[first]=.25;mass[second]=.75
    z=sample_grid(mass.ravel(),50000,np.random.default_rng())
    width=8/35
    indices=np.floor((z+4)/width).astype(int)
    which_first=np.all(indices==np.asarray(first),axis=1)
    which_second=np.all(indices==np.asarray(second),axis=1)
    assert np.all(which_first|which_second)
    assert abs(which_first.mean()-.25)<.015
    local=(z-(-4+indices*width))/width
    assert local.min()>=0 and local.max()<=1
    np.testing.assert_allclose(local.mean(axis=0),.5,rtol=0,atol=.01)
    np.testing.assert_allclose(local.var(axis=0),1/12,rtol=.03,atol=0)
