import numpy as np
from src.physics_model import cyclic_convolve,cyclic_adjoint,nearest_resize,central_crop,projected_size,source_coordinates
def test_fourier_against_direct_sum():
    x=np.arange(20,dtype=float).reshape(4,5);h=np.sin(x)
    direct=sum(x[i,j]*np.roll(h,(i,j),(0,1)) for i in range(4) for j in range(5))
    np.testing.assert_allclose(cyclic_convolve(x,h),direct,rtol=1e-13,atol=1e-12)
def test_adjoint_flux_and_shift():
    rng=np.random.default_rng(52);x=rng.normal(size=(7,8));y=rng.normal(size=x.shape);h=rng.uniform(size=x.shape)
    np.testing.assert_allclose(np.vdot(cyclic_convolve(x,h),y),np.vdot(x,cyclic_adjoint(y,h)),rtol=1e-13)
    np.testing.assert_allclose(cyclic_convolve(x,h).sum(),x.sum()*h.sum(),rtol=1e-13)
    np.testing.assert_allclose(cyclic_convolve(np.roll(x,(2,-1),(0,1)),h),np.roll(cyclic_convolve(x,h),(2,-1),(0,1)),atol=1e-12)
def test_resize_pixel_centres():
    a=np.array([[0,1],[2,3]],dtype=np.float32)
    np.testing.assert_array_equal(nearest_resize(a,4),np.repeat(np.repeat(a,2,0),2,1))
def test_geometry_and_crop():
    import pytest
    assert projected_size(50)==126
    with pytest.raises(ValueError):projected_size(0)
    a=np.arange(64).reshape(8,8)
    np.testing.assert_array_equal(central_crop(a,.5),a[2:6,2:6])
    np.testing.assert_allclose(source_coordinates(4,50),[.275,.1375,0.,-.1375],atol=1e-15)
