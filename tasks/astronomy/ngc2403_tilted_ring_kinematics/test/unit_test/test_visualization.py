import numpy as np
from src.visualization import rms,signed_norm,spectral_slice
from src.preprocessing import PIXEL_ARCSEC


def test_physical_rms_and_neutral_signed_scale():
    assert rms(np.array([3.,4.]))==np.sqrt(12.5)
    norm=signed_norm(np.array([-2.,1.]))
    assert norm(-2.)==0 and norm(0.)==.5 and norm(2.)==1
    zero=signed_norm(np.zeros(3))
    assert zero(0.)==.5 and zero.vmin<0<zero.vmax


def test_major_axis_display_preserves_channel_and_signed_pixel_geometry():
    channel,y,x=np.indices((62,150,150))
    cube=(1000*channel+10*y+x).astype(float)
    radius=np.array([-30.,0.,30.])
    actual_radius,cut=spectral_slice(cube,90.,radius)
    expected=1000*np.arange(62)[:,None]+10*77+77-radius[None]/PIXEL_ARCSEC
    np.testing.assert_array_equal(actual_radius,radius)
    np.testing.assert_allclose(cut,expected,atol=1e-10,rtol=0)
