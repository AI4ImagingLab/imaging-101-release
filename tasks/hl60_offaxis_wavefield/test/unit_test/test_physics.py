import numpy as np
from src.physics_model import intensity,sideband_peak,aperture,recover_field,unwrap_for_review
from src.preprocessing import padded_spectrum
from src.solvers import reconstruct
def test_analytic_interference_conjugate_sidebands():
    yy,xx=np.indices((64,64));reference=np.exp(-2j*np.pi*(-.25*yy-.125*xx));obj=np.ones((64,64))*np.exp(.6j)
    image=intensity(obj,reference)
    direct=2+2*np.cos(.6+2*np.pi*(-.25*yy-.125*xx));np.testing.assert_allclose(image,direct,atol=1e-13)
    peak=sideband_peak(padded_spectrum(image));assert peak==(-.25,-.125)
def test_identical_empty_field_gives_unity_transmission():
    yy,xx=np.indices((64,64));image=30+10*np.cos(2*np.pi*(.25*yy+.125*xx))
    out,_=reconstruct({'hologram':image,'empty_hologram':image.copy()})
    np.testing.assert_allclose(out['reconstruction'],1,atol=1e-14)
def test_local_unwrap_of_known_continuous_plane():
    yy,xx=np.indices((31,29));phase=.4*yy+.3*xx
    principal=(phase+np.pi)%(2*np.pi)-np.pi
    np.testing.assert_allclose(unwrap_for_review(principal),phase,atol=1e-13)
