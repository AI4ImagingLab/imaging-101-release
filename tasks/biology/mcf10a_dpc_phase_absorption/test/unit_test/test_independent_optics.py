import numpy as np
from src.physics_model import transfer_functions,forward

def test_wotf_against_independent_incoherent_plane_wave_imaging():
    """Differentiate actual pupil-filtered complex fields, not a copied WOTF."""
    n=32;model=transfer_functions((n,n));yy,xx=np.indices((n,n))
    phase=.2*np.cos(2*np.pi*xx/n)+.3*np.sin(4*np.pi*yy/n)
    absorption=.15*np.cos(2*np.pi*(xx+yy)/n)
    eps=1e-4;predicted=[]
    frequencies=np.fft.fftfreq(n)*n
    for source in model['source']:
        measurements=[]
        for sign in [1,-1]:
            # The preserved original Waller output is PHASE DELAY, not the
            # positive complex argument. Public contract: T=exp(u-i*phase).
            transmission=1+sign*eps*(absorption-1j*phase)
            total=np.zeros((n,n))
            for iy,ix in np.argwhere(source>0):
                illumination=np.exp(2j*np.pi*(frequencies[iy]*yy+frequencies[ix]*xx)/n)
                detector=np.fft.ifft2(model['pupil']*np.fft.fft2(transmission*illumination))
                total+=abs(detector)**2
            measurements.append(total/source.sum())
        predicted.append((measurements[0]-measurements[1])/(2*eps))
    linear=forward(absorption,phase,model['Hu'],model['Hp'])
    np.testing.assert_allclose(linear,predicted,rtol=1e-6,atol=1e-8)
