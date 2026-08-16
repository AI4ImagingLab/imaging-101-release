# Defocused probe builder (reference pseudocode)

```
def electron_wavelength_A(energy_eV):
    m0=9.10938e-31; e=1.60218e-19; h=6.62607e-34; c=2.99792e8
    E=energy_eV*e
    return h/np.sqrt(2*m0*E*(1+E/(2*m0*c**2)))*1e10

def build_probe(vp, defocus, energy_eV, R_pixel_size_A):
    lam=electron_wavelength_A(energy_eV)
    Qx,Qy=vp.shape
    yy,xx=np.meshgrid(np.arange(Qy),np.arange(Qx))
    vp=vp.astype(float); tot=vp.sum()
    cx=(xx*vp).sum()/tot; cy=(yy*vp).sum()/tot
    sx=Qx/2-cx; sy=Qy/2-cy
    KX,KY=np.meshgrid(np.fft.fftfreq(Qx),np.fft.fftfreq(Qy),indexing='ij')
    ramp=np.exp(-2j*np.pi*(KX*sx+KY*sy))
    centered=np.maximum(np.real(np.fft.ifft2(np.fft.fft2(vp)*ramp)),0)
    amp_k=np.sqrt(centered)
    kx=np.fft.fftfreq(Qx,d=R_pixel_size_A); ky=np.fft.fftfreq(Qy,d=R_pixel_size_A)
    KXp,KYp=np.meshgrid(kx,ky,indexing='ij')
    chi=-np.pi*lam*defocus*(KXp**2+KYp**2)
    probe_k=np.fft.fftshift(amp_k)*np.exp(1j*chi)
    return np.fft.ifft2(probe_k).astype(np.complex128)
```

Note meshgrid indexing: the reference used default meshgrid with (Qy,Qx) then paired as (KY,KX); use indexing='ij' consistently so KX indexes the first (Qx) axis. Keep amplitude and defocus-phase conventions exactly as above.