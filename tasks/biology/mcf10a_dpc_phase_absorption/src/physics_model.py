"""BSD3 Waller weak-object equations, explicit local NumPy operators."""
import numpy as np
WAVELENGTH=.514
NA=.4
PIXEL_SIZE=6.5/40
ROTATIONS=(0,180,90,270)
def transfer_functions(shape=(512,512)):
    ny,nx=shape
    fx=np.fft.ifftshift((np.arange(nx,dtype='complex64')-nx//2)*(1/nx/PIXEL_SIZE))
    fy=np.fft.ifftshift((np.arange(ny,dtype='complex64')-ny//2)*(1/ny/PIXEL_SIZE))
    pupil=fx[None,:]**2+fy[:,None]**2<=(NA/WAVELENGTH)**2
    sources=[];hu=[];hp=[]
    for rotation in ROTATIONS:
        s=np.zeros(shape,dtype='float32');angle=np.deg2rad(rotation)
        criterion=fy[:,None]*np.cos(angle)+1e-15>=fx[None,:]*np.sin(angle)
        if rotation<180:s[criterion]=1;s*=pupil
        else:s[~criterion]=-1;s*=pupil;s+=pupil
        sources.append(s)
        term=np.fft.fft2(s*pupil)*np.fft.fft2(pupil).conj();i0=(s*pupil*pupil.conj()).sum()
        hu.append(2*np.fft.ifft2(term.real)/i0)
        hp.append(2j*np.fft.ifft2(1j*term.imag)/i0)
    return {'Hu':np.asarray(hu),'Hp':np.asarray(hp),'source':np.asarray(sources),'pupil':pupil,'fx':fx,'fy':fy}
def forward(absorption,phase,hu,hp):
    """Normalized contrast for transmission exp(absorption - 1j*phase).

    `phase` is the original numerical phase-delay convention. The optical
    complex argument under exp(+i*argument) is its negative.
    """
    return np.fft.ifft2(hu*np.fft.fft2(absorption)+hp*np.fft.fft2(phase)).real
def adjoint(images,hu,hp):
    spectra=np.fft.fft2(images)
    return np.fft.ifft2(np.sum(hu.conj()*spectra,axis=0)).real,np.fft.ifft2(np.sum(hp.conj()*spectra,axis=0)).real
