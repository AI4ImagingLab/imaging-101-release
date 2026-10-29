"""Explicit phase/induction and lens-defocus operators; no domain package."""
import numpy as np
from scipy.constants import h,hbar,e,m_e,c
def wavelength_nm(voltage=200000.):
    """Relativistic electron de Broglie wavelength, nm, voltage in V."""
    epsilon=.5*e/m_e/c**2
    return h*1e9/np.sqrt(2*m_e*e)/np.sqrt(voltage+epsilon*voltage**2)
def periodic_gradient(image):
    """Centered pixel derivative with periodic wrapping, returning (dy,dx)."""
    return tuple((np.roll(image,-1,axis=k)-np.roll(image,1,axis=k))*.5 for k in [0,1])
def frequency_filters(shape,pixel_nm=5.1066,qc=0.):
    """Inverse q^2 response and native 2/3-Nyquist circular aperture.

    Frequencies are cycles/pixel, qc is cycles/nm; the zero mode is removed.
    """
    ny,nx=shape;qy,qx=np.meshgrid(np.fft.fftfreq(ny),np.fft.fftfreq(nx),indexing='ij')
    q=np.sqrt(qx*qx+qy*qy);q[0,0]=1
    inv=q*q/(q*q+(qc*pixel_nm)**2)**2 if qc>0 else 1/q**2
    inv[0,0]=0
    gy,gx=np.ogrid[-ny//2:ny//2,-nx//2:nx//2]
    aperture=np.fft.ifftshift(gy*gy+gx*gx<=(ny/3)**2)
    return inv,aperture
def even_extension(image):
    """Even mirrored extension in both axes, preserving the top-left ROI."""
    image=np.asarray(image,dtype=np.float64)
    a=np.concatenate([image,np.flip(image,axis=0)],axis=0)
    return np.concatenate([a,np.flip(a,axis=1)],axis=1)
def phase_to_induction(phase,pixel_nm=5.1066):
    """Return (Bx,By)=integral B dz in T nm using second-order edge derivatives."""
    dy,dx=np.gradient(phase,edge_order=2)
    factor=hbar/(e*pixel_nm)*1e18
    return factor*dy,-factor*dx
def defocus_derivative(phase,infocus,pixel_nm=5.1066,voltage=200000.):
    """Independent spectral TIE derivative per nm of lens defocus.

    Positive source lens defocus has transfer exp(+i*pi*lambda*df*k^2),
    opposite free-space propagation z. Thus dI/ddf=+lambda/(2pi)div(I0 grad phi).
    This diagnostic uses spectral derivatives, independently of inverse stencils.
    """
    fy,fx=np.meshgrid(np.fft.fftfreq(phase.shape[0],pixel_nm),np.fft.fftfreq(phase.shape[1],pixel_nm),indexing='ij')
    ft=np.fft.fft2(phase)
    gy=np.fft.ifft2(2j*np.pi*fy*ft).real;gx=np.fft.ifft2(2j*np.pi*fx*ft).real
    div=np.fft.ifft2(2j*np.pi*(fy*np.fft.fft2(infocus*gy)+fx*np.fft.fft2(infocus*gx))).real
    return wavelength_nm(voltage)/(2*np.pi)*div
def propagate(phase,defocus_nm,pixel_nm=5.1066,voltage=200000.,amplitude=None):
    """Coherent paraxial intensity, useful for independent weak-phase tests."""
    fy,fx=np.meshgrid(np.fft.fftfreq(phase.shape[0],pixel_nm),np.fft.fftfreq(phase.shape[1],pixel_nm),indexing='ij')
    wave=np.exp(1j*phase) if amplitude is None else amplitude*np.exp(1j*phase)
    return abs(np.fft.ifft2(np.fft.fft2(wave)*np.exp(1j*np.pi*wavelength_nm(voltage)*defocus_nm*(fx*fx+fy*fy))))**2
