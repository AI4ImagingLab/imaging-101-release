"""Local interference and finite-aperture sideband operators."""
import numpy as np
from scipy.signal import convolve
def intensity(object_wave,reference_wave):return np.abs(object_wave+reference_wave)**2
def sideband_peak(spectrum):
    values=spectrum.copy();ny,nx=values.shape;cy,cx=ny//2,nx//2
    values[cy-max(int(np.ceil(ny/42)),5):]=0
    values[cy-3:cy+3]=0;values[:,cx-3:cx+3]=0
    iy,ix=np.unravel_index(np.argmax(abs(values)),values.shape)
    return np.fft.fftshift(np.fft.fftfreq(ny))[iy],np.fft.fftshift(np.fft.fftfreq(nx))[ix]
def aperture(shape,carrier,size_fraction=.5,kind='smooth disk'):
    fy=np.fft.fftshift(np.fft.fftfreq(shape[0]))[:,None];fx=np.fft.fftshift(np.fft.fftfreq(shape[1]))[None,:]
    radius=float(np.sqrt(np.sum(np.asarray(carrier)**2))*size_fraction)
    disk=(carrier[0]-fy)**2+(carrier[1]-fx)**2<=radius**2
    if kind=='disk':return disk.astype(float)
    if kind!='smooth disk':raise ValueError('Unknown aperture')
    gaussian=np.exp(-(fy**2+fx**2)/(2*(radius/5)**2))
    window=convolve(gaussian,disk,mode='same');window/=window.max();return window
def recover_field(spectrum,window,carrier,shape):
    filtered=spectrum*window
    shift=(int(carrier[0]*spectrum.shape[0]),int(carrier[1]*spectrum.shape[1]))
    centered=np.roll(np.roll(filtered,-shift[0],axis=0),-shift[1],axis=1)
    field=np.fft.ifft2(np.fft.ifftshift(centered))[:shape[0],:shape[1]]
    return field,filtered
def unwrap_axis(phase,axis):
    d=np.diff(phase,axis=axis);wrapped=(d+np.pi)%(2*np.pi)-np.pi
    wrapped[(wrapped==-np.pi)&(d>0)]=np.pi
    correction=wrapped-d;correction[np.abs(d)<np.pi]=0
    pad=[(0,0)]*phase.ndim;pad[axis]=(1,0)
    return phase+np.pad(np.cumsum(correction,axis=axis),pad)
def unwrap_for_review(phase):return unwrap_axis(unwrap_axis(phase,1),0)
