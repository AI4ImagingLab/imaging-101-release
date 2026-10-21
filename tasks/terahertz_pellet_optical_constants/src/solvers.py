"""Explicit FFT phase referencing and closed-form slab inversion.
Adapted from THzPy, LGPL-3.0; Leung et al.2025 tutorial and Jepsen2019 phase.
"""
import numpy as np
from .preprocessing import common_window,THICKNESSES_MM
from .physics_model import slab_inverse
def transform(sample,baseline,upsampling=3,min_frequency=.2,max_frequency=2.5,peak_snr=.3,fit_range=.2):
    """(2,N) field/time_ps -> amplitude,positive_delay_phase,frequency_Hz."""
    count=(1<<(len(sample[0])-1).bit_length())*2**upsampling
    f=np.fft.fftfreq(count,np.mean(np.diff(sample[1])))
    a=np.fft.fft(sample[0],count);b=np.fft.fft(baseline[0],count)
    amplitude=abs(a)/abs(b)
    # Original phase reference deliberately uses positive peaks here; window
    # centering uses absolute peaks. Their roles are distinct and preserved.
    ts=sample[1][np.argmax(sample[0])];tr=baseline[1][np.argmax(baseline[0])]
    k1=2*np.pi*ts*f;k2=2*np.pi*tr*f
    u=np.unwrap(np.angle(a*np.e**(-1j*k1))-np.angle(b*np.e**(-1j*k2)))
    left=np.argmin(abs(f-(peak_snr-fit_range/2)));right=np.argmin(abs(f-(peak_snr+fit_range/2)))
    design=np.vstack([f[left:right],np.ones(right-left)]).T
    _,intercept=np.linalg.lstsq(design,u[left:right],rcond=None)[0]
    phase=abs(u+(k1-k2)-2*np.pi*round(intercept/(2*np.pi)))
    mask=(f>=min_frequency)&(f<=max_frequency)
    return amplitude[mask],phase[mask],f[mask]*1e12
def reconstruct(time_ps,fields,half_width=15.,window='hanning',thicknesses_mm=THICKNESSES_MM,return_diagnostics=False):
    """Full measured channels -> PE then mixture n,alpha, each (1,2,1256).
    Diagnostic experiments may alter output frequency count when window changes.
    """
    w=common_window(time_ps,fields,half_width,window);indices=[1,0];n=[];alpha=[];amps=[];phases=[]
    for channel,thickness in zip(indices,thicknesses_mm):
        amp,phase,f=transform(w[channel],w[2]);ni,ai=slab_inverse(amp,phase,f,thickness*1e-3)
        n.append(ni);alpha.append(ai);amps.append(amp);phases.append(phase)
    result=dict(real_index=np.array(n)[None],absorption_coefficient=np.array(alpha)[None])
    return(result,dict(windowed=w,frequency_hz=f,amplitude=np.array(amps),phase=np.array(phases)))if return_diagnostics else result
