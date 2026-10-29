"""Explicit exploding-reflector dispersion and sampled Stolt remapping.

GPL-3.0; adapted from ImpDAR migrationStolt, Benjamin Hills (2019).
"""
import numpy as np

def dispersion(n,nx,dt,dx,velocity):
    """Return omega,kx,kz,query-frequency,dimensionless obliquity arrays.

    omega rad/s, kx/kz rad/m; scalar-wave exploding-reflector speed v/2.
    The input and output vertical sample index corresponds to z=v*t/2.
    """
    omega=2*np.pi*np.fft.rfftfreq(n,dt)
    kx=2*np.pi*np.fft.fftfreq(nx,dx)
    kz=omega*2/velocity
    target=velocity/2*np.sqrt(kz[:,None]**2+kx[None,:]**2)
    with np.errstate(invalid='ignore'):
        scaling=kz[:,None]/np.sqrt(kx[None,:]**2+kz[:,None]**2)
    return omega,kx,kz,target,scaling

def interpolation_weights(omega,target):
    """Endpoint-extended linear interpolation indices and fractional weights."""
    index=np.searchsorted(omega,target,side='right')-1
    index=np.clip(index,0,len(omega)-2)
    fraction=np.clip((target-omega[index])/(omega[index+1]-omega[index]),0,1)
    return index,fraction

def remap(spectrum,omega,target):
    """Interpolate each fixed-kx column; preserve upstream last-bin exclusion."""
    index,fraction=interpolation_weights(omega,target)
    column=np.arange(spectrum.shape[1])[None,:]
    mapped=(1-fraction)*spectrum[index,column]+fraction*spectrum[index+1,column]
    mapped[-1]=0
    return mapped

def remap_adjoint(spectrum,omega,target):
    """Exact Euclidean adjoint of sampled remap; independently tested."""
    index,fraction=interpolation_weights(omega,target)
    out=np.zeros_like(spectrum)
    values=spectrum.copy();values[-1]=0
    columns=np.broadcast_to(np.arange(spectrum.shape[1])[None,:],index.shape)
    np.add.at(out,(index,columns),(1-fraction)*values)
    np.add.at(out,(index+1,columns),fraction*values)
    return out

def point_diffraction(x,time_s,source_x,depth,velocity,pulse_width):
    """Independent kinematic point-response used for the physics check.

    x/source_x/depth metres; time/pulse_width seconds. Returns a real
    time×trace Ricker-pulse diffraction with two-way hyperbolic travel time
    2*sqrt((x-source_x)^2+depth^2)/velocity and cylindrical 1/sqrt(R)
    spreading. It is an analytic validation helper, not a source of the
    bundled experimental observations or a replacement forward-wave solver.
    """
    radius=np.sqrt((np.asarray(x)-source_x)**2+depth**2)
    tau=(np.asarray(time_s)[:,None]-2*radius[None,:]/velocity)/pulse_width
    return (1-2*tau*tau)*np.exp(-tau*tau)/np.sqrt(radius[None,:])
