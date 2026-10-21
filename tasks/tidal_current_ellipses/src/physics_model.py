"""Local rotary harmonic basis and Cartesian/ellipse transformations.
Adapted from UTide/Codiga, MIT LICENSE; no domain package is imported.
"""
import numpy as np
from .preprocessing import gregorian_days
def harmonic_basis(time_seconds,frequency_cph,reference_days=None):
    """Complex(N,2K+1) matrix [exp(+iwt),exp(-iwt),1]."""
    t=gregorian_days(time_seconds);tref=.5*(t[0]+t[-1])if reference_days is None else reference_days
    phase=24*(t-tref)[:,None]*np.asarray(frequency_cph)[None]
    e=np.exp(1j*phase*2*np.pi)
    return np.column_stack([e,e.conj(),np.ones(len(t))]),float(tref)
def coefficients_from_rotary(rotary):
    """Complex2K+1 -> cosine(2,K),sine(2,K),mean(2),cm/s."""
    k=(len(rotary)-1)//2;p=rotary[:k];m=rotary[k:2*k]
    cosine=np.array([(p+m).real,(p+m).imag]);sine=np.array([-(p-m).imag,(p-m).real])
    return cosine,sine,np.array([rotary[-1].real,rotary[-1].imag])
def rotary_from_coefficients(cosine,sine,mean_current):
    """Two real Cartesian components -> complex rotary coefficients."""
    x=cosine[0]+1j*cosine[1];y=sine[0]+1j*sine[1]
    return np.concatenate([.5*(x-1j*y),.5*(x+1j*y),[mean_current[0]+1j*mean_current[1]]])
def predict(time_seconds,frequency_cph,cosine,sine,mean_current,reference_days):
    b,_=harmonic_basis(time_seconds,frequency_cph,reference_days)
    z=b@rotary_from_coefficients(cosine,sine,mean_current)
    return np.array([z.real,z.imag])
def ellipse_parameters(cosine,sine):
    """Return(K,4): major, signed minor(cm/s),inclination/raw phase(deg).
    Inclination[0,180),phase[0,360); zero-strength angles defined0 for display.
    Near-circular or zero harmonics have geometrically nonidentifiable angles;
    evaluation uses Cartesian coefficients rather than angle gauges.
    """
    x=cosine[0]+1j*cosine[1];y=sine[0]+1j*sine[1];p=.5*(x-1j*y);m=.5*(x+1j*y)
    major=abs(p)+abs(m);minor=abs(p)-abs(m)
    ep=np.angle(p,deg=True);em=np.angle(m,deg=True);theta=((ep+em)/2)%180;phase=(-ep+theta)%360
    theta=np.where(major==0,0,theta);phase=np.where(major==0,0,phase)
    return np.column_stack([major,minor,theta,phase])
