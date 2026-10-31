"""Normalized DFT, IRF calibration and apparent lifetime inversion.

Mathematical extraction from PhasorPy v0.7, MIT, Copyright PhasorPy
Contributors. See NOTICE for attribution. No PhasorPy runtime import.
"""
import numpy as np
FREQUENCY_MHZ=40.00017089948058
CALIBRATION_LIFETIME_NS=4.1
LASER_PERIOD_NS=24.99989318828099

def normalized_phasor(signal):
    """Return mean, cosine and positive-sine phasors along last time axis."""
    signal=np.asarray(signal,dtype=np.float64)
    n=signal.shape[-1]
    theta=2*np.pi*np.arange(n)/n
    total=signal.sum(-1)
    with np.errstate(divide='ignore',invalid='ignore'):
        return total/n,(signal@np.cos(theta))/total,(signal@np.sin(theta))/total

def single_exponential_phasor(lifetime_ns,frequency_mhz=FREQUENCY_MHZ):
    """Analytic phasor of a periodic causal exponential, dimensionless."""
    wt=2*np.pi*frequency_mhz*1e-3*np.asarray(lifetime_ns)
    return 1/(1+wt*wt),wt/(1+wt*wt)

def calibrate(real,imag,calibration):
    """Remove instrument response using independent pooled 4.1 ns dye counts."""
    _,cg,cs=normalized_phasor(calibration)
    tg,ts=single_exponential_phasor(CALIBRATION_LIFETIME_NS)
    factor=(tg+1j*ts)/complex(float(np.ravel(cg)[0]),float(np.ravel(cs)[0]))
    z=(np.asarray(real)+1j*np.asarray(imag))*factor
    return z.real,z.imag

def lifetimes(real,imag,frequency_mhz=FREQUENCY_MHZ):
    """Return ns lifetimes, retaining upstream's out-of-domain convention.

    Modulation >1 returns zero; a zero phasor returns infinity. Neither case
    belongs to the benchmark's supplied valid measurement support.
    """
    omega=2*np.pi*frequency_mhz*1e-3
    with np.errstate(divide='ignore',invalid='ignore'):
        phase=imag/(omega*real)
        radius2=real*real+imag*imag
        mod=np.where(radius2>1,0.,np.sqrt(1/radius2-1)/omega)
        phase=np.where(radius2==0,np.inf,phase)
    return phase,mod
