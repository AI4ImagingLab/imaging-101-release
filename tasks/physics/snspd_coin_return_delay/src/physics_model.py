"""Explicit discrete pulse-shift measurement physics; no reconstruction library."""
import numpy as np
LIGHT_SPEED_M_S=299792458.;TIME_BIN_S=1e-12;BIN_DEPTH_MM=LIGHT_SPEED_M_S*TIME_BIN_S*.5*1000
REFERENCE_PIXEL=(0,0)
def expected_histogram(irf,shift,signal_photons,background_per_bin):
    """Poisson mean counts(N,): A*h[(i-shift) modN]+b, unit-sum h."""
    if not isinstance(shift,(int,np.integer)):raise ValueError('Integer shift required')
    if signal_photons<0 or background_per_bin<0:raise ValueError('Nonnegative photon intensities required')
    return signal_photons*np.roll(irf,int(shift))+background_per_bin
def shifts_to_depth(shifts,time_bin_s=TIME_BIN_S):
    """Signed depth difference in mm from fixed pixel(0,0), no modulo unwrapping."""
    s=np.asarray(shifts);return (s-s[REFERENCE_PIXEL])*LIGHT_SPEED_M_S*time_bin_s*.5*1000
def fit_nuisance(hist,irf,shift):
    """Nonnegative two-column least squares for pulse count A and uniform background b.

    Diagnostic only; it does not change the original matched-filter peak.
    Explicit interior solution plus the two nonnegative boundary candidates.
    """
    y=np.asarray(hist,dtype=np.float64);h=np.roll(irf,int(shift));n=y.size
    sy=y.sum();hy=np.dot(h,y);hh=np.dot(h,h);den=n*hh-1
    A=(n*hy-sy)/den;b=(hh*sy-hy)/den
    candidates=[(max(0.,hy/hh),0.),(0.,max(0.,sy/n))]
    if A>=0 and b>=0:candidates.append((float(A),float(b)))
    A,b=min(candidates,key=lambda pair:np.sum((pair[0]*h+pair[1]-y)**2))
    model=A*h+b;return A,b,model
