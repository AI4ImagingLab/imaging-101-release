"""Explicit interference, phase-shifting and digital-reference mathematics.

SOSR adaptations retain the MIT-licensed pyDHM convention; see test/validation.md (full license record).
"""
import math
import numpy as np

WAVELENGTH=632.8e-9
DX=6.9e-6
DY=6.9e-6

def centered_spectrum(frame: np.ndarray) -> np.ndarray:
    """Unnormalized centered forward FFT, including upstream pre-FFT shift."""
    return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(frame)))

def reference_wave(shape: tuple[int,int], fx: float, fy: float,
                   wavelength: float=WAVELENGTH, dx: float=DX,
                   dy: float=DY) -> np.ndarray:
    """Unit phasor on coordinates (column-N/2,row-M/2), SI physical units."""
    m,n=shape
    x,y=np.meshgrid(np.arange(n)-n/2,np.arange(m)-m/2,indexing='xy')
    tx=math.asin((n/2-fx)*wavelength/(n*dx))
    ty=math.asin((m/2-fy)*wavelength/(m*dy))
    return np.exp(1j*(2*math.pi/wavelength)*(math.sin(tx)*x*dx+math.sin(ty)*y*dy))

def quadrature(centered: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Four synthetic-reference-weighted frames, preserving upstream sum order."""
    r0=reference; r1=1j*reference; r2=-reference; r3=-1j*reference
    return r0*centered[0]+r1*centered[1]+r2*centered[2]+r3*centered[3]

def interference(object_field: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Ideal four-frame |object + reference exp(i*j*pi/2)|^2 acquisition."""
    shifts=np.array([1,1j,-1,-1j])
    return np.abs(object_field[None]+shifts[:,None,None]*reference[None])**2

def phase_score(field: np.ndarray) -> int:
    """Count pixels in the lower tenth of min-max scaled wrapped phase."""
    phase=np.angle(field)
    span=float(np.ptp(phase))
    if span==0:
        return 0
    return int(np.count_nonzero((phase-phase.min())/span<0.1))
