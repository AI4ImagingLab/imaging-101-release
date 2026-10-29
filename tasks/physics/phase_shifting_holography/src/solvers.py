"""Quadrature recovery with finite exhaustive synthetic reference search."""
import numpy as np
from .preprocessing import center_frames
from .physics_model import centered_spectrum, reference_wave, quadrature, phase_score

SEARCH_HALF_WIDTH=1
SEARCH_STEPS=4

def find_carrier(centered_first: np.ndarray, upper: bool=True) -> tuple[int,int]:
    """Return (column,row) spectral peak, preserving upstream ROI slicing."""
    m,n=centered_first.shape
    spectrum=centered_spectrum(centered_first)
    power=np.abs(spectrum)**2
    # Negative stops are intentional upstream Python slices, documented in validation.
    if upper:
        roi=power[1:int(m/2)-int(m-m*.1),1:n]
    else:
        roi=power[1:m,1:int(n/2)-int(n-n*.1)]
    row,col=np.unravel_index(np.argmax(roi),roi.shape)
    return int(col+1),int(row+1)

def reconstruct(holograms: np.ndarray, s: float=SEARCH_HALF_WIDTH,
                steps: int=SEARCH_STEPS) -> tuple[np.ndarray,dict]:
    """Return compensated complex128 field and selected frequency/score metadata."""
    if steps<s or steps<1 or s<0:
        raise ValueError('Require steps >= s >= 0 and steps >= 1')
    h=center_frames(holograms)
    fx,fy=find_carrier(h[0])
    best=-1; carrier=None
    for x in np.linspace(fx-s,fx+s,steps):
        for y in np.linspace(fy-s,fy+s,steps):
            wave=reference_wave(h.shape[1:],x,y)
            score=phase_score(quadrature(h,wave))
            if score>best:
                best=score; carrier=(float(x),float(y))
    field=quadrature(h,reference_wave(h.shape[1:],*carrier))
    return field,dict(carrier=carrier,score=best,peak=(fx,fy),candidates=steps**2)
