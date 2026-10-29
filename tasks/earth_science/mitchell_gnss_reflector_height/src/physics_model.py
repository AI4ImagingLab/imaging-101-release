"""Explicit harmonic least-squares GNSS interferometric spectrum (GPL3 adaptation)."""
import numpy as np
def forward(coordinate,height,coefficients):
    phase=2*np.pi*height*np.asarray(coordinate)
    return coefficients[0]*np.cos(phase)+coefficients[1]*np.sin(phase)
def spectrum(coordinate,signal,max_height=8.,precision=.005,min_height=.5):
    x=np.asarray(coordinate);y=np.asarray(signal);span=np.ptp(x);n=len(x)
    ofac=1/span/precision;hifac=max_height/(n/(2*span))
    nout=int(.5*ofac*hifac*n)
    heights=np.linspace(1/(span*ofac),hifac*n/(2*span),nout)
    phase=2*np.pi*heights[:,None]*x[None,:];c=np.cos(phase);s=np.sin(phase)
    cc=np.sum(c*c,axis=1);ss=np.sum(s*s,axis=1);cs=np.sum(c*s,axis=1)
    cy=c@y;sy=s@y;det=cc*ss-cs*cs
    power=.5*(ss*cy*cy+cc*sy*sy-2*cs*cy*sy)/det
    amplitude=2*np.sqrt(np.maximum(power,0)/n);keep=heights>min_height
    return heights[keep],amplitude[keep]
