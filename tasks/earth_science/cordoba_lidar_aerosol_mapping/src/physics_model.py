"""Explicit elastic lidar acquisition and reference-centered quadrature.

MIT adaptation, copyright2021Juan V.Pallotta, LPP.
"""
import numpy as np
STEP=7.5
REFERENCE=812
START=53
OUTPUT=slice(54,813)
MOLECULAR_LIDAR_RATIO=8.496628478899588
AEROSOL_LIDAR_RATIO=50.

def reference_integral(values,reference=REFERENCE,start=START,step=STEP):
    """Original signed trapezoidal integral from reference to each range bin."""
    y=np.atleast_2d(values);out=np.zeros_like(y)
    for i in range(reference+1,y.shape[-1]):out[:,i]=out[:,i-1]+.5*step*(y[:,i]+y[:,i-1])
    for i in range(reference-1,start-1,-1):out[:,i]=out[:,i+1]-.5*step*(y[:,i+1]+y[:,i])
    return out[0] if np.ndim(values)==1 else out

def forward_signal(backscatter,molecular_backscatter,molecular_extinction,lidar_ratio=AEROSOL_LIDAR_RATIO,reference=REFERENCE,step=STEP):
    """Relative range-corrected elastic return; arbitrary instrument gain removed."""
    total=backscatter+molecular_backscatter
    optical=reference_integral(lidar_ratio*backscatter+molecular_extinction,reference,start=0,step=step)
    return total/total[:,reference,None]*np.exp(-2*optical)
