"""Explicit response weighting and photometric acquisition mathematics.

Weight conventions adapted from Francesco Banterle,HDR Toolbox,GPL3+.
"""
import numpy as np

def triangular_weights(codes_normalized:np.ndarray) -> np.ndarray:
    """Source Deb97 weights on normalized digital codes; preserve input rounding."""
    x=np.asarray(codes_normalized)
    w=np.empty(x.shape,dtype=np.float64);low=x<=.5
    w[low]=x[low];w[~low]=1.-x[~low]
    return np.clip(w/.5,0,1)

def response_system(samples:np.ndarray,times:np.ndarray,smoothing:float=128.) -> tuple[np.ndarray,np.ndarray]:
    """Build one channel's original weighted overdetermined system A q=b.

    q consists of256 log inverse-response values and one log irradiance per site.
    Data rows precede the g(128)=0 gauge and254 weighted second differences.
    The source's two unused zero rows are retained for numeric parity.
    """
    z=np.asarray(samples)
    if z.ndim!=2 or z.shape[1]!=len(times) or np.any(z<0) or np.any(z>255) or np.any(z!=np.floor(z)):
        raise ValueError('Samples must be integer-valued codes(N,J) within0..255')
    if np.any(np.asarray(times)<=0) or smoothing<0:raise ValueError('Invalid times/smoothing')
    n,j=z.shape;weights=triangular_weights(np.arange(256)/255.)
    a=np.zeros((n*j+257,256+n));b=np.zeros(n*j+257)
    row=np.arange(n*j);zi=z.astype(int).ravel();site=np.repeat(np.arange(n),j)
    a[row,zi]=weights[zi];a[row,256+site]=-weights[zi]
    b[row]=weights[zi]*np.tile(np.log(times),n)
    a[n*j,128]=1
    for i in range(254):a[n*j+1+i,i:i+3]=smoothing*weights[i+1]*np.array([1.,-2.,1.])
    return a,b

def predict_codes(irradiance:np.ndarray,response:np.ndarray,times:np.ndarray) -> np.ndarray:
    """Inverse lookup of monotone response, clipped at digital saturation limits."""
    if np.any(irradiance<=0) or np.any(response<=0) or np.any(np.diff(response,axis=0)<=0):
        raise ValueError('Positive irradiance and strictly monotone response required')
    out=np.empty((len(times),*irradiance.shape))
    for j,t in enumerate(times):
        for c in range(3):out[j,...,c]=np.interp(np.log(irradiance[...,c]*t),np.log(response[:,c]),np.arange(256))
    return out

def forward_residual(frames:np.ndarray,times:np.ndarray,irradiance:np.ndarray,response:np.ndarray) -> float:
    """Weighted ADU RMSE on fixed observed-code support8..247, all RGB channels."""
    pred=predict_codes(irradiance,response,times);obs=frames.astype(float)
    w=np.minimum(obs,255-obs)*((obs>=8)&(obs<=247))
    return float(np.sqrt(np.sum(w*(pred-obs)**2)/np.sum(w)))

def log_exposure_residual(frames:np.ndarray,times:np.ndarray,irradiance:np.ndarray,response:np.ndarray) -> float:
    """Weighted log-exposure mismatch; defined even for a nonmonotone trial curve.

    This diagnostic does not certify monotonicity or substitute for Judge's code residual.
    """
    error=mass=0.
    for j,t in enumerate(times):
        for c in range(3):
            z=frames[j,...,c];w=np.minimum(z.astype(float),255.-z)*((z>=8)&(z<=247))
            e=np.log(response[z,c])-np.log(irradiance[...,c])-np.log(t)
            error+=np.sum(w*e**2);mass+=w.sum()
    return float(np.sqrt(error/mass))
