"""Local response least-squares and weighted exposure fusion.

Adapted from gsolve/BuildHDR in Francesco Banterle's HDR Toolbox,GPL3+.
gsolve credits Paul Debevec and Jitendra Malik. No upstream runtime imports.
"""
import numpy as np
from scipy.linalg import lstsq
from .preprocessing import sample_stack
from .physics_model import triangular_weights,response_system

def fit_response(samples:np.ndarray,times:np.ndarray,smoothing:float=128.) -> np.ndarray:
    """Return(256,3) log inverse response, paper gauge g[128]=0 per channel."""
    g=np.empty((256,3),dtype=np.float64)
    for c in range(3):
        a,b=response_system(samples[...,c],times,smoothing)
        q,_,rank,_=lstsq(a,b,lapack_driver='gelsd')
        if rank<a.shape[1]:raise ValueError('Response system is rank deficient')
        g[:,c]=q[:256]
    return g

def merge_exposures(frames:np.ndarray,times:np.ndarray,response:np.ndarray|None=None,mode:str='calibrated') -> np.ndarray:
    """Full HDR Toolbox log merge, including epsilon and global endpoint weights.

    Source keeps the running numerator/denominator in float32 and returns float64.
    mode is calibrated/linear/gamma; gamma=2.2 is an explicitly uncalibrated baseline.
    """
    if mode not in {'calibrated','linear','gamma'}:raise ValueError('Unknown mode')
    if mode=='calibrated' and (response is None or response.shape!=(256,3)):raise ValueError('Missing response')
    x=frames.astype(np.float32)/np.float32(255)
    shortest=int(np.argmin(times));longest=int(np.argmax(times));delta=1/65535.
    numerator=np.zeros(frames.shape[1:],dtype=np.float32);denom=np.zeros_like(numerator)
    def linearize(v):
        if mode=='linear':return v
        if mode=='gamma':return np.power(v,np.float32(2.2))
        codes=np.floor(v*255+.5).astype(int)
        return np.stack([response[codes[...,c],c] for c in range(3)],axis=-1)
    for j,t in enumerate(times):
        v=x[j];weight=triangular_weights(v)
        if j==shortest and np.any(v>.9):weight=v
        if j==longest and np.any(v<.1):weight=1-v
        lin=linearize(v)
        # Octave/MATLAB single-double arithmetic returns single whenever one operand is single.
        if lin.dtype==np.float32:
            term=(np.log(lin+np.float32(delta))-np.float32(np.log(t))).astype(np.float32)
        else:term=np.log(lin+delta)-np.log(t)
        numerator=(numerator+weight*term).astype(np.float32)
        denom=(denom+weight).astype(np.float32)
    with np.errstate(divide='ignore',invalid='ignore'):
        out=np.exp(numerator/denom).astype(np.float32)
    mid=x[int(np.floor(len(times)/2+.5))-1]
    for bright,j in [(True,shortest),(False,longest)]:
        mask=np.any((denom<=1e-4)&((mid>.5) if bright else (mid<.5)),axis=-1)
        if np.any(mask):out[mask]=(linearize(x[j])/times[j])[mask]
    if not np.isfinite(out).all() or np.any(out<=0):raise ValueError('Unrecoverable exposure support')
    return out.astype(np.float64)

def reconstruct(frames:np.ndarray,times:np.ndarray,n_samples:int=256,smoothing:float=128.) -> dict[str,np.ndarray]:
    g=fit_response(sample_stack(frames,times,n_samples),times,smoothing)
    response=np.exp(g)
    return {'reconstruction':merge_exposures(frames,times,response)[None],'response':response[None]}
