"""Read measured JPEG codes converted losslessly to NPZ; select response samples.

Spatial sampling adapted from HDR Toolbox by Francesco Banterle,2015–2020,
GPL-3.0-or-later. Source revision and adaptations are in test/validation.md.
"""
from pathlib import Path
import numpy as np

SHAPE=(7,427,640,3)
EXPOSURE_TIMES=np.array([2.,.5,.125,0.03333333333333333,.008,.001,.00025],dtype=np.float64)

def load_observations(path: str|Path='data/raw_data.npz') -> tuple[np.ndarray,np.ndarray]:
    with np.load(path,allow_pickle=False) as f:
        frames=f['frames'];times=f['exposure_times']
        if frames.shape!=(1,*SHAPE) or frames.dtype!=np.uint8:
            raise ValueError('frames must be uint8(1,7,427,640,3)')
        if times.shape!=(1,7) or times.dtype!=np.float64 or not np.array_equal(times[0],EXPOSURE_TIMES):
            raise ValueError('Exposure times differ from recorded camera metadata')
        return frames[0].copy(),times[0].copy()

def sample_stack(frames:np.ndarray,times:np.ndarray,n_samples:int=256) -> np.ndarray:
    """Source regular grid; retain a site if ANY channel is strictly exposure-monotone.

    Returns(N,J,3) float64 original integer codes, not spatially averaged values.
    Grid order follows MATLAB column-major meshgrid flattening.
    """
    if frames.ndim!=4 or frames.shape[-1]!=3 or len(times)!=len(frames):
        raise ValueError('Expected frames(J,H,W,3) and times(J)')
    if n_samples<1:raise ValueError('n_samples must be positive')
    _,h,w,_=frames.shape
    f=int(np.floor(np.sqrt(n_samples)+1+.5))
    xx,yy=np.meshgrid(np.arange(0,w,max(int(np.ceil(w/f)),1)),np.arange(0,h,max(int(np.ceil(h/f)),1)))
    values=frames[:,yy.ravel(order='F'),xx.ravel(order='F'),:].transpose(1,0,2).astype(np.float64)
    ordered=values[:,np.argsort(-times),:]
    keep=np.any(np.all(np.diff(ordered,axis=1)<0,axis=1),axis=1)
    if not np.any(keep):raise ValueError('No strictly exposure-monotone sample sites')
    return values[keep]
