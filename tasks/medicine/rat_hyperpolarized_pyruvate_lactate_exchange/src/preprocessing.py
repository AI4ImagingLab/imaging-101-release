"""Native signed metabolite-image observations; no spatial support selection."""
from pathlib import Path
import numpy as np
TIME_STEP=2.0
T1_P=30.0
T1_L=25.0
def load_data(path):
    """Return signal(2,32,32,25) float32 and angles(2,25) float16 radians."""
    with np.load(Path(path),allow_pickle=False) as z:
        if set(z.files)!={'signal','flip_angles'}:raise ValueError('unexpected observation keys')
        s=z['signal'];a=z['flip_angles']
    if s.shape!=(1,2,32,32,25) or s.dtype!=np.float32 or not np.isfinite(s).all():raise ValueError('invalid signal')
    if a.shape!=(1,2,25) or a.dtype!=np.float16 or not np.isfinite(a).all() or np.any(a<=0) or np.any(a>=np.pi/2):raise ValueError('invalid flip angles')
    return s[0],a[0]
def prepare(signal,angles):
    """C-order native x,y voxels; preserve source zero-padding/upcast before trig."""
    s=np.asarray(signal)
    if s.ndim!=4 or s.shape[0]!=2 or s.shape[-1]!=25:raise ValueError('signal must be 2,x,y,25')
    if not np.isfinite(s).all():raise ValueError('nonfinite observations')
    n=s.shape[1]*s.shape[2]
    padded=np.concatenate((np.transpose(s,(1,2,0,3)).reshape(n,2,25),np.zeros((n,2,25))),axis=1)
    flips=np.concatenate((np.asarray(angles),np.ones((2,25))),axis=0)
    return padded,flips
