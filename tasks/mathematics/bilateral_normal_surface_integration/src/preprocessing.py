"""Load author-provided normal fields without smoothing or renormalization."""
from pathlib import Path
import numpy as np


def load_data(path='data/raw_data.npz'):
    """Return name -> (float64 H,W,3 normals, bool H,W mask)."""
    result={}
    with np.load(Path(path),allow_pickle=False) as f:
        for name,shape in [('vase',(168,168)),('tent',(128,128))]:
            n=f[name+'_normals'];m=f[name+'_mask']
            if n.shape!=(1,*shape,3) or m.shape!=(1,*shape) or n.dtype!=np.float64 or m.dtype!=np.bool_:
                raise ValueError('Invalid normal-field contract: '+name)
            if not np.isfinite(n).all() or not m.any():raise ValueError('Invalid normal support')
            result[name]=(n[0].copy(),m[0].copy())
    return result
