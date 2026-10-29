"""Load author-processed observations without clipping, resampling or fitting."""
from pathlib import Path
import numpy as np

OBS_KEYS=('time','mlt1','mcolat1','el','time_ref','mlt1_ref','mcolat1_ref','el_ref','rhs')
def load_data(path):
    """Return unbatched float64 observation dict and declared grid arrays."""
    with np.load(Path(path),allow_pickle=False) as z:
        result={}
        for key in OBS_KEYS:
            value=z[key]
            if value.dtype!=np.float64 or value.shape!=(1,504358) or not np.isfinite(value).all():
                raise ValueError(f'Invalid observation {key}')
            result[key]=value[0].copy()
        expected={'grid_time':np.arange(25,dtype=float)*3600,
                  'grid_colatitude':np.deg2rad(np.arange(2.5,180,2.5)),
                  'grid_longitude':np.deg2rad(np.arange(0.,360.,5.))}
        grids={}
        for key,target in expected.items():
            value=z[key]
            if value.dtype!=np.float64 or value.shape!=(1,len(target)) or not np.array_equal(value[0],target):
                raise ValueError(f'Invalid fixed acquisition grid {key}')
            grids[key]=value[0].copy()
    return result,grids
