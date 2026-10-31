"""Read source-normalized/binned observations without renormalizing."""
from pathlib import Path
import numpy as np
SHAPES={'spectra':(1,2,2,300,480,51),'frequency_hz':(1,2,51),'calibration_images':(1,2,300,480)}
DTYPES={'spectra':np.float64,'frequency_hz':np.float32,'calibration_images':np.float64}
def load_observations(path):
    with np.load(Path(path),allow_pickle=False) as z:out={key:z[key].copy() for key in SHAPES}
    for key,value in out.items():
        if value.shape!=SHAPES[key] or value.dtype!=DTYPES[key] or not np.isfinite(value).all():raise ValueError(f'Invalid observation {key}')
    if not np.all(np.diff(out['frequency_hz'],axis=-1)>0):raise ValueError('Frequency must increase within each window')
    return out
def normalize_and_bin(native,bin_factor=4):
    """Normalize each native pixel BEFORE disjoint spatial block averaging.

    native: (configuration,window,row,column,frequency), real measured values.
    """
    a=np.asarray(native,dtype='float64');a=a/a.max(axis=-1,keepdims=True)
    p,w,h,c,f=a.shape
    return a.reshape(p,w,h//bin_factor,bin_factor,c//bin_factor,bin_factor,f).mean(axis=(3,5))
