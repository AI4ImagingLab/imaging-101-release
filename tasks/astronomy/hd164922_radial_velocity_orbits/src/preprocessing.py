"""Strict full 401-observation interface; no nightly binning in this case."""
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def load_data(path=None):
    with np.load(ROOT/'data/raw_data.npz' if path is None else path,allow_pickle=False) as z:
        result={k:z[k][0].copy() for k in ('time','velocity','uncertainty','instrument')}
    for k,a in result.items():
        if a.shape!=(401,) or not np.isfinite(a).all():raise ValueError('invalid '+k)
    if result['instrument'].dtype!=np.int64 or set(result['instrument'])!={0,1,2}:raise ValueError('invalid instruments')
    if np.any(result['uncertainty']<=0):raise ValueError('nonpositive uncertainty')
    return result
