"""Measured parcel time courses; atlas aggregation occurred at acquisition binding."""
from pathlib import Path
import sys,numpy as np
from scipy import stats
from .physics_model import ideal_filter
def load_data(path='data/raw_data.npz'):
    with np.load(Path(path),allow_pickle=False) as f:
        bold=f['bold'][0].copy();counts=f['vertex_counts'][0].copy()
    if bold.shape!=(400,152) or bold.dtype!=np.float64 or not np.isfinite(bold).all():raise ValueError('invalid BOLD')
    if counts.shape!=(400,) or np.any(counts<=0):raise ValueError('invalid atlas counts')
    return dict(bold=bold,vertex_counts=counts)
def preprocess(bold):
    """Return first-stage and actual final filtered time-first signals.

    The upstream functions mutate and return the same signal array. Its later
    .01-.08Hz filter therefore also affects the aliased deconvolution input.
    Preserve this observed execution flow explicitly, with a diagnostic copy
    only for the independently captured first stage.
    """
    signal=stats.zscore(np.ascontiguousarray(bold.T),axis=0,ddof=1)
    signal=np.nan_to_num(signal)
    signal=ideal_filter(signal,2.,[0.,sys.float_info.max]);first=signal.copy()
    signal=ideal_filter(signal,2.,[.01,.08])
    return first,signal
