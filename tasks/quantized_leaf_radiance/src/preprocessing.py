from pathlib import Path
import numpy as np

def load_data(path=Path('data/raw_data.npz')):
    with np.load(path,allow_pickle=False) as f:lower=f['lower'][0].copy();upper=f['upper'][0].copy()
    if lower.shape!=upper.shape or lower.ndim!=3 or lower.shape[-1]!=3:raise ValueError('RGB interval shapes must agree')
    if lower.dtype!=np.float32 or upper.dtype!=np.float32:raise ValueError('intervals must retain float32')
    if not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(lower>upper) or lower.min()<0 or upper.max()>1:raise ValueError('invalid quantization interval')
    return lower,upper
