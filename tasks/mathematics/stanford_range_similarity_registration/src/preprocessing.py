"""Public laser-point observations and fixed deterministic fit selection."""
from pathlib import Path
import numpy as np
FIT_STRIDE=40

def load_data(path='data/raw_data.npz'):
    """Return unbatched float64 point tables, xyz metres, unchanged order."""
    with np.load(Path(path),allow_pickle=False) as f:
        result={key:f[key][0].copy() for key in ['target_points','source_points']}
    for key,a in result.items():
        if a.ndim!=2 or a.shape[1]!=3 or a.dtype!=np.float64 or not np.isfinite(a).all():
            raise ValueError('Invalid public point table: '+key)
    return result

def prepare(data,stride=FIT_STRIDE):
    """Every stride-th recorded point; common observed target RMS normalization.

    Returns target/source normalized point tables, common origin(m), unit(m).
    Entire native tables remain available and are transformed after fitting.
    """
    x=data['target_points'][::stride];y=data['source_points'][::stride]
    origin=x.mean(axis=0);unit=np.sqrt(np.mean((x-origin)**2))
    if not np.isfinite(unit) or unit<=0:raise ValueError('Degenerate spatial observations')
    return (x-origin)/unit,(y-origin)/unit,origin,unit
