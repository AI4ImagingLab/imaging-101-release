from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
KEYS=('k','chi','scattering_k','scattering_phase','scattering_amplitude','real_momentum','mean_free_path')
def load_observations(path=ROOT/'data/raw_data.npz'):
    """Load float64 reduced observations and theoretical atomic scattering calibration."""
    with np.load(path,allow_pickle=False) as a:d={k:a[k][0].copy() for k in KEYS}
    if any(v.ndim!=1 or v.dtype!=np.float64 or not np.isfinite(v).all() for v in d.values()):raise ValueError('invalid input')
    if d['k'].shape!=d['chi'].shape or not np.all(np.diff(d['k'])>0):raise ValueError('invalid data axis')
    if any(d[k].shape!=d['scattering_k'].shape for k in KEYS[3:]):raise ValueError('invalid calibration shape')
    return d
