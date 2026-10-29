from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
KEYS=('spin_echo_length','depolarisation','standard_error','wavelength','length_standard_error')
def load_observations(path=ROOT/'data/raw_data.npz'):
    """Return exact reduced float64 measurement arrays with batch removed."""
    with np.load(path,allow_pickle=False) as a:d={k:a[k][0].copy() for k in KEYS}
    if any(v.shape!=(40,) or v.dtype!=np.float64 or not np.isfinite(v).all() for v in d.values()):raise ValueError('invalid observations')
    if np.any(d['standard_error']<=0) or np.any(np.diff(d['spin_echo_length'])<=0):raise ValueError('invalid errors or axis')
    return d
